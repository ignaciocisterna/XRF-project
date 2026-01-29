# Importaciones externas
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import xraylib as xl
import matplotlib.pyplot as plt
import time
import itertools
import threading

# Importaciones relativas 
from . import core as core
from . import processing as prc
from . import metrics as mtr
from .config.manager import InstrumentConfig

#------------------------------------------------------------------------------#

class XRFDeconv:
    def __init__(self, energy, counts, name="Muestra", fondo="lin", instrument="S2 PICOFOX 200",
                t_real=None, t_live=None, ajustar_tau=None):
        self.E_raw = energy
        self.I_raw = counts
        self.name = name
        self.fondo = fondo
        self.offset = 11 if fondo == "lin" else 12

        self.config = InstrumentConfig.load(instrument)
        res = self.config.get_resolution_params()
        self.noise_init = res["noise"]
        self.fano_init = res["fano"]
        self.epsilon_init = res["epsilon"]

        # Tiempos y estimación de Tau
        self.t_real = t_real 
        self.t_live = t_live 
        # Si no hay tiempos, calculamos un tau inicial de 0 o uno genérico
        # y forzamos que se ajuste porque no tenemos info real.
        if t_real and t_live:
            self.tau_init = prc.estimate_tau_pileup(counts, t_real, t_live)
            # Si el usuario no especificó nada, no lo ajustamos (usamos el físico)
            self.free_tau = ajustar_tau if ajustar_tau is not None else False
        else:
            self.tau_init = 1e-6 # Valor semilla genérico
            self.t_live = 1.0    # Evitar división por cero
            # Si no hay info, es obligatorio ajustarlo para que el modelo no falle
            self.free_tau = True
        
        # Atributos que se llenarán en el proceso
        self.E, self.I = None, None
        self.I_net = None
        self.bkg = None
        self.elements = []
        self.p_actual = None
        self.pcov = None
        self.I_fit = None

#------------------------------------------------------------------------------#
    
    def prepare_data(self, e_max=17.5):
        self.E, self.I = prc.recortar_espectro(self.E_raw, self.I_raw, e_max=e_max)
        print(f"[{self.name}] Datos preparados. Rango: {self.E.min():.2f} - {self.E.max():.2f} keV")
        
#------------------------------------------------------------------------------#
   
    def run_identification(self, manual=None, ignore=None, graf=False, verbose=False, 
                           permitir_solapamientos=False, todos=False):
        """Calcula SNIP y detecta elementos."""
        self.bkg = prc.snip_trace_safe(self.E, self.I, core.fwhm_SNIP)
                               
        self.I_net = self.I - self.bkg
        self.I_net[self.I_net < 0] = 0 # Limpieza de valores negativos
                               
        self.elements = prc.detectar_elementos(self.E, self.I, self.bkg, 
                                               manual_elements=manual,
                                               ignorar=ignore,
                                               permitir_solapamientos=permitir_solapamientos, 
                                               todos=todos)
        if graf:
            mtr.graficar_deteccion_preliminar(self.E, self.I, self.elements, self.bkg)
        if verbose:
            print(f"[{self.name}] Elementos detectados: {self.elements}")

#------------------------------------------------------------------------------#

    def get_mask(self, etapa):
            """
            etapa: "K", "L", "M" o "global"
            """
            n_elem = len(self.elements)
            
            # 1. Parte Global (Siempre libre en las etapas iniciales)
            # Creamos una lista de '1's según el tamaño del offset
            mask_base = [1] * self.offset 
            
            # 2. Parte de Elementos [Area_K, Area_L, Area_M]
            element_masks = []
        
            for elem in self.elements:
                # Inicializamos los 3 slots para este elemento [K, L, M]
                slots = [0, 0, 0] 
                if etapa == "K":
                    slots[0] = 1
                elif etapa == "L":
                    try:
                        info = core.get_Xray_info(elem, families=("L",))
                        tiene_L = any((self.E.min() + 0.25) <= d['energy'] <= (self.E.max() - 0.25) for d in info.values())
                        if tiene_L: slots[1] = 1
                    except: pass
                elif etapa == "M":
                    slots[2] = 1
                elif etapa == "global":
                    slots = [1, 1, 1]
                
                element_masks.extend(slots)
                    
            return mask_base + element_masks

#------------------------------------------------------------------------------#
    
    def get_p0(self, etapa, mask=None):
        """Genera la semilla base según el modelo de fondo y etapa actual."""
        if etapa == "K":
            c0_init = np.min(self.bkg)
            c1_init = (self.bkg[-1] - self.bkg[0]) / (self.E[-1] - self.E[0])

            rayleigh_init = max(self.I) * 2
            compton_init = max(self.I) * 1

            if self.fondo == "lin":
                # a, b, c0, c1, Ray, Comp
                p0 = [0.0057, 0.00252, c0_init, c1_init, rayleigh_init, compton_init]
            else:
                # a, b, c0, c1, c2, Ray, Comp
                p0 = [0.0057, 0.00252, c0_init, c1_init, 0.0, rayleigh_init, compton_init]

            if self.I_net is None: 
                self.I_net = np.maximum(self.I - self.bkg, 0)
                
            area_init = np.trapezoid(self.I_net, self.E) / len(self.elements)

            for _ in self.elements:
                p0 += [
                    area_init,              # K
                    0,                      # L
                    0                       # M
                ]
            return p0
        elif etapa == "L" or etapa == "M" or etapa == "global":
            # Creamos una copia del estado actual de los parámetros
            p_working = self.p_actual.copy()
            
            if etapa == "L":
                # En lugar de un bucle manual, usamos la máscara que acabamos de crear
                # para saber dónde inyectar la semilla de intensidad
                for i in range(self.offset, len(p_working)):
                    # Si la máscara dice que este parámetro es una 'Area L' libre:
                    if i < len(mask) and mask[i] == 1:
                        # Solo aplicamos la semilla si el parámetro está en un slot de Area L
                        # (posiciones offset+1, offset+4, offset+7...)
                        if (i - self.offset) % 3 == 1:
                            p_working[i] = np.max(self.I_net) * 0.01
            
            # Filtramos solo los parámetros que la etapa permite mover
            p0 = [p for p, f in zip(p_working, mask) if f]
            return p0

#------------------------------------------------------------------------------#

    def run_stage_fit(self, etapa, graf=False, roi_margin=0.4, tol=1e-6):
        free_mask = self.get_mask(etapa)

        # Si es la primera etapa (K), inicializamos p_actual con p0 completo
        if etapa == "K" or self.p_actual is None:
            self.p_actual = self.get_p0("K")

        p0_free = self.get_p0(etapa, free_mask)

        def frx_wrapper(E_val, *p_free):
            p_full = core.build_p_from_free(p_free, self.p_actual, free_mask)
            params = core.pack_params(p_full, self.elements, fondo=self.fondo)
            return core.FRX_model_sdd_general(E_val, params, config=self.config)
        
        # Definimos el techo de cuentas: 1.5 veces el máximo del espectro
        # es más que suficiente para cualquier pico real.
        techo_cuentas = np.max(self.I) * 1.5

        A_LIMITS = (0.003, 0.015) 
        B_LIMITS = (0.0005, 0.005)
        
        lower_bounds = []
        upper_bounds = []
        
        for i, p in enumerate(p0_free):
            if i == 0: # Parámetro 'a'
                lower_bounds.append(A_LIMITS[0])
                upper_bounds.append(A_LIMITS[1])
            elif i == 1: # Parámetro 'b'
                lower_bounds.append(B_LIMITS[0])
                upper_bounds.append(B_LIMITS[1])
            elif i < self.offset: # El resto del fondo (c0, c1, c2, Ray, Comp)
                lower_bounds.append(0.0)
                upper_bounds.append(np.inf)
            else: # Áreas de picos
                lower_bounds.append(0.0) 
                upper_bounds.append(techo_cuentas)
        
        bounds = (lower_bounds, upper_bounds)

        roi_mask = prc.generar_mascara_roi(self.E, self.elements, margen=roi_margin)

        try:
            if etapa in ["K", "L"]:
                popt, pcov = curve_fit(frx_wrapper, 
                                      self.E[roi_mask], 
                                      self.I[roi_mask], 
                                      p0=p0_free,
                                      bounds=bounds,
                                      method='trf',
                                      x_scale='jac',
                                      loss='soft_l1',
                                      xtol=tol, 
                                      ftol=tol
                                      )
            elif etapa == "M":
                popt, pcov = curve_fit(frx_wrapper, 
                                      self.E[roi_mask], 
                                      self.I[roi_mask],
                                      p0=p0_free,
                                      bounds=bounds,
                                      xtol=tol, 
                                      ftol=tol
                                      )
            else: #global
                sigma_completo = np.sqrt(np.maximum(self.I, 1))
                sigma_roi = sigma_completo[roi_mask]

                # REFUERZO DE SEMILLA: Antes de entrar, asegurémonos de que 
                # las áreas no estén en cero si el solver se rindió antes.
                max_I = np.max(self.I_net)
                semilla_minima = max_I * 0.005 # 0.5% del pico máximo
                
                for i in range(self.offset, len(p0_free)):
                    # Solo reforzamos si el área está muy cerca de cero
                    if p0_free[i] < 0.1: 
                        # Si es una línea K (el primer slot de cada 3) le damos un empujoncito
                        if (i - self.offset) % 3 == 0:
                            p0_free[i] = semilla_minima
                        # Para L y M (slots 1 y 2), mejor dejarlos en 0 o algo ínfimo
                        else:
                            p0_free[i] = 0.01
                        
                popt, pcov = curve_fit(frx_wrapper,
                                      self.E[roi_mask],
                                      self.I[roi_mask],
                                      p0=p0_free,
                                      bounds=bounds,
                                      sigma=sigma_roi,
                                      absolute_sigma=True,
                                      method='trf', 
                                      x_scale='jac', 
                                      loss='huber',
                                      max_nfev=50000,
                                      xtol=1e-5, 
                                      ftol=1e-5 
                                  )
                self.pcov = pcov 

            self.p_actual = core.build_p_from_free(popt, self.p_actual, free_mask)
            self.I_fit = frx_wrapper(self.E, *popt) 

            if graf:
                mtr.graficar_ajuste(self.E, self.I, self.I_fit, self.elements, 
                                    popt, self.p_actual, [etapa])

        except Exception as e:
            print(f"Error al ajustar {etapa}: {e}")

#------------------------------------------------------------------------------#

    def animacion_carga(self, stop_event, mensaje):
        for puntos in itertools.cycle(["",".", "..", "..."]):
            if stop_event.is_set():
                break
            print(f"\r{mensaje}{puntos}   ", end="", flush=True)
            time.sleep(0.5)
        print("\r", end="", flush=True)

#------------------------------------------------------------------------------#

    def run_full_fit(self, graf=False, roi_margin=0.4, tol=1e-6):
        """Ejecuta el pipeline completo de ajuste secuencial."""
        for etapa in ["K", "L", "M"]:
            stop_event = threading.Event()
            t = threading.Thread(
                target=self.animacion_carga,
                args=(stop_event, f"  > Ajustando Capas {etapa}")
            )
            t.start()
            self.run_stage_fit(etapa, graf=graf, roi_margin=roi_margin, tol=tol)
            stop_event.set()
            t.join()
        
        stop_event = threading.Event()
        t = threading.Thread(
            target=self.animacion_carga,
            args=(stop_event, "  > Refinando Ajuste Global")
        )
        t.start()
        self.run_stage_fit('global', graf=graf, roi_margin=roi_margin, tol=tol)
        stop_event.set()
        t.join()
        print(f"[{self.name}] Deconvolución finalizada con éxito.")

#------------------------------------------------------------------------------#

    def report(self, filename=None, pdf=False):
        """Genera el PDF y muestra el reporte en consola."""
        if self.I_fit is None:
            raise ValueError("Debe ejecutar run_full_fit() antes de generar un reporte.")
            
        fname = filename if filename else f"Reporte_{self.name}.pdf"

        mtr.generar_reporte_completo(self.E, self.I, self.I_fit, 
                                        self.p_actual, self.elements, 
                                        nombre_muestra=self.name, fondo=self.fondo)
        if pdf:
            mtr.exportar_reporte_pdf(self.E, self.I, self.I_fit, 
                                    self.p_actual, self.elements, 
                                    nombre_muestra=self.name, 

                                    archivo=fname, fondo=self.fondo)

#------------------------------------------------------------------------------#

    def run_resolution_check(self):
        """Compara la resolución ajustada contra la nominal del equipo.
           Aplicar idealmente a ajuste completo.
        """
        params = core.pack_params(self.p_actual, self.elements, fondo=self.fondo)
        mtr.check_resolution_health(params, self.config)





















