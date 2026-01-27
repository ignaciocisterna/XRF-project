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

#------------------------------------------------------------------------------#

class XRFDeconv:
    def __init__(self, energy, counts, name="Muestra", fondo="lin"):
        self.E_raw = energy
        self.I_raw = counts
        self.name = name
        self.fondo = fondo
        self.offset = 6 if fondo == "lin" else 7
        
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
            for _ in range(n_elem):
                if etapa == "K":
                    element_masks += [1, 0, 0]

                elif etapa == "L":
                    for elem in self.elements:
                        try:
                            info = core.get_Xray_info(elem, families=("L",))
                            # Verificamos si al menos una línea L está en el rango E
                            tiene_L_en_rango = any(self.E.min() <= d['energy'] <= self.E.max() for d in info.values())
                            if tiene_L_en_rango:
                                element_masks += [0, 1, 0] # K fijo, L libre, M fijo
                            else:
                                element_masks += [0, 0, 0] # Todo fijo para este elemento
                        except:
                            element_masks += [0, 0, 0]

                elif etapa == "M":
                    element_masks += [0, 0, 1]

                else: # global
                    element_masks += [1, 1, 1]
                    
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
        elif etapa == "L":
            p_for_L = self.p_actual.copy()
            idx = self.offset
            for i, elem in enumerate(self.elements):
                # Solo inicializamos si la máscara permite que sea libre
                if idx + 1 < len(mask) and idx + 1 < len(p_for_L):
                    if mask[idx + 1] == 1:
                        # Una semilla basada en el máximo del espectro neto es más estable
                        p_for_L[idx + 1] = max(self.I_net) * 0.01
                idx += 3
            
            p0 = [p for p, f in zip(p_for_L, mask) if f]

        else: # etapa M o Global 
            p_for_M_or_G = self.p_actual.copy()
            p0 = [p for p, f in zip(p_for_M_or_G, mask) if f]

        return p0

#------------------------------------------------------------------------------#

    def run_stage_fit(self, etapa, graf=False, roi_margin=0.4, tol=1e-5):
        free_mask = self.get_mask(etapa)

        # Si es la primera etapa (K), inicializamos p_actual con p0 completo
        if etapa == "K" or self.p_actual is None:
            self.p_actual = self.get_p0("K")

        p0_free = self.get_p0(etapa, free_mask)

        def frx_wrapper(E_val, *p_free):
            p_full = core.build_p_from_free(p_free, self.p_actual, free_mask)
            params = core.pack_params(p_full, self.elements, fondo=self.fondo)
            return core.FRX_model_sdd_general(E_val, params)
        
        lower_bounds = [0] * len(p0_free)
        upper_bounds = [np.inf] * len(p0_free)
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
                popt, pcov = curve_fit(frx_wrapper,
                                      self.E[roi_mask],
                                      self.I[roi_mask],
                                      p0=p0_free,
                                      bounds=bounds,
                                      sigma=sigma_roi,
                                      absolute_sigma=True,
                                      max_nfev=50000,
                                      xtol=tol, 
                                      ftol=tol 
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

    def run_full_fit(self, graf=False, roi_margin=0.4, tol=1e-5):
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








