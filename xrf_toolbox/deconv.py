# Importaciones externas
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import xraylib as xl
import xraydb
import matplotlib.pyplot as plt
import time
import itertools
import threading
from numpy.polynomial.chebyshev import Chebyshev
# Importaciones relativas 
from . import core as core
from . import processing as prc
from . import metrics as mtr
from .config.manager import InstrumentConfig

#------------------------------------------------------------------------------#

class XRFDeconv:
    def __init__(self, energy, counts, name="Muestra", fondo="poly", grado_fondo=2, instrument="s2 picofox 200",
                t_real=None, t_live=None, ajustar_tau=None):
        self.E_raw = energy
        self.I_raw = counts
        self.name = name
        self.fondo = fondo
        self.grado_fondo = grado_fondo
        self.n_bkg = self.grado_fondo + 1
        self.offset = 11 + self.grado_fondo

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
            # Si no hay datos y el usuario pide NO ajustar, tau es 0 -> Peak Suma desactivados
            if ajustar_tau:
                self.tau_init = 1e-6 # Semilla pequeña para que el solver empiece a buscar
                self.free_tau = True
            else:
                self.tau_init = 0.0  # <--- ESTO OMITE LOS PICOS DE SUMA
                self.free_tau = False
            self.t_live = 1.0    # Evitar división por cero
        
        # Atributos que se llenarán en el proceso
        self.E, self.I = None, None
        self.I_net = None
        self.bkg_snip = None
        self.bkg_fit = None
        self.elements = []
        self.p_actual = None
        self.pcov = None
        self.I_fit = None
        self.p_dict = None

#------------------------------------------------------------------------------#
    
    def prepare_data(self, e_max=None):
        if not e_max:
            if self.config.mode == "TXRF":
                z_anod = xl.SymbolToAtomicNumber(self.config.anode)
                e_max = xl.LineEnergy(z_anod, xl.KA1_LINE)
            else:
                e_max = self.E_raw[-1]
        self.E, self.I = prc.recortar_espectro(self.E_raw, self.I_raw, e_max=e_max)
        print(f"[{self.name}] Datos preparados. Rango: {self.E.min():.2f} - {self.E.max():.2f} keV")
        
#------------------------------------------------------------------------------#
   
    def run_identification(self, manual=None, ignore=None, graf=False, verbose=False, 
                           permitir_solapamientos=False, todos=False, autodet=True):
        """Calcula SNIP y detecta elementos."""
        self.bkg_snip = prc.snip_trace_safe(self.E, self.I, core.fwhm_SNIP)
                               
        self.I_net = self.I - self.bkg_snip
        self.I_net[self.I_net < 0] = 0 # Limpieza de valores negativos
        if autodet:                 
            self.elements = prc.detectar_elementos(self.E, self.I, self.bkg_snip, 
                                                   manual_elements=manual,
                                                   ignorar=ignore,
                                                   permitir_solapamientos=permitir_solapamientos, 
                                                   todos=todos)
        else: 
            if manual: 
                self.elements = manual
            else:
                raise ValueError("Debe ingresarse una lista manual de elementos si la autodetección está desactivada")
        if graf:
            mtr.graficar_deteccion_preliminar(self.E, self.I, self.elements, self.bkg_snip, config=self.config)
        if verbose:
            print(f"[{self.name}] Reporte de Detección:")
            print("Elemento ---------- Status")
            for elem in self.elements:
                status = "Ingresado Manualmente" if elem in manual else "Autodetectado"
                if len(elem) == 2:
                    print(f"      {elem} ---------- {status}")
                else:
                    print(f"       {elem} ---------- {status}")

#------------------------------------------------------------------------------#

    def validar_familia(self, elemento, familia):
        """Función auxiliar para chequear si una familia existe y es visible"""
        # Rango útil con un pequeño margen para no cortar colas de picos en los bordes
        e_min, e_max = self.E.min() + 0.25, self.E.max() - 0.25
        try:
            # 1. ¿Es excitable por el ánodo?
            if not core.is_excitable(xl.SymbolToAtomicNumber(elemento), familia, self.config):
                return 0
            # 2. ¿Tiene líneas en el rango de energía actual?
            info = core.get_Xray_info(elemento, families=(familia,), config=self.config)
            for line_data in info.values():
                if e_min <= line_data['energy'] <= e_max:
                    return 1
            return 0
        except:
            return 0
#------------------------------------------------------------------------------#

    def identify_calibration_elements(self):
        """
        Identifica el peak de mayor intensidad y devuelve una lista de TODOS los 
        elementos/líneas que solapan en esa región para un ajuste multielemental.
        """
        # 1. Encontrar la energía del máximo global (suavizado ligeramente)
        # Filtramos energías muy bajas (ruido) o muy altas (fuera de rango útil)
        z_anod = xl.SymbolToAtomicNumber(self.config.anode)
        valid_mask = (self.E > 1.0) & (self.E < xl.LineEnergy(z_anod, xl.KA1_LINE))
        idx_max = np.argmax(self.I[valid_mask])
        peak_energy = self.E[valid_mask][idx_max]
        
        calibration_set = []
        threshold_kev = 0.20  # Ventana de solapamiento (aprox 1.5 * FWHM típico)
    
        for elem in self.elements:
            Z = xl.SymbolToAtomicNumber(elem)
            
            # Revisar Capa K
            if core.is_excitable(Z, "K", self.config):
                e_k = xl.LineEnergy(Z, xl.KA1_LINE)
                if abs(e_k - peak_energy) < threshold_kev:
                    calibration_set.append((elem, "K"))
                    continue # Si ya entró por K, no lo evaluamos por L
    
            # Revisar Capa L
            if core.is_excitable(Z, "L", self.config):
                try:
                    e_l = xl.LineEnergy(Z, xl.LA1_LINE)
                    if abs(e_l - peak_energy) < threshold_kev:
                        calibration_set.append((elem, "L"))
                except ValueError:
                    pass
    
        # Bonus: Siempre incluir el Silicio si está presente para anclar la baja energía
        if "Si" in self.elements and ("Si", "K") not in calibration_set:
            calibration_set.append(("Si", "K"))
    
        return calibration_set

#------------------------------------------------------------------------------#

    def get_mask(self, etapa):
        """
        Genera la máscara de parámetros libres.
        Estructura p: [noise, fano, eps, tau, c0, c1, (c2), RK, CK, RL, CL, Area1_K, Area1_L...]
        Etapa: "bkg", "scat"/"resol", K", "L", "M" o "global"
        """
        n_elem = len(self.elements)

        if etapa == "bkg":
            # 1. Parte Base
            # Por defecto: noise, fano, eps, tau, gain, offset, bkg, scat
            mask_base = [0, 0, 0, 0, 0, 0] # tau dependiente de free_tau
            # Fondo y Dispersión
            mask_base += [1] * self.n_bkg # c0, c1...  
            mask_base += [0] * 4     # 4 áreas de dispersión
        elif etapa == "scat":
            # EDXRF CALIBRATION MODE
            # 1. Parte Base
            # Por defecto: noise, fano, eps, tau, gain, offset, bkg, scat
            mask_base = [1, 1, 1, 0, 1, 1] # tau dependiente de free_tau
            # Fondo y Dispersión
            mask_base += [0] * self.n_bkg # c0, c1...  
            mask_base += [1] * 4     # 4 áreas de dispersión
        elif etapa == "resol":
            # TXRF CALIBRATION MODE
            # 1. Parte Base
            # Por defecto: noise, fano, eps, tau, gain, offset, bkg, scat
            mask_base = [1, 1, 1, 0, 1, 1] # tau dependiente de free_tau
            # Fondo y Dispersión
            mask_base += [0] * self.n_bkg # c0, c1...  
            mask_base += [1] * 4     # 4 áreas de dispersión
        elif etapa == "global": 
            # 1. Parte Base
            # Por defecto: noise, fano, eps, tau, gain, offset, bkg, scat
            mask_base = [0, 0, 0, 0, 0, 0] # tau dependiente de free_tau
            # Fondo y Dispersión
            mask_base += [1] * self.n_bkg # c0, c1...
            if getattr(self.config, 'mode', 'EDXRF') == "TXRF":    # Cambiar luego después de testeo
                 mask_base += [1] * 4    # 4 áreas de dispersión
            else:
                 mask_base += [1] * 4    # 4 áreas de dispersión
        else: # K, L, M
            # 1. Parte Base
            # Por defecto: noise, fano, eps, tau, gain, offset, bkg, scat
            mask_base = [0, 0, 0, (1 if self.free_tau else 0), 0, 0] # tau dependiente de free_tau
            # Fondo y Dispersión
            mask_base += [0] * self.n_bkg # c0, c1...  
            if getattr(self.config, 'mode', 'EDXRF') == "TXRF":
                 mask_base += [0] * 4    # 4 áreas de dispersión
            else:
                 mask_base += [0] * 4    # 4 áreas de dispersión
            
        # 2. Parte de Elementos [Area_K, Area_L, Area_M]
        element_masks = []
            
        for elem in self.elements:
            slots = [0, 0, 0] # [K, L, M]

            if etapa == "resol":
                # Identificamos anclas para modo resol
                calibration_list = self.identify_calibration_elements()
                # calibration_list es algo como: [("Si", "K"), ("As", "K"), ("Pb", "L")]
                for target_elem, family in calibration_list:
                    if elem == target_elem:
                        if family == "K": 
                            slots[0] = 1
                        elif family == "L": 
                            slots[1] = 1
                        elif family == "M": 
                            slots[2] = 1
            
            elif etapa == "K":
                slots[0] = self.validar_familia(elem, "K")
            elif etapa == "L":
                slots[1] = self.validar_familia(elem, "L")
            elif etapa == "M":
                slots[2] = self.validar_familia(elem, "M")
            elif etapa == "global":
                # En la global, validamos las tres familias rigurosamente
                slots[0] = self.validar_familia(elem, "K")
                slots[1] = self.validar_familia(elem, "L")
                slots[2] = self.validar_familia(elem, "M")
                
            element_masks.extend(slots)
                
        return mask_base + element_masks 

#------------------------------------------------------------------------------#
    
    def get_p0(self, etapa, mask=None):
        """Genera la semilla base según el modelo de fondo y etapa actual."""
        if etapa == "bkg":
            if self.fondo == "exp_poly":
                # Trabajamos en escala Logarítmica para las semillas
                # 1. Filtramos datos para el fit inicial (evitar log(0) y picos muy altos)
                log_bkg = np.log(np.maximum(self.bkg_snip, 1e-5))
                # 2. Ajuste lineal de Chebyshev (instantáneo y globalmente óptimo)
                # Esto es lo que recomienda Van Espen: linealizar el problema exponencial
                E_min, E_max = self.E.min(), self.E.max()
                E_norm = 2 * (self.E - E_min) / (E_max - E_min) - 1

                fit_cheb = Chebyshev.fit(E_norm, log_bkg, deg=self.grado_fondo)
                semillas_bkg = fit_cheb.coef

            else:
                c0_init = np.min(self.bkg_snip)
                c1_init = (self.bkg_snip[-1] - self.bkg_snip[0]) / (self.E[-1] - self.E[0])
                semillas_bkg = [c0_init, c1_init]
                semillas_bkg += [0.0] * (self.n_bkg - 2)

            # Semillas de dispersión basadas en el máximo del espectro
            max_counts = np.max(self.I)
            p_base = [
                self.noise_init, self.fano_init, self.epsilon_init,
                self.tau_init,
                1.0, 0.0,           # gain, offset
                ]  
            p_base.extend(semillas_bkg)    # c0, c1, ... , cN
            
            # [Ray_K, Com_K, Ray_L, Com_L]
            is_txrf = getattr(self.config, 'mode', 'EDXRF') == "TXRF"
            if is_txrf:
                 p_base += [0.0, 0.0, 0.0, 0.0] 
            else:
                 p_base += [max_counts * 2, max_counts * 1, max_counts * 0.1, max_counts * 0.05]
                 
            for elem in self.elements:
                p_base += [0, 0, 0]    # K, L, M

        elif etapa == "resol":
            # Copiamos lo actual
            p_base = self.p_actual.copy()
             
            # 1. Obtenemos la lista de elementos/familias de calibración
            calibration_list = self.identify_calibration_elements()
            num_calib = len(calibration_list)
            
            # Calculamos la semilla basada en la intensidad máxima repartida
            # Si num_calib es 0 (caso raro), usamos 1 para evitar división por cero
            seed_intensity = np.max(self.I) / max(1, num_calib)
            
            idx = self.offset
            for elem in self.elements:
                # Revisamos cada familia [K, L, M] para este elemento
                for f_idx, family in enumerate(["K", "L", "M"]):
                    # Si esta combinación (elemento, familia) está en nuestra lista VIP, inyectamos semilla
                    if (elem, family) in calibration_list:
                        p_base[idx + f_idx] = seed_intensity
                        
                # Saltamos al siguiente bloque de parámetros (3 slots por elemento)
                idx += 3
                 
        elif etapa == "K":
            # Para K partimos de lo que ya tenemos preajustado del fondo
            p_base = self.p_actual.copy()
            # Inicializar áreas de elementos detectados
            if self.I_net is None: self.I_net = np.maximum(self.I - self.bkg_snip, 0)
            area_init_gral = np.trapezoid(self.I_net, self.E) / len(self.elements)

            idx = self.offset
            for elem in self.elements:
                area_init = area_init_gral * self.validar_familia(elem, "K")
                p_base[idx] = area_init
                idx += 3

        else:
            # Para scat, L, M o global, partimos de lo que ya tenemos ajustado
            p_base = self.p_actual.copy()
            
            if etapa == "L":
                # Inyectamos semilla en slots de L si la máscara lo permite
                for i in range(self.offset, len(p_base)):
                    pos = (i - self.offset) % 3
                    if i < len(mask) and mask[i] == 1:
                        # Slot L es (índice - offset) % 3 == 1
                        if pos == 1:
                            p_base[i] = np.max(self.I_net) * 0.01
                
            
        # --- FILTRADO FINAL ÚNICO ---
        # Esto garantiza que p0_free tenga el mismo largo que los '1' en la máscara
        if mask is not None:
            p0_filtrado = [p for p, f in zip(p_base, mask) if f]
            return p0_filtrado
        
        else: return p_base

#------------------------------------------------------------------------------#

    def run_stage_fit(self, etapa, graf=False, roi_margin=0.4, tol=1e-5):
        free_mask = self.get_mask(etapa)

        # Si es la primera etapa (bkg), inicializamos p_actual con p0 completo
        if etapa == "bkg" or self.p_actual is None:
            self.p_actual = self.get_p0("bkg")

        p0_free = self.get_p0(etapa, free_mask)
        E_min, E_max = self.E.min(), self.E.max()

        if etapa == "bkg":
            def frx_wrapper(E_val, *p_free):
                p_full = core.build_p_from_free(p_free, self.p_actual, free_mask)
                params = core.pack_params(p_full, self.elements, n_bkg=self.n_bkg)
                bkg_params = params["background"]
                if self.fondo == "exp_poly":
                    # 1. Normalización idéntica a la del core (CRÍTICO)
                    # Usamos los límites de la etapa de bkg para que el dominio [-1, 1] sea consistente
                    E_norm = 2 * (E_val - E_min) / (E_max - E_min) - 1
                    # 2. Evaluación con Chebyshev (sin exp, porque ajustamos contra log(SNIP))
                    # Importante: bkg_params ya viene en orden [c0, c1...] desde pack_params
                    return core.chebval(E_norm, bkg_params)
                    
                elif self.fondo == "poly":
                    return core.continuum_bkg(E_val, params, fondo=self.fondo)
        else:
            def frx_wrapper(E_val, *p_free):
                p_full = core.build_p_from_free(p_free, self.p_actual, free_mask)
                params = core.pack_params(p_full, self.elements, n_bkg=self.n_bkg)
                return core.FRX_model_sdd_general(E_val, params, self.t_live, fondo=self.fondo, 
                                                  config=self.config, E_min=E_min, E_max=E_max) 

        # BOUNDS DINÁMICOS
        # Identificamos los índices de los parámetros que SÍ son libres
        indices_libres = [idx for idx, free in enumerate(free_mask) if free]

        lower, upper = [], []
        # Definimos el techo de cuentas: 1.5 veces el máximo del espectro
        # es más que suficiente para cualquier pico real.
        techo_cuentas = np.max(self.I) * 1.5
        for i, val in enumerate(p0_free):
            # Encontrar a qué parámetro real corresponde este p0_free[i]
            # Esto es clave para aplicar límites físicos
            p_idx = indices_libres[i]
            limite_bkg = 6 + self.n_bkg
            
            if p_idx == 0: # Noise
                lower.append(0.0035); upper.append(0.15)
            elif p_idx == 1: # Fano Factor
                lower.append(0.08);  upper.append(0.14)  # Rango físico estricto
            elif p_idx == 2: # Epsilon
                lower.append(0.0035); upper.append(0.0038) # Casi fijo, pero con un mínimo margen
            elif p_idx == 3: # Tau pileup
                lower.append(0.0); upper.append(1e-5) # Máximo 10 microsegundos
            elif p_idx == 4: # Gain Corr
                lower.append(0.98);   upper.append(1.02) # +/- 2% ganancia
            elif p_idx == 5: # Offset Corr
                lower.append(-0.05);  upper.append(0.05) # +/- 50 eV shift
            # --- Parámetros de Fondo y Dispersión (Indices 6 hasta offset) ---
            elif p_idx < self.offset:
                # Detectamos si es un coeficiente del polinomio de fondo
                # Asumiendo estructura: [Noise...Offset, c0, c1, c2, Ray, Com...]
                # Los coeficientes suelen ser los primeros después del índice 5        
                if 6 <= p_idx < limite_bkg:
                    if self.fondo == "exp_poly":
                        lower.append(-100.0); upper.append(100.0)
                    else:
                        lower.append(-np.inf); upper.append(np.inf)
                else:
                    # Es Rayleigh o Compton
                    lower.append(0.0); upper.append(np.inf)
            
            # --- Áreas de Elementos ---
            else: # Áreas de elementos
                lower.append(0.0); upper.append(techo_cuentas)
                
        bounds = (lower, upper)
        
        if etapa not in ["bkg", "scat", "resol"]:
            roi_mask = prc.generar_mascara_roi(self.E, self.elements, margen=roi_margin)

        if etapa == "bkg":
            y_target = np.log(np.maximum(self.bkg_snip, 1e-5)) if self.fondo == "exp_poly" else self.bkg_snip
            popt, pcov = curve_fit(frx_wrapper, 
                                   self.E, 
                                   y_target, 
                                   p0=p0_free, 
                                   bounds=bounds,
                                   method='trf', # Altamente recomendado para bounds
                                   xtol=tol, 
                                   ftol=tol
                                   )
        elif etapa == "scat" or etapa == "resol":
            popt, pcov = curve_fit(frx_wrapper, 
                                   self.E, 
                                   self.I,
                                   p0=p0_free,
                                   bounds=bounds,
                                   method='trf',
                                   xtol=tol, 
                                   ftol=tol
                                   )
        elif etapa in ["K", "L"]:    
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
                                   xtol=tol*2, 
                                   ftol=tol*2 
                                   )
            self.pcov = pcov 
            
        self.p_actual = core.build_p_from_free(popt, self.p_actual, free_mask)
        opt_params = core.pack_params(self.p_actual, self.elements, n_bkg=self.n_bkg)
        
        if etapa == "bkg":
            self.bkg_fit = core.continuum_bkg(self.E, opt_params, fondo=self.fondo, E_min=E_min, E_max=E_max)
        else:
            self.bkg_fit = core.continuum_bkg(self.E, opt_params, fondo=self.fondo,  E_min=E_min, E_max=E_max)
            self.I_fit = frx_wrapper(self.E, *popt) 

        if graf:
            if etapa == "bkg":
                mtr.graficar_fondo(self.E, self.I, self.bkg_fit, self.bkg_snip, fondo=self.fondo, grado_fondo=self.grado_fondo)
            elif etapa == "scat" or etapa == "resol":
                mtr.graficar_ajuste(self.E, self.I, self.I_fit, self.bkg_fit, self.elements, 
                                    popt, self.p_actual, [etapa], n_bkg=self.n_bkg, config=self.config)   
            elif etapa == "K":
                mtr.graficar_ajuste(self.E, self.I, self.I_fit, self.bkg_fit, self.elements, 
                                    popt, self.p_actual, [etapa], n_bkg=self.n_bkg,
                                    umbral_area_familia=0.5,
                                    umbral_ratio_linea=0.1, config=self.config)
            elif etapa == "L":
                mtr.graficar_ajuste(self.E, self.I, self.I_fit, self.bkg_fit, self.elements, 
                                    popt, self.p_actual, [etapa], n_bkg=self.n_bkg,
                                    config=self.config)
            elif etapa == "M":
                mtr.graficar_ajuste(self.E, self.I, self.I_fit, self.bkg_fit, self.elements, 
                                    popt, self.p_actual, [etapa],  n_bkg=self.n_bkg,
                                    umbral_area_familia=0,
                                    umbral_ratio_linea=0.1, config=self.config)
            else: # global
                mtr.graficar_ajuste(self.E, self.I, self.I_fit, self.bkg_fit, self.elements, 
                                    popt, self.p_actual, ["K", "L", "M"], n_bkg=self.n_bkg,
                                    umbral_ratio_linea=0.75, config=self.config)


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
        """Ejecuta el pipeline completo de ajuste secuencial según el modo del instrumento."""
        mode = getattr(self.config, 'mode', 'EDXRF') # Default a EDXRF por seguridad
        
        print(f" > Iniciando Deconvolución (Modo: {mode}):")
        # Preajuste de Fondo
        stop_event = threading.Event()
        t = threading.Thread(
            target=self.animacion_carga,
            args=(stop_event, "  > Preajustando Fondo Continuo")
        )
        t.start()
        self.run_stage_fit('bkg', graf=graf, roi_margin=roi_margin, tol=tol)
        stop_event.set()
        t.join()
        # --- BIFURCACIÓN DEL PIPELINE ---
        if mode == "EDXRF":
            stop_event = threading.Event()
            t = threading.Thread(
                target=self.animacion_carga,
                args=(stop_event, "  > Ajustando Dispersión (Rayleigh/Compton)")
            )
            t.start()
            self.run_stage_fit('scat', graf=graf, roi_margin=roi_margin, tol=tol)
            stop_event.set()
            t.join()

        elif mode == "TXRF":
            stop_event = threading.Event()
            t = threading.Thread(
                target=self.animacion_carga,
                args=(stop_event, "  > Calibrando Energía")
            )
            t.start()
            self.run_stage_fit('resol', graf=graf, roi_margin=roi_margin, tol=tol)
            stop_event.set()
            t.join()
        # Ajustes de capas K, L y M
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
        # Refinamiento Global
        stop_event = threading.Event()
        t = threading.Thread(
            target=self.animacion_carga,
            args=(stop_event, "  > Refinando Ajuste Global")
        )
        t.start()
        self.run_stage_fit('global', graf=graf, roi_margin=roi_margin, tol=tol)
        stop_event.set()
        t.join() 
        # El diccionario final empaquetado
        self.p_dict = core.pack_params(self.p_actual, self.elements, n_bkg=self.n_bkg)
        print(f"[{self.name}] Deconvolución finalizada con éxito.")

#------------------------------------------------------------------------------#

    def report(self, filename=None, pdf=False):
        """Genera el PDF y muestra el reporte en consola."""
        if self.I_fit is None:
            raise ValueError("Debe ejecutar run_full_fit() antes de generar un reporte.")
            
        fname = filename if filename else f"Reporte_{self.name}.pdf"

        mtr.generar_reporte_completo(self.E, self.I, self.I_fit, self.bkg_fit, 
                                        self.p_actual, self.elements, 
                                        nombre_muestra=self.name, n_bkg=self.n_bkg,
                                        config=self.config)
        if pdf:
            mtr.exportar_reporte_pdf(self.E, self.I, self.I_fit, self.bkg_fit, 
                                    self.p_actual, self.elements, 
                                    nombre_muestra=self.name, 
                                    archivo=fname, n_bkg=self.n_bkg,
                                    config=self.config)

#------------------------------------------------------------------------------#

    def run_resolution_check(self):
        """Compara la resolución ajustada contra la nominal del equipo.
           Aplicar idealmente a ajuste completo.
        """
        params = core.pack_params(self.p_actual, self.elements, n_bkg=self.n_bkg)
        mtr.check_resolution_health(params, self.config)

#------------------------------------------------------------------------------#

    def generar_tabla_resultados(self):
        res = []        
        for elem, data in self.p_dict["elements"].items():
            row = {"Elemento": elem}
            # Solo agregamos si el área es significativa (mayor a un umbral)
            if data["area_K"] > 0: row["Area_K"] = f"{data['area_K']:.2e}"
            if data["area_L"] > 0: row["Area_L"] = f"{data['area_L']:.2e}"
            if data["area_M"] > 0: row["Area_M"] = f"{data['area_M']:.2e}"
            
            if len(row) > 1: # Si tiene al menos una familia ajustada
                res.append(row)
                
        df = pd.DataFrame(res).fillna("-")
        return df





























































































