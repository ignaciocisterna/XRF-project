import numpy as np
import xraylib as xl
import xraydb
from scipy.special import voigt_profile
from numpy.polynomial.chebyshev import chebval

# Información de emisiones y probabilidades radiativas
def get_Xray_info(symb, families=("K", "L", "M"), config=None, E_ref=None):
    """
    Obtiene líneas de emisión usando Secciones Eficaces de Producción
    para corregir las proporciones entre subcapas.
    
    E_ref: Energía de excitación de referencia (keV). 
           Idealmente la energía Ka del ánodo (ej. 20.21 para Rh).
    """
    Z = xl.SymbolToAtomicNumber(symb)

    # --- Configuración de Energía de Excitación ---
    if config and not E_ref:
        Z_anode = xl.SymbolToAtomicNumber(config.anode)
        E_ref = xl.LineEnergy(Z_anode, xl.KA1_LINE)
    elif E_ref is None:
        # Si no nos dan energía de referencia, usamos una alta por defecto (50 keV)
        # para asegurar que calculamos ratios válidos incluso si el ánodo es ligero.
        E_ref = 50.0
        
    # Diccionarios maestros de líneas 
    LINE_MAPS = {
        "K": {
            "Ka1": xl.KA1_LINE, 
            "Ka2": xl.KA2_LINE,
            "Kb1": xl.KB1_LINE, 
            "Kb3": xl.KB3_LINE, 
            "Kb5": xl.KB5_LINE,
        },
        "L": {
            "La1": xl.LA1_LINE, 
            "La2": xl.LA2_LINE, 
            "Lb1": xl.LB1_LINE,
            "Lb2": xl.LB2_LINE, 
            "Lb3": xl.LB3_LINE, 
            "Lb4": xl.LB4_LINE,
            "Lg1": xl.LG1_LINE, 
            "Ll" : xl.LL_LINE,  
            "Le" : xl.LE_LINE,
        },
        "M": {
            "Ma1": xl.MA1_LINE, 
            "Ma2": xl.MA2_LINE, 
            "Mb" : xl.MB_LINE,  
            "Mg" : xl.MG_LINE,
        }
    }

    # Mapeo de líneas a sus niveles de transición para calcular el ancho (gamma)
    # Formato: (Nivel Inicial, Nivel Final)
    LINE_LEVELS = {
        "Ka1": (xl.K_SHELL, xl.L3_SHELL),
        "Ka2": (xl.K_SHELL, xl.L2_SHELL),
        "Kb1": (xl.K_SHELL, xl.M3_SHELL),
        "Kb3": (xl.K_SHELL, xl.M2_SHELL),
        "Kb5": (xl.K_SHELL, xl.M4_SHELL),
        
        "La1": (xl.L3_SHELL, xl.M5_SHELL),
        "La2": (xl.L3_SHELL, xl.M4_SHELL),
        "Lb1": (xl.L2_SHELL, xl.M4_SHELL),
        "Lb2": (xl.L3_SHELL, xl.N5_SHELL),
        "Lb3": (xl.L1_SHELL, xl.M3_SHELL),
        "Lb4": (xl.L1_SHELL, xl.M2_SHELL),
        "Lg1": (xl.L2_SHELL, xl.N4_SHELL),
        "Ll" : (xl.L3_SHELL, xl.M1_SHELL),
        "Le" : (xl.L2_SHELL, xl.M1_SHELL),
        
        "Ma1": (xl.M5_SHELL, xl.N7_SHELL),
        "Ma2": (xl.M5_SHELL, xl.N6_SHELL),
        "Mb":  (xl.M4_SHELL, xl.N6_SHELL),
        "Mg":  (xl.M3_SHELL, xl.N5_SHELL),
    }

    # Diccionario de traducción: { Nombre_XrayLib : [Posibles_Nombres_Elam] }
    ALIAS = {
        "Le": "Ln",
        "Lb2": "Lb2,15",
        "Ma1": "Ma",
        "Ma2": "Ma",
        #"Mb": "Mb", # Por si acaso
        #"Mg": "Mg"
    }
    
    info = {}
    try:
        elam_table = xraydb.xray_lines(symb)
        ###########################Temporal#############################
        #if symb == "Pb":
        #    print(f"\n--- Líneas Elam detectadas para {symb} ---")
        #    for k, v in elam_table.items():
        #        # v[0] es la energía en eV
        #        print(f"Línea: {k:6} | Energía: {v[0]/1000.0:.4f} keV")
        ################################################################
    except:
        elam_table = {}
        print(f"Advertencia: No se pudo cargar XrayDB para {symb}")

    for fam in families:
        if fam not in LINE_MAPS: continue

        temp_family_info = {}
        for name, line_code in LINE_MAPS[fam].items():
            # Extraer alias si lo hay
            if name in ALIAS:
                alt_name = ALIAS[name]
            else: 
                alt_name = None
                
            # 1. Obtener Energía de forma segura
            if name in elam_table:
                energy_ev = getattr(elam_table[name], 'energy', elam_table[name][0])
                energy = energy_ev / 1000.0  # Pasar a keV
                source = "Elam"
            elif alt_name in elam_table:
                energy_ev = getattr(elam_table[alt_name], 'energy', elam_table[alt_name][0])
                energy = energy_ev / 1000.0  # Pasar a keV
                source = "Elam"
            else:
                try:
                    energy = xl.LineEnergy(Z, line_code)
                    source = "XrayLib"
                except ValueError:
                    # ¡AQUÍ ESTABA EL ERROR! Elementos ligeros no tienen líneas L o M.
                    continue 
                
            if energy <= 0: continue

            # 2. Obtener Intensidad (CS) de forma robusta con Fallback
            try:
                cs = xl.CS_FluorLine_Kissel(Z, line_code, E_ref)
            except ValueError:
                cs = 0.0
                
            # --- EL RESCATE VITAL (Rescatado de v1.1.5) ---
            # Si E_ref no la excita, forzamos con 100 keV para obtener las proporciones, 
            # asumiendo que el Bremsstrahlung del tubo hará el trabajo empírico.
            if cs <= 0:
                try:
                    cs = xl.CS_FluorLine_Kissel(Z, line_code, 100.0)
                except ValueError:
                    pass
                    
            if cs <= 0: continue

            # 3. Calcular Gamma
            try:
                l1, l2 = LINE_LEVELS[name]
                gamma = (xl.AtomicLevelWidth(Z, l1) + xl.AtomicLevelWidth(Z, l2)) / 2.0
            except:
                gamma = 0.001

            temp_family_info[name] = {
                "energy": energy, 
                "ratio": cs, 
                "gamma": gamma, 
                "family": fam, 
                "source": source
            }

        # --- NORMALIZACIÓN POR FAMILIA ---
        if temp_family_info:
            max_cs = max(d['ratio'] for d in temp_family_info.values())
            for name in temp_family_info:
                temp_family_info[name]['ratio'] /= max_cs
            info.update(temp_family_info)

    if not info:
        raise ValueError(f"No valid X-ray lines found for {symb}")

    return info

# --- FÍSICA Y RESOLUCIÓN ---

# Correción de ganacia y deriva energéticas
def energy_corr(E, gain, offset):
    return E * gain + offset

# Resolución Energética para detector tipo SDD (usado en PICOFOX)

# Para modelo
def sigma_E(E, noise=0.06, fano=0.115, epsilon=0.00365):
    """
    E: Energía en keV
    noise: Ruido electrónico constante (FWHM en keV)
    fano: Factor de Fano (adimensional, ~0.115 para Si)
    epsilon: Energía de creación de par en Si (en keV, 0.00365)
    """
    cte_conv_geo_Gauss = 2.355
    # Contribución de ruido electrónico
    noise_term = noise**2
    # Contribución de Fano
    fano_term = (cte_conv_geo_Gauss**2) * fano * epsilon * E
    # FWHM total
    fwhm_total = np.sqrt(noise_term + fano_term)
    # Desviación estándar para perfil Voight
    return fwhm_total / cte_conv_geo_Gauss

# Para SNIP
def fwhm_SNIP(E, gamma=1.2):
    a = 0.0057    # Datos de S2 PICOFOX
    b = 0.00252   # Datos de S2 PICOFOX
    return gamma * np.sqrt(a + b*E)

# Energía de los peaks de dispersión Compton
def get_compton_energy(E0, angle_deg):
    """
    E0: Energía incidente en keV
    angle_deg: Ángulo de dispersión (90° típico en TXRF)
    """
    m_e_c2 = 510.99895  # Energía en reposo del electrón en keV
    angle_rad = np.radians(angle_deg)
    return E0 / (1 + (E0 / m_e_c2) * (1 - np.cos(angle_rad)))

# Eficiencia del detector
def get_efficiency(energy, config):
    """
    Calcula la eficiencia intrínseca del detector usando datos de config.
    """
    # Densidades (g/cm3)
    rho_be = xl.ElementDensity(4)     # Be
    rho_si = xl.ElementDensity(14)    # Si
    
    # Espesores desde config
    x_be = config.win_thick * 1e-4  # um a cm
    x_si = config.det_thick * 1e-1  # mm a cm
    # Si no existe dead_layer en el JSON, usamos 0.05 um por defecto
    x_dead = config.dead_layer * 1e-4 
    
    # 1. Transmisión ventana Be
    mu_be = xl.CS_Total(4, energy) 
    trans_be = np.exp(-mu_be * rho_be * x_be)
    
    # 2. Transmisión/Absorción en Si
    mu_si = xl.CS_Total(14, energy)
    # Transmisión a través de la capa muerta
    trans_dead = np.exp(-mu_si * rho_si * x_dead)
    # Absorción en el volumen activo
    abs_si = 1.0 - np.exp(-mu_si * rho_si * x_si)
    
    return trans_be * trans_dead * abs_si

# Probabilidad de peak de escape
def get_escape_ratio(E0):
    """
    Calcula la probabilidad de escape de Si Ka usando el modelo de Reed.
    Todo se obtiene dinámicamente de xraylib.
    """
    Z_si = 14
    edge_si = xl.EdgeEnergy(Z_si, xl.K_SHELL)
    
    if E0 < edge_si:
        return 0.0
    
    # Energías y constantes
    E_si = xl.LineEnergy(Z_si, xl.KA1_LINE)
    omega_k = xl.FluorYield(Z_si, xl.K_SHELL)
    jump_k = xl.JumpFactor(Z_si, xl.K_SHELL)
    
    # Probabilidad de fotoabsorción en la capa K
    r_k = (jump_k - 1) / jump_k
    
    # Coeficientes de atenuación masica (Total es buena aprox para CS_Photo en Si)
    mu_inc = xl.CS_Total(Z_si, E0)
    mu_si = xl.CS_Total(Z_si, E_si)
    
    # Modelo de Reed: 0.5 * omega * r * [1 - (mu_det/mu_inc) * ln(1 + mu_inc/mu_det)]
    ratio = 0.5 * omega_k * r_k * (1 - (mu_si / mu_inc) * np.log(1 + mu_inc / mu_si))
    
    return max(0, ratio)

# Modificador de peak para añadir peak de escape y suma
def add_detector_artifacts(E, spectrum, area, E0, gamma, params, live_time, config, doppler_width=0.0):
    """
    Agrega picos de Escape y Suma a un espectro a partir de un pico principal.
    
    area: área del pico principal.
    E0: energía del pico principal.
    params: debe contener 'tau_pileup' y 'live_time'.
    """
    # 1. PICO DE ESCAPE (Si K-alpha)
    # Usamos la función zero-hardcode que definimos antes
    ratio_esc = get_escape_ratio(E0)
    if ratio_esc > 0:
        E_esc = E0 - xl.LineEnergy(14, xl.KA1_LINE)  # Energía desplazada por el escape del Si
        # Recalculamos la resolución para la nueva energía
        s_esc = sigma_E(E_esc, params["noise"], params["fano"], params["epsilon"])
        # La sigma total hereda la varianza Doppler del padre (si la hay)
        s_esc_total = np.sqrt(s_esc**2 + doppler_width**2)
        spectrum += voigt_peak(E, area * ratio_esc, E_esc, s_esc_total, gamma)

    # 2. PICO DE SUMA (Pile-up: E0 + E0)
    # Obtenemos tau del diccionario de parámetros
    tau = params.get("tau_pileup", 0.0)
    
    if tau > 0 and area > 0:
        # Probabilidad de suma: R * tau, donde R es la tasa de cuentas (Area / T_live)
        prob_sum = (area / live_time) * tau
        E_sum = E0 * 2
        s_sum = sigma_E(E_sum, params["noise"], params["fano"], params["epsilon"])
        # En la suma, la varianza Doppler se suma (Doppler1 + Doppler2) -> factor sqrt(2)
        s_sum_total = np.sqrt(s_sum**2 + 2 * (doppler_width**2))

       # El ancho natural (Lorentziano) se duplica en una convolución simple
        gamma_sum = gamma * 2
        
        # Eliminamos eff_corr: el pileup ocurre con fotones que ya superaron la eficiencia
        spectrum += voigt_peak(E, area * prob_sum, E_sum, s_sum_total, gamma_sum)
        
    return spectrum

# --- MODELO DE ESPECTRO ---

# Perfil de los peaks elementales y dispersión de Rayleigh
def voigt_peak(E, area, E0, sigma, gamma):
    """
    area : área total del pico
    gamma: componente Lorentziana (cola instrumental)
    """
    return area * voigt_profile(E - E0, sigma, gamma)

def get_doppler_width(E_inc, angle_deg):
    """
    Calcula una sigma de Doppler aproximada.
    Para TXRF con muestras ligeras, el ensanchamiento es ~0.3% de la E_inc.
    """
    m_e_c2 = 510.99895  # Energía en reposo del electrón en keV
    angle_rad = np.radians(angle_deg)
    # Constante de momento electrónico típico de la matriz (ajustar)
    cte_mom_elec = 0.362
    factor_geometrico = np.sin(angle_rad / 2)
    return (E_inc**2 / m_e_c2) * factor_geometrico *  cte_mom_elec

# Identificador de familias de señal
def line_family(line_name):
    if line_name.startswith("K"):
        return "K"
    if line_name.startswith("L"):
        return "L"
    if line_name.startswith("M"):
        return "M"
    return None

# Identificador de excitabilidad
def is_excitable(Z, family, config):
    """
    Chequea si el ánodo del equipo puede excitar una familia específica.
    Usa la línea Ka1 del ánodo como referencia de energía de excitación.
    """
    # 1. Obtener energía de la línea Ka1 del ánodo
    Z_anode = xl.SymbolToAtomicNumber(config.anode)
    E_excit = xl.LineEnergy(Z_anode, xl.KA1_LINE)
    
    # 2. Mapeo de bordes de absorción principales
    SHELLS = {"K": xl.K_SHELL, "L": xl.L3_SHELL, "M": xl.M5_SHELL}
    
    try:
        if family in SHELLS:
            edge = xl.EdgeEnergy(Z, SHELLS[family])
            return edge < E_excit
        return False
    except ValueError:
        return False

# Modelo Fondo continuo
def continuum_bkg(E, params, fondo="poly", E_min=None, E_max=None):
    """ Función del fondo ajustable """
    bkg_coeffs = params["background"] # Ya viene con el largo correcto
    
    if fondo == "poly":
        # Calculamos el polinomio base (funciona para cualquier grado)
        return np.polyval(bkg_coeffs[::-1], E)
        
    elif fondo == "exp_poly":
        E_norm = 2 * (E - E_min) / (E_max - E_min) - 1
        poly = chebval(E_norm, bkg_coeffs)
        return np.exp(np.clip(poly, -700, 700))

# Modelo Dispersión
def add_anode_scattering(spectrum, E, params, live_time, config):
    """
    Agrega los picos de dispersión (Rayleigh y Compton) del ánodo.
    Detecta automáticamente las familias K y L del material del ánodo.
    """
    # 1. Parámetros instrumentales globales
    noise, fano, epsilon = params["noise"], params["fano"], params["epsilon"]
    # 2. Obtenemos información física del ánodo
    tube_info = get_Xray_info(config.anode, families=("K", "L"), E_ref=config.voltaje)
    # 3. Extraemos parámetros de amplitud (Ajustados por el fit)
    # Ahora esperamos: scat_ray_K, scat_com_K, scat_ray_L, scat_com_L
    scat_areas = params.get("scat_areas", {}) 
    
    for line_name, data in tube_info.items():
        E_tube = data["energy"]
        ratio = data["ratio"]
        gamma_tube = data["gamma"]
        fam = line_family(line_name) # "K" o "L"
    
        # Obtenemos el área base para esta familia (si no existe, 0)
        a_ray = scat_areas.get(f"ray_{fam}", 0.0)
        a_com = scat_areas.get(f"com_{fam}", 0.0)
        
        # --- RAYLEIGH (Elástico) ---
        if a_ray > 0:
            # Calculamos el área EFECTIVA (lo que realmente "ve" el detector)
            area_efectiva = a_ray * ratio * get_efficiency(E_tube, config)
            s_ray = sigma_E(E_tube, noise, fano, epsilon)
            
            # Agregamos el pico principal al espectro
            spectrum += voigt_peak(E, area_efectiva, E_tube, s_ray, gamma_tube) 
            
            # Generamos sus artefactos (Rayleigh no tiene Doppler, pasamos 0.0)
            add_detector_artifacts(E, spectrum, area_efectiva, E_tube, gamma_tube, 
                                   params, live_time, config)
            
        # --- COMPTON (Inelástico) ---
        if a_com > 0:
            # Calcular cambio de energía según ángulo
            E_com = get_compton_energy(E_tube, config.angle)
            # Verificar si el pico cae dentro del rango del espectro
            if E.min() < E_com < E.max():
                area_efectiva = a_com * ratio * get_efficiency(E_com, config)
                doppler_w = get_doppler_width(E_tube, config.angle)
                s_com = sigma_E(E_com, noise, fano, epsilon)
                s_com_total = np.sqrt(s_com**2 + doppler_w**2)
                
                # Agregamos el pico Compton (gamma residual muy pequeño)
                gamma_compton = 0.01
                spectrum += voigt_peak(E, area_efectiva, E_com, s_com_total, gamma_compton)
                
                # Generamos artefactos PASANDO la anchura Doppler para que la hereden
                add_detector_artifacts(E, spectrum, area_efectiva, E_com, gamma_compton, 
                                       params, live_time, config, doppler_width=doppler_w)
                
    return spectrum

# Modelo Espectro
def FRX_model_sdd_general(E_raw, params, live_time, fondo="poly", config=None, E_min=None, E_max=None):
    """
    Modelo FRX generalizado con áreas independientes por familia K, L y M.
    La excitación del ánodo y los efectos instrumentales están absorbidos
    en area_K, area_L y area_M respectivamente.
    """

    gain = params.get("gain_corr", 1.0)
    offset = params.get("offset_corr", 0.0)
    E = energy_corr(E_raw, gain, offset)
    
    noise, fano, epsilon = params["noise"], params["fano"], params["epsilon"]

    spectrum = np.zeros_like(E)

    for elem, elem_params in params["elements"].items():

        try:
            info = get_Xray_info(elem, config=config)
        except ValueError:
            continue

        Z = xl.SymbolToAtomicNumber(elem)

        for line, line_data in info.items():
            E0 = line_data["energy"]
            r  = line_data["ratio"]
            gamma = line_data["gamma"]

            fam = line_family(line)
            if fam is None:
                continue

            A = elem_params.get(f"area_{fam}", 0.0)
            if A <= 0:
                continue

            # Excitación por el ánodo
            if not is_excitable(Z, fam, config):
                continue

            # Solo líneas dentro del rango experimental
            if E0 < E.min() or E0 > E.max():
                continue

            sigma = sigma_E(E0, noise, fano, epsilon)
            # 1. Pico Principal: Área * Ratio * Perfil * Eficiencia_Instrumental
            spectrum += voigt_peak(E, A * r, E0, sigma, gamma) * get_efficiency(E0, config)

            # 2. Artefactos: Escape y Suma
            # Nota: Pasamos el área para que la función maneje los satélites
            spectrum = add_detector_artifacts(E, spectrum, A * r, E0, gamma, params, live_time, config)

    # --- PICOS DE DISPERSIÓN ---
    spectrum = add_anode_scattering(spectrum, E, params, live_time, config)
  
   # --- FONDO CONTINUO ---
    background = continuum_bkg(E, params, fondo=fondo,  E_min=E_min, E_max=E_max)
    
    # spectrum es la suma de picos calculada previamente
    return spectrum + background

# --- TRATAMIENTO DE PARÁMETROS ---

# Vectorización de parámetros
def pack_params(p, elements, n_bkg=2):
    """
    Empaqueta los parámetros en un diccionario, adaptándose al modelo de fondo.
    """
    params = {
        "noise": p[0], "fano": p[1], "epsilon": p[2],
        "tau_pileup": p[3],
        "gain_corr": p[4], "offset_corr": p[5]
    }
    
    idx_start = 6
    idx_end = idx_start + n_bkg
    
    params["background"] = p[idx_start:idx_end]

    params["scat_areas"] = {
            "ray_K": p[idx_end], "com_K": p[idx_end + 1], 
            "ray_L": p[idx_end + 2], "com_L": p[idx_end + 3]
        }
    idx = idx_end + 4
    # Empaquetado de elementos (común a ambos)
    params["elements"] = {}
    for elem in elements:
        params["elements"][elem] = {
            "area_K": p[idx],
            "area_L": p[idx + 1],
            "area_M": p[idx + 2],
        }
        idx += 3
    return params

def build_p_from_free(p_free, p_fixed, free_mask):
    """
    Reconstruye el vector completo de parámetros
    a partir de los libres y fijos.
    """
    p = np.zeros(len(free_mask))
    j = 0
    for i, free in enumerate(free_mask):
        if free:
            p[i] = p_free[j]
            j += 1
        else:
            p[i] = p_fixed[i]
    return p





























































