import numpy as np
import xraylib as xl
from scipy.special import voigt_profile

# Información de emisiones y probabilidades radiativas
def get_Xray_info(symb, families=("K", "L", "M")):
    Z = xl.SymbolToAtomicNumber(symb)
    K_LINES = {
        "Ka1": xl.KA1_LINE,
        "Ka2": xl.KA2_LINE,
        "Kb1": xl.KB1_LINE,
        "Kb3": xl.KB3_LINE,
        "Kb5": xl.KB5_LINE,
    }

    L_LINES = {
        "La1": xl.LA1_LINE,
        "La2": xl.LA2_LINE,
        "Lb1": xl.LB1_LINE,
        "Lb2": xl.LB2_LINE,
        "Lb3": xl.LB3_LINE,
        "Lb4": xl.LB4_LINE,
        "Lg1": xl.LG1_LINE,
    }

    M_LINES = {
        "Ma1": xl.MA1_LINE,
        "Ma2": xl.MA2_LINE,
        "Mb":  xl.MB_LINE,
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
        "Ma1": (xl.M5_SHELL, xl.N7_SHELL),
        "Ma2": (xl.M5_SHELL, xl.N6_SHELL),
        "Mb":  (xl.M4_SHELL, xl.N6_SHELL),
    }
    
    info = {}

    families_def = {
        "K": (K_LINES, "Ka1"),
        "L": (L_LINES, "La1"),
        "M": (M_LINES, "Ma1"),
    }

    for fam in families:
        lines, ref_name = families_def[fam]

        # línea de referencia (Ka1, La1, Ma1)
        try:
            ref_rate = xl.RadRate(Z, lines[ref_name])
        except ValueError:
            continue

        if ref_rate <= 0:
            continue

        for name, line_id in lines.items():
            try:
                energy = xl.LineEnergy(Z, line_id)
                rate = xl.RadRate(Z, line_id)

                if energy > 0 and rate > 0:
                    # --- CÁLCULO DE GAMMA (Lorentziano) ---
                    # El ancho de la línea es la suma de los anchos de los niveles involucrados
                    lvl_init, lvl_final = LINE_LEVELS[name]
                    gamma_total_keV = (xl.AtomicLevelWidth(Z, lvl_init) + 
                                       xl.AtomicLevelWidth(Z, lvl_final))
                    
                    # Para voigt_profile(sigma, gamma), gamma suele ser el HWHM 
                    # xraylib devuelve el ancho total (FWHM), por eso dividimos por 2
                    gamma_hwhm = gamma_total_keV / 2.0
                    
                    info[name] = {
                        "energy": energy,
                        "ratio": rate / ref_rate,
                        "gamma": gamma_hwhm
                    }

            except ValueError:
                pass

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
    rho_be = xl.ElementDensity(4)
    rho_si = xl.ElementDensity(14)
    
    # Espesores desde config
    x_be = config.win_thick * 1e-4  # um a cm
    x_si = config.det_thick * 1e-1  # mm a cm
    # Si no existe dead_layer en el JSON, usamos 0.05 um por defecto
    x_dead = config.dead_layer * 1e-4 
    
    # 1. Transmisión ventana Be
    mu_be = xl.CS_Total(4, E) 
    trans_be = np.exp(-mu_be * rho_be * x_be)
    
    # 2. Transmisión/Absorción en Si
    mu_si = xl.CS_Total(14, E)
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
def add_detector_artifacts(E, spectrum, area, E0, sigma, gamma, params, live_time, config):
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
        E_esc = E0 - 1.74  # Energía desplazada por el escape del Si
        # Recalculamos la resolución para la nueva energía
        s_esc = sigma_E(E_esc, params["noise"], params["fano"], params["epsilon"])
        spectrum += voigt_peak(E, area * ratio_esc, E_esc, s_esc, gamma)

    # 2. PICO DE SUMA (Pile-up: E0 + E0)
    # Obtenemos tau del diccionario de parámetros
    tau = params.get("tau_pileup", 0.0)
    
    if tau > 0 and area > 0:
        # Probabilidad de suma: R * tau, donde R es la tasa de cuentas (Area / T_live)
        prob_sum = (area / live_time) * tau
        E_sum = E0 * 2
        s_sum = sigma_E(E_sum, params["noise"], params["fano"], params["epsilon"])

        # Corrección de eficiencia: El detector es menos eficiente a E_sum que a E0
        # Multiplicamos por (Eff_sum / Eff_original)
        eff_corr = get_efficiency(E_sum, config) / get_efficiency(E0, config)
        
        # El pico de suma hereda el perfil (gamma) del pico original
        spectrum += voigt_peak(E, area, E_sum, s_sum, gamma) * prob_sum * eff_corr
        
    return spectrum

# --- MODELO DE ESPECTRO ---

# Perfil de los peaks elementales y dispersión de Rayleigh
def voigt_peak(E, area, E0, sigma, gamma):
    """
    area : área total del pico
    gamma: componente Lorentziana (cola instrumental)
    """
    return area * voigt_profile(E - E0, sigma, gamma)

def get_doppler_width(E_inc):
    """
    Calcula una sigma de Doppler aproximada.
    Para TXRF con muestras ligeras, el ensanchamiento es ~0.3% de la E_inc.
    """
    # Constante empírica para un ajuste balanceado (puede requerir ajuste para muestras sólidas)
    cte_ensanchamiento = 0.0028    
    return E_inc * cte_ensanchamiento  

# Perfil de los peaks de dispersión de Compton
def compton_peak(E, area, E_com, E_inc, sigma_inst):
    """
    E: Vector de energías.
    E_com: Centro del pico Compton.
    E_inc: Energía original del fotón del ánodo.
    sigma_inst: Resolución instrumental (sigma_E).
    """
    # Calcular el ensanchamiento de Doppler
    doppler_width = get_doppler_width(E_inc)
    
    # La sigma total es la suma cuadrática de la resolución y el Doppler
    sigma_total = np.sqrt(sigma_inst**2 + doppler_width**2)
    
    # Usamos una Gaussiana (o Voigt con gamma muy pequeña)
    # El perfil Compton tiende a ser más ancho que el Rayleigh
    return area * voigt_profile(E - E_com, sigma_total, 0.001)


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

# Modelo
def FRX_model_sdd_general(E_raw, params, live_time, config):
    """
    Modelo FRX generalizado con áreas independientes por familia K, L y M.
    La excitación de Mo y los efectos instrumentales están absorbidos
    en area_K, area_L y area_M respectivamente.
    """

    gain = params.get("gain_corr", 1.0)
    offset = params.get("offset_corr", 0.0)
    E = energy_corr(E_raw, gain, offset)
    
    noise, fano, epsilon = params["noise"], params["fano"], params["epsilon"]

    spectrum = np.zeros_like(E)

    for elem, elem_params in params["elements"].items():

        try:
            info = get_Xray_info(elem)
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

            # Excitación por Mo
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
            spectrum = add_detector_artifacts(E, spectrum, A * r, E0, sigma, gamma, params, live_time, config)

    # --- PICOS DE DISPERSIÓN ---
    # Rayleigh: Elástico ; Compton: Inelástico (depende del ángulo del detector)
    
    # Obtenemos info completa del ánodo (K y L)
    tube_info = get_Xray_info(config.anode, families=("K", "L"))
    
    # Extraemos las áreas de dispersión del diccionario params
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
    
        if a_ray > 0:
            s_ray = sigma_E(E_tube, noise, fano, epsilon)
            spectrum += voigt_peak(E, a_ray * ratio, E_tube, s_ray, gamma_tube) * get_efficiency(E_tube, config)
    
        if a_com > 0:
            E_com = get_compton_energy(E_tube, config.angle)
            s_com = sigma_E(E_com, noise, fano, epsilon)
            # El ensanchamiento Doppler es más notable en líneas de alta energía (K)
            spectrum += compton_peak(E, a_com * ratio, E_com, E_tube, s_com) * get_efficiency(E_com, config)

   # --- FONDO DINÁMICO ---
    bkg_coeffs = params["background"]
    # np.polyval usa el orden [cn, ..., c1, c0], así que invertimos la lista
    background = np.polyval(bkg_coeffs[::-1], E)
    
    # spectrum es la suma de picos calculada previamente
    return spectrum + background

# --- TRATAMIENTO DE PARÁMETROS ---

# Vectorización de parámetros
def pack_params(p, elements, fondo="lin"):
    """
    Empaqueta los parámetros en un diccionario, adaptándose al modelo de fondo.
    """
    params = {
        "noise": p[0], "fano": p[1], "epsilon": p[2],
        "tau_pileup": p[3],
        "gain_corr": p[4], "offset_corr": p[5]
    }

    if fondo == "lin":
        params["background"] = (p[6], p[7])
        params["scat_areas"] = {
            "ray_K": p[8], "com_K": p[9], 
            "ray_L": p[10], "com_L": p[11]
        }
        idx = 12
    elif fondo == "cuad":
        params["background"] = (p[6], p[7], p[8])
        params["scat_areas"] = {
            "ray_K": p[9], "com_K": p[10], 
            "ray_L": p[11], "com_L": p[12]
        }
        idx = 13
    else:
        raise ValueError("Fondo no soportado, el fondo debe ser 'lin' o 'cuad'")

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













