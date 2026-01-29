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

# Perfil de los peaks
def voigt_peak(E, area, E0, sigma, gamma):
    """
    area : área total del pico
    gamma: componente Lorentziana (cola instrumental)
    """
    return area * voigt_profile(E - E0, sigma, gamma)

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
def is_excitable_Mo(Z, family):
    """
    Fuente: Mo Kα ≈ 17.44 keV
    """
    if family == "K":
        try:
            return xl.EdgeEnergy(Z, xl.K_SHELL) < 17.44
        except ValueError:
            return False

    # L y M son excitables con Mo
    return True

def get_compton_energy(E0, angle_deg):
    """
    E0: Energía incidente en keV
    angle_deg: Ángulo de dispersión (90° típico en TXRF)
    """
    m_e_c2 = 510.99895  # Energía en reposo del electrón en keV
    angle_rad = np.radians(angle_deg)
    return E0 / (1 + (E0 / m_e_c2) * (1 - np.cos(angle_rad)))

# Modelo
def FRX_model_sdd_general(E_raw, params, config=None):
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
            sigma = sigma_E(E0, noise, fano, epsilon)

            fam = line_family(line)
            if fam is None:
                continue

            A = elem_params.get(f"area_{fam}", 0.0)
            if A <= 0:
                continue

            # Excitación por Mo
            if not is_excitable_Mo(Z, fam):
                continue

            # Solo líneas dentro del rango experimental
            if E0 < E.min() or E0 > E.max():
                continue

            spectrum += voigt_peak(
                E,
                A * r,
                E0,
                sigma,
                gamma
            )

    # --- PICOS DE DISPERSIÓN ---
    # Rayleigh: Elástico ; Compton: Inelástico (depende del ángulo del detector)
    
    # Obtenemos info completa del ánodo (K y L)
    tube_info = get_Xray_info(config["anode"], families=("K", "L"))
    
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
            spectrum += voigt_peak(E, a_ray * ratio, E_tube, s_ray, gamma_tube)
    
        if a_com > 0:
            E_com = get_compton_energy(E_tube, config["geometry_angle"])
            s_com = sigma_E(E_com, noise, fano, epsilon)
            # El ensanchamiento Doppler es más notable en líneas de alta energía (K)
            doppler = 0.005 if fam == "K" else 0.002
            spectrum += voigt_peak(E, a_com * ratio, E_com, s_com, gamma_tube + doppler)

   # --- FONDO DINÁMICO ---
    bkg_coeffs = params["background"]
    # np.polyval usa el orden [cn, ..., c1, c0], así que invertimos la lista
    background = np.polyval(bkg_coeffs[::-1], E)
    
    # spectrum es la suma de picos calculada previamente
    return spectrum + background

# Vectorización de parámetros
def pack_params(p, elements, fondo="lin"):
    """
    Empaqueta los parámetros en un diccionario, adaptándose al modelo de fondo.
    """
    params = {
        "noise": p[0],
        "fano": p[1],
        "epsilon": p[2],
        "elements": {}
    }

    if fondo == "lin":
        params["background"] = (p[3], p[4])      # c0, c1
        params["scat_areas"] = (p[5], p[6])      # Rayleigh, Compton
        idx_start_elements = 7
    elif fondo == "cuad":
        params["background"] = (p[3], p[4], p[5]) # c0, c1, c2
        params["scat_areas"] = (p[6], p[7])      # Rayleigh, Compton
        idx_start_elements = 8
    else:
        raise ValueError("El fondo debe ser 'lin' o 'cuad'")

    # Empaquetado de elementos (común a ambos)
    idx = idx_start_elements
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




