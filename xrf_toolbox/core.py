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
                    info[name] = {
                        "energy": energy,
                        "ratio": rate / ref_rate
                    }

            except ValueError:
                pass

    if not info:
        raise ValueError(f"No valid X-ray lines found for {symb}")

    return info

# Resolución Energética para detector tipo SDD (usado en PICOFOX)

# Para modelo
def sigma_E(E, a=0.0057, b=0.00252):
    #a = 0.0057    # Datos de S2 PICOFOX
    #b = 0.00252   # Datos de S2 PICOFOX
    return np.sqrt(a + b * E)

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

# Modelo
def FRX_model_sdd_general(E, params):
    """
    Modelo FRX generalizado con áreas independientes por familia K, L y M.
    La excitación de Mo y los efectos instrumentales están absorbidos
    en area_K, area_L y area_M respectivamente.
    """

    gamma = 0.02  # keV, típico SDD (corroborar)
    a, b = params["a"], params["b"]

    sigma = sigma_E(E, a, b)
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

    # Picos de Dispersión (Rayleigh y Compton del Mo)
    # Rayleigh: Elástico (17.44 keV)
    # Compton: Inelástico (aprox 17.1 keV, depende del ángulo del detector)
    area_ray, area_com = params.get("scat_areas", (0, 0))
    spectrum += voigt_peak(E, area_ray, 17.44, sigma, gamma) # Rayleigh
    spectrum += voigt_peak(E, area_com, 17.10, sigma, gamma) # Compton (ajustable)

    # Fondo dinámico
    bkg_coeffs = params["background"]
    if len(bkg_coeffs) == 2:   # Lineal: c0 + c1*E
        background = bkg_coeffs[0] + bkg_coeffs[1] * E
    elif len(bkg_coeffs) == 3: # Cuadrático: c0 + c1*E + c2*E^2
        background = bkg_coeffs[0] + bkg_coeffs[1] * E + bkg_coeffs[2] * E**2
    
    # spectrum es la suma de picos calculada previamente
    return spectrum + background

# Vectorización de parámetros
def pack_params(p, elements, fondo="lin"):
    """
    Empaqueta los parámetros en un diccionario, adaptándose al modelo de fondo.
    """
    params = {
        "a": p[0],
        "b": p[1],
        "elements": {}
    }

    if fondo == "lin":
        params["background"] = (p[2], p[3])      # c0, c1
        params["scat_areas"] = (p[4], p[5])      # Rayleigh, Compton
        idx_start_elements = 6
    elif fondo == "cuad":
        params["background"] = (p[2], p[3], p[4]) # c0, c1, c2
        params["scat_areas"] = (p[5], p[6])      # Rayleigh, Compton
        idx_start_elements = 7
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

