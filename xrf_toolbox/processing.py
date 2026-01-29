import numpy as np
import xraylib as xl
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .core import fwhm_SNIP, get_Xray_info

# Estimación del fondo por algoritmo SNIP
def snip_trace_safe(
    energy,
    counts,
    fwhm_func,
    n_iter=40,
    beta=1.5,
    alpha=0.05
):
    """
    energy    : eje energético [keV]
    counts    : espectro original
    fwhm_func : función que devuelve FWHM en [keV]
    """
    # 1. Cálculo del ancho de canal (keV por bin)
    delta_E = np.mean(np.diff(energy))

    # 2. Transformación Logarítmica para manejar el rango dinámico
    c = max(1.0, alpha * np.percentile(counts, 5))
    y = np.log(np.maximum(counts, 0) + c)
    y_final = y.copy()

    # 3. Algoritmo SNIP
    for k in range(n_iter, 0, -1):
        for i in range(len(y)):
            # Convertir FWHM(keV) en Canales
            fwhm_channels = fwhm_func(energy[i]) / delta_E
            m = int(beta * fwhm_channels * (k / n_iter))

            if m < 1:
                m = 1

            if i - m >= 0 and i + m < len(y):
                y_clip = 0.5 * (y_final[i - m] + y_final[i + m])
                y_final[i] = min(y_final[i], y_clip)

    # 4. Regresar al espacio original
    background = np.exp(y_final) - c
    return np.clip(background, 0, None)

def detectar_elementos(E, I, bkg_snip, manual_elements=None, ignorar=None, tolerance=0.05, sigma_umbral=8, 
                       permitir_solapamientos=False, todos=False):
    """
    Autodetección robusta basada en significancia estadística y probabilidad.
    """
    I_net = I - bkg_snip
    max_counts = np.max(I_net)
    # Calculamos la desviación estándar del ruido basada en el fondo (Poisson)
    # Un pico es real si supera el fondo en N sigmas.
    std_ruido = np.sqrt(np.maximum(bkg_snip, 1))
    
    # Buscamos picos que destaquen sobre el ruido local
    indices, _ = find_peaks(I_net, height=sigma_umbral * std_ruido, distance=15, prominence=max_counts * 0.005)
    energias_picos = E[indices]

    # 1. Inclusión de elementos manuales
    manuales = set(manual_elements) if manual_elements else set()
    elementos_finales = manuales.copy()

    # Pre-calculo de energías de manuales para control de solapamiento
    energias_manuales = []
    if not permitir_solapamientos:
        for m in manuales:
            try:
                info_m = get_Xray_info(m)
                if "Ka1" in info_m: energias_manuales.append(info_m["Ka1"]["energy"])
                if "La1" in info_m: energias_manuales.append(info_m["La1"]["energy"])
            except: continue

    # Grupos de control
    PRIORIDAD = {'Si', 'Ar', 'Ti', 'Fe', 'Cu', 'Zn', 'As', 'Se', 'Sr', 'Sb', 'Pb', 'Ca', 'K', 'Cl', 'S', 'Ni', 'Cr', 'Mn'}
    TIERRAS_RARAS = {'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
    ESCASOS = {'Os', 'Ir', 'Re', 'Ru', 'Rh', 'Pd', 'Pt', 'Au', 'Hf', 'Ta'}
    if not todos:
        EXCLUIR = {'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Pu', 'Kr', 'Xe'}
    else:
        EXCLUIR = {}

    if ignorar:
        EXCLUIR.update(ignorar)
    
    zona_exclusion = (16.8, 17.8) # Zona del Mo/Dispersión

    for ep in energias_picos:
        # Filtros básicos de energía
        if (zona_exclusion[0] < ep < zona_exclusion[1]) or ep < 1.0: continue

        # Protección de manuales: si el pico es "propiedad" de un manual, saltar
        if not permitir_solapamientos and any(abs(ep - em) < tolerance for em in energias_manuales):
            continue

        candidatos_locales = []
        for z in range(11, 84):
            sym = xl.AtomicNumberToSymbol(z)
            
            # Si el elemento ya está en manuales, no lo buscamos de nuevo como auto-detectado
            if sym in elementos_finales and not permitir_solapamientos:
                continue
            if sym in EXCLUIR: continue
            
            try:
                info = get_Xray_info(sym)
                for fam in ["Ka1", "La1"]:
                    if fam in info:
                        e_theo = info[fam]["energy"]
                        if abs(e_theo - ep) < tolerance:
                            score = 15 if sym in PRIORIDAD else 5
                            
                            # Penalización de tierras raras
                            if sym in TIERRAS_RARAS or sym in ESCASOS:
                                score -= 12 
                            
                            # Validación de pareja (Kb o Lb)
                            check_line = "Kb1" if fam == "Ka1" else "Lb1"
                            if check_line in info:
                                e_check = info[check_line]["energy"]
                                idx_c = np.abs(E - e_check).argmin()
                                if I_net[idx_c] > 4 * std_ruido[idx_c]:
                                    score += 25 
                                else:
                                    score -= 10
                            
                            candidatos_locales.append({'sym': sym, 'score': score, 'diff': abs(e_theo - ep)})
            except: continue
        
        if candidatos_locales:
            mejor = max(candidatos_locales, key=lambda x: (x['score'], -x['diff']))
            if mejor['score'] > 8:
                elementos_finales.add(mejor['sym'])

    return sorted(list(elementos_finales))

def recortar_espectro(E, I, e_min_busqueda=1.2, e_max=17.5, offset_kev=0.25):
    """
    Recorta el espectro eliminando el ruido electrónico inicial mediante 
    el análisis de la segunda derivada (curvatura) y aplica el límite superior.
    """
    # 1. Aislar zona de ruido inicial para el análisis
    mask_inicio = E <= e_min_busqueda
    E_sub = E[mask_inicio]
    I_sub = I[mask_inicio]
    
    if len(E_sub) < 10:
        # Si hay muy pocos datos en la zona de búsqueda, usar un fallback seguro
        e_min_detectado = 0.1
    else:
        # 2. Transformación logarítmica y suavizado
        # El logaritmo ayuda a resaltar el cambio de curvatura en caídas exponenciales
        I_log = np.log(I_sub + 1.0)
        I_smooth = gaussian_filter1d(I_log, sigma=1.5)
        
        # 3. Calcular la segunda derivada (Aceleración de la curva)
        # El punto de máxima curvatura positiva indica el "frenazo" del ruido
        d2 = np.gradient(np.gradient(I_smooth))
        
        # Buscamos el índice del máximo de la segunda derivada 
        idx_corte = np.argmax(d2)
        
        # 4. Aplicar offset de seguridad
        # Desplazamos unos cuantos keV a la derecha para no morder el flanco del ruido
        e_min_detectado = E_sub[idx_corte] + offset_kev

    # 5. Validaciones de seguridad
    if e_min_detectado < 0.05:
        e_min_detectado = 0.05
    
    # 6. Aplicar máscara final al espectro completo
    mask = (E >= e_min_detectado) & (E <= e_max)
    

    return E[mask], I[mask]

def generar_mascara_roi(E, elementos, margen=0.4, borde=0.1):
    """
    Crea una máscara booleana: True solo en las zonas donde 
    sabemos que hay líneas de los elementos detectados.
    """
    E_work = (E >= E.min() + borde) & (E <= E.max() - borde)
    mask = np.zeros_like(E_work, dtype=bool)
    for sym in elementos:
        try:
            info = get_Xray_info(sym)
            for datos in info.values():
                en = datos['energy']
                # Marcamos como True el rango alrededor de cada línea
                mask |= (E_work >= en - margen) & (E_work <= en + margen)
        except:
            continue
    return mask

def estimate_tau_pileup(counts, T_real, T_live):
    """
    Estima el tiempo de resolución (tau) del sistema a partir del espectro.
    
    counts: Array o lista con las cuentas por canal.
    T_real: Tiempo de reloj (s) - Tiempo transcurrido total.
    T_live: Tiempo de vida (s) - Tiempo que el detector estuvo disponible.
    
    Retorna:
    tau (float): Tiempo de resolución estimado en segundos.
    """
    # 1. Integración del área total (N total de pulsos procesados)
    n_total = np.sum(counts)
    
    # 2. Validación de tiempos
    if T_real <= T_live:
        # Esto ocurre si no hay tiempo muerto o los metadatos están mal
        return 0.0
    
    if n_total <= 0:
        return 0.0
    
    # 3. Cálculo de tau
    # Basado en el modelo de tiempo muerto: T_dead = T_real - T_live
    # El tiempo muerto por pulso es el tiempo muerto total entre el total de eventos
    tau = (T_real - T_live) / n_total
    
    return tau









