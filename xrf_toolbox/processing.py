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

def detectar_elementos(E, I, bkg_snip, manual_elements=None, tolerance=0.05, sigma_umbral=8, permitir_solapamientos=False, todos=False):
    """
    Autodetección robusta basada en significancia estadística y probabilidad.
    """
    I_net = I - bkg_snip
    # Calculamos la desviación estándar del ruido basada en el fondo (Poisson)
    # Un pico es real si supera el fondo en N sigmas.
    std_ruido = np.sqrt(np.maximum(bkg_snip, 1))
    
    # Buscamos picos que destaquen sobre el ruido local
    indices, props = find_peaks(I_net, height=sigma_umbral * std_ruido, distance=15, prominence=sigma_umbral*2)
    energias_picos = E[indices]

    elementos_finales = set(manual_elements) if manual_elements else set()

    # Filtro de exclusión total de elementos "imposibles" en XRF común
    # Elementos altamente radiactivos o inestables que ensucian el ajuste
    if not todos:
        EXCLUIR = {'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Pu', 'Am', 'Cm'}
    else:
        EXCLUIR = {}
    
    zona_exclusion = (16.8, 17.8) # Zona del Mo/Dispersión

    for ep in energias_picos:
        # Filtros básicos de energía
        if (zona_exclusion[0] < ep < zona_exclusion[1]) or ep < 1.0:
            continue

        candidatos = []
        for z in range(11, 84):
            sym = xl.AtomicNumberToSymbol(z)
            if sym in EXCLUIR: continue
            
            try:
                info_lineas = get_Xray_info(sym)
                
                # Buscamos si el pico actual coincide con alguna línea fuerte (Ka1 o La1)
                for line_name in ["Ka1", "La1"]:
                    if line_name in info_lineas:
                        e_theo = info_lineas[line_name]["energy"]
                        
                        if abs(e_theo - ep) < tolerance:
                            # --- VALIDACIÓN DE CONSISTENCIA ---
                            # Si es Ka1, buscamos su pareja Kb1
                            check_line = "Kb1" if line_name == "Ka1" else "Lb1"
                            
                            if check_line in info_lineas:
                                e_check = info_lineas[check_line]["energy"]
                                ratio_theo = info_lineas[check_line]["ratio"]
                                
                                # Buscamos la intensidad real en el espectro en la posición de la línea de chequeo
                                idx_check = np.abs(E - e_check).argmin()
                                # El pico de chequeo debe ser al menos proporcional a la intensidad esperada
                                # (Damos un margen amplio porque la eficiencia del detector varía)
                                if I_net[idx_check] > (sigma_umbral * 0.5):
                                    score = 10 if line_name == "Ka1" else 5
                                    candidatos.append({'sym': sym, 'score': score, 'diff': abs(e_theo - ep)})
                            else:
                                # Si el elemento no tiene Kb1/Lb1 en xraylib (raro), confiamos con menor score
                                candidatos.append({'sym': sym, 'score': 2, 'diff': abs(e_theo - ep)})
                                
            except: continue
            
        if candidatos:
            # Seleccionamos el que tenga mejor score y mejor coincidencia de energía
            mejor = max(candidatos, key=lambda x: (x['score'], -x['diff']))
            elementos_finales.add(mejor['sym'])

    return sorted(list(elementos_finales))

def recortar_espectro(E, I, e_min_busqueda=1.2, e_max=17.5, offset_bins=2):
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
        I_smooth = gaussian_filter1d(I_log, sigma=2)
        
        # 3. Calcular la segunda derivada (Aceleración de la curva)
        # El punto de máxima curvatura positiva indica el "frenazo" del ruido
        d2 = np.gradient(np.gradient(I_smooth))
        
        # Buscamos el índice del máximo de la segunda derivada
        idx_corte = np.argmax(d2)
        
        # 4. Aplicar offset de seguridad
        # Desplazamos un par de canales a la derecha para no morder el flanco del ruido
        idx_final = min(idx_corte + offset_bins, len(E_sub) - 1)
        e_min_detectado = E_sub[idx_final]

    # 5. Validaciones de seguridad
    if e_min_detectado < 0.05:
        e_min_detectado = 0.05
    
    # 6. Aplicar máscara final al espectro completo
    mask = (E >= e_min_detectado) & (E <= e_max)
    

    return E[mask], I[mask]



