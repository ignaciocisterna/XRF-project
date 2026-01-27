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
    alturas_picos = I_net[indices]

    elementos_finales = set(manual_elements) if manual_elements else set()

    # Filtro de exclusión total de elementos "imposibles" en XRF común
    # Elementos altamente radiactivos o inestables que ensucian el ajuste
    if not todos:
        EXCLUIR = {'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Pu', 'Am', 'Cm'}
    else:
        EXCLUIR = {}
    
    # Elementos con alta probabilidad (Metales de transición y geológicos comunes)
    PRIORIDAD = {'Fe', 'Cu', 'Zn', 'Pb', 'As', 'Sr', 'Ti', 'Cr', 'Mn', 'Ni', 'Ca', 'K', 'Cl', 'S', 'Si', 'Al'} 
    
    zona_exclusion = (16.8, 17.8) # Zona del Mo/Dispersión

    for ep, ap in zip(energias_picos, alturas_picos):
        # Filtros básicos de energía
        if (zona_exclusion[0] < ep < zona_exclusion[1]) or ep < 1.0:
            continue

        candidatos_locales = []
        for z in range(11, 93):
            sym = xl.AtomicNumberToSymbol(z)
            if sym in EXCLUIR: continue
            
            try:
                # Chequeamos las dos líneas principales
                ka = xl.LineEnergy(z, xl.KA1_LINE)
                la = xl.LineEnergy(z, xl.LA1_LINE)
                
                for e_theo in [ka, la]:
                    if abs(e_theo - ep) < tolerance:
                        # Calculamos un score de confianza
                        score = 1
                        if sym in PRIORIDAD: score += 2
                        # Si es una línea K, le damos más peso que a una M o L difusa
                        if e_theo == ka: score += 1 
                        
                        candidatos_locales.append({'sym': sym, 'score': score, 'diff': abs(e_theo - ep)})
            except: continue
        
        if candidatos_locales:
            # Elegimos el que tenga mejor score, y en caso de empate, el de menor diferencia de energía
            mejor = max(candidatos_locales, key=lambda x: (x['score'], -x['diff']))
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

