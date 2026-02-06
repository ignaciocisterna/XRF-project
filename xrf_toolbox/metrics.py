import numpy as np
import matplotlib.pyplot as plt
import xraylib as xl
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from . import core

def graficar_deteccion_preliminar(E, I, elementos_detectados, bkg_snip=None):
    """
    Grafica el espectro y etiqueta solo la línea más fuerte (mayor ratio)
    de cada familia (K, L, M) por cada elemento.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(E, I, label='Espectro Experimental', color='grey', lw=1, alpha=0.7)

    if bkg_snip is not None:
        plt.plot(E, bkg_snip, label='Fondo SNIP', color='orange', linestyle='--', alpha=0.5)

    plt.axvspan(16.8, E.max(), color='red', alpha=0.08, label='Zona Exclusión Mo')

    etiquetas_finales = []

    for symb in elementos_detectados:
        try:
            # Obtenemos todas las líneas disponibles
            info_lineas = core.get_Xray_info(symb, families=("K", "L", "M"))

            # Diccionario temporal para encontrar la mejor línea por familia
            # Estructura: {'K': {'e': energy, 'ratio': max_ratio, 'name': 'Ka1'}, ...}
            mejor_por_familia = {}

            for nombre_linea, datos in info_lineas.items():
                e_theo = datos['energy']

                # Solo considerar si está en el rango del eje X
                if E.min() <= e_theo <= E.max():
                    # Ignorar zona de exclusión (excepto para Mo)
                    if (16.8 < e_theo < 17.7) and symb != "Mo":
                        continue

                    # Identificar familia (primera letra: K, L o M)
                    familia = nombre_linea[0]

                    # Si la familia no está o esta línea tiene mayor ratio que la guardada
                    if familia not in mejor_por_familia or datos['ratio'] > mejor_por_familia[familia]['ratio']:
                        mejor_por_familia[familia] = {
                            'e': e_theo,
                            'ratio': datos['ratio'],
                            'label': f"{symb}-{familia}" # Formato simplificado: Si-K, Pb-L, etc.
                        }

            # Añadir las mejores de cada familia a la lista global
            for datos_finales in mejor_por_familia.values():
                etiquetas_finales.append(datos_finales)

        except ValueError:
            continue

    # Ordenar por energía para el escalonamiento vertical
    etiquetas_finales.sort(key=lambda x: x['e'])

    max_counts = np.max(I)
    for i, tag in enumerate(etiquetas_finales):
        e0 = tag['e']
        if 16.8 < e0:
            continue
        idx = np.abs(E - e0).argmin()
        y_val = I[idx]

        # Escalonamiento de niveles
        nivel = i % 6
        y_text = y_val + (max_counts * (0.08 + nivel * 0.12))

        plt.vlines(e0, y_val, y_text, color='royalblue', alpha=0.2, lw=0.6)
        plt.text(e0, y_text, tag['label'],
                 rotation=0, va='bottom', ha='center', fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.1))

    plt.title("Identificación de Elementos (Línea principal por familia)")
    plt.xlabel("Energía (keV)")
    plt.ylabel("Cuentas")
    #plt.ylim(-max_counts*0.05, max_counts * 2.5)
    plt.xlim(E.min(), E.max())
    plt.legend(loc='upper right')
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()

def graficar_fondo(E, I, bkg_fit, bkg_snip, figsize=(12, 8), show=True, fondo="poly", grado_fondo=2):
    if show: plt.figure(figsize=figsize)
    plt.plot(E, I, label='Datos Experimentales', color='grey', alpha=0.4)
    plt.plot(E, bkg_fit, label='Fondo Modelo', color='orange', alpha=0.8, linestyle='--')
    plt.plot(E, bkg_snip, label='Fondo SNIP', color='darkblue', alpha=0.6, linestyle='--')
    plt.xlabel('Energía (keV)')
    plt.ylabel('Cuentas')
    if fondo == "poly":
        modelo_fondo = "Polinomial"
    elif fondo == "exp_poly":
        modelo_fondo = "Exponencial Polinomial"
    plt.title(f'Ajuste de Fondo Continuo con modelo {modelo_fondo} de grado {grado_fondo}')
    #plt.xlim(E.min(), E.max()*1.05)
    #plt.ylim(0, max(I) * 2) # Más espacio arriba para etiquetas
    plt.grid(True, alpha=0.05)
    plt.legend()
    if show: plt.show()

def graficar_ajuste(E, I, I_fit, bkg_fit, elementos, popt, p=None, shells=["K", "L", "M"], 
                       umbral_area_familia=5, umbral_ratio_linea=0.5, figsize=(12, 8),
                       n_bkg=2,show=True, config=None):    
    
    # 1. Preparar parámetros
    p_to_use = p if p is not None else popt
    final_params = core.pack_params(p_to_use, elementos, n_bkg=n_bkg)
    
    if show: plt.figure(figsize=figsize)
    plt.plot(E, I, label='Datos Experimentales', color='grey', alpha=0.4)
    plt.plot(E, I_fit, label='Modelo Ajustado', color='red', lw=1.5, alpha=0.8)
    plt.plot(E, bkg_fit, label='Fondo Modelo', color='orange', alpha=0.8, linestyle='--')

    etiquetas_info = []

    # --- IDENTIFICACIÓN DE LÍNEAS ATÓMICAS ---
    if shells in ["K", "L", "M"]:
        for elem in elementos:
            if elem not in final_params["elements"]: 
                continue
            
            elem_data = final_params["elements"][elem]
            try:
                # Reutilizamos la función del paquete para obtener energías
                info = core.get_Xray_info(elem, families=tuple(shells))
            except:
                continue
    
            for fam in shells:
                area_fam = elem_data.get(f"area_{fam}", 0)
                # Solo etiquetamos si el área es significativa
                if area_fam > umbral_area_familia:
                    # Buscamos la línea principal de la familia (alfa) para poner la etiqueta
                    lines_in_fam = {k: v for k, v in info.items() if k.startswith(fam)}
                    if not lines_in_fam: continue
                    
                    # Seleccionamos la línea con mayor ratio para posicionar el texto
                    main_line = max(lines_in_fam, key=lambda x: lines_in_fam[x]["ratio"])
                    e0 = lines_in_fam[main_line]["energy"]
    
                    if E.min() < e0 < E.max():
                        # Evitar Mo en zona de dispersión si el ánodo es Mo
                        if config and config.anode == "Mo" and elem == "Mo" and (17.0 < e0 < 17.6):
                            continue
                            
                        etiquetas_info.append({
                            'e': e0,
                            'name': f"{elem}-{fam}\n({area_fam:.1e})", # Agregamos el área aquí
                            'type': 'elem'
                        })

    # --- INDICADORES DE ARTEFACTOS (ESCAPE Y SUMA) ---
    # Solo graficamos si el pico principal es muy fuerte (> 20% del max global)
        for elem, data in final_params["elements"].items():
            area_k = data.get("area_K", 0)
            if area_k > (np.max(I) * 0.2): 
                e_ka = xl.LineEnergy(xl.SymbolToAtomicNumber(elem), xl.KA1_LINE)
                
                # Escape (Si Ka - 1.74 keV)
                e_esc = e_ka - 1.74
                if E.min() < e_esc < E.max():
                    etiquetas_info.append({'e': e_esc, 'name': f"esc-{elem}", 'type': 'art'})
                
                # Suma (Ka + Ka) - Solo si tau > 0
                if final_params.get("tau_pileup", 0) > 0:
                    e_sum = e_ka * 2
                    if E.min() < e_sum < E.max():
                        etiquetas_info.append({'e': e_sum, 'name': f"sum-{elem}", 'type': 'art'})

    # --- IDENTIFICACIÓN DE DISPERSIÓN ---
    if config:
        scat = final_params.get("scat_areas", {})
        tube_info = core.get_Xray_info(config.anode, families=("K", "L"))
        # Nos enfocamos en las líneas más intensas del tubo para dispersión
        scat_peaks = []
        
        for line, data in tube_info.items():
            # Buscamos Ka1 o La1
            if line == "Ka1" or line == "La1":
                E_tube = data["energy"]
                E_com = core.get_compton_energy(E_tube, config.angle)
                fam = core.line_family(line)
                
                # Rayleigh
                area_ray = scat.get(f"ray_{fam}", 0)
                if area_ray > umbral_area_familia:
                    scat_peaks.append({'e': E_tube, 'name': f"Ray-{fam}\n({area_ray:.1e})"})
                
                # Compton
                area_com = scat.get(f"com_{fam}", 0)
                if area_com > umbral_area_familia:
                    scat_peaks.append({'e': E_com, 'name': f"Com-{fam}\n({area_com:.1e})"})

        for s_peak in scat_peaks:
            if E.min() < s_peak['e'] < E.max():
                etiquetas_info.append({'e': s_peak['e'], 'name': s_peak['name'], 'type': 'scat'}) 

    # --- RENDERIZADO DE ETIQUETAS ---
    etiquetas_info.sort(key=lambda x: x['e'])
    
    # Ajuste dinámico de niveles para evitar colisiones
    for i, tag in enumerate(etiquetas_info):
        e0 = tag['e']
        idx_e0 = np.abs(E - e0).argmin()
        # Buscamos el máximo en los datos cerca de la energía para posicionar el texto
        y_peak = np.max(I[max(0, idx_e0-5):min(len(I), idx_e0+5)])

        nivel = i % 6 
        y_text = y_peak + (max(I) * (0.04 + nivel * 0.14))

        color = 'black' if tag['type'] == 'elem' else 'darkblue'
        alpha = 0.8 if tag['type'] == 'elem' else 0.6
        fontstyle = 'italic' if tag['type'] == 'art' else 'normal'
        fontweight = 'normal'
        
        plt.vlines(e0, y_peak, y_text, color=color, linestyle=':', alpha=0.3, lw=0.7)
        plt.text(e0, y_text, tag['name'], fontsize=8, rotation=0, alpha=alpha, color=color,
                 ha='center', va='bottom', fontweight=fontweight, fontstyle=fontstyle,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))

    plt.xlabel('Energía (keV)')
    plt.ylabel('Cuentas')
    titulo = 'Deconvolución sobre Peaks Dispersivos' if shells == 'scat' else f'Deconvolución sobre Capas {", ".join(shells)} y Dispersión'
    plt.title(titulo)
    plt.xlim(E.min(), E.max()*1.05)
    plt.ylim(0, max(I) * 2) # Más espacio arriba para etiquetas
    plt.grid(True, alpha=0.05)
    plt.legend()
    if show: plt.show()

def evaluar_ajuste_global(E, I, I_fit, n_parametros, pcov=None, verbose=True):
    """
    Calcula indicadores de calidad globales para el ajuste FRX.
    """
    n_puntos = len(I)
    grados_libertad = n_puntos - n_parametros

    # Chi-cuadrado Reducido (pesos de Poisson)
    sigma2 = np.maximum(I, 1.0)
    residuos = I - I_fit
    chi_cuadrado = np.sum((residuos**2) / sigma2)
    chi_reducido = chi_cuadrado / grados_libertad

    # FOM (Figure of Merit)
    fom = (np.sum(np.abs(residuos)) / np.sum(I)) * 100

    # R-cuadrado
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((I - np.mean(I))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Cálculo de incertidumbres de los parámetros (1-sigma)
    if pcov is not None:
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = "Matriz de covarianza no entregada"

    if verbose:
        print("-" * 35)
        print("MÉTRICAS GLOBALES DEL AJUSTE")
        print("-" * 35)
        print(f"Chi-cuadrado (χ²ν):  {chi_reducido:.4f}")
        print(f"FOM (Error Rel.):    {fom:.2f} %")
        print(f"R² (Det.):           {r2:.5f}")
        print("-" * 35)

        if chi_reducido > 2.0:
            print("Aviso: El χ²ν es alto. El modelo podría estar subestimando picos.")

        elif chi_reducido < 0.8:
            print("Aviso: El χ²ν es bajo. Posible sobreajuste (overfitting).")

        else:
            print("Resultado: El ajuste es estadísticamente excelente.")

    return {"chi": chi_reducido, "fom": fom, "r2": r2, "perr": perr}

def evaluar_elemento_local(elem, E, I, I_fit, padding=0.2):
    """
    Calcula métricas de ajuste en la Región de Interés (ROI) de un elemento.
    """
    try:
        Z = xl.SymbolToAtomicNumber(elem)
        # Seleccionar línea de referencia según energía
        e_k = xl.LineEnergy(Z, xl.KA1_LINE)
        e_l = xl.LineEnergy(Z, xl.LA1_LINE)

        # Si la línea K está en el rango, la usamos, si no, probamos con L
        e_ref = e_k if (e_k > E.min() and e_k < E.max()) else e_l

        if e_ref <= 0 or e_ref < E.min() or e_ref > E.max():
            return None

        e_min, e_max = e_ref - padding, e_ref + padding
        mask = (E >= e_min) & (E <= e_max)

        if np.sum(mask) < 3: return None

        I_roi = I[mask]
        I_fit_roi = I_fit[mask]
        residuos = I_roi - I_fit_roi
        sigma2 = np.maximum(I_roi, 1.0)

        chi_local = np.sum((residuos**2) / sigma2) / len(I_roi)
        fom_local = (np.sum(np.abs(residuos)) / np.sum(I_roi)) * 100

        return {
            "elemento": elem,
            "range": (e_min, e_max),
            "chi_reducido": chi_local,
            "fom": fom_local
        }
    except:
        return None

def graficar_residuos_globales(E, I, I_fit, show=True):
    """
    Muestra la distribución de residuos en todo el rango energético.
    """
    residuos_norm = (I - I_fit) / np.sqrt(np.maximum(I, 1.0))
    
    plt.figure(figsize=(12, 4))
    plt.scatter(E, residuos_norm, s=1, color='blue', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Distribución Global de Residuos Normalizados")
    plt.xlabel("Energía (keV)")
    plt.ylabel("Residuos (σ)")
    plt.grid(True, alpha=0.2)
    if show: plt.show()

def identificar_candidato(energia_pico, tolerancia=0.05):
    """Busca en xraylib qué elemento tiene una línea K o L cerca de la energía dada."""
    candidatos = []
    # Buscamos desde el Litio (3) hasta el Uranio (92)
    for z in range(3, 93):
        symbol = xl.AtomicNumberToSymbol(z)
        # Revisamos líneas K y L principales
        for line_id in [xl.KA1_LINE, xl.LA1_LINE]:
            try:
                e_line = xl.LineEnergy(z, line_id)
                if abs(e_line - energia_pico) < tolerancia:
                    candidatos.append(f"{symbol} ({e_line:.2f} keV)")
            except:
                continue
    return ", ".join(candidatos) if candidatos else "Desconocido"

def detectar_elementos_omitidos(E, I, I_fit, umbral_sigma=10, output=True, verbose=True):
    """Detecta picos en residuos, los identifica y ordena por importancia."""
    residuos = I - I_fit
    # Error estándar de Poisson
    sigma_poisson = np.sqrt(np.maximum(I_fit, 1.0))
    res_norm = residuos / sigma_poisson
    
    # Encontramos picos en los residuos normalizados
    peaks, props = find_peaks(res_norm, height=umbral_sigma, distance=15)
    
    if len(peaks) > 0 and verbose:
        # Extraemos alturas para ordenar
        alturas = props['peak_heights']
        # Ordenar índices de mayor a menor sigma
        idx_ordenado = np.argsort(alturas)[::-1]
        
        print(f"\n{'-'*17} ¡ALERTA: PICOS NO AJUSTADOS!  {'-'*17}")
        print(f"{'Energía (keV)':<15} | {'Significancia':<15}  | {'Posibles Candidatos'}")
        print("-" * 65)
        
        for i in idx_ordenado:
            p = peaks[i]
            sig = alturas[i]
            ene = E[p]
            candidatos = identificar_candidato(ene)
            print(f"{ene:<15.3f} | {sig:<15.1f}σ | {candidatos}")
    else:
        if verbose: print("\n✅ No se detectaron picos significativos omitidos.")
    
    return peaks if output else None

def check_resolution_health(params, config):
    """Compara la resolución ajustada contra la nominal y detecta saturación de bounds."""
    E_mn = 5.895 # Mn Ka1 en keV
    
    # 1. Cálculo de Resolución (FWHM)
    # sigma = sqrt( noise^2 + F*epsilon*E )
    s_mn = np.sqrt(params['noise']**2 + params['fano'] * params['epsilon'] * E_mn)
    fwhm_ev = (s_mn * 2.355) * 1000
    nominal = config.res_mn_ka
    diff = fwhm_ev - nominal
    
    print("-" * 45)
    print(f"DIAGNÓSTICO DEL DETECTOR: {config.name}")
    print(f"  Resolución Nominal: {nominal:.1f} eV")
    print(f"  Resolución Ajustada: {fwhm_ev:.1f} eV ({'+' if diff>0 else ''}{diff:.1f} eV)")
    
    # 2. Detección de Bounds (Salud del ajuste)
    # Valores críticos definidos en deconv.py
    alertas = []
    if params['fano'] >= 0.19: alertas.append("FACTOR FANO AL LÍMITE (Posible sobre-ensanchamiento)")
    if params['fano'] <= 0.06: alertas.append("FACTOR FANO MUY BAJO (Picos inusualmente delgados)")
    if params['noise'] >= 0.029: alertas.append("RUIDO ELECTRÓNICO AL MÁXIMO (Fondo inestable)")
    if params['epsilon'] >= 0.00375 or params['epsilon'] <= 0.00355:
        alertas.append("EPSILON DESVIADO (Problema de calibración ADC)")

    # 3. Impresión de Resultados
    if not alertas:
        if diff > 15:
            print("  Estado: ⚠️ DEGRADADO (Pérdida de resolución significativa)")
        else:
            print("  Estado: ✅ ÓPTIMO")
    else:
        print("  Estado: ❌ CRÍTICO - ALERTAS DE AJUSTE:")
        for a in alertas:
            print(f"    - {a}")
    print("-" * 45)

    fig_h = plt.figure()
    ax_graph = fig_h.add_axes([0.1, 0.1, 0.8, 0.4])
    e_plot = np.linspace(1, 20, 100)
    sig_plot = np.sqrt(params['noise']**2 + params['fano'] * params['epsilon'] * e_plot) * 2.355 * 1000
    ax_graph.plot(e_plot, sig_plot, color='green', label='Curva de Resolución Fit')
    sig_plot_nom = np.sqrt(config.noise**2 + config.fano * config.epsilon * e_plot) * 2.355 * 1000
    ax_graph.plot(e_plot, sig_plot_nom, color='grey', label='Curva de Resolución Nominal', linestyle='--', alpha=0.5)
    ax_graph.set_xlabel("Energía (keV)"); ax_graph.set_ylabel("FWHM (eV)"); ax_graph.grid(alpha=0.3)
    ax_graph.set_title("Gráfico de función de resolución Sigma(E)")
    ax_graph.legend()
    plt.show()

def generar_reporte_completo(E, I, I_fit, bkg_fit, popt, elementos, nombre_muestra="Muestra", n_top=3, 
                             n_bkg=2, config=None):
    """
    Genera tabla de calidad por elemento y gráficos de diagnóstico de los peores ajustes.
    """
    print(f"\n{'='*25} REPORTE: {nombre_muestra} {'='*25}")

    graficar_ajuste(E, I, I_fit, bkg_fit, elementos, popt, n_bkg=n_bkg, umbral_ratio_linea=0.75, config=config)
     
    _ = evaluar_ajuste_global(E, I, I_fit, len(popt))
  
    graficar_residuos_globales(E, I, I_fit)

    resultados = []
    for elem in elementos:
        res = evaluar_elemento_local(elem, E, I, I_fit)
        if res: resultados.append(res)

    if not resultados:
        print("Aviso: No se encontraron ROIs válidas para los elementos proporcionados.")
        return

    # Tabla resumen ordenada por calidad
    print("CALIDAD DEL AJUSTE POR ELEMENTO")
    print(f"{'Elem':<6} | {'Status':<12} | {'χ² Local':<10} | {'FOM (%)':<10}")
    print("-" * 55)

    for r in sorted(resultados, key=lambda x: x['chi_reducido']):
        if r['chi_reducido'] < 1.5: status = "✅ OK"
        elif r['chi_reducido'] < 4.0: status = "⚠️ WARNING"
        else: status = "❌ ERROR"
        print(f"{r['elemento']:<6} | {status:<12} | {r['chi_reducido']:<10.2f} | {r['fom']:<10.2f}")

    # Gráfico de diagnóstico (elementos con peor ajuste)
    print(f"Gráficos Diagnóstico para los {n_top} peores ajustes")
    peores = sorted(resultados, key=lambda x: x['chi_reducido'], reverse=True)[:n_top]
    n_plots = len(peores)
    
    if n_plots > 0:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 * n_plots))
        if n_plots == 1: axes = [axes]

        for i, res in enumerate(peores):
            mask = (E >= res['range'][0]) & (E <= res['range'][1])
            axes[i].plot(E[mask], I[mask], 'ko', alpha=0.3, label='Experimental', markersize=4)
            axes[i].plot(E[mask], I_fit[mask], 'r-', label='Ajuste Modelo', lw=1.5)
            
            # Sombreado de residuos
            diff = I[mask] - I_fit[mask]
            axes[i].fill_between(E[mask], diff, color='blue', alpha=0.15, label='Residuo')
            
            # Emisión característica
            info = core.get_Xray_info(res['elemento'])
            for line_name, line_data in info.items():
                y_text = I[mask].max() * line_data['ratio']
                if line_data['energy'] > res['range'][0] and line_data['energy'] < res['range'][1]:
                    axes[i].vlines(line_data['energy'], 0, y_text, color='blue', linestyle='-', lw=0.75)
                    axes[i].text(line_data['energy'], y_text, line_name,
                         fontsize=8, 
                         rotation=0,
                         ha='center',
                         va='bottom',
                         fontweight='normal',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
            axes[i].set_title(f"Diagnóstico Local: {res['elemento']} (χ²: {res['chi_reducido']:.2f})")
            axes[i].set_ylabel("Cuentas")
            axes[i].legend(fontsize='small')
            axes[i].grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()

    # Gráfico de diagnóstico (elementos con mejor ajuste)
    print(f"Gráficos Diagnóstico para los {n_top} mejores ajustes")
    mejores = sorted(resultados, key=lambda x: x['chi_reducido'], reverse=False)[:n_top]
    n_plots = len(mejores)
    
    if n_plots > 0:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 * n_plots))
        if n_plots == 1: axes = [axes]

        for i, res in enumerate(mejores):
            mask = (E >= res['range'][0]) & (E <= res['range'][1])
            axes[i].plot(E[mask], I[mask], 'ko', alpha=0.3, label='Experimental', markersize=4)
            axes[i].plot(E[mask], I_fit[mask], 'r-', label='Ajuste Modelo', lw=1.5)
            
            # Sombreado de residuos
            diff = I[mask] - I_fit[mask]
            axes[i].fill_between(E[mask], diff, color='blue', alpha=0.15, label='Residuo')
            
            axes[i].set_title(f"Diagnóstico Local: {res['elemento']} (χ²: {res['chi_reducido']:.2f})")
            axes[i].set_ylabel("Cuentas")
            axes[i].legend(fontsize='small')
            axes[i].grid(True, alpha=0.2)
            
            # Emisión característica
            info = core.get_Xray_info(res['elemento'])
            for line_name, line_data in info.items():
                y_text = I[mask].max() * line_data['ratio']
                if line_data['energy'] > res['range'][0] and line_data['energy'] < res['range'][1]:
                    axes[i].vlines(line_data['energy'], 0, y_text, color='blue', linestyle='-', lw=0.75)
                    axes[i].text(line_data['energy'], y_text, line_name,
                         fontsize=8, 
                         rotation=0,
                         ha='center',
                         va='bottom',
                         fontweight='normal',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
        plt.tight_layout()
        plt.show()

    detectar_elementos_omitidos(E, I, I_fit, output=False)

    if config:
        params = core.pack_params(popt, elementos, n_bkg=n_bkg)
        check_resolution_health(params, config)

def exportar_reporte_pdf(E, I, I_fit, bkg_fit, popt, elementos, config, nombre_muestra="Muestra", 
                         archivo="Reporte_FRX.pdf", n_bkg=2):
    PAGE_SIZE = (11, 8.5)
    with PdfPages(archivo) as pdf:
        # --- PÁGINA 1: AJUSTE GLOBAL Y TABLA ---
        fig1 = plt.figure(figsize=PAGE_SIZE)
        graficar_ajuste(E, I, I_fit, bkg_fit, elementos, popt, figsize=PAGE_SIZE, show=False, n_bkg=n_bkg)
        plt.subplots_adjust(bottom=0.38)
        ax_table = fig1.add_axes([0.1, 0.05, 0.8, 0.25])
        ax_table.axis('off')
        
        # Llenado por columnas
        n_cols = 8
        elementos_sorted = sorted(elementos)
        n_rows = int(np.ceil(len(elementos_sorted) / n_cols))
        grid = np.full((n_rows, n_cols), "", dtype=object)
        for i, elem in enumerate(elementos_sorted):
            grid[i % n_rows, i // n_rows] = elem
        
        # Corregido: edges='' para eliminar bordes y evitar ValueError
        tab = ax_table.table(cellText=grid, loc='center', cellLoc='center', edges='')
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        ax_table.set_title("Elementos Ajustados en la Deconvolución", fontweight='bold', pad=5)
        pdf.savefig(fig1); plt.close(fig1)

        # --- PÁGINA 2: MÉTRICAS Y RESIDUOS ---
        fig2, (ax_t, ax_r) = plt.subplots(2, 1, figsize=PAGE_SIZE, gridspec_kw={'height_ratios': [1, 2]})
        res_g = evaluar_ajuste_global(E, I, I_fit, len(popt), verbose=False)
        info = f"MÉTRICAS GLOBALES - {nombre_muestra}\n{'-'*50}\nχ²ν: {res_g['chi']:.4f} | FOM: {res_g['fom']:.2f}% | R²: {res_g['r2']:.5f}"
        ax_t.axis('off'); ax_t.text(0.5, 0.5, info, ha='center', va='center', fontsize=12, family='monospace')
        ax_r.scatter(E, (I-I_fit)/np.sqrt(np.maximum(I_fit, 1.0)), s=1.5, color='blue', alpha=0.4)
        ax_r.axhline(0, color='red', linestyle='--'); ax_r.set_title("Residuos Normalizados"); pdf.savefig(fig2); plt.close(fig2)

        # --- PÁGINA 3: TABLA CALIDAD ---
        fig3, ax3 = plt.subplots(figsize=PAGE_SIZE); ax3.axis('off')
        res_loc = sorted([r for r in [evaluar_elemento_local(e, E, I, I_fit) for e in elementos] if r], key=lambda x: x['chi_reducido'])
        txt = f"{'Elem':<8} | {'Status':<10} | {'χ² Local':<10} | {'FOM (%)':<10}\n" + "-"*50 + "\n"
        for r in res_loc:
            st = "OK" if r['chi_reducido'] < 1.5 else "WARN" if r['chi_reducido'] < 4.0 else "ERR"
            txt += f"{r['elemento']:<8} | {st:<10} | {r['chi_reducido']:<10.2f} | {r['fom']:<10.2f}\n"
        ax3.text(0.1, 0.95, f"CALIDAD DEL AJUSTE POR ELEMENTO\n\n{txt}", fontsize=10, family='monospace', va='top'); pdf.savefig(fig3); plt.close(fig3)

        # --- PÁGINA 4: DIAGNÓSTICO EXTREMOS ---
        peores, mejores = res_loc[::-1][:3], res_loc[:3]
        fig4, axes = plt.subplots(6, 1, figsize=(11, 15))
        # Ajustado: hspace más grande para evitar solapamiento
        plt.subplots_adjust(hspace=1.0, top=0.9, bottom=0.05)
        fig4.suptitle("Diagnóstico de extremos", fontsize=16, fontweight='bold', y=0.97)
        fig4.text(0.5, 0.93, "Peores Ajustes", ha='center', fontsize=13, fontstyle='italic')
        # Corregido: Posición de 'Mejores Ajustes' para que no solape
        fig4.text(0.5, 0.47, "Mejores Ajustes", ha='center', fontsize=13, fontstyle='italic', va='center')
        for i, res in enumerate(peores + mejores):
            m = (E >= res['range'][0]) & (E <= res['range'][1])
            axes[i].plot(E[m], I[m], 'ko', alpha=0.3, markersize=3)
            axes[i].plot(E[m], I_fit[m], 'r-', lw=1.2)
            axes[i].fill_between(E[m], I[m]-I_fit[m], color='blue', alpha=0.1)
            # Emisión característica
            info = core.get_Xray_info(res['elemento'])
            for line_name, line_data in info.items():
                ymin, ymax = axes[i].get_ylim()
                npoint = 0.9 * (ymin + ymax)  
                y_text = npoint * line_data['ratio']
                if line_data['energy'] > res['range'][0] and line_data['energy'] < res['range'][1]:
                    axes[i].vlines(line_data['energy'], 0, y_text, color='blue', linestyle='-', lw=0.75)
                    axes[i].text(line_data['energy'], y_text, line_name,
                         fontsize=7, 
                         rotation=0,
                         ha='center',
                         va='bottom',
                         fontweight='normal',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
            axes[i].set_title(f"ROI: {res['elemento']} (χ²: {res['chi_reducido']:.2f})", fontsize=9, pad=3)
        pdf.savefig(fig4); plt.close(fig4)

        # --- PÁGINA 5: OMITIDOS ---
        fig5, ax5 = plt.subplots(figsize=PAGE_SIZE); ax5.axis('off')
        res_n = (I-I_fit)/np.sqrt(np.maximum(I_fit, 1.0))
        pks, prps = find_peaks(res_n, height=10, distance=15)
        txt_o = f"PICOS NO AJUSTADOS - {nombre_muestra}\n" + "="*55 + "\n\n"
        if len(pks) > 0:
            idx = np.argsort(prps['peak_heights'])[::-1]
            txt_o += f"{'Energía':<10} | {'Significancia':<12} | {'Candidatos'}\n" + "-"*55 + "\n"
            for i in idx: 
                txt_o += f"{E[pks[i]]:<10.3f} | {prps['peak_heights'][i]:<12.1f}σ | {identificar_candidato(E[pks[i]])}\n"
        else: txt_o += "✅ Sin omisiones significativas."
        ax5.text(0.1, 0.95, txt_o, fontsize=10, family='monospace', va='top'); pdf.savefig(fig5); plt.close(fig5)

        # --- PÁGINA 6: SALUD DEL DETECTOR ---
        fig_h, ax_h = plt.subplots(figsize=PAGE_SIZE)
        ax_h.axis('off')
        
        # Obtenemos los datos de salud
        res_mn_nom = config.res_mn_ka
        final_params = core.pack_params(p_to_use, elementos, n_bkg=n_bkg)
        s_mn = np.sqrt(final_params['noise']**2 + final_params['fano'] * final_params['epsilon'] * 5.895)
        res_mn_fit = (s_mn * 2.355) * 1000
        
        health_txt = (
            f"CHEQUEO DE RESOLUCIÓN Y HARDWARE\n"
            f"{'='*40}\n\n"
            f"Instrumento: {config.name}\n"
            f"Resolución Mn-Ka (Nominal): {res_mn_nom:.1f} eV\n"
            f"Resolución Mn-Ka (Ajustada): {res_mn_fit:.1f} eV\n"
            f"Desviación: {res_mn_fit - res_mn_nom:+.1f} eV\n\n"
            f"PARÁMETROS FÍSICOS AJUSTADOS:\n"
            f"{'-'*40}\n"
            f"Ruido Electrónico (σ): {final_params['noise']*1000:.2f} eV\n"
            f"Factor de Fano:        {final_params['fano']:.3f}\n"
            f"Epsilon (eV/eh):       {final_params['epsilon']*1000:.1f} eV\n"
            f"Tau Pile-up (s):       {final_params['tau_pileup']:.2e} s\n\n"
        )
        
        # Lógica de semáforo
        status = "ÓPTIMO"
        if abs(res_mn_fit - res_mn_nom) > 15: status = "DEGRADADO"
        if final_params['fano'] > 0.18 or final_params['noise'] > 0.025: status = "CRÍTICO (Revisar Bounds)"
        
        health_txt += f"ESTADO GENERAL: {status}"
        
        ax_h.text(0.1, 0.9, health_txt, fontsize=12, family='monospace', va='top')
        
        # Opcional: Un pequeño gráfico de la función de resolución Sigma(E)
        ax_graph = fig_h.add_axes([0.1, 0.1, 0.8, 0.4])
        e_plot = np.linspace(1, 20, 100)
        
        sig_plot = np.sqrt(final_params['noise']**2 + final_params['fano'] * final_params['epsilon'] * e_plot) * 2.355 * 1000
        ax_graph.plot(e_plot, sig_plot, color='green', label='Curva de Resolución Fit')
        
        sig_plot_nom = np.sqrt(config.noise**2 + config.fano * config.epsilon * e_plot) * 2.355 * 1000
        ax_graph.plot(e_plot, sig_plot_nom, color='grey', label='Curva de Resolución Nominal', linestyle='--', alpha=0.5)
        
        ax_graph.set_xlabel("Energía (keV)"); ax_graph.set_ylabel("FWHM (eV)"); ax_graph.grid(alpha=0.3)
        ax_graph.set_title("Gráfico de función de resolución Sigma(E)")
        ax_graph.legend()
        
        pdf.savefig(fig_h); plt.close(fig_h)
    
    print(f"✅ Reporte final generado: {archivo}")


        
