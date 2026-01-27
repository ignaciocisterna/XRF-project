# Importar la clase principal para que esté disponible directamente
from .deconv import XRFDeconv

# Funciones útiles de procesamiento, métricas y core
from .processing import recortar_espectro
from .processing import detectar_elementos

from .core import get_Xray_info

from .metrics import graficar_deteccion_preliminar
from .metrics import graficar_ajuste
from .metrics import generar_reporte_completo
from .metrics import exportar_reporte_pdf


# Versión del paquete 
__version__ = "1.0.1"

# Controlar qué se importa con "from xrf_toolbox import *"
__all__ = ["XRFDeconv", "recortar_espectro", "get_Xray_info", "detectar_elementos",
            "graficar_deteccion_preliminar", "graficar_ajuste", "generar_reporte_completo",

            "exportar_reporte_pdf"]
