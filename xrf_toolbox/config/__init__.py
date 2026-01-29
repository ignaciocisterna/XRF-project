from .manager import InstrumentConfig

def load_config(name):
    """Atajo para cargar la configuración de un equipo por nombre."""
    return InstrumentConfig.load(name)

__all__ = ["InstrumentConfig", "load_config"]
