import json
import os

class InstrumentConfig:
    def __init__(self, data):
        # --- Identificación ---
        self.name = data.get("name", "Unknown Instrument")
        self.anode = data.get("anode", "Mo")
        
        # --- Geometría y Trayecto ---
        geom = data.get("geometry", {})
        path = geom.get("path", {})
        self.angle = geom.get("angle_deg", 90.0)
        self.atm = path.get("atmosphere", "air")
        self.path_dist = path.get("distance_mm")
        pd = path.get("distance_mm")
        self.path_dist = 5.0 if pd is None else pd
        
        # --- Detector y Física de Resolución ---
        det = data.get("detector", {})
        self.det_material = det.get("material", "Si")
        self.det_thick = det.get("thickness_mm", 0.45)
        self.active_area = det.get("active_area_mm2", 30.0)
        dl = det.get("dead_layer_um")
        self.dead_layer = 0.05 if dl is None else dl
        
        # Valores para el ajuste (Semillas p0)
        self.noise = det.get("noise_default_kev", 0.065)
        self.fano = det.get("fano_default", 0.115)
        self.epsilon = det.get("epsilon_kev", 0.00365)
        self.res_mn_ka = det.get("resolution_mn_ka_ev", 149.0)
        
        # --- Ventana ---
        win = data.get("window", {})
        self.win_material = win.get("material", "Be")
        self.win_thick = win.get("thickness_um", 100.0)

    @classmethod
    def load(cls, instrument_name):
        """Carga el JSON desde la carpeta config del paquete."""
        base_path = os.path.dirname(__file__)
        filename = f"{instrument_name.lower().replace(' ', '_')}.json"
        path = os.path.join(base_path, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el perfil del equipo: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return cls(json.load(f))

    def get_resolution_params(self):
        """Devuelve el trío de resolución para p0: noise, fano y epsilon"""
       return {"noise": self.noise, "fano": self.fano, "epsilon": self.epsilon}








