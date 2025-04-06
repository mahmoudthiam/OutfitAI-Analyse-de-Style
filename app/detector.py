
import os
import sys
import torch
from ultralytics import YOLO
import cv2

# Configuration et patches critiques
os.environ.update({
    'YOLO_DISABLE_HUB': '1',
    'YOLO_DO_NOT_TRACK': '1',
    'YOLO_VERBOSE': 'False'
})

class FashionDetector:
    def __init__(self, model_path="https://github.com/mahmoudthiam/OutfitAI-Analyse-de-Style/blob/main/yolov8n-fashion.pt
"):
        """Initialisation robuste du modèle"""
        # Solution définitive pour weights_only
        self.original_load = torch.load
        self._patch_torch_load()
        
        try:
            self.model = YOLO(model_path)
            # Vérification que le modèle est bien chargé
            if not hasattr(self.model, 'predict'):
                raise RuntimeError("Échec du chargement du modèle YOLO")
        except Exception as e:
            self._restore_torch_load()
            raise e

        self.class_map = {
            0: "top", 1: "bottom", 
            2: "dress", 3: "jacket",
            4: "shoes", 5: "accessory"
        }

    def _patch_torch_load(self):
        """Patch temporaire de torch.load"""
        def safe_load(*args, **kwargs):
            kwargs.pop('weights_only', None)  # Évite le conflit de paramètres
            return self.original_load(*args, **kwargs)
        torch.load = safe_load

    def _restore_torch_load(self):
        """Restaure la fonction originale"""
        torch.load = self.original_load

    def detect(self, image_path):
        """Détection robuste avec gestion d'erreur améliorée"""
        try:
            # Conversion directe pour éviter les fichiers temporaires
            if isinstance(image_path, (str, bytes)):
                results = self.model.predict(
                    source=image_path,
                    conf=0.25,  # Seuil de confiance minimum
                    stream=False,
                    verbose=False
                )
            else:
                raise ValueError("Type d'image non supporté")
            
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        "type": self.class_map.get(int(box.cls[0]), "unknown"),
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0])
                    })
            return detections
        
        except Exception as e:
            print(f"[ERREUR] Échec de détection : {str(e)}")
            return []
        finally:
            self._restore_torch_load()
