import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image, k=3):
    """Extrait les k couleurs dominantes (format BGR)"""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10)  # n_init pour éviter les warnings
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

def analyze_fit(bbox, image):
    """Détermine si un vêtement est 'slim' ou 'oversized'"""
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]
    
    # Ratio largeur/hauteur normalisé
    ratio = (x2 - x1) / (y2 - y1)
    return "slim" if ratio < 0.7 else "oversized"

def load_image(image_path):
    """Charge une image en BGR (compatible OpenCV)"""
    return cv2.imread(image_path)