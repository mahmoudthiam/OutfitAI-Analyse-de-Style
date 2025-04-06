import streamlit as st

# DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(page_title="OutfitAI - Analyse de Style", layout="wide")

import cv2
import numpy as np
import tempfile
import os
from detector import FashionDetector
from style_analyzer import *

# Initialisation du détecteur
@st.cache_resource
def load_detector():
    return FashionDetector()

detector = load_detector()

# Titre de l'application (après set_page_config)
st.title("👗 OutfitAI - Analyseur de Tenue")

# Le reste de votre code reste inchangé...
with st.sidebar:
    st.header("Paramètres")
    confidence_threshold = st.slider("Seuil de confiance", 0.1, 1.0, 0.5)
    show_dominant_colors = st.checkbox("Afficher les couleurs dominantes", True)
    show_technical = st.checkbox("Afficher les détails techniques", False)

# [...] (le reste de votre code original)
# Zone de téléchargement
uploaded_file = st.file_uploader("Téléversez votre tenue", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Lecture de l'image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Sauvegarde temporaire pour la détection
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    # Détection
    with st.spinner("Analyse en cours..."):
        try:
            detections = detector.detect(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    if not detections:
        st.warning("Aucun vêtement détecté ! Essayez avec une autre image.")
    else:
        # Affichage en deux colonnes
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image Originale")
            st.image(uploaded_file, use_column_width=True)
            
        with col2:
            # Dessiner les bounding boxes
            annotated_image = image.copy()
            for det in detections:
                if det["confidence"] >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det['type']} {det['confidence']:.0%}"
                    cv2.putText(annotated_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            st.subheader("Détections")
            st.image(annotated_image, channels="BGR", use_column_width=True)
        
        # Section d'analyse
        st.divider()
        st.subheader("📊 Analyse Complète")
        
        # Filtrage des détections par seuil de confiance
        filtered_detections = [d for d in detections if d["confidence"] >= confidence_threshold]
        
        # Correction critique : vérification du nombre de détections
        num_columns = max(1, min(4, len(filtered_detections)))  # Garantit au moins 1 colonne
        cols = st.columns(num_columns)
        
        for idx, det in enumerate(filtered_detections[:num_columns]):
            with cols[idx]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                crop_img = image[y1:y2, x1:x2]
                
                with st.expander(f"{det['type'].upper()} ({det['confidence']:.0%})"):
                    st.image(crop_img, channels="BGR")
                    st.write(f"**Type:** {det['type']}")
                    st.write(f"**Confiance:** {det['confidence']:.0%}")
                    st.write(f"**Coupe:** {analyze_fit(det['bbox'], image)}")
                    
                    if show_dominant_colors:
                        try:
                            colors = get_dominant_color(crop_img)
                            st.write("**Couleurs dominantes:**")
                            cols_color = st.columns(len(colors))
                            for i, color in enumerate(colors):
                                with cols_color[i]:
                                    st.color_picker(
                                        label="", 
                                        value=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", 
                                        key=f"color_{idx}_{i}"
                                    )
                        except Exception as e:
                            st.warning(f"Erreur d'analyse couleur : {str(e)}")

        # Suggestions basiques
        if len(filtered_detections) >= 2:
            tops = [d for d in filtered_detections if d["type"] in ["top", "jacket"]]
            bottoms = [d for d in filtered_detections if d["type"] == "bottom"]
            
            if tops and bottoms:
                st.divider()
                st.subheader("💡 Suggestions d'Association")
                
                try:
                    # Analyse des couleurs
                    top_colors = get_dominant_color(image[
                        int(tops[0]["bbox"][1]):int(tops[0]["bbox"][3]),
                        int(tops[0]["bbox"][0]):int(tops[0]["bbox"][2])
                    ])
                    
                    bottom_colors = get_dominant_color(image[
                        int(bottoms[0]["bbox"][1]):int(bottoms[0]["bbox"][3]),
                        int(bottoms[0]["bbox"][0]):int(bottoms[0]["bbox"][2])
                    ])
                    
                    # Affichage des résultats
                    col_top, col_bottom = st.columns(2)
                    with col_top:
                        st.write(f"🔷 **Haut {tops[0]['type']}**")
                        st.color_picker(
                            "Couleur dominante", 
                            value=f"#{top_colors[0][0]:02x}{top_colors[0][1]:02x}{top_colors[0][2]:02x}",
                            key="top_color"
                        )
                    
                    with col_bottom:
                        st.write(f"🔶 **Bas {bottoms[0]['type']}**")
                        st.color_picker(
                            "Couleur dominante", 
                            value=f"#{bottom_colors[0][0]:02x}{bottom_colors[0][1]:02x}{bottom_colors[0][2]:02x}",
                            key="bottom_color"
                        )
                    
                    # Logique de compatibilité améliorée
                    color_diff = np.mean(np.abs(top_colors[0] - bottom_colors[0]))
                    if color_diff > 100:
                        st.success("✅ Bon contraste de couleurs !")
                    elif color_diff > 50:
                        st.info("🌤️ Combinaison neutre, peut être améliorée")
                    else:
                        st.warning("⚠️ Contrast faible, essayez avec des couleurs plus opposées")
                
                except Exception as e:
                    st.error(f"Erreur d'analyse de couleur : {str(e)}")

        # Détails techniques (optionnel)
        if show_technical and detections:
            st.divider()
            with st.expander("🔍 Détails techniques (debug)"):
                st.json(detections)