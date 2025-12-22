import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image


st.set_page_config(page_title="YZ Savunma Analizi v4", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()



ALLOWED_CLASSES = [
    0,  # Person (Asker)
    2,  # Car (Araç)
    4,  # Airplane (Uçak)
    5,  # Bus
    6,  # Train
    7,  # Truck (Kamyon)
    8,  # Boat (Gemi)
    25, # Umbrella (YOLO bunu genelde ÇADIR olarak algılar)
    24, # Backpack (Mühimmat Çantası)
    26, # Handbag (Şüpheli Paket)
    28, # Suitcase (Mühimmat Kutusu)
    39, # Bottle (Havan mermisi/Silindirik cisim)
    32  # Sports ball (Yuvarlak Mayın)
]


TR_NAMES = {
    "person": "Personel / Asker",
    "car": "Araç",
    "airplane": "Hava Aracı",
    "truck": "Askeri Kamyon",
    "boat": "Deniz Aracı",
    "umbrella": "Çadır / Barınak", 
    "backpack": "Sırt Çantası / Teçhizat",
    "suitcase": "Mühimmat Kutusu",
    "handbag": "Şüpheli Paket",
    "sports ball": "Mayın (Yuvarlak)",
    "bottle": "Mühimmat (Silindirik)"
}

# --- FONKSİYON: EŞİKLEME (KONTRAST) MODU ---

def detect_dark_spots(image, threshold_val, min_area, max_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
  
    kernel = np.ones((5,5), np.uint8) 
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    processed_img = image.copy()
    all_valid_points = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
           
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            
          
            if circularity > 0.5 and solidity > 0.8:
                all_valid_points.append(cnt)
   
    if len(all_valid_points) > 0:
        combined_points = np.vstack(all_valid_points)
        x, y, w, h = cv2.boundingRect(combined_points)
       
        cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        detections.append((x, y, w, h))
            
    return processed_img, detections, clean_mask

# --- ARAYÜZ ---
st.sidebar.title("🛡️ Kontrol Paneli")
mode = st.sidebar.radio("Tarama Modu:", ["🧠 Yapay Zeka (Genel Tarama)", "⚫ Kontrast (Mayın/Mühimmat)"])
st.sidebar.markdown("---")

if mode == "⚫ Kontrast (Mayın/Mühimmat)":
    st.sidebar.subheader("Mayın Tespit Ayarları")
    threshold_val = st.sidebar.slider("Koyuluk Eşiği", 0, 255, 45)
    min_area = st.sidebar.slider("Min Boyut", 1, 500, 100)
    max_area = st.sidebar.slider("Max Boyut", 500, 50000, 45000)
else:
    st.sidebar.subheader("Yapay Zeka Ayarları")
    conf_threshold = st.sidebar.slider("Güven Eşiği", 0.0, 1.0, 0.20)

st.title("🛡️ Entegre Savunma Analiz Sistemi")

uploaded_file = st.file_uploader("Görüntü Yükle...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Orijinal Görüntü")
        st.image(image_rgb, use_container_width=True)

   
    if mode == "⚫ Kontrast (Mayın/Mühimmat)":
        res_img, detections, mask_img = detect_dark_spots(image_bgr, threshold_val, min_area, max_area)
        res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("Hedef Bölge")
            st.image(res_rgb, use_container_width=True)
            with st.expander("Maske Görünümü"):
                st.image(mask_img)
        
        if len(detections) > 0:
            st.error("⚠️ Kritik bölge işaretlendi.")


    else:
        with st.spinner("YZ Taraması Yapılıyor..."):
           
            results = model.predict(
                image_bgr, 
                conf=conf_threshold, 
                classes=ALLOWED_CLASSES
            )
            
          
            res_plotted = results[0].plot(labels=False, conf=False)
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Tespit Edilen Unsurlar")
                st.image(res_rgb, use_container_width=True)
            
           
            st.markdown("---")
            found_objects = results[0].boxes
            if len(found_objects) > 0:
                st.info(f"Toplam {len(found_objects)} unsur tespit edildi.")
                for box in found_objects:
                    cls_id = int(box.cls[0])
                    raw_name = model.names[cls_id]
                  
                    display_name = TR_NAMES.get(raw_name, raw_name).upper()
                    conf = float(box.conf[0])
                    
                    st.write(f"📍 **{display_name}** - Güven: %{conf*100:.1f}")
            else:
                st.warning("Bu ayarlarda bir tehdit unsuru bulunamadı.")
