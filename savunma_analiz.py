import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image


@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()


TR_NAMES = {
    "person": "Personel",
    "bicycle": "Bisiklet",
    "car": "Araç",
    "motorcycle": "Motosiklet",
    "airplane": "Hava Aracı / Uçak",
    "bus": "Otobüs / Nakliye",
    "train": "Tren",
    "truck": "Askeri Kamyon / Lojistik",
    "boat": "Deniz Aracı / Bot",
    "bird": "Kuş",
    "backpack": "Sırt Çantası",
    "cell phone": "Telefon"

}


st.set_page_config(page_title="YZ Savunma Analizi", page_icon="🛡️", layout="wide")


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2592/2592201.png", width=100)
st.sidebar.title("Kontrol Merkezi")
st.sidebar.markdown("---")


conf_threshold = st.sidebar.slider(
    "Algılama Hassasiyeti",
    0.01, 1.0, 0.15,
    help="Düşük değerler gizli hedefleri bulmaya yardımcı olur ancak hata payı artabilir."
)

img_size = st.sidebar.selectbox(
    "Tarama Çözünürlüğü",
    [640, 1024, 1280],
    index=1,
    help="Yüksek çözünürlük küçük nesneleri (mühimmat, uzak hedefler) daha iyi yakalar."
)

st.title("🛡️ Yapay Zeka Destekli Stratejik Analiz Sistemi")


uploaded_file = st.file_uploader("Analiz edilecek askeri/stratejik görseli seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orijinal Görüntü")
        st.image(image_rgb, use_container_width=True)


    with st.spinner("Yapay zeka katmanları taranıyor, hedefler analiz ediliyor..."):
        results = model.predict(source=image_bgr, conf=conf_threshold, imgsz=img_size, augment=True)

        with col2:
            st.subheader("Analiz Sonucu")
            
            res_plotted = results[0].plot(labels=False, conf=False)
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, use_container_width=True)


    st.markdown("---")
    found_objects = results[0].boxes

    if len(found_objects) > 0:
        st.subheader(f"🚩 {len(found_objects)} Kritik Hedef Tespit Edildi!")

        for box in found_objects:
            raw_name = model.names[int(box.cls[0])]
           
            name_tr = TR_NAMES.get(raw_name, raw_name).upper()
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

           
            with st.expander(f"HEDEF DETAYI: {name_tr} (Güven: %{conf*100:.1f})"):
                st.write(f"**Tanımlanan Unsur:** {name_tr}")
                st.write(f"**Tespit Doğruluğu:** %{conf*100:.2f}")
                st.write(f"**Konum Bilgisi (Piksel):** Sol Üst: ({int(coords[0])}, {int(coords[1])}) | Sağ Alt: ({int(coords[2])}, {int(coords[3])})")

                if raw_name == "airplane":
                    st.error("⚠️ ANALİZ: Tanımlanamayan hava aracı tespit edildi. Hava sahası ihlali kontrol edilmelidir.")
                elif raw_name == "truck":
                    st.warning("⚠️ ANALİZ: Lojistik veya askeri taşıma aracı olabilir. Hareket yönü takip edilmelidir.")
                elif raw_name == "person":
                    st.info("⚠️ ANALİZ: Bölgede personel hareketliliği saptandı. Kimlik doğrulama gereklidir.")
                elif raw_name == "boat":
                    st.error("⚠️ ANALİZ: Deniz taşıtı tespit edildi. Kıyı güvenliği bilgilendirilmelidir.")
                elif raw_name == "stop sign":
                    st.info("ℹ️ NOT: Trafik işareti tespit edildi. (Düşük hassasiyette yanıltıcı olabilir)")
                else:
                   
                    st.write(f"🔍 ANALİZ: {name_tr} olarak sınıflandırıldı. Ancak bu bir askeri mühimmat olabilir. Uzman incelemesi önerilir.")
    else:
        st.error("Herhangi bir stratejik hedef saptanamadı. Lütfen 'Algılama Hassasiyeti' ayarını kontrol edin.")

st.sidebar.markdown("---")
