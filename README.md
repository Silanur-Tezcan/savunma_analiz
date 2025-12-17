# 🛡️ Yapay Zeka Destekli Stratejik Görüntü Analiz Sistemi

Bu proje, **YOLOv8 derin öğrenme mimarisi** kullanarak yüklenen görüntüler üzerinde
otomatik nesne tespiti yapan, **Streamlit tabanlı interaktif bir analiz platformudur**.

## 🎯 Projenin Amacı

- Görüntüler üzerinden otonom hedef tespiti yapmak
- Savunma, güvenlik ve stratejik analiz senaryolarını simüle etmek
- Yapay zekâ destekli karar destek sistemlerine örnek oluşturmak
- YOLOv8 ve Streamlit entegrasyonunu göstermek

## 🧠 Nasıl Çalışır?

1. YOLOv8-Nano modeli uygulama başlatıldığında yüklenir.
2. Kullanıcı arayüz üzerinden bir görüntü yükler.
3. Kullanıcı:
   - Algılama hassasiyetini (confidence)
   - Görüntü çözünürlüğünü (img size)
   ayarlayabilir.
4. Yapay zekâ modeli görüntüyü tarar ve nesneleri tespit eder.
5. Tespit edilen nesneler:
   - Görsel üzerinde kutularla gösterilir
   - Türkçeleştirilmiş sınıf isimleriyle raporlanır
6. Nesne türüne göre **akıllı analiz ve uyarı mesajları** oluşturulur.

## 🛠 Kullanılan Teknolojiler ve Kütüphaneler

- **Python**
- **Streamlit**
  - Web tabanlı kullanıcı arayüzü
- **OpenCV (cv2)**
  - Görüntü işleme ve format dönüşümleri
- **Ultralytics YOLOv8**
  - Derin öğrenme tabanlı nesne tespiti
- **NumPy**
  - Veri ve görüntü işleme
- **Pillow (PIL)**
  - Görüntü yükleme ve işleme

## 📦 Gereksinimler

```bash
pip install streamlit opencv-python ultralytics numpy pillow
