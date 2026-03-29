# 🚨 Afet Kurtarma Rota Optimizasyon Sistemi
### TUA Astro Hackathon 2026 — Demo Projesi

Kahramanmaraş 6 Şubat 2023 depremini baz alarak, uydu verisi YZ analizi ile
kurtarma ekiplerine en güvenli rotayı gösteren interaktif sistem.

---

## ⚡ Hızlı Başlangıç

### 1. Python ortamı oluştur

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 3. Uygulamayı başlat

```bash
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

---

## 📁 Proje Yapısı

```
rescue_route/
├── app.py              ← Ana Streamlit arayüzü
├── damage_model.py     ← Uydu YZ analizi (hasar skoru hesabı)
├── router.py           ← A* rota optimizasyon motoru
├── requirements.txt    ← Bağımlılıklar
├── cache/              ← OSM yol ağı önbelleği (otomatik oluşur)
└── README.md
```

---

## 🗺️ İlk Açılışta Ne Olur?

1. **OSM yol ağı indirilir** (~30 sn, internet gerekli)  
   Kahramanmaraş şehir merkezi ve çevresi (`37.45–37.65°N, 36.82–37.05°E`)

2. **Hasar skorları hesaplanır**  
   Her yol kenarı için `damage_model.py` çalışır (episantr uzaklığı modeli)

3. **Harita yüklenir**  
   Yollar yeşil/sarı/kırmızı renk kodlamasıyla görüntülenir

Sonraki açılışlarda `cache/` dizininden yüklenerek **anında** başlar.

---

## 🔬 Gerçek Sistemde Nasıl Çalışır?

`damage_model.py` → `compute_damage_score()` fonksiyonunu şununla değiştir:

```python
from segment_anything import SamPredictor
import rasterio

def compute_damage_score(lat, lon, road_type=None, seed=None):
    # 1. Copernicus'tan Sentinel-2 görüntüsü indir
    # 2. SAM modeli ile segmentasyon çalıştır
    # 3. Koordinattaki piksel değerini döndür
    ...
```

---

## 🎯 Demo Senaryosu (Sunum için)

1. Başlangıç: **Pazarcık Merkez (Episantr Yakını)**
2. Hedef: **KMaraş Eğitim Araştırma Hastanesi**
3. "Güvenli Rotayı Hesapla" → A* çalışır
4. Alternatif rotaları karşılaştır
5. α/β/γ ağırlıklarını değiştir → farklı rotalar görün

---

## 🛠️ Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `osmnx` indirme hatası | İnternet bağlantısını kontrol et |
| `streamlit_folium` bulunamadı | `pip install streamlit-folium` |
| Harita yüklenmez | Tarayıcı konsolunu kontrol et |
| Rota bulunamadı | Farklı başlangıç/bitiş noktası seç |
