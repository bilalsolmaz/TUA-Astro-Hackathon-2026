<div align="center">
  <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Police%20car%20light/3D/police_car_light_3d.png" width="80" alt="logo" />
  <h1>RescueRoute | Afet Rota Optimizasyon Sistemi</h1>
  <p><b>TUA Astro Hackathon 2026 Sunum Projesi</b></p>

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![AI & Satellite Analytics](https://img.shields.io/badge/AI_%26_Uydu-Copernicus-004481?style=for-the-badge)](https://www.copernicus.eu/en)

  <p>
    <em>Kahramanmaraş 6 Şubat depremi senaryosu üzerinden modellenmiş, uydu verisi vizyonu ve yapay zeka destekli hasar tespit mantığı ile kurtarma ekiplerine <b>en hızlı ve güvenli rotayı</b> çizen akıllı navigasyon ve lojistik sistemi.</em>
  </p>
</div>

---

## 🌍 Neden RescueRoute?

Standart navigasyon uygulamaları (Google Maps, Yandex) afet anında yıkılmış binaları, çökmüş köprüleri ve enkazla kapanmış yolları hesaba katamaz. **RescueRoute**, kritik ilk 72 saatte arama kurtarma, ambulans ve yardım filolarının afet bölgesinde zaman ve hayat kaybetmesini önlemek için tasarlandı.

- ⏱ **Saniyeler İçinde Hayati Karar:** Hasarlı ve riskli yolları algılayarak alternatif güvenli güzergah planlar.
- 🛰 **Uydu Verisi Hazır Mimari:** Copernicus Sentinel-2 sistemi ve gelişmiş segmentasyon (SAM) modellerine entegre edilebilecek modüler bir altyapı üzerine kuruludur.
- 🎛 **Dinamik Önceliklendirme (A*):** Ekipler anlık duruma göre riskten kaçınma (α), kısa mesafe (β) veya tahmini varış süresi (γ) önceliklerini esnek olarak belirleyebilir.

---

## 🚀 Jüri İçin "Wow" Demo Senaryosu

Jüri değerlendirmesinde projenin kapasitesini sadece birkaç saniyede, en çarpıcı şekilde göstermek için şu adımları izleyin:

1. **Önce Durumu Anlatın:** Sistemdeki yolların **Yeşil (Açık)**, **Sarı (Riskli)** ve **Kırmızı (Göçük/Kapalı)** olarak yapay zeka ile haritalandırıldığını gösterin.
2. **Başlangıç & Hedef:** 
   - Başlangıç Noktası: **"Pazarcık Merkez (Episantr Yakını)"**
   - Hedef Noktası: **"KMaraş Eğitim Araştırma Hastanesi"**
3. **Parametre Etkisi (Kritik Aşama):**
   - Sol panelden *Hasar Risk Ağırlığını* (α) önce **1.0 (Düşük)** olarak seçip *Rotayı Hesapla*'ya basın. Sistem kısa olduğu için riskli (kırmızı) sokakların içinden geçmeyi önerecektir.
   - Ardından ağırlığı **10.0 (Yüksek)** olarak ayarlayıp tekrar hesaplayın.
4. **Sonuç:** Algoritmanın kırmızılı alanlardan asimptotik olarak nasıl kaçtığını, belki daha uzun ama **tamamen yeşil ve güvenli** çevre yolunu nasıl bulduğunu gösterin!

---

## ⚙️ Teknik Altyapı & Teknolojiler

Proje, hızlı veri işleme ve anlık reaksiyon yeteneği gözetilerek Python veri bilimi ekosistemi üzerinde geliştirilmiştir.

| Mimari | Teknoloji | Amaç |
| :--- | :--- | :--- |
| **Ön Yüz (UI)** | `Streamlit`, `Folium` | Etkileşimli analitik paneli ve katmanlı dinamik vektör haritalar. |
| **Topoloji & Çizge Alg.** | `OSMnx`, `NetworkX` | Hedef bölgenin tüm vektörel cadde/sokak ağını düğüm ve kenarlar halinde (graph) belleğe alır. |
| **Optimizasyon Motoru** | `Custom A* (A-Star)` | Kendi yazdığımız özel A* heuristik fonksiyonu ile mesafe, süre ve **Yapay Zeka Risk Skorunu** birleştirir. |

---

## 💻 Kurulum ve Çalıştırma (Demo İçin)

Proje sadece birkaç komutla çalışmaya hazır hale gelir:

### 1. Sanal Ortam (Virtual Environment)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Gerekli Paketlerin Yüklenmesi
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatma
```bash
streamlit run app.py
```
> Otomatik olarak tarayıcınızda `http://localhost:8501` adresinde açılacaktır.
> 💡 **Önemli Not:** İlk açılışta OSM yol ağı canlı olarak indirilir ve `cache/` dizinine kaydedilir (~30-40 sn sürebilir). Sonraki açılışlarda anında yüklenir!

---

## 🔭 Gelecek Vizyonu (Prodüksiyon Modeli)

Şu anki demoda hasar durumu episantr uzaklığı ile simule edilmektedir. Gerçeğe dönüşecek vizyonumuzda sistem şu mimari üzerinden çalışacaktır:

```python
from segment_anything import SamPredictor
import rasterio

def compute_damage_score(lat, lon):
    # 1. Copernicus / Göktürk uydularından kriz anında en güncel görüntüyü çek.
    # 2. SAM (Segment Anything Model) ile yıkıntıları görüntü üzerinden çıkar (segmentasyon).
    # 3. OSMNx düğümleri ile çakıştır ve her yola %0-100 enkaz/kapanma risk skoru ata.
    return ai_generated_real_risk
```

<br>
<div align="center">
  <p><b>TUA Astro Hackathon 2026 için gökyüzüne bakarak, dünyayı kurtarmak için geliştirildi. 🚀</b></p>
</div>
