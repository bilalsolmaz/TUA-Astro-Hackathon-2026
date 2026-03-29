<div align="center">
  <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Police%20car%20light/3D/police_car_light_3d.png" width="80" alt="logo" />
  <h1>RescueRoute | Akıllı Afet Rota Optimizasyon Sistemi</h1>
  <p><b>TUA Astro Hackathon 2026 Finalist Projesi</b></p>

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![Copernicus EMS](https://img.shields.io/badge/Gerçek_Uydu_Verisi-Copernicus-004481?style=for-the-badge)](https://emergency.copernicus.eu/mapping/list-of-components/EMSR648)
  [![Live Demo](https://img.shields.io/badge/Canl%C4%B1_Yay%C4%B1nda-Web_Sitesi-10a37f?style=for-the-badge)](https://bilalsolmaz.com/)

  <p>
    <em>Kahramanmaraş 6 Şubat depremi senaryosu üzerinden modellenmiş, <b>Copernicus Gerçek Uydu Verisi (EMSR648)</b> ile anlık enkaz analizi yapan ve kurtarma ekiplerine 2 aşamalı <b>en güvenli rotayı</b> çizen lojistik yönetim platformu.</em>
  </p>
  <h3>🔴 CANLI DEMO: <a href="https://bilalsolmaz.com/">bilalsolmaz.com</a></h3>
</div>

---

## 🌍 Neden RescueRoute?

Standart navigasyon uygulamaları (Google Maps, Yandex vb.) olağanüstü afet durumlarında yıkılmış binaları, çökmüş köprüleri ve enkazla kapanmış yolları hesaba katamaz. **RescueRoute**, kritik ilk 72 saatte arama kurtarma, ambulans ve yardım filolarının afet bölgesinde zaman ve hayat kaybetmesini önlemek için tasarlandı.

### ⭐ Öne Çıkan Özellikler

- 🛰 **İki Katmanlı Hasar Modeli (Gerçek Veri):** Sadece bir simülasyon değil! **Copernicus EMS EMSR648** aktivasyonundan alınan gerçek uydu analizlerini kullanarak bina yıkım haritalarını yol ağlarına yansıtır (`scipy.spatial.cKDTree` ile optimize edilmiştir). Veri olmayan kör noktalarda algoritmik **GMPE (Ground Motion Prediction Equation)** modelini devreye sokar.
- 🚒 **Çoklu Ekip Planlaması:** İl içi (AFAD, İtfaiye, UMKE) ekiplerinin yanı sıra Şehir Dışından yardıma gelen (Adana, Gaziantep vb. karayolu giriş noktalarından) kurtarma konvoyları için de dinamik rotalama sağlar.
- 🔁 **2 Aşamalı Operasyon (Faz 1 & Faz 2):** AFAD Koordinasyon Merkezinden → Enkaza (Faz 1) ve Enkazdan → Hastaneye (Faz 2) tek tıklamayla birleşik kurtarma rotası planlar.
- 🎛 **Dinamik Optimizasyon Profilleri (A* ve Algoritma Ağırlıkları):** Saha komutanları duruma göre **🛡️ Güvenlik Öncelikli**, **⚡ Hız Öncelikli** veya **📐 Dengeli** algoritmalarını (Mesafe `α`, Hasar Risk `β`, Süre `γ` parametrelerini) manipüle edebilir.
- 📡 **Çevrimdışı (Offline) Yönlendirme Kartları:** Afet alanında internet kesilme ihtimaline karşı rota üzerindeki kritik kavşak koordinatları `.json` veya harici veri formatlarında cihaza kaydedilip (Çevrimdışı Kart), sahaya otonom aktarılabilir.

---

## 🚀 Jüri İçin Demo Senaryosu (Nasıl Test Edilmeli?)

Projenin haritalama ve A-Star algoritması üzerindeki başarısını [canlı sistem üzerinde](https://bilalsolmaz.com/) sadece birkaç saniyede jüriye kanıtlamak için:

1. **Operasyon Seçimi:** Sol sekmeden "2️⃣ İki Aşamalı (Merkez → Enkaz → Hastane)" seçeneğini işaretleyin.
2. **Akıllı Geocoding (Enkaz Arama):** Enkaz noktası yöntemi için **"🔍 Adres / yer adı ara"** moduna geçin ve `Sofular Mah, Pazarcık` yazın. Sistem OSM üzerinden veriyi indirecek ve tam konumlandıracaktır.
3. **Parametre Farkını (Gücü) Gösterin (Kritik Adım):**
   - Ayarlardan hızlı profil olarak **"⚡ Hız Öncelikli"** (veya Manuel ayardan Mesafe(α)=0.70, Hasar(β)=0.10) yapın. *Haritayı Hesapla* butonuna basın. Rota kısa yolu seçecek ve riskli (kırmızı/sarı) sokaklara girmek zorunda kalacaktır.
   - Ardından profili **"🛡️ Güvenlik Öncelikli"** (Manuel ayar Hasar Risk(β)=0.80) olarak güncelleyin ve tekrar hesaplayın.
4. **Wow Etkisi:** Algoritmanın enkaz, göçük ve kırmızı işaretli riskli alanlardan nasıl asimptotik şekilde kaçtığını, çevre yolunu bularak tehlikesiz, yeşil ve güvenli bir alternatif haritada çizdiğini saniyeler içinde kanıtlayın.

---

## ⚙️ Teknik Altyapı & Mimari

Sistem yüz binlerce düğüm ve kenarı barındıran kompleks mekansal verileri milisaniyeler içinde işlemek için tasarlandı.

| Çerçeve Türü | Teknoloji / Kütüphane | Kullanım Amacı |
| :--- | :--- | :--- |
| **Arayüz (UI) & Katmanlı Harita** | `Streamlit`, `Folium` | Etkileşimli veri görselleştirme paneli, katmanlı dinamik vektör haritalar. |
| **Topoloji & Çizge Analizi** | `OSMnx`, `NetworkX` | Hedef il ve ilçenin tüm vektörel cadde ağını Graph nesnelerine dönüştürür. |
| **Uzamsal Analiz (Fast Query)** | `scipy.spatial.cKDTree`, `GeoPandas` | On binlerce Copernicus uydu hasar noktasını, cadde ve sokak yollarıyla uzamsal algoritmalarla çok hızlı eşleyerek hasar yüzdesi çıkarır. |
| **Optimizasyon Motoru** | `Custom A-Star (A*) / Dijkstra` | Yen's K-Shortest Paths ve özelleştirilmiş Heuristik. Hasar seviyesine göre taşıt hızları 50 km/s'den 1.5 km/s'ye kadar dinamik düşürülür ve süre uzatılır. |

---

## 💻 Kurulum ve Çalıştırma (Lokal Cihaz İçin)

Projeyi kendi ortamınızda test etmek için:

### 1. Ortam Hazırlığı
```bash
python -m venv venv

# Windows için:
venv\Scripts\activate

# macOS / Linux için:
source venv/bin/activate
```

### 2. Gerekli Kütüphanelerin Kurulumu
```bash
pip install -r requirements.txt
```

### 3. Sistemi Başlatma
```bash
streamlit run app.py
```
> Otomatik olarak tarayıcınızda `http://localhost:8501` adresinde açılacaktır. İlk açılışta `cache/` (önbellek) klasörüne yol ağları kaydedildiği için 30-40 sn sürebilir. Sonraki kullanımlar anında gerçekleşir.

---

## 🔭 Gelecek Vizyonu ve Bu Deponun (Repository) Kapsamı

Şu an incelemekte olduğunuz bu GitHub reposu, TUA Astro Hackathon 2026 değerlendirmesi için sunulmuş **tamamen çalışan ve test edilebilir** ana prototiptir. 

Mevcut sistemde Avrupa Uzay Ajansı'nın **Copernicus EMS** verilerini aktif olarak işleyebilen işlevsel bir altyapı mevcuttur. **Nihai hedefimiz ise:** Olası bir afet anında kendi yerli uydularımızdan (`Göktürk Serisi`, `İMECE`) gelen ham `.tiff` raster uydu görsellerini bulut sunucularda `Segment Anything Model (SAM)` ve uzamsal `U-Net` yapay zeka mimarileriyle otonom olarak parçalara ayırmaktır. 

Bu repo, uydudan alınan hasar verisini yeryüzündeki ekiplere **saniyeler içinde** rota olarak aktaracak o devasa "Gerçek Zamanlı Karar Destek Sisteminin" kanıtlanmış ilk çekirdeğidir.

<br>
<div align="center">
  <p><b>TUA Astro Hackathon 2026 Jürisine Saygılarımızla...</b><br><i>Gökyüzüne bakarak, hayat kurtarmak için geliştirildi. 🚀</i></p>
</div>
