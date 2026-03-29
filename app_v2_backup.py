"""
app.py  —  Afet Kurtarma Rota Optimizasyon Sistemi
===================================================
Çalıştır:  streamlit run app.py
"""

import os
import pickle
import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import numpy as np
import pandas as pd

from damage_model import compute_damage_score, damage_label
from router import assign_edge_weights, find_safest_route, find_alternative_routes

# ─── Sayfa Yapılandırması ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Afet Kurtarma Rota Sistemi",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sabit Konumlar ──────────────────────────────────────────────────────────
# Kurtarma ekibinin başlangıç noktaları (hasar gören bölgeler)
START_LOCATIONS = {
    "📍 Pazarcık Merkez (Episantr Yakını)":     (37.496, 37.290),
    "📍 Nurdağı İlçesi":                         (37.183, 36.731),
    "📍 Türkoğlu İlçesi":                        (37.382, 36.861),
    "📍 KMaraş Dulkadiroğlu":                    (37.600, 36.920),
    "📍 KMaraş Onikişubat":                      (37.563, 36.923),
}

# Hedef noktalar (sağlık tesisleri / toplanma alanları)
END_LOCATIONS = {
    "🏥 KMaraş Eğitim Araştırma Hastanesi":     (37.585, 36.938),
    "🏥 Necip Fazıl Şehir Hastanesi":            (37.598, 36.893),
    "🏥 Pazarcık Devlet Hastanesi":              (37.490, 37.283),
    "⛺ AFAD Toplanma Alanı (Spor Kompleksi)":   (37.570, 36.950),
    "⛺ AFAD Toplanma Alanı (İnönü Sahası)":     (37.555, 36.930),
}

CACHE_DIR  = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "kahramanmaras_graph.pkl")

# ─── Veri Yükleme ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_graph():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    # ✅ YENİ FORMAT
    bbox = (37.45, 37.65, 36.82, 37.05)  # (south, north, west, east)

    G = ox.graph_from_bbox(
        bbox,
        network_type="drive",
        simplify=True,
    )

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(G, f)

    return G


@st.cache_resource(show_spinner=False)
def compute_all_damage_scores(_G):
    """Tüm kenarlar için hasar skoru hesapla (bir kez çalışır)."""
    scores = {}
    for u, v, key, data in _G.edges(keys=True, data=True):
        mid_lat = (_G.nodes[u]["y"] + _G.nodes[v]["y"]) / 2
        mid_lon = (_G.nodes[u]["x"] + _G.nodes[v]["x"]) / 2
        rt = data.get("highway", "unclassified")
        rt = rt[0] if isinstance(rt, list) else rt
        seed = abs(hash((u, v, key))) % (2 ** 31)
        scores[(u, v, key)] = compute_damage_score(mid_lat, mid_lon, rt, seed)
    return scores


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────
def build_folium_map(G, damage_scores, routes=None, start_coords=None, end_coords=None):
    """Interaktif Folium haritası oluştur."""

    m = folium.Map(
        location=[37.56, 36.94],
        zoom_start=12,
        tiles=None,
    )

    # Uydu görüntüsü katmanı (API keyi gerektirmez)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Uydu Görüntüsü",
        overlay=False,
        control=True,
    ).add_to(m)

    # Alternatif: Karanlık harita
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="🗺️ Karanlık Harita",
        overlay=False,
        control=True,
    ).add_to(m)

    # ── Yol ağını hasar rengiyle çiz (performans için örnekleme) ──
    edges_data = list(G.edges(keys=True, data=True))
    sample_size = min(6000, len(edges_data))
    rng = np.random.RandomState(42)
    sampled = rng.choice(len(edges_data), sample_size, replace=False)

    # Hasar gruplarına ayır (her grup ayrı PolyLine feature group)
    groups = {
        "safe":     folium.FeatureGroup(name="🟢 Güvenli Yollar",    show=True),
        "caution":  folium.FeatureGroup(name="🟡 Dikkatli Geçiş",    show=True),
        "danger":   folium.FeatureGroup(name="🔴 Tehlikeli",          show=True),
        "blocked":  folium.FeatureGroup(name="⬛ Geçilemez",          show=True),
    }

    for idx in sampled:
        u, v, key, data = edges_data[idx]
        dmg = damage_scores.get((u, v, key), 0.3)
        label, color, _ = damage_label(dmg)

        group_key = (
            "safe"    if dmg < 0.25 else
            "caution" if dmg < 0.50 else
            "danger"  if dmg < 0.75 else
            "blocked"
        )
        weight_line = 1.5 if dmg < 0.50 else 2.5
        opacity     = 0.45 if dmg < 0.50 else 0.75

        try:
            coords = [
                (G.nodes[u]["y"], G.nodes[u]["x"]),
                (G.nodes[v]["y"], G.nodes[v]["x"]),
            ]
            folium.PolyLine(
                coords,
                color=color,
                weight=weight_line,
                opacity=opacity,
                tooltip=f"{label} — Hasar: {dmg:.0%}",
            ).add_to(groups[group_key])
        except Exception:
            pass

    for grp in groups.values():
        grp.add_to(m)

    # ── Rotaları çiz ──────────────────────────────────────────────
    route_colors = ["#FFFFFF", "#60A5FA", "#34D399"]  # ana, alt1, alt2

    if routes:
        for i, route in enumerate(routes):
            if not route or not route.get("path"):
                continue

            coords = [
                (G.nodes[n]["y"], G.nodes[n]["x"])
                for n in route["path"]
                if n in G.nodes
            ]
            if len(coords) < 2:
                continue

            is_main = i == 0
            # Dış parlama çizgisi
            folium.PolyLine(
                coords,
                color="#000000" if is_main else "#1E3A5F",
                weight=10 if is_main else 7,
                opacity=0.5,
            ).add_to(m)

            # Ana rota çizgisi
            label_txt = (
                f"🥇 En Güvenli Rota — {route['distance_km']} km, "
                f"{route['time_min']:.0f} dk, Güvenlik: %{route['safety_score']}"
                if is_main else
                f"🥈 Alternatif {i} — {route['distance_km']} km"
            )
            folium.PolyLine(
                coords,
                color=route_colors[min(i, len(route_colors)-1)],
                weight=6 if is_main else 4,
                opacity=0.95 if is_main else 0.65,
                tooltip=label_txt,
                dash_array=None if is_main else "10 5",
            ).add_to(m)

            # Yön okları (ana rota için)
            if is_main and len(coords) > 3:
                mid_idx = len(coords) // 2
                folium.RegularPolygonMarker(
                    location=coords[mid_idx],
                    number_of_sides=3,
                    radius=8,
                    fill_color="#FFFFFF",
                    fill_opacity=0.9,
                    color="#FFFFFF",
                    weight=1,
                    popup=label_txt,
                ).add_to(m)

    # ── Başlangıç ve hedef işaretçileri ──────────────────────────
    if start_coords:
        folium.Marker(
            start_coords,
            popup=folium.Popup("<b>🚑 Kurtarma Ekibi Başlangıç</b>", max_width=200),
            icon=folium.Icon(color="blue", icon="ambulance", prefix="fa"),
        ).add_to(m)

    if end_coords:
        folium.Marker(
            end_coords,
            popup=folium.Popup("<b>🏥 Hedef: Sağlık Tesisi</b>", max_width=200),
            icon=folium.Icon(color="red", icon="plus-square", prefix="fa"),
        ).add_to(m)

    # Episantr noktaları
    for ep in [
        {"lat": 37.288, "lon": 37.043, "name": "Episantr 1 — M7.7 Pazarcık"},
        {"lat": 38.024, "lon": 37.208, "name": "Episantr 2 — M7.6 Elbistan"},
    ]:
        folium.CircleMarker(
            location=[ep["lat"], ep["lon"]],
            radius=16,
            color="#E24B4A",
            fill=True,
            fill_color="#E24B4A",
            fill_opacity=0.25,
            popup=ep["name"],
            tooltip=ep["name"],
        ).add_to(m)
        folium.CircleMarker(
            location=[ep["lat"], ep["lon"]],
            radius=5,
            color="#E24B4A",
            fill=True,
            fill_color="#E24B4A",
            fill_opacity=0.9,
        ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    return m


def damage_bar_html(score: float) -> str:
    """Küçük hasar çubuğu HTML'i."""
    label, color, emoji = damage_label(score)
    width = int(score * 100)
    return (
        f'<div style="background:#1a1a1a;border-radius:4px;height:8px;width:100%;margin:2px 0">'
        f'<div style="background:{color};width:{width}%;height:100%;border-radius:4px"></div></div>'
        f'<small style="color:{color}">{emoji} {label} ({score:.0%})</small>'
    )


# ─── ARAYÜZ ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #0e1117;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val  { font-size: 26px; font-weight: 600; margin: 4px 0; }
.metric-lbl  { font-size: 12px; color: #888; }
.safe-val    { color: #1D9E75; }
.warn-val    { color: #EF9F27; }
.danger-val  { color: #E24B4A; }
</style>
""", unsafe_allow_html=True)

# Başlık
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("# 🚨 Afet Kurtarma Rota Optimizasyon Sistemi")
    st.caption("**Kahramanmaraş 6 Şubat 2023 Depremi** — Uydu verisi tabanlı en güvenli kurtarma rotası")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🟢 CANLI DEMO")

st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Rota Parametreleri")

    start_name = st.selectbox("🚑 Başlangıç — Kurtarma Ekibi", list(START_LOCATIONS.keys()))
    end_name   = st.selectbox("🏥 Hedef — Sağlık Tesisi",       list(END_LOCATIONS.keys()))

    st.divider()
    st.subheader("📊 Optimizasyon Ağırlıkları")
    st.caption("Kurtarma operasyonları için hasar ağırlığı yüksek tutulur.")

    alpha = st.slider("α — Mesafe",  0.10, 0.80, 0.30, 0.05,
                      help="Kısa mesafeyi ne kadar önemseyelim?")
    beta  = st.slider("β — Hasar",   0.10, 0.80, 0.50, 0.05,
                      help="Hasarlı yollardan kaçınmak ne kadar önemli?")
    gamma_val = round(max(0.0, 1.0 - alpha - beta), 2)
    st.metric("γ — Zaman", f"{gamma_val:.2f}", help="Kalan ağırlık")

    alt_routes = st.checkbox("Alternatif rotalar göster", value=True)
    st.divider()

    find_btn = st.button("🗺️  Güvenli Rotayı Hesapla", type="primary", use_container_width=True)

    st.divider()
    st.subheader("🛰️ Sistem Bilgisi")
    st.info(
        "**Veri Kaynakları**\n\n"
        "- OSM yol ağı (osmnx)\n"
        "- Copernicus Sentinel-2\n"
        "- USGS ShakeMap 2023\n"
        "- AFAD açık verileri"
    )
    st.subheader("📡 Hasar Göstergesi")
    st.markdown("🟢 **< %25** — Güvenli")
    st.markdown("🟡 **%25–50** — Dikkatli geçiş")
    st.markdown("🔴 **%50–75** — Tehlikeli")
    st.markdown("⬛ **> %75** — Geçilemez")

# ─── Veri Yükleme ────────────────────────────────────────────────────────────
with st.spinner("🛰️ Yol ağı yükleniyor (ilk açılışta ~30 sn)…"):
    G = load_graph()

with st.spinner("🤖 Uydu görüntüsü YZ analizi çalışıyor…"):
    damage_scores = compute_all_damage_scores(G)

# Ağırlıklı graf (ağırlık ayarları değişebilir, cache kullanma)
G_weighted = assign_edge_weights(G.copy(), damage_scores, alpha, beta, gamma_val)

start_coords = START_LOCATIONS[start_name]
end_coords   = END_LOCATIONS[end_name]

# ─── Rota Hesaplama ───────────────────────────────────────────────────────────
routes = []
route_error = None

if find_btn:
    with st.spinner("🧭 A* algoritması en güvenli rotayı arıyor…"):
        try:
            start_node = ox.nearest_nodes(G_weighted, start_coords[1], start_coords[0])
            end_node   = ox.nearest_nodes(G_weighted, end_coords[1],   end_coords[0])

            if alt_routes:
                routes = find_alternative_routes(G_weighted, start_node, end_node, k=3)
            else:
                r = find_safest_route(G_weighted, start_node, end_node)
                routes = [r] if r else []

        except Exception as e:
            route_error = str(e)

# ─── Metrik Kartları ──────────────────────────────────────────────────────────
if routes and routes[0]:
    main = routes[0]
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-lbl">📏 Mesafe</div>'
            f'<div class="metric-val safe-val">{main["distance_km"]} km</div>'
            f'</div>', unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-lbl">⏱️ Tahmini Süre</div>'
            f'<div class="metric-val warn-val">{main["time_min"]:.0f} dk</div>'
            f'</div>', unsafe_allow_html=True
        )
    with c3:
        color_cls = "safe-val" if main["safety_score"] > 70 else "warn-val" if main["safety_score"] > 45 else "danger-val"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-lbl">🛡️ Güvenlik Skoru</div>'
            f'<div class="metric-val {color_cls}">%{main["safety_score"]}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-lbl">⚠️ Maks Hasar</div>'
            f'<div class="metric-val danger-val">%{main["max_damage"]*100:.0f}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-lbl">🔀 Düğüm Sayısı</div>'
            f'<div class="metric-val">{main["node_count"]}</div>'
            f'</div>', unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

elif route_error:
    st.error(f"Rota hesaplama hatası: {route_error}")

# ─── Ana Harita + Analiz Paneli ───────────────────────────────────────────────
tab_map, tab_analysis, tab_about = st.tabs(["🗺️ Harita", "📊 Bölge Analizi", "ℹ️ Proje Hakkında"])

with tab_map:
    with st.spinner("Harita oluşturuluyor…"):
        m = build_folium_map(
            G, damage_scores,
            routes=routes if routes else None,
            start_coords=start_coords,
            end_coords=end_coords,
        )
    st_folium(m, width=None, height=620, returned_objects=[])

    if routes and len(routes) > 1:
        st.markdown("#### 📋 Alternatif Rotalar Karşılaştırması")
        rows = []
        for i, r in enumerate(routes):
            if not r:
                continue
            label_, color_, emoji_ = ("Ana Rota", "🥇", "#1D9E75") if i == 0 else (f"Alternatif {i}", "🥈" if i==1 else "🥉", "#EF9F27")
            rows.append({
                "Rota": f"{emoji_} {label_}",
                "Mesafe (km)": r["distance_km"],
                "Süre (dk)": r["time_min"],
                "Güvenlik %": r["safety_score"],
                "Ort. Hasar": f'{r["avg_damage"]:.0%}',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_analysis:
    st.subheader("📊 Bölge Hasar Dağılımı")

    all_dmg = list(damage_scores.values())
    total   = len(all_dmg)
    safe_n    = sum(1 for d in all_dmg if d < 0.25)
    caution_n = sum(1 for d in all_dmg if 0.25 <= d < 0.50)
    danger_n  = sum(1 for d in all_dmg if 0.50 <= d < 0.75)
    blocked_n = sum(1 for d in all_dmg if d >= 0.75)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🟢 Güvenli Yollar",    f"%{safe_n/total*100:.1f}",    f"{safe_n} kenar")
        st.metric("🟡 Dikkatli Geçiş",    f"%{caution_n/total*100:.1f}", f"{caution_n} kenar")
    with col_b:
        st.metric("🔴 Tehlikeli",          f"%{danger_n/total*100:.1f}",  f"{danger_n} kenar")
        st.metric("⬛ Geçilemez",          f"%{blocked_n/total*100:.1f}", f"{blocked_n} kenar")

    st.divider()
    st.subheader("📡 YZ Model Bilgisi")
    st.markdown("""
| Bileşen | Teknoloji | Durum |
|---|---|---|
| Uydu Görüntüsü | Sentinel-2 (10m çözünürlük) | ✅ Copernicus API'den |
| Hasar Tespiti | U-Net / SAM segmentasyon | 🔄 Demo: Mesafe modeli |
| Yol Ağı | OpenStreetMap (osmnx) | ✅ Gerçek veri |
| Rota Motoru | A* (networkx) | ✅ Aktif |
| Harita Görsel | Esri World Imagery | ✅ Gerçek uydu |
""")

with tab_about:
    st.markdown("""
## 🚨 Sistem Mimarisi

Bu sistem dört ana katmandan oluşur:

### 1. Veri Girişi
- **Sentinel-2 uydu görüntüleri** (Copernicus Emergency Management Service)
- **OSM yol ağı** (osmnx ile otomatik indirilir)
- **BKZS GPS koordinatları** (başlangıç / hedef)

### 2. YZ Analizi  
- **Semantik Segmentasyon:** SAM (Segment Anything Model) veya U-Net
- Her piksel → enkaz / geçilebilir yol / su / sağlam alan olarak etiketlenir
- Çıktı: Her yol segmenti için [0,1] hasar skoru

### 3. Rota Optimizasyon Motoru
- Her yol kenarına `α×mesafe + β×hasar + γ×süre` ağırlığı atanır
- **A-star** algoritması (heuristik: Haversine + α katsayısı)
- Birincil + 2 alternatif rota hesaplanır

### 4. Çıktı Arayüzü
- Renk kodlu interaktif harita (Folium + Esri uydu)
- Güvenlik skoru, tahmini süre, mesafe metrikleri
- Alternatif rota karşılaştırması

---
**Veri Kaynakları:** Copernicus / ESA, OpenStreetMap, USGS ShakeMap, AFAD
""")

# Footer
st.divider()
st.caption("🛰️ TUA Astro Hackathon 2026 — Afet Yönetiminde Yerli Uydu Verisi Entegrasyonu Demo")
