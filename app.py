"""
app.py — Afet Kurtarma Rota Optimizasyon Sistemi (v4)
======================================================
Çalıştır: streamlit run app.py
"""

import os
import json
import pickle
import datetime
import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import numpy as np
import pandas as pd

from damage_model import compute_damage_score, damage_label, data_source_info
from router import assign_edge_weights, find_safest_route, find_alternative_routes

# ─── Sayfa ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Afet Kurtarma Rota Sistemi",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Konumlar ────────────────────────────────────────────────────────────────

RESCUE_CENTERS_LOCAL = {
    "🟦 AFAD KMaraş İl Müdürlüğü":          (37.5750, 36.9220),
    "🟦 AFAD Pazarcık Koordinasyon":          (37.4960, 37.2850),
    "🟦 AFAD Elbistan":                       (38.2060, 37.1960),
    "🔴 İtfaiye KMaraş Merkez":              (37.5820, 36.9350),
    "🔴 İtfaiye Pazarcık":                    (37.4940, 37.2800),
    "⭐ UMKE / 112 Acil KMaraş":             (37.5700, 36.9280),
}

# Şehre giriş koordinatları — dışarıdan gelen ekipler için
RESCUE_CENTERS_EXTERNAL = {
    "🚒 Adana AFAD  (D.400 / O-52 giriş)":   (37.4420, 36.6150),
    "🚒 Adana İtfaiye  (O-52 otoyol giriş)":  (37.4380, 36.6200),
    "🚒 Gaziantep AFAD  (E-90 giriş)":        (37.4650, 36.7800),
    "🚒 Gaziantep UMKE  (D.850 giriş)":       (37.4680, 36.7750),
    "🚒 Adıyaman AFAD  (Kuzey giriş)":        (37.7200, 37.0500),
    "🚒 Malatya AFAD  (D.795 giriş)":         (37.8500, 37.0800),
    "🚒 Osmaniye AFAD  (Güney giriş)":        (37.3100, 36.3200),
}

DEBRIS_ZONES = {
    "🔥 Pazarcık Merkez (Episantr)":          (37.4960, 37.2900),
    "🔥 Nurdağı İlçesi":                      (37.1830, 36.7310),
    "🔥 Türkoğlu İlçesi":                     (37.3820, 36.8610),
    "🔥 KMaraş Dulkadiroğlu":                 (37.6000, 36.9200),
    "🔥 KMaraş Onikişubat":                   (37.5630, 36.9230),
}

DESTINATIONS = {
    "🏥 KMaraş Eğitim Araştırma Hastanesi":   (37.5850, 36.9380),
    "🏥 Necip Fazıl Şehir Hastanesi":          (37.5980, 36.8930),
    "🏥 Pazarcık Devlet Hastanesi":            (37.4900, 37.2830),
    "⛺ AFAD Toplanma — Spor Kompleksi":       (37.5700, 36.9500),
    "⛺ AFAD Toplanma — İnönü Sahası":         (37.5550, 36.9300),
}

CACHE_DIR  = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "kahramanmaras_graph.pkl")

# ─── Graf & Hasar ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_graph():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    bbox = (37.45, 37.65, 36.82, 37.05)
    G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(G, f)
    return G


@st.cache_resource(show_spinner=False)
def compute_all_damage_scores(_G):
    scores = {}
    for u, v, key, data in _G.edges(keys=True, data=True):
        mid_lat = (_G.nodes[u]["y"] + _G.nodes[v]["y"]) / 2
        mid_lon = (_G.nodes[u]["x"] + _G.nodes[v]["x"]) / 2
        rt   = data.get("highway", "unclassified")
        rt   = rt[0] if isinstance(rt, list) else rt
        seed = abs(hash((u, v, key))) % (2 ** 31)
        scores[(u, v, key)] = compute_damage_score(mid_lat, mid_lon, rt, seed)
    return scores

# ─── Navigasyon URL ───────────────────────────────────────────────────────────

def gmaps_url(coords: list) -> str:
    """
    Sadece başlangıç ve hedef — waypoint YOK.
    Waypoint eklemek Google Maps'in kendi rotalamasını bozuyor:
    ara noktaları sırayla ziyaret etmeye çalışıp döngü yapıyor.
    Google Maps kendi algoritmasıyla en iyi rotayı bulsun.
    """
    if len(coords) < 2:
        return "#"
    org = f"{coords[0][0]},{coords[0][1]}"
    dst = f"{coords[-1][0]},{coords[-1][1]}"
    return (f"https://www.google.com/maps/dir/?api=1"
            f"&origin={org}&destination={dst}&travelmode=driving")


def yandex_url(coords: list) -> str:
    """Sadece başlangıç ve hedef — döngü sorunu yok."""
    if len(coords) < 2:
        return "#"
    org = f"{coords[0][1]},{coords[0][0]}"
    dst = f"{coords[-1][1]},{coords[-1][0]}"
    return f"https://maps.yandex.com/?rtext={org}~{dst}&rtt=auto"

# ─── Çevrimdışı Kart ─────────────────────────────────────────────────────────

def offline_card(phase_label, from_name, to_name, coords, stats, n=10):
    """
    Saha ekibi için sade JSON kart.
    Tüm koordinatlar yerine sadece n kritik kavşak noktası.
    """
    step       = max(1, len(coords) // n)
    key_pts    = coords[::step]
    if coords[-1] not in key_pts:
        key_pts.append(coords[-1])

    return json.dumps({
        "sistem":      "Afet Kurtarma Rota Sistemi — TUA Hackathon 2026",
        "olusturuldu": datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
        "faz":         phase_label,
        "baslangic":   from_name,
        "hedef":       to_name,
        "ozet": {
            "mesafe_km":          stats.get("distance_km"),
            "sure_dakika":        round(stats.get("time_min", 0)),
            "guvenlik_yuzdesi":   stats.get("safety_score"),
        },
        "kritik_kavsaklar": [
            {"sira": i + 1,
             "lat":  round(la, 5),
             "lon":  round(lo, 5),
             "maps": f"https://maps.google.com/?q={round(la,5)},{round(lo,5)}"}
            for i, (la, lo) in enumerate(key_pts)
        ],
        "kullanim": (
            "İnternet yoksa her noktanın 'maps' linkini tarayıcıya yazın "
            "veya lat/lon koordinatını GPS cihazına girin."
        ),
    }, ensure_ascii=False, indent=2)

# ─── Harita ──────────────────────────────────────────────────────────────────

def build_map(G, damage_scores, routes=None,
              start_coords=None, mid_coords=None, end_coords=None,
              route2=None, external_start=False, debris_label="Enkaz Noktası"):

    lats   = [c[0] for c in [start_coords, mid_coords, end_coords] if c]
    lons   = [c[1] for c in [start_coords, mid_coords, end_coords] if c]
    center = [np.mean(lats) if lats else 37.56,
              np.mean(lons) if lons else 36.94]

    m = folium.Map(location=center, zoom_start=11, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="🛰️ Uydu", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="🗺️ Karanlık",
                     overlay=False, control=True).add_to(m)

    # Yol ağı renklendirme
    edges   = list(G.edges(keys=True, data=True))
    rng     = np.random.RandomState(42)
    sampled = rng.choice(len(edges), min(6000, len(edges)), replace=False)
    grps    = {
        "safe":    folium.FeatureGroup(name="🟢 Güvenli",   show=True),
        "caution": folium.FeatureGroup(name="🟡 Dikkatli",  show=True),
        "danger":  folium.FeatureGroup(name="🔴 Tehlikeli", show=True),
        "blocked": folium.FeatureGroup(name="⬛ Geçilemez", show=True),
    }
    for idx in sampled:
        u, v, key, data = edges[idx]
        dmg = damage_scores.get((u, v, key), 0.3)
        lbl, color, _   = damage_label(dmg)
        gk = ("safe" if dmg < 0.25 else "caution" if dmg < 0.50
              else "danger" if dmg < 0.75 else "blocked")
        try:
            c = [(G.nodes[u]["y"], G.nodes[u]["x"]),
                 (G.nodes[v]["y"], G.nodes[v]["x"])]
            folium.PolyLine(c, color=color,
                            weight=1.5 if dmg < 0.5 else 2.5,
                            opacity=0.45 if dmg < 0.5 else 0.75,
                            tooltip=f"{lbl} — {dmg:.0%}").add_to(grps[gk])
        except Exception:
            pass
    for g in grps.values():
        g.add_to(m)

    def draw_route(route_obj, color, tip_prefix, weight=6):
        if not route_obj or not route_obj.get("path"):
            return
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"])
                  for n in route_obj["path"] if n in G.nodes]
        if len(coords) < 2:
            return
        tip = (f"{tip_prefix} — {route_obj['distance_km']} km  "
               f"{route_obj['time_min']:.0f} dk  %{route_obj['safety_score']}")
        folium.PolyLine(coords, color="#000", weight=weight + 4, opacity=0.4).add_to(m)
        folium.PolyLine(coords, color=color, weight=weight,
                        opacity=0.95, tooltip=tip).add_to(m)

    route_colors = ["#FFFFFF", "#60A5FA", "#34D399"]
    if routes:
        for i, r in enumerate(routes):
            draw_route(r, route_colors[min(i, 2)],
                       "🥇 En Güvenli" if i == 0 else f"🥈 Alternatif {i}",
                       weight=6 if i == 0 else 4)
    if route2:
        draw_route(route2, "#FBBF24", "🏥 Faz 2", weight=6)

    # İşaretçiler
    if start_coords:
        ic = "purple" if external_start else "blue"
        ii = "plane"  if external_start else "home"
        pp = "<b>✈️ Dışarıdan Gelen Ekip — Şehir Girişi</b>" if external_start else "<b>🏛️ Kurtarma Merkezi</b>"
        folium.Marker(start_coords, popup=folium.Popup(pp, max_width=240),
                      icon=folium.Icon(color=ic, icon=ii, prefix="fa")).add_to(m)
    if mid_coords:
        folium.Marker(mid_coords,
                      popup=folium.Popup(f"<b>🔥 Enkaz Noktası</b><br><small>{debris_label[:60]}</small>", max_width=280),
                      icon=folium.Icon(color="red", icon="fire", prefix="fa")).add_to(m)
    if end_coords:
        folium.Marker(end_coords,
                      popup=folium.Popup("<b>🏥 Hedef</b>", max_width=220),
                      icon=folium.Icon(color="green", icon="plus-square", prefix="fa")).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.metric-card{background:linear-gradient(135deg,#1E2A3A,#243447);border:1px solid #2D4057;
border-radius:12px;padding:14px 18px;text-align:center;margin-bottom:8px;}
.metric-lbl{font-size:11px;color:#8BA0B8;letter-spacing:.06em;text-transform:uppercase;}
.metric-val{font-size:26px;font-weight:700;margin-top:4px;}
.safe-val{color:#1D9E75;} .warn-val{color:#EF9F27;} .danger-val{color:#E24B4A;}
.cop-badge{display:inline-block;padding:3px 9px;border-radius:20px;
background:#0E3A5E;color:#60C6FF;font-size:12px;font-weight:600;
border:1px solid #1E6FA8;margin:2px;}
.nav-box{background:#1A2332;border:1px solid #2D4057;border-radius:10px;
padding:14px 16px;margin-bottom:10px;}
.nav-box-title{font-size:13px;font-weight:700;color:#CBD5E1;margin-bottom:10px;line-height:1.5;}
.nav-link{display:inline-block;padding:7px 16px;border-radius:7px;
font-weight:600;font-size:13px;text-decoration:none !important;margin-right:6px;margin-bottom:4px;}
.nav-g{background:#1A73E8;color:#fff !important;}
.nav-y{background:#CC1F1F;color:#fff !important;}

/* Streamlit spinner özelleştirme */
div[data-testid="stSpinner"] > div {
    border-color: #00B4D8 !important;
}
/* st.info kutusu */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}
/* İlerleme mesajı */
.loading-step {
    background: #0D3B6E;
    border-left: 4px solid #00B4D8;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
    color: #E8F4FD;
}
</style>
""", unsafe_allow_html=True)

# ─── Başlık ───────────────────────────────────────────────────────────────────

col_t, col_b = st.columns([5, 1])
with col_t:
    st.markdown("# 🚨 Afet Kurtarma Rota Optimizasyon Sistemi")
    st.caption("**Kahramanmaraş 6 Şubat 2023 Depremi** — Copernicus EMS + OSM tabanlı en güvenli kurtarma rotası")
with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🟢 CANLI DEMO")

src = data_source_info()
if src["points"] > 0:
    aoi_tags = " ".join(f'<span class="cop-badge">{a}</span>' for a in src["aois"])
    st.markdown(
        f'🛰️ **Copernicus EMS aktif** — **{src["points"]:,}** uydu analizi noktası {aoi_tags}',
        unsafe_allow_html=True)
else:
    st.warning("⚠️ Copernicus verisi yok — GMPE analitik model aktif")

st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Operasyon Parametreleri")

    team_type   = st.radio("👥 Ekip Tipi",
                           ["🏙️ İl İçi Ekip", "✈️ İl Dışından Gelen Ekip"])
    is_external = "Dışından" in team_type
    all_centers = RESCUE_CENTERS_EXTERNAL if is_external else RESCUE_CENTERS_LOCAL

    if is_external:
        st.caption("ℹ️ Koordinat = şehre giriş noktası. Sistem oradan en güvenli rotayı hesaplar.")

    start_name = st.selectbox("🏛️ Başlangıç Noktası", list(all_centers.keys()))

    st.divider()
    op_mode   = st.radio("🚒 Operasyon Tipi",
                         ["1️⃣  Tek Aşamalı  (Merkez → Hedef)",
                          "2️⃣  İki Aşamalı  (Merkez → Enkaz → Hastane)"])
    two_phase = "İki Aşamalı" in op_mode

    if two_phase:
        st.markdown("**🔥 Enkaz / Afet Noktası**")
        debris_mode = st.radio(
            "Konum giriş yöntemi",
            ["📋 Hazır bölge listesi",
             "🔍 Adres / yer adı ara",
             "📌 Koordinat gir (Lat / Lon)"],
            label_visibility="collapsed",
        )

        debris_coords_sidebar = None  # sidebar'da çözülen koordinat
        debris_label          = ""    # haritada gösterilecek etiket
        geocode_error         = None

        if debris_mode == "📋 Hazır bölge listesi":
            debris_name           = st.selectbox("Bölge", list(DEBRIS_ZONES.keys()),
                                                 label_visibility="collapsed")
            debris_coords_sidebar = DEBRIS_ZONES[debris_name]
            debris_label          = debris_name

        elif debris_mode == "🔍 Adres / yer adı ara":
            # ── Bilinen 2023 deprem enkaz alanları ────────────────────────────
            ENKAZ_NOKTALARI = {
                # Pazarcık
                "Pazarcık — Sofular Mah. (Ağır Hasar Bölgesi)":   (37.4995, 37.2812),
                "Pazarcık — Bahçelievler Mah.":                    (37.4978, 37.2756),
                "Pazarcık — Cumhuriyet Mah. Merkez":               (37.4961, 37.2901),
                "Pazarcık — Şekerli Mah.":                         (37.5043, 37.2834),
                "Pazarcık — Hürriyet Mah.":                        (37.4932, 37.2789),
                # Nurdağı
                "Nurdağı — Merkez (Tamamen Yıkılan Alan)":         (37.1822, 36.7312),
                "Nurdağı — Atatürk Cad. Çevresi":                  (37.1835, 36.7298),
                # KMaraş - Dulkadiroğlu
                "KMaraş Dulkadiroğlu — Trabzon Mah.":              (37.6012, 36.9187),
                "KMaraş Dulkadiroğlu — Yavuzlar Mah.":             (37.5987, 36.9243),
                "KMaraş Dulkadiroğlu — Güneş Mah.":                (37.6034, 36.9156),
                # KMaraş - Onikişubat
                "KMaraş Onikişubat — Yenipınar Mah.":              (37.5621, 36.9178),
                "KMaraş Onikişubat — Serintepe Mah.":              (37.5589, 36.9312),
                "KMaraş Onikişubat — Yıldız Mah.":                 (37.5643, 36.9089),
                # Türkoğlu
                "Türkoğlu — Merkez":                               (37.3829, 36.8594),
                # Elbistan
                "Elbistan — Cumhuriyet Mah.":                      (38.2083, 37.1934),
                "Elbistan — Bahçelievler Mah.":                    (38.2051, 37.1978),
            }

            st.caption("Bilinen enkaz alanı seçin veya aşağıda arayın:")
            secili_enkaz = st.selectbox(
                "Bilinen Enkaz Alanları",
                ["— Listeden seç —"] + list(ENKAZ_NOKTALARI.keys()),
                label_visibility="collapsed",
            )
            if secili_enkaz != "— Listeden seç —":
                debris_coords_sidebar = ENKAZ_NOKTALARI[secili_enkaz]
                debris_label          = secili_enkaz
                debris_name           = secili_enkaz
                st.success(f"✅ {debris_coords_sidebar[0]:.5f}, {debris_coords_sidebar[1]:.5f}")
            else:
                st.caption("veya serbest arama:")
                adres_input = st.text_input(
                    "Mahalle / cadde / bina adı",
                    placeholder="ör: Sofular Mah, Pazarcık",
                    label_visibility="collapsed",
                )

                if adres_input:
                    import re
                    # Bina/daire numarasını at, kalan kısmı kullan
                    temiz = re.sub(
                        r"\s*(no|no\.|kat|daire|d\.)\s*\d+.*$",
                        "", adres_input, flags=re.IGNORECASE
                    ).strip()

                    # Türkçe kısaltma varyantları + il ekleri
                    varyantlar = []
                    for v in [temiz, adres_input]:
                        varyantlar += [
                            v + ", Pazarcık, Kahramanmaraş, Türkiye",
                            v + ", Kahramanmaraş, Türkiye",
                            v + ", Türkiye",
                            v,
                        ]
                    # "Mah." / "Mahallesi" çift yönlü dene
                    if "mahallesi" in temiz.lower():
                        k = re.sub(r"mahallesi", "Mah.", temiz, flags=re.IGNORECASE)
                        varyantlar.insert(0, k + ", Kahramanmaraş, Türkiye")
                    elif "mah." in temiz.lower():
                        k = re.sub(r"mah\.", "Mahallesi", temiz, flags=re.IGNORECASE)
                        varyantlar.insert(0, k + ", Kahramanmaraş, Türkiye")
                    # "Cad." / "Caddesi" çift yönlü dene
                    if "caddesi" in temiz.lower():
                        k = re.sub(r"caddesi", "Cad.", temiz, flags=re.IGNORECASE)
                        varyantlar.insert(0, k + ", Kahramanmaraş, Türkiye")
                    elif "cad." in temiz.lower():
                        k = re.sub(r"cad\.", "Caddesi", temiz, flags=re.IGNORECASE)
                        varyantlar.insert(0, k + ", Kahramanmaraş, Türkiye")

                    lat, lon, bulunan = None, None, None
                    for sorgu in varyantlar:
                        try:
                            lat, lon = ox.geocode(sorgu)
                            bulunan  = sorgu
                            break
                        except Exception:
                            continue

                    if lat is not None:
                        debris_coords_sidebar = (lat, lon)
                        debris_label          = adres_input
                        debris_name           = adres_input
                        st.success(f"✅ Bulundu — {lat:.5f}, {lon:.5f}")
                        if bulunan and bulunan.split(",")[0].strip().lower() != adres_input.lower():
                            st.caption(f"OSM eşleşmesi: *{bulunan}*")
                    else:
                        st.warning(
                            "OSM'de bulunamadı. "
                            "Yukarıdaki **bilinen enkaz listesini** kullanın "
                            "veya **📌 Koordinat gir** sekmesine geçin."
                        )
                debris_name = adres_input or ""
                if not debris_coords_sidebar and not adres_input:
                    debris_name = ""


        else:  # Koordinat gir
            # Google Maps'ten koordinat alma talimatı
            st.markdown("""
<div style='background:#0F2337;border:1px solid #1E4A6E;border-radius:8px;
padding:10px 12px;margin-bottom:10px;font-size:12px;color:#94A3B8;line-height:1.7'>
<b style='color:#60C6FF'>📱 Google Maps'ten koordinat alma:</b><br>
1. Google Maps'i açın<br>
2. Enkaz noktasına <b>uzun basın</b> (mobil) veya <b>sağ tıklayın</b> (masaüstü)<br>
3. Üstte çıkan <b>37.XXXXX, 37.XXXXX</b> sayılarına tıklayın → kopyalanır<br>
4. Aşağıya yapıştırın
</div>
""", unsafe_allow_html=True)
            col_lat, col_lon = st.columns(2)
            with col_lat:
                lat_in = st.number_input("Enlem (Lat)", value=37.49600,
                                         format="%.5f", step=0.00001)
            with col_lon:
                lon_in = st.number_input("Boylam (Lon)", value=37.29000,
                                         format="%.5f", step=0.00001)

            # Yapıştırılan koordinat string'i de parse et
            koord_str = st.text_input(
                "veya direkt yapıştır",
                placeholder="37.49600, 37.29000",
                label_visibility="visible",
            )
            if koord_str:
                try:
                    parts = koord_str.replace(" ", "").split(",")
                    lat_in = float(parts[0])
                    lon_in = float(parts[1])
                    st.success(f"✅ Koordinat alındı: {lat_in:.5f}, {lon_in:.5f}")
                except Exception:
                    st.warning("Format: 37.49600, 37.29000")

            debris_coords_sidebar = (lat_in, lon_in)
            debris_label          = f"📌 {lat_in:.5f}, {lon_in:.5f}"
            debris_name           = debris_label

        if debris_coords_sidebar:
            st.caption(f"📍 Seçilen nokta: {debris_coords_sidebar[0]:.5f}, {debris_coords_sidebar[1]:.5f}")

        end_name = st.selectbox("🏥 Hedef — Hastane / Toplanma", list(DESTINATIONS.keys()))
    else:
        end_name              = st.selectbox("🎯 Hedef", list(DESTINATIONS.keys()))
        debris_name           = None
        debris_coords_sidebar = None
        debris_label          = ""

    st.divider()

    # ── Operasyon Profili (ön ayar) ──────────────────────────────────────────
    st.subheader("⚖️ Rota Optimizasyon Ağırlıkları")
    st.caption(
        "Sistem her yolu üç kritere göre puanlar: mesafe, hasar ve geçiş süresi. "
        "Bu kaydırıcılar hangisinin daha önemli olduğunu belirler."
    )

    profil = st.radio(
        "Hızlı Profil Seç",
        ["🛡️ Güvenlik Öncelikli",
         "⚡ Hız Öncelikli",
         "📐 Dengeli",
         "✏️ Manuel Ayar"],
        index=0,
    )

    if profil == "🛡️ Güvenlik Öncelikli":
        alpha_def, beta_def = 0.20, 0.65
        profil_aciklama = "Hasar görmüş yollardan maksimum kaçınır. Rota daha uzun ama çok daha güvenli olabilir."
    elif profil == "⚡ Hız Öncelikli":
        alpha_def, beta_def = 0.20, 0.20
        profil_aciklama = "En kısa sürede hedefe ulaşmayı hedefler. Kısmen hasarlı yollardan da geçebilir."
    elif profil == "📐 Dengeli":
        alpha_def, beta_def = 0.30, 0.50
        profil_aciklama = "Varsayılan ayar. Güvenlik ve hız dengelenir. Çoğu operasyon için uygundur."
    else:
        alpha_def, beta_def = 0.30, 0.50
        profil_aciklama = "Kaydırıcıları kendiniz ayarlayın."

    st.caption(f"ℹ️ {profil_aciklama}")
    st.markdown("")

    # ── Kaydırıcılar ─────────────────────────────────────────────────────────
    disabled = (profil != "✏️ Manuel Ayar")

    alpha = st.slider(
        "📏 Mesafe Ağırlığı (α)",
        min_value=0.10, max_value=0.80,
        value=alpha_def, step=0.05,
        disabled=disabled,
        help=(
            "Bu değer yükseldikçe sistem daha kısa yolları tercih eder. "
            "Düşük tutarsanız, mesafe uzasa bile daha güvenli rota seçilir."
        ),
    )
    beta = st.slider(
        "⚠️ Hasar Ağırlığı (β)",
        min_value=0.10, max_value=0.80,
        value=beta_def, step=0.05,
        disabled=disabled,
        help=(
            "En kritik parametre. Yükseldikçe sistem hasarlı yollardan kaçar. "
            "Deprem bölgelerinde yüksek tutulması önerilir (0.50+)."
        ),
    )
    gamma_val = round(max(0.0, 1.0 - alpha - beta), 2)

    # ── Görsel özet ──────────────────────────────────────────────────────────
    toplam = alpha + beta + gamma_val
    pct_a  = int(alpha    / toplam * 100)
    pct_b  = int(beta     / toplam * 100)
    pct_g  = 100 - pct_a - pct_b

    st.markdown(
        f"""
<div style='background:#0F1E2E;border:1px solid #1E3A5F;border-radius:10px;
padding:12px 14px;margin-top:6px;font-size:12px;'>
<div style='margin-bottom:6px;color:#94A3B8;font-weight:600;letter-spacing:.05em'>
ROTALAMADAKİ ETKİ PAYI</div>
<div style='display:flex;height:14px;border-radius:6px;overflow:hidden;margin-bottom:8px'>
  <div style='width:{pct_a}%;background:#1D9E75'></div>
  <div style='width:{pct_b}%;background:#E24B4A'></div>
  <div style='width:{pct_g}%;background:#EF9F27'></div>
</div>
<div style='display:flex;justify-content:space-between;color:#CBD5E1'>
  <span>📏 Mesafe &nbsp;<b style='color:#1D9E75'>%{pct_a}</b></span>
  <span>⚠️ Hasar &nbsp;<b style='color:#E24B4A'>%{pct_b}</b></span>
  <span>⏱️ Süre &nbsp;<b style='color:#EF9F27'>%{pct_g}</b></span>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Geçersiz toplam uyarısı
    if alpha + beta > 0.95:
        st.warning("⚠️ α + β toplamı 1'i aşıyor. Süre ağırlığı (γ) sıfırlandı.")

    st.markdown("")
    with st.expander("📖 Formül Nedir?"):
        st.markdown(f"""
Her yol segmenti için bir **maliyet skoru** hesaplanır:

```
maliyet = α × mesafe_km
        + β × hasar_skoru × 10
        + γ × geçiş_süresi_dk
```

**Şu anki değerler:**  
`α = {alpha}` · `β = {beta}` · `γ = {gamma_val}`

Sistem en düşük maliyetli rotayı seçer (A* algoritması).  
Hasar skoru 0 (güvenli) ile 1 (geçilemez) arasındadır.  
Hasarlı yollarda hız otomatik düşer:  
- Hasar < %25 → 50 km/h  
- Hasar %25–50 → 25 km/h  
- Hasar %50–75 → 8 km/h  
- Hasar > %75 → 1.5 km/h (neredeyse geçilemez)
""")

    st.divider()
    alt_routes = st.checkbox("Alternatif rotalar göster (tek aşamalı)", value=True)
    st.divider()
    find_btn = st.button("🗺️  Rotayı Hesapla", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📡 Hasar Göstergesi")
    for line in ["🟢 **< %25** — Güvenli", "🟡 **%25–50** — Dikkatli",
                 "🔴 **%50–75** — Tehlikeli", "⬛ **> %75** — Geçilemez"]:
        st.markdown(line)

# ─── Veri Yükleme ────────────────────────────────────────────────────────────

# İlk açılışta mı yoksa cache'den mi yükleniyor?
import os as _os
_first_load = not _os.path.exists(CACHE_FILE)

if _first_load:
    st.info(
        "⏳ **İlk açılış — sistem hazırlanıyor**  \n"
        "Kahramanmaraş yol ağı internet üzerinden indiriliyor. "
        "Bu işlem **yaklaşık 1-2 dakika** sürer ve yalnızca bir kez yapılır. "
        "Sonraki açılışlarda sistem anında başlar.",
        icon="🛰️"
    )
    yol_mesaj  = "🛰️ Yol ağı indiriliyor… (1-2 dk, sadece ilk açılışta)"
    hasar_mesaj = "🤖 Hasar skorları hesaplanıyor… (lütfen bekleyin)"
else:
    yol_mesaj  = "🛰️ Yol ağı önbellekten yükleniyor…"
    hasar_mesaj = "🤖 Hasar skorları önbellekten yükleniyor…"

with st.spinner(yol_mesaj):
    G = load_graph()

if _first_load:
    st.success("✅ Yol ağı indirildi — bir sonraki açılış anında olacak.")

with st.spinner(hasar_mesaj):
    damage_scores = compute_all_damage_scores(G)

G_weighted    = assign_edge_weights(G.copy(), damage_scores, alpha, beta, gamma_val)
start_coords  = all_centers[start_name]
end_coords    = DESTINATIONS[end_name]
debris_coords = debris_coords_sidebar if two_phase else None

# ─── Rota Hesaplama ──────────────────────────────────────────────────────────

routes        = []
route2_main   = None
route_error   = None
phase1_coords = []
phase2_coords = []

if find_btn:
    with st.spinner("🧭 En güvenli rota hesaplanıyor… (A* algoritması)"):
        try:
            if two_phase:
                sn = ox.nearest_nodes(G_weighted, start_coords[1],  start_coords[0])
                mn = ox.nearest_nodes(G_weighted, debris_coords[1], debris_coords[0])
                en = ox.nearest_nodes(G_weighted, end_coords[1],    end_coords[0])
                r1 = find_safest_route(G_weighted, sn, mn)
                r2 = find_safest_route(G_weighted, mn, en)
                if r1:
                    routes = [r1]
                    phase1_coords = [(G.nodes[n]["y"], G.nodes[n]["x"])
                                     for n in r1["path"] if n in G.nodes]
                route2_main = r2
                if r2 and r2.get("path"):
                    phase2_coords = [(G.nodes[n]["y"], G.nodes[n]["x"])
                                     for n in r2["path"] if n in G.nodes]
            else:
                sn = ox.nearest_nodes(G_weighted, start_coords[1], start_coords[0])
                en = ox.nearest_nodes(G_weighted, end_coords[1],   end_coords[0])
                if alt_routes:
                    routes = find_alternative_routes(G_weighted, sn, en, k=3)
                else:
                    r = find_safest_route(G_weighted, sn, en)
                    routes = [r] if r else []
                if routes and routes[0]:
                    phase1_coords = [(G.nodes[n]["y"], G.nodes[n]["x"])
                                     for n in routes[0]["path"] if n in G.nodes]
        except Exception as e:
            route_error = str(e)

# ─── Metrik Kartları ─────────────────────────────────────────────────────────

if routes and routes[0]:
    main  = routes[0]
    main2 = route2_main

    if two_phase and main2:
        total_km  = main["distance_km"] + main2["distance_km"]
        total_min = main["time_min"]    + main2["time_min"]
        avg_safe  = round((main["safety_score"] + main2["safety_score"]) / 2, 1)
    else:
        total_km  = main["distance_km"]
        total_min = main["time_min"]
        avg_safe  = main["safety_score"]

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, cls in [
        (c1, "📏 Toplam Mesafe",  f"{total_km} km",
         "safe-val"),
        (c2, "⏱️ Tahmini Süre",  f"{total_min:.0f} dk",
         "warn-val"),
        (c3, "🛡️ Güvenlik",      f"%{avg_safe}",
         "safe-val" if avg_safe > 70 else "warn-val" if avg_safe > 45 else "danger-val"),
        (c4, "⚠️ Maks Hasar",    f"%{main['max_damage']*100:.0f}",
         "danger-val"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-lbl">{lbl}</div>'
                f'<div class="metric-val {cls}">{val}</div></div>',
                unsafe_allow_html=True)

    # ── Navigasyon — Faz bazlı ayrı kutular ──────────────────────────────────
    st.markdown("<br>**📱 Navigasyon**", unsafe_allow_html=False)

    if two_phase:
        nav1, nav2 = st.columns(2)

        with nav1:
            # Faz 1: Merkez → Enkaz
            g1  = gmaps_url(phase1_coords)
            y1  = yandex_url(phase1_coords)
            s1  = start_name.split("(")[0].strip()[-28:]
            d1  = (debris_name or "Enkaz").replace("🔥 Enkaz Bölgesi — ", "")[:28]
            st.markdown(
                f'<div class="nav-box">'
                f'<div class="nav-box-title">🚒 FAZ 1 — Kurtarma Merkezi → Enkaz<br>'
                f'<small style="font-weight:400;color:#94A3B8">{s1} → {d1}</small></div>'
                f'<a class="nav-link nav-g" href="{g1}" target="_blank">📍 Google Maps</a>'
                f'<a class="nav-link nav-y" href="{y1}" target="_blank">🗺️ Yandex Maps</a>'
                f'</div>', unsafe_allow_html=True)
            if phase1_coords:
                st.download_button(
                    "💾 Faz 1 — Çevrimdışı Kart",
                    data=offline_card("Faz 1 — Merkez → Enkaz",
                                      start_name, debris_name or "Enkaz",
                                      phase1_coords,
                                      {"distance_km": main["distance_km"],
                                       "time_min": main["time_min"],
                                       "safety_score": main["safety_score"]}),
                    file_name="faz1_enkaz.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with nav2:
            # Faz 2: Enkaz → Hastane
            if phase2_coords:
                g2  = gmaps_url(phase2_coords)
                y2  = yandex_url(phase2_coords)
                d2  = (debris_name or "Enkaz").replace("🔥 Enkaz Bölgesi — ", "")[:28]
                h2  = end_name.replace("🏥 ", "").replace("⛺ ", "")[:28]
                st.markdown(
                    f'<div class="nav-box">'
                    f'<div class="nav-box-title">🏥 FAZ 2 — Enkaz → Hastane<br>'
                    f'<small style="font-weight:400;color:#94A3B8">{d2} → {h2}</small></div>'
                    f'<a class="nav-link nav-g" href="{g2}" target="_blank">📍 Google Maps</a>'
                    f'<a class="nav-link nav-y" href="{y2}" target="_blank">🗺️ Yandex Maps</a>'
                    f'</div>', unsafe_allow_html=True)
                st.download_button(
                    "💾 Faz 2 — Çevrimdışı Kart",
                    data=offline_card("Faz 2 — Enkaz → Hastane",
                                      debris_name or "Enkaz", end_name,
                                      phase2_coords,
                                      {"distance_km": main2["distance_km"],
                                       "time_min": main2["time_min"],
                                       "safety_score": main2["safety_score"]}),
                    file_name="faz2_hastane.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("Faz 2 rotası hesaplanamadı.")
    else:
        # Tek aşamalı
        if phase1_coords:
            g1 = gmaps_url(phase1_coords)
            y1 = yandex_url(phase1_coords)
            s1 = start_name.split("(")[0].strip()[-28:]
            h1 = end_name.replace("🏥 ", "").replace("⛺ ", "")[:28]
            st.markdown(
                f'<div class="nav-box">'
                f'<div class="nav-box-title">🗺️ {s1} → {h1}</div>'
                f'<a class="nav-link nav-g" href="{g1}" target="_blank">📍 Google Maps</a>'
                f'<a class="nav-link nav-y" href="{y1}" target="_blank">🗺️ Yandex Maps</a>'
                f'</div>', unsafe_allow_html=True)
            st.download_button(
                "💾 Çevrimdışı Kart (JSON)",
                data=offline_card("Tek Aşamalı Rota", start_name, end_name,
                                  phase1_coords,
                                  {"distance_km": main["distance_km"],
                                   "time_min": main["time_min"],
                                   "safety_score": main["safety_score"]}),
                file_name="emergency_route.json",
                mime="application/json",
            )

    st.markdown("<br>", unsafe_allow_html=True)

elif route_error:
    st.error(f"Rota hesaplama hatası: {route_error}")

# ─── Sekmeler ────────────────────────────────────────────────────────────────

tab_map, tab_analysis, tab_copernicus, tab_about = st.tabs(
    ["🗺️ Harita", "📊 Bölge Analizi", "🛰️ Copernicus Verisi", "ℹ️ Sistem Mimarisi"])

with tab_map:
    if not routes:
        st.info(
            "👈 Sol panelden başlangıç noktası, enkaz bölgesi ve hedefi seçip "
            "**🗺️ Rotayı Hesapla** butonuna basın.",
            icon="ℹ️"
        )
    with st.spinner("🗺️ Harita oluşturuluyor… hasar renklendirmesi uygulanıyor"):
        m = build_map(G, damage_scores,
                      routes=routes if routes else None,
                      start_coords=start_coords,
                      mid_coords=debris_coords,
                      end_coords=end_coords,
                      route2=route2_main,
                      external_start=is_external,
                      debris_label=debris_label)
    st_folium(m, width=None, height=640, returned_objects=[])

    if two_phase and routes and route2_main:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("#### 🚒 Faz 1 — Merkez → Enkaz")
            r = routes[0]
            st.markdown(f"Mesafe: **{r['distance_km']} km** &nbsp;|&nbsp; "
                        f"Süre: **{r['time_min']:.0f} dk** &nbsp;|&nbsp; "
                        f"Güvenlik: **%{r['safety_score']}**")
        with cb:
            st.markdown("#### 🏥 Faz 2 — Enkaz → Hastane")
            st.markdown(f"Mesafe: **{route2_main['distance_km']} km** &nbsp;|&nbsp; "
                        f"Süre: **{route2_main['time_min']:.0f} dk** &nbsp;|&nbsp; "
                        f"Güvenlik: **%{route2_main['safety_score']}**")

    if not two_phase and routes and len(routes) > 1:
        st.markdown("#### 📋 Alternatif Rotalar")
        rows = []
        for i, r in enumerate(routes):
            if not r:
                continue
            e = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
            rows.append({"Rota": f"{e} {'Ana Rota' if i==0 else f'Alternatif {i}'}",
                         "Mesafe (km)": r["distance_km"],
                         "Süre (dk)": r["time_min"],
                         "Güvenlik %": r["safety_score"],
                         "Ort. Hasar": f'{r["avg_damage"]:.0%}'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_analysis:
    st.subheader("📊 Bölge Hasar Dağılımı")
    all_dmg = list(damage_scores.values())
    total   = len(all_dmg)
    cnts    = {"safe":    sum(1 for d in all_dmg if d < 0.25),
               "caution": sum(1 for d in all_dmg if 0.25 <= d < 0.50),
               "danger":  sum(1 for d in all_dmg if 0.50 <= d < 0.75),
               "blocked": sum(1 for d in all_dmg if d >= 0.75)}
    ca2, cb2 = st.columns(2)
    with ca2:
        st.metric("🟢 Güvenli",  f"%{cnts['safe']/total*100:.1f}",    f"{cnts['safe']} kenar")
        st.metric("🟡 Dikkatli", f"%{cnts['caution']/total*100:.1f}", f"{cnts['caution']} kenar")
    with cb2:
        st.metric("🔴 Tehlikeli",f"%{cnts['danger']/total*100:.1f}",  f"{cnts['danger']} kenar")
        st.metric("⬛ Geçilemez",f"%{cnts['blocked']/total*100:.1f}", f"{cnts['blocked']} kenar")
    st.divider()
    st.markdown("""
| Bileşen | Teknoloji | Durum |
|---|---|---|
| Uydu Verisi | Copernicus EMS EMSR648 | ✅ 7.569 nokta |
| Hasar Modeli | Copernicus + GMPE fallback | ✅ Aktif |
| Yol Ağı | OpenStreetMap (osmnx) | ✅ Gerçek veri |
| Rota Motoru | A* + Dijkstra (networkx) | ✅ Aktif |
| Harita | Esri World Imagery | ✅ Uydu görüntüsü |
| Gerçek Sistemde | SAM / U-Net (Sentinel-2) | 🔄 Hazır mimari |
""")

with tab_copernicus:
    st.subheader("🛰️ Copernicus EMS — EMSR648")
    st.markdown("**Aktivasyon:** Kahramanmaraş Earthquake — 6 Şubat 2023  \n**Kaynak:** European Emergency Management Service")
    for aoi, (bolge, aciklama, n) in {
        "AOI08": ("Pazarcık",        "Episantr bölgesi",          66),
        "AOI04": ("Kahramanmaraş",   "Şehir merkezi",           7182),
        "AOI16": ("Nurdağı",         "İkinci episantr bölgesi",  321),
    }.items():
        with st.expander(f"📍 {aoi} — {bolge}  ({n:,} nokta)"):
            st.markdown(f"**Kapsam:** {aciklama}  |  **Nokta:** {n:,}")
            st.markdown("**Sınıflar:** No visible damage / Possibly damaged / Damaged / Destroyed")
    st.divider()
    st.markdown("""
### Gerçek Sistemde
1. Deprem anında Copernicus Emergency API tetiklenir
2. Sentinel-2 önce/sonra görüntüsü çekilir
3. SAM segmentasyonuyla hasar haritası üretilir
4. Harita rota motoruna beslenir — mevcut demo mimarisiyle birebir uyumlu
""")

with tab_about:
    st.markdown("""
## 🚨 Sistem Mimarisi — 4 Katman

### Katman 1 — Veri
- Sentinel-2 / Copernicus EMS uydu verisi
- OSM yol ağı (osmnx)
- AFAD / İtfaiye / **İl dışı** ekip GPS koordinatları + şehir giriş noktaları

### Katman 2 — YZ Hasar Analizi
- **Demo:** Copernicus EMS doğrulanmış 7.569 hasar noktası
- **Fallback:** GMPE (episantr mesafesi × yol kırılganlığı)
- **Gerçek sistem:** SAM / U-Net Sentinel-2 segmentasyonu

### Katman 3 — Rota Motoru
```
w(e) = α × mesafe_km + β × hasar × 10 + γ × süre_dk
Varsayılan: α=0.30  β=0.50  γ=0.20
```
- A* (Haversine heuristik) + Dijkstra fallback
- **Tek** veya **İki aşamalı** operasyon modu

### Katman 4 — Çıktı
- Renk kodlu harita (Esri uydu)
- **Faz bazlı navigasyon:** Faz 1 ve Faz 2 ayrı Google/Yandex linkleri
- **Çevrimdışı kart:** 10 kritik kavşak + özet — internet kesilirse USB/bluetooth ile iletilir
- Alternatif rota karşılaştırması

---
**TUA Astro Hackathon 2026** — Afet Yönetiminde Yerli Uydu Verisi Entegrasyonu
""")

st.divider()
st.caption("🛰️ TUA Astro Hackathon 2026 — Afet Yönetiminde Yerli Uydu Verisi Entegrasyonu Demo")