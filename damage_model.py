"""
damage_model.py
---------------
Hasar skoru hesaplama motoru — İKİ KATMANLI MODEL

KATMAN 1 — Copernicus EMS Gerçek Uydu Verisi (öncelikli)
  - EMSR648 aktivasyonu: 6 Şubat 2023 Kahramanmaraş depremi
  - builtUpP dosyaları: uydu görüntüsü analiziyle tespit edilmiş bina hasarı
  - Her yol segmenti için yakın binalara bakıp ağırlıklı skor hesaplar

KATMAN 2 — GMPE Analitik Model (fallback)
  - Copernicus verisi olmayan bölgeler için
  - Ground Motion Prediction Equation + yol tipi kırılganlığı
  - Deterministik, tekrarlanabilir

Gerçek sistemde Katman 1:
  - Sentinel-2 görüntüsü (rasterio) + SAM / U-Net segmentasyonu ile üretilir
  - Bu demo: hazır Copernicus hasar noktaları kullanılıyor
"""

import os
from scipy.spatial import cKDTree
import numpy as np
import geopandas as gpd
from math import radians, sin, cos, sqrt, atan2
from functools import lru_cache
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─── 6 Şubat 2023 Depremleri ─────────────────────────────────────────────────
EPICENTERS = [
    {"lat": 37.288, "lon": 37.043, "magnitude": 7.7, "name": "Pazarcık (Ana)"},
    {"lat": 38.024, "lon": 37.208, "magnitude": 7.6, "name": "Elbistan (İkinci)"},
]

# Yol tipi → kırılganlık katsayısı
ROAD_VULNERABILITY = {
    "motorway":        0.35, "motorway_link":  0.35,
    "trunk":           0.45, "trunk_link":     0.45,
    "primary":         0.60, "primary_link":   0.60,
    "secondary":       0.78, "secondary_link": 0.78,
    "tertiary":        0.88, "tertiary_link":  0.88,
    "residential":     1.00, "living_street":  1.00,
    "unclassified":    0.92, "service":        1.00,
    "pedestrian":      1.00, "path":           1.00,
}

# Copernicus hasar sınıfı → 0–1 skoru
COPERNICUS_DAMAGE_MAP = {
    "no visible damage":  0.05,
    "possibly damaged":   0.40,
    "damaged":            0.65,
    "destroyed":          0.95,
}

# Hangi AOI dosyaları yüklenecek (proje klasöründe olması yeterli)
COPERNICUS_FILES = [
    "EMSR648_AOI08_GRA_PRODUCT_builtUpP_r1_v1.json",  # Pazarcık (episantr)
    "EMSR648_AOI04_GRA_PRODUCT_builtUpA_r1_v1.json",  # Kahramanmaraş
    "EMSR648_AOI16_GRA_PRODUCT_builtUpA_r1_v1.json",  # Nurdağı
]

# Yakın bina arama yarıçapı (derece cinsinden, ~500m)
SEARCH_RADIUS_DEG = 0.005


# ─── Copernicus Verisi Yükleme ────────────────────────────────────────────────

def _load_copernicus_data() -> Optional[gpd.GeoDataFrame]:
    """
    Mevcut AOI dosyalarını yükler ve birleştirir.
    Dosya bulunamazsa None döner → GMPE fallback devreye girer.
    """
    gdfs = []
    for fname in COPERNICUS_FILES:
        if os.path.exists(fname):
            try:
                gdf = gpd.read_file(fname)
                # damage_gra sütununu normalize et
                gdf["damage_score"] = (
                    gdf["damage_gra"]
                    .str.lower()
                    .str.strip()
                    .map(COPERNICUS_DAMAGE_MAP)
                    .fillna(0.30)  # Bilinmeyen → orta hasar varsayımı
                )
                gdfs.append(gdf[["damage_score", "geometry"]])
                print(f"[Copernicus] ✅ Yüklendi: {fname} ({len(gdf)} nokta)")
            except Exception as e:
                print(f"[Copernicus] ⚠️ Yüklenemedi: {fname} — {e}")
        else:
            print(f"[Copernicus] ℹ️ Dosya yok: {fname} — GMPE kullanılacak")

    if not gdfs:
        return None

    combined = gpd.pd.concat(gdfs, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")
    print(f"[Copernicus] Toplam {len(combined)} hasar noktası yüklendi.")
    return combined


# Modül yüklendiğinde bir kere çalışır
_COPERNICUS_GDF: Optional[gpd.GeoDataFrame] = _load_copernicus_data()
_COPERNICUS_AVAILABLE = _COPERNICUS_GDF is not None


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki koordinat arasındaki küresel mesafeyi km cinsinden hesaplar."""
    R = 6371.0
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = sin(dφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(dλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(max(0, 1 - a)))


def _gmpe_score(lat: float, lon: float, road_type: str, seed: int) -> float:
    """
    GMPE tabanlı analitik hasar skoru.
    Copernicus verisi yokken devreye girer.
    """
    base_damage = 0.0
    for ep in EPICENTERS:
        dist = haversine_km(lat, lon, ep["lat"], ep["lon"])
        pga = (10 ** ((ep["magnitude"] - 6.0) * 0.5)) * np.exp(-dist / 55.0)
        base_damage = max(base_damage, float(np.clip(pga, 0.0, 1.0)))

    rt = road_type if isinstance(road_type, str) else "unclassified"
    vulnerability = ROAD_VULNERABILITY.get(rt, 0.90)

    _seed = int(seed % (2 ** 31)) if seed is not None else int(abs(lat * lon * 1e4)) % (2 ** 31)
    rng = np.random.RandomState(_seed)
    noise = rng.normal(0.0, 0.04)

    return float(np.clip(base_damage * vulnerability + noise, 0.0, 1.0))


# Modül yüklenince bir kez KD-Tree kur
_KDTREE = None
_KDTREE_SCORES = None

def _build_kdtree():
    global _KDTREE, _KDTREE_SCORES
    if not _COPERNICUS_AVAILABLE or _KDTREE is not None:
        return
    cx = _COPERNICUS_GDF.geometry.centroid.x.values
    cy = _COPERNICUS_GDF.geometry.centroid.y.values
    _KDTREE = cKDTree(np.column_stack([cx, cy]))
    _KDTREE_SCORES = _COPERNICUS_GDF["damage_score"].values
    print(f"[Copernicus] KD-Tree kuruldu — {len(_KDTREE_SCORES)} nokta")

_build_kdtree()


def _copernicus_score(lat: float, lon: float) -> Optional[float]:
    if not _COPERNICUS_AVAILABLE or _KDTREE is None:
        return None

    # 500m yarıçapındaki tüm noktaları bul (derece ≈ 0.005)
    indices = _KDTREE.query_ball_point([lon, lat], r=SEARCH_RADIUS_DEG)

    if not indices:
        return None

    scores  = _KDTREE_SCORES[indices]
    # Ağırlık: merkeze yakın nokta daha etkili
    pts     = _KDTREE.data[indices]
    dists   = np.sqrt((pts[:, 0] - lon)**2 + (pts[:, 1] - lat)**2)
    weights = 1.0 / np.maximum(dists**2, 1e-8)

    return float(np.average(scores, weights=weights))


# ─── ANA FONKSIYON ───────────────────────────────────────────────────────────

def compute_damage_score(
    lat: float,
    lon: float,
    road_type: str = "unclassified",
    seed: int = None,
) -> float:
    """
    Bir koordinat noktası için [0.0, 1.0] arasında hasar skoru hesaplar.

    0.0 = tamamen güvenli
    1.0 = tamamen hasar görmüş / geçilemez

    Önce Copernicus gerçek uydu verisi denenir.
    Veri yoksa GMPE analitik modele düşer.

    Parameters
    ----------
    lat, lon   : Yol segmentinin orta noktası
    road_type  : OSM highway tag değeri
    seed       : Deterministik rastgelelik için seed
    """
    cop_score = _copernicus_score(lat, lon)

    if cop_score is not None:
        # ─── KATMAN 1: Copernicus Gerçek Veri ───
        # Yol tipi kırılganlığını Copernicus skoruna da uygula
        rt = road_type if isinstance(road_type, str) else "unclassified"
        vulnerability = ROAD_VULNERABILITY.get(rt, 0.90)

        # Copernicus bina hasarı × yol kırılganlığı
        # (Binalar yıkıldıysa yolda da enkaz/dolgu olasılığı yüksek)
        blended = cop_score * (0.7 + 0.3 * vulnerability)
        return float(np.clip(blended, 0.0, 1.0))

    else:
        # ─── KATMAN 2: GMPE Analitik Model (Fallback) ───
        return _gmpe_score(lat, lon, road_type, seed)


def damage_label(score: float) -> tuple:
    """Hasar skorunu insan okunabilir etiket + renge çevirir."""
    if score < 0.25:
        return "Güvenli",   "#1D9E75", "🟢"
    elif score < 0.50:
        return "Dikkatli",  "#EF9F27", "🟡"
    elif score < 0.75:
        return "Tehlikeli", "#E24B4A", "🔴"
    else:
        return "Geçilemez", "#501313", "⬛"


def data_source_info() -> dict:
    """Hangi veri kaynağının aktif olduğunu döner (UI için)."""
    if _COPERNICUS_AVAILABLE:
        n = len(_COPERNICUS_GDF)
        aois = [f.split("_")[1] for f in COPERNICUS_FILES if os.path.exists(f)]
        return {
            "source": "Copernicus EMS (EMSR648)",
            "points": n,
            "aois": aois,
            "fallback": "GMPE (Copernicus verisi olmayan bölgeler)",
            "icon": "🛰️"
        }
    else:
        return {
            "source": "GMPE Analitik Model",
            "points": 0,
            "aois": [],
            "fallback": None,
            "icon": "📐"
        }