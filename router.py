"""
router.py
---------
Ağırlıklı yol grafı üzerinde A* rota optimizasyonu.

Graf ağırlık formülü:
    w(e) = α × mesafe_km + β × hasar × 10 + γ × süre_dakika

Kurtarma operasyonları için varsayılan: β yüksek tutulur (hasar öncelikli).
"""

import networkx as nx
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from typing import Optional


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = sin(dφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(dλ / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(max(0, 1 - a)))


def assign_edge_weights(
    G,
    damage_scores: dict,
    alpha: float = 0.30,
    beta: float  = 0.50,
    gamma: float = 0.20,
):
    """
    Her yol kenarına bileşik ağırlık ata.

    Parameters
    ----------
    G             : osmnx MultiDiGraph
    damage_scores : {(u, v, key): float}
    alpha         : Mesafe katsayısı
    beta          : Hasar katsayısı   ← kurtarma operasyonlarında en kritik
    gamma         : Zaman katsayısı
    """
    for u, v, key, data in G.edges(keys=True, data=True):
        damage = damage_scores.get((u, v, key), 0.30)

        dist_km = data.get("length", 100) / 1000.0

        # Hasar durumuna göre gerçekçi hız tahmini (km/h)
        if damage < 0.25:
            speed_kmh = 50.0   # Normal, enkazdan arındırılmış
        elif damage < 0.50:
            speed_kmh = 25.0   # Hafif hasar, dikkatli geçiş
        elif damage < 0.75:
            speed_kmh = 8.0    # Ciddi hasar, yavaş geçiş
        else:
            speed_kmh = 1.5    # Kritik hasar, neredeyse imkansız

        time_min = (dist_km / speed_kmh) * 60.0 if speed_kmh > 0 else 9999.0

        weight = (
            alpha * dist_km
            + beta  * damage * 10.0
            + gamma * time_min
        )

        G[u][v][key].update(
            {
                "weight":    max(weight, 1e-6),
                "damage":    damage,
                "speed_kmh": speed_kmh,
                "time_min":  round(time_min, 2),
            }
        )

    return G


def find_safest_route(G, start_node: int, end_node: int) -> Optional[dict]:
    """
    A* algoritması ile en güvenli rotayı bul.
    Fallback: Dijkstra (A* başarısız olursa).

    Returns
    -------
    dict veya None
        path, distance_km, time_min, avg_damage, max_damage, safety_score
    """

    def heuristic(n1, n2):
        # Admissible heuristic: alpha × doğrusal mesafe
        # (gerçek ağırlık her zaman ≥ buna olduğu için kabul edilebilir)
        d = haversine_km(
            G.nodes[n1]["y"], G.nodes[n1]["x"],
            G.nodes[n2]["y"], G.nodes[n2]["x"],
        )
        return 0.30 * d  # α katsayısıyla uyumlu

    for algo in ["astar", "dijkstra"]:
        try:
            if algo == "astar":
                path = nx.astar_path(
                    G, start_node, end_node,
                    heuristic=heuristic, weight="weight"
                )
            else:
                path = nx.dijkstra_path(
                    G, start_node, end_node, weight="weight"
                )
            return {"path": path, **_path_statistics(G, path)}
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NodeNotFound):
            continue

    return None


def find_alternative_routes(G, start_node: int, end_node: int, k: int = 3) -> list:
    """
    k adet alternatif rota bul (Yen's k-shortest paths).
    Ana rota dahil toplam k rota döner.
    """
    routes = []
    try:
        paths = list(
            nx.shortest_simple_paths(G, start_node, end_node, weight="weight")
        )
        for path in paths[:k]:
            routes.append({"path": path, **_path_statistics(G, path)})
    except Exception:
        main = find_safest_route(G, start_node, end_node)
        if main:
            routes.append(main)
    return routes


def _path_statistics(G, path: list) -> dict:
    """Rota üzerindeki istatistikleri hesapla."""
    total_dist = 0.0
    total_time = 0.0
    damage_vals = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if v not in G[u]:
            continue
        # Paralel kenarlar arasından en hafif olanı seç
        best = min(G[u][v].values(), key=lambda d: d.get("weight", 9999))
        total_dist += best.get("length", 0) / 1000.0
        total_time += best.get("time_min", 0)
        damage_vals.append(best.get("damage", 0.0))

    avg_dmg = float(np.mean(damage_vals)) if damage_vals else 0.0
    max_dmg = float(np.max(damage_vals))  if damage_vals else 0.0

    return {
        "distance_km":   round(total_dist, 2),
        "time_min":      round(total_time, 1),
        "avg_damage":    round(avg_dmg, 3),
        "max_damage":    round(max_dmg, 3),
        "safety_score":  round((1 - avg_dmg) * 100, 1),
        "node_count":    len(path),
    }
