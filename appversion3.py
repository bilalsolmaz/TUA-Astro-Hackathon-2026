"""
app.py — OPTİMİZE AFET KURTARMA ROTA SİSTEMİ (v2)
=================================================

Çalıştır: streamlit run app.py
"""

import os
import pickle
import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import numpy as np
import pandas as pd

# ─── SAYFA ─────────────────────────────────────────────────────

st.set_page_config(
page_title="Afet Kurtarma Rota Sistemi",
page_icon="🚨",
layout="wide"
)

# ─── SABİTLER ──────────────────────────────────────────────────

START_LOCATIONS = {
"📍 Pazarcık": (37.496, 37.290),
"📍 Nurdağı": (37.183, 36.731),
"📍 Türkoğlu": (37.382, 36.861),
}

END_LOCATIONS = {
"🏥 KMaraş Hastane": (37.585, 36.938),
"🏥 Necip Fazıl": (37.598, 36.893),
}

CACHE_FILE = "graph.pkl"

# ─── GRAPH LOAD (OSMnx 2.x FIXED) ──────────────────────────────

@st.cache_resource
def load_graph():
if os.path.exists(CACHE_FILE):
with open(CACHE_FILE, "rb") as f:
return pickle.load(f)

```
bbox = (37.45, 37.65, 36.82, 37.05)  # south, north, west, east

G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

with open(CACHE_FILE, "wb") as f:
    pickle.dump(G, f)

return G
```

# ─── GERÇEKÇİ HASAR MODELİ ─────────────────────────────────────

def compute_damage_score(lat, lon, road_type, seed):
epicenter = (37.288, 37.043)

```
dist = np.sqrt((lat - epicenter[0])**2 + (lon - epicenter[1])**2)
base = np.exp(-dist * 15)

road_factor = {
    "motorway": 0.6,
    "primary": 0.8,
    "secondary": 1.0,
    "residential": 1.2,
}.get(road_type, 1.0)

noise = (seed % 100) / 500
return min(1.0, base * road_factor + noise)
```

# ─── LOKAL DAMAGE (HIZLI) ──────────────────────────────────────

def compute_local_damage(G, center, radius=0.15):
scores = {}
for u, v, k, data in G.edges(keys=True, data=True):
lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2

```
    if abs(lat - center[0]) > radius or abs(lon - center[1]) > radius:
        continue

    rt = data.get("highway", "residential")
    rt = rt[0] if isinstance(rt, list) else rt

    seed = hash((u, v, k)) & 0xffffffff
    scores[(u, v, k)] = compute_damage_score(lat, lon, rt, seed)

return scores
```

# ─── EDGE WEIGHT ───────────────────────────────────────────────

def assign_weights(G, damage, alpha, beta, gamma):
for u, v, k, d in G.edges(keys=True, data=True):
dist = d.get("length", 1)
time = d.get("travel_time", 1)
dmg = damage.get((u, v, k), 0.3)

```
    d["weight"] = alpha*dist + beta*(dmg*1000) + gamma*time
return G
```

# ─── ROUTE ─────────────────────────────────────────────────────

def get_route(G, start, end):
import networkx as nx
try:
path = nx.shortest_path(G, start, end, weight="weight")
return path
except:
return None

# ─── HARİTA ────────────────────────────────────────────────────

def build_map(G, damage, route, start, end):
m = folium.Map(location=[37.56, 36.94], zoom_start=12)

```
edges = list(G.edges(keys=True, data=True))
sample = edges[:2500]

for u, v, k, d in sample:
    dmg = damage.get((u, v, k), 0.2)

    if dmg > 0.8:
        continue

    color = "green" if dmg < 0.3 else "orange" if dmg < 0.6 else "red"

    coords = [
        (G.nodes[u]["y"], G.nodes[u]["x"]),
        (G.nodes[v]["y"], G.nodes[v]["x"])
    ]

    folium.PolyLine(coords, color=color, weight=2, opacity=0.6).add_to(m)

if route:
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
    folium.PolyLine(coords, color="blue", weight=6).add_to(m)

folium.Marker(start, icon=folium.Icon(color="blue")).add_to(m)
folium.Marker(end, icon=folium.Icon(color="red")).add_to(m)

return m
```

# ─── UI ────────────────────────────────────────────────────────

st.title("🚨 Afet Kurtarma Rota Sistemi (Optimize)")

start_name = st.selectbox("Başlangıç", list(START_LOCATIONS.keys()))
end_name   = st.selectbox("Hedef", list(END_LOCATIONS.keys()))

alpha = st.slider("Mesafe", 0.1, 0.8, 0.3)
beta  = st.slider("Hasar", 0.1, 0.8, 0.5)
gamma = 1 - alpha - beta

if st.button("Rota Hesapla"):
G = load_graph()

```
start_coords = START_LOCATIONS[start_name]
end_coords   = END_LOCATIONS[end_name]

start_node = ox.nearest_nodes(G, start_coords[1], start_coords[0])
end_node   = ox.nearest_nodes(G, end_coords[1], end_coords[0])

damage = compute_local_damage(G, start_coords)

G = assign_weights(G, damage, alpha, beta, gamma)

route = get_route(G, start_node, end_node)

m = build_map(G, damage, route, start_coords, end_coords)

st_folium(m, width=1000, height=600)
```
