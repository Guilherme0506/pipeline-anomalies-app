import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from pyproj import Transformer
from folium import FeatureGroup
from pyproj import Transformer
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import requests
import xml.etree.ElementTree as ET

# === Load Data ===
# === Load Data ===
@st.cache_data
def load_data():
    df_morken = pd.read_excel("https://raw.githubusercontent.com/Guilherme0506/pipeline-anomalies-app/main/Updated_Morken_Anomalies_With_Status.xlsx")
    df_anomalies = pd.read_excel("https://raw.githubusercontent.com/Guilherme0506/pipeline-anomalies-app/main/List%20of%20Anomalies_Rosen_UTM.xlsx")
    return df_morken, df_anomalies

df_morken, df_anomalies = load_data()

# === VENN PLOT TAB ===
tab1, tab2 = st.tabs(["📍 Mapa Interativo", "📊 Diagrama de Venn"])

with tab1:
    @st.cache_data
    def load_kml_route(url):
        response = requests.get(url)
        kml_content = response.text
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
        root = ET.fromstring(kml_content)
        for placemark in root.findall(".//kml:Placemark", namespace):
            line_string = placemark.find(".//kml:LineString", namespace)
            if line_string is not None:
                coordinates = line_string.find(".//kml:coordinates", namespace).text.strip()
                coord_list = [(float(c.split(',')[1]), float(c.split(',')[0])) for c in coordinates.split()]
                return coord_list
        return []

# === Coordinate Transformation ===
transformer = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
df_morken["long"], df_morken["lat"] = zip(*df_morken.apply(
    lambda row: transformer.transform(row["Leste"], row["Norte"]) if pd.notnull(row["Leste"]) and pd.notnull(row["Norte"]) else (None, None),
    axis=1
))


# === Sidebar Filters ===
st.sidebar.header("🔎 Filtros Interativos")
wl_min = float(df_anomalies["wl [%]"].min())
wl_max = float(df_anomalies["wl [%]"].max())
selected_wl = st.sidebar.slider("Wall Loss (%)", min_value=wl_min, max_value=wl_max, value=(wl_min, wl_max))

prio_min = int(df_morken["Prioridade Final"].min())
prio_max = int(df_morken["Prioridade Final"].max())
selected_prio = st.sidebar.slider("Prioridade Final (Morken)", min_value=prio_min, max_value=prio_max, value=(prio_min, prio_max))

morken_status_filter = st.sidebar.multiselect(
    "Status de Anomalias Morken",
    options=["Matched", "Unmatched"],
    default=["Matched", "Unmatched"]
)

rosen_types = df_anomalies["anom. type/ident"].dropna().unique()
selected_types = st.sidebar.multiselect("Tipos de Anomalia (Rosen)", options=sorted(rosen_types), default=list(rosen_types))

# === Apply Filters ===
df_anomalies_filtered = df_anomalies[
    (df_anomalies["wl [%]"] >= selected_wl[0]) &
    (df_anomalies["wl [%]"] <= selected_wl[1]) &
    (df_anomalies["anom. type/ident"].isin(selected_types))
]

# Filter matched Morken anomalies only if the corresponding Rosen anomaly still exists
df_morken_matched = df_morken[df_morken["Match_Status"] == "Matched"].copy()
df_morken_unmatched = df_morken[df_morken["Match_Status"] == "Unmatched"].copy()

if "ID_Rosen" in df_morken_matched.columns and "ID" in df_anomalies_filtered.columns:
    matched_ids = set(df_anomalies_filtered["ID"].unique())
    df_morken_matched = df_morken_matched[df_morken_matched["ID_Rosen"].isin(matched_ids)]

# Filter unmatched anomalies by priority
df_morken_unmatched = df_morken_unmatched[
    (df_morken_unmatched["Prioridade Final"] >= selected_prio[0]) &
    (df_morken_unmatched["Prioridade Final"] <= selected_prio[1])
]

# Combine both matched and unmatched based on filters
df_morken_filtered = pd.concat([df_morken_matched, df_morken_unmatched], ignore_index=True)
df_morken_filtered = df_morken_filtered[df_morken_filtered["Match_Status"].isin(morken_status_filter)]

# === Convert to GeoDataFrames ===
gdf_anomalies = gpd.GeoDataFrame(df_anomalies_filtered.dropna(subset=["long", "lat"]),
                                 geometry=gpd.points_from_xy(df_anomalies_filtered["long"], df_anomalies_filtered["lat"]),
                                 crs="EPSG:4326")

gdf_morken = gpd.GeoDataFrame(df_morken_filtered.dropna(subset=["long", "lat"]),
                              geometry=gpd.points_from_xy(df_morken_filtered["long"], df_morken_filtered["lat"]),
                              crs="EPSG:4326")

    # === Anomaly Groups & Colors ===
anomaly_groups = {
        "Corrosion": ["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"],
        "Dent": ["Anomaly  / Dent"],
        "Girth Weld Anomaly": ["Anomaly  / Girth weld anomaly"],
        "Grinding": ["Anomaly  / Grinding"],
        "Lamination": ["Anomaly  / Lamination"],
        "Milling": ["Anomaly  / Milling"],
        "Morken Matched": [],
        "Morken Unmatched": []
}

anomaly_colors = {
        "Corrosion": "red",
        "Dent": "orange",
        "Girth Weld Anomaly": "green",
        "Grinding": "blue",
        "Lamination": "purple",
        "Milling": "brown",
        "Morken Matched": "yellow",
        "Morken Unmatched": "magenta"
}

# === Create Map ===
m = folium.Map(location=[-20, -45], zoom_start=6)
fgs = {k: folium.FeatureGroup(name=k).add_to(m) for k in anomaly_groups}

# Load and add pipeline route
kml_url = "https://raw.githubusercontent.com/Guilherme0506/pipeline-anomalies-app/main/Rota_Mineroduto.kml"
pipeline_route = load_kml_route(kml_url)

if pipeline_route:
    folium.PolyLine(
        locations=pipeline_route,
        color="blue",
        weight=3,
        opacity=0.7,
        tooltip="Pipeline Route"
    ).add_to(m)

for _, row in gdf_anomalies.iterrows():
    for group, types in anomaly_groups.items():
        if row["anom. type/ident"] in types:
            folium.CircleMarker(
                location=[row["lat"], row["long"]],
                radius=5,
                color=anomaly_colors[group],
                fill=True,
                fill_opacity=0.7,
                popup=f"Rosen - {row['anom. type/ident']}<br>WL: {row['wl [%]']}%"
            ).add_to(fgs[group])
            break

for _, row in gdf_morken.iterrows():
    status = "Morken Matched" if row["Match_Status"] == "Matched" else "Morken Unmatched"
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=6,
        color=anomaly_colors[status],
        fill=True,
        fill_opacity=0.7,
        popup=f"Morken - {row['Match_Status']}<br>Prioridade: {row['Prioridade Final']}"
    ).add_to(fgs[status])

folium.LayerControl(collapsed=False).add_to(m)

if not gdf_anomalies.empty:
    bounds = gdf_anomalies.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
elif not gdf_morken.empty:
    bounds = gdf_morken.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

st_folium(m, width=1000, height=600)

# === VENN PLOT ===
with tab2:
    rosen_ids = set(df_anomalies_filtered["ID"].dropna().astype(str))
    morken_matched_ids = set(df_morken_matched["ID_Rosen"].dropna().astype(str))
    morken_unmatched_ids = set(df_morken_unmatched["ID_Rosen"].dropna().astype(str))

    shared_ids = rosen_ids & morken_matched_ids
    unique_rosen = rosen_ids - shared_ids
    unique_morken = morken_unmatched_ids

    fig, ax = plt.subplots(figsize=(6, 6))
    venn = venn2(subsets=(len(unique_rosen), len(unique_morken), len(shared_ids)),
                 set_labels=("Rosen Anomalias", "Morken Anomalias"))

    venn.get_label_by_id("10").set_text(str(len(unique_rosen)))
    venn.get_label_by_id("01").set_text(str(len(unique_morken)))
    venn.get_label_by_id("11").set_text(str(len(shared_ids)))

    legend_labels = [
        f"🔴 Exclusivas da Rosen: {len(unique_rosen)}",
        f"🟣 Exclusivas da Morken: {len(unique_morken)}",
        f"🟢 Compartilhadas: {len(shared_ids)}"
    ]
    ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title("🔍 Sobreposição de Anomalias")
    st.pyplot(fig)
