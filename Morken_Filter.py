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

    unique_types = df_anomalies["anom. type/ident"].dropna().unique()
    selected_types = st.sidebar.multiselect("Tipos de Anomalia", options=sorted(unique_types), default=list(unique_types))

    # === Apply filters ===
    df_anomalies_filtered = df_anomalies[
        (df_anomalies["wl [%]"] >= selected_wl[0]) &
        (df_anomalies["wl [%]"] <= selected_wl[1]) &
        (df_anomalies["anom. type/ident"].isin(selected_types))
    ]
    df_morken_filtered = df_morken[
        (df_morken["Prioridade Final"] >= selected_prio[0]) &
        (df_morken["Prioridade Final"] <= selected_prio[1])
    ]

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
        "Milling": ["Anomaly  / Milling"]
    }
    anomaly_colors = {
        "Corrosion": "red", "Dent": "orange", "Girth Weld Anomaly": "green",
        "Grinding": "blue", "Lamination": "purple", "Milling": "brown"
    }

    # === Create Map ===
    m = folium.Map(location=[-20, -45], zoom_start=6)
    fgs = {k: FeatureGroup(name=k).add_to(m) for k in anomaly_groups}
    fg_matched = FeatureGroup(name="Morken Matched", show=True).add_to(m)
    fg_unmatched = FeatureGroup(name="Morken Unmatched", show=True).add_to(m)

    # Plot Rosen Anomalies
    for _, row in gdf_anomalies.iterrows():
        for group, types in anomaly_groups.items():
            if row["anom. type/ident"] in types:
                folium.CircleMarker(
                    location=[row["lat"], row["long"]],
                    radius=5,
                    color=anomaly_colors[group],
                    fill=True,
                    fill_opacity=0.7
                ).add_to(fgs[group])
                break

    # Plot Morken Anomalies
    for _, row in gdf_morken.iterrows():
        color = "yellow" if row["Match_Status"] == "Matched" else "magenta"
        group = fg_matched if row["Match_Status"] == "Matched" else fg_unmatched
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(group)

    # Auto-zoom
    if not gdf_anomalies.empty:
        bounds = gdf_anomalies.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    elif not gdf_morken.empty:
        bounds = gdf_morken.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Show map only
    st_folium(m, width=1000, height=600)


with tab2:
    # Calculate Venn counts
    rosen_anomalies = len(gdf_anomalies)
    shared_anomalies = df_morken['Match_Status'].value_counts().get('Matched', 0)
    unique_morken = df_morken['Match_Status'].value_counts().get('Unmatched', 0)
    unique_rosen = rosen_anomalies - shared_anomalies

    # Create Venn diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    venn = venn2(subsets=(unique_rosen, unique_morken, shared_anomalies),
                 set_labels=('Rosen Anomalias', 'Morken Anomalias'))

    # Label sections with counts
    venn.get_label_by_id('10').set_text(str(unique_rosen))      # Unique Rosen
    venn.get_label_by_id('01').set_text(str(unique_morken))     # Unique Morken
    venn.get_label_by_id('11').set_text(str(shared_anomalies))  # Shared

    # Add legend
    legend_labels = [
        f"🔴 Exclusivas da Rosen: {unique_rosen}",
        f"🟣 Exclusivas da Morken: {unique_morken}",
        f"🟢 Compartilhadas: {shared_anomalies}"
    ]
    ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title("🔍 Sobreposição de Anomalias")

    st.pyplot(fig)

