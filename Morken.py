##########################################################################################################################################################

import pandas as pd
import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from pyproj import Transformer
import os
import webbrowser
from shapely.geometry import LineString, Point
import xml.etree.ElementTree as ET
import numpy as np
from matplotlib_venn import venn2

# Load datasets
Morken = input("Informe o Caminho do Arquivo Contendo as Anomalias:")
df_morken = pd.read_excel(Morken, sheet_name="MATRIZ GUT")
df_anomalies = pd.read_excel(r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\List of Anomalies_Rosen_UTM.xlsx")
kml_filepath = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Rota Mineroduto.kml"

# Define anomaly groups
ANOMALY_GROUPS = {
    "Corrosion": ["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"],
    "Dent": ["Anomaly  / Dent"],
    "Girth Weld Anomaly": ["Anomaly  / Girth weld anomaly"],
    "Grinding": ["Anomaly  / Grinding"],
    "Lamination": ["Anomaly  / Lamination"],
    "Milling": ["Anomaly  / Milling"],
}

# Define colors for anomaly groups
ANOMALY_COLORS = {
    "Corrosion": "red",
    "Dent": "orange",
    "Girth Weld Anomaly": "green",
    "Grinding": "blue",
    "Lamination": "purple",
    "Milling": "brown",
}

# Function to load the pipeline route from KML
def load_kml_route(kml_file):
    with open(kml_file, 'r', encoding='utf-8') as f:
        kml_content = f.read()
    
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    root = ET.fromstring(kml_content)
    
    for placemark in root.findall(".//kml:Placemark", namespace):
        line_string = placemark.find(".//kml:LineString", namespace)
        if line_string is not None:
            coordinates = line_string.find(".//kml:coordinates", namespace).text.strip()
            coord_list = []
            for coord in coordinates.split():
                lon, lat, *_ = map(float, coord.split(','))
                coord_list.append((lon, lat))
            return LineString(coord_list)  # Convert to Shapely LineString
    
    raise ValueError("❌ No LineString pipeline route found in the KML file.")

# Function to find nearby matches
def find_nearby_matches(gdf1, gdf2):
    # Ensure both are in a projected CRS (e.g., EPSG:3857 - units in meters)
    gdf1 = gdf1.to_crs(epsg=3857)
    gdf2 = gdf2.to_crs(epsg=3857)

    threshold = 3  # radius in meters
    gdf2_sindex = gdf2.sindex
    matches = []

    for idx1, row1 in gdf1.iterrows():
        # Spatial index bounding box filter
        possible_matches_idx = list(gdf2_sindex.intersection(row1.geometry.buffer(threshold).bounds))
        possible_matches = gdf2.iloc[possible_matches_idx]
        
        for idx2, row2 in possible_matches.iterrows():
            if row1.geometry.distance(row2.geometry) <= threshold:
                matches.append((idx1, idx2))

    return matches

# Load pipeline route
gdf_route = gpd.GeoDataFrame(geometry=[load_kml_route(kml_filepath)], crs="EPSG:4326")

# Convert "wl [%]" to numeric
df_anomalies["wl [%]"] = pd.to_numeric(df_anomalies["wl [%]"], errors="coerce")

# Filter anomalies based on type and depth
df_anomalies_filtered = df_anomalies.copy()

# Apply depth filter only for corrosion anomalies
df_anomalies_filtered = df_anomalies_filtered[
    (~df_anomalies_filtered["anom. type/ident"].isin(ANOMALY_GROUPS["Corrosion"])) | 
    (df_anomalies_filtered["anom. type/ident"].isin(ANOMALY_GROUPS["Corrosion"]) & (df_anomalies_filtered["wl [%]"] >= 0))
]

# Ensure valid coordinates
df_anomalies_filtered = df_anomalies_filtered.dropna(subset=["long", "lat"])
df_anomalies_filtered["long"] = pd.to_numeric(df_anomalies_filtered["long"], errors="coerce")
df_anomalies_filtered["lat"] = pd.to_numeric(df_anomalies_filtered["lat"], errors="coerce")

# Convert filtered DataFrame to GeoDataFrame
gdf_anomalies_filtered = gpd.GeoDataFrame(
    df_anomalies_filtered, geometry=gpd.points_from_xy(df_anomalies_filtered["long"], df_anomalies_filtered["lat"]), crs="EPSG:4326"
)

# Convert and filter Morken anomalies
df_morken = df_morken[df_morken['Prioridade Final'] >=0 ]
transformer = Transformer.from_crs("EPSG:31983", "EPSG:4326", always_xy=True)
df_morken["long"], df_morken["lat"] = zip(*df_morken.apply(
    lambda row: transformer.transform(row["Leste"], row["Norte"]) if pd.notnull(row["Leste"]) and pd.notnull(row["Norte"]) else (None, None),
    axis=1
))
df_morken = df_morken.dropna(subset=["long", "lat"])
gdf_morken = gpd.GeoDataFrame(df_morken, geometry=gpd.points_from_xy(df_morken["long"], df_morken["lat"]), crs="EPSG:4326")

# Convert datasets to Web Mercator
gdf_anomalies_3857 = gdf_anomalies_filtered.to_crs(epsg=3857)
gdf_morken_3857 = gdf_morken.to_crs(epsg=3857)

# Find matches
matches = find_nearby_matches(gdf_morken_3857, gdf_anomalies_3857)
print(f"✅ Found {len(matches)} matches within 2 meters.")

# Update matched and unmatched Morken anomalies color
gdf_morken_3857["color"] = "magenta"  # Default color for unmatched anomalies
matched_indices = [idx1 for idx1, _ in matches]
gdf_morken_3857.loc[gdf_morken_3857.index.isin(matched_indices), "color"] = "yellow"  # Matched anomalies

# Add a "Match_Status" column to df_morken
df_morken["Match_Status"] = "Unmatched"  # Default value for all anomalies
df_morken.loc[matched_indices, "Match_Status"] = "Matched"  # Update matched entries

# Save the updated DataFrame with Match_Status
df_morken.to_excel("Updated_Morken_Anomalies_With_Status.xlsx", index=False)

# Create summary plot
fig1, ax = plt.subplots(figsize=(16, 16))
gdf_route.to_crs(epsg=3857).plot(ax=ax, color='cyan', linewidth=2, label="Pipeline Route")

for anomaly_type, color in ANOMALY_COLORS.items():
    subset = gdf_anomalies_3857[gdf_anomalies_3857["anom. type/ident"].isin(ANOMALY_GROUPS[anomaly_type])]
    subset.plot(ax=ax, color=color, marker='o', markersize=100, alpha=1, edgecolor='black', label=anomaly_type)

gdf_morken_3857.plot(ax=ax, color=gdf_morken_3857["color"], marker='D', markersize=150, alpha=1, edgecolor='black', label="Matched Morken Anomalies")

gdf_morken_3857[gdf_morken_3857["color"] == "magenta"].plot(ax=ax, color="magenta", marker='D', markersize=150, alpha=1, edgecolor='black', label="Unmatched Morken Anomalies")

ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs='EPSG:3857')
plt.legend()
plt.savefig("Pipeline_Map_Matches.png", dpi=300, bbox_inches="tight")
plt.show()


# Create the Venn diagram
# Calculate unique anomalies

rosen_anomalies= len(gdf_anomalies_3857) 
shared_anomalies = df_morken['Match_Status'].value_counts().get('Matched', 0)  # Count of "Matched"
unique_morken = df_morken['Match_Status'].value_counts().get('Unmatched', 0)  # Count of "Unmatched"
unique_rosen = rosen_anomalies - shared_anomalies  # Remaining anomalies in Rosen dataset

# Create the Venn diagram
fig, ax = plt.subplots(figsize=(8, 8))
venn = venn2(subsets=(unique_rosen, unique_morken, shared_anomalies), 
             set_labels=('Rosen Anomalies', 'Morken Anomalies'))

# Label the Venn diagram with counts
venn.get_label_by_id('10').set_text(f"{unique_rosen}")  # Unique to Rosen Anomalies
venn.get_label_by_id('01').set_text(f"{unique_morken}")  # Unique to Morken Anomalies
venn.get_label_by_id('11').set_text(f"{shared_anomalies}")  # Shared anomalies

# Adjust the legend position to avoid overlap
legend_labels = [
    f"Unique to Rosen Anomalies: {unique_rosen}",
    f"Unique to Morken Anomalies: {unique_morken}",
    f"Shared Anomalies: {shared_anomalies}"
]
ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

# Title and saving
ax.set_title("Overlap of Rosen and Morken Anomalies")
plt.savefig("Venn_Diagram.png", dpi=300, bbox_inches="tight")
plt.show()





# fig, ax = plt.subplots(figsize=(8, 8))
# venn = venn2(subsets=(len(gdf_anomalies_3857) - len(matches), 
#                       len(gdf_morken_3857) - len(matches), 
#                       len(matches)), 
#              set_labels=('Total Anomalies', 'Morken Anomalies'))

# # Labeling each part of the Venn diagram
# venn.get_label_by_id('10').set_text("Left (Unique to Total Anomalies)")  # Left circle, no intersection
# venn.get_label_by_id('01').set_text("Right (Unique to Morken Anomalies)")  # Right circle, no intersection
# venn.get_label_by_id('11').set_text("Intersection (Shared Anomalies)")  # Intersection

# # Title and saving
# ax.set_title("Overlap of Anomalies and Morken Dataset")
# plt.savefig("Venn_Diagram.png", dpi=300, bbox_inches="tight")
# plt.show()


###############################################################################################################


# Function to load processed Morken data
def load_processed_morken(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"❌ Error loading processed Morken file: {e}")
        return None

# Load processed Morken data
df_morken = load_processed_morken("Updated_Morken_Anomalies_With_Status.xlsx")

# Load anomalies dataset
df_anomalies = pd.read_excel(r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\List of Anomalies_Rosen_UTM.xlsx")
kml_filepath = r"C:\Users\User\OneDrive\Documentos\Área de Trabalho\Mosaic\Rota Mineroduto.kml"

# Function to load the pipeline route from KML
def load_kml_route(kml_file):
    with open(kml_file, 'r', encoding='utf-8') as f:
        kml_content = f.read()
    
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    root = ET.fromstring(kml_content)
    
    for placemark in root.findall(".//kml:Placemark", namespace):
        line_string = placemark.find(".//kml:LineString", namespace)
        if line_string is not None:
            coordinates = line_string.find(".//kml:coordinates", namespace).text.strip()
            coord_list = [(float(coord.split(',')[1]), float(coord.split(',')[0])) for coord in coordinates.split()]
            return coord_list  # Return list of (lat, lon) tuples
    
    return []  # Return empty list if no route found

# Load pipeline route
pipeline_route = load_kml_route(kml_filepath)

# Define anomaly groups
ANOMALY_GROUPS = {
    "Corrosion": ["Anomaly  / Corrosion", "Anomaly  / Corrosion cluster"],
    "Dent": ["Anomaly  / Dent"],
    "Girth Weld Anomaly": ["Anomaly  / Girth weld anomaly"],
    "Grinding": ["Anomaly  / Grinding"],
    "Lamination": ["Anomaly  / Lamination"],
    "Milling": ["Anomaly  / Milling"],
}

# Define colors for anomaly groups
ANOMALY_COLORS = {
    "Corrosion": "red",
    "Dent": "orange",
    "Girth Weld Anomaly": "green",
    "Grinding": "blue",
    "Lamination": "purple",
    "Milling": "brown",
}

# Apply depth filter only for corrosion anomalies
df_anomalies_filtered = df_anomalies.copy()
df_anomalies_filtered = df_anomalies_filtered[
    (~df_anomalies_filtered["anom. type/ident"].isin(ANOMALY_GROUPS["Corrosion"])) | 
    (df_anomalies_filtered["anom. type/ident"].isin(ANOMALY_GROUPS["Corrosion"]) & (df_anomalies_filtered["wl [%]"] >= 0))
]

# Convert DataFrames to GeoDataFrames
gdf_anomalies = gpd.GeoDataFrame(df_anomalies_filtered.dropna(subset=["long", "lat"]), geometry=gpd.points_from_xy(df_anomalies_filtered["long"], df_anomalies_filtered["lat"]), crs="EPSG:4326")
gdf_morken = gpd.GeoDataFrame(df_morken, geometry=gpd.points_from_xy(df_morken["long"], df_morken["lat"]), crs="EPSG:4326")

# Create interactive map
the_map = folium.Map(location=[-20, -45], zoom_start=6)

# Add pipeline route
if pipeline_route:
    folium.PolyLine(locations=pipeline_route, color="blue", weight=3, opacity=0.7, tooltip="Pipeline Route").add_to(the_map)

# Feature groups for filtering
anomaly_groups = {key: folium.FeatureGroup(name=key).add_to(the_map) for key in ANOMALY_GROUPS.keys()}
morken_matched = folium.FeatureGroup(name="Morken Matched Anomalies", overlay=True).add_to(the_map)
morken_unmatched = folium.FeatureGroup(name="Morken Unmatched Anomalies", overlay=True).add_to(the_map)

# Add anomalies with correct colors
for _, row in gdf_anomalies.iterrows():
    for anomaly_group, anomaly_types in ANOMALY_GROUPS.items():
        if row["anom. type/ident"] in anomaly_types:
            color = ANOMALY_COLORS.get(anomaly_group, "black")
            folium.CircleMarker(
                location=[row["lat"], row["long"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=row["anom. type/ident"]
            ).add_to(anomaly_groups[anomaly_group])

# Add Morken anomalies with different colors for matched/unmatched
for _, row in gdf_morken.iterrows():
    color = "yellow" if row["Match_Status"] == "Matched" else "magenta"
    group = morken_matched if row["Match_Status"] == "Matched" else morken_unmatched
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"Morken Anomaly - {row['Match_Status']}"
    ).add_to(group)

# Add layer control
folium.LayerControl().add_to(the_map)

# Save interactive map
the_map.save("Pipeline_Anomalies_Map.html")
print("✅ Interactive map updated and saved as Pipeline_Anomalies_Map.html")
print("✅ Updated Morken anomalies loaded from Updated_Morken_Anomalies_With_Status.xlsx")
