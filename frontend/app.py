import streamlit as st
import geemap
import ee
import os
import json
import zipfile
import shutil

# Make sure backend is in path to importgee pipeline if needed
import sys
import asyncio

# Fix for "There is no current event loop in thread 'ScriptRunner.scriptThread'."
# localtileserver needs an asyncio event loop to run.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from gee_pipeline import get_aoi

st.set_page_config(layout="wide", page_title="Riverbank Encroachment Detection")

# Initialize Earth Engine
@st.cache_resource
def init_ee():
    try:
        ee.Initialize()
        return True, ""
    except Exception as e:
        return False, str(e)

st.title("🌊 Riverbank Encroachment Detection")
st.markdown("Monitor the Mula-Mutha river (or custom location) using Google Earth Engine satellite imagery.")

success, error_msg = init_ee()
if not success:
    st.error(f"Earth Engine Initialization Failed: {error_msg}")
    st.info("If the error mentions a missing project ID, you can fix this by running `earthengine set_project YOUR_PROJECT_ID` in your terminal, or you can manually edit `app.py` to use `ee.Initialize(project='your-project-id')`.")
    st.stop()

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# 1. Location Selection
st.sidebar.subheader("1. Area of Interest")
location_mode = st.sidebar.radio("Location Mode", ["Default (Warje, Pune)", "Custom Coordinates"])

if location_mode == "Default (Warje, Pune)":
    bbox = [73.76, 18.47, 73.83, 18.52]
    aoi = get_aoi(bbox)
else:
    min_lon = st.sidebar.number_input("Min Longitude", value=73.76)
    min_lat = st.sidebar.number_input("Min Latitude", value=18.47)
    max_lon = st.sidebar.number_input("Max Longitude", value=73.83)
    max_lat = st.sidebar.number_input("Max Latitude", value=18.52)
    bbox = [min_lon, min_lat, max_lon, max_lat]
    aoi = get_aoi(bbox)

# 2. Year Range
st.sidebar.subheader("2. Time Period")
start_year, end_year = st.sidebar.slider("Select Year Range", 2005, 2025, (2015, 2025))

# --- Main Dashboard ---
row1_col1, row1_col2 = st.columns([2, 1])

from gee_pipeline import get_aoi, fetch_imagery

with row1_col1:
    st.subheader("Geospatial Visualization")
    # Initialize Map
    m = geemap.Map(center=[(bbox[1] + bbox[3])/2, (bbox[0] + bbox[2])/2], zoom=13)
    m.addLayer(aoi, {'color': 'red'}, "Area of Interest", False)
    
    # Add Earth Engine layers directly for the selected years!
    try:
        # Fetch imagery for Start Year
        img_start, _ = fetch_imagery(start_year, aoi)
        vis_params = {'bands': ['Red', 'Green', 'Blue'], 'min': 0, 'max': 0.3}
        # Add start year as base
        m.addLayer(img_start, vis_params, f"Satellite Image ({start_year})", False)
        
        # Fetch imagery for End Year
        img_end, _ = fetch_imagery(end_year, aoi)
        m.addLayer(img_end, vis_params, f"Satellite Image ({end_year})", True)
        
        # We can also try adding the local change map if it exists, or just the EE images.
        # Let's add the local change map if it's there
        base_dir_app = os.path.dirname(os.path.dirname(__file__))
        dset_name = f"dataset_{start_year}_{end_year}"
        change_map_path = os.path.join(base_dir_app, 'datasets', dset_name, 'changes', f'change_map_{start_year}_{end_year}.tif')
        
        if os.path.exists(change_map_path):
            # geemap can add local raster
            m.add_raster(change_map_path, bands=[3], colormap="reds", layer_name="Detected Encroachment (Red)", opacity=0.7)
    except Exception as e:
        st.warning(f"Could not load interactive map layers: {e}")
    
    m.to_streamlit(height=600)

with row1_col2:
    st.subheader("Statistical Reports")
    
    # Mocking or reading actual generated reports
    st.info("Run the backend pipeline to generate local datasets and change reports. Once generated, the statistics will appear here.")
    
    # Checking if reports exist
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_name = f"dataset_{start_year}_{end_year}"
    changes_dir = os.path.join(base_dir, 'datasets', dataset_name, 'changes')
    
    if os.path.exists(changes_dir):
        reports = [f for f in os.listdir(changes_dir) if f.startswith('report_') and f.endswith('.json')]
        if reports:
            latest_report = max(reports) # simplistic way to get highest years
            with open(os.path.join(changes_dir, latest_report), 'r') as f:
                data = json.load(f)
                st.metric("Water Loss (Hectares)", f"{data['water_loss_hectares']:.2f}")
                st.metric("New Built-Up Area (Hectares)", f"{data['new_built_up_hectares']:.2f}")
                st.metric("Detected Encroachment (Hectares)", f"{data['direct_encroachment_hectares']:.2f}", delta="- Warning", delta_color="inverse")
        else:
            st.warning("No reports generated for this time frame yet.")
    else:
        st.warning(f"No changes directory found for dataset: {dataset_name}. Run `backend/change_detection.py` first.")

    st.subheader("Download Dataset")
    ds_path = os.path.join(base_dir, 'datasets', dataset_name)
    zip_path = os.path.join(base_dir, 'datasets', f"{dataset_name}.zip")
    
    if st.button("Generate & Download ZIP"):
        if os.path.exists(ds_path):
            with st.spinner("Zipping dataset..."):
                shutil.make_archive(zip_path.replace('.zip', ''), 'zip', ds_path)
            
            with open(zip_path, "rb") as fp:
                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name=f"{dataset_name}.zip",
                    mime="application/zip"
                )
        else:
            st.error("Dataset folder does not exist. Please run `dataset_generator.py` for this year range.")
