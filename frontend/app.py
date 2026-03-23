import streamlit as st
import geemap
import ee
import os
import json
import zipfile
import shutil

# Make sure backend is in path to import gee pipeline if needed
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
AOI_PRESETS = {
    "Warje, Pune (Default)": {
        "bbox": [73.76, 18.47, 73.83, 18.52],
        "dataset_slug": "satellite_images",
    },
    "Baner (clear Mula river stretch)": {
        "bbox": [73.762, 18.545, 73.812, 18.585],
        "dataset_slug": "preset_baner",
    },
    "Sangamwadi (Mula-Mutha confluence)": {
        "bbox": [73.845, 18.515, 73.895, 18.555],
        "dataset_slug": "preset_sangamwadi",
    },
    "Balewadi (upstream Mula view)": {
        "bbox": [73.745, 18.555, 73.790, 18.595],
        "dataset_slug": "preset_balewadi",
    },
    "Pashan (river corridor)": {
        "bbox": [73.775, 18.520, 73.825, 18.560],
        "dataset_slug": "preset_pashan",
    },
}

location_mode = st.sidebar.radio("Location Mode", ["Preset Pune AOIs", "Custom Coordinates"])

if location_mode == "Preset Pune AOIs":
    selected_preset = st.sidebar.selectbox("Preset Location", list(AOI_PRESETS.keys()), index=0)
    bbox = AOI_PRESETS[selected_preset]["bbox"]
    selected_dataset_slug = AOI_PRESETS[selected_preset]["dataset_slug"]
    aoi = get_aoi(bbox)
    st.sidebar.caption(
        f"Selected BBox: {bbox[0]:.4f}, {bbox[1]:.4f} to {bbox[2]:.4f}, {bbox[3]:.4f}"
    )
else:
    coord_input = st.sidebar.text_input("Center Coordinate (Lat, Lon)", value="18.471412, 73.823849")
    try:
        # Parse the input string
        lat_str, lon_str = coord_input.split(',')
        center_lat = float(lat_str.strip())
        center_lon = float(lon_str.strip())
        
        # Calculate bounding box (roughly 2.2km wide/high: roughly ~0.02 degrees)
        offset = 0.02
        min_lon = center_lon - offset
        max_lon = center_lon + offset
        min_lat = center_lat - offset
        max_lat = center_lat + offset
        
        bbox = [min_lon, min_lat, max_lon, max_lat]
        aoi = get_aoi(bbox)
        st.sidebar.caption(f"Calculated BBox: {min_lon:.4f}, {min_lat:.4f} to {max_lon:.4f}, {max_lat:.4f}")
    except ValueError:
        st.sidebar.error("Invalid format. Please use 'Lat, Lon' (e.g., 18.471412, 73.823849).")
        st.stop()

# 2. Year Range
st.sidebar.subheader("2. Time Period")
start_year, end_year = st.sidebar.slider("Select Year Range", 2005, 2025, (2015, 2025))

# --- Helpers ---
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import numpy as np
from PIL import Image

def tif_to_rgb_image(tif_path):
    """Read bands 1,2,3 (R,G,B) from a GeoTIFF, stretch to 0-255, return PIL Image."""
    with rasterio.open(tif_path) as src:
        r = src.read(1).astype(float)
        g = src.read(2).astype(float)
        b = src.read(3).astype(float)

    def stretch(band):
        p2, p98 = np.nanpercentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        band = np.clip(band, p2, p98)
        band = (band - p2) / (p98 - p2 + 1e-9) * 255
        return band.astype(np.uint8)

    rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
    return Image.fromarray(rgb)


def create_overlay_image(satellite_tif_path, change_map_tif_path, overlay_band=4, alpha=0.5):
    """
    Create an RGB satellite image with a colored overlay from the change map.
    
    Change map bands:
        Band 1: Water (old year)
        Band 2: Water (new year) 
        Band 3: New built-up areas
        Band 4: Encroachment
    
    overlay_band: which band to overlay (default 4 = encroachment)
    alpha: transparency of overlay (0-1)
    """
    # Get RGB base image
    with rasterio.open(satellite_tif_path) as src:
        r = src.read(1).astype(float)
        g = src.read(2).astype(float)
        b = src.read(3).astype(float)
        sat_width = src.width
        sat_height = src.height

    def stretch(band):
        p2, p98 = np.nanpercentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        band = np.clip(band, p2, p98)
        band = (band - p2) / (p98 - p2 + 1e-9) * 255
        return band.astype(np.float64)

    r_s, g_s, b_s = stretch(r), stretch(g), stretch(b)

    # Read change map and handle potential size mismatch
    with rasterio.open(change_map_tif_path) as cm_src:
        cm_width = cm_src.width
        cm_height = cm_src.height
        
        if cm_width != sat_width or cm_height != sat_height:
            # Resize the change map to match satellite image using nearest neighbor
            mask_data = cm_src.read(overlay_band).astype(float)
            mask_img = Image.fromarray(mask_data)
            mask_img = mask_img.resize((sat_width, sat_height), Image.NEAREST)
            mask_data = np.array(mask_img)
        else:
            mask_data = cm_src.read(overlay_band).astype(float)

    # Create an overlay: red for encroachment, blue for water, orange for new built-up
    overlay_r = np.zeros_like(r_s)
    overlay_g = np.zeros_like(g_s)
    overlay_b = np.zeros_like(b_s)

    mask = mask_data > 0

    if overlay_band == 1:  # Water old - blue
        overlay_r[mask] = 50
        overlay_g[mask] = 100
        overlay_b[mask] = 255
    elif overlay_band == 2:  # Water new - cyan
        overlay_r[mask] = 0
        overlay_g[mask] = 200
        overlay_b[mask] = 255
    elif overlay_band == 3:  # New built-up - orange
        overlay_r[mask] = 255
        overlay_g[mask] = 165
        overlay_b[mask] = 0
    elif overlay_band == 4:  # Encroachment - red
        overlay_r[mask] = 255
        overlay_g[mask] = 30
        overlay_b[mask] = 30

    # Blend
    r_out = np.where(mask, r_s * (1 - alpha) + overlay_r * alpha, r_s)
    g_out = np.where(mask, g_s * (1 - alpha) + overlay_g * alpha, g_s)
    b_out = np.where(mask, b_s * (1 - alpha) + overlay_b * alpha, b_s)

    rgb = np.stack([r_out.astype(np.uint8), g_out.astype(np.uint8), b_out.astype(np.uint8)], axis=-1)
    return Image.fromarray(rgb)


def create_multi_overlay(satellite_tif_path, change_map_tif_path, show_water=True, show_built=True, show_encroach=True, alpha=0.5):
    """
    Create an RGB satellite image with multiple colored overlays from the change map.
    
    Color scheme:
        Blue: Water bodies (old year)
        Cyan: Water bodies (new year)
        Orange: New built-up areas  
        Red: Encroachment zones
    """
    with rasterio.open(satellite_tif_path) as src:
        r = src.read(1).astype(float)
        g = src.read(2).astype(float)
        b = src.read(3).astype(float)
        sat_w, sat_h = src.width, src.height

    def stretch(band):
        p2, p98 = np.nanpercentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
        band = np.clip(band, p2, p98)
        band = (band - p2) / (p98 - p2 + 1e-9) * 255
        return band.astype(np.float64)

    r_s, g_s, b_s = stretch(r), stretch(g), stretch(b)

    with rasterio.open(change_map_tif_path) as cm:
        cm_w, cm_h = cm.width, cm.height
        bands = {}
        for i in range(1, cm.count + 1):
            band_data = cm.read(i).astype(float)
            if cm_w != sat_w or cm_h != sat_h:
                band_img = Image.fromarray(band_data)
                band_data = np.array(band_img.resize((sat_w, sat_h), Image.NEAREST))
            bands[i] = band_data > 0

    # Layer overlays with priority: encroachment > new_built > water_new > water_old
    layers = []
    if show_water and 1 in bands:
        layers.append((bands[1], (50, 100, 255)))    # Water old = blue
    if show_water and 2 in bands:
        layers.append((bands[2], (0, 200, 255)))      # Water new = cyan
    if show_built and 3 in bands:
        layers.append((bands[3], (255, 165, 0)))       # New built-up = orange
    if show_encroach and 4 in bands:
        layers.append((bands[4], (255, 30, 30)))       # Encroachment = red

    r_out, g_out, b_out = r_s.copy(), g_s.copy(), b_s.copy()

    for mask, (cr, cg, cb) in layers:
        r_out = np.where(mask, r_out * (1 - alpha) + cr * alpha, r_out)
        g_out = np.where(mask, g_out * (1 - alpha) + cg * alpha, g_out)
        b_out = np.where(mask, b_out * (1 - alpha) + cb * alpha, b_out)

    rgb = np.stack([
        np.clip(r_out, 0, 255).astype(np.uint8),
        np.clip(g_out, 0, 255).astype(np.uint8),
        np.clip(b_out, 0, 255).astype(np.uint8)
    ], axis=-1)
    return Image.fromarray(rgb)


def show_responsive_image(image_obj, caption=None):
    """Render image full-width across Streamlit versions.
    Streamlit <=1.32 uses `use_column_width`; newer versions use `use_container_width`.
    """
    kwargs = {}
    if caption is not None:
        kwargs["caption"] = caption
    try:
        st.image(image_obj, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image_obj, use_column_width=True, **kwargs)


# --- Main Dashboard ---
row1_col1, row1_col2 = st.columns([2, 1])

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if location_mode == "Preset Pune AOIs":
    dataset_folder = os.path.join(base_dir, 'datasets', selected_dataset_slug)
else:
    # Use formatted coordinates for custom folder
    coord_clean = coord_input.replace(", ", "_").replace(",", "_")
    dataset_folder = os.path.join(base_dir, 'datasets', f"coords_{coord_clean}")

images_dir = os.path.join(dataset_folder, 'images')
changes_dir = os.path.join(dataset_folder, 'changes')
backend_dir = os.path.join(base_dir, 'backend')

with row1_col1:
    st.subheader("Satellite Image Comparison")

    img_col1, img_col2 = st.columns(2)

    def show_year_image(col, year):
        tif_path = os.path.join(images_dir, f'composite_{year}.tif')
        with col:
            st.markdown(f"**{year}**")
            if os.path.exists(tif_path):
                try:
                    img = tif_to_rgb_image(tif_path)
                    show_responsive_image(img)
                except Exception as e:
                    st.error(f"Could not read image for {year}: {e}")
            else:
                st.warning(f"No composite image found for {year}. Run `dataset_generator.py` first.")

    show_year_image(img_col1, start_year)
    show_year_image(img_col2, end_year)

    missing_imagery = not os.path.exists(os.path.join(images_dir, f'composite_{start_year}.tif')) or \
                      not os.path.exists(os.path.join(images_dir, f'composite_{end_year}.tif'))
    
    button_label = "Download Missing Imagery" if missing_imagery else "Force Redownload Imagery for Selected Area"
    button_type = "primary" if missing_imagery else "secondary"

    if st.button(button_label, type=button_type):
        st.info(f"Downloading imagery for {start_year} and {end_year}...")
        log_placeholder = st.empty()
        
        # Use Popen to stream logs
        import subprocess
        gen_script = os.path.join(backend_dir, 'dataset_generator.py')
        
        # Helper to run script and stream output
        def run_and_stream(year):
            cmd = [sys.executable, gen_script, "--start", str(year), "--end", str(year), "--out_dir", dataset_folder]
            cmd.extend(["--bbox", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])])
            
            process = subprocess.Popen(cmd, cwd=base_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logs = f"--- Starting processing for {year} ---\n"
            for line in iter(process.stdout.readline, ''):
                logs += line
                log_placeholder.code(logs, language="bash")
            process.communicate() # Wait for process to terminate
            if process.returncode != 0:
                st.error(f"Failed for {year}. Exit code: {process.returncode}")
                return False
            return True

        success1 = run_and_stream(start_year)
        if success1:
            run_and_stream(end_year)
            st.success("Download complete!")
            import time
            time.sleep(1)
            st.rerun()

# --- Change Detection Overlay ---
change_map_path = os.path.join(changes_dir, f'change_map_{start_year}_{end_year}.tif')
new_sat_path = os.path.join(images_dir, f'composite_{end_year}.tif')

if os.path.exists(change_map_path) and os.path.exists(new_sat_path):
    st.subheader("🗺️ Encroachment & Change Map")
    st.caption("Overlaid on the newer satellite image")

    overlay_col1, overlay_col2 = st.columns([3, 1])
    
    with overlay_col2:
        st.markdown("**Overlay Settings**")
        show_water = st.checkbox("Show Water Bodies", value=True)
        show_built = st.checkbox("Show New Built-up", value=True)
        show_encroach = st.checkbox("Show Encroachment", value=True)
        overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.5, 0.05)
        
        st.markdown("**Legend**")
        st.markdown("🔵 Water (old) &nbsp; 🔹 Water (new)")
        st.markdown("🟠 New built-up &nbsp; 🔴 Encroachment")

    with overlay_col1:
        try:
            overlay_img = create_multi_overlay(
                new_sat_path, change_map_path,
                show_water=show_water,
                show_built=show_built,
                show_encroach=show_encroach,
                alpha=overlay_alpha
            )
            show_responsive_image(overlay_img, caption=f"Change map: {start_year} → {end_year}")
        except Exception as e:
            st.error(f"Could not generate overlay: {e}")

with row1_col2:
    st.subheader("Statistical Reports")
    
    report_name = f"report_{start_year}_{end_year}.json"
    report_path = os.path.join(changes_dir, report_name)
    
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data = json.load(f)
            st.metric("Water Loss (Hectares)", f"{data['water_loss_hectares']:.2f}")
            st.metric("New Built-Up Area (Hectares)", f"{data['new_built_up_hectares']:.2f}")
            st.metric("Detected Encroachment (Hectares)", f"{data['direct_encroachment_hectares']:.2f}", delta="- Warning", delta_color="inverse")
            
            # Show detailed breakdown if available
            if 'summary' in data:
                with st.expander("Detailed Breakdown"):
                    summary = data['summary']
                    st.write(f"**Pixel area**: {data.get('pixel_area_m2', 'N/A'):.1f} m²")
                    st.write(f"**Total valid pixels**: {data.get('total_valid_pixels', 'N/A'):,}")
                    for key, val in summary.items():
                        label = key.replace('_', ' ').title()
                        st.write(f"**{label}**: {val:,}")
    else:
        st.warning(f"No report found for {start_year} vs {end_year}.")
        if os.path.exists(os.path.join(images_dir, f'composite_{start_year}.tif')) and \
           os.path.exists(os.path.join(images_dir, f'composite_{end_year}.tif')):
            if st.button("Run Change Detection"):
                with st.spinner("Analyzing changes..."):
                    import subprocess
                    cd_script = os.path.join(backend_dir, 'change_detection.py')
                    result = subprocess.run(
                        [sys.executable, cd_script, "--old", str(start_year), "--new", str(end_year), "--out_dir", dataset_folder],
                        cwd=base_dir, capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        st.error(f"Change detection failed:\n{result.stderr}")
                    else:
                        st.success(result.stdout)
                    st.rerun()
        else:
            st.info("Download imagery first to enable change detection.")

    st.subheader("Download Results")
    dataset_name = f"encroachment_{start_year}_{end_year}"
    zip_path = os.path.join(base_dir, 'datasets', f"{dataset_name}.zip")
    
    if st.button("Generate & Download ZIP"):
        # Check if we have the files
        img_old = os.path.join(images_dir, f'composite_{start_year}.tif')
        img_new = os.path.join(images_dir, f'composite_{end_year}.tif')
        report = os.path.join(changes_dir, f'report_{start_year}_{end_year}.json')
        
        if os.path.exists(img_old) and os.path.exists(img_new) and os.path.exists(report):
            with st.spinner("Zipping results..."):
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Copy relevant files to temp dir
                    target_images = os.path.join(tmp_dir, 'images')
                    target_reports = os.path.join(tmp_dir, 'reports')
                    os.makedirs(target_images)
                    os.makedirs(target_reports)
                    
                    shutil.copy(img_old, target_images)
                    shutil.copy(img_new, target_images)
                    shutil.copy(report, target_reports)
                    
                    # Also include the change map if it exists
                    change_map = os.path.join(changes_dir, f'change_map_{start_year}_{end_year}.tif')
                    if os.path.exists(change_map):
                        shutil.copy(change_map, target_reports)

                    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', tmp_dir)
            
            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name=f"{dataset_name}.zip",
                    mime="application/zip"
                )
        else:
            st.error("Missing files. Please ensure imagery and change detection have been run for these years.")
