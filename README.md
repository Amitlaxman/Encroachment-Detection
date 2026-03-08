# Riverbank Encroachment Detection using Satellite Imagery

This project is a full-stack system to automatically collect satellite imagery, generate machine-learning ready datasets, and detect riverbank encroachments over time. It is specifically configured for the **Mula–Mutha river near Warje, Pune**, but can be customized to monitor any region.

## Project Structure
- `backend/gee_pipeline.py`: Google Earth Engine logic to fetch and cloud-mask satellite images with a cascading fallback strategy (Sentinel-2 -> Landsat 8 -> Landsat 5 -> Landsat 7).
- `backend/image_processing.py`: Calculation of water indices (MNDWI, optimized for turbid river water) and built-up indices (NDBI / ESA WorldCover) to generate segmentation masks.
- `backend/dataset_generator.py`: Iterates through year intervals, fetching satellite composites and calculating masks via GEE.
- `backend/patch_masks_local.py`: A utility script to recalculate MNDWI and built-up masks in-place on existing datasets without re-downloading from Earth Engine.
- `backend/tiling.py`: Slices the downloaded large GeoTIFFs into ML-ready appropriately-padded `512x512` chunks.
- `backend/change_detection.py`: Compares imagery between two years to identify water surface loss and direct encroachment (in hectares) by new built-up areas.
- `frontend/app.py`: Interactive Streamlit dashboard for visualizing the region, statistics, before/after map layers, and downloading generated datasets.
- `datasets/`: Local storage for datasets and reports. *(Note: Generated datasets are explicitly ignored by Git to keep the repository lightweight.)*

## Setup Instructions

### 1. Prerequisites
Ensure you have Python installed. You must also have a Google account authorized for Google Earth Engine.

### 2. Virtual Environment
Run the following in your terminal to initialize the environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r backend\requirements.txt -r frontend\requirements.txt
```

### 3. Authentication
You **MUST** authenticate Earth Engine before running the tools. In your active virtual environment, run:
```powershell
earthengine authenticate
```
Follow the browser prompts to log in and select your Google Cloud project. 

If you encounter an Earth Engine initialization error afterward, ensure your project ID is set properly in your environment:
```powershell
earthengine set_project YOUR_PROJECT_ID
```

### 4. Running the Pipeline
Run the following scripts from the root directory of the project to generate datasets, cut tiles, and build change detection statistics:
```powershell
python backend/dataset_generator.py
python backend/tiling.py
python backend/change_detection.py
```
*(Optional: If you need to tweak the water index or built-up thresholds later, you can modify `backend/patch_masks_local.py` and run it instead of `dataset_generator.py` to save hours of download time!)*

### 5. Launching the Dashboard
Launch the visualization dashboard:
```powershell
streamlit run frontend/app.py
```
