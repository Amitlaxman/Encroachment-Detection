import os
import geemap
from gee_pipeline import get_aoi, initialize_ee
from image_processing import process_year

# Define scales based on the satellite
SCALE_S2 = 10  # Sentinel-2 resolution is 10m
SCALE_L8 = 30  # Landsat 8 resolution is 30m

def generate_dataset(start_year, end_year, output_dir, bbox=None):
    """
    Iterates from start_year to end_year, processes satellite imagery,
    and downloads the composite + masks as a GeoTIFF.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    aoi = get_aoi(bbox) if bbox else get_aoi()
    
    print(f"Generating dataset from {start_year} to {end_year} into {output_dir}")
    
    for year in range(start_year, end_year + 1):
        out_path = os.path.join(images_dir, f'composite_{year}.tif')
        
        # Check if already downloaded
        if os.path.exists(out_path):
            print(f"Year {year} already exists, skipping...")
            continue
            
        print(f"Processing Year {year}...")
        try:
            img, source = process_year(year, aoi)
            
            # Select bands to export: Sentinel-2 and Landsat have different names but we renamed them
            # B4, B3, B2, B8/B5, B11/B6 -> Red, Green, Blue, NIR, SWIR1
            # added bands: NDWI, water_mask, NDBI/built_up_mask
            bands_to_export = ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'NDWI', 'water_mask', 'built_up_mask']
            
            export_img = img.select(bands_to_export)
            scale = SCALE_S2 if source == 'Sentinel-2' else SCALE_L8
            
            # Download via geemap using local compute
            print(f"Downloading {year} ({source}) at {scale}m scale...")
            geemap.ee_export_image(export_img, filename=out_path, scale=scale, region=aoi, file_per_band=False)
            print(f"Successfully downloaded {year}.")
            
        except Exception as e:
            print(f"Failed to process year {year}: {e}")

if __name__ == "__main__":
    initialize_ee()
    
    # We define datasets to generate as requested
    base_dir = os.path.dirname(os.path.dirname(__file__))
    datasets_dir = os.path.join(base_dir, 'datasets')
    
    # Example ranges:
    # Dataset 1: 2005-2025
    generate_dataset(2005, 2025, os.path.join(datasets_dir, 'dataset_2005_2025'))
    # Dataset 2: 2010-2025
    generate_dataset(2010, 2025, os.path.join(datasets_dir, 'dataset_2010_2025'))
    # Dataset 3: 2015-2025
    generate_dataset(2015, 2025, os.path.join(datasets_dir, 'dataset_2015_2025'))
