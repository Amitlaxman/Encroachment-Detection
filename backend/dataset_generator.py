import os
import geemap
from gee_pipeline import get_aoi, initialize_ee
from image_processing import process_year



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
            # Download via geemap using local compute
            # Force scale=10 to maximize quality and ensure perfectly matching
            # image dimensions for Sentinel-2 (10m) and Landsat (resampled to 10m)
            print(f"Downloading {year} ({source}) at 10m scale...")
            geemap.ee_export_image(export_img, filename=out_path, scale=10, region=aoi, file_per_band=False)
            print(f"Successfully downloaded {year}.")
            
        except Exception as e:
            print(f"Failed to process year {year}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download satellite imagery for a range of years.')
    parser.add_argument('--start', type=int, default=2005, help='Start year')
    parser.add_argument('--end', type=int, default=2025, help='End year')
    parser.add_argument('--bbox', type=float, nargs=4, help='Bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--out_dir', type=str, help='Output directory for datasets')
    
    args = parser.parse_args()
    
    initialize_ee()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Central storage for all satellite images
    output_dir = args.out_dir if args.out_dir else os.path.join(base_dir, 'datasets', 'satellite_images')
    
    generate_dataset(args.start, args.end, output_dir, bbox=args.bbox)
