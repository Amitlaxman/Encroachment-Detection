import os
import rasterio
import numpy as np
import json
import pandas as pd

def detect_changes(image_path_old, image_path_new, output_dir, year_old, year_new):
    """
    Compares two years of processed satellite imagery.
    Detects water reduction and new built-up areas.
    """
    if not os.path.exists(image_path_old) or not os.path.exists(image_path_new):
        print(f"Images for {year_old} or {year_new} not found.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f'report_{year_old}_{year_new}.json')
    change_map_path = os.path.join(output_dir, f'change_map_{year_old}_{year_new}.tif')

    try:
        with rasterio.open(image_path_old) as src_old, rasterio.open(image_path_new) as src_new:
            # Read masks
            # Band 7: water_mask, Band 8: built_up_mask
            water_old = src_old.read(7)
            water_new = src_new.read(7)
            built_old = src_old.read(8)
            built_new = src_new.read(8)

            # Change logic:
            # Water reduction: was water before, not water now
            water_loss = (water_old > 0) & (water_new == 0)
            
            # Built-up increase: not built-up before, is built-up now
            new_built_up = (built_old == 0) & (built_new > 0)
            
            # Encroachment (strictly): was water or near water, and is now built up
            # For simplicity: water_loss AND new_built_up
            encroachment = water_loss & new_built_up

            transform = src_old.transform
            
            # If the CRS is geographic (like EPSG:4326), transform values are in degrees
            # 1 degree latitude ~= 111,320 meters. 
            # 1 degree longitude ~= 111,320 * cos(latitude) meters.
            if src_old.crs and src_old.crs.is_geographic:
                # Approximate center latitude from the transform and image height
                center_lat = transform[5] + (transform[4] * src_old.height / 2)
                lat_rad = np.radians(center_lat)
                pixel_width_m = abs(transform[0]) * 111320 * np.cos(lat_rad)
                pixel_height_m = abs(transform[4]) * 111320
                pixel_area_m2 = pixel_width_m * pixel_height_m
            else:
                pixel_area_m2 = abs(transform[0] * transform[4])
                
            water_loss_ha = np.sum(water_loss) * pixel_area_m2 / 10000
            new_built_up_ha = np.sum(new_built_up) * pixel_area_m2 / 10000
            encroachment_ha = np.sum(encroachment) * pixel_area_m2 / 10000

            # Save report
            report = {
                "period": f"{year_old}-{year_new}",
                "water_loss_hectares": float(water_loss_ha),
                "new_built_up_hectares": float(new_built_up_ha),
                "direct_encroachment_hectares": float(encroachment_ha)
            }
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)

            # Save change map (3 bands: Water Loss, New Built-up, Encroachment)
            kwargs = src_old.meta.copy()
            kwargs.update({
                'count': 3,
                'dtype': 'uint8'
            })
            
            with rasterio.open(change_map_path, 'w', **kwargs) as dst:
                dst.write(water_loss.astype('uint8'), 1)
                dst.write(new_built_up.astype('uint8'), 2)
                dst.write(encroachment.astype('uint8'), 3)

            print(f"Change detection successful: {encroachment_ha:.2f} ha of direct encroachment.")
            return report

    except Exception as e:
        print(f"Error generating change detection: {e}")
        return None

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    datasets = [
        ("2005", "2025", "dataset_2005_2025"),
        ("2010", "2025", "dataset_2010_2025"),
        ("2015", "2025", "dataset_2015_2025")
    ]
    
    for start_year, end_year, folder in datasets:
        dset_path = os.path.join(base_dir, 'datasets', folder, 'images')
        changes_dir = os.path.join(base_dir, 'datasets', folder, 'changes')
        
        img_old = os.path.join(dset_path, f'composite_{start_year}.tif')
        img_new = os.path.join(dset_path, f'composite_{end_year}.tif')
        
        print(f"Detecting changes for {folder}...")
        detect_changes(img_old, img_new, changes_dir, start_year, end_year)
