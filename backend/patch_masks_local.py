import os
import rasterio
import numpy as np


def patch_masks(images_dir):
    """
    Recompute water and built-up masks on existing composite GeoTIFFs
    using calibrated thresholds optimized for the Mula-Mutha river area.
    
    Band layout: 1=Red, 2=Green, 3=Blue, 4=NIR, 5=SWIR1, 6=NDWI, 7=water_mask, 8=built_up_mask
    """
    for f in os.listdir(images_dir):
        if f.startswith("composite_") and f.endswith(".tif"):
            path = os.path.join(images_dir, f)
            print(f"Patching masks for {f}...")

            with rasterio.open(path, 'r+') as src:
                green = src.read(2).astype(np.float64)
                red = src.read(1).astype(np.float64)
                nir = src.read(4).astype(np.float64)
                swir1 = src.read(5).astype(np.float64)

                # Valid data mask
                valid = (red > 0) | (green > 0) | (nir > 0) | (swir1 > 0)

                # MNDWI = (Green - SWIR1) / (Green + SWIR1)
                # Standard threshold: > 0.0 identifies water bodies
                # Works consistently across Landsat and Sentinel-2
                mndwi_denom = green + swir1
                mndwi_denom[mndwi_denom == 0] = 1e-10
                mndwi = (green - swir1) / mndwi_denom

                new_water = ((mndwi > 0.0) & valid).astype(src.dtypes[6])

                # --- NDBI for built-up ---
                # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
                ndbi_denom = swir1 + nir
                ndbi_denom[ndbi_denom == 0] = 1e-10
                ndbi = (swir1 - nir) / ndbi_denom

                # Threshold 0.0: standard for built-up
                new_built_up = ((ndbi > 0.0) & valid).astype(src.dtypes[7])

                # Also update NDWI band (band 6) with the correct MNDWI values
                src.write(mndwi.astype(src.dtypes[5]), 6)
                src.write(new_water, 7)
                src.write(new_built_up, 8)

                water_pct = 100 * np.sum(new_water > 0) / new_water.size
                built_pct = 100 * np.sum(new_built_up > 0) / new_built_up.size
                print(f"  -> Water: {water_pct:.1f}%, Built-up: {built_pct:.1f}%")

    print("Done patching.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Patch water and built-up masks locally on existing GeoTIFFs.')
    parser.add_argument('--dir', type=str, help='Directory containing composite images')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    images_dir = args.dir if args.dir else os.path.join(base_dir, 'datasets', 'satellite_images', 'images')

    if os.path.exists(images_dir):
        patch_masks(images_dir)
    else:
        print(f"Directory {images_dir} not found.")
