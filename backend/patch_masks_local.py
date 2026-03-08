import os
import rasterio
import numpy as np

def patch_masks(images_dir):
    for f in os.listdir(images_dir):
        if f.startswith("composite_") and f.endswith(".tif"):
            path = os.path.join(images_dir, f)
            print(f"Patching masks for {f}...")
            
            with rasterio.open(path, 'r+') as src:
                # Band 2: Green, Band 4: NIR, Band 5: SWIR1
                green = src.read(2).astype(float)
                nir = src.read(4).astype(float)
                swir1 = src.read(5).astype(float)
                
                # Check for zero denominators
                mndwi_denom = (green + swir1)
                mndwi_denom[mndwi_denom == 0] = 1e-10
                mndwi = (green - swir1) / mndwi_denom
                
                # New water mask (MNDWI > -0.1 to account for turbidity)
                new_water = (mndwi > -0.1).astype('uint8')
                
                # Built up mask: NDBI = (SWIR1 - NIR)/(SWIR1 + NIR)
                ndbi_denom = (swir1 + nir)
                ndbi_denom[ndbi_denom == 0] = 1e-10
                ndbi = (swir1 - nir) / ndbi_denom
                
                # Use a slightly lower threshold for NDBI (e.g. -0.1) to catch more built-up areas
                new_built_up = (ndbi > -0.1).astype('uint8')
                
                # Write back to Band 7 (water_mask) and Band 8 (built_up_mask)
                src.write(new_water, 7)
                src.write(new_built_up, 8)
                
    print("Done patching.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # Assuming datasets 2005-2025 holds everything
    patch_masks(os.path.join(base_dir, 'datasets', 'dataset_2005_2025', 'images'))
    
    # We also update 2010 and 2015 folders if they got generated, but the user ran generator so let's patch all of them
    for dataset in ['dataset_2010_2025', 'dataset_2015_2025']:
        p = os.path.join(base_dir, 'datasets', dataset, 'images')
        if os.path.exists(p):
            patch_masks(p)
