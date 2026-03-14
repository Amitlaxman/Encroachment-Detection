import os
import rasterio
from rasterio.windows import Window
import numpy as np

TILE_SIZE = 512

def tile_image(image_path, output_dir, year):
    """
    Reads a raster image and slices it into 512x512 chunks.
    Separates the composite bands from the masks (water, built_up).
    """
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    water_masks_dir = os.path.join(output_dir, 'water_masks')
    built_up_masks_dir = os.path.join(output_dir, 'built_up_masks')
    metadata_dir = os.path.join(output_dir, 'metadata')
    
    for d in [images_dir, water_masks_dir, built_up_masks_dir, metadata_dir]:
        os.makedirs(d, exist_ok=True)
    
    try:
        with rasterio.open(image_path) as src:
            # Expected bands: Red, Green, Blue, NIR, SWIR1, NDWI, water_mask, built_up_mask
            width = src.width
            height = src.height
            
            tile_count = 0
            for row in range(0, height, TILE_SIZE):
                for col in range(0, width, TILE_SIZE):
                    window = Window(col, row, TILE_SIZE, TILE_SIZE)
                    
                    # Read all bands for the window
                    data = src.read(window=window)
                    
                    # If window is smaller than 512x512 (e.g. at the edge or small Landsat image), pad it
                    _, h, w = data.shape
                    if h < TILE_SIZE or w < TILE_SIZE:
                        pad_h = TILE_SIZE - h
                        pad_w = TILE_SIZE - w
                        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                    # Bands 1-3: RGB, Band 4: NIR, Band 5: SWIR1, Band 6: NDWI
                    # Band 7: water_mask, Band 8: built_up_mask
                    rgb_nir_swir = data[0:5, :, :]
                    water_mask = data[6, :, :]
                    built_up_mask = data[7, :, :]
                    
                    tile_name = f"{year}_tile_{tile_count}"
                    
                    # Save as npy or tif. Let's save as multi-band TIF for imagery and TIF for masks
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'height': TILE_SIZE,
                        'width': TILE_SIZE,
                        'transform': src.window_transform(window)
                    })
                    
                    # Save image (Bands 1-5)
                    kwargs['count'] = 5
                    with rasterio.open(os.path.join(images_dir, f"{tile_name}.tif"), 'w', **kwargs) as dst:
                        dst.write(rgb_nir_swir)
                        
                    # Save water mask
                    kwargs['count'] = 1
                    with rasterio.open(os.path.join(water_masks_dir, f"{tile_name}.tif"), 'w', **kwargs) as dst:
                        dst.write(water_mask, 1)
                        
                    # Save built up mask
                    with rasterio.open(os.path.join(built_up_masks_dir, f"{tile_name}.tif"), 'w', **kwargs) as dst:
                        dst.write(built_up_mask, 1)
                        
                    # Save metadata
                    with open(os.path.join(metadata_dir, f"{tile_name}.txt"), 'w') as f:
                        f.write(f"Year: {year}\n")
                        f.write(f"Row: {row}\n")
                        f.write(f"Col: {col}\n")
                        f.write(f"Transform: {kwargs['transform']}\n")
                        
                    tile_count += 1
            print(f"Generated {tile_count} tiles for {year}")
    except Exception as e:
        print(f"Error tiling {image_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Tile satellite imagery into patches.')
    parser.add_argument('--year', type=int, help='Year to tile (if not specified, all found years will be tiled)')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    source_dir = os.path.join(base_dir, 'datasets', 'satellite_images', 'images')
    output_base = os.path.join(base_dir, 'datasets', 'satellite_images', 'tiles')
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found.")
    else:
        files_to_tile = []
        if args.year:
            files_to_tile = [f'composite_{args.year}.tif']
        else:
            files_to_tile = [f for f in os.listdir(source_dir) if f.startswith('composite_') and f.endswith('.tif')]
            
        for f in files_to_tile:
            # Extract year
            try:
                year = f.split('_')[1].split('.')[0]
                img_path = os.path.join(source_dir, f)
                # Output to a subfolder for each year's tiles within the main tiles directory
                year_tile_dir = os.path.join(output_base, year)
                print(f"Tiling {img_path} into {year_tile_dir}...")
                tile_image(img_path, year_tile_dir, year)
            except Exception as e:
                print(f"Skipping {f}: {e}")
