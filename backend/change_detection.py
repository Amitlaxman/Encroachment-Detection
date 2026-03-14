import os
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import numpy as np
import json


def _reproject_to_match(src_path, ref_path, tmp_path):
    """
    Reprojects the image at src_path to match the resolution, CRS, and
    extent of ref_path, writing the result to tmp_path.
    Returns tmp_path.
    """
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(src_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height,
        })

        with rasterio.open(tmp_path, 'w', **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
    return tmp_path


def _compute_masks_from_bands(src):
    """
    Recompute water and built-up masks from the raw spectral bands.
    This ensures consistent thresholds regardless of what was baked 
    into the GeoTIFF at download time.
    
    Band layout: 1=Red, 2=Green, 3=Blue, 4=NIR, 5=SWIR1, 6=NDWI, 7=water_mask, 8=built_up_mask
    
    Returns: (water_mask, built_up_mask, valid_mask) as boolean arrays
    """
    green = src.read(2).astype(np.float64)
    nir = src.read(4).astype(np.float64)
    swir1 = src.read(5).astype(np.float64)
    red = src.read(1).astype(np.float64)

    # Identify valid pixels (where there is actual satellite data)
    valid = (red > 0) | (green > 0) | (nir > 0) | (swir1 > 0)

    # --- MNDWI for water ---
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    # Standard threshold: MNDWI > 0.0 identifies water bodies.
    # This works consistently across Landsat and Sentinel-2.
    mndwi_denom = green + swir1
    mndwi_denom[mndwi_denom == 0] = 1e-10
    mndwi = (green - swir1) / mndwi_denom

    water_mask = (mndwi > 0.0) & valid

    # --- NDBI for built-up ---
    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    # Standard threshold: NDBI > 0 indicates built-up areas.
    ndbi_denom = swir1 + nir
    ndbi_denom[ndbi_denom == 0] = 1e-10
    ndbi = (swir1 - nir) / ndbi_denom

    built_up_mask = (ndbi > 0.0) & valid

    return water_mask, built_up_mask, valid, mndwi, ndbi


def detect_changes(image_path_old, image_path_new, output_dir, year_old, year_new):
    """
    Compares two years of processed satellite imagery.
    Detects water reduction and new built-up areas.
    Handles resolution mismatches by reprojecting to a common grid.
    """
    if not os.path.exists(image_path_old) or not os.path.exists(image_path_new):
        print(f"Images for {year_old} or {year_new} not found.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f'report_{year_old}_{year_new}.json')
    change_map_path = os.path.join(output_dir, f'change_map_{year_old}_{year_new}.tif')

    try:
        # --- Handle resolution mismatch ---
        # If one image is higher-res (e.g. Sentinel-2 @10m) and the other is
        # lower-res (e.g. Landsat @30m), we reproject the high-res one to
        # match the lower-res grid so comparisons are apples-to-apples.
        with rasterio.open(image_path_old) as s1, rasterio.open(image_path_new) as s2:
            dims_old = (s1.width, s1.height)
            dims_new = (s2.width, s2.height)

        actual_old_path = image_path_old
        actual_new_path = image_path_new
        tmp_file = None

        if dims_old != dims_new:
            print(f"  Resolution mismatch: old={dims_old}, new={dims_new}. Reprojecting...")
            # We reproject the larger (higher-res) image to match the smaller one
            if dims_old[0] * dims_old[1] > dims_new[0] * dims_new[1]:
                # Old is bigger, reproject old to match new
                tmp_file = os.path.join(output_dir, f'_tmp_reprojected_{year_old}.tif')
                _reproject_to_match(image_path_old, image_path_new, tmp_file)
                actual_old_path = tmp_file
            else:
                # New is bigger, reproject new to match old
                tmp_file = os.path.join(output_dir, f'_tmp_reprojected_{year_new}.tif')
                _reproject_to_match(image_path_new, image_path_old, tmp_file)
                actual_new_path = tmp_file

        with rasterio.open(actual_old_path) as src_old, rasterio.open(actual_new_path) as src_new:
            # Recompute masks from the spectral bands with consistent thresholds
            water_old, built_old, valid_old, mndwi_old, ndbi_old = _compute_masks_from_bands(src_old)
            water_new, built_new, valid_new, mndwi_new, ndbi_new = _compute_masks_from_bands(src_new)

            # Only consider pixels where BOTH images have valid data
            both_valid = valid_old & valid_new

            # --- Create a water buffer zone ---
            # Encroachment doesn't only happen exactly AT water pixels;
            # it happens in a zone around water bodies (floodplain, banks).
            # Dilate the water mask by a few pixels to create a buffer.
            # Pure numpy dilation (no scipy needed):
            buffer_px = 3  # ~90m buffer at 30m resolution
            water_zone_old = water_old.copy()
            for _ in range(buffer_px):
                padded = np.pad(water_zone_old, 1, mode='constant', constant_values=False)
                water_zone_old = (
                    padded[1:-1, 1:-1] |  # center
                    padded[:-2, 1:-1] |    # up
                    padded[2:, 1:-1] |     # down
                    padded[1:-1, :-2] |    # left
                    padded[1:-1, 2:]       # right
                )

            # --- Change logic ---
            # Water reduction: was water before, not water now (in valid areas only)
            water_loss = water_old & ~water_new & both_valid

            # Built-up increase: not built-up before, is built-up now
            new_built_up = ~built_old & built_new & both_valid

            # Encroachment (strict): was water, is now built-up
            direct_encroachment = water_old & built_new & ~water_new & both_valid

            # Encroachment (broad): was in water zone, is now built-up but wasn't before
            zone_encroachment = water_zone_old & ~built_old & built_new & both_valid

            # Total encroachment is the union
            encroachment = direct_encroachment | zone_encroachment

            # --- Compute pixel area ---
            transform = src_old.transform
            if src_old.crs and src_old.crs.is_geographic:
                center_lat = transform[5] + (transform[4] * src_old.height / 2)
                lat_rad = np.radians(center_lat)
                pixel_width_m = abs(transform[0]) * 111320 * np.cos(lat_rad)
                pixel_height_m = abs(transform[4]) * 111320
                pixel_area_m2 = pixel_width_m * pixel_height_m
            else:
                pixel_area_m2 = abs(transform[0] * transform[4])

            water_loss_ha = float(np.sum(water_loss) * pixel_area_m2 / 10000)
            new_built_up_ha = float(np.sum(new_built_up) * pixel_area_m2 / 10000)
            encroachment_ha = float(np.sum(encroachment) * pixel_area_m2 / 10000)

            total_valid_pixels = int(np.sum(both_valid))
            water_old_pixels = int(np.sum(water_old & both_valid))
            water_new_pixels = int(np.sum(water_new & both_valid))
            built_old_pixels = int(np.sum(built_old & both_valid))
            built_new_pixels = int(np.sum(built_new & both_valid))

            # Save report
            report = {
                "period": f"{year_old}-{year_new}",
                "water_loss_hectares": water_loss_ha,
                "new_built_up_hectares": new_built_up_ha,
                "direct_encroachment_hectares": encroachment_ha,
                "pixel_area_m2": float(pixel_area_m2),
                "total_valid_pixels": total_valid_pixels,
                "summary": {
                    f"water_pixels_{year_old}": water_old_pixels,
                    f"water_pixels_{year_new}": water_new_pixels,
                    f"built_up_pixels_{year_old}": built_old_pixels,
                    f"built_up_pixels_{year_new}": built_new_pixels,
                    "water_loss_pixels": int(np.sum(water_loss)),
                    "new_built_up_pixels": int(np.sum(new_built_up)),
                    "encroachment_pixels": int(np.sum(encroachment)),
                }
            }
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)

            # Save change map (4 bands: Water Old, Water New, Built-up New, Encroachment)
            kwargs = src_old.meta.copy()
            kwargs.update({
                'count': 4,
                'dtype': 'uint8'
            })

            with rasterio.open(change_map_path, 'w', **kwargs) as dst:
                dst.write(water_old.astype('uint8'), 1)
                dst.write(water_new.astype('uint8'), 2)
                dst.write(new_built_up.astype('uint8'), 3)
                dst.write(encroachment.astype('uint8'), 4)

            print(f"  Total valid pixels: {total_valid_pixels}")
            print(f"  Water {year_old}: {water_old_pixels} px | Water {year_new}: {water_new_pixels} px")
            print(f"  Built-up {year_old}: {built_old_pixels} px | Built-up {year_new}: {built_new_pixels} px")
            print(f"  Water loss: {water_loss_ha:.4f} ha")
            print(f"  New built-up: {new_built_up_ha:.4f} ha")
            print(f"  Direct encroachment: {encroachment_ha:.4f} ha")

            # Clean up temp file
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

            return report

    except Exception as e:
        import traceback
        print(f"Error generating change detection: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Detect changes between two years of satellite imagery.')
    parser.add_argument('--old', type=int, default=2015, help='Older year')
    parser.add_argument('--new', type=int, default=2025, help='Newer year')
    parser.add_argument('--out_dir', type=str, help='Output directory for datasets')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path setup
    images_dir = os.path.join(args.out_dir if args.out_dir else os.path.join(base_dir, 'datasets', 'satellite_images'), 'images')
    output_dir = os.path.join(args.out_dir if args.out_dir else os.path.join(base_dir, 'datasets', 'satellite_images'), 'changes')
    
    image_old = os.path.join(images_dir, f'composite_{args.old}.tif')
    image_new = os.path.join(images_dir, f'composite_{args.new}.tif')

    print(f"Detecting changes between {args.old} and {args.new}...")
    detect_changes(image_old, image_new, output_dir, args.old, args.new)
