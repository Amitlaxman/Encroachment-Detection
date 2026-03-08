import ee
from gee_pipeline import get_aoi, fetch_imagery

def calculate_ndwi(image, source="Sentinel-2"):
    """
    Calculates the Modified Normalized Difference Water Index (MNDWI)
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    Much better for urban and turbid water separation than NDWI.
    """
    # We rename to NDWI just to keep pipeline consistent downstream
    mndwi = image.normalizedDifference(['Green', 'SWIR1']).rename('NDWI')
    
    # A threshold of 0.0 is standard for MNDWI
    water_mask = mndwi.gt(0).rename('water_mask')
    
    return image.addBands([mndwi, water_mask])


def get_built_up_mask_esa(aoi, year):
    """
    Fetches ESA WorldCover map for a specific year and extracts the built-up areas.
    ESA WorldCover is available for 2020 and 2021. For other years, we could use proxies
    or Dynamic World (available 2015+), or NDBI for Landsat.
    For simplicity, let's use a proxy index NDBI if ESA is not available.
    """
    if year in [2020, 2021]:
        # ESA WorldCover: 50 is Built-up
        esa = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(aoi).first()
        if esa is not None:
             built_up = esa.select('Map').eq(50).rename('built_up_mask')
             return built_up

    # If year < 2015 or ESA not available, return None so we can use NDBI
    return None

def calculate_ndbi(image):
    """
    Calculates Normalized Difference Built-up Index (NDBI)
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    """
    ndbi = image.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI')
    
    # NDBI > 0.0 often indicates built-up, but thresholding can be tricky.
    # We use > 0.1 as a stricter threshold for built-up to reduce false positives.
    built_up_mask = ndbi.gt(0.1).rename('built_up_mask')
    
    return image.addBands([ndbi, built_up_mask])


def process_year(year, aoi):
    """
    Fetches imagery, calculates indices, and returns the processed image.
    """
    img, src = fetch_imagery(year, aoi)
    
    # 1. Water Mask
    img = calculate_ndwi(img, src)
    
    # 2. Built-up Mask
    esa_mask = get_built_up_mask_esa(aoi, year)
    if esa_mask is not None:
        img = img.addBands(esa_mask)
    else:
        # Fallback to NDBI
        img = calculate_ndbi(img)
        
    return img, src
