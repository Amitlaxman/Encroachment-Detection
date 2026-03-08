import ee
import geemap
import os

# Define the default Area of Interest: Warje, Pune
DEFAULT_AOI_COORDS = [73.7969, 18.4901] 
# Approximate bounds around Warje
DEFAULT_BBOX = [73.76, 18.47, 73.83, 18.52]

def initialize_ee():
    """Authenticates and initializes the Earth Engine API."""
    try:
        ee.Initialize()
    except Exception as e:
        print("Earth Engine not authenticated. Trying to authenticate...")
        ee.Authenticate()
        ee.Initialize()
    print("Earth Engine Initialized Successfully.")

def get_aoi(bbox=DEFAULT_BBOX):
    """Returns an Earth Engine geometry for the Area of Interest."""
    return ee.Geometry.BBox(*bbox)

def mask_s2_clouds(image):
    """Cloud masking function for Sentinel-2."""
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
             .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

def mask_l8_clouds(image):
    """Cloud masking function for Landsat 8."""
    qa = image.select('QA_PIXEL')
    dilatedCloud = 1 << 1
    cloud = 1 << 3
    cloudShadow = 1 << 4
    mask = qa.bitwiseAnd(dilatedCloud).eq(0) \
             .And(qa.bitwiseAnd(cloud).eq(0)) \
             .And(qa.bitwiseAnd(cloudShadow).eq(0))
    return image.updateMask(mask).multiply(0.0000275).add(-0.2)

def fetch_imagery(year, aoi):
    """
    Fetches the median composite image for a given year with robust fallback.
    Priority: Sentinel-2 (>2015) -> Landsat 8 (>2013) -> Landsat 5 (<2012) -> Landsat 7.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # 1. Try Sentinel-2 (2015+)
    if year >= 2015:
        s2_col = (ee.ImageCollection('COPERNICUS/S2')
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.listContains('system:band_names', 'QA60')))
        
        if s2_col.size().getInfo() > 0:
            try:
                composite = s2_col.map(mask_s2_clouds).median().clip(aoi)
                composite = composite.select(
                    ['B4', 'B3', 'B2', 'B8', 'B11'],
                    ['Red', 'Green', 'Blue', 'NIR', 'SWIR1']
                )
                return composite, "Sentinel-2"
            except Exception as e:
                print(f"Sentinel-2 parsing failed for {year}: {e}")

    # 2. Try Landsat 8 (2013+)
    if year >= 2013:
        l8_col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date))
                  
        if l8_col.size().getInfo() > 0:
            try:
                composite = l8_col.map(mask_l8_clouds).median().clip(aoi)
                composite = composite.select(
                    ['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5', 'SR_B6'],
                    ['Red', 'Green', 'Blue', 'NIR', 'SWIR1']
                )
                return composite, "Landsat 8"
            except Exception as e:
                print(f"Landsat 8 parsing failed for {year}: {e}")

    # 3. Try Landsat 5 (Up to 2011)
    if year <= 2011:
        l5_col = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date))
                  
        if l5_col.size().getInfo() > 0:
            try:
                composite = l5_col.map(mask_l8_clouds).median().clip(aoi)
                composite = composite.select(
                    ['SR_B3', 'SR_B2', 'SR_B1', 'SR_B4', 'SR_B5'],
                    ['Red', 'Green', 'Blue', 'NIR', 'SWIR1']
                )
                return composite, "Landsat 5"
            except Exception as e:
                print(f"Landsat 5 parsing failed for {year}: {e}")

    # 4. Ultimate Fallback: Landsat 7 (1999+)
    print(f"Falling back to Landsat 7 for {year}")
    l7_col = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date))
                  
    composite = l7_col.map(mask_l8_clouds).median().clip(aoi)
    composite = composite.select(
        ['SR_B3', 'SR_B2', 'SR_B1', 'SR_B4', 'SR_B5'],
        ['Red', 'Green', 'Blue', 'NIR', 'SWIR1']
    )
    return composite, "Landsat 7"

if __name__ == "__main__":
    initialize_ee()
    aoi = get_aoi()
    print("Testing 2010 imagery (Landsat 8):")
    img_2010, src_2010 = fetch_imagery(2010, aoi)
    print(f"Bands for 2010: {img_2010.bandNames().getInfo()}")
    
    print("Testing 2020 imagery (Sentinel-2):")
    img_2020, src_2020 = fetch_imagery(2020, aoi)
    print(f"Bands for 2020: {img_2020.bandNames().getInfo()}")
