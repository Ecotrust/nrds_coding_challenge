import io
import numpy as np
import requests
from imageio import imread


def naip2016_from_orgeo(bbox, width, height, epsg=5070, num_retries=3, **kwargs):
    """
    Retrieves a 2016 NAIP image from Oregon Geospatial Enterprise Office.

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    width, height : int
      width and height (in pixels) of image to be returned
    epsg : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    img : array
      NAIP image as a 3-band or 4-band array
    """
    BASE_URL = ''.join([
        'https://imagery.oregonexplorer.info/arcgis/rest/services/'
        'NAIP_2016/NAIP_2016_SL/ImageServer/exportImage?'
    ])
    params = dict(
        bbox=','.join([str(x) for x in bbox]),
        bboxSR=epsg,
        size=f'{width},{height}',
        imageSR=epsg,
        format='tiff',
        pixelType='U8',
        noData=None,
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression=None,
        compressionQuality=None,
        bandIds=None,
        mosaicRule=None,
        renderingRule=None,
        f='image'
    )
    for key, value in kwargs.items():
        params.update({key: value})

    img = None
    retries = num_retries

    while retries > 0 and img is None:
        try:
            r = requests.get(BASE_URL, params=params)
            img = imread(io.BytesIO(r.content))
        except:
            retries -= 1

    if img is None:
        print(f'Failed to get image after {num_retries} retries.')
        return
    else:
        return img


def nlcd_from_mrlc(bbox, width, height,
                   layer='NLCD_2016_Land_Cover_L48',
                   epsg=5070,
                   **kwargs):
    """
    Retrieves National Land Cover Data (NLCD) Layers from the Multiresolution
    Land Characteristics Consortium's web service, with NLCD cover types mapped
    to four simplified classes: 1) water or ice; 2)

    Parameters
    ----------
    bbox : list-like
      list of bounding box coordinates (minx, miny, maxx, maxy)
    width, height : int
      width and height (in pixels) of image to be returned
    layer : str
      title of layer to retrieve (e.g., 'NLCD_2001_Land_Cover_L48')
    epsg : int
      spatial reference for bounding box, such as an EPSG code (e.g., 4326)

    Returns
    -------
    img : numpy array
      map image as array
    """
    BASE_URL = ''.join([
        'https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2016_Land_Cover_L48/',
        'wms?service=WMS&request=GetMap',
    ])

    params = dict(bbox=','.join([str(x) for x in bbox]),
                  crs=f'epsg:{epsg}',
                  width=width,
                  height=height,
                  format='image/geotiff',
                  layers=layer)
    for key, value in kwargs.items():
        params.update({key: value})

    r = requests.get(BASE_URL, params=params)
    img = imread(io.BytesIO(r.content), format='tiff')

    REMAP = {
            1: 1,  # open water
            2: 1,  # perennial ice/snow
            3: 5,  # developed, open space
            4: 5,  # developed, low intensity
            5: 5,  # developed, medium intensity
            6: 5,  # developed, high intensity
            7: 4,  # barren land (rock/stand/clay)
            8: 4,  # unconsolidated shore
            9: 2,  # deciduous forest
            10: 2,  # evergreen forest
            11: 2,  # mixed forest
            12: 3,  # dwarf scrub (AK only)
            13: 3,  # shrub/scrub
            14: 3,  # grasslands/herbaceous,
            15: 3,  # sedge/herbaceous (AK only)
            16: 3,  # lichens (AK only)
            17: 3,  # moss (AK only)
            18: 3,  # pasture/hay
            19: 3,  # cultivated crops
            20: 2,  # woody wetlands
            21: 3,  # emergent herbaceous wetlands
        }
    k = np.array(list(REMAP.keys()), dtype=np.uint8)
    v = np.array(list(REMAP.values()), dtype=np.uint8)

    mapping_ar = np.zeros(k.max() + 1, dtype=np.uint8)
    mapping_ar[k] = v
    img = mapping_ar[img]

    return img


def colorize_landcover(img):
    """Assigns colors to a land cover map.

    Parameters
    ----------
    img : arr, shape (H, W) or (H, W, num_classes)
      a HxW array with acceptable integer values of {0, 1, 2, 3, 4, 5, 255}.

    Returns
    -------
    land_color : arr, shape (H, W, 3)
      RGB image of land cover types
    """
    COLOR_MAP = {
        0: [1.0, 1.0, 1.0],  # unlabeled but mapped
        1: [0.0, 0.0, 1.0],  # water or ice/snow
        2: [0.0, 0.5, 0.0],  # trees
        3: [0.5, 1.0, 0.5],  # non-forest, vegetated
        4: [0.5, 0.375, 0.375],  # barren/non-vegetated
        5: [0.0, 0.0, 0.0],  # developed/building
        255: [1.0, 0.0, 0.0]  # unmapped, nodata
    }
    cover_colors = np.zeros((img.shape[0], img.shape[1], 3))

    for cov in np.unique(img):
        mask = img == cov
        cover_colors[mask] = COLOR_MAP[cov]

    land_color = (cover_colors * 255).astype(np.uint8)

    return land_color
