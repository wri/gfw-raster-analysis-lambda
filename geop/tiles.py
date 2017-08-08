from PIL import Image
from io import BytesIO
import numpy as np

from geop.geo_utils import tile_read
from geop.geoprocessing import reclassify_from_data, weighted_overlay_from_data


def render_tile(geom, raster_path, user_palette=None):
    """
    Generates a visual PNG map tile from a vector polygon

    Args:
        geom (Shapely Geometry): A polygon corresponding to a TMS tile
            request boundary

        raster_path (string): A local file path to a raster in EPSG:3857
            to generate visual tile from

        uer_palette (optional list): A sequence of RGB triplets whose index
            corresponds to the raster value which will be rendered. If
            provided, will override a ColorTable defined in the raster
    Returns:
        Byte Array of image in the PNG format
    """
    tile, palette = tile_read(geom, raster_path)
    return render_tile_from_data(tile, user_palette or palette)


def render_tile_from_data(tile, palette):
    """
    Generates a visual PNG map tile from an ndarray of raster data

    Args:
        tile (ndarray) : A square 256x256 array of raster values

        palette (list<int>): A list of RGB values to render `tile`

    Returns:
        Byte Array of image in the PNG format
    """
    img = Image.fromarray(tile, mode='P')

    if len(palette):
        img.putpalette(palette)

    img_data = BytesIO()
    img.save(img_data, 'png')
    img_data.seek(0)
    return img_data


def weighted_overlay_tile(bbox, urban, forest, nlcd_path, soil_path):

    # Decimated read for the bbox of each layer.
    nlcd_tile, _ = tile_read(bbox, nlcd_path)
    soil_tile, _ = tile_read(bbox, soil_path)

    # Reclassify both the nlcd and soils data sets into a priority map of
    # 0 (low) to 10 (high) normalized values.  For example, NLCD 21-24 are
    # highly impervious and are rated as a 10, where 42-43 are forested and
    # marked a low priority
    nlcd_reclass = [[11, 0], [(21, 24), urban], [31, 7], [(41, 43), forest],
                    [(51, 52), 6], [(71, 74), 4], [(81, 82), 5], [(90, 95), 2]]

    # Soil values aren't linearly worse, 3&4 have the slowest infiltration,
    # followed by 6&7.  Ordering is important so a reclassed value doesn't
    # get reclassed again by a subsequent rule
    soil_reclass = [[255, 0], [3, 8], [4, 10], [(6, 7), 8], [5, 6], [2, 5]]

    nlcd_priority = reclassify_from_data(nlcd_tile, nlcd_reclass)
    soil_priority = reclassify_from_data(soil_tile, soil_reclass)

    # Use the two relative priority layers and weight them, giving the NLCD
    # layer more weight when determining an overall priority map.  The result
    # is a layer with values of 0 - 10 that identifies areas more in need
    # of Green Stormwater Infrastructure projects, based on the defined
    # scores and preferences.
    layers = [nlcd_priority, soil_priority]
    weights = [0.65, 0.35]
    priority = weighted_overlay_from_data(layers, weights)

    # The weighted overlay will produce floats, but for rendering purposes we
    # can round to ints so that we can create a straightforward palette
    priority_rounded = priority.astype(np.uint8)

    # A color palette from 0 (white -> green) to 10 (red) for our overall
    # site priorities
    palette = [255,255,255, 0,104,55, 26,152,80, 102,189,99, 166,217,106, 217,239,139, 254,224,139, 253,174,97, 244,109,67, 215,48,39, 165,0,38]  # noqa

    # Render the image tile for this priority map with the new palette
    return render_tile_from_data(priority_rounded, palette)
