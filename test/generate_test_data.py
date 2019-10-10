import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import os
import os.path


def generate_test_data():
    """
    Generate rasters in the test_data folder to use for unit testing
    """
    if not os.path.exists("test_data"):
        os.mkdir("test_data")

    # keep it small grid we can easily look at in NumPy Array Viewer for debugging
    base = np.zeros((40, 20)).astype(np.uint8)

    # generate NumPy arrays for different layers we want
    analysis_layer = create_analysis_layer(np.copy(base))
    contextual_layer_1, contextual_layer_2 = create_contextual_layers(base)
    filter_layer = create_filter_layer(np.copy(base))
    sum_layer = create_sum_layer(np.copy(analysis_layer).astype(np.float))

    # generate rasters from NumPy arrays
    create_test_raster("analysis_layer", analysis_layer)
    create_test_raster("contextual_layer_1", contextual_layer_1)
    create_test_raster("contextual_layer_2", contextual_layer_2)
    create_test_raster("filter_layer", filter_layer)
    create_test_raster("sum_layer", sum_layer)


def create_test_raster(name, arr):
    # arbitrary location somewhere in brazil
    transform = from_origin(-60, -10, 0.00025, 0.00025)

    dataset = rasterio.open(
        "test_data/" + name + ".tif",
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=0,
    )
    dataset.write(arr, 1)
    dataset.close()


def create_analysis_layer(base):
    """
    Dot small square of values 1-4 around grid to emulate something like loss layer
    :param base: base grid to create layer from
    :return: analysis layer
    """
    analysis_base = np.array([[1, 2], [3, 4]]).astype(np.uint8)

    base[9:11, 4:6] += analysis_base
    base[14:16, 4:6] += analysis_base
    base[24:26, 4:6] += analysis_base
    base[29:31, 4:6] += analysis_base

    base[9:11, 14:16] += analysis_base
    base[14:16, 14:16] += analysis_base
    base[24:26, 14:16] += analysis_base
    base[29:31, 14:16] += analysis_base

    base[14:16, 9:11] += analysis_base
    base[19:21, 14:16] += analysis_base

    return base


def create_contextual_layers(base):
    """
    Create two contextual layers of large rectangles overlapping some of the analysis squares,
    and overlapping each other and an analysis square to get interesting intersection results
    :param base: base grid to create layer from
    :return: 2-tuple of contextual layers
    """
    contextual_layer_1 = np.copy(base)
    contextual_base = np.ones((11, 4)).astype(np.uint8)
    contextual_layer_1[7:18, 3:7] += contextual_base
    contextual_layer_1[22:33, 13:17] += contextual_base

    contextual_layer_2 = np.copy(base)
    contextual_layer_2[13:17, 3:17] += np.ones((4, 14)).astype(np.uint8)

    return contextual_layer_1, contextual_layer_2


def create_filter_layer(base):
    """
    Create filter layer consisting of a rectangle overlapping both contextual layers, as well as analysis squares
    both inside and outside the contextual layers
    :param base: base grid to create layer from
    :return: filter layer
    """
    base += 10
    beneath_threshold = (np.ones((16, 4)).astype(np.uint8)) * 5
    base[12:28, 13:17] -= beneath_threshold

    return base


def create_sum_layer(base):
    """
    Create sum layer with float values overlapping the analysis layer values

    :param base: base analysis layer to create layer from
    :return: sum layer
    """
    base[base == 1] = 0.5
    base[base == 2] = 1.5
    base[base == 3] = 2.5
    base[base == 4] = 3.5

    return base
