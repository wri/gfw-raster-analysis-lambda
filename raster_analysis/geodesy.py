import math

import numpy as np


def get_area(lat: float, pixel_width: float) -> float:
    """
    Calculate geodesic area for Hansen data, assuming a fix pixel size of 0.00025 * 0.00025 degree
    using WGS 1984 as spatial reference.
    Pixel size various with latitude, which is it the only input parameter.
    """
    a = 6378137.0  # Semi major axis of WGS 1984 ellipsoid
    b = 6356752.314245179  # Semi minor axis of WGS 1984 ellipsoid

    d_lat = pixel_width
    d_lon = pixel_width

    pi = math.pi

    q = d_lon / 360
    e = math.sqrt(1 - (b / a) ** 2)

    area = (
        abs(
            (
                pi
                * b ** 2
                * (
                    2 * np.arctanh(e * np.sin(np.radians(lat + d_lat))) / (2 * e)
                    + np.sin(np.radians(lat + d_lat))
                    / (
                        (1 + e * np.sin(np.radians(lat + d_lat)))
                        * (1 - e * np.sin(np.radians(lat + d_lat)))
                    )
                )
            )
            - (
                pi
                * b ** 2
                * (
                    2 * np.arctanh(e * np.sin(np.radians(lat))) / (2 * e)
                    + np.sin(np.radians(lat))
                    / (
                        (1 + e * np.sin(np.radians(lat)))
                        * (1 - e * np.sin(np.radians(lat)))
                    )
                )
            )
        )
        * q
    )

    return area
