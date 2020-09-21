import random
import asyncio
import http3
from shapely.geometry import box, mapping
import asyncclick as click

LAMBDA_NAME = "raster-analysis-tiled_raster_analysis-default"
API_URI = "https://staging-data-api.globalforestwatch.org/analysis/zonal"
GEOSTORE_URI = "http://staging-api.globalforestwatch.org/geostore"
PROFILE = "gfw-staging"


@click.command()
@click.option(
    "--requests", type=int, required=True, help="Number of concurrent requests to make"
)
@click.option(
    "--size",
    type=int,
    required=True,
    help="Size of each request, in number of 1x1 degree tiles.",
)
async def stress_test(requests, size):
    geoms = [get_random_box(size) for i in range(0, requests)]
    for geom in geoms:
        print(geom.wkt)

    response_futures = []

    async with http3.AsyncClient(timeout=60) as client:
        for geom in geoms:
            payload = {
                "geometry": mapping(geom),
                "geostore_origin": "rw",
                "group_by": ["umd_tree_cover_loss__year"],
                "filters": [
                    "is__umd_regional_primary_forest_2001",
                    "umd_tree_cover_density_2000__30",
                ],
                "sum": ["area__ha"],
            }

            print("Sending request...")

            response_futures.append(client.post(API_URI, json=payload))

        responses = await asyncio.gather(*response_futures)

    for response in responses:
        if response.status_code != 200:
            print("ERROR")
            print(geom)
            print(response.content)

        response = response.json()
        if response["status"] != "success":
            print("ERROR")
            print(geom)
            print(response)

        print("SUCCESS")


def get_random_box(size):
    region = random.randint(0, 2)

    if region == 0:  # South America
        bottom_left_x = random.randint(-70, -50 - size)
        bottom_left_y = random.randint(-20, 0 - size)
    if region == 1:  # Africa
        bottom_left_x = random.randint(10, 40 - size)
        bottom_left_y = random.randint(-15, 15 - size)
    if region == 2:  # Southeast Asia
        bottom_left_x = random.randint(93, 108 - size)
        bottom_left_y = random.randint(-5, 22 - size)

    return box(bottom_left_x, bottom_left_y, bottom_left_x + size, bottom_left_y + size)


if __name__ == "__main__":
    stress_test(_anyio_backend="asyncio")
