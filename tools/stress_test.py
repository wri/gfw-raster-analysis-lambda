import asyncio
import json
import random
from typing import Coroutine, List

import asyncclick as click
import httpx
from shapely import to_wkt
from shapely.geometry import box, mapping, shape

LAMBDA_NAME = "raster-analysis-tiled_raster_analysis-default"
API_URI = "https://data-api.globalforestwatch.org/analysis/zonal"


@click.command()
@click.option(
    "--api_key", type=str, required=True, help="GFW Data API key in the target environment"
)
@click.option(
    "--requests", type=int, required=True, help="Number of concurrent requests to make"
)
@click.option(
    "--size",
    type=int,
    required=True,
    help="Size of each request, in number of 1x1 degree tiles.",
)
@click.option(
    "--seed",
    type=int,
    required=False,
    help="Random seed, if desired",
)
async def stress_test(api_key, requests, size, seed):
    if seed is not None:
        random.seed(seed)

    geoms: box = [get_random_box(size) for _ in range(0, requests)]
    print(f"Getting stats for the following geoms: ")
    for geom in geoms:
        print(geom.wkt)

    print("\nSending requests...")

    futures: List[Coroutine] = []

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    async with httpx.AsyncClient(timeout=60, headers=headers) as client:
        for geom in geoms:
            payload = {
                "geometry": mapping(geom),
                "group_by": ["umd_tree_cover_loss__year"],
                "filters": [
                    "is__umd_regional_primary_forest_2001",
                    "umd_tree_cover_density_2000__30",
                ],
                "sum": ["area__ha"],
            }

            futures.append(client.post(
                API_URI,
                json=payload
            ))

        responses: tuple[httpx.Response] = await asyncio.gather(*futures)

    for response in responses:
        geojson = json.loads(response.request.content)["geometry"]
        if response.status_code != 200 or response.json()["status"] != "success":
            print(f"ERROR on {to_wkt(shape(geojson))}: {response.content.decode('utf-8')}")
        else:
            print(f"SUCCESS on {to_wkt(shape(geojson))} in {response.elapsed}")

    print("\nDONE")


def get_random_box(size: int) -> box:
    region = random.randint(0, 2)

    if region == 0:  # South America
        bottom_left_x = random.randint(-70, -50 - size)
        bottom_left_y = random.randint(-20, 0 - size)
    elif region == 1:  # Africa
        bottom_left_x = random.randint(10, 40 - size)
        bottom_left_y = random.randint(-15, 15 - size)
    else:  # Southeast Asia
        bottom_left_x = random.randint(93, 108 - size)
        bottom_left_y = random.randint(-5, 22 - size)

    return box(
        bottom_left_x,
        bottom_left_y,
        bottom_left_x + size,
        bottom_left_y + size
    )


if __name__ == "__main__":
    stress_test(_anyio_backend="asyncio")
