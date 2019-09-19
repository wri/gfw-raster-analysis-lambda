# GFW raster analsyis in AWS Lambda

### Functionality

Run zonal statistics on any given combination of rasters for a given geometry.

This function produces the following statistics:

#### Count pixels
Counts number of pixels for unique value combinations of the given input bands inside the given geometry.

#### Calculate area
Calculates the geodesic area for unique value combinations of the given input bands inside the given geometry.
Precision is best for smaller geometries. The function currently calculates the mean area for the input geometry
and multiplies this area with the pixel count for unique value combinations.
Use the Sum function together with an area raster for more precise results.

#### Sum values
Calculates the sum of pixel values for any given raster layer with `Float` datatype for unique value combinations
of the given input bands with `Integer` datatype inside the given geometry.


### Input Parameters

|Parameter|Type|Description|Example|
|---------|----|-----------|-------|
|raster_ids| [String] | List of raster ids to be used for analysis | ["loss", "wdpa"] |
|analysis| String | Analysis to be performed | One of: `area`, `sum`, `count` |
|geometry| Object | A valid GeoJSON geometry (see further specification in `Limitations and assumtions` | `"geometry": {"type": "Polygon", "coordinates": [[[9, 4.1], [9.1, 4.1], [9.1, 4.2], [9, 4.2], [9, 4.1]]],}`|
|filters| [Object] | A list of rasters to use a mask on the geometry during analysis. Includes the raster id and a threshold value. | `[{"raster_id": "tcd_2000", "threshold": 30}]`|

#### Examples

Request:
```python
import requests
import json

url = "https://hjebg1jly1.execute-api.us-east-1.amazonaws.com/default/gfw-raster-analysis"

payload = {
        "raster_ids": ["loss", "wdpa"],
        "analysis": "area",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[9, 4.1], [9.1, 4.1], [9.1, 4.2], [9, 4.2], [9, 4.1]]],
        },
        "filters": [
            {
                "raster_id": "tcd_2000",
                "threshold": 30
            }
        ]
    }

headers = {
    "Content-Type": "application/json",
    }

response = requests.request("POST", url, data=json.dumps(payload), headers=headers)

```

Response:
```JSON
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
        "extent": 11830.244769451865,
        "threshold": 30,
        "data": [
            [0, 0, 2106.3891750831863],
            [0, 1, 9088.39998493891],
            [1, 0, 10.539639042600312],
            [1, 1, 5.539080372753449],
            [2, 0, 76.7008768282665],
            [2, 1, 0.6923850465941811],
            [3, 0, 6.462260434879024],
            [3, 1, 4.308173623252682],
            [5, 0, 4.231241951408885],
            [5, 1, 0.3846583592189895],
            [6, 0, 67.46907620701076],
            [6, 1, 4.462036966940278],
            [7, 1, 0.2307950155313937],
            [8, 0, 1.7694284524073516],
            [8, 1, 5.000558669846863],
            [9, 1, 0.9231800621255748],
            [10, 0, 154.4787970623462],
            [10, 1, 32.8498238773017],
            [11, 0, 32.695960533614105],
            [11, 1, 3.000335201908118],
            [12, 0, 0.1538633436875958],
            [13, 0, 12.924520869758048],
            [13, 1, 2.7695401863767244],
            [14, 0, 22.156321491013795],
            [14, 1, 73.62360995451459],
            [15, 0, 0.5385217029065853],
            [15, 1, 1.1539750776569684],
            [16, 0, 8.847142262036758],
            [17, 0, 48.31308991790508],
            [17, 1, 5.077490341690662],
            [18, 0, 45.92820809074735],
            [18, 1, 2.231018483470139],
        ],
        "dtype": [["loss", "|u1"], ["wdpa", "|u1"], ["AREA", "<f8"]],
    }
}

```


### Endpoints

https://hjebg1jly1.execute-api.us-east-1.amazonaws.com/default/gfw-raster-analysis

Max memory: 3GB


### Assumptions and limitations

This function works best with smaller geometries and lower number of input rasters.
Maximum number of rasters you will be able to process at oncedepends on available memory of Lambda function (up to 3GB),
data type of rasters and size of your geometry.

To speed up requests, break down your geometry into smaller blocks and send them as multiple requests,
allowing AWS Lambda to process them in parallel.

GFW raster tiles are organized in 10 x 10 Degree grids and have a pixel size of 0.00025 x 0.00025 degree.
They are saved as Cloud Optimized TIFFs with 400 x 400 pixels blocks.
Ideally, input geometries width and height should be a multiple of 400 pixel and align with 0.1 x 0.1 degree grid
to minimize read requests to S3. Geometries must always be inside a 10 x 10 degree grid cell.

The current function does not try to correct geometries or optimize requests.
This efforts will still need to be done on the client side.

## Deployment

We need to package our lambda function together with its dependencies. We can do this inside Docker:

Build Docker image

`docker build -t gfw/raster-analysis-lambda .`

Copy lambda package from Docker file

```
docker run --name lambda -itd gfw/raster-analysis-lambda /bin/bash
docker cp lambda:/tmp/package.zip package.zip
docker stop lambda
docker rm lambda
```


Deploy lambda function by uploading ZIP archive to AWS using cli, console or any other deployment method.
To enable HTTP requests, you will need to add the lambda function to an API Gateway.

When deploying the function use the following settings:

```
Runtime: Python 3.6
Handler: lambda_function.lambda_handler
```

