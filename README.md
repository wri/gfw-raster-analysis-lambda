# GFW raster analsyis in AWS Lambda

### Functionality

Run zonal statistics on any given combination of rasters for a given geometry, returning two serialized pandas DataFrames:

(1) A summary table providing the analysis aggregated across the entire geometry
(2) A detailed table broken down by year and only include only NoData results masked by the filter

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
|analysis_raster_id| String | Primary raster to base analysis off of. Any NoData pixels in this raster will be ignored.| "loss" |
|contextual_raster_ids| [String] | List of rasters to contextualize analysis. Analysis results will be aggregated by unique combinations of contextual and analysis raster values. | ["wdpa"] |
|analyses| [String] | Analyses to be performed | One or more of: `area`, `sum`, `count` |
|geometry| Object | A valid GeoJSON geometry (see further specification in `Limitations and assumtions` | `"geometry": {"type": "Polygon", "coordinates": [[[9, 4.1], [9.1, 4.1], [9.1, 4.2], [9, 4.2], [9, 4.1]]],}`|
|filters| [Object] | A list of rasters to use a mask on the geometry during analysis. Includes the raster id and a threshold value. | `[{"raster_id": "tcd_2000", "threshold": 30}]`|
|aggregate_rasters_ids| [String] | List of value rasters to aggregate. Certain analyses (currently only `sum`) will aggregate the pixel values of these rasters across unique combinations of contextual layers. | ["tcd_2000", "tcd_2010"] |

#### Examples

Request:
```python
import requests
import json

url = "https://hjebg1jly1.execute-api.us-east-1.amazonaws.com/default/gfw-raster-analysis"

payload = {
        "analysis_raster_id": "loss",
        "contextual_raster_ids": ["wdpa"],
        "analyses": ["count", "area"],
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
    "detailed_table": {
      "loss": {
        "0": 1,
        "1": 1,
        "2": 2,
        "3": 2,
        "4": 3,
        "5": 3,
        "6": 5,
        "7": 5,
        "8": 6,
        "9": 6,
        "10": 7,
        "11": 8,
        "12": 8,
        "13": 9,
        "14": 10,
        "15": 10,
        "16": 11,
        "17": 11,
        "18": 12,
        "19": 13,
        "20": 13,
        "21": 14,
        "22": 14,
        "23": 15,
        "24": 15,
        "25": 16,
        "26": 17,
        "27": 17,
        "28": 18,
        "29": 18
      },
      "wdpa": {
        "0": 0,
        "1": 1,
        "2": 0,
        "3": 1,
        "4": 0,
        "5": 1,
        "6": 0,
        "7": 1,
        "8": 0,
        "9": 1,
        "10": 1,
        "11": 0,
        "12": 1,
        "13": 1,
        "14": 0,
        "15": 1,
        "16": 0,
        "17": 1,
        "18": 0,
        "19": 0,
        "20": 1,
        "21": 0,
        "22": 1,
        "23": 0,
        "24": 1,
        "25": 0,
        "26": 0,
        "27": 1,
        "28": 0,
        "29": 1
      },
      "count": {
        "0": 137,
        "1": 72,
        "2": 997,
        "3": 9,
        "4": 84,
        "5": 56,
        "6": 55,
        "7": 5,
        "8": 877,
        "9": 58,
        "10": 3,
        "11": 23,
        "12": 65,
        "13": 12,
        "14": 2008,
        "15": 427,
        "16": 425,
        "17": 39,
        "18": 2,
        "19": 168,
        "20": 36,
        "21": 288,
        "22": 957,
        "23": 7,
        "24": 15,
        "25": 115,
        "26": 628,
        "27": 66,
        "28": 597,
        "29": 29
      },
      "area": {
        "0": 10.539639042600312,
        "1": 5.539080372753449,
        "2": 76.7008768282665,
        "3": 0.6923850465941811,
        "4": 6.462260434879024,
        "5": 4.308173623252682,
        "6": 4.231241951408885,
        "7": 0.3846583592189895,
        "8": 67.46907620701076,
        "9": 4.462036966940278,
        "10": 0.2307950155313937,
        "11": 1.7694284524073516,
        "12": 5.000558669846863,
        "13": 0.9231800621255748,
        "14": 154.4787970623462,
        "15": 32.8498238773017,
        "16": 32.695960533614105,
        "17": 3.000335201908118,
        "18": 0.1538633436875958,
        "19": 12.924520869758048,
        "20": 2.7695401863767244,
        "21": 22.156321491013795,
        "22": 73.62360995451459,
        "23": 0.5385217029065853,
        "24": 1.1539750776569684,
        "25": 8.847142262036758,
        "26": 48.31308991790508,
        "27": 5.077490341690662,
        "28": 45.92820809074735,
        "29": 2.231018483470139
      }
    },
    "summary_table": {
      "wdpa": {
        "0": 0,
        "1": 1
      },
      "count": {
        "0": 33860,
        "1": 126140
      },
      "area": {
        "0": 2604.9064086309972,
        "1": 9704.16108637667
      },
      "filtered_area": {
        "0": 2594.1359745728655,
        "1": 9180.717991151465
      },
      "loss_area": {
        "0": 493.2089481905882,
        "1": 142.24666123918232
      }
    }
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

