# GFW raster analsyis in AWS Lambda

### Functionality

Run zonal statistics on tree cover loss, GLAD alerts, or arbitrary contextual layers defined in our data lake.

See [a gfw-data-api](https://github.com/gfw-api/gfw-data-api) for how to access through analysis API.


### Query Parameters

All layers should be referred to by their standard data lake column name: <data lake layer name>__<data type> or is__<data lake layer name> for boolean layers.

See the data API for a full list of registered layers.


|Parameter|Type|Description|Example|
|---------|----|-----------|-------|
|geostore_id (required)| String | A valid geostore ID containing the GeoJSON for the geometry of interest (see further specification in `Limitations and assumtions` | cb64960710fd11925df3fda6a2005ca9 |
|group_by| [String] | Rasters with categorical pixel values to group by.| umd_tree_cover_loss__year, umd_glad_alerts, tsc_tree_cover_loss_drivers__type |
|filters| [String] | Rasters to apply as a mask. Pixels with NoData values will be filtered out of the final result. For umd_tree_cover_density_2000/2010, you can put a threshold number as the data type, and it will apply a filter for that threshold|  is__umd_regional_primary_forest_2001, umd_tree_cover_density_2000__30|
|sum| [String] | Pixel values will be summed based on intersection with group_by layers. If there are no group_by layers, all pixel values will be summed to a single number. Pixel value must be numerical. This field can also include area__ha or alert__count, which will give the pixel count or area.| area__ha, whrc_aboveground_co2_emissions__Mg |
|start| Date | Filters date group_by columns to this start date. Must be a year or a YYYY-MM-DD formatted date. | 2015, 2015-02-04 |
|end| Date | Same format as 'start'. Must come after start. | 2016 , 2016-02-10 |
#### Examples

Request:
```JSON
{
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[9, 4.1], [9.1, 4.1], [9.1, 4.2], [9, 4.2], [9, 4.1]]],
    },
    "group_by": ["umd_tree_cover_loss__year"],
    "filters": [
        "is__umd_regional_primary_forest_2001",
        "umd_tree_cover_density_2000__30",
    ],
    "sum": ["area__ha", "whrc_aboveground_co2_emissions__Mg"],
}
```

Response:
```JSON
{
   "status":"success",
   "data":[
      {
         "umd_tree_cover_loss__year":2001,
         "area__ha":9.894410216509604,
         "whrc_aboveground_co2_emissions__Mg":3560.875476837158
      },
      {
         "umd_tree_cover_loss__year":2002,
         "area__ha":40.0378459923877,
         "whrc_aboveground_co2_emissions__Mg":14713.026161193848
      },
      {
         "umd_tree_cover_loss__year":2003,
         "area__ha":6.442871768889975,
         "whrc_aboveground_co2_emissions__Mg":2568.1107501983643
      },
      {
         "umd_tree_cover_loss__year":2005,
         "area__ha":3.2214358844449875,
         "whrc_aboveground_co2_emissions__Mg":1274.5636539459229
      },
      {
         "umd_tree_cover_loss__year":2006,
         "area__ha":22.01314521037408,
         "whrc_aboveground_co2_emissions__Mg":8167.388116836548
      },
      {
         "umd_tree_cover_loss__year":2007,
         "area__ha":0.23010256317464195,
         "whrc_aboveground_co2_emissions__Mg":136.68091201782227
      },
      {
         "umd_tree_cover_loss__year":2008,
         "area__ha":3.7583418651858187,
         "whrc_aboveground_co2_emissions__Mg":1579.5646076202393
      },
      {
         "umd_tree_cover_loss__year":2009,
         "area__ha":0.7670085439154732,
         "whrc_aboveground_co2_emissions__Mg":226.95782279968262
      },
      {
         "umd_tree_cover_loss__year":2010,
         "area__ha":108.37830725525636,
         "whrc_aboveground_co2_emissions__Mg":41855.43841171265
      },
      {
         "umd_tree_cover_loss__year":2011,
         "area__ha":12.88574353777995,
         "whrc_aboveground_co2_emissions__Mg":4887.8897132873535
      },
      {
         "umd_tree_cover_loss__year":2012,
         "area__ha":0.07670085439154732,
         "whrc_aboveground_co2_emissions__Mg":23.061389923095703
      },
      {
         "umd_tree_cover_loss__year":2013,
         "area__ha":1.6107179422224938,
         "whrc_aboveground_co2_emissions__Mg":601.4241733551025
      },
      {
         "umd_tree_cover_loss__year":2014,
         "area__ha":54.30420490921551,
         "whrc_aboveground_co2_emissions__Mg":22433.24832725525
      },
      {
         "umd_tree_cover_loss__year":2015,
         "area__ha":0.3068034175661893,
         "whrc_aboveground_co2_emissions__Mg":119.5254955291748
      },
      {
         "umd_tree_cover_loss__year":2016,
         "area__ha":5.752564079366049,
         "whrc_aboveground_co2_emissions__Mg":2075.9469604492188
      },
      {
         "umd_tree_cover_loss__year":2017,
         "area__ha":24.774375968469784,
         "whrc_aboveground_co2_emissions__Mg":9848.338472366333
      },
      {
         "umd_tree_cover_loss__year":2018,
         "area__ha":29.75993150392036,
         "whrc_aboveground_co2_emissions__Mg":11987.563570022583
      },
      {
         "umd_tree_cover_loss__year":2019,
         "area__ha":27.382205017782393,
         "whrc_aboveground_co2_emissions__Mg":10558.882364273071
      }
   ]
}
```


### Endpoints
```http request
https://staging-data-api.globalforestwatch.org/analysis
https://data-api.globalforestwatch.org/analysis
```


### Assumptions and limitations

GFW raster tiles are organized in 10 x 10 Degree grids and have a pixel size of 0.00025 x 0.00025 degree.
They are saved as Cloud Optimized TIFFs with 400 x 400 pixels blocks.

## Deployment

Use terraform:

```
./scripts/infra plan
./scripts/infra apply
```

```
Runtime: Python 3.7
Handler: lambda_function.lambda_handler
```

