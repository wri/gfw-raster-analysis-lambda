# GFW raster analysis in AWS Lambda

### Functionality

Run raster zonal analysis on data in [a gfw-data-api](https://github.com/gfw-api/gfw-data-api). Use the lambda to run on one geometry, or the step function to run on a list of geometries. Supported analyses:

- Zonal statistics across multiple layers at once, including masks and grouping
- Pulling point data in a zone, including latitude and longitude

See **Raster SQL** for how to query datasets.

### Query Parameters

|Parameter|Type|Description|
|---------|----|-----------|
|geometry (required)| GeoJSON | A valid GeoJSON geometry to run analysis on. Must be a Polygon or MultiPolygon. |
|sql (required)| String | A **Raster SQL** query string for the analysis you want to run. See below for more details.| SELECT SUM(area__ha), umd_tree_cover_loss__year FROM data WHERE umd_tree_cover_density__percent > 30 GROUP BY umd_tree_cover_loss__year |
|data_environment (required) | Dict | A config telling the raster analysis how to match layer names in the query with the actual source. This is typically created by [a gfw-data-api](https://github.com/gfw-api/gfw-data-api) automatically using layers in the API.|
#### Examples

Request:
```JSON
{
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[9, 4.1], [9.1, 4.1], [9.1, 4.2], [9, 4.2], [9, 4.1]]],
    },
    "sql": "SELECT umd_tree_cover_loss__year, SUM(area__ha), SUM(whrc_aboveground_co2_emissions__Mg) FROM data WHERE umd_tree_cover_density_2000__percent > 30 GROUP BY umd_tree_cover_loss__year
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

### Assumptions and limitations

GFW raster tiles are organized in 10 x 10 Degree grids and have a pixel size of 0.00025 x 0.00025 degree.
They are saved as Cloud Optimized TIFFs with 400 x 400 pixels blocks.

Because we can scale up parallel processing with lambda, size of the geometry shouldn't be an issue unless getting to massive scales (> 1 billion ha). But each lambda has in-memory cap of 3 GB, so currently only so many rasters can be loaded into memory at once. The limit depends on the size of the raster values (e.g. binary is way less memory than float), but generally max 4 or 5 raster layers is a good rule of thumb.

To optimize speed for area calculations, we assume each tile has roughly similar pixel area, and only use the pixel area of the centroid pixel for calculations. This may introduce some inaccuracies of the tile is sparsely covered. 

## Deployment

Use terraform:

```
./scripts/infra plan
./scripts/infra apply
```

```
Runtime: Python 3.10
Handler: lambda_function.lambda_handler
```
