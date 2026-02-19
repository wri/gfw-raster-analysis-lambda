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
    "sql": "SELECT umd_tree_cover_loss__year, SUM(area__ha), SUM(whrc_aboveground_co2_emissions__Mg) FROM data
                WHERE umd_tree_cover_density_2000__percent > 30 GROUP BY umd_tree_cover_loss__year"
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

### Raster SQL

WRI has a strong need for complex analytics beyond just simple zonal statistics. Often, we need to apply multiple masks based on pixel value, group by pixel values, and aggregate multiple values at once. Traditional GIS involves pregenerating and applying masks on each layer, then running a zonal statistics analysis. Because these analyses have a strong overlap with SQL, we invented a subset SQL called Raster SQL, that allows for expressive queries that apply the necessary GIS operations on-the-fly.

Our SQL subset supports the following:
- `SELECT` statements, which can be either use aggregate functions or pull out pixel values as "rows"
- `WHERE` statements, supporting `AND`/`OR` operators, basic comparison operators and nested expression (e.g. `x > 2 AND (y = 3 OR z < 4)`)
- `GROUP BY` statements, which can include multiple GROUP BY layers
- The aggregates `COUNT` (just counts pixels), SUM (sums values of pixels), and AVG (average values of pixels)

This translate to GIS operations in the following ways:

**Basics**

`SELECT COUNT(*) FROM data`

While the query will looks like this, based on which datasets in the API you query, it will actually be replaced under the hood to look something like this:

`SELECT COUNT(*) FROM umd_tree_cover_loss__year`

Each field name references either an actual raster layer, or special reserved word. In this basic analysis, the `umd_tree_cover_loss__year` will be applied as a basic mask, and then zonal statistics will be collected on the mask in the `geometry`. To perform zonal statistics, the geometry is rasterized and applied as an additional mask. The `COUNT` aggregate will just count all the non-masked pixels. This will return something like:

```JSON
{
   "status":"success",
   "data": [
        {
            "count": 242
        }
    ]
}
```

The `count` is the number of non-NoData pixels in the `umd_tree_cover_loss__year` raster that intersect the `geometry`.

**Area**

Usually we care about the actual hectare area of loss, not just the count of pixels. All of our analysis in WGS84, a geographic coordinate system, so pixels are degrees rather than meters. The actual meter length of 1 degree varies based on latitude because of the projection. To calculate real hectare, we introduced a special reserved field `area__ha`:

`SELECT SUM(area__ha) FROM umd_tree_cover_loss__year`

Now rather just counting the number of pixels, we're getting the sum of the hectare area of the pixel. When the query executor sees `area__ha`, it will calculate the hectare area of the pixels based on their latitude, and return the sum of area of non-masked pixels. So the results would now look like:

```JSON
{
   "status":"success",
   "data": [
        {
            "area__ha": 175.245
        }
    ]
}
```

Which is the actual hectare of loss in the `geometry`.

**Aggregation**

We can also aggregate the values of raster layers, like carbon emissions or biomass. All we need to is apply the `SUM` function to the layer:

`SELECT SUM(whrc_aboveground_co2_emissions__Mg) FROM umd_tree_cover_loss__year`

To get the amount of carbon emissions due to tree cover loss. The result might look like:

```JSON
{
   "status":"success",
   "data": [
        {
            "whrc_aboveground_co2_emissions__Mg": 245.325
        }
    ]
}
```

**Masking**

Often, we want to apply different filters (masks) to our data to get more specific information. For example, people often care about loss specifically in primary forests, since these forests are especially valuable. Using `WHERE` statements, we can add additional masks to our calculation:

`SELECT SUM(area__ha) FROM umd_tree_cover_loss__year WHERE is__umd_regional_primary_forest_2001 = 'true'`

`is__umd_regional_primary_forest_2001` is a boolean raster layer showing the extent of primary forest in 2001. The statement `is__umd_regional_primary_forest_2001 = 'true'` will load the raster into an area and apply the filter to include `true` values. For this simple boolean raster, this just means any non-NoData pixels. We then apply this as a mask to `umd_tree_cover_loss__year` before running the aggregation and zonal statics. Our result might now look like:

```JSON
{
   "status":"success",
   "data": [
        {
            "area__ha": 75.467
        }
    ]
}
```

We can also apply multiple masks using `AND`/`OR` conditions. For example, different countries usually have different definitions of how dense a cluster of trees needs to be to be called a "forest" legally. We measure this density by the percent of "canopy cover" in a pixel. We can apply this canopy cover layer as an additional mask with a query like:

`SELECT SUM(area__ha) FROM umd_tree_cover_loss__year WHERE is__umd_regional_primary_forest_2001 = 'true' AND umd_tree_cover_density_2000__percent > 30`

Now, we'll calculate loss area only in primary forests with a canopy cover percent greater than 30. Internally, the umd_tree_cover_density_2000__percent will be loaded, and the comparison operation will be applied to generate a boolean array that's true where the pixel value is greater than 30. We then combine with the overall mask using a bitwise & operation. If we used an `OR` statement in the query, we would use a bitwise | operation instead. Now our result might look like:

```JSON
{
   "status":"success",
   "data": [
        {
            "area__ha": 25.945
        }
    ]
}
```

**Grouping**

The final SQL statement we support is `GROUP BY`. This will group the results by unique pixel values in a raster, like a histogram. For example, the loss raster pixel values are actually the year loss occurred, from 2001 to the current year. We often want to know the amount of loss per year, to track how it changed over time. Now we can finally examine the query in our initial example:

`SELECT umd_tree_cover_loss__year, SUM(area__ha), SUM(whrc_aboveground_co2_emissions__Mg) FROM data WHERE umd_tree_cover_density_2000__percent > 30 GROUP BY umd_tree_cover_loss__year`

This will first apply the masks, then applies a weighted histogram function where each unique value in `umd_tree_cover_loss__year` becomes a bucket, and the weights are the values of the pixels we want to aggregate (in this case, `area__ha` and `whrc_aboveground_co2_emissions__Mg`). We then end up with aggregation results by unique value, and the results look like:

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

### Architecture

To support fast on-the-fly analytics for both small and large areas, we use a serverless AWS Lambda-based architecture. Lambdas allow us to scale up massive parallel processing very quickly, since requests usually come sporadically and may require very variable workloads.

<img width="1171" alt="image" src="https://github.com/user-attachments/assets/4e359df9-c5c9-4a32-9e7a-1e6359fd03ab">

We have three different lambas. 

**tiled-raster-analysis**: This is the entrypoint function. This function checks the size of the geometry, and splits it up into many chunks if the geometry is large. The chunk size depends on the resolution: 1.25x1.25 degrees for 30m resolution data, and 0.5x0.5 degrees for 10m resolution data. It then invokes a lambda function for each chunk, and waits for the results to be written to a DynamoDB table. Each lambda invocation has 256 KB payload limit, so it may compress or simplify the geometry chunks if the geometry is too complicated. Once all the results are in, it will aggregate the results and return them to the client.

**fanout**: This lambda is optional. One of the bottlenecks for this architecture is the lambda invocations themselves. If a geometry requires 100+ processing lambdas, it can take some time to invoke it all from one lambda. If more than 10 lambdas need to be invoked, the `tiled-raster-analysis` function will actually split the lambdas into groups of 10, and send each to a `fanout` lambda. All the fanout lambda does is invoke those 10 lambdas. This massively speeds up the invocation time for huge amounts of lambdas.

**raster-analysis**: This is the core analysis function. This processes each geometry chunk, reads all necessary rasters, runs the query execution, and writes the results out to DynamoDB. Each of these currently runs with 3 GB of RAM.

### Assumptions and limitations

GFW raster tiles are organized in 10 x 10 Degree grids and have a pixel size of 0.00025 x 0.00025 degree.
They are saved as Cloud Optimized TIFFs with 400 x 400 pixels blocks.

Because we can scale up parallel processing with lambda, size of the geometry shouldn't be an issue unless getting to massive scales (> 1 billion ha). But each lambda has in-memory cap of 3 GB, so currently only so many rasters can be loaded into memory at once. The limit depends on the size of the raster values (e.g. binary is way less memory than float), but generally max 4 or 5 raster layers is a good rule of thumb.

To optimize speed for area calculations, we assume each tile has roughly similar pixel area, and only use the pixel area of the centroid pixel for calculations. This may introduce some inaccuracies of the tile is sparsely covered. 

## Deployment

Use terraform:

```
./scripts/cibuild
./scripts/infra plan
./scripts/infra apply
```

```
Runtime: Python 3.10
Handler: lambda_function.lambda_handler
```
