# GFW raster analsyis in AWS Lambda

### Functionality

Run zonal statistics on tree cover loss, GLAD alerts, or arbitrary contextual layers defined in our data lake.*

Currently supports three endpoints:
1) /analysis/treecoverloss - get statistics on hectares of tree cover loss per year in a geometry. This can be thresholded by a given tree cover density from either the 2000 or 2010 tree cover density baseline.
2) /analysis/gladalerts - get statistics on number of glad alerts per day in geometry.
3) /analysis/summary - get statistics on hectare areas of arbitrary combinations of layers. This can be used to get total forest extent area or total area of contextual layers intersecting the geometry (i.e. for percents of tree cover loss).

*Date lake is not yet complete, so currently only these layers are supported:
* wdpa
* primary_forest
* plantations
* ifl
* drivers
* biomass
* emissions


### Query Parameters

|Parameter|Type|Description|Example|
|---------|----|-----------|-------|
|geostore_id (required)| String | A valid geostore ID containing the GeoJSON for the geometry of interest (see further specification in `Limitations and assumtions` | cb64960710fd11925df3fda6a2005ca9 |
|contextual_raster_ids| [String] | List of rasters to contextualize analysis. Analysis results will be aggregated by unique combinations of contextual and analysis raster values. | plantations,wdpa |
|aggregate_rasters_ids| [String] | List of value rasters to aggregate. These layers will have their pixel values summed unique combinations of contextual layers. | biomass,emissions |
|threshold| Integer | A percent threshold of tree cover density, using either 2000 or 2010 TCD layers. | 30 |
|extent_year| Year | The year to use for tree cover density. Must be either 2000 or 2010. Default is 2000. | 2000 |
|start| Date | This should be a year for tree cover loss, or a date in YYYY-MM-DD format for GLAD alerts (ignored for summary). | 2015, 2015-02-04 |
|end| Date | Same format as 'start'. Must come after start. | 2016 , 2016-02-10 |
#### Examples

Request:
```http request
https://gad5b4taw3.execute-api.us-east-1.amazonaws.com/default/analysis/treecoverloss?analyses=area&geostore_id=cb64960710fd11925df3fda6a2005ca9&contextual_raster_ids=wdpa&contextual_raster_ids=primary_forest&threshold=30
```

Response:
```JSON
[
   {
      "loss":2001,
      "wdpa":0,
      "primary_forest":0,
      "area":804.1504851401617
   },
   {
      "loss":2001,
      "wdpa":0,
      "primary_forest":1,
      "area":342.39460212739095
   },
   {
      "loss":2001,
      "wdpa":1,
      "primary_forest":0,
      "area":0.23072412542277018
   },
   {
      "loss":2002,
      "wdpa":0,
      "primary_forest":0,
      "area":12.84364298186754
   },
   {
      "loss":2002,
      "wdpa":0,
      "primary_forest":1,
      "area":403.921035573463
   },
   {
      "loss":2003,
      "wdpa":0,
      "primary_forest":0,
      "area":30.071044346767714
   },
   {
      "loss":2003,
      "wdpa":0,
      "primary_forest":1,
      "area":362.69832516459473
   },
   {
      "loss":2004,
      "wdpa":0,
      "primary_forest":0,
      "area":25.84110204735026
   },
   {
      "loss":2004,
      "wdpa":0,
      "primary_forest":1,
      "area":320.09127000318983
   },
   {
      "loss":2005,
      "wdpa":0,
      "primary_forest":0,
      "area":11.99765452198405
   },
   {
      "loss":2005,
      "wdpa":0,
      "primary_forest":1,
      "area":148.81706089768676
   },
   {
      "loss":2005,
      "wdpa":1,
      "primary_forest":1,
      "area":0.07690804180759006
   },
   {
      "loss":2006,
      "wdpa":0,
      "primary_forest":0,
      "area":45.99100900093886
   },
   {
      "loss":2006,
      "wdpa":0,
      "primary_forest":1,
      "area":441.6828841009897
   },
   {
      "loss":2006,
      "wdpa":1,
      "primary_forest":1,
      "area":0.30763216723036024
   },
   {
      "loss":2007,
      "wdpa":0,
      "primary_forest":0,
      "area":32.76282581003336
   },
   {
      "loss":2007,
      "wdpa":0,
      "primary_forest":1,
      "area":275.10006554574966
   },
   {
      "loss":2007,
      "wdpa":1,
      "primary_forest":1,
      "area":0.15381608361518012
   },
   {
      "loss":2008,
      "wdpa":0,
      "primary_forest":0,
      "area":30.455584555805665
   },
   {
      "loss":2008,
      "wdpa":0,
      "primary_forest":1,
      "area":366.4668192131666
   },
   {
      "loss":2008,
      "wdpa":1,
      "primary_forest":1,
      "area":0.6921723762683105
   },
   {
      "loss":2009,
      "wdpa":0,
      "primary_forest":0,
      "area":19.688458702743056
   },
   {
      "loss":2009,
      "wdpa":0,
      "primary_forest":1,
      "area":366.6975433385894
   },
   {
      "loss":2010,
      "wdpa":0,
      "primary_forest":0,
      "area":19.765366744550647
   },
   {
      "loss":2010,
      "wdpa":0,
      "primary_forest":1,
      "area":220.4953558623607
   },
   {
      "loss":2010,
      "wdpa":1,
      "primary_forest":1,
      "area":0.5383562926531305
   },
   {
      "loss":2011,
      "wdpa":0,
      "primary_forest":0,
      "area":87.29062745161472
   },
   {
      "loss":2011,
      "wdpa":0,
      "primary_forest":1,
      "area":716.7829496467393
   },
   {
      "loss":2011,
      "wdpa":1,
      "primary_forest":1,
      "area":3.383953839533963
   },
   {
      "loss":2012,
      "wdpa":0,
      "primary_forest":0,
      "area":71.98592713190429
   },
   {
      "loss":2012,
      "wdpa":0,
      "primary_forest":1,
      "area":627.4927131081273
   },
   {
      "loss":2012,
      "wdpa":1,
      "primary_forest":1,
      "area":2.3841492960352917
   },
   {
      "loss":2013,
      "wdpa":0,
      "primary_forest":0,
      "area":50.06713521674113
   },
   {
      "loss":2013,
      "wdpa":0,
      "primary_forest":1,
      "area":248.56679112213106
   },
   {
      "loss":2013,
      "wdpa":1,
      "primary_forest":1,
      "area":1.076712585306261
   },
   {
      "loss":2014,
      "wdpa":0,
      "primary_forest":0,
      "area":91.90510996007012
   },
   {
      "loss":2014,
      "wdpa":0,
      "primary_forest":1,
      "area":663.2549525486567
   },
   {
      "loss":2014,
      "wdpa":1,
      "primary_forest":1,
      "area":0.07690804180759006
   },
   {
      "loss":2015,
      "wdpa":0,
      "primary_forest":0,
      "area":122.36069451587579
   },
   {
      "loss":2015,
      "wdpa":0,
      "primary_forest":1,
      "area":1131.1634789060347
   },
   {
      "loss":2015,
      "wdpa":1,
      "primary_forest":1,
      "area":1.8457930033821615
   },
   {
      "loss":2016,
      "wdpa":0,
      "primary_forest":0,
      "area":336.39577486639894
   },
   {
      "loss":2016,
      "wdpa":0,
      "primary_forest":1,
      "area":3808.255506186437
   },
   {
      "loss":2016,
      "wdpa":1,
      "primary_forest":1,
      "area":0.5383562926531305
   },
   {
      "loss":2017,
      "wdpa":0,
      "primary_forest":0,
      "area":328.4742465602171
   },
   {
      "loss":2017,
      "wdpa":0,
      "primary_forest":1,
      "area":3306.8919816427574
   },
   {
      "loss":2017,
      "wdpa":1,
      "primary_forest":1,
      "area":0.3845402090379503
   },
   {
      "loss":2018,
      "wdpa":0,
      "primary_forest":0,
      "area":652.7954588628245
   },
   {
      "loss":2018,
      "wdpa":0,
      "primary_forest":1,
      "area":5202.598304158045
   },
   {
      "loss":2018,
      "wdpa":1,
      "primary_forest":1,
      "area":0.9998045434986708
   }
]
```


### Endpoints
```http request
https://gad5b4taw3.execute-api.us-east-1.amazonaws.com/default/analysis/treecoverloss
https://gad5b4taw3.execute-api.us-east-1.amazonaws.com/default/analysis/gladalerts
https://gad5b4taw3.execute-api.us-east-1.amazonaws.com/default/analysis/summary
```


### Assumptions and limitations

GFW raster tiles are organized in 10 x 10 Degree grids and have a pixel size of 0.00025 x 0.00025 degree.
They are saved as Cloud Optimized TIFFs with 400 x 400 pixels blocks.

Because we can scale up parallel processing with lambda, size of the geometry shouldn't be an issue unless getting to massive scales (> 1 billion ha). But each lambda has in-memory cap of 3 GB, so currently only so many rasters can be loaded into memory at once. The limit depends on the size of the raster values (e.g. binary is way less memory than float), but generally max 4 or 5 raster layers is a good rule of thumb.

## Deployment

Use terraform:

```
./scripts/infra plan
./scripts/infra apply
```

```
Runtime: Python 3.6
Handler: lambda_function.lambda_handler
```

## Future Development

### Data Lake

The GFW data lake is now in production, so this service will soon point to that instead of just test layers. Once it does, all data lake layers should be available for analysis. This currently includes:

* aqueduct_baseline_water_stress
* aqueduct_erosion_risk
* birdlife_alliance_for_zero_extinction_site
* birdlife_endemic_bird_areas
* birdlife_key_biodiversity_area
* bra_biomes
* esa_land_cover_2015
* gfw_aboveground_carbon_stock_2000
* gfw_aboveground_carbon_stock_in_emissions_year
* gfw_aboveground_carbon_stock_in_emissions_year__biomass_swap
* gfw_aboveground_carbon_stock_in_emissions_year__legal_amazon_loss
* gfw_aboveground_carbon_stock_in_emissions_year__no_primary_gain
* gfw_aboveground_carbon_stock_in_emissions_year__us_removals
* gfw_belowground_carbon_stock_2000
* gfw_belowground_carbon_stock_in_emissions_year
* gfw_deadwood_carbon_stock_2000
* gfw_deadwood_carbon_stock_in_emissions_year
* gfw_forest_age_category
* gfw_gross_annual_removals_biomass
* gfw_gross_cumul_removals_co2
* gfw_gross_cumul_removals_co2__biomass_swap
* gfw_gross_cumul_removals_co2__legal_amazon_loss
* gfw_gross_cumul_removals_co2__maxgain
* gfw_gross_cumul_removals_co2__no_primary_gain
* gfw_gross_cumul_removals_co2__us_removals
* gfw_gross_emissions_co2e_co2_only
* gfw_gross_emissions_co2e_co2_only__biomass_swap
* gfw_gross_emissions_co2e_co2_only__convert_to_grassland
* gfw_gross_emissions_co2e_co2_only__legal_amazon_loss
* gfw_gross_emissions_co2e_co2_only__no_primary_gain
* gfw_gross_emissions_co2e_co2_only__no_shifting_ag
* gfw_gross_emissions_co2e_co2_only__soil_only
* gfw_gross_emissions_co2e_co2_only__us_removals
* gfw_gross_emissions_co2e_non_co2
* gfw_gross_emissions_co2e_non_co2__biomass_swap
* gfw_gross_emissions_co2e_non_co2__convert_to_grassland
* gfw_gross_emissions_co2e_non_co2__legal_amazon_loss
* gfw_gross_emissions_co2e_non_co2__no_primary_gain
* gfw_gross_emissions_co2e_non_co2__no_shifting_ag
* gfw_gross_emissions_co2e_non_co2__soil_only
* gfw_gross_emissions_co2e_non_co2__us_removals
* gfw_intact_or_primary_forest_2000
* gfw_land_rights
* gfw_litter_carbon_stock_2000
* gfw_litter_carbon_stock_in_emissions_year
* gfw_managed_forests
* gfw_mining
* gfw_net_flux_co2e
* gfw_net_flux_co2e__biomass_swap
* gfw_net_flux_co2e__convert_to_grassland
* gfw_net_flux_co2e__legal_amazon_loss
* gfw_net_flux_co2e__maxgain
* gfw_net_flux_co2e__no_primary_gain
* gfw_net_flux_co2e__no_shifting_ag
* gfw_net_flux_co2e__us_removals
* gfw_oil_gas
* gfw_oil_palm
* gfw_peatlands
* gfw_peatlands__flux
* gfw_pixel_area
* gfw_plantations
* gfw_resource_rights
* gfw_soil_carbon_stock_2000
* gfw_soil_carbon_stock_in_emissions_year
* gfw_soil_carbon_stock_in_emissions_year__biomass_swap
* gfw_soil_carbon_stock_in_emissions_year__legal_amazon_loss
* gfw_soil_carbon_stock_in_emissions_year__no_primary_gain
* gfw_soil_carbon_stock_in_emissions_year__us_removals
* gfw_tiger_landscapes
* gfw_total_carbon_stock_2000
* gfw_total_carbon_stock_in_emissions_year
* gfw_wood_fiber
* gmw_mangroves_1996
* gmw_mangroves_2016
* idn_forest_area
* idn_forest_moratorium
* idn_land_cover_2017
* idn_primary_forest_2000
* ifl_intact_forest_landscapes
* jpl_mangrove_aboveground_biomass_stock_2000
* jpl_tropics_abovegroundbiomass_2000
* landmark_land_rights
* mapbox_river_basins
* mex_forest_zoning
* mex_payment_ecosystem_services
* mex_protected_areas
* per_forest_concessions
* per_permanent_production_forests
* per_protected_areas
* rspo_oil_palm
* tnc_urban_water_intake
* tsc_tree_cover_loss_drivers
* umd_regional_primary_forest_2001
* umd_tree_cover_density_2000
* umd_tree_cover_density_2010
* umd_tree_cover_gain
* umd_tree_cover_loss
* usfs_fia_regions
* wdpa_protected_areas
* whrc_aboveground_biomass_stock_2000
* wwf_eco_regions

### VIIRS/MODIS Alerts

These alerts are currently unsupported because we don't rasterize these layers. Instead, we store all enriched points in an document dataset. You can do on-the-fly analysis for these via SQL. (TBD: do we want to just forward that through here so there's only one endpoint?)

### Aggregation

There will be a new `String` parameter called `agg` that will accept one of `day | week | month | year` and return results aggregated by that timeline.

### Whitelist

There will be a new endpoint that will return a whitelist of whether layers intersect the input geometry. Details TBD.

### Misc Layers

Need to decide if/how we will support miscellanious layers on GFW but not maintained in the data lake, like PRODES and Terra-i. TBD.

### Lat/Lon Coordinates

A new analysis will be added to retrieve lat/lon coordinates of points (e.g. for GLAD alerts).
