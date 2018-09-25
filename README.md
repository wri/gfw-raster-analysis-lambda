# glad-raster-analysis-lambda

### Background

This repo takes Matt McFarland's [raster lambda demo](https://github.com/mmcfarland/foss4g-lambda-demo) and uses it to provide stats and raster-to-vector downloads for our GLAD data.

### Data

After receiving an input polygon and an operation type (stats or download), the code looks at the GLAD raster mosaic stored in data/glad.vrt. This points to a bunch of rasters, like this one over part of Nigeria: s3://palm-risk-poc/data/glad/analysis-staging/afr_asia/tiles/000E_00N_010E_10N/raster_processing/date_conf/1_prep/date_conf_all_nd_0.tif

### Endpoints

The endpoints deployed are designed to exactly mimic existing GFW API endpoints, making it easy to 'plug' this service into existing code. The current exposed endpoints are as follows:

Base URL:
https://0kepi1kf41.execute-api.us-east-1.amazonaws.com/dev/

/glad-alerts
Matches the /glad-alerts endpoint

/glad-alerts/download
Allows for vector download of GLAD data in CSV or JSON format. Called when someone wants to download an AOI or geostore from: https://github.com/gfw-api/glad-analysis-tiled

### Limitations

Our main obstacle here is speed; particularly large areas may time out. If we need to cross this bridge, the subdivide_polygon function in the [original repo](https://github.com/mmcfarland/foss4g-lambda-demo/blob/master/handler.py#L63) will likely help.


## Development
1. Clone locally
2. Create .env file to store AWS credentials in the root of this project
```
AWS_ACCESS_KEY_ID=<my key id>
AWS_SECRET_ACCESS_KEY=<my key>
```
3. Spin up the docker container and ssh in
```
docker-compose run base
```

4. Change to shared directory
```
cd /home/geolambda/
```

5. Run handler.py to test analysis and alerts endpoints (see `if __name__ == '__main__':` block)


## Deployment
1. install serverless `npm install -g serverless`

2. Test! Run `docker-compose run test` to automatically spin up the docker container and run your tests (in the `/test` folder)

3. Package - `docker-compose run package`

4. Deploy - `serverless deploy -v`

