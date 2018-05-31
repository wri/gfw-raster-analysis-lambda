# gfw-raster-analysis-lambda

### Background

This repo takes Matt McFarland's [raster lambda demo](https://github.com/mmcfarland/foss4g-lambda-demo) and uses it to provide a scalable alternative to the GEE-backed [umd-loss-gain](https://github.com/wri/gfw-umd-loss-gain-lambda).

This usage grew out of the need to process 2000 palm oil mills in under 5 minutes, per the requirements for the upcoming GFW Pro application. 

### Endpoints

The endpoints deployed are designed to exactly mimic existing GFW API endpoints, making it easy to 'plug' this service into existing code. The current exposed endpoints are as follows:

Base URL:
https://0yvx7602sb.execute-api.us-east-1.amazonaws.com/dev/

/umd-loss-gain:
Replicates the existing umd-loss-gain endpoint

/analysis:
Runs analysis for loss | gain | extent against a polygon AOI. These requests are parallelized under the hood as part of the umd-loss-gain endpoint

/landcover:
Matches the /landcover endpoint on the GFW API, but only provides analysis for `primary-forest` data (the only layer required by GFW Pro batch analysis at this time)

/loss-by-landcover:
Matches the /loss-by-landcover endpoint, again only providing results for `primary-forest` data

/glad-alerts
Matches the /glad-alerts endpoint, but requires aggregate_values to be set to True (designed for batch processing only)

### Limitations

Our main obstacle here is speed; particularly large areas may time out. This isn't an issue for the palm-risk use case (all areas are 50 km buffers of palm oil mills) but we could run into this if we expand the usage of this repo.

If we need to cross this bridge, the subdivide_polygon function in the [original repo](https://github.com/mmcfarland/foss4g-lambda-demo/blob/master/handler.py#L63) will likely help.


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

