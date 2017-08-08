## Raster Processing on Serverless Architecture
Source code for a presentation on using Rasterio and Numpy to do raster processing on AWS Lambda

To run:

1. Install Serverless Framework
```
npm install
```

2. Install python dependencies in virtualenv
```
./scripts/setup.sh
```

3. Activate virtualenv
```
source env/bin/activate
```

4. Update `serverless.yml` to specify your AWS profile and s3 buckets

5. Create serverless stack and deploy
```
./scripts/publish.sh
```

