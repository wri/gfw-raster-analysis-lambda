#!/bin/bash
set -e

echo "Packaging dependencies..."
mkdir dist -p

pushd . > /dev/null
cd ./env/lib/python2.7/site-packages

# Delete the duplicated geos file and symlink it to the existing .so
rm shapely/.libs/libgeos-3-fc05f4c1.5.0.so
ln -s ../../rasterio/.libs/libgeos-3-fc05f4c1.5.0.so shapely/.libs/libgeos-3-fc05f4c1.5.0.so

# Zip the code and dependencies, ignoring things we know to exist on
# the lambda runtime already.  Set to not resolve symlinks
zip -9 -rqy ../../../../dist/raster-ops-deploy.zip * \
    -x \*-info\* \
    -x boto\*\* \
    -x pip\* \
    -x docutils\* \
    -x s3transfer\* \
    -x setuptools\* \
    -x jmespath\* \
    -x pkg_resources\* \

popd  > /dev/null
zip -9 -rq dist/raster-ops-deploy.zip geop/* handler.py

# Deploy the function
serverless deploy -v
