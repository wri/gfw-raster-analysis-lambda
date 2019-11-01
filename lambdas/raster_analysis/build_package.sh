#!/usr/bin/env bash
IMAGE_NAME="gfw/raster-analysis-lambda"

docker build -t ${IMAGE_NAME} -f ./Dockerfile ../..
docker run --name lambda -itd ${IMAGE_NAME} /bin/bash
docker cp lambda:/tmp/package.zip package.zip
docker stop lambda
docker rm lambda