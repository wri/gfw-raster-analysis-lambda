FROM --platform=linux/amd64 ghcr.io/lambgeo/lambda-gdal:3.8-python3.13

# Make the dir and to install all packages into packages/
COPY raster_analysis/ raster_analysis/
COPY lambdas/ lambdas/
COPY setup.py .
COPY requirements-dev.txt .

# installing dependencies to build package
ENV SETUPTOOLS_USE_DISTUTILS stdlib
RUN pip3 install -r requirements-dev.txt
RUN pip3 install .

CMD ["/bin/bash"]