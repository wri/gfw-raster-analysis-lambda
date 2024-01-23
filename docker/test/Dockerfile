FROM ghcr.io/lambgeo/lambda-gdal:3.6-python3.10

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