FROM remotepixel/amazonlinux-gdal:3.0.1

# Make the dir and to install all packages into packages/
COPY raster_analysis/ raster_analysis/
COPY lambdas/ lambdas/
COPY setup.py .
COPY requirements-dev.txt .

# installing dependencies to build package
RUN pip3 install -r requirements-dev.txt
RUN pip3 install .

CMD ["/bin/bash"]