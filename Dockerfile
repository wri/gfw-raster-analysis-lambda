FROM lambci/lambda:build-python3.7

WORKDIR /var/task
ENV WORKDIR /var/task

# Make the dir and to install all packages into packages/
COPY setup.py "$WORKDIR/setup.py"
COPY raster_analysis "$WORKDIR/raster_analysis"


RUN mkdir -p packages/ \
    && pip install . -t packages --no-binary numpy/

RUN rm packages/shapely/.libs/libgeos-3-6a255356.6.2.so \
    && rm packages/shapely/.libs/libgeos_c-bd8d3f16.so.1.10.2 \
    && ln -s packages/rasterio/.libs/libgeos-3-cd838e67.6.2.so packages/shapely/.libs/libgeos-3-6a255356.6.2.so \
    && ln -s packages/rasterio/.libs/libgeos_c-595de9d4.so.1.10.2 packages/shapely/.libs/libgeos_c-bd8d3f16.so.1.10.2

RUN python -m compileall .

RUN find packages/ -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-37//'); cp $f $n; done;
RUN find packages/ -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN find packages/ -type f -a -name '*.py' -print0 | xargs -0 rm -f

# Copy initial source codes into container.
COPY lambda_function.py "$WORKDIR/lambda_function.py"

COPY .lambdaignore "$WORKDIR/.lambdaignore"
# Compress all source codes.
RUN cat .lambdaignore | xargs zip -9qyr $WORKDIR/lambda.zip packages/ lambda_function.py -x

CMD ["/bin/bash"]

 
