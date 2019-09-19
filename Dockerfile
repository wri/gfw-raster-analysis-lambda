FROM remotepixel/amazonlinux-gdal:2.4.0-light

WORKDIR /tmp/python
ENV WORKDIR /tmp/python

# Make the dir and to install all packages into packages/
COPY setup.py "$WORKDIR/setup.py"
COPY raster_analysis "$WORKDIR/raster_analysis"

RUN mkdir -p $WORKDIR
RUN CFLAGS="--std=c99" pip3 install . --no-binary numpy,rasterio,shapely -t $WORKDIR -U

RUN python -m compileall .
RUN find -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-36//'); cp $f $n; done;
RUN find -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN find -type f -a -name '*.py' -print0 | xargs -0 rm -f

# Copy initial source codes into container.
COPY lambda_function.py "$WORKDIR/lambda_function.py"
COPY .lambdaignore "$WORKDIR/.lambdaignore"

RUN cd /tmp/python && \
    cat .lambdaignore | xargs zip -9qyr /tmp/package.zip . -x
RUN cd /var/task; zip -r9q --symlinks /tmp/package.zip lib/*.so*
#RUN cd /var/task; zip -r9q --symlinks /tmp/package.zip lib64/*.so*
RUN cd /var/task; zip -r9q /tmp/package.zip share

CMD ["/bin/bash"]


