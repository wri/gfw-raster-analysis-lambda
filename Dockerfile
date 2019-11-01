FROM remotepixel/amazonlinux-gdal:2.4.0-light

WORKDIR /tmp/python
ENV WORKDIR /tmp/python

# Make the dir and to install all packages into packages/
COPY requirements.txt "$WORKDIR/requirements.txt"

RUN mkdir -p $WORKDIR
RUN CFLAGS="--std=c99" pip3 install -r requirements.txt --no-binary numpy,rasterio,shapely -t $WORKDIR -U

RUN python -m compileall .
RUN find -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-36//'); cp $f $n; done;
RUN find -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN find -type f -a -name '*.py' -print0 | xargs -0 rm -f