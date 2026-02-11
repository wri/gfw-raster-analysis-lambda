FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

WORKDIR /var/task
ENV WORKDIR=/var/task

WORKDIR $WORKDIR
RUN mkdir -p python

# Make the dir and install all packages into packages/
COPY raster_analysis/ raster_analysis/
COPY setup.py .
COPY docker/raster_analysis/.lambdaignore .

# Install dependencies to build package
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
RUN pip install . -t python

RUN yum install -y zip

# Precompile all python packages and remove .py files
RUN python -m compileall .
RUN find python/ -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-313//'); cp $f $n; done;
RUN find python/ -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN find python/ -type f -a -name '*.py' -print0 | xargs -0 rm -f

# Compress all source codes expect files listed in .lambdaignore
RUN cat .lambdaignore | xargs zip -9qyr layer.zip python -x

CMD ["/bin/bash"]