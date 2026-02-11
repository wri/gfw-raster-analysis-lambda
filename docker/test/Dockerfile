FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

# Make the dir and to install all packages into packages/
COPY raster_analysis/ raster_analysis/
COPY lambdas/ lambdas/
COPY setup.py .
COPY requirements-dev.txt .

# installing dependencies to build package
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
RUN pip install -r requirements-dev.txt
RUN pip install .

CMD ["/bin/bash"]