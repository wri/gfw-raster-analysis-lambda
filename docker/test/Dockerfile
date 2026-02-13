FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

ENV UV_VERSION="0.10.2"

# Make the dir and install all packages into packages/
COPY raster_analysis/ raster_analysis
COPY lambdas/ lambdas
#COPY requirements-dev.txt ./
#COPY pyproject.toml ./
#COPY uv.lock ./
#
## Set uv env variables for behavior and venv directory
#ENV USR_LOCAL_BIN="/usr/local/bin"
#RUN mkdir -p ${USR_LOCAL_BIN}
#
#ENV UV_LINK_MODE=copy \
#    UV_COMPILE_BYTECODE=1 \
#    UV_UNMANAGED_INSTALL=${USR_LOCAL_BIN}
#
#RUN dnf install -y expat tar gzip && \
#    curl -LsSf https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh | sh
#
#ENV PATH=${USR_LOCAL_BIN}:${PATH}
#
#RUN uv pip install --system -r requirements-dev.txt && \
#    uv pip install --system .
#
#CMD ["/bin/bash"]

WORKDIR /opt

RUN dnf install -y unzip

ENV PYTHONPATH=/opt/python

ENV RUNTIME_PREFIX="python3.13-py313" \
    MERCANTILE_VERSION_SUFFIX="121_1.2.1" \
    NUMPY_VERSION_SUFFIX="242_2.4.2" \
    PILLOW_VERSION_SUFFIX="1210_12.1.0" \
    RASTERIO_VERSION_SUFFIX="144_1.4.4"

ENV MERCANTILE_FILENAME="${RUNTIME_PREFIX}_mercantile_${MERCANTILE_VERSION_SUFFIX}.zip" \
    NUMPY_FILENAME="${RUNTIME_PREFIX}_numpy_${NUMPY_VERSION_SUFFIX}.zip" \
    PILLOW_FILENAME="${RUNTIME_PREFIX}_pillow_${PILLOW_VERSION_SUFFIX}.zip" \
    RASTERIO_FILENAME="${RUNTIME_PREFIX}_rasterio_${RASTERIO_VERSION_SUFFIX}.zip"

RUN for filename in $MERCANTILE_FILENAME $NUMPY_FILENAME $PILLOW_FILENAME $RASTERIO_FILENAME; do \
      unzip /usr/local/app/tests/fixtures/${filename} \
    done
