FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.10

RUN yum update -y && \
    yum install -y gzip unzip tar && \
    yum clean all

# App code goes under /var/task (Lambda task root)
WORKDIR /var/task
COPY raster_analysis raster_analysis
COPY lambdas lambdas

# Layer zips
COPY docker/layers /layers

# Unpack all layer zips into /opt (like Lambda)
RUN set -eux; \
    for filename in /layers/*.zip; do \
      echo "Unpacking $filename"; \
      unzip -q -n "$filename" -d /opt; \
    done

# Install test dependencies in a venv so entrypoints exist (moto_server, pytest, etc.)
# Using pyproject.toml with [test] optional dependencies
ENV UV_VERSION="0.9.11"
ENV USR_LOCAL_BIN="/usr/local/bin"
RUN mkdir -p ${USR_LOCAL_BIN}

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_UNMANAGED_INSTALL=${USR_LOCAL_BIN}

RUN curl -LsSf https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh | sh

ENV PATH=${USR_LOCAL_BIN}:${PATH}

COPY pyproject.toml ./pyproject.toml

RUN python -m venv /venv && \
    uv pip install --python /venv/bin/python ".[test]"

ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="/var/task:/opt/python"
ENV LD_LIBRARY_PATH="/opt/lib:${LD_LIBRARY_PATH}"
# Preload the SQLite library from /opt/lib to ensure GDAL uses the correct version
ENV LD_PRELOAD="/opt/lib/libsqlite3.so.0"

ENTRYPOINT ["python", "-m", "pytest", "--cov-report", "term", "--cov-report", "xml:/var/task/tests/cobertura.xml", "--cov=/var/task"]
