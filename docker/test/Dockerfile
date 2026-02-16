FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.10

RUN yum install -y unzip && yum clean all

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

# Test deps in a venv so entrypoints exist (moto_server, pytest, etc.)
COPY requirements-test.txt /tmp/requirements-test.txt
RUN python -m venv /venv && \
    /venv/bin/pip install -r /tmp/requirements-test.txt

ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="/var/task:/opt/python"
ENV LD_LIBRARY_PATH="/opt/lib:${LD_LIBRARY_PATH}"
# Preload the SQLite library from /opt/lib to ensure GDAL uses the correct version
ENV LD_PRELOAD="/opt/lib/libsqlite3.so.0"

# Entrypoint wrapper that always runs the intended tests but allows extra flags
COPY docker/test/pytest_entrypoint.sh /usr/local/bin/pytest_entrypoint.sh
RUN chmod +x /usr/local/bin/pytest_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/pytest_entrypoint.sh"]
