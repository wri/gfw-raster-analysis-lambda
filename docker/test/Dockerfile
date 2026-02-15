FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

RUN dnf install -y unzip && dnf clean all

# App code goes under /var/task (Lambda task root)
WORKDIR /var/task
COPY raster_analysis raster_analysis
COPY lambdas lambdas

# Layer zips
COPY docker/layers /layers

# Unpack all layer zips into /opt (like Lambda)
RUN set -eux; \
    for filename in /layers/*.zip; do \
      unzip -q -n "$filename" -d /opt; \
    done

# Test deps in a venv so entrypoints exist (moto_server, pytest, etc.)
COPY requirements-test.txt /tmp/requirements-test.txt
RUN python -m venv /venv && \
    /venv/bin/pip install -r /tmp/requirements-test.txt

ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="/var/task:/opt/python"

# Entrypoint wrapper that always runs the intended tests but allows extra flags
COPY docker/test/pytest_entrypoint.sh /usr/local/bin/pytest_entrypoint.sh
RUN chmod +x /usr/local/bin/pytest_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/pytest_entrypoint.sh"]