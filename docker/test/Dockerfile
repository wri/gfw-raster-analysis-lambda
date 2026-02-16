FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.10

RUN yum install -y unzip && yum clean all

# App code goes under /var/task (Lambda task root)
WORKDIR /var/task
COPY raster_analysis raster_analysis
COPY lambdas lambdas

# Layer zips
COPY docker/layers /layers

# Debug: show what was copied
RUN echo "=== Contents of /layers ===" && ls -lh /layers/ && echo "========================="

# Unpack all layer zips into /opt (like Lambda)
RUN set -eux; \
    for filename in /layers/*.zip; do \
      echo "Unpacking $filename"; \
      unzip -q -n "$filename" -d /opt; \
    done

# Debug: show what's in /opt/python after unpacking
RUN echo "=== Contents of /opt/python ===" && ls -lh /opt/python/ && echo "========================="

# Debug: show what's in /opt/lib (GDAL and dependencies)
RUN echo "=== Contents of /opt/lib ===" && ls -lh /opt/lib/ 2>/dev/null || echo "No /opt/lib directory" && echo "========================="

# Debug: check if SQLite library is present and verify LD_LIBRARY_PATH
RUN echo "=== Checking for SQLite libraries ===" && \
    ls -lh /opt/lib/libsqlite3* 2>/dev/null || echo "No SQLite libraries found in /opt/lib" && \
    echo "========================="

# Test deps in a venv so entrypoints exist (moto_server, pytest, etc.)
COPY requirements-test.txt /tmp/requirements-test.txt
RUN python -m venv /venv && \
    /venv/bin/pip install -r /tmp/requirements-test.txt

ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="/var/task:/opt/python"
ENV LD_LIBRARY_PATH="/opt/lib:${LD_LIBRARY_PATH}"
# Preload the SQLite library from /opt/lib to ensure GDAL uses it
ENV LD_PRELOAD="/opt/lib/libsqlite3.so.0"

# Entrypoint wrapper that always runs the intended tests but allows extra flags
COPY docker/test/pytest_entrypoint.sh /usr/local/bin/pytest_entrypoint.sh
RUN chmod +x /usr/local/bin/pytest_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/pytest_entrypoint.sh"]