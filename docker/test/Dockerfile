FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

ENV UV_VERSION="0.10.2"

COPY raster_analysis raster_analysis
COPY lambdas lambdas
COPY layers layers

COPY requirements-test.txt ./

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

RUN dnf install -y unzip

RUN for filename in `ls layers/*.zip`; do \
      unzip ${filename} -d /opt; \
    done

# RUN pip install -r requirements-test.txt -t /opt/python

WORKDIR /opt
ENV PYTHONPATH=/opt/python

ENTRYPOINT ["python", "-m", "pytest", "--cov-report", "term", "--cov-report", "xml:/usr/local/app/tests/cobertura.xml", "--cov=/usr/local/app", "/usr/local/app/tests/" ]
