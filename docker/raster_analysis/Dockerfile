FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.10

ENV UV_VERSION="0.10.2"

WORKDIR /var/task
ENV WORKDIR=/var/task

RUN mkdir -p python

# Make the dir and install all packages into packages/
COPY raster_analysis/ raster_analysis/
COPY pyproject.toml ./
COPY uv.lock ./
COPY docker/raster_analysis/.lambdaignore ./

# Set uv env variables for behavior and venv directory
ENV USR_LOCAL_BIN="/usr/local/bin"
RUN mkdir -p ${USR_LOCAL_BIN}

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_UNMANAGED_INSTALL=${USR_LOCAL_BIN}

RUN dnf update -y && \
    dnf install -y findutils gzip tar zip && \
    curl -LsSf https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh | sh

ENV PATH=${USR_LOCAL_BIN}:${PATH}

RUN uv pip install . -t python

# Precompile all python packages and remove .py files
#RUN python -m compileall .
#RUN find python/ -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-313//'); cp $f $n; done;
#RUN find python/ -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
#RUN find python/ -type f -a -name '*.py' -print0 | xargs -0 rm -f

# Compress all source codes except files listed in .lambdaignore
RUN cat .lambdaignore | xargs zip -9qyr layer.zip python -x

CMD ["/bin/bash"]