FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

ENV UV_VERSION="0.9.30"

# Make the dir and to install all packages into packages/
COPY raster_analysis/ raster_analysis/
COPY lambdas/ lambdas/
COPY requirements-dev.txt ./
COPY pyproject.toml ./
COPY uv.lock ./

# Set uv env variables for behavior and venv directory
ENV USR_LOCAL_BIN="/usr/local/bin"
RUN mkdir -p ${USR_LOCAL_BIN}

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_UNMANAGED_INSTALL=${USR_LOCAL_BIN}

RUN dnf install tar && \
    curl -LsSf https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh | sh

ENV PATH=${USR_LOCAL_BIN}:${PATH}

RUN uv pip install --system -r requirements-dev.txt && \
    uv pip install --system .

CMD ["/bin/bash"]