FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

# Make the dir and to install all packages into packages/
COPY raster_analysis/ raster_analysis/
COPY lambdas/ lambdas/
COPY requirements-dev.txt ./
COPY pyproject.toml ./
COPY uv.lock ./

# Set uv env variables for behavior and venv directory
ENV PATH=${USR_LOCAL_BIN}:${PATH} \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=${VENV_DIR} \
    UV_UNMANAGED_INSTALL=${USR_LOCAL_BIN}

# Create a virtual environment with uv inside the container
RUN curl -LsSf https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh | sh && \
    uv pip install --system -r requirements-dev.txt && \
    uv pip install --system .

CMD ["/bin/bash"]