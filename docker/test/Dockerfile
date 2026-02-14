FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.13

ENV UV_VERSION="0.10.2"

COPY raster_analysis raster_analysis
COPY lambdas lambdas
COPY docker/layers layers

COPY requirements-test.txt requirements-test.txt

RUN dnf install -y unzip

RUN for filename in `ls layers/*.zip`; do \
      unzip -n ${filename} -d /opt; \
    done

RUN pip install -r requirements-test.txt -t /opt/python

WORKDIR /opt
ENV PYTHONPATH=/opt/python

ENTRYPOINT ["python", "-m", "pytest", "--cov-report", "term", "--cov-report", "xml:/usr/local/app/tests/cobertura.xml", "--cov=/usr/local/app", "/usr/local/app/tests/" ]
