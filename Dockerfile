FROM lambci/lambda:build-python3.7

WORKDIR /var/task
ENV WORKDIR /var/task

# Make the dir and to install all packages into packages/
RUN mkdir -p packages/ && \
    pip install git+https://https://github.com/gfw-api/gfw-raster-analysis-lambda#egg=raster-analysis -t packages/

# Copy initial source codes into container.
COPY lambda_function.py "$WORKDIR/lambda_function.py"

# Compress all source codes.
# RUN zip -r9 $WORKDIR/lambda.zip packages/ lambda_function.py

# CMD ["/bin/bash"]

 
