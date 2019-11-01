FROM jterry64/base-raster-analysis-lambda:latest

WORKDIR /tmp/python
ENV WORKDIR /tmp/python

# Copy initial source codes into container.
COPY lambdas/tiled_analysis/lambda_function.py "$WORKDIR/lambda_function.py"
COPY lambdas/tiled_analysis/.lambdaignore "$WORKDIR/.lambdaignore"

RUN cd /tmp/python && \
    cat .lambdaignore | xargs zip -9qyr /tmp/package.zip . -x
RUN cd /var/task; zip -r9q --symlinks /tmp/package.zip lib/*.so*
#RUN cd /var/task; zip -r9q --symlinks /tmp/package.zip lib64/*.so*
RUN cd /var/task; zip -r9q /tmp/package.zip share

CMD ["/bin/bash"]
