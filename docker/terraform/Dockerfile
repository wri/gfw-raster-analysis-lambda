FROM hashicorp/terraform:0.12.26

RUN \
    apk add --no-cache \
        bash \
        make \
        python3 \
        py3-pip \
        zip \
    && pip3 install awscli

COPY scripts/infra /usr/local/bin/
COPY scripts/select_workspace /usr/local/bin/
COPY scripts/delete_workspace /usr/local/bin/