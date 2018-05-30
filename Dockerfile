FROM lambdalinux/baseimage-amzn:2017.03-004

RUN yum install python27-devel gcc python27-pip zip
COPY . /build
WORKDIR /build

RUN \
    pip install -r requirements.txt; \
    rm -rf /build/*
    
 
