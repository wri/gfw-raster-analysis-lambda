#!/bin/bash

# Create and activate a virtualevn
virtualenv env
source env/bin/activate
pip install pip --upgrade

# install Cython first - required when installing
# fiona directly from github, and for some reason
# doesn't know to install automatically
pip install Cython
pip install -r requirements.txt

echo "-----"
printf "Virtualenv created, activate with: \nsource env/bin/activate"
