#!/bin/bash

# Create and activate a virtualevn
virtualenv env
source env/bin/activate
pip install pip --upgrade
pip install -r requirements.txt

echo "-----"
printf "Virtualenv created, activate with: \nsource env/bin/activate"
