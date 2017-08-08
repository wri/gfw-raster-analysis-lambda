#!/bin/bash

# Create and activate a virtualevn
virtualenv env --python=/home/mmcfarland/.localpython/bin/python2.7
source env/bin/activate
pip install pip --upgrade
pip install -r requirements.txt

echo "-----"
printf "Virtualenv created, activate with: \nsource env/bin/activate"
