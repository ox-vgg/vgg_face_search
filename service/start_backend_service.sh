#!/bin/bash

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<additional paths>
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
source ../../bin/activate
python backend.py
