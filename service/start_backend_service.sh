#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/webapps/vgg_face_search/dependencies/boost_1_57_0/lib:/usr/local/cuda-7.5/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/webapps/vgg_face_search/dependencies/cudnn-v4.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/webapps/vgg_face_search/dependencies/cudnn-v5.0/lib64
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
source ../bin/activate
python backend.py
