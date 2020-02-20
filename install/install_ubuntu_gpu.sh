#!/bin/bash

# - This script is to be run in a clean Ubuntu 16 LTS machine, by a sudoer user.
# - VGG_FACE_SRC_FOLDER should not exist
# - All python dependencies are installed in a python virtual environment to avoid conflicts with pre-installed python packages
# - Make sure the NVIDIA CUDA Toolkit is installed and that 'nvcc' is reachable. See the commented environment variable
#   definitions below. The same variables should be available when actually running the service, and should be added to:
#     * $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
#     * $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
# - This script does not include the use of cuDNN. If you want to use it, you will need to change the Makefile.config of
#   caffe-fast-rcnn and recompile it.

#export PATH=/usr/local/cuda/bin:$PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib
VGG_FACE_INSTALL_FOLDER="$HOME"
VGG_FACE_SRC_FOLDER="$VGG_FACE_INSTALL_FOLDER/vgg_face_search"
VGG_FACE_DEPENDENCIES_FOLDER="$VGG_FACE_SRC_FOLDER/dependencies"

# update repositories
sudo apt-get update

# Caffe  dependencies
sudo apt-get install -y cmake
sudo apt-get install -y pkg-config
sudo apt-get install -y libgoogle-glog-dev
sudo apt-get install -y libhdf5-serial-dev
sudo apt-get install -y liblmdb-dev
sudo apt-get install -y libleveldb-dev
sudo apt-get install -y libprotobuf-dev
sudo apt-get install -y protobuf-compiler
sudo apt-get install -y libopencv-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libsnappy-dev
sudo apt-get install -y libgflags-dev
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y wget unzip

# pip and other python dependencies
sudo apt-get install -y python-pip python3-pip
sudo apt-get install -y python-dev python3-dev
sudo apt-get install -y gfortran
sudo apt-get install -y libz-dev libjpeg-dev libfreetype6-dev
sudo apt-get install -y libxml2-dev libxslt1-dev
sudo apt-get install -y python-opencv python3-tk

# setup folders and download git repo
cd $VGG_FACE_INSTALL_FOLDER
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
sed -i 's/CUDA_ENABLED = False/CUDA_ENABLED = True/g' $VGG_FACE_SRC_FOLDER/service/settings.py
sed -i 's/resnet50_256/senet50_256/g' $VGG_FACE_SRC_FOLDER/service/settings.py

# create virtual environment and install python dependencies
cd $VGG_FACE_SRC_FOLDER
sudo pip install virtualenv
pip install --upgrade pip
pip install zipp
virtualenv -p python3 .
source ./bin/activate
pip install torch==1.1.0
pip install Pillow==6.1.0
pip install PyWavelets==1.1.1
pip install torchvision==0.3.0
pip install scipy==1.2.0
pip install opencv-python==4.2.0.32
pip install scikit-image==0.14.2
pip install simplejson==3.8.2
pip install matplotlib==2.1.0
pip install protobuf==3.11

# download caffe
wget https://github.com/BVLC/caffe/archive/1.0.zip -P /tmp
unzip /tmp/1.0.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/caffe* $VGG_FACE_DEPENDENCIES_FOLDER/caffe

# download SENet modifications to caffe (Sep 2017) and apply them
wget https://github.com/lishen-shirley/SENet/archive/c8f7b4e311fc9b5680047e14648fde86fb23cb17.zip -P /tmp
unzip /tmp/c8f7b4e311fc9b5680047e14648fde86fb23cb17.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/SENet* $VGG_FACE_DEPENDENCIES_FOLDER/SENet
cp -v $VGG_FACE_DEPENDENCIES_FOLDER/SENet/include/caffe/layers/* $VGG_FACE_DEPENDENCIES_FOLDER/caffe/include/caffe/layers/
cp -v $VGG_FACE_DEPENDENCIES_FOLDER/SENet/src/caffe/layers/* $VGG_FACE_DEPENDENCIES_FOLDER/caffe/src/caffe/layers/

# download Pytorch_Retinaface (Dec 2019)
wget https://github.com/biubug6/Pytorch_Retinaface/archive/96b72093758eeaad985125237a2d9d34d28cf768.zip -P /tmp
unzip /tmp/96b72093758eeaad985125237a2d9d34d28cf768.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/Pytorch_Retinaface* $VGG_FACE_DEPENDENCIES_FOLDER/Pytorch_Retinaface
mkdir $VGG_FACE_DEPENDENCIES_FOLDER/Pytorch_Retinaface/weights

# download models
cd $VGG_FACE_SRC_FOLDER/models
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.caffemodel
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.prototxt
cd $VGG_FACE_DEPENDENCIES_FOLDER/Pytorch_Retinaface/weights
wget http://www.robots.ox.ac.uk/~vgg/software/vff/downloads/models/Pytorch_Retinaface/Resnet50_Final.pth

# download static ffmpeg
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O /tmp/ffmpeg-release-amd64-static.tar.xz
tar -xf /tmp/ffmpeg-release-amd64-static.tar.xz -C $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg*  $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg
sed -i "s|ffmpeg|${VGG_FACE_DEPENDENCIES_FOLDER}/ffmpeg/ffmpeg|g" $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# remove all zips
rm -rf /tmp/*.zip
rm -rf /tmp/*.tar*

# compile caffe
cd $VGG_FACE_DEPENDENCIES_FOLDER/caffe
cp Makefile.config.example Makefile.config
sed -i 's/# WITH_PYTHON_LAYER/WITH_PYTHON_LAYER/g' Makefile.config
sed -i 's|PYTHON_INCLUDE := /usr/include/python2.7|#PYTHON_INCLUDE := /usr/include/python2.7|g' Makefile.config
sed -i 's|/usr/lib/python2.7/|#/usr/lib/python2.7/|g' Makefile.config
sed -i 's|# PYTHON_LIBRARIES := boost_python3|PYTHON_LIBRARIES := boost_python-py35|g' Makefile.config
sed -i 's|# PYTHON_INCLUDE := /usr/include/python3.5m|PYTHON_INCLUDE := /usr/include/python3.5m '$VGG_FACE_SRC_FOLDER'/lib/python3.5/site-packages/numpy/core/include/ #|g' Makefile.config
sed -i 's/INCLUDE_DIRS :=/INCLUDE_DIRS := \/usr\/include\/hdf5\/serial\/ /g' Makefile.config
sed -i 's/LIBRARY_DIRS :=/LIBRARY_DIRS := \/usr\/lib\/x86_64-linux-gnu\/hdf5\/serial\/ /g' Makefile.config
sed -i 's/# Configure build/CXXFLAGS += -std=c++11/g' Makefile
make all
make pycaffe

# compile shot detector
cd $VGG_FACE_SRC_FOLDER/pipeline
mkdir build
cd build
cmake -DBoost_INCLUDE_DIR=/usr/include/ ../
make

# some minor adjustments
sed -i 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
sed -i 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
