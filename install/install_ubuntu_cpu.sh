#!/bin/bash

# - This script is to be run in a clean Ubuntu 16 LTS machine, by a sudoer user.
# - VGG_FACE_SRC_FOLDER should not exist
# - Caffe is compiled for CPU use only.

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
sudo apt-get install -y python-pip
sudo apt-get install -y python-dev
sudo apt-get install -y gfortran
sudo apt-get install -y libz-dev libjpeg-dev libfreetype6-dev
sudo apt-get install -y libxml2-dev libxslt1-dev
sudo apt-get install -y python-opencv python-tk

# setup folders and download git repo
cd $VGG_FACE_INSTALL_FOLDER
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
sed -i 's/resnet50_256/senet50_256/g' $VGG_FACE_SRC_FOLDER/service/settings.py

# create virtual environment and install python dependencies
cd $VGG_FACE_SRC_FOLDER
sudo pip install virtualenv
virtualenv .
source ./bin/activate
pip install --upgrade pip
pip install setuptools==40.4.3
pip install simplejson==3.8.2
pip install Pillow==6.1.0
pip install numpy==1.13.3
pip install lxml==4.1.1
pip install scipy==0.18.1
pip install matplotlib==2.1.0
pip install scikit-image==0.13.1
pip install scikit-learn==0.19.1
pip install dill==0.2.8.2
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl

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

# download davidsandberg's facenet (Dec 2017)
wget https://github.com/davidsandberg/facenet/archive/28d3bf2fa7254037229035cac398632a5ef6fc24.zip -P /tmp
unzip /tmp/28d3bf2fa7254037229035cac398632a5ef6fc24.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/facenet* $VGG_FACE_DEPENDENCIES_FOLDER/facenet

# download models
cd $VGG_FACE_SRC_FOLDER/models
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.caffemodel
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.prototxt

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
sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
sed -i 's/\/usr\/include\/python2.7/\/usr\/include\/python2.7 \/usr\/local\/lib\/python2.7\/dist-packages\/numpy\/core\/include/g' Makefile.config
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

# make cv2 available in the virtualenv
cp /usr/lib/python2.7/dist-packages/cv2*.so $VGG_FACE_SRC_FOLDER/lib/python2.7/cv2.so

# some minor adjustments
sed -i 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
sed -i 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
