#!/bin/bash

# - This script is to be run in a clean Ubuntu 16 LTS machine, by a sudoer user.
# - VGG_FACE_SRC_FOLDER should not exist.
# - Python 3 is required.
# - All python dependencies are installed in a python virtual environment to avoid conflicts
#   with pre-installed python packages.
# - If you have a GPU and NVIDIA drivers in your PC, then PyTorch should support GPU automatically.
#   Otherwise, please refer to the PyTorch web page https://pytorch.org/ for specific installation instructions.

VGG_FACE_INSTALL_FOLDER="$HOME"
VGG_FACE_SRC_FOLDER="$VGG_FACE_INSTALL_FOLDER/vgg_face_search"
VGG_FACE_DEPENDENCIES_FOLDER="$VGG_FACE_SRC_FOLDER/dependencies"

# update repositories
sudo apt-get update

# install  dependencies
sudo apt-get install -y cmake
sudo apt-get install -y pkg-config
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y wget unzip

# pip and other python dependencies
sudo apt-get install -y python-pip python3-pip
sudo apt-get install -y python-dev python3-dev
sudo apt-get install -y libz-dev libjpeg-dev libfreetype6-dev
sudo apt-get install -y python3-tk

# setup folders and download git repo
cd $VGG_FACE_INSTALL_FOLDER
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
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
pip install scikit-image==0.14.2
pip install simplejson==3.8.2
pip install matplotlib==2.1.0
pip install opencv-python==4.2.0.32

# download Pytorch_Retinaface (Dec 2019)
wget https://github.com/biubug6/Pytorch_Retinaface/archive/96b72093758eeaad985125237a2d9d34d28cf768.zip -P /tmp
unzip /tmp/96b72093758eeaad985125237a2d9d34d28cf768.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/Pytorch_Retinaface* $VGG_FACE_DEPENDENCIES_FOLDER/Pytorch_Retinaface
mkdir $VGG_FACE_DEPENDENCIES_FOLDER/Pytorch_Retinaface/weights

# download models
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_256_pytorch.tar.gz -P /tmp/
tar -xvzf /tmp/senet50_256_pytorch.tar.gz -C $VGG_FACE_SRC_FOLDER/models/
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

# compile shot detector
cd $VGG_FACE_SRC_FOLDER/pipeline
mkdir build
cd build
cmake -DBoost_INCLUDE_DIR=/usr/include/ ../
make

# some minor adjustments
sed -i 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
sed -i 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
