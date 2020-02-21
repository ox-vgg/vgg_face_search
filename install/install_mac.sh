#!/bin/bash

# - This script has been tested in a clean macOS High Sierra 10.13.3
# - It assumes Homebrew is available in the system (https://brew.sh/).
# - If used to install pip, it will require a sudoer user.
# - All python dependencies are installed in a python virtual environment to avoid conflicts
#   with pre-installed python packages
# - VGG_FACE_SRC_FOLDER should not exist
# - If you have a GPU and NVIDIA drivers in your PC, then PyTorch should support GPU automatically.
#   Otherwise, please refer to the PyTorch web page https://pytorch.org/ for specific installation instructions.

VGG_FACE_INSTALL_FOLDER="$HOME"
VGG_FACE_SRC_FOLDER="$VGG_FACE_INSTALL_FOLDER/vgg_face_search"
VGG_FACE_DEPENDENCIES_FOLDER="$VGG_FACE_SRC_FOLDER/dependencies"

# update repositories
brew update

# install some utils
brew install wget
brew install cmake
brew install jpeg libpng libtiff freetype

# install pip and virtualenv, which requires sudo access
#wget https://bootstrap.pypa.io/get-pip.py -P /tmp
#sudo python /tmp/get-pip.py

# dependencies
brew install -vd python3
brew install -vd boost boost-python3
brew install -vd libomp

# download vgg_face_search git repo
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
sed -i '.sed' 's/resnet50_256/senet50_256/g' $VGG_FACE_SRC_FOLDER/service/settings.py

# create virtual environment and install python dependencies
cd $VGG_FACE_SRC_FOLDER
pip3 install virtualenv
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
wget https://ffmpeg.zeranoe.com/builds/macos64/static/ffmpeg-4.1.1-macos64-static.zip -P /tmp
unzip /tmp/ffmpeg-4.1.1-macos64-static.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg*  $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg
sed -i '.sed' "s|ffmpeg|${VGG_FACE_DEPENDENCIES_FOLDER}/ffmpeg/bin/ffmpeg|g" $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# remove all zips
rm -rf /tmp/*.zip

# compile shot detector
BREW_BOOST_ROOT=$(brew info boost | grep Cellar/boost | awk '{print $1}' )
cd $VGG_FACE_SRC_FOLDER/pipeline
mkdir build
cd build
cmake -DBOOST_ROOT=$BREW_BOOST_ROOT ../
make

# some minor adjustments
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
