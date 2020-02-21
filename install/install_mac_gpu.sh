#!/bin/bash

# - This script has been tested in a clean macOS High Sierra 10.13.3
# - It assumes Homebrew is available in the system (https://brew.sh/).
# - If used to install pip, it will require a sudoer user.
# - All python dependencies are installed in a python virtual environment to avoid conflicts with pre-installed python packages
# - It asumes the CUDA driver and toolkit are already installed in standard locations. Define CUDA_HOME if possible. See the
#   commented environment variable definitions below.
# - VGG_FACE_SRC_FOLDER should not exist
# - See also the last part of the script where $HOME/.profile is modified.
# - Compilation notes:
#   - Use Xcode CommandLineTools for macOS 10.12 (v8.1 or above)
#   - See https://github.com/CharlesShang/TFFRCNN/issues/21

#export PATH=/usr/local/cuda/bin:$PATH
#export CUDAHOME=/usr/local/cuda
VGG_FACE_INSTALL_FOLDER="$HOME"
VGG_FACE_SRC_FOLDER="$VGG_FACE_INSTALL_FOLDER/vgg_face_search"
VGG_FACE_DEPENDENCIES_FOLDER="$VGG_FACE_SRC_FOLDER/dependencies"

# update repositories
brew update

# install some utils
brew install wget
brew install cmake
brew install jpeg libpng libtiff

# install pip and virtualenv, which requires sudo access
#wget https://bootstrap.pypa.io/get-pip.py -P /tmp
#sudo python /tmp/get-pip.py

# caffe dependencies
brew install -vd snappy leveldb gflags glog szip lmdb
brew install -vd hdf5
brew install -vd opencv@2
brew install -vd tesseract
brew install -vd python3
brew install -vd boost boost-python3
brew install -vd openblas
brew link --force opencv@2

# torch dependencies
brew install libomp

# download vgg_face_search git repo and create virtual environment
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
sed -i '.sed' 's/CUDA_ENABLED = False/CUDA_ENABLED = True/g' $VGG_FACE_SRC_FOLDER/service/settings.py
sed -i '.sed' 's/resnet50_256/senet50_256/g' $VGG_FACE_SRC_FOLDER/service/settings.py

# python dependencies
cd $VGG_FACE_SRC_FOLDER
pip3 install virtualenv
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

# download, compile and install protobuf-3.1.0, newer versions of protobuf won't work
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.zip -O /tmp/protobuf-cpp-3.1.0.zip
unzip /tmp/protobuf-cpp-3.1.0.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
cd $VGG_FACE_DEPENDENCIES_FOLDER/protobuf-3.1.0/
./configure CC=clang CXX=clang++ CXXFLAGS='-std=c++11 -stdlib=libc++ -O3 -g' LDFLAGS='-stdlib=libc++' LIBS="-lc++ -lc++abi"
make -j 4
make install
pip install protobuf==3.1.0

# download caffe 1.0
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
wget https://ffmpeg.zeranoe.com/builds/macos64/static/ffmpeg-4.1.1-macos64-static.zip -P /tmp
unzip /tmp/ffmpeg-4.1.1-macos64-static.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg*  $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg
sed -i '.sed' "s|ffmpeg|${VGG_FACE_DEPENDENCIES_FOLDER}/ffmpeg/bin/ffmpeg|g" $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# remove all zips
rm -rf /tmp/*.zip

# obtain the location of python3 dev files
PYTHON3_DIR=$(brew --prefix python3)

# compile caffe
cd $VGG_FACE_DEPENDENCIES_FOLDER/caffe
cp Makefile.config.example Makefile.config
sed -i '.sed' 's|PYTHON_INCLUDE := /usr/include/python2.7|#PYTHON_INCLUDE := /usr/include/python2.7|g' Makefile.config
sed -i '.sed' 's|/usr/lib/python2.7/|#/usr/lib/python2.7/|g' Makefile.config
sed -i '.sed' 's|BLAS := atlas|BLAS := open|g' Makefile.config
sed -i '.sed' 's/# BLAS_INCLUDE := $(/BLAS_INCLUDE := $(/g' Makefile.config
sed -i '.sed' 's/# BLAS_LIB := $(/BLAS_LIB := $(/g' Makefile.config
sed -i '.sed' 's|# PYTHON_INCLUDE := /usr/include/python3.5m|PYTHON_INCLUDE := '$PYTHON3_DIR'/Frameworks/Python.framework/Versions/3.7/include/python3.7m '$VGG_FACE_SRC_FOLDER'/lib/python3.7/site-packages/numpy/core/include/ #|g' Makefile.config
sed -i '.sed' 's|# PYTHON_LIBRARIES := boost_python3 python3.5m|PYTHON_LIBRARIES := boost_python37 python3.7m|g' Makefile.config
sed -i '.sed' 's|# WITH_PYTHON_LAYER|WITH_PYTHON_LAYER|g' Makefile.config
sed -i '.sed' 's|PYTHON_LIB :=|PYTHON_LIB :='$PYTHON3_DIR'/Frameworks/Python.framework/Versions/3.7/lib|g' Makefile.config
sed -i '.sed' 's|# Configure build|CXXFLAGS += -std=c++11|g' Makefile
make all
make pycaffe

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

# add dylb paths to caffe and cuda library dependencies.
# If preferred, instead of modifying $HOME/.profile, the 'export' command should be executed before calling python in:
#    $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
#    $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
echo "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDAHOME/lib" >> $HOME/.profile
source $HOME/.profile
