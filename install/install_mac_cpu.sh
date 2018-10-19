#!/bin/bash

# - This script has been tested in a clean macOS High Sierra 10.13.3
# - It assumes Homebrew is available in the system (https://brew.sh/).
# - If used to install pip and virtualenv, it will require a sudoer user.
# - All python dependencies are installed in a python virtual environment to avoid conflicts with pre-installed python packages
# - VGG_FACE_SRC_FOLDER should not exist
# - Caffe is compiled for CPU use only.

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
#sudo pip install virtualenv

# caffe dependencies
brew install -vd snappy leveldb gflags glog szip lmdb
brew install -vd hdf5 opencv
brew install -vd tesseract
brew install -vd protobuf@2.6
brew install -vd boost@1.59 boost-python@1.59
brew install -vd openblas
brew link --force protobuf@2.6
brew link --force boost@1.59
brew link --force boost-python@1.59

# Download vgg_face_search git repo and create virtual environment
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
sed -i '.sed' 's/resnet50_256/senet50_256/g' $VGG_FACE_SRC_FOLDER/service/settings.py
cd $VGG_FACE_SRC_FOLDER
virtualenv .
source ./bin/activate

# register the numpy version used by opencv, so that python-opencv can be used in the virtualenv
BREW_NUMPY_VERSION=$(brew info numpy | grep Cellar/numpy | awk -F '[/| |_]' '{print $6}'  )

# python dependencies
pip install simplejson==3.8.2
pip install Pillow==2.3.0
pip install numpy==$BREW_NUMPY_VERSION
pip install lxml==4.1.1
pip install scipy==0.18.1
pip install matplotlib==2.1.0
pip install scikit-image==0.13.1
pip install scikit-learn==0.19.1
pip install dill==0.2.8.2
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.0-py2-none-any.whl

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

# download davidsandberg's facenet (Dec 2017)
wget https://github.com/davidsandberg/facenet/archive/28d3bf2fa7254037229035cac398632a5ef6fc24.zip -P /tmp
unzip /tmp/28d3bf2fa7254037229035cac398632a5ef6fc24.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/facenet* $VGG_FACE_DEPENDENCIES_FOLDER/facenet

# download models
cd $VGG_FACE_SRC_FOLDER/models
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.caffemodel
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.prototxt

# download static ffmpeg
wget https://ffmpeg.zeranoe.com/builds/macos64/static/ffmpeg-20180411-9825f77-macos64-static.zip -P /tmp
unzip /tmp/ffmpeg-20180411-9825f77-macos64-static.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg*  $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg
sed -i '.sed' "s|ffmpeg|${VGG_FACE_DEPENDENCIES_FOLDER}/ffmpeg/bin/ffmpeg|g" $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# remove all zips
rm -rf /tmp/*.zip

# compile caffe
cd $VGG_FACE_DEPENDENCIES_FOLDER/caffe
cp Makefile.config.example Makefile.config
sed -i '.sed' 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
sed -i '.sed' 's/# OPENCV_VERSION := 3/OPENCV_VERSION := 3/g' Makefile.config # homebrew will install opencv3
sed -i '.sed' 's/BLAS := atlas/BLAS := open/g' Makefile.config
sed -i '.sed' 's/# BLAS_INCLUDE := $(/BLAS_INCLUDE := $(/g' Makefile.config
sed -i '.sed' 's/# BLAS_LIB := $(/BLAS_LIB := $(/g' Makefile.config
sed -i '.sed' 's/# PYTHON_INCLUDE +=/PYTHON_INCLUDE +=/g' Makefile.config
make all
make pycaffe

# compile shot detector
BREW_BOOST_ROOT=$(brew info boost@1.59 | grep Cellar/boost | awk '{print $1}' )
cd $VGG_FACE_SRC_FOLDER/pipeline
mkdir build
cd build
cmake -DBOOST_ROOT=$BREW_BOOST_ROOT ../
make

# some minor adjustments
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# Make cv2 available in the virtualenv
CV2_LOCATION=$(brew info opencv | grep /usr/local/Cellar | cut -d' ' -f1)
ln -s $CV2_LOCATION/lib/python2.7/site-packages/cv2.so $VGG_FACE_SRC_FOLDER/lib/python2.7/cv2.so
