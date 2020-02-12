#!/bin/bash

# - This script has been tested in a clean macOS High Sierra 10.13.3
# - It assumes Homebrew is available in the system (https://brew.sh/).
# - If used to install pip and protobuf, it will require a sudoer user.
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
brew install -vd boost@1.59 boost-python
brew install -vd openblas
brew link --force opencv@2
brew link --overwrite --force boost@1.59

# download vgg_face_search git repo and create virtual environment
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
sed -i '.sed' 's/CUDA_ENABLED = False/CUDA_ENABLED = True/g' $VGG_FACE_SRC_FOLDER/service/settings.py
sed -i '.sed' 's/resnet50_256/senet50_256/g' $VGG_FACE_SRC_FOLDER/service/settings.py

# download, compile and install protobuf-3.1.0, newer versions of protobuf won't work
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.zip -O /tmp/protobuf-cpp-3.1.0.zip
unzip /tmp/protobuf-cpp-3.1.0.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
cd $VGG_FACE_DEPENDENCIES_FOLDER/protobuf-3.1.0/
./configure CC=clang CXX=clang++ CXXFLAGS='-std=c++11 -stdlib=libc++ -O3 -g' LDFLAGS='-stdlib=libc++' LIBS="-lc++ -lc++abi"
make -j 4
sudo make install

# python dependencies
cd $VGG_FACE_SRC_FOLDER
pip install simplejson==3.8.2
pip install Pillow==2.3.0
pip install numpy==1.16.2
pip install Cython==0.27.3
pip install scipy==0.18.1
pip install matplotlib==2.1.0
pip install scikit-image==0.13.1
pip install protobuf==3.1.0
pip install easydict==1.7
pip install pyyaml==3.12
pip install six==1.11.0
pip install dill==0.2.8.2

# install face-py-faster-rcnn
wget https://github.com/playerkk/face-py-faster-rcnn/archive/9d8c143e0ff214a1dcc6ec5650fb5045f3002c2c.zip -P /tmp
unzip /tmp/9d8c143e0ff214a1dcc6ec5650fb5045f3002c2c.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn-* $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn
wget https://github.com/rbgirshick/caffe-fast-rcnn/archive/0dcd397b29507b8314e252e850518c5695efbb83.zip -P /tmp
unzip /tmp/0dcd397b29507b8314e252e850518c5695efbb83.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn
rm -r $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn
mv $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn-* $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn
cd $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/lib
make
mkdir $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/data/faster_rcnn_models
cd $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/data/faster_rcnn_models
wget http://supermoe.cs.umass.edu/%7Ehzjiang/data/vgg16_faster_rcnn_iter_80000.caffemodel

# download SENet modifications to caffe (Sep 2017) and apply them to caffe-fast-rcnn
wget https://github.com/lishen-shirley/SENet/archive/c8f7b4e311fc9b5680047e14648fde86fb23cb17.zip -P /tmp
unzip /tmp/c8f7b4e311fc9b5680047e14648fde86fb23cb17.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/SENet* $VGG_FACE_DEPENDENCIES_FOLDER/SENet
CAFFE_FASTER_RCNN_FOLDER="$VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn"
cp -v $VGG_FACE_DEPENDENCIES_FOLDER/SENet/include/caffe/layers/* $CAFFE_FASTER_RCNN_FOLDER/include/caffe/layers/
cp -v $VGG_FACE_DEPENDENCIES_FOLDER/SENet/src/caffe/layers/* $CAFFE_FASTER_RCNN_FOLDER/src/caffe/layers/

# download models
cd $VGG_FACE_SRC_FOLDER/models
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.caffemodel
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/senet50_256.prototxt

# download static ffmpeg
wget https://ffmpeg.zeranoe.com/builds/macos64/static/ffmpeg-4.1.1-macos64-static.zip -P /tmp
unzip /tmp/ffmpeg-4.1.1-macos64-static.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg*  $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg
sed -i '.sed' "s|ffmpeg|${VGG_FACE_DEPENDENCIES_FOLDER}/ffmpeg/bin/ffmpeg|g" $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# remove all zips
rm -rf /tmp/*.zip

# compile caffe-fast-rcnn
cd $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn
cp Makefile.config.example Makefile.config
sed -i '.sed' 's/BLAS := atlas/BLAS := open/g' Makefile.config
sed -i '.sed' 's/# BLAS_INCLUDE := $(/BLAS_INCLUDE := $(/g' Makefile.config
sed -i '.sed' 's/# BLAS_LIB := $(/BLAS_LIB := $(/g' Makefile.config
sed -i '.sed' 's/# PYTHON_INCLUDE +=/PYTHON_INCLUDE +=/g' Makefile.config
sed -i '.sed' 's/# Configure build/CXXFLAGS += -std=c++11/g' Makefile
sed -i '.sed' 's/boost_python/boost_python27/g' Makefile
make all
make pycaffe

# compile shot detector
BREW_BOOST_ROOT=$(brew info boost@1.59 | grep Cellar/boost | awk '{print $1}' )
cd $VGG_FACE_SRC_FOLDER/pipeline
mkdir build
cd build
cmake -DBOOST_ROOT=$BREW_BOOST_ROOT ../
make

# Make cv2 available locally
CV2_LOCATION=$(brew info opencv@2 | grep /usr/local/Cellar | cut -d' ' -f1)
cp $CV2_LOCATION/lib/python2.7/site-packages/cv2.so $HOME/Library/Python/2.7/lib/python/site-packages

# some minor adjustments
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# add dylb paths to caffe and cuda library dependencies.
# If preferred, instead of modifying $HOME/.profile, the 'export' command should be executed before calling python in:
#    $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
#    $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
echo "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDAHOME/lib:$VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn/build/lib:$BREW_BOOST_ROOT/lib" >> $HOME/.profile
source $HOME/.profile
