#!/bin/bash

# - This script has been tested in a clean macOS High Sierra 10.13.3
# - It assumes Homebrew is available in the system (https://brew.sh/).
# - If used to install pip and virtualenv, it will require a sudoer user.
# - It asumes the CUDA driver and toolkit are already installed in standard locations. Define CUDA_HOME if possible. See the
#   commented environment variable definitions below.
# - All python dependencies are installed in a python virtual environment to avoid conflicts with pre-installed python packages
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
#sudo pip install virtualenv

# caffe dependencies
brew install -vd snappy leveldb gflags glog szip lmdb
brew install -vd hdf5 opencv
brew install -vd tesseract
brew install -vd protobuf
brew install -vd boost@1.59 boost-python@1.59
brew install -vd openblas
brew link --force boost@1.59
brew link --force boost-python@1.59

# download vgg_face_search git repo and create virtual environment
wget https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip
unzip /tmp/vgg_face_search.zip -d $VGG_FACE_INSTALL_FOLDER/
mv $VGG_FACE_INSTALL_FOLDER/vgg_face_search*  $VGG_FACE_SRC_FOLDER
cd $VGG_FACE_SRC_FOLDER
virtualenv .
sed -i '.sed' 's/CUDA_ENABLED = False/CUDA_ENABLED = True/g' $VGG_FACE_SRC_FOLDER/service/settings.py
source ./bin/activate

# register the numpy version used by opencv, so that python-opencv can be used in the virtualenv
BREW_NUMPY_VERSION=$(brew info numpy | grep Cellar/numpy | awk -F '[/| |_]' '{print $6}' )

# register the protobuf installed by homebrew, so that pycaffe can be used in the virtualenv
PROTOBUF_NUMPY_VERSION=$(brew info protobuf | grep Cellar/protobuf | awk -F '[/| |_]' '{print $6}' )

# python dependencies
pip install simplejson==3.8.2
pip install Pillow==2.3.0
pip install numpy==$BREW_NUMPY_VERSION
pip install Cython==0.27.3
pip install scipy==0.18.1
pip install matplotlib==2.1.0
pip install scikit-image==0.13.1
pip install protobuf==$PROTOBUF_NUMPY_VERSION
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

# download models
cd $VGG_FACE_SRC_FOLDER/models
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/resnet50_256.caffemodel
wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/256/resnet50_256.prototxt

# download static ffmpeg
wget https://ffmpeg.zeranoe.com/builds/macos64/static/ffmpeg-20180411-9825f77-macos64-static.zip -P /tmp
unzip /tmp/ffmpeg-20180411-9825f77-macos64-static.zip -d $VGG_FACE_DEPENDENCIES_FOLDER/
mv $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg*  $VGG_FACE_DEPENDENCIES_FOLDER/ffmpeg
sed -i '.sed' "s|ffmpeg|${VGG_FACE_DEPENDENCIES_FOLDER}/ffmpeg/bin/ffmpeg|g" $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# remove all zips
rm -rf /tmp/*.zip

# compile caffe-fast-rcnn
cd $VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn
cp Makefile.config.example Makefile.config
sed -i '.sed' 's/# OPENCV_VERSION := 3/OPENCV_VERSION := 3/g' Makefile.config  # homebrew will install opencv3
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

# Make cv2 available in the virtualenv
CV2_LOCATION=$(brew info opencv | grep /usr/local/Cellar | cut -d' ' -f1)
ln -s $CV2_LOCATION/lib/python2.7/site-packages/cv2.so $VGG_FACE_SRC_FOLDER/lib/python2.7/cv2.so

# some minor adjustments
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
sed -i '.sed' 's/source ..\//source /g' $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh

# add dylb paths to caffe and cuda library dependencies.
# If preferred, instead of modifying $HOME/.profile, the 'export' command should be executed before calling python in:
#    $VGG_FACE_SRC_FOLDER/service/start_backend_service.sh
#    $VGG_FACE_SRC_FOLDER/pipeline/start_pipeline.sh
echo "export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDAHOME/lib:$VGG_FACE_DEPENDENCIES_FOLDER/face-py-faster-rcnn/caffe-fast-rcnn/build/lib:$BREW_BOOST_ROOT/lib" >> $HOME/.profile
source $HOME/.profile
