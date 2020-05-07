# Dockerfile for the VFS application. See https://gitlab.com/vgg/vgg_face_search

# Start from a clean Ubuntu Xenial machine
FROM ubuntu:xenial

# For GPU support, start from an nvidia image
#FROM nvidia/cuda:8.0-devel-ubuntu16.04

# Update repositories and install all apt-get dependencies
RUN apt-get update && apt-get install -y \
    python-pip \
    python-dev \
    libz-dev \
    libjpeg-dev \
    libfreetype6-dev \
    screen \
    cmake \
    libsm6 \
    pkg-config \
    python3-pip \
    python3-dev \
    python3-tk \
    unzip \
    wget \
    nano \
    && apt-get install -y --no-install-recommends libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Change the default shell to be able to work with the virtualenv below
SHELL ["/bin/bash", "-c"]

# Create requirements file, install all requirements, download git repo and configure vgg_face_search
RUN pip install --upgrade pip \
    && pip install virtualenv zipp \
    && mkdir /webapps/ \
    && chmod 777 /webapps/ \
    && cd /webapps/ \
    && wget -t 0 https://gitlab.com/vgg/vgg_face_search/-/archive/master/vgg_face_search-master.zip -O /tmp/vgg_face_search.zip \
    && unzip /tmp/vgg_face_search.zip -d /webapps/ \
    && mv /webapps/vgg_face_search* /webapps/vgg_face_search \
    && cd /webapps/vgg_face_search \
    && virtualenv -p python3 . \
    && source ./bin/activate \
    && printf "torch==1.1.0\nPillow==6.1.0\nPyWavelets==1.1.1\ntorchvision==0.3.0\nscipy==1.2.0\nscikit-image==0.14.2\nsimplejson==3.8.2\nmatplotlib==2.1.0\nopencv-python==4.2.0.32" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm  /tmp/requirements.txt \
    && sed -i 's/source ..\//source /g' /webapps/vgg_face_search/service/start_backend_service.sh \
    && sed -i 's/source ..\//source /g' /webapps/vgg_face_search/pipeline/start_pipeline.sh \
    && sed -i "s|ffmpeg|/webapps/vgg_face_search/dependencies/ffmpeg/ffmpeg|g" /webapps/vgg_face_search/pipeline/start_pipeline.sh \
    && sed -i 's/resnet50_256/senet50_256/g' /webapps/vgg_face_search/service/settings.py \
    && sed -i 's/localhost/0.0.0.0/g' /webapps/vgg_face_search/service/settings.py \
    && rm -rf /tmp/*.zip \
    && rm -rf /tmp/*.tar*

# Install Pytorch_Retinaface (Dec 2019) and download static ffmpeg
RUN wget https://github.com/biubug6/Pytorch_Retinaface/archive/96b72093758eeaad985125237a2d9d34d28cf768.zip -P /tmp \
    && unzip /tmp/96b72093758eeaad985125237a2d9d34d28cf768.zip -d /webapps/vgg_face_search/dependencies/ \
    && mv /webapps/vgg_face_search/dependencies/Pytorch_Retinaface* /webapps/vgg_face_search/dependencies/Pytorch_Retinaface \
    && mkdir /webapps/vgg_face_search/dependencies/Pytorch_Retinaface/weights \
    && wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O /tmp/ffmpeg-release-amd64-static.tar.xz \
    && tar -xf /tmp/ffmpeg-release-amd64-static.tar.xz -C /webapps/vgg_face_search/dependencies/ \
    && mv /webapps/vgg_face_search/dependencies/ffmpeg* /webapps/vgg_face_search/dependencies/ffmpeg \
    && rm -rf /tmp/*.zip \
    && rm -rf /tmp/*.tar*

# Compile shot detector
RUN cd /webapps/vgg_face_search/pipeline \
    && mkdir build \
    && cd build \
    && cmake -DBoost_INCLUDE_DIR=/usr/include/ ../ \
    && make

# Download models. This is only here for reference. The models should not be copied inside the image to reduce the image size.
# Use the suggested VOLUME below to mount the storage containing the models
#RUN wget http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/pytorch/senet50_256_pytorch.tar.gz -P /tmp/ \
#    && tar -xvzf /tmp/senet50_256_pytorch.tar.gz -C /webapps/vgg_face_search/models/ \
#    && cd /webapps/vgg_face_search/dependencies/Pytorch_Retinaface/weights \
#    && wget http://www.robots.ox.ac.uk/~vgg/software/vff/downloads/models/Pytorch_Retinaface/Resnet50_Final.pth

# The customizations below can be done via the command line and are not needed here.
# If you need them create a new Docker image derived from this one and define them.
#EXPOSE 55302
#VOLUME /webapps/vgg_face_search/models \
#       /webapps/vgg_face_search/dependencies/Pytorch_Retinaface/weights \
#       /webapps/vgg_face_search/features \ # in case the features are precomputed
#ENTRYPOINT /webapps/vgg_face_search/service/start_backend_service.sh
