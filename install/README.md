VGG Face Search Installers
==========================

These scripts are meant for local deployment on your computer. This means that they will download and install third-party software and data in your computer, as well as compile and configure source code. The scripts are experimental and are intended for **Software Developers**. Regular users are strongly advised to use the docker version of the application.

In all cases, you will need a C++ compiler and Python 3 installed on your system. For GPU support you will need the NVIDIA drivers and the CUDA Toolkit in your system. Please be aware that the scripts for GPU might fail depending on your particular CUDA setup (version, location of the CUDA library in your system, etc.). cuDNN is not used anywhere so if you want, for instance, Caffe with cuDNN, you will need to reconfigure and recompile Caffe by yourself.

All scripts contain requirements and some instructions at the beginning of the file. Please read them before attempting the deployment.

Remember that before the first use you need to configure `vgg_face_search`. See the `Usage` section of the README in the root folder of this repository.
