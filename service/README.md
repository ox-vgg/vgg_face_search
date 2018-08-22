VGG Face Search Service
=======================

Author:

 + Ernesto Coto, University of Oxford – <ecoto@robots.ox.ac.uk>

Derived for the original version by Omkar Parkhi, VGG - University of Oxford, 2014.

Installation Instructions
-------------------------

The service can run with or without GPU support. However, a different face-detector is used on each case, so different dependencies are needed.

Without GPU support, a CPU face-detector extracted from [facenet](https://github.com/davidsandberg/facenet) is used, along with [Caffe](https://github.com/BVLC/caffe) and pycaffe.

With GPU support, the GPU [Faster R-CNN python implementation](https://github.com/rbgirshick/py-faster-rcnn ) is used. This software includes its own version of Caffe, so there is no need to download another version.

In the `install` folder at the root of the repository you will find installation scripts for Ubuntu and macOS, for the GPU-only and the CPU-only versions of the service. For having a service that supports both the GPU and the CPU just use the GPU-only installer and then download `facenet` into the `dependencies` folder. See the CPU-only installation script for the way to do it.

Usage
-----

Before running the service for the first time, please check the `settings.py` file:

 1. Check that the `CUDA_ENABLED` flag is set to `False` if you used the CPU-only installation script or set to `True` if you used the GPU-only installation.
 2. Make sure that `DEPENDENCIES_PATH` points to the location of the place where the dependency libraries (e.g. Caffe) are installed.
 3. Make sure that `DATASET_FEATS_FILE` points to the location of your dataset features file. If you do not have one, you won't be able to run the service until you run the feature computation `pipeline`. See the README in the `pipeline` directory at the root of the repository.
 4. Adjust `MAX_RESULTS_RETURN` as you wish.
 5. Only change the rest of the settings if you really know what you are doing.

If you already have adjusted the settings and have a dataset feature file, you should be ready to start the service. To do so, start a command-line terminal and then execute the `start_backend_service.sh` script (`start_backend_service.bat` for Windows). Use that script file to define or modify any environment variables required by your local setup.

The service should be reachable at the HOST and PORT specified in the settings.

Advanced Result Ranking
-----------------------

In general, the ranking of results is done by computing distances between the feature vectors extracted from the dataset and the feature vectors extracted from the training images submitted via the API of this service. For very large datasets, this distance computation can take an undesirable long time.

In order to speed up this computation, the features can be stored in KD-trees and separated from the rest of the dataset information (such as image paths and face detections). However, at present you can only "move" to this kind of computation if you already have a pre-computed dataset file in the old format. If you do have a pre-computed dataset file, take a look at `databaseutils.py`. In that file you will find functions to: a) extract the feature vectors from the dataset file and save them as KD-trees, b) Remove the feature vectors from the dataset file (since you don't want to load them in memory twice).

Note that depending on the structure of the source pre-computed dataset file, one or more KD-trees file and new dataset files will be produced.

Once you have "moved" to the new dataset representation, go to `settings.py` and set `KDTREES_RANKING_ENABLED` to `True` plus change the `DATASET_FEATS_FILE` variable to point to the new dataset file without features. Then restart the service.
