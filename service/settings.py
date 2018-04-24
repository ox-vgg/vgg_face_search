__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import os
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

HOST = 'localhost'

PORT = 55302

MAX_RESULTS_RETURN = 1000

MAX_RESULTS_SCORE = 0.9

CUDA_ENABLED = False

DEPENDENCIES_PATH = os.path.join(FILE_DIR, '..', 'dependencies')

DATASET_FEATS_FILE = os.path.join(FILE_DIR, '..', 'features', 'database.pkl')

FEATURES_CAFFE_MODEL = os.path.join(FILE_DIR, '..', 'models', 'resnet50_256.caffemodel')

FEATURES_CAFFE_PROTOTXT = os.path.join(FILE_DIR, '..', 'models', 'resnet50_256.prototxt')

FEATURES_CAFFE_LAYER = 'feat_extract'

FEATURES_VECTOR_SIZE = 256

GPU_FACE_DETECTION_CAFFE_MODEL = os.path.join(DEPENDENCIES_PATH, 'face-py-faster-rcnn', 'data', 'faster_rcnn_models', 'vgg16_faster_rcnn_iter_80000.caffemodel')

NUMBER_OF_HELPER_WORKERS = 8
