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

FEATURES_MODEL_WEIGHTS = os.path.join(FILE_DIR, '..', 'models', 'senet50_256.pth')

FEATURES_MODEL_DEF = os.path.join(FILE_DIR, '..', 'models', 'senet50_256.py')

FEATURES_MODEL_LAYER = 'feat_extract'

FEATURES_VECTOR_SIZE = 256

FEATURES_EXTRACTION_TIMEOUT = 10

NUMBER_OF_HELPER_WORKERS = 8

KDTREES_RANKING_ENABLED = False

KDTREES_DATASET_SPLIT_SIZE = 100000

KDTREES_FILE =  os.path.join(FILE_DIR, '..', 'kdtrees.pkl')

FACE_DETECTION_MODEL = os.path.join(DEPENDENCIES_PATH, 'Pytorch_Retinaface', 'weights' , 'Resnet50_Final.pth')

FACE_DETECTION_NETWORK = 'resnet50' # options are 'mobile0.25' or 'resnet50'
