__author__      = 'Ernesto Coto'
__copyright__   = 'March 2018'

import os
import sys
import numpy
import pickle
import argparse
import platform
from multiprocessing import freeze_support

# add the web service folder to the sys path
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(DIR_PATH, '..', 'service'))
import settings
import imutils

if __name__ == '__main__':
    if 'Windows' in platform.system():
        freeze_support() # a requirement for windows execution

    # check arguments before continuing
    parser = argparse.ArgumentParser(description='Face-backend features extractor')
    parser.add_argument('dataset_base_path', metavar='dataset_base_path', type=str, help='Base path of image dataset')
    parser.add_argument('images_list', metavar='images_list', type=str, help='Path to file containing the list of images to extract the features from. Image paths in the list should be paths relative to dataset_base_path')
    parser.add_argument('-o', dest='output_file', default=settings.DATASET_FEATS_FILE, help='Output file (default: file specified in the settings). If the file exist the new features will be appended to it.')
    args = parser.parse_args()

    previous_database = None
    if os.path.exists(args.output_file):
        with open(args.output_file, 'rb') as fin:
            previous_database = pickle.load(fin)
        if isinstance(previous_database, list):
            print ('ERROR: This script creates a dictionary-based database and cannot be used to add information to the existing list-based database found at %s. Aborting !.' % args.output_file)
            sys.exit(1)

    # import the appropriate face detector, depending on CUDA being enabled or not
    if settings.CUDA_ENABLED:
        import face_detection_faster_rcnn
        face_detector = face_detection_faster_rcnn.FaceDetectorFasterRCNN()
    elif 'Windows' in platform.system():
        import face_detection_dlib
        face_detector = face_detection_dlib.FaceDetectorDlib()
    else:
        import face_detection_facenet
        face_detector = face_detection_facenet.FaceDetectorFacenetMTCNN()
    # import and create face feature extractor
    import face_features
    feature_extractor = face_features.FaceFeatureExtractor()

    # Compute features for all image paths in args.images_list
    all_feats = {'paths': [], 'rois': [], 'feats': []}
    with open(args.images_list) as fin:
        for img_path in fin:
            img_path = img_path.replace('\n', '')
            if len(img_path) > 0:
                full_path = os.path.join(args.dataset_base_path, img_path)
                print ('Computing features for file %s' % (full_path))

                # read image
                img = imutils.acquire_image(full_path)

                # run face detector
                detections = face_detector.detect_faces(img)

                if numpy.all(detections != None):

                    for det in detections:

                        # The coordinates should be already integers, but some basic
                        # conversion is need for compatibility with all face detectors.
                        # Plus we have to get rid of the detection score det[4]
                        det = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]

                        # crop image to detected face area.
                        crop_img = img[det[1]:det[3], det[0]:det[2], :]

                        # compute feature
                        feat = feature_extractor.feature_compute(crop_img)

                        # Append to previous results
                        all_feats['paths'].append(img_path)
                        all_feats['rois'].append(det)
                        all_feats['feats'].append(feat)

        # load previous database file, if present
        if previous_database:
            # convert back to list before appending
            previous_database['feats'] = list(previous_database['feats'])
            # append new elements to previous database
            for idx in range(len(all_feats['paths'])):
                previous_database['feats'].append(all_feats['feats'][idx])
                previous_database['paths'].append(all_feats['paths'][idx])
                previous_database['rois'].append(all_feats['rois'][idx])
            all_feats = previous_database

        # convert to format used in the backend
        all_feats['feats'] = numpy.array(all_feats['feats'])
        # save to database file
        with open(args.output_file, 'wb') as fout:
            pickle.dump(all_feats, fout, pickle.HIGHEST_PROTOCOL)
