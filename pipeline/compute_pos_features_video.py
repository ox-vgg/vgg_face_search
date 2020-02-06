__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import os
import sys
import numpy
import pickle
import argparse
import platform
from multiprocessing import freeze_support
import shutil
import string
import re

MIN_IOU = 0.5

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Taken from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Arguments:
        bb1 : list[x1,y1,x2,y2]
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
        Keys: list[x1,y1,x2,y2]
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
    Returns:
        A float number in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

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
    parser.add_argument('video_frames_path', metavar='video_frames_path', type=str, help='Base path of video frames')
    parser.add_argument('shot_boundaries', metavar='shot_boundaries', type=str, help='Path to file containing the list of shot boundaries for the video')
    parser.add_argument('dataset_base_path', metavar='dataset_base_path', type=str, help='Base path of image dataset')
    parser.add_argument('-o', dest='output_file', default=settings.DATASET_FEATS_FILE, help='Output file (default: file specified in the settings). If the file exist the new features will be appended to it.')
    args = parser.parse_args()

    if not os.path.exists(args.video_frames_path) or not os.path.exists(args.shot_boundaries):
        print ('ERROR: Either the video frames or the shot boundaries are not found. Aborting !.')
        sys.exit(1)

    # acquire list of images
    video_frames_list = os.listdir(args.video_frames_path)
    video_frames_list.sort()
    if len(video_frames_list) == 0:
        print ('ERROR: There are no frames in the video frames path. Aborting !.')
        sys.exit(1)

    # load previous database, if present
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

    # acquire shots list
    shots_list = []
    with open(args.shot_boundaries) as fshots:
        for line in fshots:
            if len(line) > 0:
                line = line.replace('\n', '')
                ashot = line.split(' ')
                shots_list.append(ashot)

    # create final sub-folder in the dataset folder
    destination_frames_path = args.video_frames_path
    if destination_frames_path.endswith(os.path.sep):
        destination_frames_path = destination_frames_path[:-1]

    pattern = re.compile('[^a-zA-Z0-9_]')
    string_accepted = pattern.sub('', string.printable)
    destination_frames_path = destination_frames_path.split(os.path.sep)[-1]
    destination_frames_path = ''.join(filter(lambda afunc: afunc in string_accepted, destination_frames_path))
    if not os.path.exists(os.path.join(args.dataset_base_path, destination_frames_path)):
        os.makedirs(os.path.join(args.dataset_base_path, destination_frames_path))

    # go through list of shots computing tracks and features
    all_feats = {'paths': [], 'rois': [], 'feats': []}
    for shot in shots_list:

        # all files should be in jpeg format (and with extension .jpg), because we must have split
        # the video before, using exactly this format and extension
        shot_begin = shot[0] + '.jpg'
        shot_end = shot[1] + '.jpg'
        shot_begin_index = video_frames_list.index(shot_begin)
        shot_end_index = video_frames_list.index(shot_end)
        shot_detections = []
        shot_tracks = []
        shot_images = []

        #####
        # Compute face detections in shot
        #####
        for index in range(shot_begin_index, shot_end_index+1):
            img_name = video_frames_list[index]
            full_path = os.path.join(args.video_frames_path, img_name)

            # read image
            img = imutils.acquire_image(full_path)
            shot_images.append(img)

            # run face detector
            detections = face_detector.detect_faces(img)
            shot_detections.append(detections)
            if numpy.all(detections != None):
                shot_tracks.append([-1] * len(detections)) # init all tracks number with -1 ...
            else:
                shot_tracks.append(None) # ... or None if there are no detections

        #####
        # Compute face tracks in shot
        #####

        # The code below uses two pointers to the array of images: index A and index B.
        # Index A points to the current image
        # Index B is used for comparing the faces in A with the faces in the rest of the images in the shot

        face_track_counter = 0
        map_track_images_det = {}
        # iterate through images with pointer A
        for index_image_A in range(len(shot_detections)):
            if shot_detections[index_image_A]: # check for a non-empty list
                image_A_detections = shot_detections[index_image_A]
                image_A_tracks = shot_tracks[index_image_A]

                # iterate through faces in image pointed by A
                for index_faces_A in range(len(image_A_detections)):
                    image_A_face_det = image_A_detections[index_faces_A]
                    if image_A_tracks[index_faces_A] < 0: # only take into account faces with no-assigned tracks
                        image_A_tracks[index_faces_A] = face_track_counter
                        # save track info
                        if face_track_counter not in map_track_images_det.keys():
                            map_track_images_det[face_track_counter] = []
                        map_track_images_det[face_track_counter].append([index_image_A, image_A_face_det])
                        index_image_B = index_image_A + 1 # start index B in the next image

                         # iterate through images with pointer B (B>A)
                        while index_image_B < len(shot_detections):
                            if shot_detections[index_image_B]: # check for a non-empty list
                                image_B_detections = shot_detections[index_image_B]
                                image_B_tracks = shot_tracks[index_image_B]
                                found_match = False

                                # iterate through faces in image pointed by B
                                for index_faces_B in range(len(image_B_detections)):
                                    if image_B_tracks[index_faces_B] < 0: # only take into account faces with no-assigned tracks
                                        image_B_face_det = image_B_detections[index_faces_B]
                                        #print ("compare %s against %s" % (str(image_A_face_det), str(image_B_face_det)))
                                        the_iou = get_iou(image_A_face_det, image_B_face_det)
                                        if the_iou > MIN_IOU:
                                            # found match, moving to next image
                                            found_match = True
                                            # save track info
                                            image_B_tracks[index_faces_B] = face_track_counter  # save track number
                                            map_track_images_det[face_track_counter].append([index_image_B, image_B_face_det])
                                            image_A_face_det = image_B_face_det # update tracking face
                                            break
                                # end for

                                if found_match:
                                    # move to next image
                                    index_image_B = index_image_B + 1
                                else:
                                    # stop tracking and move to next image with index A, increase track counter
                                    face_track_counter = face_track_counter + 1
                                    break
                            else:
                                # an empty list means a break in the track
                                # stop tracking and move to next image with index A, increase track counter
                                face_track_counter = face_track_counter + 1
                                break
                        # end while

                        # if we reached the end, increase track counter
                        if index_image_B == len(shot_detections):
                            face_track_counter = face_track_counter + 1

        #####
        # Compute average feature per track, per image.
        # Also copy those frames representing the track to their final sub-folder in the dataset folder
        #####

        for track in map_track_images_det:
            det_pair_list = map_track_images_det[track]
            feats_accumulator = numpy.zeros((1, settings.FEATURES_VECTOR_SIZE))
            best_score = -100000
            chosen_image_path = None
            chosen_det = None
            for det_pair in det_pair_list:
                img_index = det_pair[0]
                img_path = video_frames_list[img_index + shot_begin_index]
                img = shot_images[img_index]
                det = det_pair[1]
                score = det[4]

                # The coordinates should be already integers, but some basic
                # conversion is need for compatibility with all face detectors.
                # Plus we have to get rid of the detection score det[4]
                det = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]

                # crop image to detected face area.
                crop_img = img[det[1]:det[3], det[0]:det[2], :]

                # compute feature
                feat = feature_extractor.feature_compute(crop_img)
                feats_accumulator = feats_accumulator + feat

                if score > best_score:
                    best_score = score
                    chosen_image_path = img_path
                    chosen_det = det

            # average and normalize
            feats_average = feats_accumulator / len(det_pair_list)
            feats_average_norm = numpy.linalg.norm(feats_average)
            feats_average_norm = feats_average/max(feats_average_norm, 0.00001)

            # make sure we save a simple 1D vector
            feats_average_1D = numpy.reshape(feats_average_norm, settings.FEATURES_VECTOR_SIZE)

            # append to previous results
            all_feats['paths'].append(destination_frames_path + os.path.sep + chosen_image_path)
            all_feats['rois'].append(chosen_det)
            all_feats['feats'].append(feats_average_1D)

            # copy chosen frame to final destination in dataset folder
            chose_image_path_in_datasets = os.path.join(args.dataset_base_path, destination_frames_path, chosen_image_path)
            if not os.path.exists(chose_image_path_in_datasets):
                shutil.copyfile(os.path.join(args.video_frames_path, chosen_image_path), chose_image_path_in_datasets)
                # print final frame path within the dataset folder, for other process to pick up
                print (destination_frames_path + os.path.sep + chosen_image_path)

    # after processing all shots, save the results ...

    # if there is a previous database file ...
    if previous_database:
        # ... convert back to list before appending
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
