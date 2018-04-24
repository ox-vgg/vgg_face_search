__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import os, sys
import settings
# add facenet to python path
sys.path.append(os.path.join(settings.DEPENDENCIES_PATH, 'facenet', 'src'))
import facenet
import tensorflow as tf
import numpy
import align.detect_face

# Disables warnings, e.g.: "Your CPU supports instructions that this TensorFlow binary was not compiled to use".
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Some handy constants, mostly extracted from:
# https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py
MINSIZE = 20 # minimum size of face
THRESHOLD = [0.6, 0.7, 0.7]  # three steps's threshold
FACTOR = 0.709 # scale factor
FACE_RECT_EXPAND_FACTOR = 0.4 # A different value to the other detector, to make results more similar between them

class FaceDetectorFacenetMTCNN(object):
    """
        Class to support the face detection via Multi-task CNN.
        Based on the code found at https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py.
        Note that CUDA is sill not supported, so this implementation only works on the CPU.
    """

    def __init__(self, enable_cuda=False, face_rect_expand_factor=FACE_RECT_EXPAND_FACTOR):
        """
            Initializes Multi-task CNN in Tensorflow
            Arguments:
                enable_cuda: boolean indicating whether CUDA must be used for the extraction of the features (not used for now)
                face_rect_expand_factor: Expansion factor for the detection face rectangle
        """
        self.is_cuda_enable = enable_cuda # not used for now
        self.face_rect_expand_factor = face_rect_expand_factor
        with tf.Graph().as_default():
            self.sess = tf.Session(config=tf.ConfigProto())
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, None)


    def detect_faces(self, img, return_best=False):
        """
            Computes a list of faces detected in the input image in the form of a list of bounding-boxes, one per each detected face.
            Arguments:
                img: The image to be input to the Faster R-CNN model
                return_best: boolean indicating whether to return just to best detection or the complete list of detections
            Returns:
                A list of lists. Each sublist contains the image coordinates of the corners of a bounding-box and the score of the detection
                in the form [x1,y1,x2,y2,score], where (x1,y1) are the integer coordinates of the top-left corner of the box and (x2,y2) are
                the coordinates of the bottom-right corner of the box. The score is a floating-point number.
                When return_best is True, the returned list will contain only one bounding-box
        """
        if numpy.all(img != None):
            try:

                bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, self.pnet, self.rnet, self.onet, THRESHOLD, FACTOR)
                if len(bounding_boxes) > 0:
                    if return_best:
                        # dets is ordered by confidence so the first one is the best
                        det = numpy.squeeze(bounding_boxes[0, 0:5])
                        bounding_box = numpy.zeros(5, dtype=numpy.float32)
                        # extend detection
                        extend_factor = self.face_rect_expand_factor
                        width = round(det[2]-det[0])
                        height = round(det[3]-det[1])
                        length = (width + height)/2.0
                        centrepoint = [round(det[0]) + width/2.0, round(det[1]) + height/2.0]
                        bounding_box[0] = centrepoint[0] - round((1+extend_factor)*length/2.0)
                        bounding_box[1] = centrepoint[1] - round((1+extend_factor)*length/2.0)
                        bounding_box[2] = centrepoint[0] + round((1+extend_factor)*length/2.0)
                        bounding_box[3] = centrepoint[1] + round((1+extend_factor)*length/2.0)
                        ## prevent going off image
                        bounding_box[0] = int(max(bounding_box[0], 0))
                        bounding_box[1] = int(max(bounding_box[1], 0))
                        bounding_box[2] = int(min(bounding_box[2], img.shape[1]))
                        bounding_box[3] = int(min(bounding_box[3], img.shape[0]))
                        bounding_box[4] = det[4]
                        return [bounding_box]
                    else:
                        det_list = []
                        for j in range(len(bounding_boxes)):
                            det = numpy.squeeze(bounding_boxes[j, 0:5])
                            bounding_box = numpy.zeros(5, dtype=numpy.float32)
                            # extend detection
                            extend_factor = self.face_rect_expand_factor
                            width = round(det[2]-det[0])
                            height = round(det[3]-det[1])
                            length = (width + height)/2.0
                            centrepoint = [round(det[0]) + width/2.0, round(det[1]) + height/2.0]
                            bounding_box[0] = centrepoint[0] - round((1+extend_factor)*length/2.0)
                            bounding_box[1] = centrepoint[1] - round((1+extend_factor)*length/2.0)
                            bounding_box[2] = centrepoint[0] + round((1+extend_factor)*length/2.0)
                            bounding_box[3] = centrepoint[1] + round((1+extend_factor)*length/2.0)
                            ## prevent going off image
                            bounding_box[0] = int(max(bounding_box[0], 0))
                            bounding_box[1] = int(max(bounding_box[1], 0))
                            bounding_box[2] = int(min(bounding_box[2], img.shape[1]))
                            bounding_box[3] = int(min(bounding_box[3], img.shape[0]))
                            bounding_box[4] = det[4]
                            det_list.append(bounding_box)
                        return det_list
                else:
                    return None

            except Exception as e:
                print 'Exception in FaceDetectorFacenetMTCNN:', str(e)
                pass

        return None
