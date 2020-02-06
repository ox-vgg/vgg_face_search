from __future__ import division

__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import os, sys
import settings
# suppress Caffe verbose prints. Caffe is first imported by face-py-faster-rcn.
os.environ['GLOG_minloglevel'] = '2'
# add face-py-faster-rcn dependencies to python path
sys.path.append(os.path.join(settings.DEPENDENCIES_PATH, 'face-py-faster-rcnn', 'caffe-fast-rcnn', 'python'))
sys.path.append(os.path.join(settings.DEPENDENCIES_PATH, 'face-py-faster-rcnn', 'lib'))
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy
import caffe

# Some handy constants, mostly extracted from:
# https://github.com/playerkk/face-py-faster-rcnn/blob/master/tools/run_face_detection_on_fddb.py
CONF_THRESH = 0.65
NMS_THRESH = 0.15
cfg.TEST.HAS_RPN = True  # Use RPN for proposals
cfg.GPU_ID = 0 # Change the default GPU here, if needed
FACE_RECT_EXPAND_FACTOR = 0.3

class FaceDetectorFasterRCNN(object):
    """
        Class to support the face detection via Faster R-CNN.
        Based on the code found at https://github.com/playerkk/face-py-faster-rcnn/blob/master/tools/run_face_detection_on_fddb.py.
        Note that when CUDA is not used this detector can be very slow.
    """

    def __init__(self, prototxt=os.path.join(settings.DEPENDENCIES_PATH, 'face-py-faster-rcnn', 'models', 'face', 'VGG16', 'faster_rcnn_end2end', 'test.prototxt'),
                       caffemodel=settings.GPU_FACE_DETECTION_CAFFE_MODEL,
                       face_rect_expand_factor=FACE_RECT_EXPAND_FACTOR,
                       enable_cuda=settings.CUDA_ENABLED):
        """
            Initializes the Faster R-CNN model in Caffe
            Arguments:
                prototxt: Full path to the Caffe prototxt file corresponding to the Faster R-CNN model
                caffemodel: Full path to the Faster R-CNN model
                face_rect_expand_factor: Expansion factor for the detection face rectangle
                enable_cuda: boolean indicating whether CUDA must be used for the extraction of the features
        """
        self.is_cuda_enable = enable_cuda
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.face_rect_expand_factor = face_rect_expand_factor
        self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

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
                if not self.is_cuda_enable:
                    caffe.set_mode_cpu()
                else:
                    caffe.set_mode_gpu()
                    caffe.set_device(cfg.GPU_ID)

                scores, boxes = im_detect(self.net, img)

                cls_ind = 1
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = numpy.hstack((cls_boxes,
                        cls_scores[:, numpy.newaxis])).astype(numpy.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]

                keep = numpy.where(dets[:, 4] > CONF_THRESH)
                dets = dets[keep]

                if len(dets) > 0:
                    if return_best:
                        # dets is ordered by confidence dets[:, 4], so the first one is the best
                        det = [int(dets[0, 0]), int(dets[0, 1]), int(dets[0, 2]), int(dets[0, 3]), dets[0, 4]]
                        # extend detection
                        extend_factor = self.face_rect_expand_factor
                        width = round(det[2]-det[0])
                        height = round(det[3]-det[1])
                        length = (width + height)/2.0
                        centrepoint = [round(det[0]) + width/2.0, round(det[1]) + height/2.0]
                        det[0] = centrepoint[0] - round((1+extend_factor)*length/2.0)
                        det[1] = centrepoint[1] - round((1+extend_factor)*length/2.0)
                        det[2] = centrepoint[0] + round((1+extend_factor)*length/2.0)
                        det[3] = centrepoint[1] + round((1+extend_factor)*length/2.0)
                        ## prevent going off image
                        det[0] = int(max(det[0], 0))
                        det[1] = int(max(det[1], 0))
                        det[2] = int(min(det[2], img.shape[1]))
                        det[3] = int(min(det[3], img.shape[0]))
                        return [det]
                    else:
                        det_list = []
                        for j in range(dets.shape[0]):
                            det = [int(dets[j, 0]), int(dets[j, 1]), int(dets[j, 2]), int(dets[j, 3]), dets[0, 4]]
                            # extend detection
                            extend_factor = self.face_rect_expand_factor
                            width = round(det[2]-det[0])
                            height = round(det[3]-det[1])
                            length = (width + height)/2.0
                            centrepoint = [round(det[0]) + width/2.0, round(det[1]) + height/2.0]
                            det[0] = centrepoint[0] - round((1+extend_factor)*length/2.0)
                            det[1] = centrepoint[1] - round((1+extend_factor)*length/2.0)
                            det[2] = centrepoint[0] + round((1+extend_factor)*length/2.0)
                            det[3] = centrepoint[1] + round((1+extend_factor)*length/2.0)
                            ## prevent going off image
                            det[0] = int(max(det[0], 0))
                            det[1] = int(max(det[1], 0))
                            det[2] = int(min(det[2], img.shape[1]))
                            det[3] = int(min(det[3], img.shape[0]))
                            det_list.append(det)
                        return det_list
                else:
                    return None

            except Exception as e:
                print ('Exception in FaceDetectorFasterRCNN: ' + str(e))
                pass

        return None
