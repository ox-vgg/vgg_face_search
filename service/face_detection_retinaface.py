__author__      = 'Ernesto Coto'
__copyright__   = 'February 2020'

import os, sys
import settings
import torch
import torch.backends.cudnn as cudnn
import numpy
# add RetinaFace to python path
sys.path.append(os.path.join(settings.DEPENDENCIES_PATH, 'Pytorch_Retinaface'))
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

# Some handy constants, mostly extracted from
# https://github.com/biubug6/Pytorch_Retinaface/blob/master/test_fddb.py
CONF_THRESH = 0.65
NMS_THRESH = 0.4
FACE_RECT_EXPAND_FACTOR = 0.3

class FaceDetectorRetinaFace(object):
    """
        Class to support the face detection via RetinaFace
        Based on the code found at https://github.com/biubug6/Pytorch_Retinaface/blob/master/test_fddb.py
    """

    def __init__(self, enable_cuda=settings.CUDA_ENABLED,
                       face_rect_expand_factor=FACE_RECT_EXPAND_FACTOR,
                       trained_model=settings.FACE_DETECTION_MODEL,
                       network=settings.FACE_DETECTION_NETWORK,
                       ):
        """
            Initializes the RetinaFace in PyTorch
            Arguments:
                enable_cuda: boolean indicating whether CUDA must be used for the extraction of the features
                face_rect_expand_factor: Expansion factor for the detection face rectangle
                trained_model: Path to a pretrained model file with weights
                network: Name of the network used for the detection. The options are 'mobile0.25' or 'resnet50'.
        """
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        self.is_cuda_enable = enable_cuda
        self.face_rect_expand_factor = face_rect_expand_factor
        self.trained_model = trained_model
        self.cfg = None
        if network == 'mobile0.25':
            self.cfg = cfg_mnet
        elif network == 'resnet50':
            self.cfg = cfg_re50
        assert self.cfg != None, "Network name can only be 'resnet50' or 'mobile0.25' !"
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = self.load_model(self.net, self.trained_model, not self.is_cuda_enable)
        self.net.eval()
        self.device = torch.device('cpu' if not self.is_cuda_enable else 'cuda')
        self.net = self.net.to(self.device)


    def check_keys(self, model, pretrained_state_dict):
        """
            Checks missing dictionary keys in the pretrained model.

            Extracted 'as is' from https://github.com/biubug6/Pytorch_Retinaface/blob/master/test_fddb.py
        """
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        """
            Old style model is stored with all names of parameters sharing common prefix 'module.'

            Extracted 'as is' from https://github.com/biubug6/Pytorch_Retinaface/blob/master/test_fddb.py
        """
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        """
            Loads the specified trained model
            Arguments:
                load_to_cpu: boolean indicating whether to load the model on the CPU or the GPU
                model: RetinaFace model object
                pretrained_path: Path to a pretrained model file with weights

            Extracted 'as is' from https://github.com/biubug6/Pytorch_Retinaface/blob/master/test_fddb.py
        """
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model


    def detect_faces(self, img, return_best=False):
        """
            Computes a list of faces detected in the input image in the form of a list of bounding-boxes, one per each detected face.
            Arguments:
                img: The image to be input to the RetinaFace model
                return_best: boolean indicating whether to return just to best detection or the complete list of detections
            Returns:
                A list of arrays. Each array contains the image coordinates of the corners of a bounding-box and the score of the detection
                in the form [x1,y1,x2,y2,score], where (x1,y1) are the integer coordinates of the top-left corner of the box and (x2,y2) are
                the coordinates of the bottom-right corner of the box. The score is a floating-point number.
                When return_best is True, the returned list will contain only one bounding-box
        """
        if numpy.all(img != None):
            try:
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img = numpy.float32(img)
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(self.device)
                scale = scale.to(self.device)

                # note below that the landmarks (3rd returned value) are ignored
                loc, conf, _ = self.net(img)

                priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(self.device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
                boxes = boxes * scale
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

                # ignore low scores
                inds = numpy.where(scores > CONF_THRESH)[0]
                boxes = boxes[inds]
                scores = scores[inds]

                # keep top-K before NMS
                # order = scores.argsort()[::-1][:args.top_k]
                order = scores.argsort()[::-1]
                boxes = boxes[order]
                scores = scores[order]

                # do NMS
                dets = numpy.hstack((boxes, scores[:, numpy.newaxis])).astype(numpy.float32, copy=False)
                keep = py_cpu_nms(dets, NMS_THRESH)

                # keep top-K faster NMS
                detections = dets[keep, :]

                if len(detections) > 0:
                    if return_best:
                        # detections is ordered by confidence so the first one is the best
                        det = numpy.squeeze(detections[0, 0:5])
                        bounding_box = numpy.zeros(5, dtype=numpy.float32)
                        # extend detection
                        extend_factor = self.face_rect_expand_factor
                        width = round(det[2]-det[0]+1)
                        height = round(det[3]-det[1]+1)
                        length = (width + height)/2.0
                        centrepoint = [round(det[0]) + width/2.0, round(det[1]) + height/2.0]
                        bounding_box[0] = centrepoint[0] - round((1+extend_factor)*length/2.0)
                        bounding_box[1] = centrepoint[1] - round((1+extend_factor)*length/2.0)
                        bounding_box[2] = centrepoint[0] + round((1+extend_factor)*length/2.0)
                        bounding_box[3] = centrepoint[1] + round((1+extend_factor)*length/2.0)
                        # prevent going off image
                        bounding_box[0] = int(max(bounding_box[0], 0))
                        bounding_box[1] = int(max(bounding_box[1], 0))
                        bounding_box[2] = int(min(bounding_box[2], img.shape[3]))
                        bounding_box[3] = int(min(bounding_box[3], img.shape[2]))
                        bounding_box[4] = det[4]
                        return [bounding_box]
                    else:
                        det_list = []
                        for j in range(len(detections)):
                            det = numpy.squeeze(detections[j, 0:5])
                            bounding_box = numpy.zeros(5, dtype=numpy.float32)
                            # extend detection
                            extend_factor = self.face_rect_expand_factor
                            width = round(det[2]-det[0]+1)
                            height = round(det[3]-det[1]+1)
                            length = (width + height)/2.0
                            centrepoint = [round(det[0]) + width/2.0, round(det[1]) + height/2.0]
                            bounding_box[0] = centrepoint[0] - round((1+extend_factor)*length/2.0)
                            bounding_box[1] = centrepoint[1] - round((1+extend_factor)*length/2.0)
                            bounding_box[2] = centrepoint[0] + round((1+extend_factor)*length/2.0)
                            bounding_box[3] = centrepoint[1] + round((1+extend_factor)*length/2.0)
                            # prevent going off image
                            bounding_box[0] = int(max(bounding_box[0], 0))
                            bounding_box[1] = int(max(bounding_box[1], 0))
                            bounding_box[2] = int(min(bounding_box[2], img.shape[3]))
                            bounding_box[3] = int(min(bounding_box[3], img.shape[2]))
                            bounding_box[4] = det[4]
                            det_list.append(bounding_box)
                        return det_list
                else:
                    return None

            except Exception as e:
                print ('Exception in FaceDetectorRetinaFace: ' + str(e))
                pass

        return None
