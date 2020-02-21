__author__      = 'Ernesto Coto'
__copyright__   = 'February 2020'

import numpy
import os
import sys
import multiprocessing
import skimage.transform
import settings
import torch
import torch.backends.cudnn as cudnn
import PIL.Image

cudnn.benchmark = True
torch.set_grad_enabled(False)

# add model definition folder to the sys path
sys.path.append(os.path.dirname(settings.FEATURES_MODEL_DEF))
# then import the model
import senet50_256 as model

class FaceFeatureExtractor(object):
    """ Class to support the face-feature extraction """

    def __init__(self, model_weights=settings.FEATURES_MODEL_WEIGHTS,
                       model_def=settings.FEATURES_MODEL_DEF,
                       feature_layer=settings.FEATURES_MODEL_LAYER,
                       feature_vector_size=settings.FEATURES_VECTOR_SIZE,
                       enable_cuda=settings.CUDA_ENABLED):
        """
            Initializes the face-feature extraction CNN model
            Arguments:
                model_weights: Full path to the file containing the weights of the model
                model_def: Full path to the model
                feature_layer: name of the layer from where to extract the features
                feature_vector_size: the length of the feature vector output by the CNN
                enable_cuda: boolean indicating whether CUDA must be used for the extraction of the features
        """
        self.is_cuda_enable = enable_cuda
        self.model_weights = model_weights
        self.model_def = model_def
        self.feature_layer = feature_layer
        self.feature_vector_size = feature_vector_size
        self.net_lock = multiprocessing.Lock()

        # Load model here
        self.network = model.Senet50_256()
        self.device = torch.device('cpu' if not self.is_cuda_enable else 'cuda')
        if not self.is_cuda_enable:
            pretrained_dict = torch.load(self.model_weights, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(self.model_weights, map_location=lambda storage, loc: storage.cuda(self.device))
        self.network.load_state_dict(pretrained_dict)
        self.network.eval()
        self.network = self.network.to(self.device)


    def feature_compute(self, image):
        """
            Inputs an image to the CNN and computes a vector of face-features
            The vector is extracted from the layer specified when the object was instantiated
            This method will try to use CUDA if enabled when the object was instantiated
            Arguments:
                image: input image
            Returns:
                A 1D normalized vector with the length specified when the object was instantiated
                Returns None in case of error
        """
        if numpy.all(image != None):

            try:

                # the input to the network is 224x224, so we need to resize the image.
                # Unfortunately, the resizing has to be done with Pillow to follow
                # a similar procedure to
                # https://github.com/ox-vgg/vgg_face2/blob/master/standard_evaluation/pytorch_feature_extractor.py
                # or the results are not the same because the resizing results with
                # skimage are different

                pil_img = PIL.Image.fromarray(image)
                pil_img = pil_img.resize(size=(244, 244), resample=PIL.Image.BILINEAR)

                # now we can convert back to numpy to continue
                img_prepared = numpy.array(pil_img)
                img_prepared = img_prepared - self.network.meta['mean']
                im_array = numpy.array([img_prepared])
                img_torch = torch.Tensor(im_array.transpose(0, 3, 1, 2))
                img_torch = img_torch.to(self.device)
                
                # lock acquire
                self.net_lock.acquire()

                # evaluate input
                feat = self.network(img_torch)[1].detach().cpu().numpy()[: , :, 0, 0]

                # make sure the output is a simple 1D vector
                feat = numpy.reshape(feat, self.feature_vector_size)

                # lock release
                self.net_lock.release()

                # normalize
                feat = feat / numpy.sqrt(numpy.sum(feat ** 2, -1, keepdims=True))

                return feat

            except Exception as e:
                print ('Exception in FaceFeatureExtractor: ' + str(e))
                pass

        return None
