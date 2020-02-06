__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import numpy
import os
import sys
import multiprocessing
import skimage
import settings

# add pycaffe to the sys path, reuse the Caffe within face-py-faster-rcnn if possible
caffe_fast_rcnn_path = os.path.join(settings.DEPENDENCIES_PATH, 'face-py-faster-rcnn', 'caffe-fast-rcnn', 'python')
if caffe_fast_rcnn_path not in sys.path:
    if os.path.exists(caffe_fast_rcnn_path):
        sys.path.append(caffe_fast_rcnn_path)
    else:
        standard_caffe_path = os.path.join(settings.DEPENDENCIES_PATH, 'caffe', 'python')
        sys.path.append(standard_caffe_path)
# suppress Caffe verbose prints
os.environ['GLOG_minloglevel'] = '2'
# finally import Caffe
import caffe

class FaceFeatureExtractor(object):
    """ Class to support the face-feature extraction """

    def __init__(self, caffe_prototxt=settings.FEATURES_CAFFE_PROTOTXT,
                       caffe_model=settings.FEATURES_CAFFE_MODEL,
                       feature_layer=settings.FEATURES_CAFFE_LAYER,
                       feature_vector_size=settings.FEATURES_VECTOR_SIZE,
                       enable_cuda=settings.CUDA_ENABLED):
        """
            Initializes the face-feature extraction CNN model in Caffe
            Arguments:
                caffe_prototxt: Full path to the Caffe prototxt file corresponding to the model
                caffe_model: Full path to Caffe CNN model
                feature_layer: name of the layer from where to extract the features
                feature_vector_size: the length of the feature vector output by the CNN
                enable_cuda: boolean indicating whether CUDA must be used for the extraction of the features
        """
        self.is_cuda_enable = enable_cuda
        self.caffe_prototxt = caffe_prototxt
        self.caffe_model = caffe_model
        self.feature_layer = feature_layer
        self.feature_vector_size = feature_vector_size
        self.net_lock = multiprocessing.Lock()

        # Load caffe model here. Take the help from
        # http://caffe.berkeleyvision.org/tutorial/interfaces.html

        self.net = caffe.Net(self.caffe_prototxt, self.caffe_model, caffe.TEST)
        self.net.blobs['data'].reshape(1, 3, 224, 224)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))

        # Change this code to set mean array from the mean RGB values. Note that mean
        # subtraction is done after the channel swap.

        self.transformer.set_mean('data', numpy.array([91.4953, 103.8827, 131.0912]) ) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB


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

                # Setting the Caffe mode has to be done PER THREAD, so this needs
                # to be executed every time this method is called
                if self.is_cuda_enable:
                    caffe.set_mode_gpu()
                    #caffe.set_device(0) # Change the default GPU here, if needed
                else:
                    caffe.set_mode_cpu()

                # the input to the network is 224x224. If not done here, transformer.preprocess() will do it anyway
                img_scaled = skimage.transform.resize(image, (224, 224), mode='constant')

                # lock acquire
                self.net_lock.acquire()

                # use transformer to prepare input data
                self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img_scaled)
                # evaluate input
                out = self.net.forward()
                # extract features from configured model layer
                feat = self.net.blobs[self.feature_layer].data[0];
                # make sure the output is a simple 1D vector
                feat = numpy.reshape(feat, self.feature_vector_size)

                # lock release
                self.net_lock.release()

                # normalize
                norm = numpy.linalg.norm(feat)
                feat = feat/max(norm, 0.00001)

                return feat

            except Exception as e:
                print ('Exception in FaceFeatureExtractor: ' + str(e))
                pass

        return None
