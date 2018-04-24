__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import numpy
import dlib
import scipy
from scipy.ndimage.interpolation import rotate
import multiprocessing

# IMAGE PROCESSING constants
NO_FACE_DETECTED_SCORE = -999999
FACE_RECT_EXPAND_FACTOR = 0.3
NUMBER_OF_HELPER_WORKERS = 8

def rotate_and_detect(image_and_angle):
    """
        Method to rotate an input image and then perform face detection on it.
        Made to be invoked in parallel for rotation over several angles.
        Parameters:
            image_and_angle: Tuple containing the image object and the angle
            by which it must be rotated
        Returns:
            Tuple containing the detection rectangle, the score of the detection,
            the rotated image, and the angle by which it was rotated
    """
    try:
        img = image_and_angle[0]
        angle = image_and_angle[1]
        if angle > 0:
            img_rotated = rotate(img, angle)
        else:
            img_rotated = img
        detector = dlib.get_frontal_face_detector()
        dets, scores, idx = detector.run(img_rotated, 1, -1)  # set score threshod to -1 to
                                                              # increase number of detections
        try:
            # This is were return_best should be used
            scores = scores[0]
        except:
            scores = NO_FACE_DETECTED_SCORE
        return (dets, scores, img_rotated, angle)
    except Exception as e:
        #print str(e)
        pass
    return (None, 0, None)


class FaceDetectorDlib(object):
    """
        Class to support the face detection via dlib.
        It runs dlib's HOG-based frontal face detector over four different
        orientations of an image and returns the best detection.
    """

    def __init__(self, face_rect_expand_factor=FACE_RECT_EXPAND_FACTOR):
        """
            Initializes the detector.
            Arguments:
                face_rect_expand_factor: Expansion factor for the detection face rectangle
        """
        self.face_rect_expand_factor = face_rect_expand_factor
        self.worker_pool = multiprocessing.Pool(processes=NUMBER_OF_HELPER_WORKERS)


    def detect_faces(self, img, return_best=True):
        """
            Computes a list of faces detected in the input image in the form of a list of bounding-boxes, one per each detected face.
            Currently only ONE face is detected and therefore the return_best parameter is not used.
            Arguments:
                img: The image to be input to the Faster R-CNN model
                return_best: Boolean indicating whether to return just to best detection or the complete list of detections
            Returns:
                A list containing only ONE sublist, which contains the image coordinates of the corners of a bounding-box and the score
                of the detection in the form [x1,y1,x2,y2,score], where (x1,y1) are the integer coordinates of the top-left corner of the
                box and (x2,y2) are the coordinates of the bottom-right corner of the box. The score is a floating-point number.
        """
        # return_best not used for the moment
        if numpy.all(img != None):

            try:

                # remember img.shape[1] --> image width
                # remember img.shape[0] --> image height
                # Processing original image plus 3 rotated images.
                results = self.worker_pool.map(rotate_and_detect,
                                                [(img, 0), (img, 90), (img, 180), (img, 270)]
                                              )

                # Select the best orientation
                thedet = results[0][0]
                thescore = results[0][1]
                theim = results[0][2]
                theangle = results[0][3]
                for i in range(1, len(results)):
                    scores = results[i][1]
                    if scores > thescore:
                        thedet = results[i][0]
                        thescore = results[i][1]
                        theim = results[i][2]
                        theangle = results[i][3]

                if thescore == NO_FACE_DETECTED_SCORE:
                    return None

                # Once selected the best orientation, extract detection rectangle (det)
                mydets = [(d.left(), d.top(), d.right(), d.bottom()) for i, d  in enumerate(thedet)]
                det = mydets[numpy.argmax(scores)]
                det = list(det)
                det = [float(i) for i in det]

                # extend detection square
                # x1==det[0], y1==det[1], x2==det[2], y2==det[3]
                extend_factor = self.face_rect_expand_factor
                width = round(det[2]-det[0])
                height = round(det[3]-det[1])
                length = (width + height)/2.0
                centrepoint = [round(det[0]) + width/2.0, round(det[1]) + height/2.0]
                det[0] = centrepoint[0] - round((1+extend_factor)*length/2.0)
                det[1] = centrepoint[1] - round((1+extend_factor)*length/2.0)
                det[2] = centrepoint[0] + round((1+extend_factor)*length/2.0)
                det[3] = centrepoint[1] + round((1+extend_factor)*length/2.0)

                # prevent going off image
                det[0] = int(max(det[0], 0))
                det[1] = int(max(det[1], 0))
                det[2] = int(min(det[2], theim.shape[1]))
                det[3] = int(min(det[3], theim.shape[0]))

                # Revert initial rotation
                if theangle == 90:
                    det = [theim.shape[0]-det[3], det[0], theim.shape[0]-det[1], det[2]]
                if theangle == 180:
                    det = [theim.shape[1]-det[2], theim.shape[0]-det[3], theim.shape[1]-det[0], theim.shape[0]-det[1]]
                if theangle == 270:
                    det = [det[1], theim.shape[1]-det[2], det[3], theim.shape[1]-det[0]]

                # append detection score
                det.append(thescore)

                # return a list of detections
                return [det]

            except Exception as e:
                print 'Exception in FaceDetectorDlib:', str(e)
                pass

        return None
