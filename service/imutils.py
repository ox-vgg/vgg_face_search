__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import skimage
from skimage import io
from skimage import color

def acquire_image(img_path):
    """
        Utility function to read an image and covert it to RGB if needed.
        Arguments:
           img_path: Full path to the image file to be read
        Returns:
           An MxNx3 array corresponding to the contents of an MxN image in RGB format.
           Returns None in case of errors
    """
    try:
        if img_path:

            # read image
            img = skimage.io.imread(img_path)
            # if RGBA, drop Alpha channel
            if len(img.shape) > 2 and img.shape[2] == 4:
                img = img[:, :, :3].copy()
            # if only one channel, convert to RGB
            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)

            return img
    except Exception as e:
        print e
        pass

    return None


def save_image(img, img_path):
    """
        Utility function to save an image to a local path.
        Arguments:
           img: ndarray containing the pixel values of the image
           img_path: Full path to the image file to be created
        Returns:
           An MxNx3 array corresponding to the contents of an MxN image in RGB format.
    """
    try:
        skimage.io.imsave(img_path, img)
    except Exception as e:
        print e
        pass
