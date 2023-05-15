import numpy as np
import PIL
import random


class HorizontalFlip(object):
    """
    Random Horizontally flip the video.
    """

    def __init__(self, use=True) -> None:
        self.use = use

    def __call__(self, image):
        if not self.use:
            return image

        if isinstance(image, np.ndarray):
            return np.fliplr(image)
        elif isinstance(image, PIL.Image.Image):
            return image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(image)))


class RandomHorizontalFlip(object):
    """
    Random Horizontally flip the video.
    """

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image, {'RandomHorizontalFlip': False}

        if isinstance(image, np.ndarray):
            return np.fliplr(image), {'RandomHorizontalFlip': True}
        elif isinstance(image, PIL.Image.Image):
            return image.transpose(PIL.Image.FLIP_LEFT_RIGHT), {'RandomHorizontalFlip': True}
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(image)))