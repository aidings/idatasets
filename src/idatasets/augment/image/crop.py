import numpy as np
import PIL
import numbers
import random

class Crop:
    def __init__(self, tlrb) -> None:
        self.box = tlrb

    def __call__(self, image):
        x1, y1, x2, y2 = [int(c) for c in self.box]
        if isinstance(image, np.ndarray):
            im_h, im_w, im_c = image.shape
        elif isinstance(image, PIL.Image.Image):
            im_w, im_h = image.size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(image)))

        assert 0<=x1<x2<=im_w and 0<=y1<y2<=im_h, "%d %d %d %d %d %d" % (x1, y1, x2, y2, im_w, im_h)

        if isinstance(image, np.ndarray):
            return image[y1:y2, x1:x2, :]
        elif isinstance(image, PIL.Image.Image):
            return image.crop((x1, y1, x2, y2))

class CenterCrop(object):
    """
    Extract center crop of thevideo.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, image):
        crop_h, crop_w = self.size
        if isinstance(image, np.ndarray):
            im_h, im_w, im_c = image.shape
        elif isinstance(image, PIL.Image.Image):
            im_w, im_h = image.size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(image)))

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)

        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))

        if isinstance(image, np.ndarray):
            return image[h1:h1 + crop_h, w1:w1 + crop_w, :], {'CenterCrop': (h1, w1, h1+crop_h, w1+crop_w)}
        elif isinstance(image, PIL.Image.Image):
            return image.crop((w1, h1, w1 + crop_w, h1 + crop_h)), {'CenterCrop': (h1, w1, h1+crop_h, w1+crop_w)}


class CornerCrop(object):
    """
    Extract corner crop of the video.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).

        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size, crop_position=None):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

        if crop_position is None:
            self.randomize = True
        else:
            if crop_position not in ['c', 'tl', 'tr', 'bl', 'br']:
                raise ValueError("crop_position should be one of " +
                                 "['c', 'tl', 'tr', 'bl', 'br']")
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, image):
        crop_h, crop_w = self.size
        if isinstance(image, np.ndarray):
            im_h, im_w, im_c = image.shape
        elif isinstance(image, PIL.Image.Image):
            im_w, im_h = image.size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(image)))

        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(0,len(self.crop_positions) - 1)]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((im_w - crop_w) / 2.))
            y1 = int(round((im_h - crop_h) / 2.))
            x2 = x1 + crop_w
            y2 = y1 + crop_h
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_w
            y2 = crop_h
        elif self.crop_position == 'tr':
            x1 = im_w - crop_w
            y1 = 0
            x2 = im_w
            y2 = crop_h
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = im_h - crop_h
            x2 = crop_w
            y2 = im_h
        elif self.crop_position == 'br':
            x1 = im_w - crop_w
            y1 = im_h - crop_h
            x2 = im_w
            y2 = im_h

        if isinstance(image, np.ndarray):
            return image[y1:y2, x1:x2, :], {'CornerCrop': (x1,y1, x2, y2)}
        elif isinstance(image, PIL.Image.Image):
            return image.crop((x1, y1, x2, y2)), {'CornerCrop': (x1, y1, x2, y1)}


class RandomCrop(object):
    """
    Extract random crop of the video.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).

        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, image):
        crop_h, crop_w = self.size
        if isinstance(image, np.ndarray):
            im_h, im_w, im_c = image.shape
        elif isinstance(image, PIL.Image.Image):
            im_w, im_h = image.size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(image)))

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)

        w1 = random.randint(0, im_w - crop_w)
        h1 = random.randint(0, im_h - crop_h)

        if isinstance(image, np.ndarray):
            return image[h1:h1 + crop_h, w1:w1 + crop_w, :], {'RandomCrop': (h1, w1, h1+crop_h, w1+crop_w)}
        elif isinstance(image, PIL.Image.Image):
            return image.crop((w1, h1, w1 + crop_w, h1 + crop_h)), {'RandomCrop': (h1, w1, h1+crop_h, w1+crop_w)}