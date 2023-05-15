import PIL
import numpy as np
from torchvision import transforms

__all__ = ['Pad']

class Pad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        
        self.fill = fill
        self.padding_mode = padding_mode 
    
    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            h, w, _ = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            w, h = clip[0].size
        else:
            raise ValueError("make sure image type is np.ndarray or PIL.Image")
        
        pt, pb, pr, pl = self.__shape2board(w, h)

        if isinstance(clip, np.ndarray):
            clip = np.pad(clip, ((0, 0), (pt, pb), (pr, pl), (0, 0)), mode=self.padding_mode, constant_values=self.constant_values)
        elif isinstance(clip[0], np.ndarray):
            clip = [np.pad(clip, ((pt, pb), (pr, pl), (0, 0)), mode=self.padding_mode, constant_values=self.constant_values) for i in range(len(clip))]
        else:
            pad = transforms.Pad((pt, pl, pb, pr), fill=self.fill, padding_mode=self.padding_mode)
            clip = [pad(image) for image in clip] 
        
        return clip

    def __shape2board(self, width, height):
        pt = (self.height - height) >> 1
        pr = (self.width - width) >> 1

        pb = self.height - pt 
        pl = self.width - pr 

        return pt, pb, pr, pl