import PIL
import numpy as np
from torchvision import transforms


class Pad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __shape2board(self, width, height):
        
        pt = (self.height - height) >> 1
        pr = (self.width - width) >> 1

        pb = self.height - pt 
        pl = self.width - pr 

        return pt, pb, pr, pl
    
    def __call__(self, image):
        if isinstance(image, PIL.Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[:2]
        else:
            raise ValueError("make sure image type is np.ndarray or PIL.Image")

        pt, pb, pr, pl = self.__shape2board(img_w, img_h)        

        if isinstance(image, PIL.Image.Image):
            image = transforms.Pad((pt, pl, pb, pr), fill=self.fill, padding_mode=self.padding_mode)(image)
        else:
            image = np.pad(image, ((pt, pb), (pr, pl), (0, 0)), mode=self.padding_mode, constant_values=self.constant_values)

        return image
