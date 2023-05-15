import random 
import PIL
import numpy as np


class RandomMultiScale:
    """随机多尺度图像（一般用于collate_fn）
    """
    def __init__(self, scale_sizes, resample=PIL.Image.BILINEAR):
        """构造方法

        Args:
            scale_sizes (int or tuple2d): 输入多种尺寸
            resample (resize type, optional): 重采样方法. Defaults to PIL.Image.BILINEAR.
            transform (transform方法, optional): 变换方法. Defaults to None.
        """
        self.scale_sizes = []
        for size in scale_sizes:
            if isinstance(size, int):
                size = (size, size)
            else:
                size = (size[0], size[1])
            self.scale_sizes.append(size)

        self.resample = resample

    
    def __call__(self, images: list):
        """多尺度图像

        Args:
            images (list): 输入图像序列（PIL）

        Returns:
            list : 输出图像序列（PIL）
            np.ndarry : 输出缩放序列 [n, 2] (w, h)
        """
        w, h = random.choice(self.scale_sizes)
        
        imgs = []
        scls = []
        for image in images:
            sw, sh = image.size
            image = image.resize((w, h), resample=self.resample)
            image = self.transform(image)
            imgs.append(image)
            scls.append((w / sw, h / sh))

        return imgs, np.array(scls)