import random
import numpy as np


class RandomReplicate:
    def __init__(self, bdim=0):
        self.bdim = bdim 

    def __call__(self, image, targets):
        h, w = image.shape[:2]
        boxes = targets[:, self.bdim:self.bdim+4].astype(int)
        x1, y1, x2, y2 = boxes.T
        s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
        for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
            x1b, y1b, x2b, y2b = boxes[i]
            bh, bw = y2b - y1b, x2b - x1b
            yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
            x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
            image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            boxes = np.append(boxes, [[x1a, y1a, x2a, y2a]], axis=0)

        return image, boxes 