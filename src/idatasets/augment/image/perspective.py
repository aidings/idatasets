import cv2
import math
import random
import numpy as np

class RandomPerspective:
    """随机透视变换
        Args:
            degrees (int, tuple) : 旋转的角度, 默认为10, 范围0~360度
            translate (float) : 平移的百分比，范围0~1.0
            scale (float) : 尺度百分比，范围0~1.0
            shear (float, tuple) : 错切角度，默认为10，范围0~180度
            perspective (0, 1)
    """
    def __init__(self, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
        self.degrees = (-degrees, degrees) if isinstance(degrees, int) else degrees
        self.translate = translate
        self.scale = scale
        self.shear = (shear, shear) if isinstance(shear, int) else shear
        self.perspective = (perspective, perspective) if isinstance(perspective, int) else perspective
        self.border = border

    @staticmethod 
    def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

    def __call__(self, image, targets):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [xyxy]

        height = image.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = image.shape[1] + self.border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -width / 2  # x translation (pixels)
        C[1, 2] = -height / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
        # ax[0].imshow(img[:, :, ::-1])  # base
        # ax[1].imshow(img2[:, :, ::-1])  # warped

        # Transform label coordinates
        if targets and len(targets):
            
            n = len(targets)
            # warp points
            xy = np.ones((n * 4, 3))
            # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            idxs = [0, 1, 2, 3, 0, 3, 2, 1]
            xy[:, :2] = targets[:, idxs].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self.box_candidates(box1=targets[:, 0:4].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, 0:4] = xy[i]

        return image, targets