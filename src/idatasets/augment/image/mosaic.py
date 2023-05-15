import cv2
import random
import numpy as np


class MosaicDataset:
    """MosaicDataset
        Args:
            out_size: 1/4的窗口大小
            xyc: 窗口中心的变化范围

    """
    def __init__(self, out_size, dataset, xyc=(0.5, 1.5)) -> None:
        if isinstance(out_size, int):
            self.input_w = out_size
            self.input_h = out_size
        elif isinstance(out_size, tuple) or isinstance(out_size, list):
            self.input_h, self.input_h = out_size[:2]
        
        self.__dataset = dataset
        self.__xyc = xyc
    
    def get_mosaic_coordinate(self, mosaic_index, xc, yc, w, h):
        # TODO update doc
        # index0 to top left part of image
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, self.input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(self.input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, self.input_w * 2), min(self.input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
        return (x1, y1, x2, y2), small_coord
    
    def __getitem__(self, idx):
        mosaic_labels = []

        # yc, xc = s, s  # mosaic center x, y
        yc = int(random.uniform(self.__xyc[0] * self.input_h, self.__xyc[1] * self.input_h))
        xc = int(random.uniform(self.__xyc[0] * self.input_w, self.__xyc[1] * self.input_w))

        # 3 additional image indices
        indices = [idx] + [random.randint(0, len(self.__dataset) - 1) for _ in range(3)]

        for i_mosaic, index in enumerate(indices):
            datas = self.__dataset[index]
            img, _labels = datas[:2]
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * self.input_h / h0, 1. * self.input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )
            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((self.input_h * 2, self.input_w * 2, c), 114, dtype=np.uint8)

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            labels = _labels.copy()
            # Normalized xywh to pixel xyxy format
            if _labels.size > 0:
                labels[:, 0] = scale * _labels[:, 0] + padw
                labels[:, 1] = scale * _labels[:, 1] + padh
                labels[:, 2] = scale * _labels[:, 2] + padw
                labels[:, 3] = scale * _labels[:, 3] + padh
            mosaic_labels.append(labels)

        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 0], 0, 2 * self.input_w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 * self.input_h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * self.input_w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * self.input_h, out=mosaic_labels[:, 3])

        return mosaic_img, mosaic_labels