import cv2
cv2.setNumThreads(0)
import os
import glob
import random
import requests
from io import BytesIO
import numpy as np
from PIL import Image
from typing import Callable, List, Any
from .parameters import comkey
from .. import gformat

try:
    import decord
    DECORD_FLAG = True
except ImportError:
    DECORD_FLAG = False
    # print("if you use decord, pip install decord, please get some more information: https://github.com/dmlc/decord.git")

try:
    from turbojpeg import TurboJPEG
    TBJPEG_FLAG = True
except ImportError:
    TBJPEG_FLAG = False
    # print('if you use torbojpeg, please get some more information: https://github.com/lilohuang/PyTurboJPEG.git')

__all__ = ["ImageReader", "VideoReader", "VisualAutoReader", "VisualLoopReader"]


class ImageReader(object):
    """视觉资源读取

        Args:
            read_type (str, optional): 读取类型，支持pillow|opencv. Defaults to 'pillow'.
        Raises:
            ValueError: 不支持上述两种类型
            ImportError: 缺少PyTurboJPEG库或以来库
    """
    IMAGE_PILLOW = 'pillow'
    IMAGE_OPENCV = 'opencv'
    IMAGE_TBJPEG = 'turbojpeg'

    def __init__(self, read_type='pillow', *args, **kwargs):
        if read_type == 'turbojpeg':
            if not TBJPEG_FLAG:
                raise ImportError("please pip PyTurboJPEG library before use it, get some more information: https://github.com/lilohuang/PyTurboJPEG.git") 
            else:
                lib_path = kwargs.get('lib_path', None)
                self.jpeg = TurboJPEG(lib_path=lib_path)
                if 'lib_path' in kwargs:
                    kwargs.pop('lib_path')
                # pixel_format = kwargs.get('pixel_format', None)
                # if pixel_format is None:
                #     kwargs['pixel_format'] = TJPF_RGB
                self.kwargs = kwargs

        self.__decodes = {
            "pillow": self.__read_impil,
            "opencv": self.__read_imocv,
            'turbojpeg': self.__read_imjpg
        }

        try:
            self.__decode = self.__decodes[read_type]
        except KeyError:
            raise ValueError("invalid read type('pillow, opencv'): {}".format(read_type))

    def decode(self, inp_buf):
        return self.__decode(inp_buf)

    def image_decode(self, inp_buf):
        return self.__decode(inp_buf)

    @staticmethod
    def __read_impil(inp_buf):
        if isinstance(inp_buf, bytes):
            img = np.asarray(bytearray(inp_buf), dtype='uint8')
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(inp_buf, str) and os.path.exists(inp_buf):
            image = Image.open(inp_buf)
            image = image.convert('RGB')
        elif isinstance(inp_buf, str) and 'http' == inp_buf.strip()[:4]:
            resp = requests.get(inp_buf)
            image = Image.open(BytesIO(resp.content))
            image = image.convert('RGB')
        elif isinstance(inp_buf, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(inp_buf, cv2.COLOR_BGR2RGB))
        elif isinstance(inp_buf, Image.Image):
            image = inp_buf
        else:
            raise ValueError("Error: Not support this type image buffer(byte, str, np.ndarry, PIL.Image)")
        
        return image

    @staticmethod
    def __read_imocv(inp_buf):
        if isinstance(inp_buf, bytes):
            img = np.asarray(bytearray(inp_buf), dtype='uint8')
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        elif isinstance(inp_buf, str) and os.path.isfile(inp_buf):
            image = cv2.imread(inp_buf, cv2.IMREAD_COLOR)
        elif isinstance(inp_buf, str) and 'http' == inp_buf.strip()[:4]:
            resp = requests.get(inp_buf)
            image = cv2.imdecode(np.fromstring(resp.content, np.unit8), 1)
        elif isinstance(inp_buf, np.ndarray):
            image = inp_buf
        elif isinstance(inp_buf, Image.Image):
            image = cv2.cvtColor(np.asarray(inp_buf), cv2.COLOR_RGB2BGR)
        else:
            image = None
            raise ValueError('Error: Not support this type image buffer(byte, str, np.ndarry, PIL.Image)')

        return image
    
    def __read_imjpg(self, jpg_buf):
        jpg_ext = ('.jpg', '.jpeg', '.JPG', '.JPEG')
        if isinstance(jpg_buf, bytes):
            image = self.jpeg.decode(jpg_buf, **self.kwargs)
        elif isinstance(jpg_buf, str) and jpg_buf.endswith(jpg_ext) and os.path.isfile(jpg_buf):
            jfile = open(jpg_buf, 'rb')
            image = self.jpeg.decode(jfile.read(), **self.kwargs)
        elif isinstance(jpg_buf, np.ndarray):
            image = jpg_buf
        elif isinstance(jpg_buf, Image.Image):
            image = np.asarray(jpg_buf)
        else:
            image = None
            raise ValueError('Error: Not support this type image buffer(byte, str(.jpeg, .jpg), np.ndarry, PIL.Image')
        
        return image


class VideoReader(ImageReader):
    """ 视频读取

        Args:
            read_type (str/tuple, optional): 视频帧读取类型(video, image)，支持opencv|decord|frames. Defaults to pillow.
            type_name (str, optional): frames中图片的扩展名. Defaults to '.jpg'.
            min_vdim_frame (int): 读取视频中最小帧数
            gen_seq_idx (function): 读取视频帧索引函数，gen_seq_idx(capture, nframe)
            **decord_kwargs (dict, optional): decord读取视频时的参数
        Raises:
            ValueError: 不支持上述三种类型
        Note:
            opencv: 读取视频结果，[Image(BGR), ..., Image(BGR)]
            decord: 读取视频结果, Numpy(N,3,H,W), RGB
            frames: 读取视频结果，[Image(BGR), ..., Image(BGR)] or [Image(Pillow), ..., Image(Pillow)]
    """
    VIDEO_OPENCV = 'opencv'
    VIDEO_DECORD = 'decord'
    VIDEO_FRAMES = 'frames'
    def __init__(self, read_type=('decord', 'pillow'), type_name='.jpg', min_vdim_frame=1,
                 gen_seq_idx=None, **kwargs):

        if isinstance(read_type, str):
            read_type = (read_type, 'pillow')
        elif isinstance(read_type, (tuple, list)) and len(read_type) == 1:
            read_type = (read_type[0], 'pillow')
        else:
            pass
        image_read_type = read_type[1] if isinstance(read_type, tuple) and len(read_type) == 2 else ImageReader.IMAGE_PILLOW
        super(VideoReader, self).__init__(read_type=image_read_type, **kwargs)

        
        self.type_name = type_name
        self.min_vdim_frame = min_vdim_frame

        self.__decodes = {
            "opencv": self.__read_vdcv,
            "decord": self.__read_vdde,
            "frames": self.__read_vdim
        }

        try:
            self.__decode = self.__decodes[read_type[0]]
        except KeyError:
            raise ValueError("invalid video read type(decord, opencv, frames): {}".format(read_type))
        if DECORD_FLAG:
            self.decord_kwargs = comkey(decord.VideoReader, kwargs)
            if 'ctx' not in self.decord_kwargs.keys():
                self.decord_kwargs['ctx'] = decord.cpu(0)
            if 'num_threads' not in self.decord_kwargs.keys():
                self.decord_kwargs['num_threads'] = 2

        self._gen_seq_idx_ = gen_seq_idx or self._gen_seq_idx

    def decode(self, inp_buf):
        return self.__decode(inp_buf)

    def video_decode(self, inp_buf):
        return self.__decode(inp_buf)

    def _gen_seq_idx(self, cap, nframes):
        return [i for i in range(nframes)]

    def __read_vdcv(self, vid_pth):
        cap = cv2.VideoCapture(vid_pth)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = self._gen_seq_idx_(cap, nframes)

        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            flag, image = cap.read()
            frames.append(image)

        return frames

    def __read_vdde(self, vid_pth):
        cap = decord.VideoReader(vid_pth, **self.decord_kwargs)
        idxs = self._gen_seq_idx_(cap, len(cap))
        frames = cap.get_batch(idxs).asnumpy()
        return frames

    def __read_vdim(self, vid_pth):
        if not isinstance(vid_pth, list):
            paths = glob.glob(os.path.join(vid_pth, '*'+self.type_name))
            paths = sorted(paths, key=lambda x: os.path.getctime(x))
        else:
            paths = vid_pth

        idxs = self._gen_seq_idx_(paths, len(paths))

        frames = []
        for idx in idxs:
            try:
                image = self.image_decode(paths[idx])
                assert image is not None
            except:
                continue
            frames.append(image)
        assert len(frames) > self.min_vdim_frame, "can not read {} frames from {}".format(self.min_vdim_frame, vid_pth)

        return frames


class VisualAutoReader(VideoReader):
    """ 视觉数据读取

        Args:
            video_extention (tuple, optional): 视频文件的扩展名. Defaults to ('.mp4').
    """
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        super(VisualAutoReader, self).__init__(*args, **kwargs)

    def decode(self, inp_buf):
        if os.path.isdir(inp_buf) or inp_buf.lower().endswith(gformat['video']):
            return self.video_decode(inp_buf)
        else:
            return [self.image_decode(inp_buf)]


class VisualLoopReader(VisualAutoReader):
    """ 视觉资源循环读取

        Args:
            list_data (List): 输入包含图像或视频资源的列表类型数据
            retry (int, optional): 读取失败尝试的次数. Defaults to -1.
            args (Any): VisualReader参数
            kwargs (Any): VisualReader参数
    """
    def __init__(self, list_data:List, get_img:Callable[[Any], str], retry=-1, *args, **kwargs):
        super(VisualLoopReader, self).__init__(*args, **kwargs)
        self.datas = list_data
        self.retry = retry
        self.__get_img = get_img
    
    def _back_list(self, idx) ->List:
        return []

    def __getitem__(self, idx):
        return self.read(idx) 
    
    def read(self, index):
        img = None
        retry = self.retry
        while retry != 0 or retry < 0:
            try:
                data_unit = self.datas[index]
                img_dirs = self.__get_img(data_unit)
                img = self.decode(img_dirs)
                assert self._valid(img)
            except Exception as e:
                back_list = self._back_list(index)
                if back_list is not None and len(back_list):
                    idx = random.choice(back_list)
                else:
                    idx = random.randint(0, len(self.datas))
            retry -= 1
        return img, idx