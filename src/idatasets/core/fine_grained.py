from ..utils import ClassifyListData, ImageReader
from torchvision import transforms
from PIL import Image


class FGDataset:
    def __init__(self, list_data, mode, data_root='', inp_size=600, img_size=448, **kwargs):
        if isinstance(inp_size, int):
            inp_size = (inp_size, inp_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.mode = mode
        self.img_size = img_size
        self.inp_size = inp_size
        if mode == 'inference':
            self.datas = []
        else:
            self.datas = ClassifyListData(list_data, data_root)
        
        trans = kwargs.get('transforms', None)
        if self.mode == 'train':
            transform = transforms.Compose([transforms.Resize(inp_size),
                                            transforms.RandomCrop(img_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([transforms.Resize(inp_size),
                                            transforms.CenterCrop(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            if self.mode == 'inference':
                self.reader = ImageReader()
        self.transform = trans or transform
        self.post_proc = self.__post_proc \
            if 'with_path' in kwargs and kwargs['with_path'] else lambda image, label, path: (image, label)

    @staticmethod
    def __post_proc(image, label, path):
        return image, label, path

    def __getitem__(self, index):
        img_dir, img_lab = self.datas[index]
        image = Image.open(img_dir).convert('RGB')
        image = self.transform(image)
        return self.post_proc(image, img_lab, img_dir)

    def __call__(self, img_buf):
        assert self.mode == 'inference', 'please use mode "inference"'
        image = self.reader.decode(img_buf)
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.datas)
