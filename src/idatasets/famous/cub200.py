import os
from ..core import FGDataset


class CUB200:
    def __init__(self, data_root):
        list_file = {"0": [], "1": []}

        image_list = {}
        with open(os.path.join(data_root, "images.txt"), 'r') as f:
            for line in f:
                idx, pth = line.strip().split()
                image_list[idx] = {'path': pth, "label": -1}
        
        with open(os.path.join(data_root, "image_class_labels.txt"), 'r') as f:
            for line in f:
                idx, lab = line.strip().split()
                image_list[idx]['label'] = str(int(lab)-1)

        with open(os.path.join(data_root, "train_test_split.txt"), 'r') as f:
            for line in f:
                idx, istrain = line.strip().split()
                list_file[istrain].append(image_list[idx]['path']+' '+image_list[idx]['label'])
        
        self.list_file = {"train": list_file['1'], "test": list_file['0']}
        
        self.data_root = os.path.join(data_root, 'images')
    
    def train(self, inp_size=256, img_size=224, trans=None, with_path=False):
        return FGDataset(self.list_file["train"], "train", data_root=self.data_root, inp_size=inp_size, img_size=img_size, transforms=trans, with_path=with_path)
    
    def test(self, inp_size=256, img_size=224, trans=None, with_path=False):
        return FGDataset(self.list_file["test"], "test", data_root=self.data_root, inp_size=inp_size, img_size=img_size, transforms=trans, with_path=with_path)

    
     

        
        
