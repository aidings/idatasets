import os
import pandas as pd


class ListData:
    def __init__(self, list_data, data_root='./', **kwargs):
        """ 基础数据类型

        Args:
            list_data (str, list): 输入列表数据|输入列表文件路径
            kwargs (dict): pandas parameters
        """
        if isinstance(list_data, list):
            datas = list_data
        elif isinstance(list_data, str) and list_data.endswith(('.list', '.txt', '.lst', '.train', '.test')):
            datas = open(list_data, 'r')
        elif isinstance(list_data, str) and list_data.endswith('.csv'):
            df = pd.read_csv(list_data, **kwargs)
            datas = df.to_dict(orient='records')
        elif isinstance(list_data, str) and list_data.endswith('.xlsx'):
            df = pd.read_excel(list_data, **kwargs)
            datas = df.to_dict(orient='records')
        else:
            raise ValueError('Not support this type')

        self.data_root = data_root
        self.kwargs = kwargs
        flag = self.__load_list(datas)
        assert flag, "Error: can not load some data into memory."

    def _parse_line(self, line):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.datas)

    @property 
    def size(self):
        return len(self.datas)
    
    @staticmethod
    def _line_cnts(label_file):
        count = -1
        with open(label_file, 'rb') as f:
            count = 0
            last_data = '\n'
            while True:
                data = f.read(0x400000)
                if not data:
                    break
                count += data.count(b'\n')
                last_data = data
            if last_data[-1:] != b'\n':
                count += 1 # Remove this if a wc-like count is needed
        return count

    def __load_list(self, list_data):
        self.datas = []
        for line in list_data:
            data_dict = self._parse_line(line)
            self.datas.append(data_dict)
            if data_dict is None:
                self.datas.pop()
        return len(self.datas) > 0
    
    def __getitem__(self, index):
        return self.datas[index]
    

class ClassifyListData(ListData):
    def _parse_line(self, line):
        img_dir, img_lab = line.strip().split()
        img_dir = os.path.join(self.data_root, img_dir)
        if not os.path.exists(img_dir):
            return None

        return [img_dir, int(img_lab)]


if __name__ == '__main__':
    datas = ListData('/data/image/FGCV/CUB_200_2011/list/cub_bird.test', '/data/image/FGCV/CUB_200_2011')

    print(datas[10])