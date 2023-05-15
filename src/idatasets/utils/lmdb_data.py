import lmdb
from typing import List, Any
from .parameters import comkey

class LmdbData:
    def __init__(self, file_name, index_formats:List=[], **kwargs):
        
        # lmdb.open parameters
        lmdb_args = comkey(['readahead', 'create', 'readonly', 'lock', 'map_size'], kwargs)
        self.env = lmdb.open(file_name, **lmdb_args)
        write = True if len(index_formats) > 0 else False
        if write:
            with self.env.begin(write=True) as lmdb_file:
                self.format = index_formats
                encode_formats = '|||'.join(index_formats)
                lmdb_file.put('index_formats'.encode(), encode_formats.encode())
                self.n = 0
        else:
            with self.env.begin(write=False) as lmdb_file:
                self.format = lmdb_file.get('index_formats'.encode()).decode().split('|||')
                self.n = int(lmdb_file.get('ndata'.encode()))

    def __len__(self):
        return self.n

    @property
    def size(self):
        return self.n

    def encode(self, data:Any) -> List:
        pass

    def decode(self, data:Any) -> List:
        pass

    def save(self, datas:List):
        bidx = self.n
        with self.env.begin(write=True) as lmdb_file:
            for index, data in enumerate(datas):
                edata = self.encode(data)
                for idx, value in enumerate(edata):
                    key = self.format[idx].format(bidx+index)
                    lmdb_file.put(key.encode(), value)

            self.n += len(datas)
            lmdb_file.put('ndata'.encode(), str(self.n).encode())

    def load(self, index=None):
        datas = []
        if index is None:
            for i in range(self.n):
                ddata = []
                with self.env.begin(write=False) as lmdb_file:
                    for idx_str in self.format:
                        key = idx_str.format(index)
                        value = lmdb_file.get(key.encode())
                        ddata.append(value)
                data = self.decode(ddata)
                datas.append(data)
        elif isinstance(index, list):
            for i in index:
                ddata = []
                with self.env.begin(write=False) as lmdb_file:
                    for idx_str in self.format:
                        key = idx_str.format(index)
                        value = lmdb_file.get(key.encode())
                        ddata.append(value)
                data = self.decode(ddata)
                datas.append(data)
        else:
            raise ValueError('Not support this index')

        return datas

    def __getitem__(self, index):
        return self.get(index)
    
    def get(self, index):
        ddata = []
        with self.env.begin(write=False) as lmdb_file:
            for idx_str in self.format:
                key = idx_str.format(index)
                value = lmdb_file.get(key.encode())
                ddata.append(value)
        data = self.decode(ddata)
        return data

    def close(self):
        self.env.close()


class ClassifyLmdbData(LmdbData):
    def encode(self, data):
        img_dir, img_lab = data
        if isinstance(img_dir, list):
            enc_str = '|||'.join(img_dir)
            enc = enc_str.encode()
        else:
            enc = img_dir.encode()
        
        return enc, str(img_lab).encode()
    
    def decode(self, data):
        img_dir = data[0].decode().split('|||')
        if len(img_dir) == 1:
            return img_dir[0], data[1].decode()
        else:
            return img_dir, data[1].decode()


def show_lmdb(filename):
    lmdb_env = lmdb.open(filename)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    n_keys = 0
    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:  
                print(key, value)
                n_keys = n_keys + 1

    print('n_keys',n_keys)