import sys

# import faulthandler
# faulthandler.enable()
# import cv2
# cv2.setNumThreads(0)

sys.path.insert(0, '../src')

from idatasets import ImageReader
# import idatasets

# reader = ImageReader(ImageReader.IMAGE_PILLOW)

def check_unit(data):
    img_dir, img_lab = data
    try:
        reader.decode(img_dir)
    except:
        return False
    return True

def pool_data():
    from idatasets import PoolData, ClassifyListData
    data = ClassifyListData('./cub_bird.test', '/data/image/FGCV/CUB_200_2011/')
    pool = PoolData(data, nproc=5)
    res = pool.run(check_unit)
    print(res)

def test_lmdb():
    from idatasets import ClassifyLmdbData, ClassifyListData, show_lmdb
    data = ClassifyListData('./cub_bird.test', '/data/image/FGCV/CUB_200_2011')
    lmdb = ClassifyLmdbData('./cub_bird.db', index_formats=['image:{}', "label:{}"]) 

    lmdb.save(data)

    show_lmdb('./cub_bird.db')

    lmdb_file = ClassifyLmdbData('./cub_bird.db')
    print('key:500',lmdb_file[500])

def test_fgdata():
    from idatasets.core import FGDataset, BuildDataloader
    dataset = FGDataset(list_data='./cub_bird.test', data_root='/data/image/FGCV/CUB_200_2011/', mode='train')

    # for data in dataset:
    #     print(data[0].shape, data[1])

    import torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    print(len(loader))
    # loader = create_dataloader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=8)
    for data in loader:
        print(data[0].shape, data[1])

def test_fgdata_sampler():
    from idatasets.core import FGDataset, BuildDataloader, BucketSampler
    dataset = FGDataset(list_data='./cub_bird.test', data_root='/data/image/FGCV/CUB_200_2011/', mode='train')

    bs = BucketSampler(dataset.bucket(), 16, True, batch_class=[2, 3, 4, 5], each_mins=3)
    loader = BuildDataloader(pin_memory=True, num_workers=8, batch_sampler=bs)(dataset)
    ddict = {}
    for data in loader:
        print(data[0].shape, data[1])
        label = data[1]
        for lab in data[1]:
            lab = lab.item()
            if lab not in ddict.keys():
                ddict[lab] = 0
            ddict[lab] += 1
    
    for key in ddict.keys():
        print(key, max(bs.bucket.shape(key) - ddict[key], 0))

    # for key in bs.bucket.keys():
        # print(key, bs.bucket.size(key))
    
    print(len(bs), len(dataset))


def test_lcdata():
    from idatasets.core import LCDataset, BuildDataloader

    dataset = LCDataset(scale=[1.2], list_data='./cub_bird.test', data_root='/data/image/FGCV/CUB_200_2011/', mode='train')

    loader = BuildDataloader(batch_size=16, shuffle=True, pin_memory=True, num_workers=8)(dataset)
    for data in loader:
        print(data[0].shape, data[1], data[2]['1.2'].shape)


def test_mnist():
    import torch
    # from idatasets.core import create_dataloader
    from torchvision import transforms, datasets
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
    dataset = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = True)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=8, num_workers=8)

    for data in loader:
        print(data[0].shape, data[1])

if __name__ == '__main__':
    # pool_data()
    # test_lmdb()
    test_fgdata_sampler()
    # test_lcdata()
    # test_mnist()
    
