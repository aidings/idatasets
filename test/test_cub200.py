import sys

sys.path.insert(0, "../src")


from idatasets.famous import CUB200


if __name__ == "__main__":
    # cub = CUB200(data_root='f:/dbase/CUB_200_2011/CUB_200_2011')
    cub = CUB200(data_root='/data/image/FGCV/CUB_200_2011')

    dataset = cub.create('test', batch_size=16, shuffle=True, num_workers=4, pin_memory=True, with_path=True)

    for data in dataset:
        print(data[0].shape, data[1], data[2])
