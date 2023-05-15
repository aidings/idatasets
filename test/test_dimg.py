import sys

sys.path.insert(0, '../src')

from idatasets import ImageReader


if __name__ == '__main__':

    imread = ImageReader(ImageReader.IMAGE_TBJPEG)

    img_pth = '/data/image/FGCV/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
    img = imread.decode(img_pth)

    print(img)