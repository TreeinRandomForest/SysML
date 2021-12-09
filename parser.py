from PIL import Image
import numpy as np
import matplotlib.pylab as plt

import glob

plt.ion()

DATA_PATH = '/home/sanjay/data/sanjay/SysML/data/'

train_list = glob.glob(f'{DATA_PATH}/train/*.bmp')
test_list = glob.glob(f'{DATA_PATH}/test/*.bmp')

def load_all_data(file_list):
    img_list, target_list = [], []
    
    for f in file_list:
        try:
            img, target = read_file(f)
        except:
            print(f)
            continue

        img_list.append(img)
        target_list.append(target)

    return img_list, target_list


def read_file(fname):
    img = np.array(Image.open(fname)).astype(int)

    target = fname.split('/')[-1].split('.')[0].split('_')[1].split(':')
    target = [int(t) for t in target]

    return img, target
