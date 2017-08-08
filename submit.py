import os

import numpy as np


def main():
    if not os.path.exists('1pixel'):
        os.makedirs('1pixel')

    p1 = np.zeros([500, 500], dtype=np.uint8)
    p1[0, 0] = 1

    for f in os.listdir('data/training/truth'):
        if f[-4:] == '.png':
            txt_path = '1pixel/' + f.split('.')[0] + '.txt'
            np.savetxt(txt_path, p1, fmt='%d', delimiter='')
    for f in os.listdir('data/testing/images'):
        if f[-4:] == '.tif':
            txt_path = '1pixel/' + f.split('.')[0] + '_mask.txt'
            np.savetxt(txt_path, p1, fmt='%d', delimiter='')

if __name__ == '__main__':
    main()
