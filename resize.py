import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from argparse import ArgumentParser
from glob import glob

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--dataset_root', type=str)

    args = parser.parse_args()
    dataset_root = args.dataset_root

    for f2 in ['A', 'B', 'label']:
        print(f'Processing {f2}')

        for fi1 in glob(os.path.join(dataset_root, f2, '*.png')):
            #print(f'Resizing {fi1}')
            img = cv2.imread(fi1)
            resized = cv2.resize(img, (128, 128))

            if f2 == 'label':
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(fi1, resized)
