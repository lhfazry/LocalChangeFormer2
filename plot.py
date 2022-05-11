import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_root', type=str)

    args = parser.parse_args()

    train_acc = np.load(os.path.join(args.checkpoint_root, 'train_acc.npy'))
    val_acc = np.load(os.path.join(args.checkpoint_root, 'val_acc.npy'))
    print(f'train_acc: \n{train_acc}\n')
    print(f'val_acc: \n{val_acc}\n')

    epochs_range = range(len(train_acc))

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plot_image = os.path.join(args.checkpoint_root, 'plot.png')

    if os.path.isfile(plot_image):
        os.remove(plot_image) 

    plt.savefig(plot_image)
    print('plot image saved')
