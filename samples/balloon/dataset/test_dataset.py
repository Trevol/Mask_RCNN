import numpy as np
import cv2
import matplotlib.pyplot as plt
from samples.balloon.dataset.BaloonDataset import BalloonDataset


def main():
    dataset = BalloonDataset()
    dataset.load_balloon('few_images', 'train')
    dataset.prepare()
    for imageId in dataset.image_ids:
        image = dataset.load_image(imageId)
        mask, classIds = dataset.load_mask(imageId)
        print(mask.shape, mask.dtype)
        ax1, ax2 = plt.figure().subplots(1, 2)
        ax1.imshow(image)
        ax2.imshow(image)
    plt.show()

main()
