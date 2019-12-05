from time import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from samples.balloon.dataset.BaloonDataset import BalloonDataset


def main():
    dataset = BalloonDataset()
    dataset.load_balloon('few_images', 'train')
    dataset.prepare()
    axes = plt.figure().subplots(len(dataset.image_ids), 2)
    for imageId, (ax1, ax2) in zip(dataset.image_ids, axes):
        image = dataset.load_image(imageId)
        masks, classIds = dataset.load_mask(imageId)
        print(classIds)
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

        aggregatedMask = np.sum(masks, -1, keepdims=True)
        splash = np.where(aggregatedMask, image, gray)

        ax1.imshow(image)
        ax2.imshow(splash)
    plt.show()


main()
