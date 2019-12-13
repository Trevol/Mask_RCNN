import cv2
import numpy as np
from samples.iterative_training.Timer import timeit


def main():
    imageFile = "f_7926_528400.00_528.40.jpg"
    maskFile = "f_7926_528400.00_528.40_masks.npy"
    image = cv2.imread(imageFile)
    masks = np.load(maskFile)
    print(masks.shape, masks.dtype)

    color = np.uint8([0, 200, 0])
    with timeit('applyMasks', autoreport=False) as t:
        applyMasks(image, masks, color)
    print(t.report())

    cv2.imshow('image', image)
    cv2.waitKey()


def applyMasks(image, masks, color):
    n = masks.shape[-1]
    durations = []
    for i in range(n):
        with timeit('apply mask', autoreport=False) as t:
            apply_mask_2(image, masks[..., i], color)
        durations.append(t.duration)
    print(np.mean(durations))


def apply_mask_1(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def apply_mask_2(image, mask, color, alpha=0.5):
    maskedImage = image[mask]
    image[mask] = maskedImage * (1 - alpha) + alpha * color
    return image


main()
