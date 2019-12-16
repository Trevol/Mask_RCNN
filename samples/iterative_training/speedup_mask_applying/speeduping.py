import cv2
import numpy as np
from samples.iterative_training.Timer import timeit
from samples.iterative_training.Utils import Utils


def main():
    imageFile = "f_7926_528400.00_528.40.jpg"
    maskFile = "f_7926_528400.00_528.40_masks.npy"
    image = cv2.imread(imageFile)
    masks = np.load(maskFile)

    color = np.uint8([0, 200, 0])

    # copy = image.copy()
    # colorMap = np.reshape(Utils.random_colors(256), [256, 1, 3]).astype(np.uint8)
    # colorMap[0, 0] = (0, 0, 0)
    # with timeit('applyMasks_0'):
    #     maskedImage_0 = InitialMasks.applyMasks_0(copy, masks, colorMap)

    # with timeit('applyMasks_1'):
    #     maskedImage_1 = applyMasks_1(image.copy(), masks, color)

    # copy = image.copy()
    # with timeit('applyMasks_aggregatedAny_setAlphaColorWhereMask'):
    #     maskedImage_2 = applyMasks_aggregatedAny_setAlphaColorWhereMask(copy, masks, color)
    #
    # copy = image.copy()
    # with timeit('applyMasks_aggregatedAny_setColorWhereMask'):
    #     maskedImage_2_1 = applyMasks_aggregatedAny_setColorWhereMask(copy, masks, color)

    # copy = image.copy()
    # colorMap = np.reshape(Utils.random_colors(256), [256, 1, 3]).astype(np.uint8)
    # colorMap[0, 0] = (0, 0, 0)
    # with timeit('applyMasks_multiply_max_colorMap'):
    #     maskedImage_3 = applyMasks_multiply_max_colorMap(copy, masks, colorMap)

    colorMap = np.reshape(Utils.random_colors(256), [256, 1, 3]).astype(np.uint8)

    copy = image.copy()
    with timeit('applyMasks_argmax_colorMap'):
        maskedImage_4 = applyMasks_argmax_colorMap(copy, masks, colorMap)
    copy = image.copy()
    with timeit('applyMasks_argmax_colorMap'):
        maskedImage_4 = applyMasks_argmax_colorMap(copy, masks, colorMap)
    copy = image.copy()
    with timeit('applyMasks_argmax_colorMap'):
        maskedImage_4 = applyMasks_argmax_colorMap(copy, masks, colorMap)

    cv2.imshow('maskedImage_4', maskedImage_4)
    cv2.waitKey()


def applyMasks_argmax_colorMap(image, masks, colorMap, alpha=.5):
    zeroInstanceMask = masks[..., 0]
    instanceIndexes = np.argmax(masks, -1).astype(np.uint8)
    # First instance marked by 0 and BG has 0 value
    # so set different value for first instance
    instanceIndexes[zeroInstanceMask] = 255
    instancesMask = instanceIndexes.astype(np.bool)
    coloredInstances = cv2.applyColorMap(instanceIndexes, colorMap)
    blendedAndMasked = cv2.addWeighted(image, 1 - alpha, coloredInstances, alpha, 0)[instancesMask]
    image[instancesMask] = blendedAndMasked
    return image


##############################################
######################
def applyMasks_multiply_max_colorMap(image, masks, colorMap):
    n = masks.shape[-1]
    instanceMap = np.multiply(masks, list(range(1, n + 1)), dtype=np.uint8, casting='unsafe')
    instanceMap = np.max(instanceMap, -1)
    return cv2.applyColorMap(instanceMap, colorMap)


def applyMasks_aggregatedAny_setAlphaColorWhereMask(image, masks, color):
    aggregatedMask = np.any(masks, -1)
    return set_alphaColor_whereMask(image, aggregatedMask, color)


def applyMasks_aggregatedAny_setColorWhereMask(image, masks, color):
    aggregatedMask = np.any(masks, -1)
    return set_color_whereMask(image, aggregatedMask, color)


def applyMasks_everyMask_alphaColor(image, masks, color):
    n = masks.shape[-1]
    for i in range(n):
        set_alphaColor_whereMask(image, masks[..., i], color)
    return image


def set_alphaColor_whereMask(image, mask, color, alpha=0.5):
    maskedImage = image[mask]
    image[mask] = maskedImage * (1 - alpha) + alpha * color
    return image


def set_color_whereMask(image, mask, color):
    image[mask] = color
    return image


### MASK 0
class InitialMasks:
    @staticmethod
    def applyMasks_0(image, masks, colorMap):
        n = masks.shape[-1]
        for i in range(n):
            InitialMasks.apply_mask_0(image, masks[..., i], colorMap[i, 0])
        return image

    @staticmethod
    def apply_mask_0(image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = np.where(mask,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        return image


main()
