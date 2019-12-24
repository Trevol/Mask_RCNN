import skimage.io
import numpy as np
from cv2 import cv2

from samples.iterative_training.ImshowWindow import ImshowWindow
from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Timer import timeit
from samples.iterative_training.Utils import Utils
from samples.iterative_training.pins.PinsConfig import PinsInferenceConfig
from mrcnn import utils


def main_predict_and_save_minimasks(pickleFile):
    config = PinsInferenceConfig()
    modelDir = 'logs'
    model = MaskRCNNEx(mode='inference', config=config, model_dir=modelDir)
    model.load_weights('mask_rcnn_pins_0008.h5', by_name=True)

    image = skimage.io.imread('f_7926_528400.00_528.40.jpg')

    with timeit():
        r = model.detect_minimasks([image])[0]

    Utils.savePickle(r, pickleFile)


def main_load_and_display_minimasks(pickleFile):
    r = Utils.loadPickle(pickleFile)

    boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
    colors = Utils.random_colors(256)
    colorMap = colors #np.reshape(colors, [256, 3]).astype(np.uint8)

    image = skimage.io.imread('f_7926_528400.00_528.40.jpg')

    copy = image.copy()
    with timeit('applyBboxMasks'):
        instancesImage = Utils.applyMiniMasks(copy, boxes, masks, colorMap)
    copy = image.copy()
    with timeit('applyBboxMasks'):
        instancesImage = Utils.applyMiniMasks(copy, boxes, masks, colorMap)
    copy = image.copy()
    with timeit('applyBboxMasks'):
        instancesImage = Utils.applyMiniMasks(copy, boxes, masks, colorMap)

    copy = image.copy()
    with timeit('applyBboxMasks_alpha'):
        instancesImage = Utils.applyMiniMasks_alpha(copy, boxes, masks, colorMap)
    copy = image.copy()
    with timeit('applyBboxMasks_alpha'):
        instancesImage = Utils.applyMiniMasks_alpha(copy, boxes, masks, colorMap)
    copy = image.copy()
    with timeit('applyBboxMasks_alpha'):
        instancesImage = Utils.applyMiniMasks_alpha(copy, boxes, masks, colorMap)

    ImshowWindow('').imshow(instancesImage)
    cv2.waitKey()


# main_predict_and_save_minimasks('minimasks.pickle')
main_load_and_display_minimasks('minimasks.pickle')
