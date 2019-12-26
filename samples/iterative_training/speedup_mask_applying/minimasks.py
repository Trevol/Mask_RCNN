import skimage.io
import numpy as np
from cv2 import cv2

from samples.iterative_training.ImshowWindow import ImshowWindow
from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Timer import timeit
from samples.iterative_training.Utils import Utils
from mrcnn import utils
from samples.iterative_training.pins.rough_dataset.RoughAnnotatedPinsConfig import RoughAnnotatedPinsInferenceConfig


def main_predict_and_save_minimasks(pickleFile, imageFile):
    config = RoughAnnotatedPinsInferenceConfig()
    modelDir = 'logs'
    model = MaskRCNNEx(mode='inference', config=config, model_dir=modelDir)
    model.load_weights('mask_rcnn_pins_0047.h5', by_name=True)

    image = skimage.io.imread(imageFile)

    with timeit():
        r = model.detect_minimasks([image])[0]

    Utils.savePickle(r, pickleFile)


def main_load_and_display_minimasks(pickleFile, imageFile):
    r = Utils.loadPickle(pickleFile)

    boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
    colors = Utils.random_colors(256)

    image = skimage.io.imread(imageFile)

    # TODO: display classes
    print(np.unique(classIds))

    classColors = [
        None,  # 0
        (255, 0, 0),  # 1 - pin
        (0, 255, 0),  # 2 - solder

    ]

    copy = image.copy()
    with timeit('applyBboxMasks_alpha'):
        # instancesImage = Utils.applyMiniMasks_alpha(copy, boxes, masks, colors)
        instancesImage = Utils.display_instances(copy, boxes, masks, classIds, scores, mode='classes',
                                                 colors=classColors)

    ImshowWindow('').imshow(instancesImage)
    cv2.waitKey()


pickleFile = 'minimasks.pickle'
imageFile = 'f_1946_129733.33_129.73.jpg' #'f_1955_130333.33_130.33.jpg' # 'f_7926_528400.00_528.40.jpg'
main_predict_and_save_minimasks(pickleFile, imageFile)
main_load_and_display_minimasks(pickleFile, imageFile)
