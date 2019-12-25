import skimage.io
import numpy as np
from cv2 import cv2

from samples.iterative_training.ImshowWindow import ImshowWindow
from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Timer import timeit
from samples.iterative_training.Utils import Utils
from mrcnn import utils
from samples.iterative_training.pins.rough_dataset.RoughAnnotatedPinsConfig import RoughAnnotatedPinsInferenceConfig


def main_predict_and_save_minimasks(pickleFile):
    config = RoughAnnotatedPinsInferenceConfig()
    modelDir = 'logs'
    model = MaskRCNNEx(mode='inference', config=config, model_dir=modelDir)
    model.load_weights('mask_rcnn_pins_0040.h5', by_name=True)

    image = skimage.io.imread('f_7926_528400.00_528.40.jpg')

    with timeit():
        r = model.detect_minimasks([image])[0]

    Utils.savePickle(r, pickleFile)


class Vis:
    @staticmethod
    def applyMiniMasks_alpha(image, bboxes, masks, colorMap, alpha=.5):
        for i, ((y1, x1, y2, x2), mask) in enumerate(zip(bboxes, masks)):
            color = colorMap[i]
            patch = image[y1:y2, x1:x2].copy()
            patch[mask] = color
            cv2.addWeighted(image[y1:y2, x1:x2], alpha, patch, 1 - alpha, 0,
                            dst=image[y1:y2, x1:x2], dtype=cv2.CV_8U)
        return image

    @classmethod
    def display_instances(cls, image, boxes, miniMasks, instancesClassIds, scores, mode='instances', colors=None):
        assert mode in ('instances', 'classes')
        N = boxes.shape[0]  # Number of instances
        if not N:
            print("\n*** No instances to display *** \n")
            return image
        assert boxes.shape[0] == len(miniMasks) == instancesClassIds.shape[0]

        # Generate random colors
        colors = cls.createInstanceColors(N, instancesClassIds, mode, colors)

        masked_image = image
        masked_image = cls.applyMiniMasks_alpha(masked_image, boxes, miniMasks, colors)

        for i in range(N):
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), colors[i], 1)

            classId = instancesClassIds[i]
            score = scores[i]
            score = int(score * 100)
            if score == 100:
                instanceLabel = f'{classId}'
            else:
                instanceLabel = f'{classId}/{score}'
            cv2.putText(masked_image, instanceLabel, ((x1 + x2) // 2, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))

        return masked_image

    @classmethod
    def display_instances_proto(cls, image, boxes, miniMasks, class_ids, scores, mode='instances', colors=None):
        # select color for instance
        # create map from instance index to color
        N = boxes.shape[0]
        if not N:
            return image
        instanceColors = cls.createInstanceColors(N, class_ids, mode, colors)

    @staticmethod
    def createInstanceColors(n, instancesClassIds, mode, colors=None):
        assert n > 0
        assert instancesClassIds.shape[0] == n
        assert mode in ['instances', 'classes']
        assert colors is None or isinstance(colors, (int, tuple, list))

        if mode == 'instances':
            if colors is None:
                return Utils.random_colors(n)
            if isinstance(colors, (int, tuple)):  # single color for all instances
                return [colors] * n
            # colors is list => user mappings for each instance
            assert len(colors) >= n
            return colors

        if mode == 'classes':  # prepare list of
            if isinstance(colors, (int, tuple)):  # single color for all instances
                return [colors] * n
            maxClassId = np.max(instancesClassIds)
            if colors is None:
                classColors = Utils.random_colors(maxClassId + 1)
                return [classColors[i] for i in instancesClassIds]  # map instance => classId => color
            # colors is list => user mappings for each class
            assert len(colors) > maxClassId
            return [colors[i] for i in instancesClassIds]  # map instance => classId => color

        raise Exception('Unexpected Params')


def main_load_and_display_minimasks(pickleFile):
    r = Utils.loadPickle(pickleFile)

    boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
    colors = Utils.random_colors(256)

    image = skimage.io.imread('f_7926_528400.00_528.40.jpg')

    # TODO: display classes
    print(np.unique(classIds))

    classColors = [
        None,  # 0
        (0, 255, 0),  # 1 - pin
        (255, 0, 0),  # 2 - solder
    ]

    copy = image.copy()
    with timeit('applyBboxMasks_alpha'):
        # instancesImage = Utils.applyMiniMasks_alpha(copy, boxes, masks, colors)
        instancesImage = Vis.display_instances(copy, boxes, masks, classIds, scores, mode='classes', colors=classColors)

    ImshowWindow('').imshow(instancesImage)
    cv2.waitKey()


# main_predict_and_save_minimasks('minimasks.pickle')
main_load_and_display_minimasks('minimasks.pickle')
