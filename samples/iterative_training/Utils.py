import colorsys
import random
import cv2
import numpy as np
import os
import pickle


class Utils:
    @staticmethod
    def applyMiniMasks_alpha(image, bboxes, masks, colorMap, alpha=.5):
        for i, ((y1, x1, y2, x2), mask) in enumerate(zip(bboxes, masks)):
            color = colorMap[i]
            patch = image[y1:y2, x1:x2].copy()
            patch[mask] = color
            cv2.addWeighted(image[y1:y2, x1:x2], alpha, patch, 1 - alpha, 0,
                            dst=image[y1:y2, x1:x2], dtype=cv2.CV_8U)
        return image

    @staticmethod
    def applyMiniMasks(image, bboxes, masks, colorMap):
        for i, ((y1, x1, y2, x2), mask) in enumerate(zip(bboxes, masks)):
            color = colorMap[i]
            image[y1:y2, x1:x2][mask] = color
        return image

    @staticmethod
    def display_instances(image, boxes, miniMasks, class_ids, scores):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            return image
        else:
            assert boxes.shape[0] == len(miniMasks) == class_ids.shape[0]

        # Generate random colors
        colors = Utils.random_colors(256)

        masked_image = image
        masked_image = Utils.applyMiniMasks_alpha(masked_image, boxes, miniMasks, colors)

        for i in range(N):
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), colors[i], 1)

            classId = class_ids[i]
            score = scores[i]
            score = int(score * 100)
            if score == 100:
                instanceLabel = f'{classId}'
            else:
                instanceLabel = f'{classId}/{score}'
            cv2.putText(masked_image, instanceLabel, ((x1 + x2) // 2, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))

        return masked_image

    @staticmethod
    def display_instances_fullsizedMasks(image, boxes, masks, class_ids, scores):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            return image
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # Generate random colors
        colors = Utils.random_colors(256)
        colorMap = np.reshape(colors, [256, 1, 3]).astype(np.uint8)

        masked_image = image
        masked_image = Utils.applyFullsizedMasks(masked_image, masks, colorMap)

        for i in range(N):
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), colors[i], 1)

            classId = class_ids[i]
            score = scores[i]
            score = int(score * 100)
            if score == 100:
                instanceLabel = f'{classId}'
            else:
                instanceLabel = f'{score}/{classId}'
            cv2.putText(masked_image, instanceLabel, ((x1 + x2) // 2, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255))

        return masked_image

    @staticmethod
    def applyFullsizedMasks(image, masks, colorMap, alpha=.5):
        if colorMap is None:
            colorMap = Utils.createColorMap()
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

    @staticmethod
    def createColorMap():
        return np.reshape(Utils.random_colors(256), [256, 1, 3]).astype(np.uint8)

    @staticmethod
    def rgb2bgr(rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def random_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: Utils.hsv2rgb(c), hsv))
        random.shuffle(colors)
        return colors

    @staticmethod
    def hsv2rgb(hsv):
        r, g, b = colorsys.hsv_to_rgb(*hsv)
        return int(r * 255), int(g * 255), int(b * 255)

    @staticmethod
    def exploreDatasets(*datasets):
        import cv2
        if len(datasets) == 0:
            print('exploreDatasets: no datasets to explore')
            return
        for dataset in datasets:
            for imageId in dataset.image_ids:
                image = dataset.load_image(imageId)
                fileName = dataset.image_annotation(imageId).name
                masks, classIds = dataset.load_mask(imageId)

                instancesImage = Utils.applyFullsizedMasks(image.copy(), masks, colorMap=None)
                cv2.imshow('Instances', Utils.rgb2bgr(instancesImage))
                cv2.setWindowTitle('Instances', fileName)
                cv2.waitKey()
        while cv2.waitKey() != 27: pass

    @staticmethod
    def findFilePath(searchPaths, fileName):
        if os.path.isfile(fileName):
            return fileName
        for searchPath in searchPaths:
            filePath = os.path.join(searchPath, fileName)
            if os.path.isfile(filePath):
                return filePath
        return None

    @staticmethod
    def savePickle(obj, picklePath):
        with open(picklePath, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadPickle(picklePath):
        with open(picklePath, 'rb') as f:
            return pickle.load(f)


class contexts:
    def __init__(self, *contextObjects):
        self.contextObjects = contextObjects

    def __enter__(self):
        for o in self.contextObjects:
            o.__enter__()
        return self.contextObjects

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            for o in self.contextObjects:
                o.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.contextObjects = None
