import colorsys
import random
import cv2
import numpy as np


class Utils:
    @staticmethod
    def display_instances(image, boxes, masks, class_ids, scores):
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
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # Generate random colors
        colors = Utils.random_colors(256)
        colorMap = np.reshape(colors, [256, 1, 3]).astype(np.uint8)

        masked_image = image
        for i in range(N):
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), colors[i], 1)

        masked_image = Utils.applyMasks(masked_image, masks, colorMap)

        return masked_image

    @staticmethod
    def applyMasks(image, masks, colorMap, alpha=.5):
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
    def apply_mask_slow(image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = np.where(mask,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        return image

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
