from mrcnn import utils
import numpy as np
import cv2
import math
import random
import os
import skimage.io
import skimage.color


class PinsDataset(utils.Dataset):
    @staticmethod
    def loadImageByPath(imagePath):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(imagePath)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    @staticmethod
    def makeMask(size, polygons, labels):
        polygons = [p for p in polygons if p.label in labels]
        assert len(polygons) > 0
        masks = [np.zeros(size, np.uint8) for _ in range(len(polygons))]
        for i, p in enumerate(polygons):
            cv2.fillPoly(masks[i], np.array([p.points]), 1)
            # cv2.polylines(masks[i], np.array([p.points]), True, 1)
        mask = np.dstack(masks).astype(np.bool)
        # Map class names to class IDs.
        class_ids = np.array([labels.index(p.label) + 1 for p in polygons])
        return mask, class_ids.astype(np.int32)

    def __init__(self, labels, imagesDir, imageAnnotations):
        super(PinsDataset, self).__init__()
        self.labels = labels
        for i, label in enumerate(labels):
            self.add_class("pins", i, label)

        for i, imageAnnotation in enumerate(imageAnnotations):
            imagePath = os.path.join(imagesDir, imageAnnotation.name)
            image = self.loadImageByPath(imagePath)
            mask = self.makeMask(image.shape[:2], imageAnnotation.polygons, labels)

            self.add_image('pins', i, imagePath, annotation=imageAnnotation, image=image, mask=mask)
        self.prepare()

    def load_image(self, image_id):
        return self.image_info[image_id]['image']

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pins":
            return info["pins"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def image_annotation(self, image_id):
        return self.image_info[image_id]['annotation']

    def load_mask(self, image_id):
        return self.image_info[image_id]['mask']
