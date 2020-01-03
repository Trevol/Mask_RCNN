import cv2
import numpy as np
import skimage.color
import skimage.io

from mrcnn import utils
from samples.iterative_training.Utils import Utils


class CVATDataset(utils.Dataset):
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
    def makeMask(size, polygons, boxes, labels):
        polygons = [p for p in polygons if p.label in labels]
        boxes = [b for b in boxes if b.label in labels]
        assert len(polygons) > 0 or len(boxes) > 0

        polyMasks = [np.zeros(size, np.uint8) for _ in range(len(polygons))]
        for i, p in enumerate(polygons):
            cv2.fillPoly(polyMasks[i], np.array([p.points]), 1)

        boxMasks = [np.zeros(size, np.uint8) for _ in range(len(boxes))]
        for i, b in enumerate(boxes):
            pt1 = b.xtl, b.ytl
            pt2 = b.xbr, b.ybr
            cv2.rectangle(boxMasks[i], pt1, pt2, 1, -1)

        mask = np.dstack(polyMasks + boxMasks).astype(np.bool)
        # Map class names to class IDs.
        class_ids = np.int32([labels.index(p.label) + 1 for p in polygons + boxes])
        return mask, class_ids

    def __init__(self, name, labels, imagesDirs, imageAnnotations):
        super(CVATDataset, self).__init__()
        self.name = name

        assert imagesDirs is None or isinstance(imagesDirs, (str, list))
        if imagesDirs is None:
            imagesDirs = []
        if isinstance(imagesDirs, str):
            imagesDirs = [imagesDirs]

        self.labels = labels
        for i, label in enumerate(labels):
            self.add_class(name, i, label)

        for i, imageAnnotation in enumerate(imageAnnotations):
            imagePath = Utils.findFilePath(imagesDirs, imageAnnotation.name)
            if not imagePath:
                raise Exception(f'Can not find {imageAnnotation.name}')
            image = self.loadImageByPath(imagePath)
            mask = self.makeMask(image.shape[:2], imageAnnotation.polygons, imageAnnotation.boxes, labels)

            self.add_image(name, i, imagePath, annotation=imageAnnotation, image=image, mask=mask)
        self.prepare()

    def load_image(self, image_id):
        return self.image_info[image_id]['image']

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.name:
            return info[self.name]
        else:
            super(self.__class__).image_reference(self, image_id)

    def image_annotation(self, image_id):
        return self.image_info[image_id]['annotation']

    def load_mask(self, image_id):
        return self.image_info[image_id]['mask']
