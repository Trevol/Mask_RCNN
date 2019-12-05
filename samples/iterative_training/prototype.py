import os
import numpy as np
import cv2

from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt
from mrcnn.visualize import random_colors, apply_mask
from samples.iterative_training.ShapesDataset import ShapesDataset


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNEx(MaskRCNN):
    def findLastWeights(self):
        """Finds the last checkpoint file of the last trained model in the
                model directory.
                Returns:
                    The path of the last checkpoint file
                """
        return self.findLastWeightsInModelDir(self.model_dir, self.config.NAME.lower())

    @staticmethod
    def findLastWeightsInModelDir(modelDir, modelName):
        """Finds the last checkpoint file of the last trained model in the
                        model directory.
                        Returns:
                            The path of the last checkpoint file
                        """
        # Get directory names. Each directory corresponds to a model
        for _, dir_names, _ in list(os.walk(modelDir))[:1]:
            dir_names = filter(lambda f: f.startswith(modelName), dir_names)
            for dirName in sorted(dir_names, reverse=True):
                dirName = os.path.join(modelDir, dirName)
                # Find the last checkpoint
                checkpointFiles = next(os.walk(dirName))[2]
                checkpointFiles = filter(lambda f: f.startswith("mask_rcnn") and f.endswith('.h5'), checkpointFiles)
                checkpointFiles = sorted(checkpointFiles)
                if len(checkpointFiles):
                    return os.path.join(dirName, checkpointFiles[-1])
        return None


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

        # If no axis is passed, create one and automatically call show()
        auto_show = False

        # Generate random colors
        colors = random_colors(N)

        # Show area outside image boundaries.
        height, width = image.shape[:2]

        masked_image = image.copy()
        for i in range(N):
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]

            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 1)

            # Label
            # ax.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            # padded_mask = np.zeros(
            #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            # padded_mask[1:-1, 1:-1] = mask
            # contours = find_contours(padded_mask, 0.5)
            # for verts in contours:
            #     # Subtract the padding and flip (y, x) to (x, y)
            #     verts = np.fliplr(verts) - 1
            #     p = Polygon(verts, facecolor="none", edgecolor=color)
            #     ax.add_patch(p)

        return masked_image.astype(np.uint8)

    @staticmethod
    def rgb2bgr(rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class IterativeTrainer():
    modelDir = './logs'
    trainingConfig = ShapesConfig()

    def findLastWeights(self):
        return MaskRCNNEx.findLastWeightsInModelDir(self.modelDir, self.trainingConfig.NAME.lower())

    def makeTrainableModel(self):
        model = MaskRCNNEx(mode='training', config=self.trainingConfig, model_dir=self.modelDir)
        lastWeights = model.findLastWeights()
        if lastWeights:
            model.load_weights(lastWeights)
        else:
            # starts with coco weights
            model.load_weights('mask_rcnn_coco.h5', by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        return model

    def makeInferenceModel(self, weights):
        pass

    def train(self):
        model = self.makeTrainableModel()

        trainDataset = ShapesDataset()
        trainDataset.load_shapes(50, self.trainingConfig.IMAGE_SHAPE[0], self.trainingConfig.IMAGE_SHAPE[1])
        trainDataset.prepare()
        validationDataset = ShapesDataset()
        validationDataset.load_shapes(5, self.trainingConfig.IMAGE_SHAPE[0], self.trainingConfig.IMAGE_SHAPE[1])
        validationDataset.prepare()

        # enter training loop:
        model.train(trainDataset, validationDataset, self.trainingConfig.LEARNING_RATE, epochs=model.epoch + 1,
                    layers='heads')
        model.train(trainDataset, validationDataset, self.trainingConfig.LEARNING_RATE / 10, epochs=model.epoch + 1,
                    layers='all')

        # lastWeights = MaskRCNNEx.findLastWeightsInModelDir('./logs', ShapesConfig.NAME)
        return model.findLastWeights()

    def visualizePredictability(self):

        weights = self.findLastWeights()
        print('Visualizing weights: ', weights)
        inferenceModel = MaskRCNNEx(mode='inference', config=InferenceConfig(), model_dir='./logs')
        inferenceModel.load_weights(weights, by_name=True)

        imageHeight, imageWidth = self.trainingConfig.IMAGE_SHAPE[0:2]
        hSpacer = np.full([imageHeight, 10, 3], 255, np.uint8)

        while True:
            cv2.destroyWindow('Predictability')

            dataset = ShapesDataset()
            dataset.load_shapes(5, imageHeight, imageWidth)
            dataset.prepare()

            rows = []
            for imageId in dataset.image_ids:
                image = dataset.load_image(imageId)
                r = inferenceModel.detect([image])[0]
                boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']

                rowImage = Utils.display_instances(image, boxes, masks, classIds, scores)
                rowImage = np.hstack([image, hSpacer, rowImage])
                vSpacer = np.full([10, rowImage.shape[1], 3], 255, np.uint8)
                rows.extend([rowImage, vSpacer])
            total = np.vstack(rows)
            cv2.imshow('Predictability', Utils.rgb2bgr(total))

            while True:
                key = cv2.waitKey()
                if key == 27:
                    return 'esc'
                if key == ord('t'):  # require training
                    return 'train'
                if key == ord('n'):  # next visualization
                    break

    def trainingLoop(self, startWithVisualize):
        if not startWithVisualize:
            self.train()
        self.visualizePredictability()


def parseArgs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=False, default='', help="'vis' or other")
    return parser.parse_args()


def main():
    args = parseArgs()
    IterativeTrainer().trainingLoop(args.start == 'vis')


main()
