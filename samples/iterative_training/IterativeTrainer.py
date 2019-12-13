import os

import cv2
import numpy as np

from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Timer import timeit
from samples.iterative_training.Utils import Utils


class IterativeTrainer():
    def __init__(self, trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                 modelDir='./logs'):
        self.validationDataset = validationDataset
        self.trainingDataset = trainingDataset
        self.testingGenerator = testingGenerator
        self.inferenceConfig = inferenceConfig
        self.trainingConfig = trainingConfig
        self.modelDir = modelDir
        self._trainableModel = None
        self._inferenceModel = None

    def getTrainingDataset(self):
        dataset = self.trainingDataset
        return dataset() if callable(dataset) else dataset

    def getValidationDataset(self):
        dataset = self.validationDataset
        return dataset() if callable(dataset) else dataset

    def getTestingGenerator(self):
        generator = self.testingGenerator
        return generator() if callable(generator) else generator

    def findLastWeights(self):
        return MaskRCNNEx.findLastWeightsInModelDir(self.modelDir, self.trainingConfig.NAME.lower())

    def getTrainableModel(self, loadWeights):
        if not self._trainableModel:
            self._trainableModel = MaskRCNNEx(mode='training', config=self.trainingConfig, model_dir=self.modelDir)
        if loadWeights:
            lastWeights = self.findLastWeights()
            if lastWeights:
                self._trainableModel.load_weights(lastWeights)
            else:
                # starts with coco weights
                exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
                self._trainableModel.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=exclude)

        return self._trainableModel

    def getInferenceModel(self):
        if not self._inferenceModel:
            self._inferenceModel = MaskRCNNEx(mode='inference', config=self.inferenceConfig, model_dir=self.modelDir)
        return self._inferenceModel

    def train(self):
        trainableModel = self.getTrainableModel(loadWeights=True)
        trainingDataset = self.getTrainingDataset()
        validationDataset = self.getValidationDataset()
        trainableModel.train(trainingDataset, validationDataset, self.trainingConfig.LEARNING_RATE,
                             epochs=trainableModel.epoch + 1, layers='heads')
        trainableModel.train(trainingDataset, validationDataset, self.trainingConfig.LEARNING_RATE / 10,
                             epochs=trainableModel.epoch + 1, layers='all')

    def visualizePredictability(self):
        weights = self.findLastWeights()
        print('Visualizing weights: ', weights)
        inferenceModel = self.getInferenceModel()
        inferenceModel.load_weights(weights, by_name=True)
        weightsFile = weights.split("/")[-1]

        WND_NAME = 'Predictability'
        while True:
            imageGenerator = self.getTestingGenerator()
            for i, (imageFile, image) in enumerate(imageGenerator):
                print('--------------------------')
                with timeit('detect'):
                    # r = inferenceModel.detect([image])[0]
                    r = self.detect_DEBUG(inferenceModel, [image], verbose=0)[0]

                boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']

                with timeit('display_instances(apply masks)'):
                    instancesImage = Utils.display_instances(image.copy(), boxes, masks, classIds, scores)

                cv2.imshow(WND_NAME, Utils.rgb2bgr(instancesImage))

                cv2.setWindowTitle(WND_NAME, f'{imageFile} Predictability {weightsFile}')

                while True:
                    key = cv2.waitKey()
                    if key == 27:
                        cv2.destroyWindow(WND_NAME)
                        return 'esc'
                    elif key == ord('s'):
                        # save image and masks
                        self.save(imageFile, image, masks)
                    elif key == ord('t'):  # require training
                        cv2.destroyWindow(WND_NAME)
                        return 'train'
                    elif key in [ord('n'), ord(' '), 13]:  # next visualization on n, space or enter
                        cv2.destroyWindow(WND_NAME)
                        break

    @staticmethod
    def save(imageFile, image, masks, verbose=True):
        cv2.imwrite(imageFile, Utils.rgb2bgr(image))
        nameWithoutExt = os.path.splitext(imageFile)[0]
        masksFile = nameWithoutExt + '_masks.npy'
        np.save(masksFile, masks)
        if verbose:
            print('Saved.', imageFile, masksFile)


    @staticmethod
    def detect_DEBUG(model, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert model.mode == "inference", "Create model in inference mode."
        assert len(
            images) == model.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # if verbose:
        #     log("Processing {} images".format(len(images)))
        #     for image in images:
        #         log("image", image)

        # Mold inputs to format expected by the neural network
        with timeit('mold_detections'):
            molded_images, image_metas, windows = model.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        with timeit('get_anchors and broadcast_to'):
            anchors = model.get_anchors(image_shape)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        # if verbose:
        #     log("molded_images", molded_images)
        #     log("image_metas", image_metas)
        #     log("anchors", anchors)
        # Run object detection

        with timeit('model.keras_model.predict'):
            detections, _, _, mrcnn_mask, _, _, _ = \
                model.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        # Process detections
        results = []

        for i, image in enumerate(images):
            with timeit('unmold_detections'):
                final_rois, final_class_ids, final_scores, final_masks = \
                    model.unmold_detections(detections[i], mrcnn_mask[i],
                                            image.shape, molded_images[i].shape,
                                            windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def trainingLoop(self, startWithVisualize):
        if startWithVisualize:
            interactionResult = self.visualizePredictability()
            if interactionResult == 'esc':
                return
        while True:
            self.train()
            interactionResult = self.visualizePredictability()
            if interactionResult == 'esc':
                break
