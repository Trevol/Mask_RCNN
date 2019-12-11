from time import time

import cv2
import numpy as np

from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Utils import Utils


class IterativeTrainer():
    def __init__(self, trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig,
                 modelDir='./logs'):
        self.validationDataset = validationDataset
        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset
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

    def getTestingDataset(self):
        dataset = self.testingDataset
        return dataset() if callable(dataset) else dataset

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

        WND_NAME = 'Predictability'
        while True:
            dataset = self.getTestingDataset()
            for i, imageId in enumerate(dataset.image_ids, 1):
                image = dataset.load_image(imageId)
                t0 = time()
                r = inferenceModel.detect([image])[0]
                t1 = time()
                print('detect', t1 - t0)
                boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']

                instancesImage = Utils.display_instances(image, boxes, masks, classIds, scores)

                cv2.imshow(WND_NAME, Utils.rgb2bgr(instancesImage))
                cv2.setWindowTitle(WND_NAME, f'{i}/{len(dataset.image_ids)} Predictability {weights.split("/")[-1]}')

                while True:
                    key = cv2.waitKey()
                    if key == 27:
                        cv2.destroyWindow(WND_NAME)
                        return 'esc'
                    if key == ord('t'):  # require training
                        cv2.destroyWindow(WND_NAME)
                        return 'train'
                    if key in [ord('n'), ord(' '), 13]:  # next visualization on n, space or enter
                        cv2.destroyWindow(WND_NAME)
                        break

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
