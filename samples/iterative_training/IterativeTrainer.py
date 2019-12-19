import os
from time import sleep

import cv2
import numpy as np

from samples.iterative_training.ImshowWindow import ImshowWindow
from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Utils import Utils, contexts


class IterativeTrainer():
    def __init__(self, trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                 initialWeights, modelDir='./logs'):
        self.initialWeights = initialWeights
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
            elif self.initialWeights:
                # starts with initial weights
                exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
                self._trainableModel.load_weights(self.initialWeights, by_name=True, exclude=exclude)

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

        with contexts(ImshowWindow('Predictability'),
                      ImshowWindow('Original')) as (predWindow, origWindow):
            while True:
                imageGenerator = self.getTestingGenerator()
                for i, (imageFile, image) in enumerate(imageGenerator):
                    predWindow.setBusy()
                    origWindow.setBusy()
                    r = inferenceModel.detect([image])[0]
                    boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
                    instancesImage = Utils.display_instances(image.copy(), boxes, masks, classIds, scores)

                    predWindow.imshow(instancesImage)
                    predWindow.setTitle(f'{imageFile} Predictability {weightsFile}')
                    origWindow.imshow(image)
                    origWindow.setTitle(f'{imageFile}')

                    while True:
                        key = cv2.waitKey(60000)
                        if key == 27:
                            return 'esc'
                        elif key == ord('s'):
                            self.save(imageFile, image, masks)
                        elif key in [-1, ord('t')]:  # require training by t or timeout
                            return 'train'
                        elif key in [ord('n'), ord(' '), 13]:  # next visualization on n, space or enter
                            break

    @staticmethod
    def save(imageFile, image, masks, verbose=True):
        cv2.imwrite(imageFile, Utils.rgb2bgr(image))
        nameWithoutExt = os.path.splitext(imageFile)[0]
        masksFile = nameWithoutExt + '_masks.npy'
        np.save(masksFile, masks)
        if verbose:
            print('Saved.', imageFile, masksFile)

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


