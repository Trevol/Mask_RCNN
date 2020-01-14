import json
import os
import warnings

import cv2
import numpy as np
import skimage.io

from samples.iterative_training.ImshowWindow import ImshowWindow
from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Utils import Utils, contexts


class IterativeTrainer():
    def __init__(self, trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                 initialWeights, modelDir, visualize, classBGR, augmentation, checkpointFileName):
        self.augmentation = augmentation
        self.classBGR = classBGR
        self.visualize = visualize
        self.initialWeights = initialWeights
        self.validationDataset = validationDataset
        self.trainingDataset = trainingDataset
        self.testingGenerator = testingGenerator
        self.inferenceConfig = inferenceConfig
        self.trainingConfig = trainingConfig
        self.modelDir = modelDir
        self._trainableModel = None
        self._inferenceModel = None
        self.checkpointFileName = checkpointFileName

    def getTrainingDataset(self):
        dataset = self.trainingDataset
        return dataset() if callable(dataset) else dataset

    def getValidationDataset(self):
        dataset = self.validationDataset
        return dataset() if callable(dataset) else dataset

    def getTestingGenerator(self, outerGenerator):
        generator = outerGenerator or self.testingGenerator
        return generator() if callable(generator) else generator

    def findLastWeights(self):
        modelName = (self.trainingConfig or self.inferenceConfig).NAME.lower()
        return MaskRCNNEx.findLastWeightsInModelDir(self.modelDir, modelName)

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

    def getInferenceModel(self, loadLastWeights):
        if not self._inferenceModel:
            self._inferenceModel = MaskRCNNEx(mode='inference', config=self.inferenceConfig, model_dir=self.modelDir)
        lastWeights = None
        if loadLastWeights:
            lastWeights = self.findLastWeights()
            if lastWeights:
                self._inferenceModel.load_weights(lastWeights, by_name=True)
        return self._inferenceModel, lastWeights

    def train(self, lr):
        trainableModel = self.getTrainableModel(loadWeights=True)
        trainingDataset = self.getTrainingDataset()
        validationDataset = self.getValidationDataset()

        lr = lr or self.trainingConfig.LEARNING_RATE
        print('Training stage 1: HEADS.')

        trainableModel.train(trainingDataset, validationDataset, lr,
                             epochs=trainableModel.epoch + 1, layers='heads', augmentation=self.augmentation)

        print('Training stage 2: Finetune layers from ResNet stage 4 and up.')
        trainableModel.train(trainingDataset, validationDataset, lr,
                             epochs=trainableModel.epoch + 1, layers='4+',
                             augmentation=self.augmentation)
        print('Training stage 3: Finetune all layers')
        history = trainableModel.train(trainingDataset, validationDataset, lr / 10,
                                       epochs=trainableModel.epoch + 1, layers='all', augmentation=self.augmentation)

        self.saveCheckpoint(trainableModel, history)

    def saveCheckpoint(self, model, history):
        allEpochsHistory = history.history
        lastEpochHistory = {k: v[-1] for k, v in allEpochsHistory.items()}

        name = model.config.NAME.lower()
        epoch = model.epoch
        checkpointFile = self.checkpointFileName.format(name=name, epoch=epoch, **lastEpochHistory)
        checkpoint_path = os.path.join(model.log_dir, checkpointFile + '.h5')
        model.keras_model.save_weights(checkpoint_path, overwrite=True)

        jsonHistoryPath = os.path.join(model.log_dir, checkpointFile + '.history.json')
        with open(jsonHistoryPath, "wt") as f:
            json.dump(dict(lastEpoch=lastEpochHistory, all=allEpochsHistory), f, indent=2)

    def visualizePredictability(self, imageGenerator=None):
        if not self.visualize:
            # TODO: save some predictions to file system
            return 'train'
        inferenceModel, weights = self.getInferenceModel(loadLastWeights=True)
        print('Using weights: ', weights)
        weightsFile = os.path.basename(weights)

        with contexts(ImshowWindow('Predictability'),
                      ImshowWindow('Original')) as (predWindow, origWindow):
            while True:
                imageGenerator = self.getTestingGenerator(imageGenerator)
                for i, (imageFile, image) in enumerate(imageGenerator):
                    predWindow.setBusy()
                    origWindow.setBusy()
                    r = inferenceModel.detect_minimasks([image])[0]
                    boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
                    instancesImage = Utils.display_instances(image.copy(), boxes, masks, classIds, scores,
                                                             'classes', self.classBGR)
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

    def trainingLoop(self, startWithVisualization, lr=None):
        if startWithVisualization:
            interactionResult = self.visualizePredictability()
            if interactionResult == 'esc':
                return
        while True:
            self.train(lr)
            interactionResult = self.visualizePredictability()
            if interactionResult == 'esc':
                break

    #############
    def saveDetections(self, imagesGenerator, saveDir):
        model, weights = self.getInferenceModel(loadLastWeights=True)
        print('Using weights ', weights)

        os.makedirs(saveDir, exist_ok=True)
        for i, (imageFile, image) in enumerate(imagesGenerator):
            r = model.detect_minimasks([image])[0]
            nameWithoutExt = os.path.splitext(imageFile)[0]
            outFile = os.path.join(saveDir, nameWithoutExt + '.pickle')
            Utils.savePickle(r, outFile)
            if i > 0 and i % 50 == 0:
                print(f'{i} images processed')

    def saveDetectionsV2(self, imagesGenerator, batchSize, pickleDir, imagesDir, withBoxes: bool, onlyMasks: bool,
                         imageQuality=75):
        assert pickleDir or imagesDir
        assert imagesDir and (withBoxes or onlyMasks)

        model, weights = self.getInferenceModel(loadLastWeights=True)
        print('Using weights ', weights)

        if pickleDir:
            os.makedirs(pickleDir, exist_ok=True)

        withBoxesDir, onlyMasksDir = None, None
        if imagesDir:
            if withBoxes:
                withBoxesDir = os.path.join(imagesDir, 'withBoxes')
                os.makedirs(withBoxesDir, exist_ok=True)
            if onlyMasks:
                onlyMasksDir = os.path.join(imagesDir, 'onlyMasks')
                os.makedirs(onlyMasksDir, exist_ok=True)

        def saveStuff(r, imageFile, image):
            if pickleDir:
                nameWithoutExt = os.path.splitext(imageFile)[0]
                pickleFile = os.path.join(pickleDir, nameWithoutExt + '.pickle')
                Utils.savePickle(r, pickleFile)

            if imagesDir:
                boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
                if withBoxes:
                    imageWithBoxes = Utils.display_instances(image.copy(), boxes, masks, classIds, scores,
                                                             'classes', self.classBGR, True, True, True)
                    skimage.io.imsave(os.path.join(withBoxesDir, imageFile), imageWithBoxes, quality=imageQuality)
                if onlyMasks:
                    imageOnlyMasks = Utils.display_instances(image.copy(), boxes, masks, classIds, scores,
                                                             'classes', self.classBGR, False, True, False)
                    skimage.io.imsave(os.path.join(onlyMasksDir, imageFile), imageOnlyMasks, quality=imageQuality)

        for i, batch in enumerate(Utils.batchFlow(imagesGenerator, batchSize)):
            imBatch = list(map(lambda b: b[1], batch))
            resultsBatch = model.detect_minimasks(imBatch)
            for r, (imageFile, image) in zip(resultsBatch, batch):
                saveStuff(r, imageFile, image)
            if i > 0 and i % 100 == 0:
                print(f'{i} batches processed')

    def showSavedDetections(self, saveDir, inReverseOrder, imagesDirs, imageExt, step, saveVisualizationToDir=None):
        def genDetectionsAndImage():
            import glob
            picklePaths = glob.glob(os.path.join(saveDir, '*.pickle'), recursive=False)
            for picklePath in sorted(picklePaths, reverse=inReverseOrder)[::step]:
                r = Utils.loadPickle(picklePath)
                boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
                imageFileName = os.path.splitext(os.path.basename(picklePath))[0] + '.' + imageExt
                imagePath = Utils.findFilePath(imagesDirs, imageFileName)
                image = cv2.imread(imagePath)
                yield boxes, masks, classIds, scores, image, imageFileName

        def showDetections():
            with contexts(ImshowWindow('Predictability'), ImshowWindow('Original')) as (instancesWindow, origWindow):
                for boxes, masks, classIds, scores, image, imageFileName in genDetectionsAndImage():
                    instancesImage = Utils.display_instances(image.copy(), boxes, masks, classIds, scores,
                                                             'classes', self.classBGR)
                    instancesWindow.imshow(instancesImage, imgInRgb=False)
                    instancesWindow.setTitle(f'{imageFileName}')
                    origWindow.imshow(image, imgInRgb=False)
                    origWindow.setTitle(f'{imageFileName}')

                    key = cv2.waitKey(10000)
                    if key == 27:
                        return 'esc'

        def saveVisualizations():
            assert saveVisualizationToDir
            JP = os.path.join
            withBoxesDir = JP(saveVisualizationToDir, 'withBoxes')
            onlyMasksDir = JP(saveVisualizationToDir, 'onlyMasks')
            os.makedirs(withBoxesDir, exist_ok=True)
            os.makedirs(onlyMasksDir, exist_ok=True)
            for i, (boxes, masks, classIds, scores, image, imageFileName) in enumerate(genDetectionsAndImage()):
                imageWithBoxes = Utils.display_instances(image.copy(), boxes, masks, classIds, scores,
                                                         'classes', self.classBGR, True, True, True)
                imageOnlyMasks = Utils.display_instances(image.copy(), boxes, masks, classIds, scores,
                                                         'classes', self.classBGR, False, True, False)
                cv2.imwrite(JP(withBoxesDir, imageFileName), imageWithBoxes)
                cv2.imwrite(JP(onlyMasksDir, imageFileName), imageOnlyMasks)
                if i > 0 and i % 50 == 0:
                    print(f'{i} images processed')

        if saveVisualizationToDir:
            saveVisualizations()
        else:
            showDetections()
