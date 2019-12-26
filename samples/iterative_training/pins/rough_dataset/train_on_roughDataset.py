import os

from samples.iterative_training.ImshowWindow import ImshowWindow
from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Utils import Utils, contexts
from samples.iterative_training.pins.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.arguments import parser
from samples.iterative_training.pins.rough_dataset.RoughAnnotatedPinsConfig import RoughAnnotatedPinsConfig, \
    RoughAnnotatedPinsInferenceConfig


def imagesGenerator(reverse, step, paths, ext):
    assert isinstance(paths, (list, str))
    if isinstance(paths, str):
        paths = [paths]
    assert len(paths)

    import glob, skimage.io
    for path in paths:
        imagePaths = glob.glob(os.path.join(path, f'*.{ext}'), recursive=False)
        for imagePath in sorted(imagePaths, reverse=reverse)[::step]:
            yield os.path.basename(imagePath), skimage.io.imread(imagePath)


def prepareTrainerInput():
    labels = ['pin', 'pin_w_solder']

    trainingConfig = RoughAnnotatedPinsConfig()
    inferenceConfig = RoughAnnotatedPinsInferenceConfig()

    imagesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6'
    dataDir = './data'

    trainAnnotationFile = os.path.join(dataDir, '4_8_point_pin_train.xml')
    trainLabels, trainImageAnnotations = CvatAnnotation.parse(trainAnnotationFile)

    trainAnnotationFile2 = os.path.join(dataDir, '6_8_point_pin_train_2.xml')
    trainLabels2, trainImageAnnotations2 = CvatAnnotation.parse(trainAnnotationFile2)

    trainAnnotationFile3 = os.path.join(dataDir, '7_8_point_pin_train_3.xml')
    trainLabels3, trainImageAnnotations3 = CvatAnnotation.parse(trainAnnotationFile3)

    valAnnotationFile = os.path.join(dataDir, '5_8_point_pin_val.xml')
    valLabels, valImageAnnotations = CvatAnnotation.parse(valAnnotationFile)

    valAnnotationFile2 = os.path.join(dataDir, '7_8_point_pin_val_3.xml')
    valLabels2, valImageAnnotations2 = CvatAnnotation.parse(valAnnotationFile2)

    assert trainLabels == labels
    assert trainLabels2 == labels
    assert trainLabels3 == labels
    assert valLabels == labels
    assert valLabels2 == labels

    trainingDataset = PinsDataset(labels, [imagesDir, dataDir],
                                  trainImageAnnotations + trainImageAnnotations2 + trainImageAnnotations3)
    validationDataset = PinsDataset(labels, [imagesDir, dataDir], valImageAnnotations + valImageAnnotations2)

    return trainingDataset, validationDataset, imagesGenerator(True, 10, [imagesDir],
                                                               'jpg'), trainingConfig, inferenceConfig


def main_train():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig = prepareTrainerInput()
    initialWeights = '../../mask_rcnn_coco.h5'
    # initialWeights = None
    trainer = IterativeTrainer(trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                               initialWeights=initialWeights)

    startWithVisualization = parser.parse_args().start == 'vis'
    trainer.trainingLoop(startWithVisualization)
    # trainer.visualizePredictability()


def main_explore_dataset():
    trainingDataset, validationDataset, testingGenerator, _, _ = prepareTrainerInput()
    Utils.exploreDatasets(trainingDataset, validationDataset)


def saveOrShowDetections(save, saveStep, show, showInReverseOrder):
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    inferenceConfig = RoughAnnotatedPinsInferenceConfig()
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None)

    saveDir = '/home/trevol/HDD_DATA/TMP/frames/detect_all'
    imagesDirs = ['/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6']
    imageExt = 'jpg'

    if save:
        imagesGen = imagesGenerator(False, saveStep, imagesDirs, imageExt)
        trainer.saveDetections(imagesGen, saveDir)
    if show:
        trainer.showSavedDetections(saveDir, showInReverseOrder, imagesDirs, imageExt, 1)


# main_explore_dataset()
# saveOrShowDetections(save=False, saveStep=1, show=True, showInReverseOrder=False)
main_train()

# export PYTHONPATH=$PYTHONPATH:../../../..
