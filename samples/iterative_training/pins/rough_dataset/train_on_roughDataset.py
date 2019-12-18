import itertools

from mrcnn.visualize import random_colors
from samples.iterative_training.Utils import Utils
from samples.iterative_training.pins.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsConfig import PinsConfig, PinsInferenceConfig
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.arguments import parser
from samples.iterative_training.pins.rough_dataset.RoughAnnotatedPinsConfig import RoughAnnotatedPinsConfig, \
    RoughAnnotatedPinsInferenceConfig


def imagesGenerator(*paths):
    import glob, os, skimage.io
    for path in paths:
        imagePaths = glob.glob(os.path.join(path, '*.jpg'), recursive=False)
        for imagePath in sorted(imagePaths, reverse=True):
            yield os.path.basename(imagePath), skimage.io.imread(imagePath)


def prepareTrainerInput():
    labels = ['pin', 'pin_w_solder']

    trainingConfig = RoughAnnotatedPinsConfig()
    inferenceConfig = RoughAnnotatedPinsInferenceConfig()

    imagesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6'
    trainAnnotationFile = '4_8_point_pin_train.xml'
    trainLabels, trainImageAnnotations = CvatAnnotation.parse(trainAnnotationFile)
    valAnnotationFile = '5_8_point_pin_val.xml'
    valLabels, valImageAnnotations = CvatAnnotation.parse(valAnnotationFile)
    assert trainLabels == valLabels
    assert trainLabels == labels

    trainingDataset = PinsDataset(labels, imagesDir, trainImageAnnotations)
    validationDataset = PinsDataset(labels, imagesDir, valImageAnnotations)

    return trainingDataset, validationDataset, imagesGenerator(imagesDir), trainingConfig, inferenceConfig


def main_train():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig = prepareTrainerInput()
    initialWeights = '../../mask_rcnn_coco.h5'
    # initialWeights = None
    trainer = IterativeTrainer(trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                               initialWeights=initialWeights)

    trainer.trainingLoop(parser.parse_args().start == 'vis')
    # trainer.visualizePredictability()


def main_explore_dataset():
    trainingDataset, validationDataset, testingGenerator, _, _ = prepareTrainerInput()
    Utils.exploreDatasets(trainingDataset, validationDataset)


# main_explore_dataset()
main_train()

# export PYTHONPATH=$PYTHONPATH:../../../..
