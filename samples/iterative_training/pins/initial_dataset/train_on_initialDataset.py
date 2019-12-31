import itertools

from mrcnn.visualize import random_colors
from samples.iterative_training.Utils import Utils
from samples.iterative_training.cvat.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsConfig import PinsConfig, PinsInferenceConfig
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.arguments import parser


def testImagesGenerator(*paths):
    import glob, os, skimage.io
    for path in paths:
        imagePaths = glob.glob(os.path.join(path, '*.jpg'), recursive=False)
        for imagePath in sorted(imagePaths):
            yield os.path.basename(imagePath), skimage.io.imread(imagePath)


def prepareTrainerInput():
    trainingConfig = PinsConfig()
    inferenceConfig = PinsInferenceConfig()

    imagesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6'
    annotationFile1 = '1_TestSegmentation.xml'
    labels, imageAnnotations1 = CvatAnnotation.parse(annotationFile1)
    annotationFile2 = '2_TestSegmentation_2.xml'
    labels2, imageAnnotations2 = CvatAnnotation.parse(annotationFile2)
    assert labels2 == labels
    # labels.remove('background')
    labels = ['pin']  # пока только пин

    trainAnnotations, valAnnotations = trainValAnnotations([*imageAnnotations1, *imageAnnotations2])
    trainingDataset = PinsDataset(labels, imagesDir, trainAnnotations)
    validationDataset = PinsDataset(labels, imagesDir, valAnnotations)

    return trainingDataset, validationDataset, testImagesGenerator(imagesDir), trainingConfig, inferenceConfig


def trainValAnnotations(imageAnnotations):
    """
    filename         count(pin)  count(pin_w_solder)
    f_0230_15333.33_15.33.jpg 6 0
    f_0350_23333.33_23.33.jpg 10 0
    f_0669_44600.00_44.60.jpg 20 0
    f_3975_265000.00_265.00.jpg 88 0

    f_3439_229266.67_229.27.jpg 74 0

    f_4446_296400.00_296.40.jpg 85 1
    f_4765_317666.67_317.67.jpg 80 3
    f_7618_507866.67_507.87.jpg 24 60
    f_8499_566600.00_566.60.jpg 6 79
    """

    trainImages = ['f_0230_15333.33_15.33.jpg',
                   'f_0350_23333.33_23.33.jpg',
                   'f_0669_44600.00_44.60.jpg',
                   'f_3975_265000.00_265.00.jpg']
    valImages = ['f_3439_229266.67_229.27.jpg']
    testImages = ['f_4446_296400.00_296.40.jpg',
                  'f_4765_317666.67_317.67.jpg',
                  'f_7618_507866.67_507.87.jpg',
                  'f_8499_566600.00_566.60.jpg']
    trainAnnotations = [ann for ann in imageAnnotations if ann.name in trainImages]
    valAnnotations = [ann for ann in imageAnnotations if ann.name in valImages]
    return trainAnnotations, valAnnotations


def main_train():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig = prepareTrainerInput()
    trainer = IterativeTrainer(trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                               initialWeights='../../mask_rcnn_coco.h5')

    trainer.trainingLoop(parser.parse_args().start == 'vis')
    # trainer.visualizePredictability()


def main_explore_dataset():
    trainingDataset, validationDataset, testingGenerator, _, _ = prepareTrainerInput()
    Utils.exploreDatasets(trainingDataset, validationDataset)


main_explore_dataset()
# main_train()
