import itertools

from mrcnn.visualize import random_colors
from samples.iterative_training.Utils import Utils
from samples.iterative_training.pins.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsConfig import PinsConfig, PinsInferenceConfig
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.arguments import parser


def prepareTrainerInput():
    trainingConfig = PinsConfig()
    inferenceConfig = PinsInferenceConfig()

    imagesDir = '/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6'
    annotationFile1 = 'pins/1_TestSegmentation.xml'
    labels, imageAnnotations1 = CvatAnnotation.parse(annotationFile1)
    annotationFile2 = 'pins/2_TestSegmentation_2.xml'
    labels2, imageAnnotations2 = CvatAnnotation.parse(annotationFile2)
    assert labels2 == labels
    # labels.remove('background')
    labels = ['pin']  # пока только пин

    trainAnnotations, valAnnotations, testAnnotations = trainValTestAnnotations(
        [*imageAnnotations1, *imageAnnotations2])
    trainingDataset = PinsDataset(labels, imagesDir, trainAnnotations)
    validationDataset = PinsDataset(labels, imagesDir, valAnnotations)
    testingDataset = PinsDataset(labels, imagesDir, testAnnotations)

    return trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig


def main_train():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig = prepareTrainerInput()
    trainer = IterativeTrainer(trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig)

    # trainer.trainingLoop(parser.parse_args().start == 'vis')
    trainer.visualizePredictability()


def trainValTestAnnotations(imageAnnotations):
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
    testAnnotations = [ann for ann in imageAnnotations if ann.name in testImages]
    return trainAnnotations, valAnnotations, testAnnotations


def main_explore_dataset():
    import cv2
    import numpy as np

    labels = ['pin']  # пока только пин

    trainingDataset, validationDataset, testingDataset, _, _ = prepareTrainerInput()

    labelColors = {l: tuple(map(lambda ch: int(ch * 255), c)) for l, c in zip(labels, random_colors(len(labels)))}

    numOfMasks = 3
    for dataset in [trainingDataset, validationDataset, testingDataset]:
        for imageId in dataset.image_ids:
            image = dataset.load_image(imageId)
            fileName = dataset.image_annotation(imageId).name
            mask, classIds = dataset.load_mask(imageId)

            for i, classId in enumerate(classIds[:numOfMasks]):
                instanceMask = mask[..., i]
                instanceMask = np.multiply(instanceMask, 255, dtype=np.uint8)
                cv2.imshow(f'Instance {i}', instanceMask)

            cv2.imshow('Vis', Utils.rgb2bgr(image))
            cv2.setWindowTitle('Vis', fileName)
            cv2.waitKey()
    while cv2.waitKey() != 27: pass


# main_explore_dataset()
main_train()

# TODO: add weighted for displaying masks
# TODO: inference on CPU???
# TODO: visualize image by image
# TODO: test/visualize on all images from video_6/video_2
