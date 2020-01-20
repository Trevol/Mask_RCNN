import os
from mrcnn.model import data_generator
from samples.iterative_training.cvat.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.pins.rough_dataset.RoughAnnotatedPinsConfig import RoughAnnotatedPinsConfig


def getTrainDataset():
    trainXmlAnnotations = ['4_8_point_pin_train.xml', '6_8_point_pin_train_2.xml',
                           '7_8_point_pin_train_3.xml', '9_8_point_pin_train_4.xml',
                           '10_8_point_pin_train_5.xml']
    imagesDir = '/hdd/Computer_Vision_Task/frames_6'
    dataDir = './data'
    pjn = os.path.join
    trainLabelsAndImageAnnotations = [CvatAnnotation.parse(pjn(dataDir, x)) for x in trainXmlAnnotations]
    trainImageAnnotations = []
    for annotLabels, imageAnnotations in trainLabelsAndImageAnnotations:
        trainImageAnnotations.extend(imageAnnotations)
    labels = ['pin', 'pin_w_solder']
    trainingDataset = PinsDataset(labels, [imagesDir, dataDir], trainImageAnnotations)
    return trainingDataset


def main():
    trainDataset = getTrainDataset()
    config = RoughAnnotatedPinsConfig()
    augmentation = None
    no_augmentation_sources = None
    trainGenerator = data_generator(trainDataset, config, shuffle=True,
                                     augmentation=augmentation,
                                     batch_size=config.BATCH_SIZE,
                                     no_augmentation_sources=no_augmentation_sources)
    for i in range(1):
        r = next(trainGenerator)
        print(r)
main()
