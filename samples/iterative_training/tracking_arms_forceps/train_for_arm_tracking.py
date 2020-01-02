import os

from samples.iterative_training.Utils import Utils
from samples.iterative_training.cvat.CVATDataset import CVATDataset
from samples.iterative_training.cvat.CvatAnnotation import CvatAnnotation
from samples.iterative_training.tracking_arms_forceps.TrackingArmsForcepsConfig import TrackingArmsForcepsConfig, \
    TrackingArmsForcepsInferenceConfig
from samples.iterative_training.tracking_arms_forceps.node_config import nodeConfig


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


def prepareTrainerInput(imagesDir):
    labels = ['arm', 'forceps', 'forceps+solder', 'pin-array']

    trainingConfig = TrackingArmsForcepsConfig()
    inferenceConfig = TrackingArmsForcepsInferenceConfig()

    dataDir = './data'

    trainXmlAnnotations = ['11_arm_forceps_solder_pin-array.xml']
    valXmlAnnotations = ['11_arm_forceps_solder_pin-array.xml']

    pjn = os.path.join
    trainLabelsAndImageAnnotations = [CvatAnnotation.parse(pjn(dataDir, x)) for x in trainXmlAnnotations]
    valLabelsAndImageAnnotations = [CvatAnnotation.parse(pjn(dataDir, x)) for x in valXmlAnnotations]

    trainImageAnnotations = []
    for annotLabels, imageAnnotations in trainLabelsAndImageAnnotations:
        assert annotLabels == labels
        trainImageAnnotations.extend(imageAnnotations)

    valImageAnnotations = []
    for annotLabels, imageAnnotations in valLabelsAndImageAnnotations:
        assert annotLabels == labels
        valImageAnnotations.extend(imageAnnotations)

    trainingDataset = CVATDataset('TrackingArmsForceps', labels, [imagesDir, dataDir], trainImageAnnotations)
    validationDataset = CVATDataset('TrackingArmsForceps', labels, [imagesDir, dataDir], valImageAnnotations)

    return trainingDataset, validationDataset, imagesGenerator(True, 10, [imagesDir],
                                                               'jpg'), trainingConfig, inferenceConfig


def main_explore_dataset():
    imagesDir = os.path.join(nodeConfig.workingDir, 'frames_6')
    trainingDataset, validationDataset, testingGenerator, _, _ = prepareTrainerInput(imagesDir)
    Utils.exploreDatasets(trainingDataset, validationDataset)


main_explore_dataset()

# export PYTHONPATH=$PYTHONPATH:../../../..
