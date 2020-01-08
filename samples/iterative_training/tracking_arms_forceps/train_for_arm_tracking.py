import os
import warnings

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from samples.iterative_training.Utils import Utils
from samples.iterative_training.arguments import parser
from samples.iterative_training.cvat.CVATDataset import CVATDataset
from samples.iterative_training.cvat.CvatAnnotation import CvatAnnotation
from samples.iterative_training.tracking_arms_forceps.TrackingArmsForcepsConfig import TrackingArmsForcepsConfig, \
    TrackingArmsForcepsInferenceConfig
from samples.iterative_training.tracking_arms_forceps.node_config import nodeConfig
import imgaug as ia
import imgaug.augmenters as iaa


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
    # labels = ['arm', 'forceps', 'forceps+solder', 'pin-array']
    labels = ['forceps', 'forceps+solder']

    trainingConfig = TrackingArmsForcepsConfig()
    inferenceConfig = TrackingArmsForcepsInferenceConfig()

    dataDir = './data'

    trainXmlAnnotations = ['13_arm_forceps_solder_pin-array.xml']
    valXmlAnnotations = ['13_arm_forceps_solder_pin-array.xml']

    pjn = os.path.join
    trainLabelsAndImageAnnotations = [CvatAnnotation.parse(pjn(dataDir, x)) for x in trainXmlAnnotations]
    valLabelsAndImageAnnotations = [CvatAnnotation.parse(pjn(dataDir, x)) for x in valXmlAnnotations]

    trainImageAnnotations = []
    for annotLabels, imageAnnotations in trainLabelsAndImageAnnotations:
        # assert annotLabels == labels
        assert set(labels).issubset(set(annotLabels))
        trainImageAnnotations.extend(imageAnnotations)

    valImageAnnotations = []
    for annotLabels, imageAnnotations in valLabelsAndImageAnnotations:
        # assert annotLabels == labels
        assert set(labels).issubset(set(annotLabels))
        valImageAnnotations.extend(imageAnnotations)

    trainingDataset = CVATDataset('TrackingArmsForceps', labels, [imagesDir, dataDir], trainImageAnnotations)
    validationDataset = CVATDataset('TrackingArmsForceps', labels, [imagesDir, dataDir], valImageAnnotations)

    imGen = Utils.imageFlow(paths=[imagesDir], ext='jpg', start=None, stop=None, step=-10)
    return trainingDataset, validationDataset, imGen, trainingConfig, inferenceConfig


def main_train():
    if not nodeConfig:
        print('Node is not configured for training. Stopping...')
        return
    from samples.iterative_training.IterativeTrainer import IterativeTrainer

    initialWeights = nodeConfig.initialWeights
    # initialWeights = None

    os.makedirs(nodeConfig.workingDir, exist_ok=True)

    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    imagesDir = nodeConfig.framesDir
    TrackingArmsForcepsConfig.IMAGES_PER_GPU = nodeConfig.IMAGES_PER_GPU

    trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig = \
        prepareTrainerInput(imagesDir)

    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 1.5)),  # blur images with a sigma of 0 to 3.0
        iaa.Sharpen((0.0, 1.0)),
        iaa.Affine(rotate=(-10, 10)),
        iaa.Affine(shear=(-10, 10)),
        iaa.Affine(scale=(1, 1.1))
    ])

    trainer = IterativeTrainer(trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                               initialWeights=initialWeights, modelDir=modelDir, visualize=nodeConfig.visualize,
                               classBGR=None, augmentation=seq)

    args = parser.parse_args()
    startWithVisualization = args.start == 'vis'
    lr = args.lr
    trainer.trainingLoop(startWithVisualization, lr)
    # trainer.visualizePredictability()


def main_explore_dataset():
    imagesDir = nodeConfig.framesDir
    trainingDataset, validationDataset, testingGenerator, _, _ = prepareTrainerInput(imagesDir)
    Utils.exploreDatasets(trainingDataset, validationDataset)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ia.seed(1)
        # main_explore_dataset()
        main_train()

# export PYTHONPATH=$PYTHONPATH:../../../..
