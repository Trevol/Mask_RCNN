import os

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


def main_train():
    if not nodeConfig:
        print('Node is not configured for training. Stopping...')
        return
    from samples.iterative_training.IterativeTrainer import IterativeTrainer

    initialWeights = os.path.join(nodeConfig.workingDir, 'mask_rcnn_coco.h5')
    # initialWeights = None
    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    imagesDir = nodeConfig.framesDir
    TrackingArmsForcepsConfig.IMAGES_PER_GPU = nodeConfig.IMAGES_PER_GPU
    # TrackingArmsForcepsConfig.LEARNING_RATE = 0.0001

    trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig = \
        prepareTrainerInput(imagesDir)
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
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
    imagesDir = os.path.join(nodeConfig.workingDir, 'frames_6')
    trainingDataset, validationDataset, testingGenerator, _, _ = prepareTrainerInput(imagesDir)
    Utils.exploreDatasets(trainingDataset, validationDataset)


if __name__ == '__main__':
    ia.seed(1)
    # main_explore_dataset()
    main_train()

# export PYTHONPATH=$PYTHONPATH:../../../..
