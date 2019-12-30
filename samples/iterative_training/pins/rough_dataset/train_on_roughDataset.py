import os

from samples.iterative_training.Utils import Utils
from samples.iterative_training.pins.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.arguments import parser
from samples.iterative_training.pins.rough_dataset.RoughAnnotatedPinsConfig import RoughAnnotatedPinsConfig, \
    RoughAnnotatedPinsInferenceConfig
from samples.iterative_training.pins.rough_dataset.node_config import nodeConfig


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
    labels = ['pin', 'pin_w_solder']

    trainingConfig = RoughAnnotatedPinsConfig()
    inferenceConfig = RoughAnnotatedPinsInferenceConfig()

    dataDir = './data'

    trainXmlAnnotations = ['4_8_point_pin_train.xml', '6_8_point_pin_train_2.xml',
                           '7_8_point_pin_train_3.xml', '9_8_point_pin_train_4.xml',
                           '10_8_point_pin_train_5.xml']
    valXmlAnnotations = ['5_8_point_pin_val.xml', '7_8_point_pin_val_3.xml',
                         '9_8_point_pin_val_4.xml', '10_8_point_pin_val_5.xml']

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

    trainingDataset = PinsDataset(labels, [imagesDir, dataDir], trainImageAnnotations)
    validationDataset = PinsDataset(labels, [imagesDir, dataDir], valImageAnnotations)

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
    imagesDir = os.path.join(nodeConfig.workingDir, 'frames_6')

    trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig = \
        prepareTrainerInput(imagesDir)

    trainer = IterativeTrainer(trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                               initialWeights=initialWeights, modelDir=modelDir, visualize=nodeConfig.visualize)

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


def saveVisualizedDetections():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    inferenceConfig = RoughAnnotatedPinsInferenceConfig()
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None)

    saveDir = '/home/trevol/HDD_DATA/TMP/frames/detect_all'
    saveVisualizationToDir = '/home/trevol/HDD_DATA/TMP/frames/detect_all_visualization'
    imagesDirs = ['/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6']
    imageExt = 'jpg'

    trainer.showSavedDetections(saveDir, False, imagesDirs, imageExt, step=1,
                                saveVisualizationToDir=saveVisualizationToDir)


# saveOrShowDetections(save=True, saveStep=1, show=False, showInReverseOrder=False)
# saveVisualizedDetections()

# main_explore_dataset()
main_train()

# export PYTHONPATH=$PYTHONPATH:../../../..
