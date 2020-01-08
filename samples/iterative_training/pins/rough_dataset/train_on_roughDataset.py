import os

from samples.iterative_training.Utils import Utils
from samples.iterative_training.cvat.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.arguments import parser
from samples.iterative_training.pins.rough_dataset.RoughAnnotatedPinsConfig import RoughAnnotatedPinsConfig, \
    RoughAnnotatedPinsInferenceConfig
from samples.iterative_training.pins.rough_dataset.node_config import nodeConfig


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

    imGen = Utils.imageFlow(paths=[imagesDir], ext='jpg', start=None, stop=None, step=-10)
    return trainingDataset, validationDataset, imGen, trainingConfig, inferenceConfig


classBGR = [
    None,  # 0
    (0, 255, 255),  # 1 - pin RGB  BGR
    (0, 255, 0),  # 2 - solder

]


def main_train():
    if not nodeConfig:
        print('Node is not configured for training. Stopping...')
        return
    from samples.iterative_training.IterativeTrainer import IterativeTrainer

    initialWeights = os.path.join(nodeConfig.workingDir, 'mask_rcnn_coco.h5')
    # initialWeights = None
    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    imagesDir = nodeConfig.framesDir
    RoughAnnotatedPinsConfig.IMAGES_PER_GPU = nodeConfig.IMAGES_PER_GPU
    RoughAnnotatedPinsConfig.LEARNING_RATE = 0.0001

    trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig = \
        prepareTrainerInput(imagesDir)

    trainer = IterativeTrainer(trainingDataset, validationDataset, testingGenerator, trainingConfig, inferenceConfig,
                               initialWeights=initialWeights, modelDir=modelDir, visualize=nodeConfig.visualize,
                               classBGR=classBGR)

    startWithVisualization = parser.parse_args().start == 'vis'
    trainer.trainingLoop(startWithVisualization)
    # trainer.visualizePredictability()


def main_explore_dataset():
    imagesDir = os.path.join(nodeConfig.workingDir, 'frames_6')
    trainingDataset, validationDataset, testingGenerator, _, _ = prepareTrainerInput(imagesDir)
    Utils.exploreDatasets(trainingDataset, validationDataset)


def saveOrShowDetections(save, saveStep, show, showInReverseOrder):
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    inferenceConfig = RoughAnnotatedPinsInferenceConfig()
    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None, modelDir, False, classBGR=None)

    # saveDir = '/home/trevol/HDD_DATA/TMP/frames/detect_all'
    saveDir = os.path.join(nodeConfig.workingDir, 'detect_all')
    # imagesDirs = ['/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6']
    imagesDirs = [os.path.join(nodeConfig.workingDir, 'frames_6')]
    imageExt = 'jpg'

    if save:
        imagesGen = Utils.imageFlow(paths=imagesDirs, ext=imageExt, start=None, stop=None, step=saveStep)
        trainer.saveDetections(imagesGen, saveDir)
    if show:
        trainer.showSavedDetections(saveDir, showInReverseOrder, imagesDirs, imageExt, 1)


def saveVisualizedDetections():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    inferenceConfig = RoughAnnotatedPinsInferenceConfig()
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None, modelDir=None, visualize=False,
                               classBGR=None)

    # saveDir = '/home/trevol/HDD_DATA/TMP/frames/detect_all'
    saveDir = os.path.join(nodeConfig.workingDir, 'detect_all')
    # saveVisualizationToDir = '/home/trevol/HDD_DATA/TMP/frames/detect_all_visualization'
    saveVisualizationToDir = os.path.join(nodeConfig.workingDir, 'detect_all_visualization')
    # imagesDirs = ['/home/trevol/HDD_DATA/Computer_Vision_Task/frames_6']
    imagesDirs = [os.path.join(nodeConfig.workingDir, 'frames_6')]
    imageExt = 'jpg'

    trainer.showSavedDetections(saveDir, False, imagesDirs, imageExt, step=1,
                                saveVisualizationToDir=saveVisualizationToDir)


# saveOrShowDetections(save=True, saveStep=1, show=False, showInReverseOrder=False)
# saveVisualizedDetections()

main_explore_dataset()
# main_train()

# export PYTHONPATH=$PYTHONPATH:../../../..
