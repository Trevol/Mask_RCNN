from samples.iterative_training.pins.CvatAnnotation import CvatAnnotation
from samples.iterative_training.pins.PinsConfig import PinsConfig, PinsInferenceConfig
from samples.iterative_training.pins.PinsDataset import PinsDataset
from samples.iterative_training.arguments import parser


def prepareTrainerInput():
    trainingConfig = PinsConfig()
    inferenceConfig = PinsInferenceConfig()

    imageHeight, imageWidth = trainingConfig.IMAGE_SHAPE[0], trainingConfig.IMAGE_SHAPE[1]

    trainingDataset = PinsDataset()
    trainingDataset.load_shapes(50, imageHeight, imageWidth)
    trainingDataset.prepare()

    validationDataset = PinsDataset()
    validationDataset.load_shapes(5, imageHeight, imageWidth)
    validationDataset.prepare()

    def makeTestingDataset():
        testingDataset = PinsDataset()
        testingDataset.load_shapes(5, imageHeight, imageWidth)
        testingDataset.prepare()
        return testingDataset

    return trainingDataset, validationDataset, makeTestingDataset, trainingConfig, inferenceConfig


def main():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer
    trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig = prepareTrainerInput()
    trainer = IterativeTrainer(trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig)
    trainer.trainingLoop(parser.parse_args().start == 'vis')


def main():
    labels, imageAnnotations = CvatAnnotation.parse('1_TestSegmentation.xml')
    PinsDataset()


main()
