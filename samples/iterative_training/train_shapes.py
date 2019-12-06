from samples.iterative_training.IterativeTrainer import IterativeTrainer
from samples.iterative_training.shapes.ShapesConfig import ShapesConfig, ShapesInferenceConfig
from samples.iterative_training.shapes.ShapesDataset import ShapesDataset
from samples.iterative_training.arguments import parser


def prepareTrainerInput():
    trainingConfig = ShapesConfig()
    inferenceConfig = ShapesInferenceConfig()

    imageHeight, imageWidth = trainingConfig.IMAGE_SHAPE[0], trainingConfig.IMAGE_SHAPE[1]

    trainingDataset = ShapesDataset()
    trainingDataset.load_shapes(50, imageHeight, imageWidth)
    trainingDataset.prepare()

    validationDataset = ShapesDataset()
    validationDataset.load_shapes(5, imageHeight, imageWidth)
    validationDataset.prepare()

    def makeTestingDataset():
        testingDataset = ShapesDataset()
        testingDataset.load_shapes(5, imageHeight, imageWidth)
        testingDataset.prepare()
        return testingDataset

    return trainingDataset, validationDataset, makeTestingDataset, trainingConfig, inferenceConfig


def main():
    trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig = prepareTrainerInput()
    trainer = IterativeTrainer(trainingDataset, validationDataset, testingDataset, trainingConfig, inferenceConfig)
    trainer.trainingLoop(parser.parse_args().start == 'vis')


main()
