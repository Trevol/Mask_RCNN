import os

from samples.iterative_training.Utils import Utils
from samples.iterative_training.tracking_arms_forceps.TrackingArmsForcepsConfig import \
    TrackingArmsForcepsInferenceConfig
from samples.iterative_training.tracking_arms_forceps.node_config import nodeConfig


def main():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer

    inferenceConfig = TrackingArmsForcepsInferenceConfig()
    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    classBGR = [None, (255, 0, 0), (127, 255, 0), (0, 255, 255),
                (127, 0, 255)]  # background + arm + forceps + forceps+solder + pin-array
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None, modelDir, False, classBGR=classBGR,
                               augmentation=None)

    # pickleDir = os.path.join(nodeConfig.workingDir, 'detect_all/pickles')
    outputImagesDir = os.path.join(nodeConfig.workingDir, 'detect_all/visualization')

    imagesGen = Utils.imagesGenerator(paths=nodeConfig.framesDir, ext='jpg', start=4173, stop=None, step=1)

    trainer.saveDetectionsV2(imagesGen, pickleDir=None, imagesDir=outputImagesDir, withBoxes=True, onlyMasks=False)


if __name__ == '__main__':
    main()
