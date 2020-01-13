import argparse
import os
from pathlib import Path
from samples.iterative_training.Utils import Utils
from samples.iterative_training.tracking_arms_forceps.TrackingArmsForcepsConfig import \
    TrackingArmsForcepsInferenceConfig
from samples.iterative_training.tracking_arms_forceps.node_config import nodeConfig


class Args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_per_gpu', required=False, default=1, type=int)

    def __new__(cls, *args, **kwargs):
        return cls.parser.parse_args()


def main():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer

    inferenceConfig = TrackingArmsForcepsInferenceConfig(Args().images_per_gpu)

    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    classBGR = [None, (255, 0, 0), (127, 255, 0), (0, 255, 255),
                (127, 0, 255)]  # background + arm + forceps + forceps+solder + pin-array
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None, modelDir, False, classBGR=classBGR,
                               augmentation=None, checkpointFileName=None)

    # outputImagesDir = os.path.join(nodeConfig.workingDir, 'detect_all/frames_6/visualization')
    # imagesGen = Utils.imageFlow(paths=nodeConfig.framesDir, ext='jpg', start=4173, stop=None, step=1)

    outputImagesDir = os.path.join(nodeConfig.workingDir, 'detect_all/frames_2/visualization')
    pathToFrames2 = os.path.join(Path(nodeConfig.framesDir).parent, 'frames_2')
    imagesGen = Utils.imageFlow(paths=pathToFrames2, ext='jpg', start=1754, stop=None, step=1)

    trainer.saveDetectionsV2(imagesGen, inferenceConfig.BATCH_SIZE, pickleDir=None, imagesDir=outputImagesDir,
                             withBoxes=True, onlyMasks=False)


if __name__ == '__main__':
    main()
