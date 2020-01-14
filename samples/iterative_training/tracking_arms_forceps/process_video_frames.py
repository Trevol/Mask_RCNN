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


def getDetectionOutputDir(weightsFile):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    weightsNum = os.path.basename(weightsFile).split('_')[-1]
    return f'detect_all_{weightsNum}_{now}'


def main():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer

    inferenceConfig = TrackingArmsForcepsInferenceConfig(Args().images_per_gpu)

    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    classBGR = [None, (255, 0, 0), (127, 255, 0), (0, 255, 255),
                (127, 0, 255)]  # background + arm + forceps + forceps+solder + pin-array
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None, modelDir, False, classBGR=classBGR,
                               augmentation=None, checkpointFileName=None)

    detectAllDir = getDetectionOutputDir(trainer.findLastWeights())

    for subsetName, framesPath in [('video_6', nodeConfig.frames6Dir), ('video_2', nodeConfig.frames2Dir)]:
        outputImagesDir = os.path.join(framesPath, detectAllDir, subsetName)
        imagesGen = Utils.imageFlow(paths=nodeConfig.framesDir, ext='jpg', start=4173, stop=None, step=1)
        trainer.saveDetectionsV2(imagesGen, inferenceConfig.BATCH_SIZE, pickleDir=None, imagesDir=outputImagesDir,
                                 withBoxes=True, onlyMasks=False)


if __name__ == '__main__':
    main()
