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
    nameWithoutExt = os.path.splitext(os.path.basename(weightsFile))[0]
    weightsNum = nameWithoutExt.split('_')[-1]
    return f'detect_all/{weightsNum}_{now}'


def main():
    from samples.iterative_training.IterativeTrainer import IterativeTrainer

    inferenceConfig = TrackingArmsForcepsInferenceConfig(Args().images_per_gpu)

    modelDir = os.path.join(nodeConfig.workingDir, 'logs')
    classBGR = [None, (255, 0, 0), (127, 255, 0), (0, 255, 255),
                (127, 0, 255)]  # background + arm + forceps + forceps+solder + pin-array
    trainer = IterativeTrainer(None, None, None, None, inferenceConfig, None, modelDir, False, classBGR=classBGR,
                               augmentation=None, checkpointFileName=None)

    detectAllDir = getDetectionOutputDir(trainer.findLastWeights())

    subsets = [('video_6', nodeConfig.frames6Dir, 4173), ('video_2', nodeConfig.frames2Dir, 1754)]
    for subsetName, framesPath, startFrame in subsets:
        imagesGen = Utils.imageFlow(paths=framesPath, ext='jpg', start=startFrame, stop=None, step=1)
        for r in imagesGen:
            print(r[0])
        # outputImagesDir = os.path.join(nodeConfig.workingDir, detectAllDir, subsetName)
        # trainer.saveDetectionsV2(imagesGen, inferenceConfig.BATCH_SIZE, pickleDir=None, imagesDir=outputImagesDir,
        #                          withBoxes=True, onlyMasks=False)


if __name__ == '__main__':
    main()
