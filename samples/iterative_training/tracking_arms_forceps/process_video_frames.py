import argparse
import os

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
                               augmentation=None)

    # pickleDir = os.path.join(nodeConfig.workingDir, 'detect_all/pickles')
    outputImagesDir = os.path.join(nodeConfig.workingDir, 'detect_all/visualization')

    imagesGen = Utils.imageFlow(paths=nodeConfig.framesDir, ext='jpg', start=4173, stop=None, step=1)

    for i, batch in enumerate(Utils.batchFlow(imagesGen, 6)):
        if len(batch) != 6:
            print('DDDDD', len(batch))
        if i > 0 and i % 100 == 0:
            print(f'{i} batches processed')

    # trainer.saveDetectionsV2(imagesGen, inferenceConfig.BATCH_SIZE, pickleDir=None, imagesDir=outputImagesDir,
    #                          withBoxes=True, onlyMasks=False)


if __name__ == '__main__':
    main()
