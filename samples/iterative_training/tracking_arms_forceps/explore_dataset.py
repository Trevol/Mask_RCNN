from samples.iterative_training.Utils import Utils
from samples.iterative_training.tracking_arms_forceps.node_config import nodeConfig
from samples.iterative_training.tracking_arms_forceps.train_for_arm_tracking import prepareTrainerInput


def main_explore_dataset():
    trainingDataset, _, _ = prepareTrainerInput(nodeConfig.frames6Dir, nodeConfig.frames2Dir)
    # Utils.exploreDatasets(trainingDataset, validationDataset)
    Utils.exploreDatasets(trainingDataset)


if __name__ == '__main__':
    main_explore_dataset()
