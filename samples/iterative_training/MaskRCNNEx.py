import os

from mrcnn.model import MaskRCNN


class MaskRCNNEx(MaskRCNN):
    def findLastWeights(self):
        """Finds the last checkpoint file of the last trained model in the
                model directory.
                Returns:
                    The path of the last checkpoint file
                """
        return self.findLastWeightsInModelDir(self.model_dir, self.config.NAME.lower())

    @staticmethod
    def findLastWeightsInModelDir(modelDir, modelName):
        """Finds the last checkpoint file of the last trained model in the
                        model directory.
                        Returns:
                            The path of the last checkpoint file
                        """
        # Get directory names. Each directory corresponds to a model
        for _, dir_names, _ in list(os.walk(modelDir))[:1]:
            dir_names = filter(lambda f: f.startswith(modelName), dir_names)
            for dirName in sorted(dir_names, reverse=True):
                dirName = os.path.join(modelDir, dirName)
                # Find the last checkpoint
                checkpointFiles = next(os.walk(dirName))[2]
                checkpointFiles = filter(lambda f: f.startswith("mask_rcnn") and f.endswith('.h5'), checkpointFiles)
                checkpointFiles = sorted(checkpointFiles)
                if len(checkpointFiles):
                    return os.path.join(dirName, checkpointFiles[-1])
        return None