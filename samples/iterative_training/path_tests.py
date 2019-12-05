import os


def findLastWeights(model_dir, key):
    """Finds the last checkpoint file of the last trained model in the
            model directory.
            Returns:
                The path of the last checkpoint file
            """
    # Get directory names. Each directory corresponds to a model
    for _, dir_names, _ in list(os.walk(model_dir))[:1]:
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        for dirName in sorted(dir_names, reverse=True):
            dirName = os.path.join(model_dir, dirName)
            # Find the last checkpoint
            checkpointFiles = next(os.walk(dirName))[2]
            checkpointFiles = filter(lambda f: f.startswith("mask_rcnn") and f.endswith('.h5'), checkpointFiles)
            checkpointFiles = sorted(checkpointFiles)
            if len(checkpointFiles):
                return os.path.join(dirName, checkpointFiles[-1])
    return None

def tests():
    assert findLastWeights('./logs', 'shapes') == './logs/shapes20191204T0921/mask_rcnn_shapes_0002.h5'
    assert findLastWeights('./logs_empty', 'shapes') is None
    assert findLastWeights('./logs22', 'shapes') is None
    assert findLastWeights('./logs22', 'shapes22') is None
    assert findLastWeights('./logs', 'shapes22') is None


tests()
