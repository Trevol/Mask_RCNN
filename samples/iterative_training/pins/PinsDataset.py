from samples.iterative_training.cvat.CVATDataset import CVATDataset


class PinsDataset(CVATDataset):
    def __init__(self, labels, imagesDirs, imageAnnotations):
        super(PinsDataset, self).__init__('pins', labels, imagesDirs, imageAnnotations)
