import os, sys, timeit

from samples.balloon.dataset.BaloonDataset import BalloonDataset

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

from mrcnn import model, utils
from mrcnn.config import Config


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    BACKBONE = 'resnet50'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


def train(model):
    """Train the model."""
    dataset = './dataset'
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    start = timeit.default_timer()
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')
    duration = timeit.default_timer() - start
    print("training completed ", duration)


config = BalloonConfig()

model = model.MaskRCNN(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs"))
model.load_weights(os.path.join(ROOT_DIR, "mask_rcnn_coco.h5"), by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

train(model)
