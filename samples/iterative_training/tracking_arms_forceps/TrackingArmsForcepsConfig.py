from mrcnn.config import Config
from samples.iterative_training.pins.PinsConfig import PinsConfig


class TrackingArmsForcepsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "TrackingArmsForcepsConfig"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + arm + forceps + forceps+solder + pin-array
    STEPS_PER_EPOCH = 100

    # BACKBONE = "resnet101"
    BACKBONE = "resnet50"

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


class TrackingArmsForcepsInferenceConfig(TrackingArmsForcepsConfig):
    IMAGES_PER_GPU = 1
