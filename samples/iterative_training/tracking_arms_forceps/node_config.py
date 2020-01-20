from collections import namedtuple
import platform

NodeConfig = namedtuple("NodeConfig",
                        ["frames6Dir", "frames2Dir", 'initialWeights', "workingDir", "IMAGES_PER_GPU", "visualize"])

nodesConfigs = {
    "trevol-gpu-nb": NodeConfig(
        frames6Dir="/hdd/nfs_share/video_6",
        frames2Dir="/hdd/nfs_share/video_2",
        initialWeights='/hdd/nfs_share/mask-rcnn/mask_rcnn_coco.h5',
        # workingDir="/hdd/nfs_share/mask-rcnn/tracking_arms_forceps",
        workingDir="/hdd/nfs_share/mask-rcnn/tracking_forceps_w_solder",
        IMAGES_PER_GPU=1,
        visualize=True
    ),
    "trevol-gpu-server": NodeConfig(
        frames6Dir="/nvme_data/pin_n_solder/video_6",
        frames2Dir="/nvme_data/pin_n_solder/video_2",
        initialWeights='/nvme_data/pin_n_solder/mask-rcnn/mask_rcnn_coco.h5',
        # workingDir="/nvme_data/pin_n_solder/mask-rcnn/tracking_arms_forceps",
        workingDir="/nvme_data/pin_n_solder/mask-rcnn/tracking_forceps_w_solder",
        IMAGES_PER_GPU=2,
        visualize=False
    ),
}

nodeConfig = nodesConfigs.get(platform.node())
