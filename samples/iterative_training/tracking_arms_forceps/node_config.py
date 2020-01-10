from collections import namedtuple
import platform

NodeConfig = namedtuple("NodeConfig", ["framesDir", 'initialWeights', "workingDir", "IMAGES_PER_GPU", "visualize"])

nodesConfigs = {
    "trevol-gpu-nb": NodeConfig(
        framesDir="/HDD_DATA/nfs_share/frames_6",
        initialWeights='/HDD_DATA/nfs_share/mask-rcnn/mask_rcnn_coco.h5',
        # workingDir="/HDD_DATA/nfs_share/mask-rcnn/tracking_arms_forceps",
        workingDir="/HDD_DATA/nfs_share/mask-rcnn/tracking_forceps_w_solder",
        IMAGES_PER_GPU=1,
        visualize=True
    ),
    "trevol-gpu-server": NodeConfig(
        framesDir="/nvme_data/pin_n_solder/frames_6",
        initialWeights='/trevol_gpu_nb_share/mask-rcnn/mask_rcnn_coco.h5',
        # workingDir="/nvme_data/pin_n_solder/mask-rcnn/tracking_arms_forceps",
        workingDir="/nvme_data/pin_n_solder/mask-rcnn/tracking_forceps_w_solder",
        IMAGES_PER_GPU=2,
        visualize=False
    ),
}

nodeConfig = nodesConfigs.get(platform.node())
