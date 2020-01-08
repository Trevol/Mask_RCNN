# return platform.uname().node
# import os
# return os.uname().nodename
from collections import namedtuple
import platform

NodeConfig = namedtuple("NodeConfig", ["framesDir", 'initialWeights', "workingDir", "IMAGES_PER_GPU", "visualize"])

nodesConfigs = {
    "trevol-gpu-nb": NodeConfig(
        framesDir="/HDD_DATA/nfs_share/frames_6",
        initialWeights='/HDD_DATA/nfs_share/mask-rcnn/mask_rcnn_coco.h5',
        workingDir="/HDD_DATA/nfs_share/mask-rcnn/pins/rough_dataset",
        IMAGES_PER_GPU=1,
        visualize=True
    ),
    "trevol-gpu-server": NodeConfig(
        framesDir="/trevol_gpu_nb_share/frames_6",
        initialWeights='/trevol_gpu_nb_share/mask-rcnn/mask_rcnn_coco.h5',
        workingDir="/trevol_gpu_nb_share/mask-rcnn/pins/rough_dataset",
        IMAGES_PER_GPU=2,
        visualize=False
    ),
}

nodeConfig = nodesConfigs.get(platform.node())
