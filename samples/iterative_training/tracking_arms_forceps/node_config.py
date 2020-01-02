# return platform.uname().node
# import os
# return os.uname().nodename
from collections import namedtuple
import platform

NodeConfig = namedtuple("NodeConfig", ["workingDir", "IMAGES_PER_GPU", "visualize"])

nodesConfigs = {
    "trevol-gpu-nb": NodeConfig(
        workingDir="/HDD_DATA/nfs_share/mask-rcnn/pins/rough_dataset",
        IMAGES_PER_GPU=1,
        visualize=True
    ),
    "trevol-gpu-server": NodeConfig(
        workingDir="/trevol_gpu_nb_share/mask-rcnn/pins/rough_dataset",
        IMAGES_PER_GPU=2,
        visualize=False
    ),
}

nodeConfig = nodesConfigs.get(platform.node())
