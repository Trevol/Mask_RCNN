# return platform.uname().node
# import os
# return os.uname().nodename
from collections import namedtuple
import platform

NodeConfig = namedtuple("NodeConfig", ["workingDir", "visualize"])

nodesConfigs = {
    "trevol-gpu-nb": NodeConfig(
        workingDir="/HDD_DATA/nfs_share/mask-rcnn/pins/rough_dataset",
        visualize=True
    ),
    "trevol-gpu-server": NodeConfig(
        workingDir="/trevol_gpu_nb_share/mask-rcnn/pins/rough_dataset",
        visualize=False
    ),
}

nodeConfig = nodesConfigs.get(platform.node())
