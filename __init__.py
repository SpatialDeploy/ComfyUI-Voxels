from .nodes.MeshToVoxelGrid import *
from .nodes.VoxelGridsToVoxelVideo import *

NODE_CLASS_MAPPINGS = {
    "MeshToVoxelGrid": MeshToVoxelGrid,
    "VoxelGridsToVoxelVideo": VoxelGridsToVoxelVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshToVoxelGrid": "Mesh To Voxel Grid",
    "VoxelGridsToVoxelVideo": "Voxel Grids To Voxel Video",
}
# WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    #    "WEB_DIRECTORY"
]
