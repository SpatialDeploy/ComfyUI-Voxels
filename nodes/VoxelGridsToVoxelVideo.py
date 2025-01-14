class VoxelGridsToVoxelVideo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "voxel_blocks": ("VOXEL_BLOCK",),
                "framerate": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
        }

    # @classmethod
    # def IS_CHANGED(voxel_blocks: np.ndarray):
    #     return random.randint(0, 1000)

    INPUT_IS_LIST = True
    RETURN_TYPES = ("VOXEL_VIDEO",)
    RETURN_NAMES = ("voxel_video",)

    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Voxels"

    def run(self, voxel_blocks, framerate):
        print("Inside VoxelViewer: " + str(len(voxel_blocks)))
        dimensions = voxel_blocks[0].shape
        print(dimensions)
        voxel_blocks_as_list = []

        for voxel_block in voxel_blocks:
            voxel_blocks_as_list.append(voxel_block)
        voxel_video_json = "Holding"
        return (voxel_video_json,)
