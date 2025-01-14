
import numpy as np
import random
import glob
import json


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
            flattened_block = []
            for z in range(dimensions[2]):
                for y in range(dimensions[1]):
                    for x in range(dimensions[0]):
                        r, g, b, a = voxel_block[x, y, z]
                        if a == 0:
                            flattened_block.append(None)
                        else:
                            hex_color = f"#{r:02x}{g:02x}{b:02x}{a:02x}"
                            flattened_block.append(hex_color)
            voxel_blocks_as_list.append(flattened_block)

        voxel_video_json = {
            "Version": 0,
            "Dimensions": {
                "x": dimensions[0],
                "y": dimensions[1],
                "z": dimensions[2]
            },
            "Framerate": framerate,
            "Block_count": len(voxel_blocks),
            "Title": "Voxel Block",
            "Description": "A single frame voxel block.",
            "Blocks": voxel_blocks_as_list
        }
        # Write this to a file
        existing_files = glob.glob("output/voxel_video_*.json")
        last_index = 0

        for file in existing_files:
            index = int(file.split('_')[-1].split('.')[0])
            last_index = max(last_index, index)

        file_name = "voxel_video_%s" % (last_index + 1)
        with open(f"output/{file_name}.json", "w") as outfile:
            json.dump(voxel_video_json, outfile, indent=4)
        return (voxel_video_json,)
