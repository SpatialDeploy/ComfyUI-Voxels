import numpy as np
import random
from trimesh import Trimesh
import os
import trimesh
import openvdb as vdb
import nanovdb
from nanovdb import GridClass
from nanovdb.math import Coord, CoordBBox, Vec3f
import logging


class MeshToVoxelGrid:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "voxel_size": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
            },
        }
    # @classmethod
    # def IS_CHANGED(mesh: Trimesh):
    #    return random.randint(0, 1000)

    RETURN_TYPES = ("VOXEL_GRID",)
    RETURN_NAMES = ("Voxel Grid",)

    FUNCTION = "run"

    CATEGORY = "Voxels"

    def run(self, mesh, voxel_size): 
        # Define the voxel grid size
        if isinstance(mesh, list) and len(mesh) > 0 and isinstance(mesh[0], Trimesh):
            mesh = mesh[0]
        else:
            raise ValueError("Input mesh is not a valid Trimesh object.")
        
        
        voxel_grid = [] # voxelize(mesh, voxel_size)

        
        return (voxel_grid,)
    
    
    

# We only want to show errors, not regular information.
logging.basicConfig(level=logging.ERROR, format='[%(levelname)s] %(message)s')

# These caches help us avoid doing the same work many times.
# For example, if we look up the same texture pixel twice,
# we can just use the cached answer.
TEXTURE_CACHE = {}
MATERIAL_CACHE = {}

def rotate_mesh_90_x(mesh):
    """
    Rotate the mesh by 90 degrees around the X-axis.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input 3D mesh to rotate.

    Returns
    -------
    trimesh.Trimesh
        The rotated mesh.

    Explanation:
    Imagine you're holding a cake and you tilt it forward
    so the top is now facing you. That's like rotating around the X-axis.
    """
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    mesh.apply_transform(rotation_matrix)
    return mesh

def get_optimal_ray_resolution(voxel_grid_size, quality='balanced'):
    """
    Decide how many rays (lines) to use to scan the model for voxelization.

    Parameters
    ----------
    voxel_grid_size : int, float, or tuple of three ints/floats
        The size of the voxel grid or its largest dimension.
    quality : str, optional
        The desired quality setting: 'fast', 'balanced', or 'high'.

    Returns
    -------
    int
        A recommended number of rays to use when scanning the model.

    Explanation:
    More rays means more detail but slower processing.
    Fewer rays means faster but less detail.
    """
    if isinstance(voxel_grid_size, (int, float)):
        max_dim = int(voxel_grid_size)
    else:
        max_dim = max(voxel_grid_size)

    settings = {
        'fast': (1.5, 50, 200),
        'balanced': (2.5, 100, 400),
        'high': (4.0, 200, 800)
    }

    multiplier, min_res, max_res = settings.get(quality, settings['balanced'])
    resolution = int(max_dim * multiplier)
    return max(min(resolution, max_res), min_res)

def interpolate_uv(mesh, face_idx, barycentric_coords):
    """
    Find the texture coordinates (UV) at a point inside a triangular face.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh containing the face.
    face_idx : int
        The index of the face in the mesh.
    barycentric_coords : np.ndarray of shape (3,)
        The barycentric coordinates (u,v,w) inside the face.

    Returns
    -------
    np.ndarray of shape (2,) or None
        The interpolated UV coordinates (u,v). Returns None if no UVs found.

    Explanation:
    Each face of a model can have a texture on it. UV coordinates tell us
    which part of the picture (texture) to use. We blend the corners (face_uvs)
    using the barycentric coordinates.
    """
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        face_uvs = mesh.visual.uv[mesh.faces[face_idx]]
        return (
            barycentric_coords[0] * face_uvs[0] +
            barycentric_coords[1] * face_uvs[1] +
            barycentric_coords[2] * face_uvs[2]
        )
    return None

def get_texture_color_from_uv(mesh, uv):
    """
    Get the color from the texture at a given UV coordinate.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh which has the texture.
    uv : np.ndarray of shape (2,)
        The UV coordinates on the texture.

    Returns
    -------
    np.ndarray of shape (4,)
        The RGBA color at that UV coordinate, with values between 0.0 and 1.0.

    Explanation:
    Imagine a sticker on the object. UV coordinates tell us where on the sticker we look.
    If there's no texture or something goes wrong, we return a transparent color.
    """
    if (hasattr(mesh.visual, 'material') and
            hasattr(mesh.visual.material, 'baseColorTexture')):
        texture = mesh.visual.material.baseColorTexture
        if texture is not None:
            texture_id = id(texture)
            u = uv[0] % 1.0
            v = uv[1] % 1.0
            x = int(np.clip(u * (texture.size[0] - 1), 0, texture.size[0] - 1))
            y = int(np.clip((1 - v) * (texture.size[1] - 1), 0, texture.size[1] - 1))

            cache_key = (texture_id, x, y)
            if cache_key in TEXTURE_CACHE:
                return TEXTURE_CACHE[cache_key]

            color = texture.getpixel((x, y))
            if len(color) == 3:
                result = np.array([color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0], dtype=np.float32)
            else:
                result = np.array([c / 255.0 for c in color], dtype=np.float32)

            TEXTURE_CACHE[cache_key] = result
            return result
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

def interpolate_vertex_colors(mesh, face_idx, barycentric_coords):
    """
    Interpolate vertex colors inside a triangle using barycentric coordinates.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh which has vertex colors.
    face_idx : int
        The index of the face in the mesh.
    barycentric_coords : np.ndarray of shape (3,)
        The barycentric coordinates (u,v,w) inside the face.

    Returns
    -------
    np.ndarray of shape (4,)
        The RGBA color blended from the vertex colors. If no vertex colors,
        returns transparent.

    Explanation:
    Each corner of the triangle can have a color. We mix them together
    depending on where the point is inside the triangle.
    """
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        face_colors = mesh.visual.vertex_colors[mesh.faces[face_idx]]
        if face_colors is not None and len(face_colors) == 3:
            interpolated = (
                barycentric_coords[0] * face_colors[0] +
                barycentric_coords[1] * face_colors[1] +
                barycentric_coords[2] * face_colors[2]
            ) / 255.0
            if isinstance(interpolated, np.ndarray):
                if len(interpolated) == 3:
                    return np.array([interpolated[0], interpolated[1], interpolated[2], 1.0], dtype=np.float32)
                elif len(interpolated) == 4:
                    return interpolated.astype(np.float32)
            else:
                return np.array([interpolated, interpolated, interpolated, 1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

def get_surface_material_color(mesh, face_idx, gamma=2.2):
    """
    Get a basic color from the material if no texture or vertex color is found.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh containing material info.
    face_idx : int
        The index of the face in the mesh.
    gamma : float, optional
        Gamma correction value. Default is 2.2.

    Returns
    -------
    np.ndarray of shape (4,)
        The RGBA color from the material. If not found, returns transparent.

    Explanation:
    Some materials have a simple base color or face color. If no fancy textures
    or vertex colors are available, we fall back on this basic color.
    """
    material = getattr(mesh.visual, 'material', None)
    material_id = id(material) if material is not None else None
    cache_key = (material_id, face_idx)

    if cache_key in MATERIAL_CACHE:
        return MATERIAL_CACHE[cache_key]

    # Try face colors
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        face_color = mesh.visual.face_colors[face_idx]
        if face_color is not None:
            color = face_color.astype(np.float32)
            if np.max(color) > 1.0:
                color /= 255.0
            if len(color) == 3:
                color = np.array([*color, 1.0], dtype=np.float32)
            rgb_corrected = color[:3] ** (1.0 / gamma)
            result = np.array([*rgb_corrected, color[3]], dtype=np.float32)
            MATERIAL_CACHE[cache_key] = result
            return result

    # Try material color
    if material is not None and hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
        color = np.array(material.baseColorFactor, dtype=np.float32)
        if np.max(color) > 1.0:
            color /= 255.0
        if len(color) == 3:
            color = np.array([*color, 1.0], dtype=np.float32)
        rgb_corrected = color[:3] ** (1.0 / gamma)
        result = np.array([*rgb_corrected, color[3]], dtype=np.float32)
        MATERIAL_CACHE[cache_key] = result
        return result

    MATERIAL_CACHE[cache_key] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return MATERIAL_CACHE[cache_key]

def compute_barycentric_coords(triangle, point):
    """
    Compute barycentric coordinates of a point inside a triangle.

    Parameters
    ----------
    triangle : np.ndarray of shape (3,3)
        The vertices of the triangle.
    point : np.ndarray of shape (3,)
        The point inside the triangle.

    Returns
    -------
    np.ndarray of shape (3,) or None
        The barycentric coordinates (u,v,w). If the calculation fails, returns None.

    Explanation:
    Barycentric coordinates tell us how close the point is to each corner of the triangle.
    This is very useful for mixing colors or textures.
    """
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return None

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])

def find_mesh_for_triangle(face_idx, face_counts):
    """
    Find which original mesh a triangle belongs to after meshes are combined.

    Parameters
    ----------
    face_idx : int
        The global face index in the combined mesh.
    face_counts : list of int
        The number of faces each original mesh had.

    Returns
    -------
    (int, int)
        A tuple (mesh_idx, local_tri_idx) where:
        mesh_idx is which mesh this face belongs to,
        local_tri_idx is the face index within that mesh.
    """
    cumulative_faces = np.cumsum([0] + face_counts)
    mesh_idx = np.searchsorted(cumulative_faces, face_idx, side='right') - 1
    local_tri_idx = face_idx - cumulative_faces[mesh_idx]
    return mesh_idx, local_tri_idx

def voxelize(scene_path, voxel_grid_size=16, ray_resolution=None):
    """
    Convert a 3D model into a voxel array (like 3D pixels).

    Parameters
    ----------
    scene_path : str
        The file path to the 3D model (e.g., a .glb file).
    voxel_grid_size : int or tuple of three ints, optional
        The number of voxels along each dimension or a single size for a cube.
        Default is 16, meaning a 16x16x16 cube if int is given.
    ray_resolution : int, optional
        How many rays to shoot to scan the model. If None, chosen automatically.

    Returns
    -------
    np.ndarray or None
        A 4D array (X, Y, Z, 4) containing RGBA voxel colors.
        Returns None if something goes wrong.

    Explanation:
    We cast rays through the 3D model to find where they hit it.
    At each hit point, we find the color and store it in a voxel grid.
    This creates a '3D image' of the model.
    """
    if ray_resolution is None:
        ray_resolution = get_optimal_ray_resolution(voxel_grid_size)

    if isinstance(voxel_grid_size, (int, float)):
        voxel_dims = (int(voxel_grid_size),) * 3
    elif (isinstance(voxel_grid_size, (list, tuple, np.ndarray)) and
          len(voxel_grid_size) == 3):
        voxel_dims = tuple(int(x) for x in voxel_grid_size)
    else:
        raise ValueError("voxel_grid_size must be a number or a 3-number sequence")

    # Load the 3D model
    scene = trimesh.load(scene_path)
    original_meshes = []
    face_counts = []

    # Extract meshes and apply transforms
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        mesh_copy = trimesh.Trimesh(
            vertices=geometry.vertices.copy(),
            faces=geometry.faces.copy(),
            visual=geometry.visual
        )
        mesh_copy.apply_transform(transform)
        original_meshes.append(mesh_copy)
        face_counts.append(len(mesh_copy.faces))

    # Combine all parts into one big mesh
    combined_mesh = trimesh.util.concatenate(original_meshes)
    # Rotate it so it's oriented nicely
    combined_mesh = rotate_mesh_90_x(combined_mesh)

    # Initialize the voxel array (R,G,B,A)
    voxel_array = np.zeros((*voxel_dims, 4), dtype=np.float32)
    bounds = combined_mesh.bounds
    extents = bounds[1] - bounds[0]

    # Decide where to shoot rays from
    x = np.linspace(bounds[0][0], bounds[1][0], ray_resolution)
    y = np.linspace(bounds[0][1], bounds[1][1], ray_resolution)
    z = np.linspace(bounds[0][2], bounds[1][2], ray_resolution)

    origins = []
    directions = []
    # Shoot rays along the X, Y, and Z axes
    for direction in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        dir_arr = np.array(direction)
        if dir_arr[0] != 0:
            for yi in y:
                for zi in z:
                    origins.append([bounds[0][0] if dir_arr[0] > 0 else bounds[1][0], yi, zi])
                    directions.append(dir_arr)
        elif dir_arr[1] != 0:
            for xi in x:
                for zi in z:
                    origins.append([xi, bounds[0][1] if dir_arr[1] > 0 else bounds[1][1], zi])
                    directions.append(dir_arr)
        else:
            for xi in x:
                for yi in y:
                    origins.append([xi, yi, bounds[0][2] if dir_arr[2] > 0 else bounds[1][2]])
                    directions.append(dir_arr)

    origins = np.array(origins)
    directions = np.array(directions)

    # Ray intersection: find where rays hit the mesh
    locations, index_ray, index_tri = combined_mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions
    )

    # Fill the voxel grid with colors at hit points
    if len(locations) > 0:
        normalized_positions = (locations - bounds[0]) / np.max(extents)
        voxel_coords = (normalized_positions * (np.array(voxel_dims) - 1)).astype(int)
        voxel_coords = np.clip(voxel_coords, 0, np.array(voxel_dims) - 1)

        for coord, location, tri_idx in zip(voxel_coords, locations, index_tri):
            mesh_idx, local_tri_idx = find_mesh_for_triangle(tri_idx, face_counts)
            mesh = original_meshes[mesh_idx]

            # Which triangle did we hit?
            triangle = mesh.vertices[mesh.faces[local_tri_idx]]
            barycentric_coords = compute_barycentric_coords(triangle, location)
            if barycentric_coords is None:
                continue

            # Determine the color: try texture, then vertex colors, then material, else white
            color = np.array([0, 0, 0, 0], dtype=np.float32)

            uv = interpolate_uv(mesh, local_tri_idx, barycentric_coords)
            if uv is not None:
                texture_color = get_texture_color_from_uv(mesh, uv)
                if len(texture_color) >= 3 and np.any(texture_color[:3] != 0):
                    color = texture_color
                    if len(color) == 3:
                        color = np.append(color, 1.0)

            if np.all(color == 0):
                vertex_color = interpolate_vertex_colors(mesh, local_tri_idx, barycentric_coords)
                if len(vertex_color) >= 3 and np.any(vertex_color[:3] != 0):
                    color = vertex_color
                    if len(color) == 3:
                        color = np.append(color, 1.0)

            if np.all(color == 0):
                surface_color = get_surface_material_color(mesh, local_tri_idx)
                if len(surface_color) >= 3 and np.any(surface_color[:3] != 0):
                    color = surface_color
                    if len(color) == 3:
                        color = np.append(color, 1.0)

            if np.all(color == 0):
                # Fallback: white color if nothing else found
                color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

            voxel_array[coord[0], coord[1], coord[2]] = np.clip(color[:4], 0.0, 1.0)

    return voxel_array

def create_vdb_grids(voxel_array):
    r_grid = vdb.FloatGrid()
    g_grid = vdb.FloatGrid()
    b_grid = vdb.FloatGrid()
    a_grid = vdb.FloatGrid()

    r_grid.name = 'R'
    g_grid.name = 'G'
    b_grid.name = 'B'
    a_grid.name = 'A'

    r_acc = r_grid.getAccessor()
    g_acc = g_grid.getAccessor()
    b_acc = b_grid.getAccessor()
    a_acc = a_grid.getAccessor()

    for x in range(voxel_array.shape[0]):
        for y in range(voxel_array.shape[1]):
            for z in range(voxel_array.shape[2]):
                color = voxel_array[x, y, z]
                if color[3] > 0:
                    # Use coordinates directly since array is already transformed
                    coord = (x, y, z)
                    r_acc.setValueOn(coord, float(color[0]))
                    g_acc.setValueOn(coord, float(color[1]))
                    b_acc.setValueOn(coord, float(color[2]))
                    a_acc.setValueOn(coord, float(color[3]))

    return [r_grid, g_grid, b_grid, a_grid]

def convert_vdb_to_nvdb(vdb_path, nvdb_path):
    """
    Convert a VDB file to NanoVDB format.

    Parameters
    ----------
    vdb_path : str
        The file path to the VDB file.
    nvdb_path : str
        The file path to save the NanoVDB file.

    Returns
    -------
    None

    Explanation:
    NanoVDB is a more compact format often used for rendering. This function
    reads the VDB (R,G,B grids) and writes them into a NanoVDB file.
    """
    # Read RGB grids
    r_grid = vdb.read(vdb_path, "R")
    g_grid = vdb.read(vdb_path, "G")
    b_grid = vdb.read(vdb_path, "B")

    # Get combined bounding box
    bbox_min, bbox_max = r_grid.evalActiveVoxelBoundingBox()

    # Create dictionary to store RGB values
    rgb_values = {}

    # Create iterators for RGB grids
    r_iter = r_grid.iterOnValues()
    g_iter = g_grid.iterOnValues()
    b_iter = b_grid.iterOnValues()

    # Iterate through R grid and get corresponding values from others
    for r_item in r_iter:
        coord = r_item.min
        r_val = float(r_item.value)

        g_val = 0.0
        b_val = 0.0

        for g_item in g_iter:
            if g_item.min == coord:
                g_val = float(g_item.value)
                break

        for b_item in b_iter:
            if b_item.min == coord:
                b_val = float(b_item.value)
                break

        if r_val != 0 or g_val != 0 or b_val != 0:
            rgb_values[coord] = (r_val, g_val, b_val)

    nano_bbox = CoordBBox(
        Coord(bbox_min[0], bbox_min[1], bbox_min[2]),
        Coord(bbox_max[0], bbox_max[1], bbox_max[2])
    )

    # Create function to return Vec3f values for RGB
    def rgb_func(coord):
        if (coord[0], coord[1], coord[2]) in rgb_values:
            vals = rgb_values[(coord[0], coord[1], coord[2])]
            return Vec3f(vals[0], vals[1], vals[2])
        return Vec3f(0.0, 0.0, 0.0)

    # Create Vec3f grid for RGB
    rgb_handle = nanovdb.tools.createVec3fGrid(
        background=Vec3f(0.0, 0.0, 0.0),
        name="RGB",
        gridClass=GridClass.FogVolume,
        func=rgb_func,
        bbox=nano_bbox
    )

    # Write grid with compression
    nanovdb.io.writeGrid(nvdb_path, rgb_handle, codec=nanovdb.io.Codec.BLOSC)

def process_model(model_path, output_dir=".", voxel_grid_size=16, ray_resolution=None):
    """
    Convert a 3D model into a NanoVDB file by voxelizing and processing it.

    Parameters
    ----------
    model_path : str
        Path to the input 3D model file (like .glb).
    output_dir : str, optional
        Directory where the output files are saved. Defaults to current dir.
    voxel_grid_size : int or tuple of three ints, optional
        Size of the voxel grid. Default is 16 (16x16x16).
    ray_resolution : int, optional
        How many rays to shoot. If None, chosen automatically.

    Returns
    -------
    str or None
        The path to the created NanoVDB file, or None if something went wrong.

    Explanation:
    This function:
    1. Turns the 3D model into a voxel array.
    2. Creates OpenVDB grids from that voxel array.
    3. Saves a VDB file.
    4. Converts that VDB file into a NanoVDB file.
    5. Cleans up temporary files.
    """
    os.makedirs(output_dir, exist_ok=True)
    vdb_dir = output_dir
    os.makedirs(vdb_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(model_path))[0]

    # Convert model to voxels
    voxel_array = voxelize(model_path, voxel_grid_size, ray_resolution)
    if voxel_array is None:
        return None

    # Make VDB grids from voxel data
    grids = create_vdb_grids(voxel_array)
    if not grids:
        return None

    vdb_filename = f"{base_name}.vdb"
    vdb_path = os.path.join(vdb_dir, vdb_filename)
    vdb.write(vdb_path, grids=grids)

    # Convert VDB to NanoVDB
    nvdb_filename = f"{base_name}.nvdb"
    nvdb_path = os.path.join(output_dir, nvdb_filename)
    convert_vdb_to_nvdb(vdb_path, nvdb_path)

    # Remove temporary VDB file and directory if possible
    os.remove(vdb_path)

    return nvdb_path

# if __name__ == "__main__":
#     # Example usage:
#     # Replace "model_cake.glb" with your model file.
#     # This will create a .nvdb file in the current directory.
#     model_path = "model_cake.glb"
#     numpy_path = voxelize(model_path)
#     np.save("model_cake.npy", numpy_path)
#     nvdb_path = process_model(model_path, voxel_grid_size=16)