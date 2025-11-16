import bpy
import zmq
import json
import mathutils
import os
import tempfile
import numpy as np
import uuid
import math 
import random
# Initialize the ZMQ server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

cubes = {}
temp_dir = "tmp"
print(f"Temporary directory: {temp_dir}")

# Templates will be loaded at the end of the script
cube_template = None
vis_cube_template = None

def setup_base_render_settings():

    scene = bpy.context.scene
    # scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Raw'
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.use_stamp = False
    
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    
    bpy.context.view_layer.use_pass_z = True

def load_cube(cube_path, asset_names):
    asset_file_path = os.path.join(os.path.dirname(__file__), "3d_models", cube_path)

    if isinstance(asset_names, str):
        asset_names = [asset_names]

    loaded_object = None  # Initialize to None

    with bpy.data.libraries.load(asset_file_path, link=False) as (data_from, data_to):
        data_to.objects = asset_names

    for candidate_object in data_to.objects:
        if candidate_object:
            loaded_object = candidate_object
            break

    if not loaded_object:
        return None

    if loaded_object.name not in bpy.context.collection.objects:
        bpy.context.collection.objects.link(loaded_object)
        
    return loaded_object

def setup_camera():
    """Configure the camera to match render.py."""
    camera_name = "Camera"
    
    # Remove existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)

    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.name = camera_name
    camera.location = (0, 0, 0)
    camera.rotation_euler = (math.radians(180), 0, 0)

    scene = bpy.context.scene
    scene.camera = camera
    
    return camera

def render_scene():
    """Render the current scene and return the directory containing the images."""
    setup_base_render_settings()  # Configure common render parameters
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    render_id = str(uuid.uuid4())
    render_dir = os.path.abspath(os.path.join(temp_dir, f"render_{render_id}"))
    os.makedirs(render_dir, exist_ok=True)

    all_normal_cube_objs = []
    all_vis_cube_objs = []

    for cube_id_key in cubes:
        normal_obj_name = cubes[cube_id_key].get("object")
        vis_obj_name = cubes[cube_id_key].get("vis_object")
        if normal_obj_name:
            normal_obj = bpy.data.objects.get(normal_obj_name)
            if normal_obj:
                all_normal_cube_objs.append(normal_obj)
        if vis_obj_name:
            vis_obj = bpy.data.objects.get(vis_obj_name)
            if vis_obj:
                all_vis_cube_objs.append(vis_obj)

    original_hide_states = {}
    for obj_list in [all_normal_cube_objs, all_vis_cube_objs]:
        for obj_item in obj_list:
            original_hide_states[obj_item.name] = obj_item.hide_render

    # 1. Render full_scene.exr (standard cubes only, RGBA + depth)
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.color_mode = 'RGBA'
    view_layer.use_pass_z = True

    for vis_cb in all_vis_cube_objs:
        vis_cb.hide_render = True
    for normal_cb in all_normal_cube_objs:
        normal_cb.hide_render = False 
    
    full_scene_path = os.path.join(render_dir, "full_scene.exr")
    scene.render.filepath = full_scene_path
    bpy.ops.render.render(write_still=True)
    
    # 2. Render full_scene_vis.png (visualization cubes only, RGBA)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    view_layer.use_pass_z = False
    
    for normal_cb in all_normal_cube_objs:
        normal_cb.hide_render = True
    for vis_cb in all_vis_cube_objs:
        vis_cb.hide_render = False
    
    full_scene_vis_path = os.path.join(render_dir, "full_scene_vis.png")
    scene.render.filepath = full_scene_vis_path
    bpy.ops.render.render(write_still=True)
    
    if cubes:
    # 3. Render each standard cube individually (cube_{cube_id}.exr, RGBA + depth)
        scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        scene.render.image_settings.color_depth = '16'
        scene.render.image_settings.color_mode = 'RGBA'
        view_layer.use_pass_z = True
        
        for vis_cb in all_vis_cube_objs:  # Keep all visualization cubes hidden
            vis_cb.hide_render = True
        for normal_cb_to_hide in all_normal_cube_objs:  # Hide all standard cubes first
            normal_cb_to_hide.hide_render = True
        
        for cube_id_iter in cubes:
            obj_name = cubes[cube_id_iter].get("object")
            if obj_name:
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    obj.hide_render = False  # Show the current standard cube
                    
                    cube_path = os.path.join(render_dir, f"cube_{cube_id_iter}.exr")
                    scene.render.filepath = cube_path
                    bpy.ops.render.render(write_still=True)
                    
                    obj.hide_render = True  # Hide the cube again
        
        # 4. Render each visualization cube individually (cube_{cube_id}_vis.png, RGBA)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        view_layer.use_pass_z = False  # PNG does not require the depth pass

        for normal_cb in all_normal_cube_objs:  # Hide all standard cubes
            normal_cb.hide_render = True
        for vis_cb_to_hide in all_vis_cube_objs:  # Hide all visualization cubes first
            vis_cb_to_hide.hide_render = True

        for cube_id_iter in cubes:
            vis_obj_name = cubes[cube_id_iter].get("vis_object")
            if vis_obj_name:
                vis_obj = bpy.data.objects.get(vis_obj_name)
                if vis_obj:
                    vis_obj.hide_render = False  # Show the current visualization cube
                    
                    vis_cube_path = os.path.join(render_dir, f"cube_{cube_id_iter}_vis.png")
                    scene.render.filepath = vis_cube_path
                    bpy.ops.render.render(write_still=True)
                    
                    vis_obj.hide_render = True  # Hide the visualization cube again
    
    # Restore the original hide_render state for all cubes
    for obj_name_restore, hidden_restore in original_hide_states.items():
        obj_to_restore = bpy.data.objects.get(obj_name_restore)
        if obj_to_restore:
            obj_to_restore.hide_render = hidden_restore
    
    return render_dir

def add_object(obj_template, translation, size):
    obj = obj_template.copy()
    obj.data = obj_template.data.copy()
    bpy.context.collection.objects.link(obj)
    obj.hide_render = False
    obj.hide_viewport = False

    rotation = np.array([[0,1,0],
                         [0,0,-1],
                         [-1,0,0]])
    translation = np.array(translation)
    size = np.array(size)

    mat = np.eye(4)
    mat[:3, 3] = translation
    # Normalize rotation matrix columns and apply scaling
    for i in range(3):
        norm = np.linalg.norm(rotation[:, i])
        rotation[:, i] = rotation[:, i] / norm * size[i]
    mat[:3, :3] = rotation

    obj.matrix_world = mathutils.Matrix(mat.tolist())
    
    return obj

def add_cube(params):
    """Add a cube to the scene."""
    global cube_template, vis_cube_template


    cube_id = params["id"]
    size = params["size"]
    location = params["location"]
    
    obj = add_object(cube_template, location, size)
    obj.name = f"Cube_{cube_id}"
    
    vis_obj = None
    if vis_cube_template:
        vis_obj = add_object(vis_cube_template, location, size)
        vis_obj.name = f"Vis_Cube_{cube_id}"
        if vis_obj and obj:  # Ensure both vis_obj and obj exist
            # Ensure the initial transform matches the standard cube, since add_object recomputes matrix_world
            vis_obj.matrix_world = obj.matrix_world

    # Extract transform data from the resulting matrix_world
    actual_loc, actual_rot_quat, actual_scale = obj.matrix_world.decompose()
    # Convert quaternion rotation to Euler angles (radians) and then to degrees
    actual_rotation_euler_degrees = [math.degrees(angle) for angle in actual_rot_quat.to_euler()]

    cubes[cube_id] = {
        "id": cube_id,
        "object": obj.name,
        "vis_object": vis_obj.name if vis_obj else None,
        "size": list(actual_scale),  # Use the decomposed scale
        "location": list(actual_loc),  # Use the decomposed location
        "rotation": actual_rotation_euler_degrees,  # Use the decomposed rotation (degrees)
    }
    
    image_dir = render_scene()
    return {"status": f"Added cube: {cube_id}", "cubes": get_cubes_data(), "image_dir": image_dir}

def delete_cube(params):
    """Remove a cube from the scene."""
    cube_id = params["id"]
    
    if cube_id in cubes:
        obj_name = cubes[cube_id].get("object")
        vis_obj_name = cubes[cube_id].get("vis_object")
        
        if obj_name:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                bpy.data.objects.remove(obj)
        
        if vis_obj_name:
            vis_obj = bpy.data.objects.get(vis_obj_name)
            if vis_obj:
                bpy.data.objects.remove(vis_obj)

        del cubes[cube_id]
        image_dir = render_scene()
        return {"status": f"Deleted cube: {cube_id}", "cubes": get_cubes_data(), "image_dir": image_dir}
    else:
        image_dir = render_scene()  # Render the current scene state even when not found
        return {"status": f"Cube not found: {cube_id}", "cubes": get_cubes_data(), "image_dir": image_dir}

def transform_cube(params):
    """Transform a cube (translate, rotate, or scale)."""
    cube_id = params["id"]
    operation = params["operation"]
    axis = params["axis"]
    value = params["value"]
    
    if cube_id not in cubes:
        image_dir = render_scene()
        return {"status": f"Cube not found: {cube_id}", "cubes": get_cubes_data(), "image_dir": image_dir}
    
    obj_name = cubes[cube_id]["object"]
    obj = bpy.data.objects.get(obj_name)

    vis_obj_name = cubes[cube_id].get("vis_object")
    vis_obj = bpy.data.objects.get(vis_obj_name) if vis_obj_name else None
    
    if not obj:
        image_dir = render_scene()
        return {"status": f"Object not found: {obj_name}", "cubes": get_cubes_data(), "image_dir": image_dir}
    
    # Apply the original transform logic
    loc, rot, scale = obj.matrix_world.decompose()
    rot = rot.to_matrix()  # rot is a mathutils.Matrix

    if axis == "X":
        vec = mathutils.Vector((1, 0, 0))
    elif axis == "Y":
        vec = mathutils.Vector((0, 1, 0))
    elif axis == "Z":
        vec = mathutils.Vector((0, 0, 1))
    # Note: invalid axes are not handled here, mirroring the original behavior

    if operation == "Translate":
        loc += rot @ vec * value
        
    elif operation == "Rotate":
        rot = rot @ mathutils.Matrix.Rotation(math.radians(value), 3, vec)

    elif operation == "Scale":
        # Scale along the selected axis
        if axis == "X":
            scale[0] *= value
        elif axis == "Y":
            scale[1] *= value
        elif axis == "Z":
            scale[2] *= value
    
    # Reconstruct the matrix using location, rotation, and scale
    obj.matrix_world = mathutils.Matrix.LocRotScale(loc, rot, scale)
    
    if vis_obj:
        vis_obj.matrix_world = obj.matrix_world  # Keep the visualization cube in sync

    # Update the cube data cache
    new_loc, new_rot_quat, new_scale = obj.matrix_world.decompose()
    cubes[cube_id]["location"] = list(new_loc)
    cubes[cube_id]["rotation"] = [math.degrees(angle) for angle in new_rot_quat.to_euler()]
    cubes[cube_id]["size"] = list(new_scale)
    
    image_dir = render_scene()
    return {"status": f"{operation} cube {cube_id} along {axis} axis by {value}", "cubes": get_cubes_data(), "image_dir": image_dir}

def random_transform_cube(params):
    """Apply a random transform to a cube."""
    cube_id = params["id"]

    if cube_id not in cubes:
        image_dir = render_scene()
        return {"status": f"Cube not found: {cube_id}", "cubes": get_cubes_data(), "image_dir": image_dir}

    obj_name = cubes[cube_id]["object"]
    obj = bpy.data.objects.get(obj_name)

    vis_obj_name = cubes[cube_id].get("vis_object")
    vis_obj = bpy.data.objects.get(vis_obj_name) if vis_obj_name else None

    if not obj:
        image_dir = render_scene()
        return {"status": f"Object not found: {obj_name}", "cubes": get_cubes_data(), "image_dir": image_dir}


    trans_range = (-2.0, 2.0)
    rot_range_deg = (-180.0, 180.0)
    scale_range = (0.5, 2.0)

    loc, rot, scale = obj.matrix_world.decompose()
    rot = rot.to_matrix()

    random_translation = mathutils.Vector((random.uniform(trans_range[0], trans_range[1]),
                                           random.uniform(trans_range[0], trans_range[1]),
                                           random.uniform(trans_range[0], trans_range[1])))

    random_rotation_value = random.uniform(rot_range_deg[0], rot_range_deg[1])
    random_rotation_matrix = mathutils.Matrix.Rotation(math.radians(random_rotation_value), 3, (0, 0, 1))

    random_scale_vec = mathutils.Vector((random.uniform(scale_range[0], scale_range[1]), 
                                     random.uniform(scale_range[0], scale_range[1]),
                                     random.uniform(scale_range[0], scale_range[1])))


    loc = loc + random_translation
    rot = rot @ random_rotation_matrix  # Matrix multiplication
    

    scale[0] *= random_scale_vec[0]
    scale[1] *= random_scale_vec[1]
    scale[2] *= random_scale_vec[2]

    obj.matrix_world = mathutils.Matrix.LocRotScale(loc, rot, scale)

    if vis_obj:
        vis_obj.matrix_world = obj.matrix_world  # Keep the visualization cube in sync

    # Update the cube data cache
    new_loc, new_rot_quat, new_scale = obj.matrix_world.decompose()
    cubes[cube_id]["location"] = list(new_loc)
    cubes[cube_id]["rotation"] = [math.degrees(angle) for angle in new_rot_quat.to_euler()]
    cubes[cube_id]["size"] = list(new_scale)

    image_dir = render_scene()
    return {"status": f"Applied random transform to cube {cube_id}", "cubes": get_cubes_data(), "image_dir": image_dir}

def get_cubes_data():
    """Return metadata for all cubes."""
    return [
        {
            "id": cube_id,
            "size": cubes[cube_id]["size"],
            "location": cubes[cube_id]["location"],
            "rotation": cubes[cube_id]["rotation"]
        }
        for cube_id in cubes
    ]

def refresh():
    """Refresh the scene."""
    image_dir = render_scene()
    return {"status": "Scene refreshed", "cubes": get_cubes_data(), "image_dir": image_dir}

# Initialize the scene on startup
print("Initializing scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

setup_base_render_settings()  # Apply base render settings during initialization
setup_camera()

cube_template = load_cube("coord_cube_xyz.blend", ["Cube", "立方体"])

cube_template.hide_render = True
cube_template.hide_viewport = True

vis_cube_template = load_cube("vis_coord_cube_xyz.blend","CoordinateCube")

vis_cube_template.hide_render = True
vis_cube_template.hide_viewport = True


initial_cube_id = "initial_cube"
initial_params = {
    "id": initial_cube_id,
    "size": [1.0, 1.0, 1.0],
    "location": [0.0, 0.0, 10.0]
}

# Add an initial cube
add_cube_result = add_cube(initial_params)

rotation_params = {
    "id": initial_cube_id,
    "operation": "Rotate",
    "axis": "Z",
    "value": 45
}
transform_cube(rotation_params)

scale_params = {
    "id": initial_cube_id,
    "operation": "Scale",
    "axis": "X", 
    "value": 1.4
}
transform_cube(scale_params)

print("Initial cube setup complete!")

# Main loop
while True:
    try:
        # Wait for a request
        message = socket.recv_json()
        print(f"Received request: {message['command']}")
        
        command = message["command"]
        params = message.get("params", {})
        
        # Dispatch command
        if command == "add_cube":
            response = add_cube(params)
        elif command == "delete_cube":
            response = delete_cube(params)
        elif command == "transform_cube":
            response = transform_cube(params)
        elif command == "random_transform_cube":
            response = random_transform_cube(params)
        elif command == "refresh":
            response = refresh()
        else:
            response = {"status": f"Unknown command: {command}", "cubes": get_cubes_data()}
        
        # Send response
        socket.send_json(response)
        
    except Exception as e:
        # Return the error details to the caller
        error_response = {"status": f"Error: {str(e)}", "cubes": get_cubes_data()}
        socket.send_json(error_response)
        print(f"Error handling request: {e}")