import gradio as gr
import zmq
import json
import uuid
import numpy as np
import os
from PIL import Image
import openexr_numpy
import torch
import subprocess
import argparse
from pipelines.pipeline_scenedesigner import StableDiffusion3ControlNetPipeline
from diffusers import SD3ControlNetModel
import signal
import atexit

class SceneDesignerUI:
    def __init__(self, pipeline, global_config):
        self.pipeline = pipeline
        self.global_config = global_config
        
        # Initialize the ZMQ connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
    
    def load_images_from_dir(self, image_dir):
        if image_dir and os.path.isdir(image_dir):
            full_scene_vis_path = os.path.join(image_dir, "full_scene_vis.png")
            if os.path.exists(full_scene_vis_path):
                pil_image = Image.open(full_scene_vis_path).convert("RGB")
                return [(pil_image, "Rendered Scene (Visualization)")]
        return []
    
    def send_command(self, command, params=None):
        """Send a command to Blender and handle the returned image directory."""
        if params is None:
            params = {}
        
        message = {"command": command, "params": params}
        self.socket.send_json(message)
        response = self.socket.recv_json()
        
        image_list = []
        image_dir = None
        # Handle the image directory path
        if "image_dir" in response:
            image_dir = response["image_dir"]
            image_list = self.load_images_from_dir(image_dir)
        
        return response["status"], response.get("cubes", []), image_list, image_dir
    
    def add_cube(self, cube_name=""):
        cube_id = cube_name.strip() if cube_name.strip() else str(uuid.uuid4())
        
        params = {
            "id": cube_id,
            "size": [1.0, 1.0, 1.0],
            "location": [0.0, 0.0, 10.0]
        }
        
        status, cubes, image_list, image_dir = self.send_command("add_cube", params)
        return status, gr.update(choices=[c["id"] for c in cubes], value=cube_id), image_list, image_dir, cubes
    
    def delete_cube(self, cube_id):
        if not cube_id:
            return "Please select a cube first", gr.update(), [], None, []
        
        status, cubes, image_list, image_dir = self.send_command("delete_cube", {"id": cube_id})
        
        # Update the selection dropdown
        new_cube_ids = [c["id"] for c in cubes]
        return status, gr.update(choices=new_cube_ids, value=None if not new_cube_ids else new_cube_ids[0]), image_list, image_dir, cubes
    
    def transform_cube(self, cube_id, operation, axis, value):
        if not cube_id:
            return "Please select a cube first", gr.update(), [], None, []
        
        params = {
            "id": cube_id,
            "operation": operation,
            "axis": axis,
            "value": float(value)
        }
        
        status, cubes, image_list, image_dir = self.send_command("transform_cube", params)
        return status, gr.update(value=cube_id), image_list, image_dir, cubes
    
    def refresh_scene(self):
        """Refresh the scene to fetch the latest render."""
        status, cubes, image_list, image_dir = self.send_command("refresh")
        return status, gr.update(choices=[c["id"] for c in cubes], value=None if not cubes else cubes[0]["id"]), image_list, image_dir, cubes
    
    def update_slider_range(self, operation):
        """Update the slider range based on the selected operation."""
        if operation == "Translate":
            return gr.Slider(-10.0, 10.0, value=0.0, step=0.1, label="Translation Distance")
        elif operation == "Rotate":
            return gr.Slider(-90.0, 90.0, value=0.0, step=1.0, label="Rotation Angle (degrees)")
        elif operation == "Scale":
            return gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Scale Ratio")
    
    def generate_image(self, prompt, seed, num_inference_steps, batch_size, guidance_steps, current_image_dir):
        """Batch-generate images with ControlNet."""
        if not current_image_dir or not os.path.isdir(current_image_dir):
            return "Error: Could not find a valid render image directory. Please refresh or manipulate the scene first.", [], []

        # Locate the full_scene.exr file
        full_scene_exr_path = os.path.join(current_image_dir, "full_scene.exr")

        if not os.path.exists(full_scene_exr_path):
            return f"Error: Could not find full_scene.exr in directory {current_image_dir}.", [], []

        try:
            # Read the EXR image as the conditioning input
            condition_img_exr = openexr_numpy.imread(full_scene_exr_path, channel_names=[
                "ViewLayer.Combined.R",
                "ViewLayer.Combined.G", 
                "ViewLayer.Combined.B",
                "ViewLayer.Combined.A"
            ])

            condition_rgb = condition_img_exr[:, :, :3]
            condition_alpha = condition_img_exr[:, :, 3]
            alpha_mask = condition_alpha > 1e-5
            alpha_mask_3d = np.expand_dims(alpha_mask, axis=2)
            condition_rgb = np.where(alpha_mask_3d, condition_rgb, 0.0)

            # Duplicate the conditioning image batch_size times
            condition_tensor = torch.from_numpy(condition_rgb).permute(2, 0, 1).unsqueeze(0)
            condition_tensors = condition_tensor.repeat(batch_size, 1, 1, 1).to(self.global_config["device"], dtype=self.global_config["weight_dtype"])

            # Set the random seed
            generator = None
            if seed != -1:
                generator = torch.Generator(device=self.global_config["device"]).manual_seed(seed)
            
            # Do not pass guidance_steps when it's -1 or None to use the pipeline default
            guidance_steps_arg = guidance_steps if guidance_steps is not None and guidance_steps >= 0 else None

            with torch.autocast(self.global_config["device"]):
                results = self.pipeline(
                    prompt=[prompt] * batch_size,
                    control_image=condition_tensors,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_steps=guidance_steps_arg,
                )
            
            images = results.images if results.images else []
            return "Batch image generation succeeded", images, images 

        except Exception as e:
            return f"Error while generating images: {e}", [], []
    
    def random_transform_cube(self, cube_id):
        """Apply a random transform to the selected cube."""
        if not cube_id:
            return "Please select a cube first", gr.update(), [], None, []

        params = {"id": cube_id}
        status, cubes, image_list, image_dir = self.send_command("random_transform_cube", params)
        return status, gr.update(value=cube_id), image_list, image_dir, cubes
    
    def init_app(self):
        """Initialize the application state."""
        status, selector_update, image_list, image_dir, cubes = self.refresh_scene()
        return status, selector_update, image_list, image_dir, cubes
    
    def create_interface(self):
        """Build the Gradio interface."""
        with gr.Blocks() as app:
            gr.Markdown("# Blender Cube Real-Time Editor")
            
            # Store state for the current image directory and cube list
            current_image_dir_state = gr.State(None)
            current_cubes_state = gr.State([])
            generated_images_state = gr.State([])

            # Status display
            status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                # Left control panel
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("Cube Controls"):
                            # Add cube section
                            with gr.Group():
                                gr.Markdown("## Add New Cube")
                                cube_name_input = gr.Textbox(label="Cube Name (Optional)", placeholder="Enter a cube name; an ID will be generated if left blank", value="")
                                add_btn = gr.Button("Add Cube", variant="primary")
                            
                            # Cube selection and deletion section
                            with gr.Group():
                                gr.Markdown("## Select & Delete")
                                cube_selector = gr.Dropdown(label="Select Cube", choices=[], interactive=True)
                                delete_btn = gr.Button("Delete Selected Cube", variant="stop")
                            
                            # Cube transform section
                            with gr.Group():
                                gr.Markdown("## Transform Selected Cube")
                                operation = gr.Radio(["Translate", "Rotate", "Scale"], label="Operation Type", value="Translate")
                                axis = gr.Radio(["X", "Y", "Z"], label="Axis", value="X")
                                value = gr.Slider(-10.0, 10.0, value=0.0, step=0.1, label="Translation Distance")
                                transform_btn = gr.Button("Apply Transform")
                                random_transform_btn = gr.Button("Apply Random Transform", variant="secondary")
                            
                            # Refresh button
                            refresh_btn = gr.Button("Refresh Scene", variant="secondary")
                        
                        with gr.TabItem("Generation Settings"):
                            # ControlNet generation section
                            with gr.Group():
                                gr.Markdown("## ControlNet Image Generation")
                                
                                prompt = gr.Textbox(label="Prompt", value="A mud-splattered, dark green customized Jeep Wrangler Rubicon with oversized tires and a roof rack, navigating a rocky, treacherous mountain trail during a dramatic sunset, lens flare effect, highly detailed.", placeholder="Enter a descriptive prompt...", lines=4)
                                seed = gr.Slider(label="Random Seed", minimum=-1, maximum=2147483647, step=1, value=-1, 
                                               info="Set to -1 to randomize the seed")
                                steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, step=1, value=35)
                                
                                guidance_steps_slider = gr.Slider(
                                    label="Guidance Steps (ControlNet)",
                                    minimum=0,
                                    maximum=50,  
                                    step=1,  
                                    value=15,  
                                    info="Controls the number of ControlNet guidance steps"
                                )
                                batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=4)
                                generate_btn = gr.Button("Generate Images", variant="primary")
                
                # Right display panel
                with gr.Column(scale=2):
                    # Rendered results gallery (top)
                    render_gallery = gr.Gallery(label="Blender Render Results", columns=3, height="auto")
                    
                    # Generated results gallery (bottom)
                    generated_image_gallery = gr.Gallery(label="ControlNet Generation Results", columns=4, height="auto")
            
            # Update the slider range when the operation changes
            operation.change(
                fn=self.update_slider_range,
                inputs=[operation],
                outputs=[value]
            )
            
            # Event handlers
            add_btn.click(
                fn=self.add_cube,
                inputs=[cube_name_input],
                outputs=[status_text, cube_selector, render_gallery, current_image_dir_state, current_cubes_state]
            )
            
            delete_btn.click(
                fn=self.delete_cube,
                inputs=[cube_selector],
                outputs=[status_text, cube_selector, render_gallery, current_image_dir_state, current_cubes_state]
            )
            
            transform_btn.click(
                fn=self.transform_cube,
                inputs=[cube_selector, operation, axis, value],
                outputs=[status_text, cube_selector, render_gallery, current_image_dir_state, current_cubes_state]
            )
            
            random_transform_btn.click(
                fn=self.random_transform_cube,
                inputs=[cube_selector],
                outputs=[status_text, cube_selector, render_gallery, current_image_dir_state, current_cubes_state]
            )
            
            refresh_btn.click(
                fn=self.refresh_scene,
                inputs=[],
                outputs=[status_text, cube_selector, render_gallery, current_image_dir_state, current_cubes_state]
            )
            
            generate_btn.click(
                fn=self.generate_image,
                inputs=[prompt, seed, steps, batch_size, guidance_steps_slider, current_image_dir_state],
                outputs=[status_text, generated_image_gallery, generated_images_state]
            )
            
            # Initialize on page load
            app.load(
                fn=self.init_app,
                inputs=[],
                outputs=[status_text, cube_selector, render_gallery, current_image_dir_state, current_cubes_state]
            )
        
        return app

def run_blender(blender_path, blender_script):
    proc = subprocess.Popen([blender_path, "--background", "--python", blender_script])
    return proc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_path", type=str, default="render/blender/blender")
    parser.add_argument("--blender_script", type=str, default="render/blender_server.py")
    parser.add_argument("--base_path", type=str, default="checkpoints/stable-diffusion-3.5-medium")
    parser.add_argument("--controlnet_path", type=str, default="checkpoints/SceneDesigner/controlnet")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--weight_dtype", type=str, default="float16", choices=["float16", "float32"])
    
    args = parser.parse_args()

    # Clear all HTTP proxy settings
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)

    # Store a handle to the Blender process globally
    blender_process = None
    
    def cleanup():
        """Cleanup function to ensure the Blender process terminates correctly."""
        global blender_process
        if blender_process and blender_process.poll() is None:
            blender_process.kill()
            print("Blender process terminated")
    
    def signal_handler(signum, frame):
        cleanup()
        exit(0)
    
    # Register the cleanup function and signal handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    print("Loading Blender server...")
    blender_process = run_blender(args.blender_path, args.blender_script)

    print("Loading model...")
    # Set the weight data type
    weight_dtype = torch.float16
    
    controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=weight_dtype)

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.base_path,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
    )
    pipeline.to(args.device)

    # Build the global configuration
    global_config = {
        "device": args.device,
        "weight_dtype": weight_dtype,
        "port": args.port
    }
    
    print("Initializing UI...")
    # Create the UI instance
    ui = SceneDesignerUI(pipeline, global_config)
    app = ui.create_interface()
    
    try:
        print(f"Starting Gradio app on port {args.port}...")
        app.launch(server_name="0.0.0.0", server_port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
    

