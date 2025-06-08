from diffusers_helper.hf_login import login

import json
import os
import shutil
from pathlib import PurePath
import time
import argparse
import traceback
import einops
import numpy as np
import torch
import datetime

# Version information
from modules.version import APP_VERSION

# Set environment variables
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizers parallelism warning



import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper import lora_utils
from diffusers_helper.lora_utils import load_lora, unload_all_loras

# Import model generators
from modules.generators import create_model_generator

# Global cache for prompt embeddings
prompt_embedding_cache = {}
# Import from modules
from modules.video_queue import VideoJobQueue, JobStatus
from modules.prompt_handler import parse_timestamped_prompt
from modules.interface import create_interface, format_queue_status
from modules.settings import Settings
from modules.pipelines.metadata_utils import create_metadata
from modules.pipelines.worker import worker

# Try to suppress annoyingly persistent Windows asyncio proactor errors
if os.name == 'nt':  # Windows only
    import asyncio
    from functools import wraps
    
    # Replace the problematic proactor event loop with selector event loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Patch the base transport's close method
    def silence_event_loop_closed(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise
        return wrapper
    
    # Apply the patch
    if hasattr(asyncio.proactor_events._ProactorBasePipeTransport, '_call_connection_lost'):
        asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = silence_event_loop_closed(
            asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost)
            
# ADDED: Debug function to verify LoRA state
def verify_lora_state(transformer, label=""):
    """Debug function to verify the state of LoRAs in a transformer model"""
    if transformer is None:
        print(f"[{label}] Transformer is None, cannot verify LoRA state")
        return
        
    has_loras = False
    if hasattr(transformer, 'peft_config'):
        adapter_names = list(transformer.peft_config.keys()) if transformer.peft_config else []
        if adapter_names:
            has_loras = True
            print(f"[{label}] Transformer has LoRAs: {', '.join(adapter_names)}")
        else:
            print(f"[{label}] Transformer has no LoRAs in peft_config")
    else:
        print(f"[{label}] Transformer has no peft_config attribute")
        
    # Check for any LoRA modules
    for name, module in transformer.named_modules():
        if hasattr(module, 'lora_A') and module.lora_A:
            has_loras = True
            # print(f"[{label}] Found lora_A in module {name}")
        if hasattr(module, 'lora_B') and module.lora_B:
            has_loras = True
            # print(f"[{label}] Found lora_B in module {name}")
            
    if not has_loras:
        print(f"[{label}] No LoRA components found in transformer")


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lora", type=str, default=None, help="Lora path (comma separated for multiple)")
parser.add_argument("--offline", action='store_true', help="Run in offline mode")
args = parser.parse_args()

print(args)

if args.offline:
    print("Offline mode enabled.")
    os.environ['HF_HUB_OFFLINE'] = '1'
else:
    if 'HF_HUB_OFFLINE' in os.environ:
        del os.environ['HF_HUB_OFFLINE']

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Load models
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

# Initialize model generator placeholder
current_generator = None # Will hold the currently active model generator

# Load models based on VRAM availability later
 
# Configure models
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()

if not high_vram:
   vae.enable_slicing()
   vae.enable_tiling()


vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)

# Create lora directory if it doesn't exist
lora_dir = os.path.join(os.path.dirname(__file__), 'loras')
os.makedirs(lora_dir, exist_ok=True)

# Initialize LoRA support - moved scanning after settings load
lora_names = []
lora_values = [] # This seems unused for population, might be related to weights later

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define default LoRA folder path relative to the script directory (used if setting is missing)
default_lora_folder = os.path.join(script_dir, "loras")
os.makedirs(default_lora_folder, exist_ok=True) # Ensure default exists

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# Initialize settings
settings = Settings()

# --- Populate LoRA names AFTER settings are loaded ---
lora_folder_from_settings: str = settings.get("lora_dir", default_lora_folder) # Use setting, fallback to default
print(f"Scanning for LoRAs in: {lora_folder_from_settings}")
if os.path.isdir(lora_folder_from_settings):
    try:
        for root, _, files in os.walk(lora_folder_from_settings):
            for file in files:
                if file.endswith('.safetensors') or file.endswith('.pt'):
                    lora_relative_path = os.path.relpath(os.path.join(root, file), lora_folder_from_settings)
                    lora_name = str(PurePath(lora_relative_path).with_suffix(''))
                    lora_names.append(lora_name)
        print(f"Found LoRAs: {lora_names}")
        # Temp solution for only 1 lora
        if len(lora_names) == 1 :
            lora_names.append("dummy lora - DO NOT USE - will disappear with multiple loras")        
    except Exception as e:
        print(f"Error scanning LoRA directory '{lora_folder_from_settings}': {e}")
else:
    print(f"LoRA directory not found: {lora_folder_from_settings}")
# --- End LoRA population ---


# Create job queue
job_queue = VideoJobQueue()



# Function to load a LoRA file
def load_lora_file(lora_file: str | PurePath):
    if not lora_file:
        return None, "No file selected"
    
    try:
        # Get the filename from the path
        lora_path = PurePath(lora_file)
        lora_name = lora_path.name
        
        # Copy the file to the lora directory
        lora_dest = PurePath(lora_dir, lora_path)
        import shutil
        shutil.copy(lora_file, lora_dest)
        
        # Load the LoRA
        global current_generator, lora_names
        if current_generator is None:
            return None, "Error: No model loaded to apply LoRA to. Generate something first."
        
        # Unload any existing LoRAs first
        current_generator.unload_loras()
        
        # Load the single LoRA
        selected_loras = [lora_path.stem]
        current_generator.load_loras(selected_loras, lora_dir, selected_loras)
        
        # Add to lora_names if not already there
        lora_base_name = lora_path.stem
        if lora_base_name not in lora_names:
            lora_names.append(lora_base_name)
        
        # Get the current device of the transformer
        device = next(current_generator.transformer.parameters()).device
        
        # Move all LoRA adapters to the same device as the base model
        current_generator.move_lora_adapters_to_device(device)
        
        print(f"Loaded LoRA: {lora_name} to {current_generator.get_model_name()} model")
        
        return gr.update(choices=lora_names), f"Successfully loaded LoRA: {lora_name}"
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return None, f"Error loading LoRA: {e}"

@torch.no_grad()
def get_cached_or_encode_prompt(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, target_device):
    """
    Retrieves prompt embeddings from cache or encodes them if not found.
    Stores encoded embeddings (on CPU) in the cache.
    Returns embeddings moved to the target_device.
    """
    if prompt in prompt_embedding_cache:
        print(f"Cache hit for prompt: {prompt[:60]}...")
        llama_vec_cpu, llama_mask_cpu, clip_l_pooler_cpu = prompt_embedding_cache[prompt]
        # Move cached embeddings (from CPU) to the target device
        llama_vec = llama_vec_cpu.to(target_device)
        llama_attention_mask = llama_mask_cpu.to(target_device) if llama_mask_cpu is not None else None
        clip_l_pooler = clip_l_pooler_cpu.to(target_device)
        return llama_vec, llama_attention_mask, clip_l_pooler
    else:
        print(f"Cache miss for prompt: {prompt[:60]}...")
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        # Store CPU copies in cache
        prompt_embedding_cache[prompt] = (llama_vec.cpu(), llama_attention_mask.cpu() if llama_attention_mask is not None else None, clip_l_pooler.cpu())
        # Return embeddings already on the target device (as encode_prompt_conds uses the model's device)
        return llama_vec, llama_attention_mask, clip_l_pooler

# Set the worker function for the job queue - using the imported worker from modules/pipelines/worker.py
job_queue.set_worker_function(worker)


def process(
        model_type,
        input_image,
        end_frame_image,     # NEW
        end_frame_strength,  # NEW        
        prompt_text,
        n_prompt,
        seed, 
        total_second_length, 
        latent_window_size, 
        steps, 
        cfg, 
        gs, 
        rs, 
        use_teacache, 
        teacache_num_steps, 
        teacache_rel_l1_thresh,
        blend_sections, 
        latent_type,
        clean_up_videos,
        selected_loras,
        resolutionW,
        resolutionH,
        input_image_path,
        combine_with_source,
        num_cleaned_frames,
        *lora_args,
        save_metadata_checked=True,  # NEW: Parameter to control metadata saving
    ):
    
    # Create a blank black image if no 
    # Create a default image based on the selected latent_type
    has_input_image = True
    if input_image is None:
        has_input_image = False
        default_height, default_width = resolutionH, resolutionW
        if latent_type == "White":
            # Create a white image
            input_image = np.ones((default_height, default_width, 3), dtype=np.uint8) * 255
            print("No input image provided. Using a blank white image.")

        elif latent_type == "Noise":
            # Create a noise image
            input_image = np.random.randint(0, 256, (default_height, default_width, 3), dtype=np.uint8)
            print("No input image provided. Using a random noise image.")

        elif latent_type == "Green Screen":
            # Create a green screen image with standard chroma key green (0, 177, 64)
            input_image = np.zeros((default_height, default_width, 3), dtype=np.uint8)
            input_image[:, :, 1] = 177  # Green channel
            input_image[:, :, 2] = 64   # Blue channel
            # Red channel remains 0
            print("No input image provided. Using a standard chroma key green screen.")

        else:  # Default to "Black" or any other value
            # Create a black image
            input_image = np.zeros((default_height, default_width, 3), dtype=np.uint8)
            print(f"No input image provided. Using a blank black image (latent_type: {latent_type}).")

    
    # Handle input files - copy to input_files_dir to prevent them from being deleted by temp cleanup
    input_files_dir = settings.get("input_files_dir")
    os.makedirs(input_files_dir, exist_ok=True)
    
    # Process input image (if it's a file path)
    input_image_path = None
    if isinstance(input_image, str) and os.path.exists(input_image):
        # It's a file path, copy it to input_files_dir
        filename = os.path.basename(input_image)
        input_image_path = os.path.join(input_files_dir, f"{generate_timestamp()}_{filename}")
        try:
            shutil.copy2(input_image, input_image_path)
            print(f"Copied input image to {input_image_path}")
            # For Video model, we'll use the path
            if model_type == "Video":
                input_image = input_image_path
        except Exception as e:
            print(f"Error copying input image: {e}")
    
    # Process end frame image (if it's a file path)
    end_frame_image_path = None
    if isinstance(end_frame_image, str) and os.path.exists(end_frame_image):
        # It's a file path, copy it to input_files_dir
        filename = os.path.basename(end_frame_image)
        end_frame_image_path = os.path.join(input_files_dir, f"{generate_timestamp()}_{filename}")
        try:
            shutil.copy2(end_frame_image, end_frame_image_path)
            print(f"Copied end frame image to {end_frame_image_path}")
        except Exception as e:
            print(f"Error copying end frame image: {e}")
    
    # Extract lora_loaded_names from lora_args
    lora_loaded_names = lora_args[0] if lora_args and len(lora_args) > 0 else []
    lora_values = lora_args[1:] if lora_args and len(lora_args) > 1 else []
    
    # Create job parameters
    job_params = {
        'model_type': model_type,
        'input_image': input_image.copy() if hasattr(input_image, 'copy') else input_image,  # Handle both image arrays and video paths
        'end_frame_image': end_frame_image.copy() if end_frame_image is not None else None,
        'end_frame_strength': end_frame_strength,        
        'prompt_text': prompt_text,
        'n_prompt': n_prompt,
        'seed': seed,
        'total_second_length': total_second_length,
        'latent_window_size': latent_window_size,
        'latent_type': latent_type,
        'steps': steps,
        'cfg': cfg,
        'gs': gs,
        'rs': rs,
        'blend_sections': blend_sections,
        'use_teacache': use_teacache,
        'teacache_num_steps': teacache_num_steps,
        'teacache_rel_l1_thresh': teacache_rel_l1_thresh,
        'selected_loras': selected_loras,
        'has_input_image': has_input_image,
        'output_dir': settings.get("output_dir"),
        'metadata_dir': settings.get("metadata_dir"),
        'input_files_dir': input_files_dir,  # Add input_files_dir to job parameters
        'input_image_path': input_image_path,  # Add the path to the copied input image
        'end_frame_image_path': end_frame_image_path,  # Add the path to the copied end frame image
        'resolutionW': resolutionW, # Add resolution parameter
        'resolutionH': resolutionH,
        'lora_loaded_names': lora_loaded_names,
        'combine_with_source': combine_with_source,  # Add combine_with_source parameter
        'num_cleaned_frames': num_cleaned_frames,
        'save_metadata_checked': save_metadata_checked,  # NEW: Add save_metadata_checked parameter
    }
    
    # Print teacache parameters for debugging
    print(f"Teacache parameters: use_teacache={use_teacache}, teacache_num_steps={teacache_num_steps}, teacache_rel_l1_thresh={teacache_rel_l1_thresh}")
    
    # Add LoRA values if provided - extract them from the tuple
    if lora_values:
        # Convert tuple to list
        lora_values_list = list(lora_values)
        job_params['lora_values'] = lora_values_list
    
    # Add job to queue
    job_id = job_queue.add_job(job_params)
    
    # Set the generation_type attribute on the job object directly
    job = job_queue.get_job(job_id)
    if job:
        job.generation_type = model_type  # Set generation_type to model_type for display in queue
    print(f"Added job {job_id} to queue")
    
    queue_status = update_queue_status()
    # Return immediately after adding to queue
    # Return separate updates for start_button and end_button to prevent cross-contamination
    return None, job_id, None, '', f'Job added to queue. Job ID: {job_id}', gr.update(value="Add to Queue", interactive=True), gr.update(value="Cancel Current Job", interactive=True)



def end_process():
    """Cancel the current running job and update the queue status"""
    print("Cancelling current job")
    with job_queue.lock:
        if job_queue.current_job:
            job_id = job_queue.current_job.id
            print(f"Cancelling job {job_id}")

            # Send the end signal to the job's stream
            if job_queue.current_job.stream:
                job_queue.current_job.stream.input_queue.push('end')
                
            # Mark the job as cancelled
            job_queue.current_job.status = JobStatus.CANCELLED
            job_queue.current_job.completed_at = time.time()  # Set completion time
    
    # Force an update to the queue status
    return update_queue_status()


def update_queue_status():
    """Update queue status and refresh job positions"""
    jobs = job_queue.get_all_jobs()
    for job in jobs:
        if job.status == JobStatus.PENDING:
            job.queue_position = job_queue.get_queue_position(job.id)
    
    # Make sure to update current running job info
    if job_queue.current_job:
        # Make sure the running job is showing status = RUNNING
        job_queue.current_job.status = JobStatus.RUNNING
    
    # Update the toolbar stats
    pending_count = 0
    running_count = 0
    completed_count = 0
    
    for job in jobs:
        if hasattr(job, 'status'):
            status = str(job.status)
            if status == "JobStatus.PENDING":
                pending_count += 1
            elif status == "JobStatus.RUNNING":
                running_count += 1
            elif status == "JobStatus.COMPLETED":
                completed_count += 1
    
    return format_queue_status(jobs)


def monitor_job(job_id=None):
    """
    Monitor a specific job and update the UI with the latest video segment as soon as it's available.
    If no job_id is provided, check if there's a current job in the queue.
    ALWAYS shows the current running job, regardless of the job_id provided.
    """
    last_video = None  # Track the last video file shown
    last_job_status = None  # Track the previous job status to detect status changes
    last_progress_update_time = time.time()  # Track when we last updated the progress
    last_preview = None  # Track the last preview image shown
    force_update = True  # Force an update on first iteration
    
    # Flag to indicate we're waiting for a job transition
    waiting_for_transition = False
    transition_start_time = None
    max_transition_wait = 5.0  # Maximum time to wait for transition in seconds

    while True:
        # ALWAYS check if there's a current running job that's different from our tracked job_id
        with job_queue.lock:
            current_job = job_queue.current_job
            if current_job and current_job.id != job_id and current_job.status == JobStatus.RUNNING:
                # Always switch to the current running job
                job_id = current_job.id
                waiting_for_transition = False
                force_update = True
                # Yield a temporary update to show we're switching jobs
                yield last_video, gr.update(visible=True), '', 'Switching to current job...', gr.update(interactive=True), gr.update(value="Cancel Current Job", visible=True)
                continue
                
        # Check if we're waiting for a job transition
        if waiting_for_transition:
            current_time = time.time()
            # If we've been waiting too long, stop waiting
            if current_time - transition_start_time > max_transition_wait:
                waiting_for_transition = False
                
                # Check one more time for a current job
                with job_queue.lock:
                    current_job = job_queue.current_job
                    if current_job and current_job.status == JobStatus.RUNNING:
                        # Switch to whatever job is currently running
                        job_id = current_job.id
                        force_update = True
                        yield last_video, gr.update(visible=True), '', 'Switching to current job...', gr.update(interactive=True), gr.update(value="Cancel Current Job", visible=True)
                        continue
            else:
                # If still waiting, sleep briefly and continue
                time.sleep(0.1)
                continue

        job = job_queue.get_job(job_id)
        if not job:
            # Correctly yield 6 items for the startup/no-job case
            # This ensures the status text goes to the right component and the buttons are set correctly.
            yield None, None, 'No job ID provided', '', gr.update(value="Add to Queue", interactive=True, visible=True), gr.update(interactive=False, visible=False)
            return

        # If a new video file is available, yield it immediately
        if job.result and job.result != last_video:
            last_video = job.result
            # You can also update preview/progress here if desired
            yield last_video, gr.update(visible=True), '', '', gr.update(interactive=True), gr.update(interactive=True)

        # Handle job status and progress
        if job.status == JobStatus.PENDING:
            position = job_queue.get_queue_position(job_id)
            yield last_video, gr.update(visible=True), '', f'Waiting in queue. Position: {position}', gr.update(interactive=True), gr.update(interactive=True)

        elif job.status == JobStatus.RUNNING:
            # Only reset the cancel button when a job transitions from another state to RUNNING
            # This ensures we don't reset the button text during cancellation
            if last_job_status != JobStatus.RUNNING:
                # Check if the button text is already "Cancelling..." - if so, don't change it
                # This prevents the button from changing back to "Cancel Current Job" during cancellation
                button_update = gr.update(interactive=True, value="Cancel Current Job", visible=True)
            else:
                # Keep current text and state - important to not override "Cancelling..." text
                button_update = gr.update(interactive=True, visible=True)
                
            # Check if we have progress data and if it's time to update
            current_time = time.time()
            update_needed = force_update or (current_time - last_progress_update_time > 0.05)  # More frequent updates
            
            # Always check for progress data, even if we don't have a preview yet
            if job.progress_data and update_needed:
                preview = job.progress_data.get('preview')
                desc = job.progress_data.get('desc', '')
                html = job.progress_data.get('html', '')
                
                # Only update the preview if it has changed or we're forcing an update
                # Ensure all components get an update
                current_preview_value = job.progress_data.get('preview') if job.progress_data else None
                current_desc_value = job.progress_data.get('desc', 'Processing...') if job.progress_data else 'Processing...'
                current_html_value = job.progress_data.get('html', make_progress_bar_html(0, 'Processing...')) if job.progress_data else make_progress_bar_html(0, 'Processing...')

                if current_preview_value is not None and (current_preview_value is not last_preview or force_update):
                    last_preview = current_preview_value
                # Always update if force_update is true, or if it's time for a periodic update
                if force_update or update_needed:
                    last_progress_update_time = current_time
                    force_update = False
                    yield job.result, last_preview, current_desc_value, current_html_value, gr.update(interactive=True), button_update
            
            # Fallback for periodic update if no new progress data but job is still running
            elif current_time - last_progress_update_time > 0.5: # More frequent fallback update
                last_progress_update_time = current_time
                force_update = False # Reset force_update after a yield
                current_desc_value = job.progress_data.get('desc', 'Processing...') if job.progress_data else 'Processing...'
                current_html_value = job.progress_data.get('html', make_progress_bar_html(0, 'Processing...')) if job.progress_data else make_progress_bar_html(0, 'Processing...')
                yield job.result, last_preview, current_desc_value, current_html_value, gr.update(interactive=True), button_update

        elif job.status == JobStatus.COMPLETED:
            # Show the final video and reset the button text
            yield job.result, last_preview, 'Completed', make_progress_bar_html(100, 'Completed'), gr.update(value="Add to Queue"), gr.update(interactive=True, value="Cancel Current Job", visible=False)
            break

        elif job.status == JobStatus.FAILED:
            # Show error and reset the button text
            yield job.result, last_preview, f'Error: {job.error}', make_progress_bar_html(0, 'Failed'), gr.update(value="Add to Queue"), gr.update(interactive=True, value="Cancel Current Job", visible=False)
            break

        elif job.status == JobStatus.CANCELLED:
            # Show cancelled message and reset the button text
            yield job.result, last_preview, 'Job cancelled', make_progress_bar_html(0, 'Cancelled'), gr.update(interactive=True), gr.update(interactive=True, value="Cancel Current Job", visible=False)
            break

        # Update last_job_status for the next iteration
        last_job_status = job.status
        
        # Wait a bit before checking again
        time.sleep(0.05)  # Reduced wait time for more responsive updates


# Set Gradio temporary directory from settings
os.environ["GRADIO_TEMP_DIR"] = settings.get("gradio_temp_dir")

# Create the interface
interface = create_interface(
    process_fn=process,
    monitor_fn=monitor_job,
    end_process_fn=end_process,
    update_queue_status_fn=update_queue_status,
    load_lora_file_fn=load_lora_file,
    job_queue=job_queue,
    settings=settings,
    lora_names=lora_names # Explicitly pass the found LoRA names
)

# Launch the interface
interface.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
    allowed_paths=[settings.get("output_dir"), settings.get("metadata_dir")],
)
