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

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

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
    # DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
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
        
@torch.no_grad()
def worker(
    model_type,
    input_image,
    end_frame_image,     # NEW: The end frame image (numpy array or None)
    end_frame_strength,  # NEW: Influence of the end frame
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
    selected_loras,
    has_input_image,
    lora_values=None, 
    job_stream=None,
    output_dir=None,
    metadata_dir=None,
    input_files_dir=None,  # Add input_files_dir parameter
    input_image_path=None,  # Add input_image_path parameter
    end_frame_image_path=None,  # Add end_frame_image_path parameter
    resolutionW=640,  # Add resolution parameter with default value
    resolutionH=640,
    lora_loaded_names=[]
):
    global high_vram, current_generator, args
    
    # Ensure any existing LoRAs are unloaded from the current generator
    if current_generator is not None:
        print("Unloading any existing LoRAs before starting new job")
        current_generator.unload_loras()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    stream_to_use = job_stream if job_stream is not None else stream

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # --- Total progress tracking ---
    total_steps = total_latent_sections * steps  # Total diffusion steps over all segments
    step_durations = []  # Rolling history of recent step durations for ETA
    last_step_time = time.time()

    # Parse the timestamped prompt with boundary snapping and reversing
    # prompt_text should now be the original string from the job queue
    prompt_sections = parse_timestamped_prompt(prompt_text, total_second_length, latent_window_size, model_type)
    job_id = generate_timestamp()

    stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        if not high_vram:
            # Unload everything *except* the potentially active transformer
            # unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae)
            if current_generator is not None and current_generator.transformer is not None:
                # offload_model_from_device_for_memory_preservation(current_generator.transformer, target_device=gpu, preserved_memory_gb=8)
                pass # Keep transformer on GPU

        # --- Model Loading / Switching ---
        print(f"Worker starting for model type: {model_type}")
        
        # Create the appropriate model generator
        new_generator = create_model_generator(
            model_type,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=vae,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            high_vram=high_vram,
            prompt_embedding_cache=prompt_embedding_cache,
            offline=args.offline,
            settings=settings
        )
        
        # Update the global generator
        current_generator = new_generator
        
        # Load the transformer model
        current_generator.load_model()
        
        # Ensure the model has no LoRAs loaded
        print(f"Ensuring {model_type} model has no LoRAs loaded")
        current_generator.unload_loras()

        # Pre-encode all prompts
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding all prompts...'))))

        if not high_vram:
            # fake_diffusers_current_device(text_encoder, gpu) # Commented out as per request
            load_model_as_complete(text_encoder_2, target_device=gpu)

        # PROMPT BLENDING: Pre-encode all prompts and store in a list in order
        unique_prompts = []
        for section in prompt_sections:
            if section.prompt not in unique_prompts:
                unique_prompts.append(section.prompt)

        encoded_prompts = {}
        for prompt in unique_prompts:
            # Use the helper function for caching and encoding
            llama_vec, llama_attention_mask, clip_l_pooler = get_cached_or_encode_prompt(
                prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, gpu
            )
            encoded_prompts[prompt] = (llama_vec, llama_attention_mask, clip_l_pooler)

        # PROMPT BLENDING: Build a list of (start_section_idx, prompt) for each prompt
        prompt_change_indices = []
        last_prompt = None
        for idx, section in enumerate(prompt_sections):
            if section.prompt != last_prompt:
                prompt_change_indices.append((idx, section.prompt))
                last_prompt = section.prompt

        # Encode negative prompt
        if cfg == 1:
            llama_vec_n, llama_attention_mask_n, clip_l_pooler_n = (
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][0]),
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][1]),
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][2])
            )
        else:
             # Use the helper function for caching and encoding negative prompt
            # Ensure n_prompt is a string
            n_prompt_str = str(n_prompt) if n_prompt is not None else ""
            llama_vec_n, llama_attention_mask_n, clip_l_pooler_n = get_cached_or_encode_prompt(
                n_prompt_str, text_encoder, text_encoder_2, tokenizer, tokenizer_2, gpu
            )

        # Processing input image or video
        if model_type == "Video":
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Video processing ...'))))
            # For Video model, input_image is actually a video path
            # We'll handle the video processing in the VideoModelGenerator
            # Just set default values for now
            # Ensure resolutionH and resolutionW are integers
            resH = int(resolutionH) if isinstance(resolutionH, (int, float)) else 640
            resW = int(resolutionW) if isinstance(resolutionW, (int, float)) else 640
            height, width = find_nearest_bucket(resH, resW, resolution=(resH+resW)/2)
            input_image_np = None  # Will be set by the VideoModelGenerator
            
            if settings.get("save_metadata"):
                metadata_dict = {
                    "prompt": prompt_text, # Use the original string
                    "seed": seed,
                    "total_second_length": total_second_length,
                    "steps": steps,
                    "cfg": cfg,
                    "gs": gs,
                    "rs": rs,
                    "latent_type": latent_type,
                    "blend_sections": blend_sections,
                    "latent_window_size": latent_window_size,
                    "timestamp": time.time(),
                    "resolutionW": resolutionW,
                    "resolutionH": resolutionH,
                    "model_type": model_type,
                    "video_path": input_image  # Save the video path
                }
                
                # Create a placeholder image for the video
                placeholder_img = Image.new('RGB', (width, height), (0, 0, 128))  # Blue for video
                placeholder_img.save(os.path.join(metadata_dir, f'{job_id}.png'))
                
                with open(os.path.join(metadata_dir, f'{job_id}.json'), 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
        else:
            # Regular image processing
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

            H, W, _ = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=resolutionW if has_input_image else (resolutionH+resolutionW)/2)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            if settings.get("save_metadata"):
                metadata = PngInfo()
                # prompt_text should be a string here now
                metadata.add_text("prompt", prompt_text)
                metadata.add_text("seed", str(seed))
                Image.fromarray(input_image_np).save(os.path.join(metadata_dir, f'{job_id}.png'), pnginfo=metadata)

                metadata_dict = {
                    "prompt": prompt_text, # Use the original string
                    "seed": seed,
                    "total_second_length": total_second_length,
                    "steps": steps,
                    "cfg": cfg,
                    "gs": gs,
                    "rs": rs,
                    "latent_type" : latent_type,
                    "blend_sections": blend_sections,
                    "latent_window_size": latent_window_size,
                    "timestamp": time.time(),
                    "resolutionW": resolutionW,  # Add resolution to metadata
                    "resolutionH": resolutionH,
                    "model_type": model_type,  # Add model type to metadata
                    "end_frame_strength": end_frame_strength if end_frame_image is not None else None,
                    "end_frame_used": True if end_frame_image is not None else False,                    
                }
            # Add LoRA information to metadata if LoRAs are used and metadata saving is enabled
            if settings.get("save_metadata"):
                def ensure_list(x):
                    if isinstance(x, list):
                        return x
                    elif x is None:
                        return []
                    else:
                        return [x]

                selected_loras = ensure_list(selected_loras)
                lora_values = ensure_list(lora_values)

                if selected_loras and len(selected_loras) > 0:
                    lora_data = {}
                    for lora_name in selected_loras:
                        try:
                            idx = lora_loaded_names.index(lora_name)
                            weight = lora_values[idx] if lora_values and idx < len(lora_values) else 1.0
                            if isinstance(weight, list):
                                weight_value = weight[0] if weight and len(weight) > 0 else 1.0
                            else:
                                weight_value = weight
                            lora_data[lora_name] = float(weight_value)
                        except ValueError:
                            lora_data[lora_name] = 1.0
                    metadata_dict["loras"] = lora_data

                    with open(os.path.join(metadata_dir, f'{job_id}.json'), 'w') as f:
                        json.dump(metadata_dict, f, indent=2)
                else:
                    # Always save metadata even if no LoRAs are used
                    with open(os.path.join(metadata_dir, f'{job_id}.json'), 'w') as f:
                        json.dump(metadata_dict, f, indent=2)
                    
                    Image.fromarray(input_image_np).save(os.path.join(metadata_dir, f'{job_id}.png'))

        # Process video input for Video model
        if model_type == "Video":
            # For Video model, we'll handle the video processing in the VideoModelGenerator
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Video encoding ...'))))
            
            # Encode the video using the VideoModelGenerator
            # Get input_files_dir from settings
            input_files_dir = settings.get("input_files_dir")
            
            # Encode the video using the VideoModelGenerator
            start_latent, input_image_np, video_latents, fps, height, width, input_video_pixels = current_generator.video_encode(
                video_path=input_image,
                resolution=resolutionW,
                no_resize=False,
                vae_batch_size=16,
                device=gpu,
                input_files_dir=input_files_dir
            )
            
            # CLIP Vision encoding for the first frame
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
            
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)
                
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            
            # Store the input video pixels and latents for later use
            input_video_pixels = input_video_pixels.cpu()
            video_latents = video_latents.cpu()
            
            # For Video model, we need to ensure the generation starts from the end of the input video
            # First, store the video latents
            video_latents_cpu = video_latents.cpu()
            
            # Store the last frame of the video latents as start_latent for the model
            start_latent = video_latents_cpu[:, :, -1:].cpu()
            print(f"Using last frame of input video as start_latent. Shape: {start_latent.shape}")
            
            # Initialize history_latents with the entire video latents
            # This provides full context for the model to generate a coherent continuation
            # We'll use the last frame as the starting point for generation
            history_latents = current_generator.prepare_history_latents(height, width)
            
            # Copy the video latents into the history_latents tensor
            # This ensures the model has access to the full video context
            video_frames = video_latents_cpu.shape[2]
            if video_frames > 0:
                # Calculate how many frames we can copy (limited by history_latents size)
                max_frames = min(video_frames, history_latents.shape[2] - 1)  # Leave room for start_latent
                if max_frames > 0:
                    # Copy the last max_frames frames from the video latents
                    history_latents[:, :, 1:max_frames+1, :, :] = video_latents_cpu[:, :, -max_frames:, :, :]
                    print(f"Copied {max_frames} frames from video latents to history_latents")
                
                # Always put the last frame at position 0 (this is what the model will extend from)
                history_latents[:, :, 0:1, :, :] = start_latent
                print(f"Placed last frame of video at position 0 in history_latents")
            
            print(f"Initialized history_latents with video context. Shape: {history_latents.shape}")
            
            # Initialize total_generated_latent_frames for Video model
            # For Video model, we start with 0 since we'll be adding to the end of the video
            total_generated_latent_frames = 0
            
            # Store the number of frames in the input video for later use
            input_video_frame_count = video_latents.shape[2]
        else:
            # Regular image processing
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, vae)
            if torch.isinf(start_latent).any() or torch.isnan(start_latent).any():
                print(f"Warning: start_latent contains NaN/inf values. Clamping to a safe range. Min: {start_latent.min()}, Max: {start_latent.max()}")
                start_latent = torch.nan_to_num(start_latent, nan=0.0, posinf=20.0, neginf=-20.0)
                print(f"After clamping: Min: {start_latent.min()}, Max: {start_latent.max()}")

            # CLIP Vision
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # VAE encode end_frame_image if provided
        end_frame_latent = None
        if model_type == "Original with Endframe" and end_frame_image is not None:
            print(f"Processing end frame for {model_type} model...")
            if not isinstance(end_frame_image, np.ndarray):
                print(f"Warning: end_frame_image is not a numpy array (type: {type(end_frame_image)}). Attempting conversion or skipping.")
                try:
                    end_frame_image = np.array(end_frame_image)
                except Exception as e_conv:
                    print(f"Could not convert end_frame_image to numpy array: {e_conv}. Skipping end frame.")
                    end_frame_image = None
            
            if end_frame_image is not None:
                # Use the main job's target width/height (bucket dimensions) for the end frame
                # `width` and `height` should be available from the start_image processing
                end_frame_np = resize_and_center_crop(end_frame_image, target_width=width, target_height=height)
                
                if settings.get("save_metadata"): # Save processed end frame for debugging
                     Image.fromarray(end_frame_np).save(os.path.join(metadata_dir, f'{job_id}_end_frame_processed.png'))
                
                end_frame_pt = torch.from_numpy(end_frame_np).float() / 127.5 - 1
                end_frame_pt = end_frame_pt.permute(2, 0, 1)[None, :, None] # VAE expects [B, C, F, H, W]
                
                if not high_vram: load_model_as_complete(vae, target_device=gpu) # Ensure VAE is loaded
                end_frame_latent = vae_encode(end_frame_pt, vae)
                if end_frame_latent is not None: # Ensure it was created
                    if torch.isinf(end_frame_latent).any() or torch.isnan(end_frame_latent).any():
                        print(f"Warning: end_frame_latent contains NaN/inf values. Clamping to a safe range. Min: {end_frame_latent.min()}, Max: {end_frame_latent.max()}")
                        end_frame_latent = torch.nan_to_num(end_frame_latent, nan=0.0, posinf=20.0, neginf=-20.0)
                        print(f"After clamping: Min: {end_frame_latent.min()}, Max: {end_frame_latent.max()}")
                print("End frame VAE encoded.")
                # VAE will be offloaded later if not high_vram, after prompt dtype conversions.
        
        if not high_vram: # Offload VAE and image_encoder if they were loaded
            # offload_model_from_device_for_memory_preservation(vae, target_device=gpu, preserved_memory_gb=settings.get("gpu_memory_preservation"))
            # offload_model_from_device_for_memory_preservation(image_encoder, target_device=gpu, preserved_memory_gb=settings.get("gpu_memory_preservation"))
            pass # Keep VAE and image_encoder on GPU
        
        # Dtype
        for prompt_key in encoded_prompts:
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[prompt_key]
            llama_vec = llama_vec.to(current_generator.transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(current_generator.transformer.dtype)
            encoded_prompts[prompt_key] = (llama_vec, llama_attention_mask, clip_l_pooler)

        llama_vec_n = llama_vec_n.to(current_generator.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(current_generator.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(current_generator.transformer.dtype)

        # Sampling
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        # Initialize history latents based on model type
        if model_type != "Video":  # Skip for Video model as we already initialized it
            history_latents = current_generator.prepare_history_latents(height, width)
            
            # For F1 model, initialize with start latent
            if model_type == "F1":
                history_latents = current_generator.initialize_with_start_latent(history_latents, start_latent)
                total_generated_latent_frames = 1  # Start with 1 for F1 model since it includes the first frame
            elif model_type == "Original" or model_type == "Original with Endframe":
                total_generated_latent_frames = 0

        history_pixels = None
        
        # Get latent paddings from the generator
        latent_paddings = current_generator.get_latent_paddings(total_latent_sections)

        # PROMPT BLENDING: Track section index
        section_idx = 0

        # Load LoRAs if selected
        if selected_loras:
            current_generator.load_loras(selected_loras, lora_folder_from_settings, lora_loaded_names, lora_values)

        # --- Callback for progress ---
        def callback(d):
            nonlocal last_step_time, step_durations
            
            # Check for cancellation signal
            if stream_to_use.input_queue.top() == 'end':
                print("Cancellation signal detected in callback")
                return 'cancel'  # Return a signal that will be checked in the sampler
                
            now_time = time.time()
            # Record duration between diffusion steps (skip first where duration may include setup)
            if last_step_time is not None:
                step_delta = now_time - last_step_time
                if step_delta > 0:
                    step_durations.append(step_delta)
                    if len(step_durations) > 30:  # Keep only recent 30 steps
                        step_durations.pop(0)
            last_step_time = now_time
            avg_step = sum(step_durations) / len(step_durations) if step_durations else 0.0

            preview = d['denoised']
            preview = vae_decode_fake(preview)
            preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

            # --- Progress & ETA logic ---
            # Current segment progress
            current_step = d['i'] + 1
            percentage = int(100.0 * current_step / steps)

            # Total progress
            total_steps_done = section_idx * steps + current_step
            total_percentage = int(100.0 * total_steps_done / total_steps)

            # ETA calculations
            def fmt_eta(sec):
                try:
                    return str(datetime.timedelta(seconds=int(sec)))
                except Exception:
                    return "--:--"

            segment_eta = (steps - current_step) * avg_step if avg_step else 0
            total_eta = (total_steps - total_steps_done) * avg_step if avg_step else 0

            segment_hint = f'Sampling {current_step}/{steps}  ETA {fmt_eta(segment_eta)}'
            total_hint = f'Total {total_steps_done}/{total_steps}  ETA {fmt_eta(total_eta)}'

            # For Video model, add the input video frame count when calculating current position
            if model_type == "Video":
                # Calculate the time position including the input video frames
                input_video_time = input_video_frame_count * 4 / 30  # Convert latent frames to time
                current_pos = input_video_time + (total_generated_latent_frames * 4 - 3) / 30
                # Original position is the remaining time to generate
                original_pos = total_second_length - (total_generated_latent_frames * 4 - 3) / 30
            else:
                # For other models, calculate as before
                current_pos = (total_generated_latent_frames * 4 - 3) / 30
                original_pos = total_second_length - current_pos
            
            # Ensure positions are not negative
            if current_pos < 0: current_pos = 0
            if original_pos < 0: original_pos = 0

            hint = segment_hint  # deprecated variable kept to minimise other code changes
            desc = current_generator.format_position_description(
                total_generated_latent_frames, 
                current_pos, 
                original_pos, 
                current_prompt
            )

            progress_data = {
                'preview': preview,
                'desc': desc,
                'html': make_progress_bar_html(percentage, segment_hint) + make_progress_bar_html(total_percentage, total_hint)
            }
            if job_stream is not None:
                job = job_queue.get_job(job_id)
                if job:
                    job.progress_data = progress_data

            stream_to_use.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, segment_hint) + make_progress_bar_html(total_percentage, total_hint))))

        # --- Main generation loop ---
        # `i_section_loop` will be our loop counter for applying end_frame_latent
        for i_section_loop, latent_padding in enumerate(latent_paddings): # Existing loop structure
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream_to_use.input_queue.top() == 'end':
                stream_to_use.output_queue.push(('end', None))
                return

            # Calculate the current time position
            if model_type == "Video":
                # For Video model, add the input video time to the current position
                input_video_time = input_video_frame_count * 4 / 30  # Convert latent frames to time
                current_time_position = (total_generated_latent_frames * 4 - 3) / 30  # in seconds
                if current_time_position < 0:
                    current_time_position = 0.01
            else:
                # For other models, calculate as before
                current_time_position = (total_generated_latent_frames * 4 - 3) / 30  # in seconds
                if current_time_position < 0:
                    current_time_position = 0.01

            # Find the appropriate prompt for this section
            current_prompt = prompt_sections[0].prompt  # Default to first prompt
            for section in prompt_sections:
                if section.start_time <= current_time_position and (section.end_time is None or current_time_position < section.end_time):
                    current_prompt = section.prompt
                    break

            # PROMPT BLENDING: Find if we're in a blend window
            blend_alpha = None
            prev_prompt = current_prompt
            next_prompt = current_prompt

            # Only try to blend if we have prompt change indices and multiple sections
            if prompt_change_indices and len(prompt_sections) > 1:
                for i, (change_idx, prompt) in enumerate(prompt_change_indices):
                    if section_idx < change_idx:
                        prev_prompt = prompt_change_indices[i - 1][1] if i > 0 else prompt
                        next_prompt = prompt
                        blend_start = change_idx
                        blend_end = change_idx + blend_sections
                        if section_idx >= change_idx and section_idx < blend_end:
                            blend_alpha = (section_idx - change_idx + 1) / blend_sections
                        break
                    elif section_idx == change_idx:
                        # At the exact change, start blending
                        if i > 0:
                            prev_prompt = prompt_change_indices[i - 1][1]
                            next_prompt = prompt
                            blend_alpha = 1.0 / blend_sections
                        else:
                            prev_prompt = prompt
                            next_prompt = prompt
                            blend_alpha = None
                        break
                else:
                    # After last change, no blending
                    prev_prompt = current_prompt
                    next_prompt = current_prompt
                    blend_alpha = None

            # Get the encoded prompt for this section
            if blend_alpha is not None and prev_prompt != next_prompt:
                # Blend embeddings
                prev_llama_vec, prev_llama_attention_mask, prev_clip_l_pooler = encoded_prompts[prev_prompt]
                next_llama_vec, next_llama_attention_mask, next_clip_l_pooler = encoded_prompts[next_prompt]
                llama_vec = (1 - blend_alpha) * prev_llama_vec + blend_alpha * next_llama_vec
                llama_attention_mask = prev_llama_attention_mask  # usually same
                clip_l_pooler = (1 - blend_alpha) * prev_clip_l_pooler + blend_alpha * next_clip_l_pooler
                print(f"Blending prompts: '{prev_prompt[:30]}...' -> '{next_prompt[:30]}...', alpha={blend_alpha:.2f}")
            else:
                llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[current_prompt]

            original_time_position = total_second_length - current_time_position
            if original_time_position < 0:
                original_time_position = 0

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, '
                  f'time position: {current_time_position:.2f}s (original: {original_time_position:.2f}s), '
                  f'using prompt: {current_prompt[:60]}...')

            # Apply end_frame_latent to history_latents for Original with Endframe model
            if model_type == "Original with Endframe" and i_section_loop == 0 and end_frame_latent is not None:
                print(f"Applying end_frame_latent to history_latents with strength: {end_frame_strength}")
                actual_end_frame_latent_for_history = end_frame_latent.clone()
                if end_frame_strength != 1.0: # Only multiply if not full strength
                    actual_end_frame_latent_for_history = actual_end_frame_latent_for_history * end_frame_strength
                
                # Ensure history_latents is on the correct device (usually CPU for this kind of modification if it's init'd there)
                # and that the assigned tensor matches its dtype.
                # The `current_generator.prepare_history_latents` initializes it on CPU with float32.
                if history_latents.shape[2] >= 1: # Check if the 'Depth_slots' dimension is sufficient
                    history_latents[:, :, 0:1, :, :] = actual_end_frame_latent_for_history.to(
                        device=history_latents.device, # Assign to history_latents' current device
                        dtype=history_latents.dtype    # Match history_latents' dtype
                    )
                    print("End frame latent applied to history.")
                else:
                    print("Warning: history_latents not shaped as expected for end_frame application.")
            
            
            # Prepare indices using the generator
            clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices = current_generator.prepare_indices(latent_padding_size, latent_window_size)

            # Prepare clean latents using the generator
            clean_latents, clean_latents_2x, clean_latents_4x = current_generator.prepare_clean_latents(start_latent, history_latents)
            
            # Print debug info
            print(f"{model_type} model section {section_idx+1}/{total_latent_sections}, latent_padding={latent_padding}")

            if not high_vram:
                # Unload VAE etc. before loading transformer
                unload_complete_models(vae, text_encoder, text_encoder_2, image_encoder)
                move_model_to_device_with_memory_preservation(current_generator.transformer, target_device=gpu, preserved_memory_gb=0)
                if selected_loras:
                    current_generator.move_lora_adapters_to_device(gpu)

            if use_teacache:
                current_generator.transformer.initialize_teacache(enable_teacache=True, num_steps=teacache_num_steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                current_generator.transformer.initialize_teacache(enable_teacache=False)

            generated_latents = sample_hunyuan(
                transformer=current_generator.transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            # Update history latents using the generator
            history_latents = current_generator.update_history_latents(history_latents, generated_latents)

            if not high_vram:
                if selected_loras:
                    current_generator.move_lora_adapters_to_device(cpu) # Moving LoRAs to CPU is fine for VRAM preservation if needed during swap
                # offload_model_from_device_for_memory_preservation(current_generator.transformer, target_device=gpu, preserved_memory_gb=8)
                # Ensure VAE is loaded (it should be already if not offloaded)
                load_model_as_complete(vae, target_device=gpu)


            # Get real history latents using the generator
            real_history_latents = current_generator.get_real_history_latents(history_latents, total_generated_latent_frames)

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = current_generator.get_section_latent_frames(latent_window_size, is_last_section)
                overlapped_frames = latent_window_size * 4 - 3

                # Get current pixels using the generator
                current_pixels = current_generator.get_current_pixels(real_history_latents, section_latent_frames, vae)
                
                # Update history pixels using the generator
                history_pixels = current_generator.update_history_pixels(history_pixels, current_pixels, overlapped_frames)
                
                print(f"{model_type} model section {section_idx+1}/{total_latent_sections}, history_pixels shape: {history_pixels.shape}")

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(output_dir, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=settings.get("mp4_crf"))
            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            stream_to_use.output_queue.push(('file', output_filename))

            if is_last_section:
                break

            section_idx += 1  # PROMPT BLENDING: increment section index

        # For Video model, concatenate the input video with the generated content
        if model_type == "Video":
            print("Concatenating input video with generated content...")
            # Since the generation happens in reverse order, we need to reverse the history_pixels
            # before concatenating with the input video
            print(f"Reversing generated content. Shape before: {history_pixels.shape}")
            # Reverse the frames along the time dimension (dim=2)
            reversed_history_pixels = torch.flip(history_pixels, dims=[2])
            print(f"Shape after reversal: {reversed_history_pixels.shape}")
            
            # Get the last frame of the input video and the first frame of the reversed generated video
            last_input_frame = input_video_pixels[:, :, -1:, :, :]
            first_gen_frame = reversed_history_pixels[:, :, 0:1, :, :]
            print(f"Last input frame shape: {last_input_frame.shape}")
            print(f"First generated frame shape: {first_gen_frame.shape}")
            
            # Calculate the difference between the frames
            frame_diff = first_gen_frame - last_input_frame
            print(f"Frame difference magnitude: {torch.abs(frame_diff).mean().item()}")
            
            # Blend the first few frames of the generated video to create a smoother transition
            blend_frames = 5  # Number of frames to blend
            if reversed_history_pixels.shape[2] > blend_frames:
                print(f"Blending first {blend_frames} frames for smoother transition")
                for i in range(blend_frames):
                    # Calculate blend factor (1.0 at frame 0, decreasing to 0.0)
                    blend_factor = 1.0 - (i / blend_frames)
                    # Apply correction with decreasing strength
                    reversed_history_pixels[:, :, i:i+1, :, :] = reversed_history_pixels[:, :, i:i+1, :, :] - frame_diff * blend_factor
            
            # Concatenate the input video pixels with the reversed history pixels
            # The input video should come first, followed by the generated content
            # This makes the video extend from where the input video ends
            combined_pixels = torch.cat([input_video_pixels, reversed_history_pixels], dim=2)
            
            # Create the final video with both input and generated content
            output_filename = os.path.join(output_dir, f'{job_id}_final_with_input.mp4')
            save_bcthw_as_mp4(combined_pixels, output_filename, fps=30, crf=settings.get("mp4_crf"))
            print(f'Final video with input: {output_filename}')
            stream_to_use.output_queue.push(('file', output_filename))

        # Unload all LoRAs after generation completed
        if selected_loras:
            print("Unloading all LoRAs after generation completed")
            current_generator.unload_loras()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except:
        traceback.print_exc()
        # Unload all LoRAs after error
        if current_generator is not None and selected_loras:
            print("Unloading all LoRAs after error")
            current_generator.unload_loras()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        stream_to_use.output_queue.push(('error', f"Error during generation: {traceback.format_exc()}"))
        if not high_vram:
            # Ensure all models including the potentially active transformer are unloaded on error
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, 
                current_generator.transformer if current_generator else None
            )

    if settings.get("clean_up_videos"):
        try:
            video_files = [
                f for f in os.listdir(output_dir)
                if f.startswith(f"{job_id}_") and f.endswith(".mp4")
            ]
            print(f"Video files found for cleanup: {video_files}")
            if video_files:
                def get_frame_count(filename):
                    try:
                        # Handles filenames like jobid_123.mp4
                        return int(filename.replace(f"{job_id}_", "").replace(".mp4", ""))
                    except Exception:
                        return -1
                video_files_sorted = sorted(video_files, key=get_frame_count)
                print(f"Sorted video files: {video_files_sorted}")
                final_video = video_files_sorted[-1]
                for vf in video_files_sorted[:-1]:
                    full_path = os.path.join(output_dir, vf)
                    try:
                        os.remove(full_path)
                        print(f"Deleted intermediate video: {full_path}")
                    except Exception as e:
                        print(f"Failed to delete {full_path}: {e}")
        except Exception as e:
            print(f"Error during video cleanup: {e}")
    
    # Clean up temp folder if enabled
    if settings.get("cleanup_temp_folder"):
        try:
            temp_dir = settings.get("gradio_temp_dir")
            if temp_dir and os.path.exists(temp_dir):
                print(f"Cleaning up temp folder: {temp_dir}")
                items = os.listdir(temp_dir)
                removed_count = 0
                for item in items:
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.remove(item_path)
                            removed_count += 1
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            removed_count += 1
                    except Exception as e:
                        print(f"Error removing {item_path}: {e}")
                print(f"Cleaned up {removed_count} temporary files/folders.")
        except Exception as e:
            print(f"Error during temp folder cleanup: {e}")

    # Final verification of LoRA state
    if current_generator and current_generator.transformer:
        verify_lora_state(current_generator.transformer, "Worker end")

    stream_to_use.output_queue.push(('end', None))
    return



# Set the worker function for the job queue
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
        lora_loaded_names,
        *lora_values
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
        'lora_loaded_names': lora_loaded_names
    }
    
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
    return None, job_id, None, '', f'Job added to queue. Job ID: {job_id}', gr.update(interactive=True), gr.update(interactive=True)



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


def monitor_job(job_id):
    """
    Monitor a specific job and update the UI with the latest video segment as soon as it's available.
    """
    if not job_id:
        yield None, None, None, '', 'No job ID provided', gr.update(interactive=True), gr.update(interactive=True)
        return

    last_video = None  # Track the last video file shown
    last_job_status = None  # Track the previous job status to detect status changes

    while True:
        job = job_queue.get_job(job_id)
        if not job:
            yield None, job_id, None, '', 'Job not found', gr.update(interactive=True), gr.update(interactive=True)
            return

        # If a new video file is available, yield it immediately
        if job.result and job.result != last_video:
            last_video = job.result
            # You can also update preview/progress here if desired
            yield last_video, job_id, gr.update(visible=True), '', '', gr.update(interactive=True), gr.update(interactive=True)

        # Handle job status and progress
        if job.status == JobStatus.PENDING:
            position = job_queue.get_queue_position(job_id)
            yield last_video, job_id, gr.update(visible=True), '', f'Waiting in queue. Position: {position}', gr.update(interactive=True), gr.update(interactive=True)

        elif job.status == JobStatus.RUNNING:
            # Only reset the cancel button when a job transitions from another state to RUNNING
            # This ensures we don't reset the button text during cancellation
            if last_job_status != JobStatus.RUNNING:
                button_update = gr.update(interactive=True, value="Cancel Current Job")
            else:
                button_update = gr.update(interactive=True)  # Keep current text
                
            if job.progress_data and 'preview' in job.progress_data:
                preview = job.progress_data.get('preview')
                desc = job.progress_data.get('desc', '')
                html = job.progress_data.get('html', '')
                yield last_video, job_id, gr.update(visible=True, value=preview), desc, html, gr.update(interactive=True), button_update
            else:
                yield last_video, job_id, gr.update(visible=True), '', 'Processing...', gr.update(interactive=True), button_update

        elif job.status == JobStatus.COMPLETED:
            # Show the final video and reset the button text
            yield last_video, job_id, gr.update(visible=True), '', '', gr.update(interactive=True), gr.update(interactive=True, value="Cancel Current Job")
            break

        elif job.status == JobStatus.FAILED:
            # Show error and reset the button text
            yield last_video, job_id, gr.update(visible=True), '', f'Error: {job.error}', gr.update(interactive=True), gr.update(interactive=True, value="Cancel Current Job")
            break

        elif job.status == JobStatus.CANCELLED:
            # Show cancelled message and reset the button text
            yield last_video, job_id, gr.update(visible=True), '', 'Job cancelled', gr.update(interactive=True), gr.update(interactive=True, value="Cancel Current Job")
            break

        # Update last_job_status for the next iteration
        last_job_status = job.status
        
        # Wait a bit before checking again
        time.sleep(0.5)


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
    inbrowser=args.inbrowser
)
