from diffusers_helper.hf_login import login

import json
import os
import time
import argparse
import traceback
import einops
import numpy as np
import torch

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
args = parser.parse_args()

print(args)

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

# Initialize transformer placeholders
transformer_original = None
transformer_f1 = None
current_transformer = None # Will hold the currently active model

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
lora_folder_from_settings = settings.get("lora_dir", default_lora_folder) # Use setting, fallback to default
print(f"Scanning for LoRAs in: {lora_folder_from_settings}")
if os.path.isdir(lora_folder_from_settings):
    try:
        lora_files = [f for f in os.listdir(lora_folder_from_settings)
                     if f.endswith('.safetensors') or f.endswith('.pt')]
        for lora_file in lora_files:
            lora_names.append(lora_file.split('.')[0]) # Get name without extension
        print(f"Found LoRAs: {lora_names}")
    except Exception as e:
        print(f"Error scanning LoRA directory '{lora_folder_from_settings}': {e}")
else:
    print(f"LoRA directory not found: {lora_folder_from_settings}")
# --- End LoRA population ---


# Create job queue
job_queue = VideoJobQueue()


def move_lora_adapters_to_device(model, target_device):
    """
    Move all LoRA adapters in a model to the specified device.
    This handles the PEFT implementation of LoRA.
    """
    print(f"Moving all LoRA adapters to {target_device}")
    
    # First, find all modules with LoRA adapters
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'active_adapter') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules.append((name, module))
    
    # Now move all LoRA components to the target device
    for name, module in lora_modules:
        # Get the active adapter name
        active_adapter = module.active_adapter
        
        # Move the LoRA layers to the target device
        if active_adapter is not None:
            if isinstance(module.lora_A, torch.nn.ModuleDict):
                # Handle ModuleDict case (PEFT implementation)
                for adapter_name in list(module.lora_A.keys()):
                    # Move lora_A
                    if adapter_name in module.lora_A:
                        module.lora_A[adapter_name] = module.lora_A[adapter_name].to(target_device)
                    
                    # Move lora_B
                    if adapter_name in module.lora_B:
                        module.lora_B[adapter_name] = module.lora_B[adapter_name].to(target_device)
                    
                    # Move scaling
                    if hasattr(module, 'scaling') and isinstance(module.scaling, dict) and adapter_name in module.scaling:
                        if isinstance(module.scaling[adapter_name], torch.Tensor):
                            module.scaling[adapter_name] = module.scaling[adapter_name].to(target_device)
            else:
                # Handle direct attribute case
                if hasattr(module, 'lora_A') and module.lora_A is not None:
                    module.lora_A = module.lora_A.to(target_device)
                if hasattr(module, 'lora_B') and module.lora_B is not None:
                    module.lora_B = module.lora_B.to(target_device)
                if hasattr(module, 'scaling') and module.scaling is not None:
                    if isinstance(module.scaling, torch.Tensor):
                        module.scaling = module.scaling.to(target_device)
    
    print(f"Moved all LoRA adapters to {target_device}")
    return model


# Function to load a LoRA file
def load_lora_file(lora_file):
    if not lora_file:
        return None, "No file selected"
    
    try:
        # Get the filename from the path
        _, lora_name = os.path.split(lora_file)
        
        # Copy the file to the lora directory
        lora_dest = os.path.join(lora_dir, lora_name)
        import shutil
        shutil.copy(lora_file, lora_dest)
        
        # Load the LoRA - NOTE: This needs adjustment for multiple transformers
        global current_transformer, lora_names
        if current_transformer is None:
            return None, "Error: No model loaded to apply LoRA to. Generate something first."
        
        # ADDED: Unload any existing LoRAs first
        current_transformer = lora_utils.unload_all_loras(current_transformer)
        
        current_transformer = lora_utils.load_lora(current_transformer, lora_dir, lora_name)
        
        # Add to lora_names if not already there
        lora_base_name = lora_name.split('.')[0]
        if lora_base_name not in lora_names:
            lora_names.append(lora_base_name)
        
        # Get the current device of the transformer
        device = next(current_transformer.parameters()).device
        
        # Move all LoRA adapters to the same device as the base model
        move_lora_adapters_to_device(current_transformer, device)
        
        print(f"Loaded LoRA: {lora_name} to {type(current_transformer).__name__}")
        
        # ADDED: Verify LoRA state after loading
        verify_lora_state(current_transformer, "After loading LoRA file")
        
        return gr.update(choices=lora_names), f"Successfully loaded LoRA: {lora_name}"
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return None, f"Error loading LoRA: {e}"
        
@torch.no_grad()
def worker(
    model_type,
    input_image,
    prompt_text, 
    n_prompt, 
    seed, 
    total_second_length, 
    latent_window_size,
    steps, 
    cfg, 
    gs, 
    rs, 
    gpu_memory_preservation, 
    use_teacache,
    teacache_num_steps, 
    teacache_rel_l1_thresh,
    mp4_crf, 
    save_metadata, 
    blend_sections, 
    latent_type,
    selected_loras,
    clean_up_videos, 
    lora_values=None, 
    job_stream=None,
    output_dir=None,
    metadata_dir=None,
    resolutionW=640,  # Add resolution parameter with default value
    resolutionH=640,
    lora_loaded_names=[]
):
    global transformer_original, transformer_f1, current_transformer, high_vram
    
    # ADDED: Ensure any existing LoRAs are unloaded from the current transformer
    if current_transformer is not None:
        print("Unloading any existing LoRAs before starting new job")
        current_transformer = lora_utils.unload_all_loras(current_transformer)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ADDED: Verify LoRA state at worker start
    verify_lora_state(current_transformer, "Worker start")
    
    stream_to_use = job_stream if job_stream is not None else stream

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # Parse the timestamped prompt with boundary snapping and reversing
    # prompt_text should now be the original string from the job queue
    prompt_sections = parse_timestamped_prompt(prompt_text, total_second_length, latent_window_size, model_type)
    job_id = generate_timestamp()

    stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        if not high_vram:
            # Unload everything *except* the potentially active transformer
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae)
            if current_transformer is not None:
                offload_model_from_device_for_memory_preservation(current_transformer, target_device=gpu, preserved_memory_gb=8)

        # --- Model Loading / Switching ---
        print(f"Worker starting for model type: {model_type}")
        target_transformer_model = None
        other_transformer_model = None

        if model_type == "Original":
            if transformer_original is None:
                print("Loading Original Transformer...")
                transformer_original = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
                transformer_original.eval()
                transformer_original.to(dtype=torch.bfloat16)
                transformer_original.requires_grad_(False)
                if not high_vram:
                    DynamicSwapInstaller.install_model(transformer_original, device=gpu)
                print("Original Transformer Loaded.")
            target_transformer_model = transformer_original
            other_transformer_model = transformer_f1
        elif model_type == "F1":
            if transformer_f1 is None:
                print("Loading F1 Transformer...")
                transformer_f1 = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()
                transformer_f1.eval()
                transformer_f1.to(dtype=torch.bfloat16)
                transformer_f1.requires_grad_(False)
                if not high_vram:
                    DynamicSwapInstaller.install_model(transformer_f1, device=gpu)
                print("F1 Transformer Loaded.")
            target_transformer_model = transformer_f1
            other_transformer_model = transformer_original
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Unload the *other* model if it exists and we are in low VRAM mode
        if not high_vram and other_transformer_model is not None:
            print(f"Offloading inactive transformer: {type(other_transformer_model).__name__}")
            offload_model_from_device_for_memory_preservation(other_transformer_model, target_device=gpu, preserved_memory_gb=8)
            # Consider fully unloading if memory pressure is extreme:
            # unload_complete_models(other_transformer_model)
            # if model_type == "Original": transformer_f1 = None
            # else: transformer_original = None

        current_transformer = target_transformer_model # Set the globally accessible current model

        # ADDED: Ensure the target model has no LoRAs loaded
        print(f"Ensuring {model_type} transformer has no LoRAs loaded")
        current_transformer = lora_utils.unload_all_loras(current_transformer)
        verify_lora_state(current_transformer, "After model selection")

        # Ensure the target model is on the correct device if in high VRAM mode
        if high_vram and current_transformer.device != gpu:
            print(f"Moving {model_type} transformer to GPU (High VRAM mode)...")
            current_transformer.to(gpu)

        # Pre-encode all prompts
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding all prompts...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        # PROMPT BLENDING: Pre-encode all prompts and store in a list in order
        unique_prompts = []
        for section in prompt_sections:
            if section.prompt not in unique_prompts:
                unique_prompts.append(section.prompt)

        encoded_prompts = {}
        for prompt in unique_prompts:
            llama_vec, clip_l_pooler = encode_prompt_conds(
                prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
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
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolutionW)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        if save_metadata:
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
                "mp4_crf": mp4_crf,
                "timestamp": time.time(),
                "resolutionW": resolutionW,  # Add resolution to metadata
                "resolutionH": resolutionH,
                "model_type": model_type  # Add model type to metadata
            }
            # Add LoRA information to metadata if LoRAs are used
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
            Image.fromarray(input_image_np).save(os.path.join(metadata_dir, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        for prompt_key in encoded_prompts:
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[prompt_key]
            llama_vec = llama_vec.to(current_transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(current_transformer.dtype)
            encoded_prompts[prompt_key] = (llama_vec, llama_attention_mask, clip_l_pooler)

        llama_vec_n = llama_vec_n.to(current_transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(current_transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(current_transformer.dtype)

        # Sampling
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        if model_type == "Original":
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        else:  # F1 model
            # F1モードでは初期フレームを用意
            history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
            # 開始フレームをhistory_latentsに追加
            history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
            total_generated_latent_frames = 1  # 最初のフレームを含むので1から開始

        history_pixels = None
        if model_type == "Original":
            total_generated_latent_frames = 0
            # Original model uses reversed latent paddings
            latent_paddings = reversed(range(total_latent_sections))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:  # F1 model
            # F1 model doesn't use latent paddings in the same way
            # We'll use a fixed approach with just 0 for last section and 1 for others
            latent_paddings = [1] * (total_latent_sections - 1) + [0]

        # PROMPT BLENDING: Track section index
        section_idx = 0

        # ADDED: Completely unload all loras from the current transformer
        current_transformer = lora_utils.unload_all_loras(current_transformer)
        verify_lora_state(current_transformer, "Before loading LoRAs")

        # --- LoRA loading and scaling ---
        if selected_loras:
            for lora_name in selected_loras:
                idx = lora_loaded_names.index(lora_name)
                lora_file = None
                for ext in [".safetensors", ".pt"]:
                    # Find any file that starts with the lora_name and ends with the extension
                    matching_files = [f for f in os.listdir(lora_folder_from_settings) 
                                   if f.startswith(lora_name) and f.endswith(ext)]
                    if matching_files:
                        lora_file = matching_files[0]  # Use the first matching file
                        break
                if lora_file:
                    print(f"Loading LoRA {lora_file} to {model_type} model")
                    current_transformer = lora_utils.load_lora(current_transformer, lora_folder_from_settings, lora_file)
                    # Set LoRA strength if provided
                    if lora_values and idx < len(lora_values):
                        lora_strength = float(lora_values[idx])
                        print(f"Setting LoRA {lora_name} strength to {lora_strength}")
                        # Set scaling for this LoRA by iterating through modules
                        for name, module in current_transformer.named_modules():
                            if hasattr(module, 'scaling'):
                                if isinstance(module.scaling, dict):
                                    # Handle ModuleDict case (PEFT implementation)
                                    if lora_name in module.scaling:
                                        if isinstance(module.scaling[lora_name], torch.Tensor):
                                            module.scaling[lora_name] = torch.tensor(
                                                lora_strength, device=module.scaling[lora_name].device
                                            )
                                        else:
                                            module.scaling[lora_name] = lora_strength
                                else:
                                    # Handle direct attribute case for scaling if needed
                                    if isinstance(module.scaling, torch.Tensor):
                                        module.scaling = torch.tensor(
                                            lora_strength, device=module.scaling.device
                                        )
                                    else:
                                        module.scaling = lora_strength
                else:
                    print(f"LoRA file for {lora_name} not found!")
            
            # ADDED: Verify LoRA state after loading
            verify_lora_state(current_transformer, "After loading LoRAs")

        # --- Callback for progress ---
        def callback(d):
            preview = d['denoised']
            preview = vae_decode_fake(preview)
            preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

            if stream_to_use.input_queue.top() == 'end':
                stream_to_use.output_queue.push(('end', None))
                raise KeyboardInterrupt('User ends the task.')

            current_step = d['i'] + 1
            percentage = int(100.0 * current_step / steps)
            current_pos = (total_generated_latent_frames * 4 - 3) / 30
            original_pos = total_second_length - current_pos
            if current_pos < 0: current_pos = 0
            if original_pos < 0: original_pos = 0

            hint = f'Sampling {current_step}/{steps}'
            if model_type == "Original":
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, ' \
                       f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). ' \
                       f'Current position: {current_pos:.2f}s (original: {original_pos:.2f}s). ' \
                       f'using prompt: {current_prompt[:256]}...'
            else:  # F1 model
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, ' \
                       f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). ' \
                       f'Current position: {current_pos:.2f}s. ' \
                       f'using prompt: {current_prompt[:256]}...'

            progress_data = {
                'preview': preview,
                'desc': desc,
                'html': make_progress_bar_html(percentage, hint)
            }
            if job_stream is not None:
                job = job_queue.get_job(job_id)
                if job:
                    job.progress_data = progress_data

            stream_to_use.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))

        # --- Main generation loop ---
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream_to_use.input_queue.top() == 'end':
                stream_to_use.output_queue.push(('end', None))
                return

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

            if model_type == "Original":
                # Original model uses the standard indices approach
                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            else:  # F1 model
                # F1 model uses a different indices approach
                # latent_window_sizeが4.5の場合は特別に5を使用
                effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
                indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
                
                print(f"F1 model indices: clean_latent_indices shape={clean_latent_indices.shape}, latent_indices shape={latent_indices.shape}")

            if model_type == "Original":
                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            else:  # F1 model
                # For F1, we take the last frames for clean latents
                clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
                # For F1, we prepend the start latent to clean_latents_1x
                clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
                
                # Print debug info for F1 model
                print(f"F1 model section {section_idx+1}/{total_latent_sections}, latent_padding={latent_padding}")

            if not high_vram:
                # Unload VAE etc. before loading transformer
                unload_complete_models(vae, text_encoder, text_encoder_2, image_encoder)
                move_model_to_device_with_memory_preservation(current_transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                if selected_loras:
                    move_lora_adapters_to_device(current_transformer, gpu)

            if use_teacache:
                current_transformer.initialize_teacache(enable_teacache=True, num_steps=teacache_num_steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                current_transformer.initialize_teacache(enable_teacache=False)

            generated_latents = sample_hunyuan(
                transformer=current_transformer,
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
            if model_type == "Original":
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            else:  # F1 model
                # For F1, we append new frames to the end
                history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                if selected_loras:
                    move_lora_adapters_to_device(current_transformer, cpu)
                offload_model_from_device_for_memory_preservation(current_transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            if model_type == "Original":
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            else:  # F1 model
                # For F1, we take frames from the end
                real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                
                overlapped_frames = latent_window_size * 4 - 3

                if model_type == "Original":
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                else:  # F1 model
                    # For F1, we take frames from the end
                    print(f"F1 model section {section_idx+1}/{total_latent_sections}, section_latent_frames={section_latent_frames}")
                    print(f"F1 model real_history_latents shape: {real_history_latents.shape}, taking last {section_latent_frames} frames")
                    
                    # Get the frames from the end of real_history_latents
                    current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                    
                    print(f"F1 model current_pixels shape: {current_pixels.shape}, history_pixels shape: {history_pixels.shape if history_pixels is not None else 'None'}")
                    
                    # For F1 model, history_pixels is first, current_pixels is second
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
                    
                    print(f"F1 model after append, history_pixels shape: {history_pixels.shape}")

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(output_dir, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            stream_to_use.output_queue.push(('file', output_filename))

            if is_last_section:
                break

            section_idx += 1  # PROMPT BLENDING: increment section index

        # ADDED: Unload all LoRAs after generation completed
        if selected_loras:
            print("Unloading all LoRAs after generation completed")
            current_transformer = lora_utils.unload_all_loras(current_transformer)
            verify_lora_state(current_transformer, "After generation completed")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except:
        traceback.print_exc()
        # ADDED: Unload all LoRAs after error
        if current_transformer is not None and selected_loras:
            print("Unloading all LoRAs after error")
            current_transformer = lora_utils.unload_all_loras(current_transformer)
            verify_lora_state(current_transformer, "After error")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        stream_to_use.output_queue.push(('error', f"Error during generation: {traceback.format_exc()}"))
        if not high_vram:
            # Ensure all models including the potentially active transformer are unloaded on error
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, current_transformer
            )

    if clean_up_videos:
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

    # ADDED: Final verification of LoRA state
    verify_lora_state(current_transformer, "Worker end")

    stream_to_use.output_queue.push(('end', None))
    return



# Set the worker function for the job queue
job_queue.set_worker_function(worker)


def process(
        model_type,
        input_image,
        prompt_text,
        n_prompt,
        seed, 
        total_second_length, 
        latent_window_size, 
        steps, 
        cfg, 
        gs, 
        rs, 
        gpu_memory_preservation, 
        use_teacache, 
        teacache_num_steps, 
        teacache_rel_l1_thresh,
        mp4_crf, 
        save_metadata,
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
    if input_image is None:
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

    
    # Create job parameters
    job_params = {
        'model_type': model_type,
        'input_image': input_image.copy(),  # Make a copy to avoid reference issues
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
        'gpu_memory_preservation': gpu_memory_preservation,
        'use_teacache': use_teacache,
        'teacache_num_steps': teacache_num_steps,
        'teacache_rel_l1_thresh': teacache_rel_l1_thresh,
        'mp4_crf': mp4_crf,
        'save_metadata': save_metadata,
        'selected_loras': selected_loras,
        'clean_up_videos': clean_up_videos,
        'output_dir': settings.get("output_dir"),
        'metadata_dir': settings.get("metadata_dir"),
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
    
    return format_queue_status(jobs)


def monitor_job(job_id):
    """
    Monitor a specific job and update the UI with the latest video segment as soon as it's available.
    """
    if not job_id:
        yield None, None, None, '', 'No job ID provided', gr.update(interactive=True), gr.update(interactive=True)
        return

    last_video = None  # Track the last video file shown

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
            if job.progress_data and 'preview' in job.progress_data:
                preview = job.progress_data.get('preview')
                desc = job.progress_data.get('desc', '')
                html = job.progress_data.get('html', '')
                yield last_video, job_id, gr.update(visible=True, value=preview), desc, html, gr.update(interactive=True), gr.update(interactive=True)
            else:
                yield last_video, job_id, gr.update(visible=True), '', 'Processing...', gr.update(interactive=True), gr.update(interactive=True)

        elif job.status == JobStatus.COMPLETED:
            # Show the final video
            yield last_video, job_id, gr.update(visible=True), '', '', gr.update(interactive=True), gr.update(interactive=True)
            break

        elif job.status == JobStatus.FAILED:
            yield last_video, job_id, gr.update(visible=True), '', f'Error: {job.error}', gr.update(interactive=True), gr.update(interactive=True)
            break

        elif job.status == JobStatus.CANCELLED:
            yield last_video, job_id, gr.update(visible=True), '', 'Job cancelled', gr.update(interactive=True), gr.update(interactive=True)
            break

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
