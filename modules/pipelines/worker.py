import os
import json
import time
import traceback
import einops
import numpy as np
import torch
import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers_helper.utils import save_bcthw_as_mp4, generate_timestamp, resize_and_center_crop
from diffusers_helper.memory import cpu, gpu, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from diffusers_helper.hunyuan import vae_decode
from modules.video_queue import JobStatus
from modules.prompt_handler import parse_timestamped_prompt
from modules.generators import create_model_generator
from modules.pipelines.video_tools import combine_videos_sequentially_from_tensors
from modules import DUMMY_LORA_NAME # Import the constant
from . import create_pipeline

import __main__ as studio_module # Get a reference to the __main__ module object

@torch.no_grad()
def get_cached_or_encode_prompt(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, target_device, prompt_embedding_cache):
    """
    Retrieves prompt embeddings from cache or encodes them if not found.
    Stores encoded embeddings (on CPU) in the cache.
    Returns embeddings moved to the target_device.
    """
    from diffusers_helper.hunyuan import encode_prompt_conds, crop_or_pad_yield_mask
    
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
    end_frame_image,     # The end frame image (numpy array or None)
    end_frame_strength,  # Influence of the end frame
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
    lora_loaded_names=[],
    input_video=None,     # Add input_video parameter with default value of None
    combine_with_source=None,  # Add combine_with_source parameter
    num_cleaned_frames=5,  # Add num_cleaned_frames parameter with default value
    save_metadata_checked=True  # Add save_metadata_checked parameter
):
    """
    Worker function for video generation.
    """
    # Filter out the dummy LoRA from selected_loras at the very beginning of the worker
    actual_selected_loras_for_worker = []
    if isinstance(selected_loras, list):
        actual_selected_loras_for_worker = [lora for lora in selected_loras if lora != DUMMY_LORA_NAME]
        if DUMMY_LORA_NAME in selected_loras and DUMMY_LORA_NAME in actual_selected_loras_for_worker: # Should not happen if filter works
            print(f"Worker.py: Error - '{DUMMY_LORA_NAME}' was selected but not filtered out.")
        elif DUMMY_LORA_NAME in selected_loras:
             print(f"Worker.py: Filtered out '{DUMMY_LORA_NAME}' from selected LoRAs.")
    elif selected_loras is not None: # If it's a single string (should not happen with multiselect dropdown)
        if selected_loras != DUMMY_LORA_NAME:
            actual_selected_loras_for_worker = [selected_loras]
    selected_loras = actual_selected_loras_for_worker
    print(f"Worker: Selected LoRAs for this worker: {selected_loras}")
    
    # Import globals from the main module
    from __main__ import high_vram, args, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, image_encoder, feature_extractor, prompt_embedding_cache, settings, stream
    
    # Ensure any existing LoRAs are unloaded from the current generator
    if studio_module.current_generator is not None:
        print("Worker: Unloading LoRAs from studio_module.current_generator")
        studio_module.current_generator.unload_loras()
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

    # Initialize progress data with a clear starting message and dummy preview
    dummy_preview = np.zeros((64, 64, 3), dtype=np.uint8)
    initial_progress_data = {
        'preview': dummy_preview,
        'desc': 'Starting job...',
        'html': make_progress_bar_html(0, 'Starting job...')
    }
    
    # Store initial progress data in the job object if using a job stream
    if job_stream is not None:
        try:
            from __main__ import job_queue
            job = job_queue.get_job(job_id)
            if job:
                job.progress_data = initial_progress_data
        except Exception as e:
            print(f"Error storing initial progress data: {e}")
    
    # Push initial progress update to both streams
    stream_to_use.output_queue.push(('progress', (dummy_preview, 'Starting job...', make_progress_bar_html(0, 'Starting job...'))))
    
    # Push job ID to stream to ensure monitoring connection
    stream_to_use.output_queue.push(('job_id', job_id))
    stream_to_use.output_queue.push(('monitor_job', job_id))
    
    # Always push to the main stream to ensure the UI is updated
    from __main__ import stream as main_stream
    if main_stream:  # Always push to main stream regardless of whether it's the same as stream_to_use
        print(f"Pushing initial progress update to main stream for job {job_id}")
        main_stream.output_queue.push(('progress', (dummy_preview, 'Starting job...', make_progress_bar_html(0, 'Starting job...'))))
        
        # Push job ID to main stream to ensure monitoring connection
        main_stream.output_queue.push(('job_id', job_id))
        main_stream.output_queue.push(('monitor_job', job_id))

    try:
        # Create a settings dictionary for the pipeline
        pipeline_settings = {
            "output_dir": output_dir,
            "metadata_dir": metadata_dir,
            "input_files_dir": input_files_dir,
            "save_metadata": settings.get("save_metadata", True),
            "gpu_memory_preservation": settings.get("gpu_memory_preservation", 6),
            "mp4_crf": settings.get("mp4_crf", 16),
            "clean_up_videos": settings.get("clean_up_videos", True),
            "cleanup_temp_folder": settings.get("cleanup_temp_folder", True),
            "gradio_temp_dir": settings.get("gradio_temp_dir", "./gradio_temp"),
            "high_vram": high_vram
        }
        
        # Create the appropriate pipeline for the model type
        pipeline = create_pipeline(model_type, pipeline_settings)
        
        # Create job parameters dictionary
        job_params = {
            'model_type': model_type,
            'input_image': input_image,
            'end_frame_image': end_frame_image,
            'end_frame_strength': end_frame_strength,
            'prompt_text': prompt_text,
            'n_prompt': n_prompt,
            'seed': seed,
            'total_second_length': total_second_length,
            'latent_window_size': latent_window_size,
            'steps': steps,
            'cfg': cfg,
            'gs': gs,
            'rs': rs,
            'blend_sections': blend_sections,
            'latent_type': latent_type,
            'use_teacache': use_teacache,
            'teacache_num_steps': teacache_num_steps,
            'teacache_rel_l1_thresh': teacache_rel_l1_thresh,
            'selected_loras': selected_loras,
            'has_input_image': has_input_image,
            'lora_values': lora_values,
            'resolutionW': resolutionW,
            'resolutionH': resolutionH,
            'lora_loaded_names': lora_loaded_names,
            'input_image_path': input_image_path,
            'end_frame_image_path': end_frame_image_path,
            'combine_with_source': combine_with_source,
            'num_cleaned_frames': num_cleaned_frames,
            'save_metadata_checked': save_metadata_checked # Ensure it's in job_params for internal use
        }
        
        # Validate parameters
        is_valid, error_message = pipeline.validate_parameters(job_params)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_message}")
        
        # Prepare parameters
        job_params = pipeline.prepare_parameters(job_params)
        
        if not high_vram:
            # Unload everything *except* the potentially active transformer
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae)
            if studio_module.current_generator is not None and studio_module.current_generator.transformer is not None:
                offload_model_from_device_for_memory_preservation(studio_module.current_generator.transformer, target_device=gpu, preserved_memory_gb=8)


        # --- Model Loading / Switching ---
        print(f"Worker starting for model type: {model_type}")
        print(f"Worker: Before model assignment, studio_module.current_generator is {type(studio_module.current_generator)}, id: {id(studio_module.current_generator)}")
        
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
        # This modifies the 'current_generator' attribute OF THE '__main__' MODULE OBJECT
        studio_module.current_generator = new_generator
        print(f"Worker: AFTER model assignment, studio_module.current_generator is {type(studio_module.current_generator)}, id: {id(studio_module.current_generator)}")
        if studio_module.current_generator:
             print(f"Worker: studio_module.current_generator.transformer is {type(studio_module.current_generator.transformer)}")        
             
        # Load the transformer model
        studio_module.current_generator.load_model()
        
        # Ensure the model has no LoRAs loaded
        print(f"Ensuring {model_type} model has no LoRAs loaded")
        studio_module.current_generator.unload_loras()

        # Preprocess inputs
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Preprocessing inputs...'))))
        processed_inputs = pipeline.preprocess_inputs(job_params)
        
        # Update job_params with processed inputs
        job_params.update(processed_inputs)
        
        # Save the starting image directly to the output directory with full metadata
        # Check both global settings and job-specific save_metadata_checked parameter
        if settings.get("save_metadata") and job_params.get('save_metadata_checked', True) and job_params.get('input_image') is not None:
            try:
                # Import the save_job_start_image function from metadata_utils
                from modules.pipelines.metadata_utils import save_job_start_image, create_metadata
                
                # Create comprehensive metadata for the job
                metadata_dict = create_metadata(job_params, job_id, settings)
                
                # Save the starting image with metadata
                save_job_start_image(job_params, job_id, settings)
                
                print(f"Saved metadata and starting image for job {job_id}")
            except Exception as e:
                print(f"Error saving starting image and metadata: {e}")
                traceback.print_exc()
                
        # Pre-encode all prompts
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding all prompts...'))))
        
        # THE FOLLOWING CODE SHOULD BE INSIDE THE TRY BLOCK
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
            # Use the helper function for caching and encoding
            llama_vec, llama_attention_mask, clip_l_pooler = get_cached_or_encode_prompt(
                prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, gpu, prompt_embedding_cache
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
                n_prompt_str, text_encoder, text_encoder_2, tokenizer, tokenizer_2, gpu, prompt_embedding_cache
            )

        # Process input image or video based on model type
        if model_type == "Video" or model_type == "Video F1":
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Video processing ...'))))
            
            # Encode the video using the VideoModelGenerator
            start_latent, input_image_np, video_latents, fps, height, width, input_video_pixels, end_of_input_video_image_np, input_frames_resized_np = studio_module.current_generator.video_encode(
                video_path=job_params['input_image'],  # For Video model, input_image contains the video path
                resolution=job_params['resolutionW'],
                no_resize=False,
                vae_batch_size=16,
                device=gpu,
                input_files_dir=job_params['input_files_dir']
            )

            # RT_BORG: retained only until we make our final decisions on how to handle combining videos
            # Only necessary to retain resized frames to produce a combined video with source frames of the right dimensions 
            #if combine_with_source:
            #    # Store input_frames_resized_np in job_params for later use
            #    job_params['input_frames_resized_np'] = input_frames_resized_np
            
            # CLIP Vision encoding for the first frame
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
            
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)
                
            from diffusers_helper.clip_vision import hf_clip_vision_encode
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            end_of_input_video_embedding = hf_clip_vision_encode(end_of_input_video_image_np, feature_extractor, image_encoder).last_hidden_state
            
            # Store the input video pixels and latents for later use
            input_video_pixels = input_video_pixels.cpu()
            video_latents = video_latents.cpu()
            
            # Store the full video latents in the generator instance for preparing clean latents
            if hasattr(studio_module.current_generator, 'set_full_video_latents'):
                studio_module.current_generator.set_full_video_latents(video_latents.clone())
                print(f"Stored full input video latents in VideoModelGenerator. Shape: {video_latents.shape}")
            
            # For Video model, history_latents is initialized with the video_latents
            history_latents = video_latents
            
            # Store the last frame of the video latents as start_latent for the model
            start_latent = video_latents[:, :, -1:].cpu()
            print(f"Using last frame of input video as start_latent. Shape: {start_latent.shape}")
            print(f"Placed last frame of video at position 0 in history_latents")
            
            print(f"Initialized history_latents with video context. Shape: {history_latents.shape}")
            
            # Store the number of frames in the input video for later use
            input_video_frame_count = video_latents.shape[2]
        else:
            # Regular image processing
            input_image_np = job_params['input_image']
            height = job_params['height']
            width = job_params['width']
            
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            from diffusers_helper.hunyuan import vae_encode
            start_latent = vae_encode(input_image_pt, vae)

            # CLIP Vision
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            from diffusers_helper.clip_vision import hf_clip_vision_encode
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # VAE encode end_frame_image if provided
        end_frame_latent = None
        # VAE encode end_frame_image resized to output dimensions, if provided
        end_frame_output_dimensions_latent = None 
        end_clip_embedding = None # Video model end frame CLIP Vision embedding
        end_of_input_video_embedding = None # Video model end frame CLIP Vision embedding

        # Models with end_frame_image processing
        if (model_type == "Original with Endframe" or model_type == "Video") and job_params.get('end_frame_image') is not None:
            print(f"Processing end frame for {model_type} model...")
            end_frame_image = job_params['end_frame_image']
            
            if not isinstance(end_frame_image, np.ndarray):
                print(f"Warning: end_frame_image is not a numpy array (type: {type(end_frame_image)}). Attempting conversion or skipping.")
                try:
                    end_frame_image = np.array(end_frame_image)
                except Exception as e_conv:
                    print(f"Could not convert end_frame_image to numpy array: {e_conv}. Skipping end frame.")
                    end_frame_image = None
            
            if end_frame_image is not None:
                # Use the main job's target width/height (bucket dimensions) for the end frame
                end_frame_np = job_params['end_frame_image']
                
                if settings.get("save_metadata"):
                    Image.fromarray(end_frame_np).save(os.path.join(metadata_dir, f'{job_id}_end_frame_processed.png'))
                
                end_frame_pt = torch.from_numpy(end_frame_np).float() / 127.5 - 1
                end_frame_pt = end_frame_pt.permute(2, 0, 1)[None, :, None] # VAE expects [B, C, F, H, W]
                
                if not high_vram: load_model_as_complete(vae, target_device=gpu) # Ensure VAE is loaded
                from diffusers_helper.hunyuan import vae_encode
                end_frame_latent = vae_encode(end_frame_pt, vae)

                # end_frame_output_dimensions_latent is sized like the start_latent and generated latents
                end_frame_output_dimensions_np = resize_and_center_crop(end_frame_np, width, height)
                end_frame_output_dimensions_pt = torch.from_numpy(end_frame_output_dimensions_np).float() / 127.5 - 1
                end_frame_output_dimensions_pt = end_frame_output_dimensions_pt.permute(2, 0, 1)[None, :, None] # VAE expects [B, C, F, H, W]
                end_frame_output_dimensions_latent = vae_encode(end_frame_output_dimensions_pt, vae)

                print("End frame VAE encoded.")

                # Video Mode CLIP Vision encoding for end frame
                if model_type == "Video":
                    if not high_vram: # Ensure image_encoder is on GPU for this operation
                        load_model_as_complete(image_encoder, target_device=gpu)
                    from diffusers_helper.clip_vision import hf_clip_vision_encode
                    end_clip_embedding = hf_clip_vision_encode(end_frame_np, feature_extractor, image_encoder).last_hidden_state
                    end_clip_embedding = end_clip_embedding.to(studio_module.current_generator.transformer.dtype)
                    # Need that dtype conversion for end_clip_embedding? I don't think so, but it was in the original PR.
        
        if not high_vram: # Offload VAE and image_encoder if they were loaded
            offload_model_from_device_for_memory_preservation(vae, target_device=gpu, preserved_memory_gb=settings.get("gpu_memory_preservation"))
            offload_model_from_device_for_memory_preservation(image_encoder, target_device=gpu, preserved_memory_gb=settings.get("gpu_memory_preservation"))
        
        # Dtype
        for prompt_key in encoded_prompts:
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[prompt_key]
            llama_vec = llama_vec.to(studio_module.current_generator.transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(studio_module.current_generator.transformer.dtype)
            encoded_prompts[prompt_key] = (llama_vec, llama_attention_mask, clip_l_pooler)

        llama_vec_n = llama_vec_n.to(studio_module.current_generator.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(studio_module.current_generator.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(studio_module.current_generator.transformer.dtype)

        # Sampling
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        # Initialize total_generated_latent_frames for Video model
        total_generated_latent_frames = 0  # Default initialization for all model types

        # Initialize history latents based on model type
        if model_type != "Video" and model_type != "Video F1":  # Skip for Video models as we already initialized it
            history_latents = studio_module.current_generator.prepare_history_latents(height, width)
            
            # For F1 model, initialize with start latent
            if model_type == "F1":
                history_latents = studio_module.current_generator.initialize_with_start_latent(history_latents, start_latent)
                total_generated_latent_frames = 1  # Start with 1 for F1 model since it includes the first frame
            elif model_type == "Original" or model_type == "Original with Endframe":
                total_generated_latent_frames = 0

        history_pixels = None
        
        # Get latent paddings from the generator
        latent_paddings = studio_module.current_generator.get_latent_paddings(total_latent_sections)

        # PROMPT BLENDING: Track section index
        section_idx = 0

        # Load LoRAs if selected
        if selected_loras:
            lora_folder_from_settings = settings.get("lora_dir")
            studio_module.current_generator.load_loras(selected_loras, lora_folder_from_settings, lora_loaded_names, lora_values)

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
            from diffusers_helper.hunyuan import vae_decode_fake
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
            desc = studio_module.current_generator.format_position_description(
                total_generated_latent_frames, 
                current_pos, 
                original_pos, 
                current_prompt
            )

            # Create progress data dictionary
            progress_data = {
                'preview': preview,
                'desc': desc,
                'html': make_progress_bar_html(percentage, segment_hint) + make_progress_bar_html(total_percentage, total_hint)
            }
            
            # Store progress data in the job object if using a job stream
            if job_stream is not None:
                try:
                    from __main__ import job_queue
                    job = job_queue.get_job(job_id)
                    if job:
                        job.progress_data = progress_data
                except Exception as e:
                    print(f"Error updating job progress data: {e}")
                    
            # Always push to the job-specific stream
            stream_to_use.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, segment_hint) + make_progress_bar_html(total_percentage, total_hint))))
            
            # Always push to the main stream to ensure the UI is updated
            # This is especially important for resumed jobs
            from __main__ import stream as main_stream
            if main_stream:  # Always push to main stream regardless of whether it's the same as stream_to_use
                main_stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, segment_hint) + make_progress_bar_html(total_percentage, total_hint))))
                
            # Also push job ID to main stream to ensure monitoring connection
            if main_stream:
                main_stream.output_queue.push(('job_id', job_id))
                main_stream.output_queue.push(('monitor_job', job_id))

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

            # Only try to blend if blend_sections > 0 and we have prompt change indices and multiple sections
            try:
                blend_sections_int = int(blend_sections)
            except ValueError:
                blend_sections_int = 0 # Default to 0 if conversion fails, effectively disabling blending
                print(f"Warning: blend_sections ('{blend_sections}') is not a valid integer. Disabling prompt blending for this section.")
            if blend_sections_int > 0 and prompt_change_indices and len(prompt_sections) > 1:
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

            # Apply end_frame_latent to history_latents for models with Endframe support
            if (model_type == "Original with Endframe") and i_section_loop == 0 and end_frame_latent is not None:
                print(f"Applying end_frame_latent to history_latents with strength: {end_frame_strength}")
                actual_end_frame_latent_for_history = end_frame_latent.clone()
                if end_frame_strength != 1.0: # Only multiply if not full strength
                    actual_end_frame_latent_for_history = actual_end_frame_latent_for_history * end_frame_strength
                
                # Ensure history_latents is on the correct device (usually CPU for this kind of modification if it's init'd there)
                # and that the assigned tensor matches its dtype.
                # The `studio_module.current_generator.prepare_history_latents` initializes it on CPU with float32.
                if history_latents.shape[2] >= 1: # Check if the 'Depth_slots' dimension is sufficient
                    if model_type == "Original with Endframe":
                        # For Original model, apply to the beginning (position 0)
                        history_latents[:, :, 0:1, :, :] = actual_end_frame_latent_for_history.to(
                            device=history_latents.device, # Assign to history_latents' current device
                            dtype=history_latents.dtype    # Match history_latents' dtype
                        )
                    elif model_type == "F1 with Endframe":
                        # For F1 model, apply to the end (last position)
                        history_latents[:, :, -1:, :, :] = actual_end_frame_latent_for_history.to(
                            device=history_latents.device, # Assign to history_latents' current device
                            dtype=history_latents.dtype    # Match history_latents' dtype
                        )
                    print(f"End frame latent applied to history for {model_type} model.")
                else:
                    print("Warning: history_latents not shaped as expected for end_frame application.")
            
            
            # Video models use combined methods to prepare clean latents and indices
            if model_type == "Video":
                # Get num_cleaned_frames from job_params if available, otherwise use default value of 5
                num_cleaned_frames = job_params.get('num_cleaned_frames', 5)
                clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices, clean_latents, clean_latents_2x, clean_latents_4x = \
                studio_module.current_generator.video_prepare_clean_latents_and_indices(end_frame_output_dimensions_latent, end_frame_strength, end_clip_embedding, end_of_input_video_embedding, latent_paddings, latent_padding, latent_padding_size, latent_window_size, video_latents, history_latents, num_cleaned_frames)
            elif model_type == "Video F1":
                # Get num_cleaned_frames from job_params if available, otherwise use default value of 5
                num_cleaned_frames = job_params.get('num_cleaned_frames', 5)
                clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices, clean_latents, clean_latents_2x, clean_latents_4x = \
                studio_module.current_generator.video_f1_prepare_clean_latents_and_indices(latent_window_size, video_latents, history_latents, num_cleaned_frames)
            else:
                # Prepare indices using the generator
                clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices = studio_module.current_generator.prepare_indices(latent_padding_size, latent_window_size)

                # Prepare clean latents using the generator
                clean_latents, clean_latents_2x, clean_latents_4x = studio_module.current_generator.prepare_clean_latents(start_latent, history_latents)
            
            # Print debug info
            print(f"{model_type} model section {section_idx+1}/{total_latent_sections}, latent_padding={latent_padding}")

            if not high_vram:
                # Unload VAE etc. before loading transformer
                unload_complete_models(vae, text_encoder, text_encoder_2, image_encoder)
                move_model_to_device_with_memory_preservation(studio_module.current_generator.transformer, target_device=gpu, preserved_memory_gb=settings.get("gpu_memory_preservation"))
                if selected_loras:
                    studio_module.current_generator.move_lora_adapters_to_device(gpu)

            if use_teacache:
                studio_module.current_generator.transformer.initialize_teacache(enable_teacache=True, num_steps=teacache_num_steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                studio_module.current_generator.transformer.initialize_teacache(enable_teacache=False)

            from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
            generated_latents = sample_hunyuan(
                transformer=studio_module.current_generator.transformer,
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
            history_latents = studio_module.current_generator.update_history_latents(history_latents, generated_latents)

            if not high_vram:
                if selected_loras:
                    studio_module.current_generator.move_lora_adapters_to_device(cpu)
                offload_model_from_device_for_memory_preservation(studio_module.current_generator.transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            # Get real history latents using the generator
            real_history_latents = studio_module.current_generator.get_real_history_latents(history_latents, total_generated_latent_frames)

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = studio_module.current_generator.get_section_latent_frames(latent_window_size, is_last_section)
                overlapped_frames = latent_window_size * 4 - 3

                # Get current pixels using the generator
                current_pixels = studio_module.current_generator.get_current_pixels(real_history_latents, section_latent_frames, vae)
                
                # Update history pixels using the generator
                history_pixels = studio_module.current_generator.update_history_pixels(history_pixels, current_pixels, overlapped_frames)
                
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

            # We'll handle combining the videos after the entire generation is complete
            # This section intentionally left empty to remove the in-process combination

        # Handle the results
        result = pipeline.handle_results(job_params, output_filename)

        # Unload all LoRAs after generation completed
        if selected_loras:
            print("Unloading all LoRAs after generation completed")
            studio_module.current_generator.unload_loras()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        traceback.print_exc()
        # Unload all LoRAs after error
        if studio_module.current_generator is not None and selected_loras:
            print("Unloading all LoRAs after error")
            studio_module.current_generator.unload_loras()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        stream_to_use.output_queue.push(('error', f"Error during generation: {traceback.format_exc()}"))
        if not high_vram:
            # Ensure all models including the potentially active transformer are unloaded on error
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, 
                studio_module.current_generator.transformer if studio_module.current_generator else None
            )
    finally:
        # This finally block is associated with the main try block (starts around line 154)
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
                if temp_dir and os.path.exists(temp_dir): # Check if temp_dir is not None before os.path.exists
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
                                import shutil # Import shutil here as it's only used in this block
                                shutil.rmtree(item_path)
                                removed_count += 1
                        except Exception as e:
                            print(f"Error removing {item_path}: {e}")
                    print(f"Cleaned up {removed_count} temporary files/folders.")
            except Exception as e:
                print(f"Error during temp folder cleanup: {e}")

        # Check if the user wants to combine the source video with the generated video
        # This is done after the video cleanup routine to ensure the combined video is not deleted
        # RT_BORG: Retain (but suppress) this original way to combine videos until the new combiner is proven.
        combine_v1 = False
        if combine_v1 and (model_type == "Video" or model_type == "Video F1") and combine_with_source and job_params.get('input_image_path'):
            print("Creating combined video with source and generated content...")
            try:
                input_video_path = job_params.get('input_image_path')
                if input_video_path and os.path.exists(input_video_path):
                    final_video_path_for_combine = None # Use a different variable name to avoid conflict
                    video_files_for_combine = [
                        f for f in os.listdir(output_dir)
                        if f.startswith(f"{job_id}_") and f.endswith(".mp4") and "combined" not in f
                    ]
                    
                    if video_files_for_combine:
                        def get_frame_count_for_combine(filename): # Renamed to avoid conflict
                            try:
                                return int(filename.replace(f"{job_id}_", "").replace(".mp4", ""))
                            except Exception:
                                return float('inf') 
                                
                        video_files_sorted_for_combine = sorted(video_files_for_combine, key=get_frame_count_for_combine)
                        if video_files_sorted_for_combine: # Check if the list is not empty
                             final_video_path_for_combine = os.path.join(output_dir, video_files_sorted_for_combine[-1])
                    
                    if final_video_path_for_combine and os.path.exists(final_video_path_for_combine):
                        combined_output_filename = os.path.join(output_dir, f'{job_id}_combined_v1.mp4')
                        combined_result = None
                        try:
                            if hasattr(studio_module.current_generator, 'combine_videos'):
                                print(f"Using VideoModelGenerator.combine_videos to create side-by-side comparison")
                                combined_result = studio_module.current_generator.combine_videos(
                                    source_video_path=input_video_path,
                                    generated_video_path=final_video_path_for_combine, # Use the correct variable
                                    output_path=combined_output_filename
                                )
                                
                                if combined_result:
                                    print(f"Combined video saved to: {combined_result}")
                                    stream_to_use.output_queue.push(('file', combined_result))
                                else:
                                    print("Failed to create combined video, falling back to direct ffmpeg method")
                                    combined_result = None 
                            else:
                                print("VideoModelGenerator does not have combine_videos method. Using fallback method.")
                        except Exception as e_combine: # Use a different exception variable name
                            print(f"Error in combine_videos method: {e_combine}")
                            print("Falling back to direct ffmpeg method")
                            combined_result = None 
                            
                        if not combined_result:
                            print("Using fallback method to combine videos")
                            from modules.toolbox.toolbox_processor import VideoProcessor
                            from modules.toolbox.message_manager import MessageManager
                            
                            message_manager = MessageManager()
                            # Pass settings.settings if it exists, otherwise pass the settings object
                            video_processor_settings = settings.settings if hasattr(settings, 'settings') else settings
                            video_processor = VideoProcessor(message_manager, video_processor_settings)
                            ffmpeg_exe = video_processor.ffmpeg_exe
                            
                            if ffmpeg_exe:
                                print(f"Using ffmpeg at: {ffmpeg_exe}")
                                import subprocess
                                temp_list_file = os.path.join(output_dir, f'{job_id}_filelist.txt')
                                with open(temp_list_file, 'w') as f:
                                    f.write(f"file '{input_video_path}'\n")
                                    f.write(f"file '{final_video_path_for_combine}'\n") # Use the correct variable
                                
                                ffmpeg_cmd = [
                                    ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                                    "-i", temp_list_file, "-c", "copy", combined_output_filename
                                ]
                                print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
                                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                                if os.path.exists(temp_list_file):
                                    os.remove(temp_list_file)
                                print(f"Combined video saved to: {combined_output_filename}")
                                stream_to_use.output_queue.push(('file', combined_output_filename))
                            else:
                                print("FFmpeg executable not found. Cannot combine videos.")
                    else:
                        print(f"Final video not found for combining with source: {final_video_path_for_combine}")
                else:
                    print(f"Input video path not found: {input_video_path}")
            except Exception as e_combine_outer: # Use a different exception variable name
                print(f"Error combining videos: {e_combine_outer}")
                traceback.print_exc()
    
        # Combine input frames (resized and center cropped if needed) with final generated history_pixels tensor sequentially ---
        # This creates ID_combined.mp4
        # RT_BORG: Be sure to add this check if we decide to retain the processed input frames for "small" input videos 
        # and job_params.get('input_frames_resized_np') is not None 
        if (model_type == "Video" or model_type == "Video F1") and combine_with_source and history_pixels is not None:
            print(f"Creating combined video ({job_id}_combined.mp4) with processed input frames and generated history_pixels tensor...")
            try:
                # input_frames_resized_np = job_params.get('input_frames_resized_np')

                # RT_BORG: I cringe calliing methods on BaseModelGenerator that only exist on VideoBaseGenerator, until we refactor
                input_frames_resized_np, fps, target_height, target_width = studio_module.current_generator.extract_video_frames(
                    is_for_encode=False,
                    video_path=job_params['input_image'],
                    resolution=job_params['resolutionW'],
                    no_resize=False,
                    input_files_dir=job_params['input_files_dir']
                )

                # history_pixels is (B, C, T, H, W), float32, [-1,1], on CPU
                if input_frames_resized_np is not None and history_pixels.numel() > 0 : # Check if history_pixels is not empty
                    combined_sequential_output_filename = os.path.join(output_dir, f'{job_id}_combined.mp4')
                    
                    # fps variable should be from the video_encode call earlier.
                    input_video_fps_for_combine = fps 
                    current_crf = settings.get("mp4_crf", 16)

                    # Call the new function from video_tools.py
                    combined_sequential_result_path = combine_videos_sequentially_from_tensors(
                        processed_input_frames_np=input_frames_resized_np,
                        generated_frames_pt=history_pixels,
                        output_path=combined_sequential_output_filename,
                        target_fps=input_video_fps_for_combine,
                        crf_value=current_crf
                    )
                    if combined_sequential_result_path:
                        stream_to_use.output_queue.push(('file', combined_sequential_result_path))
            except Exception as e:
                print(f"Error creating combined video ({job_id}_combined.mp4): {e}")
                traceback.print_exc()
    
    # Final verification of LoRA state
    if studio_module.current_generator and studio_module.current_generator.transformer:
        # Verify LoRA state
        has_loras = False
        if hasattr(studio_module.current_generator.transformer, 'peft_config'):
            adapter_names = list(studio_module.current_generator.transformer.peft_config.keys()) if studio_module.current_generator.transformer.peft_config else []
            if adapter_names:
                has_loras = True
                print(f"Transformer has LoRAs: {', '.join(adapter_names)}")
            else:
                print(f"Transformer has no LoRAs in peft_config")
        else:
            print(f"Transformer has no peft_config attribute")
            
        # Check for any LoRA modules
        for name, module in studio_module.current_generator.transformer.named_modules():
            if hasattr(module, 'lora_A') and module.lora_A:
                has_loras = True
            if hasattr(module, 'lora_B') and module.lora_B:
                has_loras = True
                
        if not has_loras:
            print(f"No LoRA components found in transformer")

    stream_to_use.output_queue.push(('end', None))
    return
