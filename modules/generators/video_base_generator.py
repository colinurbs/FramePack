import torch
import os
import numpy as np
import math
import decord
from tqdm import tqdm
import pathlib
from PIL import Image

from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import DynamicSwapInstaller
from diffusers_helper.utils import resize_and_center_crop
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.hunyuan import vae_encode, vae_decode
from .base_generator import BaseModelGenerator

class VideoBaseModelGenerator(BaseModelGenerator):
    """
    Model generator for the Video extension of the Original HunyuanVideo model.
    This generator accepts video input instead of a single image.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Video model generator.
        """
        super().__init__(**kwargs)
        self.model_name = None # Subclass Model Specific
        self.model_path = None # Subclass Model Specific
        self.model_repo_id_for_cache = None # Subclass Model Specific
        self.full_video_latents = None # For context, set by worker() when available
        self.resolution = 640  # Default resolution
        self.no_resize = False  # Default to resize
        self.vae_batch_size = 16  # Default VAE batch size
        
        # Import decord and tqdm here to avoid import errors if not installed
        try:
            import decord
            from tqdm import tqdm
            self.decord = decord
            self.tqdm = tqdm
        except ImportError:
            print("Warning: decord or tqdm not installed. Video processing will not work.")
            self.decord = None
            self.tqdm = None
    
    def get_model_name(self):
        """
        Get the name of the model.
        """
        return self.model_name
    
    def load_model(self):
        """
        Load the Video transformer model.
        If offline mode is True, attempts to load from a local snapshot.
        """
        print(f"Loading {self.model_name} Transformer...")
        
        path_to_load = self.model_path # Initialize with the default path

        if self.offline:
            path_to_load = self._get_offline_load_path() # Calls the method in BaseModelGenerator
        
        # Create the transformer model
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            path_to_load, 
            torch_dtype=torch.bfloat16
        ).cpu()
        
        # Configure the model
        self.transformer.eval()
        self.transformer.to(dtype=torch.bfloat16)
        self.transformer.requires_grad_(False)
        
        # Set up dynamic swap if not in high VRAM mode
        if not self.high_vram:
            DynamicSwapInstaller.install_model(self.transformer, device=self.gpu)
        else:
            # In high VRAM mode, move the entire model to GPU
            self.transformer.to(device=self.gpu)
        
        print(f"{self.model_name} Transformer Loaded from {path_to_load}.")
        return self.transformer
    
    def min_real_frames_to_encode(self, real_frames_available_count):
        """
        Minimum number of real frames to encode
        is the maximum number of real frames used for generation context.
        
        The number of latents could be calculated as below for video F1, but keeping it simple for now
        by hardcoding the Video F1 value at max_latents_used_for_context = 27.

        # Calculate the number of latent frames to encode from the end of the input video
        num_frames_to_encode_from_end = 1  # Default minimum
        if model_type == "Video":
            # Max needed is 1 (clean_latent_pre) + 2 (max 2x) + 16 (max 4x) = 19
            num_frames_to_encode_from_end = 19
        elif model_type == "Video F1":
            ui_num_cleaned_frames = job_params.get('num_cleaned_frames', 5)
            # Max effective_clean_frames based on VideoF1ModelGenerator's logic.
            # Max num_clean_frames from UI is 10 (modules/interface.py).
            # Max effective_clean_frames = 10 - 1 = 9.
            # total_context_frames = num_4x_frames + num_2x_frames + effective_clean_frames
            # Max needed = 16 (max 4x) + 2 (max 2x) + 9 (max effective_clean_frames) = 27
            num_frames_to_encode_from_end = 27
        
        Note: 27 latents ~ 108 real frames = 3.6 seconds at 30 FPS.
        Note: 19 latents ~ 76 real frames ~ 2.5 seconds at 30 FPS.
        """

        max_latents_used_for_context = 27
        if self.get_model_name == "Video":
            max_latents_used_for_context = 19
        elif self.get_model_name == "Video F1":
            max_latents_used_for_context = 27  # Enough for even Video F1 with cleaned_frames input of 10
        else:
            print("======================================================")
            print(f"    *****    Warning: Unsupported video extension model type: {self.get_model_name()}.")
            print( "    *****    Using default max latents {max_latents_used_for_context} for context.")
            print( "    *****    Please report to the developers if you see this message:")
            print( "    *****    Discord: https://discord.gg/8Z2c3a4 or GitHub: https://github.com/colinurbs/FramePack-Studio")
            print("======================================================")
            # Probably better to press on with Video F1 max vs exception?
            # raise ValueError(f"Unsupported video extension model type: {self.get_model_name()}")

        latent_size_factor = 4 # real frames to latent frames conversion factor
        max_real_frames_used_for_context = max_latents_used_for_context * latent_size_factor

        # Shortest of available frames and max frames used for context
        trimmed_real_frames_count = min(real_frames_available_count, max_real_frames_used_for_context)
        if trimmed_real_frames_count < real_frames_available_count:
            print(f"Truncating video frames from {real_frames_available_count} to {trimmed_real_frames_count}, enough to populate context")

        # Truncate to nearest latent size (multiple of 4)
        frames_to_encode_count = (trimmed_real_frames_count // latent_size_factor) * latent_size_factor
        if frames_to_encode_count != trimmed_real_frames_count:
            print(f"Truncating video frames from {trimmed_real_frames_count} to {frames_to_encode_count}, for latent size compatibility")

        return frames_to_encode_count

    def extract_video_frames(self, is_for_encode, video_path, resolution, no_resize=False, input_files_dir=None):
        """
        Extract real frames from a video, resized and center cropped as numpy array (T, H, W, C).
        
        Args:
            is_for_encode: If True, results are capped at maximum frames used for context, and aligned to 4-frame latent requirement.
            video_path: Path to the input video file.
            resolution: Target resolution for resizing frames.
            no_resize: Whether to use the original video resolution.
            input_files_dir: Directory for input files that won't be cleaned up.
        
        Returns:
            A tuple containing:
            - input_frames_resized_np: All input frames resized and center cropped as numpy array (T, H, W, C)
            - fps: Frames per second of the input video
            - target_height: Target height of the video
            - target_width: Target width of the video
        """
        def time_millis():
            import time
            return time.perf_counter() * 1000.0 # Convert seconds to milliseconds
        
        encode_start_time_millis = time_millis()
           
        # Normalize video path for Windows compatibility
        video_path = str(pathlib.Path(video_path).resolve())
        print(f"Processing video: {video_path}")
        
        # Check if the video is in the temp directory and if we have an input_files_dir
        if input_files_dir and "temp" in video_path:
            # Check if there's a copy of this video in the input_files_dir
            filename = os.path.basename(video_path)
            input_file_path = os.path.join(input_files_dir, filename)
            
            # If the file exists in input_files_dir, use that instead
            if os.path.exists(input_file_path):
                print(f"Using video from input_files_dir: {input_file_path}")
                video_path = input_file_path
            else:
                # If not, copy it to input_files_dir to prevent it from being deleted
                try:
                    from diffusers_helper.utils import generate_timestamp
                    safe_filename = f"{generate_timestamp()}_{filename}"
                    input_file_path = os.path.join(input_files_dir, safe_filename)
                    import shutil
                    shutil.copy2(video_path, input_file_path)
                    print(f"Copied video to input_files_dir: {input_file_path}")
                    video_path = input_file_path
                except Exception as e:
                    print(f"Error copying video to input_files_dir: {e}")

        try:
            # Load video and get FPS
            print("Initializing VideoReader...")
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()  # Get input video FPS
            num_real_frames = len(vr)
            print(f"Video loaded: {num_real_frames} frames, FPS: {fps}")

            # Read frames
            print("Reading video frames...")

            total_frames_in_video_file = len(vr)
            if is_for_encode:
                print(f"Using minimum real frames to encode: {self.min_real_frames_to_encode(total_frames_in_video_file)}")
                num_real_frames = self.min_real_frames_to_encode(total_frames_in_video_file)
            # else left as all frames -- len(vr) with no regard for trimming or latent alignment

            # RT_BORG: Retaining this commented code for reference.
            # pftq encoder discarded truncated frames from the end of the video.
            # frames = vr.get_batch(range(num_real_frames)).asnumpy()  # Shape: (num_real_frames, height, width, channels)

            # RT_BORG: Retaining this commented code for reference.
            # pftq retained the entire encoded video.
            # Truncate to nearest latent size (multiple of 4)
            # latent_size_factor = 4
            # num_frames = (num_real_frames // latent_size_factor) * latent_size_factor
            # if num_frames != num_real_frames:
            #     print(f"Truncating video from {num_real_frames} to {num_frames} frames for latent size compatibility")
            # num_real_frames = num_frames

            # Discard truncated frames from the beginning of the video, retaining the last num_real_frames
            # This ensures a smooth transition from the input video to the generated video
            start_frame_index = total_frames_in_video_file - num_real_frames
            frame_indices_to_extract = range(start_frame_index, total_frames_in_video_file)
            frames = vr.get_batch(frame_indices_to_extract).asnumpy()  # Shape: (num_real_frames, height, width, channels)

            print(f"Frames read: {frames.shape}")

            # Get native video resolution
            native_height, native_width = frames.shape[1], frames.shape[2]
            print(f"Native video resolution: {native_width}x{native_height}")
        
            # Use native resolution if height/width not specified, otherwise use provided values
            target_height = native_height
            target_width = native_width
        
            # Adjust to nearest bucket for model compatibility
            if not no_resize:
                target_height, target_width = find_nearest_bucket(target_height, target_width, resolution=resolution)
                print(f"Adjusted resolution: {target_width}x{target_height}")
            else:
                print(f"Using native resolution without resizing: {target_width}x{target_height}")

            # Preprocess input frames to match desired resolution
            input_frames_resized_np = []
            for i, frame in tqdm(enumerate(frames), desc="Processing Video Frames", total=num_real_frames, mininterval=0.1):
                frame_np = resize_and_center_crop(frame, target_width=target_width, target_height=target_height)
                input_frames_resized_np.append(frame_np)
            input_frames_resized_np = np.stack(input_frames_resized_np)  # Shape: (num_real_frames, height, width, channels)
            print(f"Frames preprocessed: {input_frames_resized_np.shape}")

            resized_frames_time_millis = time_millis()
            print("======================================================")
            memory_bytes = input_frames_resized_np.nbytes
            memory_kb = memory_bytes / 1024
            memory_mb = memory_kb / 1024

            print(f"    *****    input_frames_resized_np: {input_frames_resized_np.shape}")
            print(f"    *****    Memory usage: {int(memory_mb)} MB")
            duration_ms = resized_frames_time_millis - encode_start_time_millis
            print(f"    *****    Time taken to process frames tensor: {duration_ms / 1000.0:.2f} seconds")
            print("======================================================")

            return input_frames_resized_np, fps, target_height, target_width
        except Exception as e:
            print(f"Error in extract_video_frames: {str(e)}")
            raise

    # RT_BORG: video_encode produce and return end_of_input_video_latent and end_of_input_video_image_np
    # which are not needed for Video models without end frame processing.
    # But these should be inexpensive and it's easier to keep the code uniform.
    @torch.no_grad()
    def video_encode(self, video_path, resolution, no_resize=False, vae_batch_size=16, device=None, input_files_dir=None):
        """
        Encode a video into latent representations using the VAE.
        
        Args:
            video_path: Path to the input video file.
            resolution: Target resolution for resizing frames.
            no_resize: Whether to use the original video resolution.
            vae_batch_size: Number of frames to process per batch.
            device: Device for computation (e.g., "cuda").
            input_files_dir: Directory for input files that won't be cleaned up.
        
        Returns:
            A tuple containing:
            - start_latent: Latent of the first frame
            - input_image_np: First frame as numpy array
            - history_latents: Latents of all frames
            - fps: Frames per second of the input video
            - target_height: Target height of the video
            - target_width: Target width of the video
            - input_video_pixels: Video frames as tensor
            - end_of_input_video_image_np: Last frame as numpy array
            - input_frames_resized_np: All input frames resized and center cropped as numpy array (T, H, W, C)
        """
        encoding = True  # Flag to indicate this is for encoding
        input_frames_resized_np, fps, target_height, target_width = self.extract_video_frames(encoding, video_path, resolution, no_resize, input_files_dir)

        try:
            if device is None:
                device = self.gpu
                
            # Check CUDA availability and fallback to CPU if needed
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA is not available, falling back to CPU")
                device = "cpu"

            # Save first frame for CLIP vision encoding
            input_image_np = input_frames_resized_np[0]
            end_of_input_video_image_np = input_frames_resized_np[-1]

            # Convert to tensor and normalize to [-1, 1]
            print("Converting frames to tensor...")
            frames_pt = torch.from_numpy(input_frames_resized_np).float() / 127.5 - 1
            frames_pt = frames_pt.permute(0, 3, 1, 2)  # Shape: (num_real_frames, channels, height, width)
            frames_pt = frames_pt.unsqueeze(0)  # Shape: (1, num_real_frames, channels, height, width)
            frames_pt = frames_pt.permute(0, 2, 1, 3, 4)  # Shape: (1, channels, num_real_frames, height, width)
            print(f"Tensor shape: {frames_pt.shape}")
            
            # Save pixel frames for use in worker
            input_video_pixels = frames_pt.cpu()

            # Move to device
            print(f"Moving tensor to device: {device}")
            frames_pt = frames_pt.to(device)
            print("Tensor moved to device")

            # Move VAE to device
            print(f"Moving VAE to device: {device}")
            self.vae.to(device)
            print("VAE moved to device")

            # Encode frames in batches
            print(f"Encoding input video frames in VAE batch size {vae_batch_size}")
            latents = []
            self.vae.eval()
            with torch.no_grad():
                frame_count = frames_pt.shape[2]
                step_count = math.ceil(frame_count / vae_batch_size)
                for i in tqdm(range(0, frame_count, vae_batch_size), desc="Encoding video frames", total=step_count, mininterval=0.1):
                    batch = frames_pt[:, :, i:i + vae_batch_size]  # Shape: (1, channels, batch_size, height, width)
                    try:
                        # Log GPU memory before encoding
                        if device == "cuda":
                            free_mem = torch.cuda.memory_allocated() / 1024**3
                        batch_latent = vae_encode(batch, self.vae)
                        # Synchronize CUDA to catch issues
                        if device == "cuda":
                            torch.cuda.synchronize()
                        latents.append(batch_latent)
                    except RuntimeError as e:
                        print(f"Error during VAE encoding: {str(e)}")
                        if device == "cuda" and "out of memory" in str(e).lower():
                            print("CUDA out of memory, try reducing vae_batch_size or using CPU")
                        raise
            
            # Concatenate latents
            print("Concatenating latents...")
            history_latents = torch.cat(latents, dim=2)  # Shape: (1, channels, frames, height//8, width//8)
            print(f"History latents shape: {history_latents.shape}")

            # Get first frame's latent
            start_latent = history_latents[:, :, :1]  # Shape: (1, channels, 1, height//8, width//8)
            print(f"Start latent shape: {start_latent.shape}")

            print("======================================================")
            memory_bytes = history_latents.nbytes
            memory_kb = memory_bytes / 1024
            memory_mb = memory_kb / 1024

            print(f"    *****    history_latents: {history_latents.shape}")
            print(f"    *****    Memory usage: {int(memory_mb)} MB")
            print("======================================================")

            # Move VAE back to CPU to free GPU memory
            if device == "cuda":
                self.vae.to(self.cpu)
                torch.cuda.empty_cache()
                print("VAE moved back to CPU, CUDA cache cleared")

            return start_latent, input_image_np, history_latents, fps, target_height, target_width, input_video_pixels, end_of_input_video_image_np, input_frames_resized_np

        except Exception as e:
            print(f"Error in video_encode: {str(e)}")
            raise
    
    # RT_BORG: Currently history_latents is initialized within worker() for all Video models as history_latents = video_latents
    # So it is a coding error to call prepare_history_latents() here.
    # Leaving in place as we will likely use it post-refactoring.
    def prepare_history_latents(self, height, width):
        """
        Prepare the history latents tensor for the Video model.
        
        Args:
            height: The height of the image
            width: The width of the image
            
        Returns:
            The initialized history latents tensor
        """
        raise TypeError(
            f"Error: '{self.__class__.__name__}.prepare_history_latents' should not be called "
            "on the Video models. history_latents should be initialized within worker() for all Video models "
            "as history_latents = video_latents."
        )

    def prepare_indices(self, latent_padding_size, latent_window_size):
        """
        Prepare the indices for the Video model.
        
        Args:
            latent_padding_size: The size of the latent padding
            latent_window_size: The size of the latent window
            
        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices)
        """
        raise TypeError(
            f"Error: '{self.__class__.__name__}.prepare_indices' should not be called "
            "on the Video models. Currently video models each have a combined method: <model>_prepare_clean_latents_and_indices."
        )

    def set_full_video_latents(self, video_latents):
        """
        Set the full video latents for context.
        
        Args:
            video_latents: The full video latents
        """
        self.full_video_latents = video_latents
    
    def prepare_clean_latents(self, start_latent, history_latents):
        """
        Prepare the clean latents for the Video model.
        
        Args:
            start_latent: The start latent
            history_latents: The history latents
            
        Returns:
            A tuple of (clean_latents, clean_latents_2x, clean_latents_4x)
        """
        raise TypeError(
            f"Error: '{self.__class__.__name__}.prepare_indices' should not be called "
            "on the Video models. Currently video models each have a combined method: <model>_prepare_clean_latents_and_indices."
        )
    
    def get_section_latent_frames(self, latent_window_size, is_last_section):
        """
        Get the number of section latent frames for the Video model.
        
        Args:
            latent_window_size: The size of the latent window
            is_last_section: Whether this is the last section
            
        Returns:
            The number of section latent frames
        """
        return latent_window_size * 2
        
    def combine_videos(self, source_video_path, generated_video_path, output_path):
        """
        Combine the source video with the generated video side by side.
        
        Args:
            source_video_path: Path to the source video
            generated_video_path: Path to the generated video
            output_path: Path to save the combined video
            
        Returns:
            Path to the combined video
        """
        try:
            import os
            import subprocess
            
            print(f"Combining source video {source_video_path} with generated video {generated_video_path}")
            
            # Get the ffmpeg executable from the VideoProcessor class
            from modules.toolbox.toolbox_processor import VideoProcessor
            from modules.toolbox.message_manager import MessageManager
            
            # Create a message manager for logging
            message_manager = MessageManager()
            
            # Import settings from main module
            try:
                from __main__ import settings
                video_processor = VideoProcessor(message_manager, settings.settings)
            except ImportError:
                # Fallback to creating a new settings object
                from modules.settings import Settings
                settings = Settings()
                video_processor = VideoProcessor(message_manager, settings.settings)
            
            # Get the ffmpeg executable
            ffmpeg_exe = video_processor.ffmpeg_exe
            
            if not ffmpeg_exe:
                print("FFmpeg executable not found. Cannot combine videos.")
                return None
            
            print(f"Using ffmpeg at: {ffmpeg_exe}")
            
            # Create a temporary directory for the filter script
            import tempfile
            temp_dir = tempfile.gettempdir()
            filter_script_path = os.path.join(temp_dir, f"filter_script_{os.path.basename(output_path)}.txt")
            
            # Get video dimensions using ffprobe
            def get_video_info(video_path):
                cmd = [
                    ffmpeg_exe, "-i", video_path, 
                    "-hide_banner", "-loglevel", "error"
                ]
                
                # Run ffmpeg to get video info (it will fail but output info to stderr)
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse the output to get dimensions
                width = height = None
                for line in result.stderr.split('\n'):
                    if 'Video:' in line:
                        # Look for dimensions like 640x480
                        import re
                        match = re.search(r'(\d+)x(\d+)', line)
                        if match:
                            width = int(match.group(1))
                            height = int(match.group(2))
                            break
                
                return width, height
            
            # Get dimensions of both videos
            source_width, source_height = get_video_info(source_video_path)
            generated_width, generated_height = get_video_info(generated_video_path)
            
            if not source_width or not generated_width:
                print("Error: Could not determine video dimensions")
                return None
            
            print(f"Source video: {source_width}x{source_height}")
            print(f"Generated video: {generated_width}x{generated_height}")
            
            # Calculate target dimensions (maintain aspect ratio)
            target_height = max(source_height, generated_height)
            source_target_width = int(source_width * (target_height / source_height))
            generated_target_width = int(generated_width * (target_height / generated_height))
            
            # Create a complex filter for side-by-side display with labels
            filter_complex = (
                f"[0:v]scale={source_target_width}:{target_height}[left];"
                f"[1:v]scale={generated_target_width}:{target_height}[right];"
                f"[left]drawtext=text='Source':x=({source_target_width}/2-50):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[left_text];"
                f"[right]drawtext=text='Generated':x=({generated_target_width}/2-70):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[right_text];"
                f"[left_text][right_text]hstack=inputs=2[v]"
            )
            
            # Write the filter script to a file
            with open(filter_script_path, 'w') as f:
                f.write(filter_complex)
            
            # Build the ffmpeg command
            cmd = [
                ffmpeg_exe, "-y",
                "-i", source_video_path,
                "-i", generated_video_path,
                "-filter_complex_script", filter_script_path,
                "-map", "[v]"
            ]
            
            # Check if source video has audio
            has_audio_cmd = [
                ffmpeg_exe, "-i", source_video_path,
                "-hide_banner", "-loglevel", "error"
            ]
            audio_check = subprocess.run(has_audio_cmd, capture_output=True, text=True)
            has_audio = "Audio:" in audio_check.stderr
            
            if has_audio:
                cmd.extend(["-map", "0:a"])
            
            # Add output options
            cmd.extend([
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium"
            ])
            
            if has_audio:
                cmd.extend(["-c:a", "aac"])
            
            cmd.append(output_path)
            
            # Run the ffmpeg command
            print(f"Running ffmpeg command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Clean up the filter script
            if os.path.exists(filter_script_path):
                os.remove(filter_script_path)
            
            print(f"Combined video saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error combining videos: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
