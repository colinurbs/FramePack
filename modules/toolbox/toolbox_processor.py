import os
import gc
import sys
import re
import numpy as np
import torch
import imageio
import gradio as gr
import subprocess
import devicetorch
import json
import math
import shutil
import traceback

from datetime import datetime
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from torchvision.transforms.functional import to_tensor, to_pil_image

from modules.toolbox.rife_core import RIFEHandler
from modules.toolbox.esrgan_core import ESRGANUpscaler
from modules.toolbox.message_manager import MessageManager

device_name_str = devicetorch.get(torch)

VIDEO_QUALITY = 8 # Used by imageio.mimwrite quality/quantizer

class VideoProcessor:
    def __init__(self, message_manager: MessageManager, settings):
        self.message_manager = message_manager
        self.rife_handler = RIFEHandler(message_manager)
        self.device_obj = torch.device(device_name_str) # Store device_obj
        self.esrgan_upscaler = ESRGANUpscaler(message_manager, self.device_obj)
        self.settings = settings

        # FFmpeg/FFprobe paths and status flags
        self.ffmpeg_exe = None
        self.ffprobe_exe = None
        self.has_ffmpeg = False
        self.has_ffprobe = False
        self.ffmpeg_source = None
        self.ffprobe_source = None

        self._tb_initialize_ffmpeg() # Finds executables and sets flags

        studio_output_dir = Path(self.settings.get("output_dir"))
        self.postprocessed_output_root_dir = studio_output_dir / "postprocessed_output"
        self._base_temp_output_dir = self.postprocessed_output_root_dir / "temp_processing"
        self._base_permanent_save_dir = self.postprocessed_output_root_dir / "saved_videos"

        self.toolbox_video_output_dir = self._base_temp_output_dir
        self.toolbox_permanent_save_dir = self._base_permanent_save_dir

        os.makedirs(self.postprocessed_output_root_dir, exist_ok=True)
        os.makedirs(self._base_temp_output_dir, exist_ok=True)
        os.makedirs(self._base_permanent_save_dir, exist_ok=True)
        
        # Note: Renamed to a more generic name as it holds more than just extracted frames now
        self.frames_io_dir = self.postprocessed_output_root_dir / "frames"
        self.extracted_frames_target_path = self.frames_io_dir / "extracted_frames"
        os.makedirs(self.extracted_frames_target_path, exist_ok=True)
        self.reassembled_video_target_path = self.frames_io_dir / "reassembled_videos"
        os.makedirs(self.reassembled_video_target_path, exist_ok=True)

    # --- NEW BATCH PROCESSING FUNCTION ---
    def tb_process_video_batch(self, video_paths: list, pipeline_config: dict, progress=gr.Progress()):
        """
        Processes a batch of videos according to a defined pipeline of operations.
        - Batch jobs are ALWAYS saved to a new, unique, timestamped folder in 'saved_videos'.
        - Single video pipeline jobs respect the 'Autosave' setting for the FINAL output only.
        - Intermediate files are always created in and cleaned from the temp directory.
        - The very last successfully processed video (from single or batch) is kept for the UI.
        """
        original_autosave_state = self.settings.get("toolbox_autosave_enabled", True)
        is_batch_job = len(video_paths) > 1
        batch_output_dir = None
        last_successful_video_path_for_ui = None

        try:
            if is_batch_job:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_output_dir = self._base_permanent_save_dir / f"batch_process_{timestamp}"
                os.makedirs(batch_output_dir, exist_ok=True)
                self.message_manager.add_message(f"Batch outputs will be saved to: {batch_output_dir}", "SUCCESS")

            self.set_autosave_mode(False, silent=True)

            operations = pipeline_config.get("operations", [])
            if not operations:
                self.message_manager.add_warning("No operations were selected for the pipeline. Nothing to do.")
                return None

            op_names = [op['name'].replace('_', ' ').title() for op in operations]
            self.message_manager.add_message(f"🚀 Starting pipeline for {len(video_paths)} videos. Pipeline: {' -> '.join(op_names)}")

            total_videos = len(video_paths)

            for i, original_video_path in enumerate(video_paths):
                progress(i / total_videos, desc=f"Video {i+1}/{total_videos}: {os.path.basename(original_video_path)}")
                self.message_manager.add_message(f"\n--- Processing Video {i+1}/{total_videos}: {os.path.basename(original_video_path)} ---", "INFO")

                current_video_path = original_video_path
                video_failed = False
                path_to_clean = None
                
                for op_config in operations:
                    op_name = op_config["name"]
                    op_params = op_config["params"]
                    
                    self.message_manager.add_message(f"  -> Step: Applying {op_name.replace('_', ' ')}...")
                    output_path = None
                    try:
                        if op_name == "upscale": output_path = self.tb_upscale_video(current_video_path, **op_params, progress=progress)
                        elif op_name == "frame_adjust": output_path = self.tb_process_frames(current_video_path, **op_params, progress=progress)
                        elif op_name == "filters": output_path = self.tb_apply_filters(current_video_path, **op_params, progress=progress)
                        elif op_name == "loop": output_path = self.tb_create_loop(current_video_path, **op_params, progress=progress)

                        if output_path and os.path.exists(output_path):
                            self.message_manager.add_success(f"  -> Step '{op_name}' completed. Output: {os.path.basename(output_path)}")
                            if path_to_clean and os.path.exists(path_to_clean):
                                try:
                                    os.remove(path_to_clean)
                                    self.message_manager.add_message(f"  -> Cleaned intermediate file: {os.path.basename(path_to_clean)}", "DEBUG")
                                except OSError as e:
                                    self.message_manager.add_warning(f"Could not clean intermediate file {path_to_clean}: {e}")

                            current_video_path = output_path
                            path_to_clean = output_path
                        else:
                            video_failed = True; break
                    except Exception as e:
                        video_failed = True
                        self.message_manager.add_error(f"An unexpected error occurred during step '{op_name}': {e}")
                        self.message_manager.add_error(traceback.format_exc())
                        break

                if not video_failed:
                    final_temp_path = current_video_path
                    is_last_video_in_batch = (i == total_videos - 1)
                    
                    if is_batch_job:
                        # For batch jobs, copy the final output to the permanent batch folder.
                        final_dest_path = batch_output_dir / os.path.basename(final_temp_path)
                        shutil.copy2(final_temp_path, final_dest_path) # Use copy2 to keep temp file for UI
                        self.message_manager.add_success(f"--- Successfully processed. Final output saved to: {final_dest_path} ---")

                        if is_last_video_in_batch:
                             # This is the very last video of the whole batch, keep its temp path for the UI player.
                            last_successful_video_path_for_ui = final_temp_path
                        else:
                            # This is a completed video but not the last one in the batch, so we can clean its temp file.
                            try: os.remove(final_temp_path)
                            except OSError: pass
                    else: # Single video pipeline run.
                        if original_autosave_state:
                            final_dest_path = self._base_permanent_save_dir / os.path.basename(final_temp_path)
                            shutil.move(final_temp_path, final_dest_path) # Move, as it's saved permanently
                            self.message_manager.add_success(f"--- Successfully processed. Final output saved to: {final_dest_path} ---")
                            last_successful_video_path_for_ui = final_dest_path
                        else:
                            # Autosave off, so the final file remains in the temp folder for the UI.
                            self.message_manager.add_success(f"--- Successfully processed. Final output is in temp folder: {final_temp_path} ---")
                            last_successful_video_path_for_ui = final_temp_path
                else:
                    self.message_manager.add_warning(f"--- Processing failed for {os.path.basename(original_video_path)} ---")
                    if path_to_clean and os.path.exists(path_to_clean):
                        try: os.remove(path_to_clean)
                        except OSError as e: self.message_manager.add_warning(f"Could not clean failed intermediate file {path_to_clean}: {e}")

                gc.collect()
                devicetorch.empty_cache(torch)

            progress(1.0, desc="Pipeline complete.")
            self.message_manager.add_message("\n✅ Pipeline processing finished.", "SUCCESS")
            return last_successful_video_path_for_ui
            
        finally:
            # Restore the user's original autosave setting silently.
            self.set_autosave_mode(original_autosave_state, silent=True)

    def _tb_initialize_ffmpeg(self):
        """Finds FFmpeg/FFprobe and sets status flags and sources."""
        (
            self.ffmpeg_exe,
            self.ffmpeg_source,
            self.ffprobe_exe,
            self.ffprobe_source,
        ) = self._tb_find_ffmpeg_executables()

        self.has_ffmpeg = bool(self.ffmpeg_exe)
        self.has_ffprobe = bool(self.ffprobe_exe)

        self._report_ffmpeg_status()

    def _tb_find_ffmpeg_executables(self):
        """
        Finds ffmpeg and ffprobe with a priority system.
        Priority: 1. Bundled -> 2. System PATH -> 3. imageio-ffmpeg
        Returns (ffmpeg_path, ffmpeg_source, ffprobe_path, ffprobe_source)
        """
        ffmpeg_path, ffprobe_path = None, None
        ffmpeg_source, ffprobe_source = None, None
        ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        ffprobe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            bin_dir = os.path.join(script_dir, 'bin')
            bundled_ffmpeg = os.path.join(bin_dir, ffmpeg_name)
            bundled_ffprobe = os.path.join(bin_dir, ffprobe_name)
            if os.path.exists(bundled_ffmpeg):
                ffmpeg_path = bundled_ffmpeg
                ffmpeg_source = "Bundled"
            if os.path.exists(bundled_ffprobe):
                ffprobe_path = bundled_ffprobe
                ffprobe_source = "Bundled"
        except Exception:
            pass

        if not ffmpeg_path:
            path_from_env = shutil.which(ffmpeg_name)
            if path_from_env:
                ffmpeg_path = path_from_env
                ffmpeg_source = "System PATH"
        if not ffprobe_path:
            path_from_env = shutil.which(ffprobe_name)
            if path_from_env:
                ffprobe_path = path_from_env
                ffprobe_source = "System PATH"

        if not ffmpeg_path:
            try:
                imageio_ffmpeg_exe = imageio.plugins.ffmpeg.get_exe()
                if os.path.isfile(imageio_ffmpeg_exe):
                    ffmpeg_path = imageio_ffmpeg_exe
                    ffmpeg_source = "imageio-ffmpeg"
            except Exception:
                pass

        return ffmpeg_path, ffmpeg_source, ffprobe_path, ffprobe_source

    def _report_ffmpeg_status(self):
        """Provides a summary of FFmpeg/FFprobe status based on what was found."""
        if self.ffmpeg_source == "Bundled" and self.ffprobe_source == "Bundled":
            self.message_manager.add_message(f"Bundled FFmpeg found: {self.ffmpeg_exe}", "SUCCESS")
            self.message_manager.add_message(f"Bundled FFprobe found: {self.ffprobe_exe}", "SUCCESS")
            self.message_manager.add_message("All video and audio features are enabled.", "SUCCESS")
            return

        if self.has_ffmpeg:
            self.message_manager.add_message(f"FFmpeg found via {self.ffmpeg_source}: {self.ffmpeg_exe}", "SUCCESS")
        else:
            self.message_manager.add_error("Critical: FFmpeg executable could not be found. Most video processing operations will fail. Please try running the setup script.")

        if self.has_ffprobe:
            self.message_manager.add_message(f"FFprobe found via {self.ffprobe_source}: {self.ffprobe_exe}", "SUCCESS")
        else:
            self.message_manager.add_warning("FFprobe not found. Audio detection and full video analysis will be limited.")
            if self.ffmpeg_source != "Bundled":
                 self.message_manager.add_warning("For full functionality, please run the 'setup_ffmpeg.py' script.")

    def tb_get_frames_from_folder(self, folder_name: str) -> list:
        """
        Gets a sorted list of image file paths from a given folder name.
        This is the backend for the "Load Frames to Studio" button.
        """
        if not folder_name:
            return []

        full_folder_path = os.path.join(self.extracted_frames_target_path, folder_name)
        if not os.path.isdir(full_folder_path):
            self.message_manager.add_error(f"Cannot load frames: Directory not found at {full_folder_path}")
            return []

        frame_files = []
        try:
            for filename in os.listdir(full_folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    frame_files.append(os.path.join(full_folder_path, filename))

            # Natural sort to handle frame_0, frame_1, ... frame_10 correctly
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

            frame_files.sort(key=natural_sort_key)
            return frame_files

        except Exception as e:
            self.message_manager.add_error(f"Error reading frames from '{folder_name}': {e}")
            return []

    def tb_delete_single_frame(self, frame_path_to_delete: str) -> str:
        """Deletes a single frame file from the disk, logs the action, and returns a status message."""
        if not frame_path_to_delete or not isinstance(frame_path_to_delete, str):
            # This message is returned to the app's info box
            msg_for_infobox = "Error: Invalid frame path provided for deletion."
            # The message manager gets a more detailed log entry
            self.message_manager.add_error("Could not delete frame: Invalid path provided to processor.")
            return msg_for_infobox

        try:
            filename = os.path.basename(frame_path_to_delete)
            if os.path.isfile(frame_path_to_delete):
                os.remove(frame_path_to_delete)
                # Add a success message to the main log
                self.message_manager.add_success(f"Deleted frame: {filename}")
                # Return a concise status for the info box
                return f"✅ Deleted: {filename}"
            else:
                self.message_manager.add_error(f"Could not delete frame. File not found: {frame_path_to_delete}")
                return f"Error: Frame not found"
        except OSError as e:
            self.message_manager.add_error(f"Error deleting frame {filename}: {e}")
            return f"Error deleting frame: {e}"

    def tb_save_single_frame(self, source_frame_path: str) -> str | None:
        """Saves a copy of a single frame to the permanent 'saved_videos' directory."""
        if not source_frame_path or not os.path.isfile(source_frame_path):
            self.message_manager.add_error("Source frame to save does not exist or is invalid.")
            return None

        try:
            source_path_obj = Path(source_frame_path)
            parent_folder_name = source_path_obj.parent.name
            frame_filename = source_path_obj.name
            
            # Create a descriptive filename to avoid collisions
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            dest_filename = f"saved_frame_{parent_folder_name}_{timestamp}_{frame_filename}"
            
            destination_path = os.path.join(self.toolbox_permanent_save_dir, dest_filename)
            
            os.makedirs(self.toolbox_permanent_save_dir, exist_ok=True)
            shutil.copy2(source_frame_path, destination_path)
            
            self.message_manager.add_success(f"Saved frame to permanent storage: {destination_path}")
            return destination_path
        except Exception as e:
            self.message_manager.add_error(f"Error saving frame to permanent storage: {e}")
            self.message_manager.add_error(traceback.format_exc())
            return None


    def set_autosave_mode(self, autosave_enabled: bool, silent: bool = False):
        if autosave_enabled:
            self.toolbox_video_output_dir = self._base_permanent_save_dir
            if not silent:
                self.message_manager.add_message("Autosave ENABLED: Processed videos will be saved to the permanent folder.", "SUCCESS")
        else:
            self.toolbox_video_output_dir = self._base_temp_output_dir
            if not silent:
                self.message_manager.add_message("Autosave DISABLED: Processed videos will be saved to the temporary folder.", "INFO")

    def _tb_log_ffmpeg_error(self, e_ffmpeg: subprocess.CalledProcessError, operation_description: str):
        self.message_manager.add_error(f"FFmpeg failed during {operation_description}.")
        ffmpeg_stderr_str = e_ffmpeg.stderr.strip() if e_ffmpeg.stderr else ""
        ffmpeg_stdout_str = e_ffmpeg.stdout.strip() if e_ffmpeg.stdout else ""

        details_log = []
        if ffmpeg_stderr_str: details_log.append(f"FFmpeg Stderr: {ffmpeg_stderr_str}")
        if ffmpeg_stdout_str: details_log.append(f"FFmpeg Stdout: {ffmpeg_stdout_str}")

        if details_log:
            self.message_manager.add_message("FFmpeg Output:\n" + "\n".join(details_log), "INFO")
        else:
            self.message_manager.add_message(f"No specific output from FFmpeg. (Return code: {e_ffmpeg.returncode}, Command: '{e_ffmpeg.cmd}')", "INFO")

    def _tb_get_video_frame_count(self, video_path: str) -> int | None:
        """
        Uses ffprobe to get an accurate frame count by requesting JSON output for robust parsing.
        Tries a fast metadata read first, then falls back to a slower but more accurate full stream count.
        """
        if not self.has_ffprobe:
            self.message_manager.add_message("Cannot get frame count: ffprobe not found.", "DEBUG")
            return None

        # --- Tier 1: Fast metadata read using JSON output ---
        try:
            ffprobe_cmd_fast = [
                self.ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames", "-of", "json", video_path
            ]
            result = subprocess.run(ffprobe_cmd_fast, capture_output=True, text=True, check=True, errors='ignore')
            data = json.loads(result.stdout)
            frame_count_str = data.get("streams", [{}])[0].get("nb_frames", "N/A")

            if frame_count_str.isdigit() and int(frame_count_str) > 0:
                self.message_manager.add_message(f"Frame count from metadata: {frame_count_str}", "DEBUG")
                return int(frame_count_str)
            else:
                 self.message_manager.add_warning(f"Fast metadata frame count was invalid ('{frame_count_str}'). Falling back to full count.")
        except Exception as e:
            self.message_manager.add_warning(f"Fast metadata read failed: {e}. Falling back to full count.")

        # --- Tier 2: Slow, accurate full-stream count using JSON output ---
        try:
            self.message_manager.add_message("Performing full, accurate frame count with ffprobe (this may take a moment)...", "INFO")
            ffprobe_cmd_accurate = [
                self.ffprobe_exe, "-v", "error", "-count_frames",
                "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                "-of", "json", video_path
            ]
            result = subprocess.run(ffprobe_cmd_accurate, capture_output=True, text=True, check=True, errors='ignore')
            data = json.loads(result.stdout)
            frame_count_str = data.get("streams", [{}])[0].get("nb_read_frames", "N/A")

            if frame_count_str.isdigit() and int(frame_count_str) > 0:
                self.message_manager.add_message(f"Accurate frame count from full scan: {frame_count_str}", "DEBUG")
                return int(frame_count_str)
            else:
                 self.message_manager.add_error(f"Full ffprobe scan returned invalid frame count: '{frame_count_str}'.")
                 return None
        except Exception as e:
            self.message_manager.add_error(f"Critical error during full ffprobe frame count: {e}")
            self.message_manager.add_error(traceback.format_exc())
            return None
            
    def tb_extract_frames(self, video_path, extraction_rate, progress=gr.Progress()):
        if video_path is None:
            self.message_manager.add_warning("No input video for frame extraction.")
            return None
        if not isinstance(extraction_rate, int) or extraction_rate < 1:
            self.message_manager.add_error("Extraction rate must be a positive integer (1 for all frames, N for every Nth frame).")
            return None

        resolved_video_path = str(Path(video_path).resolve())
        output_folder_name = self._tb_generate_output_folder_path(
            resolved_video_path,
            suffix=f"extracted_every_{extraction_rate}")
        os.makedirs(output_folder_name, exist_ok=True)

        self.message_manager.add_message(
            f"Starting frame extraction for {os.path.basename(resolved_video_path)} (every {extraction_rate} frame(s))."
        )
        self.message_manager.add_message(f"Outputting to: {output_folder_name}")

        reader = None
        try:
            total_frames = self._tb_get_video_frame_count(resolved_video_path)
            
            # If we know the total frames, we can provide accurate progress.
            if total_frames:
                progress(0, desc=f"Extracting 0 / {total_frames} frames...")
            else:
                self.message_manager.add_warning("Could not determine total frames. Progress will be indeterminate.")
                progress(0, desc="Extracting frames (total unknown)...")
                
            reader = imageio.get_reader(resolved_video_path)
            extracted_count = 0
            
            # --- MANUAL PROGRESS LOOP ---
            for i, frame in enumerate(reader):
                # Update progress manually every few frames to avoid overwhelming the UI
                if total_frames and i % 10 == 0:
                    progress(i / total_frames, desc=f"Extracting {i} / {total_frames} frames...")

                if i % extraction_rate == 0:
                    frame_filename = f"frame_{extracted_count:06d}.png"
                    output_frame_path = os.path.join(output_folder_name, frame_filename)
                    imageio.imwrite(output_frame_path, frame, format='PNG')
                    extracted_count += 1
            
            # --- FINAL UPDATE ---
            progress(1.0, desc="Extraction complete.")
            self.message_manager.add_success(f"Successfully extracted {extracted_count} frames to: {output_folder_name}")
            return output_folder_name

        except Exception as e:
            self.message_manager.add_error(f"Error during frame extraction: {e}")
            self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error during extraction.")
            return None
        finally:
            if reader:
                reader.close()
            gc.collect()
            
    def tb_get_extracted_frame_folders(self) -> list:
        if not os.path.exists(self.extracted_frames_target_path):
            self.message_manager.add_warning(f"Extracted frames directory not found: {self.extracted_frames_target_path}")
            return []
        try:
            folders = [
                d for d in os.listdir(self.extracted_frames_target_path)
                if os.path.isdir(os.path.join(self.extracted_frames_target_path, d))
            ]
            folders.sort()
            return folders
        except Exception as e:
            self.message_manager.add_error(f"Error scanning for extracted frame folders: {e}")
            return []

    def tb_delete_extracted_frames_folder(self, folder_name_to_delete: str) -> bool:
        if not folder_name_to_delete:
            self.message_manager.add_warning("No folder selected for deletion.")
            return False

        folder_path_to_delete = os.path.join(self.extracted_frames_target_path, folder_name_to_delete)

        if not os.path.exists(folder_path_to_delete) or not os.path.isdir(folder_path_to_delete):
            self.message_manager.add_error(f"Folder not found or is not a directory: {folder_path_to_delete}")
            return False

        try:
            shutil.rmtree(folder_path_to_delete)
            self.message_manager.add_success(f"Successfully deleted folder: {folder_name_to_delete}")
            return True
        except Exception as e:
            self.message_manager.add_error(f"Error deleting folder '{folder_name_to_delete}': {e}")
            self.message_manager.add_error(traceback.format_exc())
            return False

    def tb_reassemble_frames_to_video(self, frames_source, output_fps, output_base_name_override=None, progress=gr.Progress()):
        if not frames_source:
            self.message_manager.add_warning("No frames source (folder or files) provided for reassembly.")
            return None

        try:
            output_fps = int(output_fps)
            if output_fps <= 0:
                self.message_manager.add_error("Output FPS must be a positive number.")
                return None
        except ValueError:
            self.message_manager.add_error("Invalid FPS value for reassembly.")
            return None

        self.message_manager.add_message(f"Starting frame reassembly to video at {output_fps} FPS.")

        frame_info_list = []
        frames_data_prepared = False

        try:
            # This logic now primarily handles a directory path string
            if isinstance(frames_source, str) and os.path.isdir(frames_source):
                self.message_manager.add_message(f"Processing frames from directory: {frames_source}")
                # Use our existing function to get a sorted list of frame paths
                sorted_frame_paths = self.tb_get_frames_from_folder(os.path.basename(frames_source))
                for full_path in sorted_frame_paths:
                    frame_info_list.append({
                        'original_like_filename': os.path.basename(full_path),
                        'temp_path': full_path
                    })
            else:
                self.message_manager.add_error("Invalid frames_source type for reassembly. Expected a directory path.")
                return None

            if not frame_info_list:
                self.message_manager.add_warning("No valid image files found in the provided source to reassemble.")
                return None

            self.message_manager.add_message(f"Found {len(frame_info_list)} frames for reassembly.")

            output_file_basename = "reassembled_video"
            if output_base_name_override and isinstance(output_base_name_override, str) and output_base_name_override.strip():
                sanitized_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in output_base_name_override.strip())
                output_file_basename = Path(sanitized_name).stem
                if not output_file_basename: output_file_basename = "reassembled_video"
                self.message_manager.add_message(f"Using custom output video base name: {output_file_basename}")

            output_video_path = self._tb_generate_output_path(
                input_material_name=output_file_basename,
                suffix=f"{output_fps}fps_reassembled",
                target_dir=self.reassembled_video_target_path,
                ext=".mp4"
            )

            frames_data = []
            frames_data_prepared = True

            self.message_manager.add_message("Reading frame images (in sorted order)...")

            frame_iterator = frame_info_list
            if frame_info_list and progress is not None and hasattr(progress, 'tqdm'):
                 frame_iterator = progress.tqdm(frame_info_list, desc="Reading frames")

            for frame_info in frame_iterator:
                frame_actual_path = frame_info['temp_path']
                filename_for_log = frame_info['original_like_filename']
                try:
                    if not filename_for_log.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        self.message_manager.add_warning(f"Skipping non-standard image file: {filename_for_log}.")
                        continue
                    frames_data.append(imageio.imread(frame_actual_path))
                except Exception as e_read_frame:
                    self.message_manager.add_warning(f"Could not read frame ({filename_for_log}): {e_read_frame}. Skipping.")

            if not frames_data:
                self.message_manager.add_error("No valid frames could be successfully read for reassembly.")
                return None

            self.message_manager.add_message(f"Writing {len(frames_data)} frames to video: {output_video_path}")
            imageio.mimwrite(output_video_path, frames_data, fps=output_fps, quality=VIDEO_QUALITY, macro_block_size=None)

            self.message_manager.add_success(f"Successfully reassembled {len(frames_data)} frames into: {output_video_path}")
            return output_video_path

        except Exception as e:
            self.message_manager.add_error(f"Error during frame reassembly: {e}")
            self.message_manager.add_error(traceback.format_exc())
            if "Could not find a backend" in str(e) or "No such file or directory: 'ffmpeg'" in str(e).lower():
                 self.message_manager.add_error("This might indicate an issue with FFmpeg backend for imageio. Ensure 'imageio-ffmpeg' is installed or FFmpeg is in PATH.")
            return None
        finally:
            if frames_data_prepared and 'frames_data' in locals():
                del frames_data
            gc.collect()

    def _tb_get_video_duration(self, video_path: str) -> str | None:
        """Uses ffprobe to get the duration of a video file as a string."""
        if not self.has_ffprobe:
            return None
        try:
            ffprobe_cmd = [
                self.ffprobe_exe, "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True, errors='ignore')
            return result.stdout.strip()
        except Exception:
            return None
            
    def tb_join_videos(self, video_paths: list, output_base_name_override=None, progress=gr.Progress()):
        if not video_paths or len(video_paths) < 2:
            self.message_manager.add_warning("Please select at least two videos to join.")
            return None
        if not self.has_ffmpeg:
            self.message_manager.add_error("FFmpeg is required for joining videos. This operation cannot proceed.")
            return None

        self.message_manager.add_message(f"🚀 Starting video join process for {len(video_paths)} videos...")
        progress(0.1, desc="Analyzing input videos...")

        # --- 1. STANDARDIZE DIMENSIONS ---
        # Get dimensions of the first video to use as the standard for all others.
        first_video_dims = self._tb_get_video_dimensions(video_paths[0])
        if not all(first_video_dims):
            self.message_manager.add_error("Could not determine dimensions of the first video. Cannot proceed.")
            return None
        target_w, target_h = first_video_dims
        self.message_manager.add_message(f"Standardizing all videos to {target_w}x{target_h} for joining.")

        # --- 2. BUILD THE FFMPEG COMMAND ---
        ffmpeg_cmd = [self.ffmpeg_exe, "-y", "-loglevel", "error"]
        filter_complex_parts = []
        video_stream_labels = []
        audio_stream_labels = []

        # Loop through each input video to prepare its streams.
        for i, path in enumerate(video_paths):
            ffmpeg_cmd.extend(["-i", str(Path(path).resolve())])
            
            # --- VIDEO STREAM PREPARATION ---
            video_label = f"v{i}"
            # Scale video, pad to fit, set aspect ratio, and ensure standard pixel format.
            filter_complex_parts.append(
                f"[{i}:v:0]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:-1:-1:color=black,setsar=1,format=yuv420p[{video_label}]"
            )
            video_stream_labels.append(f"[{video_label}]")
            
            # --- AUDIO STREAM PREPARATION ---
            audio_label = f"a{i}"
            if self._tb_has_audio_stream(path):
                # If audio exists, standardize it to a common format.
                filter_complex_parts.append(
                    f"[{i}:a:0]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[{audio_label}]"
                )
            else:
                # If no audio, get the video's duration first.
                duration = self._tb_get_video_duration(path)
                if duration:
                    # Then, generate a silent audio track of that exact duration.
                    self.message_manager.add_message(f"'{Path(path).name}' has no audio. Generating silent track of {float(duration):.2f}s.", "INFO")
                    filter_complex_parts.append(
                        f"anullsrc=channel_layout=stereo:sample_rate=44100,atrim=duration={duration}[{audio_label}]"
                    )
                else:
                    # If we can't get duration, we can't create a silent track, so we must skip it.
                    self.message_manager.add_warning(f"Could not get duration for '{Path(path).name}' to generate silent audio. This track's audio will be skipped.")
                    continue 
            audio_stream_labels.append(f"[{audio_label}]")

        # --- 3. CONCATENATE THE STREAMS ---
        # Join all the prepared video and audio streams together into final output streams.
        filter_complex_parts.append(f"{''.join(video_stream_labels)}concat=n={len(video_paths)}:v=1:a=0[outv]")
        
        # Only add the audio concat filter if we successfully prepared audio streams.
        if audio_stream_labels:
            filter_complex_parts.append(f"{''.join(audio_stream_labels)}concat=n={len(audio_stream_labels)}:v=0:a=1[outa]")
        
        final_filter_complex = ";".join(filter_complex_parts)
        ffmpeg_cmd.extend(["-filter_complex", final_filter_complex])

        # --- 4. MAP AND ENCODE THE FINAL VIDEO ---
        # Map the final concatenated video stream to the output.
        ffmpeg_cmd.extend(["-map", "[outv]"])
        # If we have a final audio stream, map that too.
        if audio_stream_labels:
            ffmpeg_cmd.extend(["-map", "[outa]"])

        # Determine the output filename.
        if output_base_name_override and isinstance(output_base_name_override, str) and output_base_name_override.strip():
             sanitized_name = "".join(c for c in output_base_name_override.strip() if c.isalnum() or c in (' ', '_', '-')).strip()
             base_name_to_use = Path(sanitized_name).stem if sanitized_name else Path(video_paths[0]).stem
        else:
            base_name_to_use = Path(video_paths[0]).stem
            
        output_path = self._tb_generate_output_path(
            base_name_to_use,
            suffix=f"joined_{len(video_paths)}_videos",
            target_dir=self.toolbox_video_output_dir
        )
        
        # Set standard, high-compatibility encoding options.
        ffmpeg_cmd.extend([
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", output_path
        ])

        # --- 5. EXECUTE THE COMMAND ---
        try:
            self.message_manager.add_message("Running FFmpeg to join videos. This may take a while...")
            progress(0.5, desc=f"Joining {len(video_paths)} videos...")
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore')
            
            progress(1.0, desc="Join complete.")
            self.message_manager.add_success(f"✅ Videos successfully joined! Output: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e_join:
            self._tb_log_ffmpeg_error(e_join, "video joining")
            return None
        except Exception as e:
            self.message_manager.add_error(f"An unexpected error occurred during video joining: {e}")
            self.message_manager.add_error(traceback.format_exc())
            return None
        finally:
            gc.collect()

    def _tb_clean_filename(self, filename):
        filename = re.sub(r'_\d{6}_\d{6}', '', filename) # Example timestamp pattern
        filename = re.sub(r'_\d{6}_\d{4}', '', filename) # Another example
        return filename.strip('_')

    def tb_export_video(self, video_path: str, export_format: str, quality_slider: int, max_width: int,
                        output_base_name_override=None, progress=gr.Progress()):
        if not video_path:
            self.message_manager.add_warning("No input video for exporting.")
            return None
        if not self.has_ffmpeg:
            self.message_manager.add_error("FFmpeg is required for exporting. This operation cannot proceed.")
            return None

        self.message_manager.add_message(f"🚀 Starting export to {export_format.upper()}...")
        progress(0, desc=f"Preparing to export to {export_format.upper()}...")
        
        resolved_video_path = str(Path(video_path).resolve())
        
        # --- Base FFmpeg Command ---
        ffmpeg_cmd = [self.ffmpeg_exe, "-y", "-loglevel", "error", "-i", resolved_video_path]
        
        # --- Video Filters (Resizing) ---
        vf_parts = []
        # The scale filter resizes while maintaining aspect ratio. '-2' ensures the height is an even number for codec compatibility.
        vf_parts.append(f"scale={max_width}:-2")
        
        # --- Format-Specific Settings ---
        ext = f".{export_format.lower()}"
        
        if export_format == "MP4":
            # CRF (Constant Rate Factor) is the quality setting for x264. Lower is higher quality.
            # We map our 0-100 slider to a good CRF range (e.g., 28 (low) to 18 (high)).
            crf_value = int(28 - (quality_slider / 100) * 10)
            self.message_manager.add_message(f"Exporting MP4 with CRF: {crf_value} (Quality: {quality_slider}%)")
            ffmpeg_cmd.extend(["-c:v", "libx264", "-crf", str(crf_value), "-preset", "medium"])
            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "128k"]) # Keep audio, but compress it
            
        elif export_format == "WebM":
            # Similar to MP4, but for the VP9 codec. A good CRF range is ~35 (low) to 25 (high).
            crf_value = int(35 - (quality_slider / 100) * 10)
            self.message_manager.add_message(f"Exporting WebM with CRF: {crf_value} (Quality: {quality_slider}%)")
            ffmpeg_cmd.extend(["-c:v", "libvpx-vp9", "-crf", str(crf_value), "-b:v", "0"])
            ffmpeg_cmd.extend(["-c:a", "libopus", "-b:a", "96k"]) # Use Opus for WebM audio

        elif export_format == "GIF":
            # High-quality GIF generation is a two-pass process.
            self.message_manager.add_message("Generating high-quality GIF (2-pass)...")
            # Pass 1: Generate a color palette.
            palette_path = os.path.join(self._base_temp_output_dir, f"palette_{Path(video_path).stem}.png")
            vf_parts.append("split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
            ffmpeg_cmd.extend(["-an"]) # No audio in GIFs

        if vf_parts:
            ffmpeg_cmd.extend(["-vf", ",".join(vf_parts)])

        # --- Output Path ---
        if output_base_name_override and isinstance(output_base_name_override, str) and output_base_name_override.strip():
             sanitized_name = "".join(c for c in output_base_name_override.strip() if c.isalnum() or c in (' ', '_', '-')).strip()
             base_name_to_use = Path(sanitized_name).stem if sanitized_name else Path(video_path).stem
        else:
            base_name_to_use = Path(video_path).stem

        output_path = self._tb_generate_output_path(
            base_name_to_use,
            suffix=f"exported_{quality_slider}q_{max_width}w",
            target_dir=self.toolbox_video_output_dir,
            ext=ext
        )
        ffmpeg_cmd.append(output_path)

        # --- Execute ---
        try:
            progress(0.5, desc=f"Encoding to {export_format.upper()}...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore')
            progress(1.0, desc="Export complete!")
            self.message_manager.add_success(f"✅ Successfully exported to {export_format.upper()}! Output: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            self._tb_log_ffmpeg_error(e, f"export to {export_format.upper()}")
            return None
        except Exception as e:
            self.message_manager.add_error(f"An unexpected error occurred during export: {e}")
            self.message_manager.add_error(traceback.format_exc())
            return None
        finally:
            gc.collect()
            
    def _tb_generate_output_path(self, input_material_name, suffix, target_dir, ext=".mp4"):
        base_name = Path(input_material_name).stem 
        if not base_name: base_name = "untitled_video" 
        cleaned_name = self._tb_clean_filename(base_name)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"{cleaned_name}_{suffix}_{timestamp}{ext}"
        return os.path.join(target_dir, filename)
    
    def _tb_generate_output_folder_path(self, input_video_path, suffix):
        base_name = Path(input_video_path).stem
        if not base_name: base_name = "untitled_video_frames"
        cleaned_name = self._tb_clean_filename(base_name)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        folder_name = f"{cleaned_name}_{suffix}_{timestamp}"
        return os.path.join(self.extracted_frames_target_path, folder_name)

    def tb_copy_video_to_permanent_storage(self, temp_video_path):
        if not temp_video_path or not os.path.exists(temp_video_path):
            self.message_manager.add_error("No video file provided or file does not exist to save.")
            return temp_video_path 

        try:
            video_filename = Path(temp_video_path).name
            permanent_video_path = os.path.join(self.toolbox_permanent_save_dir, video_filename)
            os.makedirs(self.toolbox_permanent_save_dir, exist_ok=True)
            self.message_manager.add_message(f"Copying '{video_filename}' to permanent storage: '{permanent_video_path}'")
            shutil.copy2(temp_video_path, permanent_video_path)
            self.message_manager.add_success(f"Video saved to: {permanent_video_path}")
            return permanent_video_path
        except Exception as e:
            self.message_manager.add_error(f"Error saving video to permanent storage: {e}")
            self.message_manager.add_error(traceback.format_exc())
            return temp_video_path
            
    def tb_analyze_video_input(self, video_path):
        if video_path is None:
            self.message_manager.add_warning("No video provided for analysis.")
            return "Please upload a video."
        
        resolved_video_path = str(Path(video_path).resolve())
        analysis_report_lines = [] # Use a list to build the report string

        file_size_bytes = 0
        file_size_display = "N/A"
        try:
            if os.path.exists(resolved_video_path):
                file_size_bytes = os.path.getsize(resolved_video_path)
                if file_size_bytes < 1024:
                    file_size_display = f"{file_size_bytes} B"
                elif file_size_bytes < 1024**2:
                    file_size_display = f"{file_size_bytes/1024:.2f} KB"
                elif file_size_bytes < 1024**3:
                    file_size_display = f"{file_size_bytes/1024**2:.2f} MB"
                else:
                    file_size_display = f"{file_size_bytes/1024**3:.2f} GB"
        except Exception as e:
            self.message_manager.add_warning(f"Could not get file size: {e}")
            
        # Variables to hold parsed info, initialized to defaults
        video_width, video_height = 0, 0
        num_frames_value = None # For the upscale warning
        duration_display, fps_display, resolution_display, nframes_display, has_audio_str = "N/A", "N/A", "N/A", "N/A", "No"
        analysis_source = "imageio" # Default analysis source

        if self.has_ffprobe:
            self.message_manager.add_message(f"Analyzing video with ffprobe: {os.path.basename(video_path)}")
            try:
                probe_cmd = [
                    self.ffprobe_exe, "-v", "error", "-show_format", "-show_streams",
                    "-of", "json", resolved_video_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                probe_data = json.loads(result.stdout)
                
                video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
                audio_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)

                if not video_stream:
                    self.message_manager.add_error("No video stream found in the file (ffprobe).")

                else:
                    analysis_source = "ffprobe"
                    duration_str = probe_data.get("format", {}).get("duration", "0") 
                    duration = float(duration_str) if duration_str and duration_str.replace('.', '', 1).isdigit() else 0.0
                    duration_display = f"{duration:.2f} seconds"

                    r_frame_rate_str = video_stream.get("r_frame_rate", "0/0")
                    avg_frame_rate_str = video_stream.get("avg_frame_rate", "0/0")
                    calculated_fps = 0.0

                    def parse_fps(fps_s):
                        if isinstance(fps_s, (int, float)): return float(fps_s)
                        if isinstance(fps_s, str) and "/" in fps_s:
                            try: num, den = map(float, fps_s.split('/')); return num / den if den != 0 else 0.0
                            except ValueError: return 0.0
                        try: return float(fps_s) 
                        except ValueError: return 0.0

                    r_fps_val = parse_fps(r_frame_rate_str); avg_fps_val = parse_fps(avg_frame_rate_str)

                    if r_fps_val > 0: calculated_fps = r_fps_val; fps_display = f"{r_fps_val:.2f} FPS"
                    if avg_fps_val > 0 and abs(r_fps_val - avg_fps_val) > 0.01 : # Only show average if meaningfully different
                        calculated_fps = avg_fps_val # Prefer average if it's different and valid
                        fps_display = f"{avg_fps_val:.2f} FPS (Avg, r: {r_fps_val:.2f})" 
                    elif avg_fps_val > 0 and r_fps_val <=0: 
                        calculated_fps = avg_fps_val; fps_display = f"{avg_fps_val:.2f} FPS (Average)"
                    
                    video_width = video_stream.get("width", 0)
                    video_height = video_stream.get("height", 0)
                    resolution_display = f"{video_width}x{video_height}" if video_width and video_height else "N/A"

                    nframes_str_probe = video_stream.get("nb_frames")
                    if nframes_str_probe and nframes_str_probe.isdigit():
                        num_frames_value = int(nframes_str_probe)
                        nframes_display = str(num_frames_value)
                    elif duration > 0 and calculated_fps > 0:
                        num_frames_value = int(duration * calculated_fps)
                        nframes_display = f"{num_frames_value} (Calculated)"
                    
                    if audio_stream:
                        has_audio_str = (f"Yes (Codec: {audio_stream.get('codec_name', 'N/A')}, "
                                         f"Channels: {audio_stream.get('channels', 'N/A')}, "
                                         f"Rate: {audio_stream.get('sample_rate', 'N/A')} Hz)")
                    self.message_manager.add_success("Video analysis complete (using ffprobe).")

            except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e_ffprobe:
                self.message_manager.add_warning(f"ffprobe analysis failed ({type(e_ffprobe).__name__}). Trying imageio fallback.")
                if isinstance(e_ffprobe, subprocess.CalledProcessError):
                    self._tb_log_ffmpeg_error(e_ffprobe, "video analysis with ffprobe")
                analysis_source = "imageio" # Ensure fallback if ffprobe fails midway
        
        if analysis_source == "imageio": # Either ffprobe not available, or it failed
            self.message_manager.add_message(f"Analyzing video with imageio: {os.path.basename(video_path)}")
            reader = None
            try:
                reader = imageio.get_reader(resolved_video_path)
                meta = reader.get_meta_data()
                
                duration_imgio_val = meta.get('duration')
                duration_display = f"{float(duration_imgio_val):.2f} seconds" if duration_imgio_val is not None else "N/A"
                
                fps_val_imgio = meta.get('fps')
                fps_display = f"{float(fps_val_imgio):.2f} FPS" if fps_val_imgio is not None else "N/A"
                
                size_imgio = meta.get('size')
                if isinstance(size_imgio, tuple) and len(size_imgio) == 2:
                    video_width, video_height = int(size_imgio[0]), int(size_imgio[1])
                    resolution_display = f"{video_width}x{video_height}"
                else:
                    resolution_display = "N/A"

                nframes_val_imgio_meta = meta.get('nframes') 
                if nframes_val_imgio_meta not in [float('inf'), "N/A", None] and isinstance(nframes_val_imgio_meta, (int,float)):
                    num_frames_value = int(nframes_val_imgio_meta)
                    nframes_display = str(num_frames_value)
                elif hasattr(reader, 'count_frames'):
                    try: 
                        nframes_val_imgio_count = reader.count_frames()
                        if nframes_val_imgio_count != float('inf'):
                             num_frames_value = int(nframes_val_imgio_count)
                             nframes_display = f"{num_frames_value} (Counted)"
                        else: nframes_display = "Unknown (Stream or very long)"
                    except Exception: nframes_display = "Unknown (Frame count failed)"
                
                has_audio_str = "(Audio info not available via imageio)"
                self.message_manager.add_success("Video analysis complete (using imageio).")
            except Exception as e_imgio:
                self.message_manager.add_error(f"Error analyzing video with imageio: {e_imgio}")
                import traceback
                self.message_manager.add_error(traceback.format_exc())
                return f"Error analyzing video: Both ffprobe (if attempted) and imageio failed."
            finally:
                if reader: reader.close()

        # --- Construct Main Analysis Report ---
        analysis_report_lines.append(f"Video Analysis ({analysis_source}):")
        analysis_report_lines.append(f"File: {os.path.basename(video_path)}")
        analysis_report_lines.append("------------------------------------")
        analysis_report_lines.append(f"File Size: {file_size_display}")
        analysis_report_lines.append(f"Duration: {duration_display}")
        analysis_report_lines.append(f"Frame Rate: {fps_display}")
        analysis_report_lines.append(f"Resolution: {resolution_display}")
        analysis_report_lines.append(f"Frames: {nframes_display}")
        analysis_report_lines.append(f"Audio: {has_audio_str}")
        analysis_report_lines.append(f"Source: {video_path}")

        # --- Append UPSCALE ADVISORY Conditionally ---
        if video_width > 0 and video_height > 0: # Ensure we have dimensions
            HD_WIDTH_THRESHOLD = 1920
            FOUR_K_WIDTH_THRESHOLD = 3800 
            
            is_hd_or_larger = (video_width >= HD_WIDTH_THRESHOLD or video_height >= (HD_WIDTH_THRESHOLD * 9/16 * 0.95)) # Adjusted height for aspect ratios
            is_4k_or_larger = (video_width >= FOUR_K_WIDTH_THRESHOLD or video_height >= (FOUR_K_WIDTH_THRESHOLD * 9/16 * 0.95))

            upscale_warnings = []
            if is_4k_or_larger:
                upscale_warnings.append(
                    "This video is 4K resolution or higher. Upscaling (e.g., to 8K+) will be very "
                    "slow, memory-intensive, and may cause issues. Proceed with caution."
                )
            elif is_hd_or_larger:
                upscale_warnings.append(
                    "This video is HD or larger. Upscaling (e.g., to 4K+) will be resource-intensive "
                    "and slow. Ensure your system is prepared."
                )
            
            if num_frames_value and num_frames_value > 900: # e.g., > 30 seconds at 30fps
                 upscale_warnings.append(
                    f"With {num_frames_value} frames, upscaling will also be very time-consuming."
                )

            if upscale_warnings:
                analysis_report_lines.append("\n--- UPSCALE ADVISORY ---")
                for warning_msg in upscale_warnings:
                    analysis_report_lines.append(f"⚠️ {warning_msg}")
                # analysis_report_lines.append("------------------------") # Optional closing separator
        
        return "\n".join(analysis_report_lines)


    def _tb_has_audio_stream(self, video_path_to_check):
        if not self.has_ffprobe: # Critical check
            self.message_manager.add_warning(
                "FFprobe not available. Cannot reliably determine if video has audio. "
                "Assuming no audio for operations requiring this check. "
                "Install FFmpeg with ffprobe for full audio support."
            )
            return False
        try:
            resolved_path = str(Path(video_path_to_check).resolve())
            ffprobe_cmd = [
                self.ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0", resolved_path
            ]
            # check=False because a non-zero return often means no audio stream, which is a valid outcome here.
            audio_check_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False, errors='ignore') 

            if audio_check_result.returncode == 0 and "audio" in audio_check_result.stdout.strip().lower():
                return True
            else:
                # Optionally log if ffprobe ran but found no audio, or if it errored for other reasons
                # if audio_check_result.returncode != 0 and audio_check_result.stderr:
                #    self.message_manager.add_message(f"FFprobe check for audio stream in {os.path.basename(video_path_to_check)} completed. Stderr: {audio_check_result.stderr.strip()}", "DEBUG")
                return False
        except FileNotFoundError: 
            self.message_manager.add_warning("FFprobe executable not found during audio stream check (should have been caught by self.has_ffprobe). Assuming no audio.")
            return False # Should ideally not happen if self.has_ffprobe is true and self.ffprobe_exe is set
        except Exception as e:
            self.message_manager.add_warning(f"Error checking for audio stream in {os.path.basename(video_path_to_check)}: {e}. Assuming no audio.")
            return False
            
    def tb_process_frames(self, video_path, target_fps_mode, speed_factor, progress=gr.Progress()):
        if video_path is None: self.message_manager.add_warning("No input video for frame processing."); return None
        
        # Core video processing relies on imageio for reading/writing frames, RIFE for interpolation.
        # FFmpeg is primarily for audio handling here.
        
        final_output_path = None 
        try:
            interpolation_factor = 1
            if "2x" in target_fps_mode: interpolation_factor = 2
            elif "4x" in target_fps_mode: interpolation_factor = 4
            should_interpolate = interpolation_factor > 1

            self.message_manager.add_message(
                f"Starting frame processing for {os.path.basename(video_path)}: "
                f"FPS Mode: {target_fps_mode}, Speed: {speed_factor}x"
            )
            progress(0, desc="Initializing...")
            resolved_video_path = str(Path(video_path).resolve())

            self.message_manager.add_message("Reading video frames...")
            progress(0.05, desc="Reading video...")
            reader = imageio.get_reader(resolved_video_path)
            original_fps = reader.get_meta_data().get('fps', 30.0) # Default if not found
            video_frames = [frame for frame in reader]
            reader.close()
            self.message_manager.add_message(f"Read {len(video_frames)} frames at {original_fps} FPS.")

            processed_frames = video_frames
            current_fps = original_fps # This will be the FPS for the *output video stream*

            if speed_factor != 1.0:
                self.message_manager.add_message(f"Adjusting speed by {speed_factor}x (frame sampling/duplication)...")
                progress(0.2, desc="Adjusting speed...")
                if speed_factor > 1.0: 
                    indices = np.arange(0, len(video_frames), speed_factor).astype(int)
                    processed_frames = [video_frames[i] for i in indices if i < len(video_frames)]
                else: 
                    new_len = int(len(video_frames) / speed_factor)
                    indices = np.linspace(0, len(video_frames) - 1, new_len).astype(int)
                    processed_frames = [video_frames[i] for i in indices]
                self.message_manager.add_message(f"Speed adjustment (sampling) resulted in {len(processed_frames)} frames.")
            
            if should_interpolate and len(processed_frames) > 1:
                self.message_manager.add_message(f"Attempting to load RIFE model for {interpolation_factor}x interpolation...")
                if not self.rife_handler._ensure_model_downloaded_and_loaded():
                    self.message_manager.add_error("RIFE model could not be loaded. Skipping interpolation.")
                else:
                    self.message_manager.add_message(f"RIFE model loaded. Starting RIFE {interpolation_factor}x interpolation...")
                    
                    # Loop for multiple passes of 2x interpolation (1 pass for 2x, 2 passes for 4x)
                    num_passes = int(math.log2(interpolation_factor))
                    for p in range(num_passes):
                        self.message_manager.add_message(f"RIFE Pass {p+1}/{num_passes}: Interpolating frames...")
                        interpolated_this_pass = []
                        num_pairs = len(processed_frames) - 1
                    
                        # Use tqdm for the innermost loop of each pass
                        for i in progress.tqdm(range(num_pairs), desc=f"RIFE Pass {p+1}/{num_passes} ({interpolation_factor}x)"):
                            frame1_np, frame2_np = processed_frames[i], processed_frames[i+1]
                            interpolated_this_pass.append(frame1_np) 
                            middle_frame_np = self.rife_handler.interpolate_between_frames(frame1_np, frame2_np)
                            if middle_frame_np is not None: interpolated_this_pass.append(middle_frame_np)
                            else: interpolated_this_pass.append(frame1_np) # Duplicate on failure
                        
                        interpolated_this_pass.append(processed_frames[-1])
                        processed_frames = interpolated_this_pass # Update for the next pass or for final output
                    # The video stream FPS itself doesn't change due to RIFE; it just has more frames.
                    # If RIFE is used, the perceived playback smoothness increases as if FPS doubled.
                    # The container FPS (current_fps) should reflect the intended playback rate of these frames.
                    # If original FPS was 30, and we RIFE, we now have 2x frames intended to still play over
                    # the same original duration segment, effectively meaning playback at 2*original_fps.
                    current_fps = original_fps * interpolation_factor 
                    self.message_manager.add_message(f"RIFE {interpolation_factor}x interpolation resulted in {len(processed_frames)} frames. Effective FPS: {current_fps:.2f}")
            
            elif should_interpolate and len(processed_frames) <= 1:
                self.message_manager.add_warning("Not enough frames for RIFE interpolation. Skipping.")

            op_suffix_parts = []
            if speed_factor != 1.0: op_suffix_parts.append(f"speed{speed_factor:.2f}x".replace('.',',')) 
            if should_interpolate and self.rife_handler.rife_model is not None: 
                op_suffix_parts.append(f"RIFE{interpolation_factor}x")

            op_suffix = "_".join(op_suffix_parts)

            temp_video_suffix = f"{op_suffix}_temp_video"
            video_stream_output_path = self._tb_generate_output_path(
                resolved_video_path, suffix=temp_video_suffix, target_dir=self.toolbox_video_output_dir
            )
            final_muxed_output_path = video_stream_output_path.replace("_temp_video", "")

            self.message_manager.add_message(f"Saving video stream to {video_stream_output_path} at {current_fps:.2f} FPS...")
            progress(0.85, desc="Saving video stream...")
            imageio.mimwrite(video_stream_output_path, processed_frames, fps=current_fps, quality=VIDEO_QUALITY, macro_block_size=None)

            final_output_path = final_muxed_output_path 
            can_process_audio = self.has_ffmpeg
            original_video_has_audio = self._tb_has_audio_stream(resolved_video_path) if can_process_audio else False

            if can_process_audio and original_video_has_audio:
                self.message_manager.add_message("Original video has audio. Processing audio with FFmpeg...")
                progress(0.9, desc="Processing audio...")
                ffmpeg_mux_cmd = [self.ffmpeg_exe, "-y", "-loglevel", "error", "-i", video_stream_output_path]
                
                audio_filters = []
                if speed_factor != 1.0: # Only apply atempo if speed actually changed
                    # Complex atempo for large speed changes (FFmpeg's atempo is 0.5-100.0)
                    # This simplified version handles common cases. For extreme speed_factor, might need more atempo stages.
                    if 0.5 <= speed_factor <= 100.0:
                        audio_filters.append(f"atempo={speed_factor:.4f}")
                    elif speed_factor < 0.5: # Needs multiple 0.5 steps
                        num_half_steps = int(np.ceil(np.log(speed_factor) / np.log(0.5)))
                        for _ in range(num_half_steps): audio_filters.append("atempo=0.5")
                        final_factor = speed_factor / (0.5**num_half_steps)
                        if abs(final_factor - 1.0) > 1e-4 and 0.5 <= final_factor <= 100.0: # Add final adjustment if needed
                             audio_filters.append(f"atempo={final_factor:.4f}")
                    elif speed_factor > 100.0: # Needs multiple 2.0 (or higher, like 100.0) steps
                        num_double_steps = int(np.ceil(np.log(speed_factor / 100.0) / np.log(2.0))) # Example for steps of 2 after 100
                        audio_filters.append("atempo=100.0") # Max one step
                        remaining_factor = speed_factor / 100.0
                        if abs(remaining_factor - 1.0) > 1e-4 and 0.5 <= remaining_factor <= 100.0:
                             audio_filters.append(f"atempo={remaining_factor:.4f}")


                    self.message_manager.add_message(f"Applying audio speed adjustment with atempo: {','.join(audio_filters) if audio_filters else 'None (speed_factor out of simple atempo range or 1.0)'}")

                ffmpeg_mux_cmd.extend(["-i", resolved_video_path]) # Input for audio
                ffmpeg_mux_cmd.extend(["-c:v", "copy"]) 
                
                if audio_filters:
                    ffmpeg_mux_cmd.extend(["-filter:a", ",".join(audio_filters)])
                # Always re-encode audio to AAC for MP4 compatibility, even if no speed change,
                # as original audio might not be AAC.
                ffmpeg_mux_cmd.extend(["-c:a", "aac", "-b:a", "192k"]) 

                ffmpeg_mux_cmd.extend(["-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_muxed_output_path])
                
                try:
                    subprocess.run(ffmpeg_mux_cmd, check=True, capture_output=True, text=True)
                    self.message_manager.add_success(f"Video saved with processed audio: {final_muxed_output_path}")
                except subprocess.CalledProcessError as e_mux:
                    self._tb_log_ffmpeg_error(e_mux, "audio processing/muxing")
                    self.message_manager.add_message("Saving video without audio as fallback.")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path) 
                    os.rename(video_stream_output_path, final_muxed_output_path) 
                except FileNotFoundError: # Should not happen if self.has_ffmpeg is true
                    self.message_manager.add_error(f"FFmpeg not found during muxing. This is unexpected if has_ffmpeg was True.")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path)
                    os.rename(video_stream_output_path, final_muxed_output_path)
            else: 
                if original_video_has_audio and not can_process_audio:
                     self.message_manager.add_warning("Original video has audio, but FFmpeg is not available to process it. Output will be silent. Install FFmpeg for audio support.")
                elif not original_video_has_audio:
                     self.message_manager.add_message("No audio in original or audio processing skipped (e.g. FFprobe missing for detection). Saving video-only.")
                
                if os.path.exists(final_muxed_output_path) and final_muxed_output_path != video_stream_output_path : 
                    os.remove(final_muxed_output_path) 
                os.rename(video_stream_output_path, final_muxed_output_path)


            if os.path.exists(video_stream_output_path) and video_stream_output_path != final_muxed_output_path:
                try: os.remove(video_stream_output_path)
                except Exception as e_clean: self.message_manager.add_warning(f"Could not remove temp video file {video_stream_output_path}: {e_clean}")
            
            progress(1.0, desc="Complete.")
            self.message_manager.add_success(f"Frame processing complete: {final_output_path}")
            return final_output_path

        except Exception as e:
            self.message_manager.add_error(f"Error during frame processing: {e}")
            import traceback; self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error.")
            return None 
        finally:
            if self.rife_handler: self.rife_handler.unload_model()
            devicetorch.empty_cache(torch); gc.collect()

    def tb_create_loop(self, video_path, loop_type, num_loops, progress=gr.Progress()):
        if video_path is None: self.message_manager.add_warning("No input video for loop creation."); return None
        if not self.has_ffmpeg: # FFmpeg is essential for this function's stream_loop and complex filter
            self.message_manager.add_error("FFmpeg is required for creating video loops. This operation cannot proceed.")
            return video_path # Return original video path
        if loop_type == "none": self.message_manager.add_message("Loop type 'none'. No action."); return video_path

        progress(0, desc="Initializing loop creation...")
        resolved_video_path = str(Path(video_path).resolve())
        output_path = self._tb_generate_output_path(
            resolved_video_path, 
            suffix=f"{loop_type}_{num_loops}x",
            target_dir=self.toolbox_video_output_dir
        )
        
        self.message_manager.add_message(f"Creating {loop_type} ({num_loops}x) for {os.path.basename(resolved_video_path)}...")
        
        ping_pong_unit_path = None 
        original_video_has_audio = self._tb_has_audio_stream(resolved_video_path) # Check once

        try:
            progress(0.2, desc=f"Preparing {loop_type} loop...")
            if loop_type == "ping-pong":
                ping_pong_unit_path = self._tb_generate_output_path(
                    resolved_video_path, 
                    suffix="pingpong_unit_temp", 
                    target_dir=self.toolbox_video_output_dir
                )
                # Create video-only ping-pong unit first
                ffmpeg_pp_unit_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", resolved_video_path,
                    "-vf", "split[main][tmp];[tmp]reverse[rev];[main][rev]concat=n=2:v=1:a=0", # Video only
                    "-an", ping_pong_unit_path
                ]
                subprocess.run(ffmpeg_pp_unit_cmd, check=True, capture_output=True, text=True)
                self.message_manager.add_message(f"Created ping-pong unit (video-only): {ping_pong_unit_path}")

                ffmpeg_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-stream_loop", str(num_loops - 1), # Loop the video unit N-1 times (total N plays)
                    "-i", ping_pong_unit_path,
                    "-c:v", "copy"
                ]
                if original_video_has_audio:
                    self.message_manager.add_message("Original video has audio. Will loop audio for ping-pong.")
                    # Audio duration needs to match 2T * num_loops. FFmpeg aloop is complex.
                    # A simpler approach for ping-pong audio might be to create a 2T audio segment (original + reversed original)
                    # then loop that audio segment N times and mux with the N-times looped ping-pong video.
                    # Current FFmpeg command for aloop:
                    audio_loop_count_for_ffmpeg = (num_loops * 2) -1 # Total plays of original audio needed, minus one for initial play by -i
                    ffmpeg_cmd.extend([
                        "-i", resolved_video_path, # Original video for audio source
                        "-filter_complex", f"[1:a]areverse[areva];[1:a][areva]concat=n=2:v=0:a=1[ppa];[ppa]aloop=loop={num_loops-1}:size=2147483647[a_looped]",
                        "-map", "0:v:0", "-map", "[a_looped]",
                        "-c:a", "aac", "-b:a", "192k", "-shortest"
                    ])
                else:
                    self.message_manager.add_message("No audio in original or detection issue. Creating video-only ping-pong loop.")
                    ffmpeg_cmd.extend(["-an"])
                ffmpeg_cmd.append(output_path)

            else: # Regular 'loop'
                
                # Hi Gemini/Claude!
                # The 'num_loops' parameter comes from the UI slider labeled "Number of Loops/Repeats".
                # The user's intent is that this slider value (num_loops) represents the number of 
                # *additional* times the video should be repeated after its initial playthrough.
                # For example, if num_loops = 1 (from slider), the video should play twice (original + 1 repeat).
                #
                # FFmpeg's -stream_loop option takes a value (let's call it X_ffmpeg), 
                # meaning the input is looped X_ffmpeg times *in addition* to the first play.
                # So, X_ffmpeg should be equal to the slider value 'num_loops'.
                
                ffmpeg_stream_loop_value = num_loops 
                
                # Ensure ffmpeg_stream_loop_value is non-negative.
                # Given the UI slider minimum is typically 1, num_loops should always be >= 1.
                # This check is for robustness if the input num_loops could ever be less than 0
                # (e.g., if UI constraints change or input comes from elsewhere).
                if ffmpeg_stream_loop_value < 0: 
                    ffmpeg_stream_loop_value = 0 # Should ideally not be hit if slider min is 1.

                # Total plays will be the original play + ffmpeg_stream_loop_value additional plays.
                total_plays = ffmpeg_stream_loop_value + 1
                self.message_manager.add_message(
                    f"Regular loop: original video + {ffmpeg_stream_loop_value} additional repeat(s). Total {total_plays} plays."
                )
                
                ffmpeg_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-stream_loop", str(ffmpeg_stream_loop_value), # This now uses num_loops directly
                    "-i", resolved_video_path,
                    "-c:v", "copy" 
                ]
                if original_video_has_audio:
                    self.message_manager.add_message("Original video has audio. Re-encoding to AAC for looped MP4 (if not already AAC).")
                    ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "0:a:0?"])
                else:
                    self.message_manager.add_message("No audio in original or detection issue. Looped video will be silent.")
                    ffmpeg_cmd.extend(["-an", "-map", "0:v:0"])
                ffmpeg_cmd.append(output_path)
            
            self.message_manager.add_message(f"Processing video {loop_type} with FFmpeg...")
            progress(0.5, desc=f"Running FFmpeg for {loop_type}...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore')

            progress(1.0, desc=f"{loop_type.capitalize()} loop created successfully.")
            self.message_manager.add_success(f"Loop creation complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e_loop:
            self._tb_log_ffmpeg_error(e_loop, f"{loop_type} creation")
            progress(1.0, desc=f"Error creating {loop_type}.")
            return None
        except Exception as e:
            self.message_manager.add_error(f"Error creating loop: {e}")
            import traceback; self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error creating loop.")
            return None
        finally:
            if ping_pong_unit_path and os.path.exists(ping_pong_unit_path):
                try: os.remove(ping_pong_unit_path)
                except Exception as e_clean_pp: self.message_manager.add_warning(f"Could not remove temp ping-pong unit: {e_clean_pp}")
            gc.collect()

    def _tb_get_video_dimensions(self, video_path):
        video_width, video_height = 0, 0
        # Prefer ffprobe if available for dimensions
        if self.has_ffprobe:
            try:
                probe_cmd = [self.ffprobe_exe, "-v", "error", "-select_streams", "v:0", 
                             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                w_str, h_str = result.stdout.strip().split('x')
                video_width, video_height = int(w_str), int(h_str)
                if video_width > 0 and video_height > 0: return video_width, video_height
            except Exception as e_probe_dim:
                self.message_manager.add_warning(f"ffprobe failed to get dimensions ({e_probe_dim}), trying imageio.")
        
        # Fallback to imageio
        reader = None
        try:
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            size_imgio = meta.get('size')  
            if size_imgio and isinstance(size_imgio, tuple) and len(size_imgio) == 2:
                video_width, video_height = int(size_imgio[0]), int(size_imgio[1])
        except Exception as e_meta:
            self.message_manager.add_warning(f"Error getting video dimensions for vignette (imageio): {e_meta}. Defaulting aspect to 1/1.")
        finally:
            if reader: reader.close()
        return video_width, video_height # Might be 0,0 if all failed
        
    def _tb_create_vignette_filter(self, strength_percent, width, height):
        min_angle_rad = math.pi / 3.5; max_angle_rad = math.pi / 2    
        normalized_strength = strength_percent / 100.0 
        angle_rad = min_angle_rad + normalized_strength * (max_angle_rad - min_angle_rad)
        vignette_aspect_ratio_val = "1/1" 
        if width > 0 and height > 0: vignette_aspect_ratio_val = f"{width/height:.4f}" 
        return f"vignette=angle={angle_rad:.4f}:mode=forward:eval=init:aspect={vignette_aspect_ratio_val}"

    def tb_apply_filters(self, video_path, brightness, contrast, saturation, temperature,
                      sharpen, blur, denoise, vignette, s_curve_contrast, film_grain_strength,
                      progress=gr.Progress()):
        if video_path is None: self.message_manager.add_warning("No input video for filters."); return None
        if not self.has_ffmpeg: # FFmpeg is essential for this function
            self.message_manager.add_error("FFmpeg is required for applying video filters. This operation cannot proceed.")
            return video_path 

        progress(0, desc="Initializing filter application...")
        resolved_video_path = str(Path(video_path).resolve())
        output_path = self._tb_generate_output_path(resolved_video_path, "filtered", self.toolbox_video_output_dir)
        self.message_manager.add_message(f"🎨 Applying filters to {os.path.basename(resolved_video_path)}...")

        video_width, video_height = 0,0
        if vignette > 0: # Only get dimensions if vignette is used
            video_width, video_height = self._tb_get_video_dimensions(resolved_video_path)
            if video_width > 0 and video_height > 0: self.message_manager.add_message(f"Video dimensions for vignette: {video_width}x{video_height}", "DEBUG")
            
        filters, applied_filter_descriptions = [], []

        # Filter definitions
        if denoise > 0: filters.append(f"hqdn3d={denoise*0.8:.1f}:{denoise*0.6:.1f}:{denoise*0.7:.1f}:{denoise*0.5:.1f}"); applied_filter_descriptions.append(f"Denoise (hqdn3d)")
        if temperature != 0: mid_shift = (temperature/100.0)*0.3; filters.append(f"colorbalance=rm={mid_shift:.2f}:bm={-mid_shift:.2f}"); applied_filter_descriptions.append(f"Color Temp")
        eq_parts = []; desc_eq = []
        if brightness != 0: eq_parts.append(f"brightness={brightness/100.0:.2f}"); desc_eq.append(f"Brightness")
        if contrast != 1: eq_parts.append(f"contrast={contrast:.2f}"); desc_eq.append(f"Contrast (Linear)")
        if saturation != 1: eq_parts.append(f"saturation={saturation:.2f}"); desc_eq.append(f"Saturation")
        if eq_parts: filters.append(f"eq={':'.join(eq_parts)}"); applied_filter_descriptions.append(" & ".join(desc_eq))
        if s_curve_contrast > 0: s = s_curve_contrast/100.0; y1 = 0.25-s*(0.25-0.10); y2 = 0.75+s*(0.90-0.75); filters.append(f"curves=all='0/0 0.25/{y1:.2f} 0.75/{y2:.2f} 1/1'"); applied_filter_descriptions.append(f"S-Curve Contrast")
        if blur > 0: filters.append(f"gblur=sigma={blur*0.4:.1f}"); applied_filter_descriptions.append(f"Blur")
        if sharpen > 0: filters.append(f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={sharpen*0.3:.2f}"); applied_filter_descriptions.append(f"Sharpen")
        if film_grain_strength > 0: filters.append(f"noise=alls={film_grain_strength*0.5:.1f}:allf=t+u"); applied_filter_descriptions.append(f"Film Grain")
        if vignette > 0: filters.append(self._tb_create_vignette_filter(vignette, video_width, video_height)); applied_filter_descriptions.append(f"Vignette")

        # --- CORRECTED LOGIC ---
        if applied_filter_descriptions:
            self.message_manager.add_message("🔧 Applying FFmpeg filters: " + ", ".join(applied_filter_descriptions))
        else:
            self.message_manager.add_message("ℹ️ No filters selected. Passing video through (re-encoding).")
        
        progress(0.2, desc="Preparing filter command...")
        original_video_has_audio = self._tb_has_audio_stream(resolved_video_path)
        
        try:
            ffmpeg_cmd = [
                self.ffmpeg_exe, "-y", "-loglevel", "error", "-i", resolved_video_path
            ]
            # Conditionally add the video filter flag only if there are filters to apply
            if filters:
                ffmpeg_cmd.extend(["-vf", ",".join(filters)])
            
            # Add the rest of the encoding options
            ffmpeg_cmd.extend([
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-map", "0:v:0" 
            ])

            if original_video_has_audio:
                self.message_manager.add_message("Original video has audio. Re-encoding to AAC for filtered video.", "INFO")
                ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", "-map", "0:a:0?"])
            else:
                self.message_manager.add_message("No audio in original or detection issue. Filtered video will be silent.", "INFO")
                ffmpeg_cmd.extend(["-an"])
            
            ffmpeg_cmd.append(output_path)

            self.message_manager.add_message("🔄 Processing with FFmpeg...")
            progress(0.5, desc="Running FFmpeg for filters...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore') 
            
            progress(1.0, desc="Filters applied successfully.")
            self.message_manager.add_success(f"✅ Filter step complete! Output: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e_filters:
            self._tb_log_ffmpeg_error(e_filters, "filter application")
            progress(1.0, desc="Error applying filters."); return None
        except Exception as e:
            self.message_manager.add_error(f"❌ An unexpected error occurred: {e}")
            import traceback; self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error applying filters."); return None
        finally: gc.collect()
            

    def tb_upscale_video(self, video_path, model_key: str, output_scale_factor_ui: float, 
                         tile_size: int, enhance_face: bool, 
                         denoise_strength_ui: float | None, # New parameter
                         progress=gr.Progress()):
        if video_path is None: self.message_manager.add_warning("No input video for upscaling."); return None
        
        final_output_path = None; reader = None
        try:
            if model_key not in self.esrgan_upscaler.supported_models:
                self.message_manager.add_error(f"Upscale model key '{model_key}' not found in supported models.")
                return None
            
            model_native_scale = self.esrgan_upscaler.supported_models[model_key].get('scale', 0)

            tile_size_str_for_log = str(tile_size) if tile_size > 0 else "Auto"
            face_enhance_str_for_log = "+FaceEnhance" if enhance_face else ""
            denoise_str_for_log = ""
            if model_key == "RealESR-general-x4v3" and denoise_strength_ui is not None:
                denoise_str_for_log = f", DNI: {denoise_strength_ui:.2f}"


            self.message_manager.add_message(
                f"Preparing to load ESRGAN model '{model_key}' for {output_scale_factor_ui:.2f}x target upscale "
                f"(Native: {model_native_scale}x, Tile: {tile_size_str_for_log}{face_enhance_str_for_log}{denoise_str_for_log})."
            )
            progress(0.05, desc=f"Loading ESRGAN model '{model_key}' (Tile: {tile_size_str_for_log}{denoise_str_for_log})...")
            
            # Pass denoise_strength_ui to load_model
            upsampler_instance = self.esrgan_upscaler.load_model(
                model_key=model_key, 
                tile_size=tile_size,
                denoise_strength=denoise_strength_ui if model_key == "RealESR-general-x4v3" else None
            )
            if not upsampler_instance:
                self.message_manager.add_error(f"Could not load ESRGAN model '{model_key}'. Aborting."); return None 

            if enhance_face: # Face enhancer loading logic
                if not self.esrgan_upscaler._load_face_enhancer(bg_upsampler=upsampler_instance):
                    self.message_manager.add_warning("Failed to load GFPGAN for face enhancement. Proceeding without it.")
                    enhance_face = False 
                    face_enhance_str_for_log = "" # Update log string if face enhance fails

            self.message_manager.add_message(
                f"ESRGAN model '{model_key}' (Native: {model_native_scale}x, Tile: {tile_size_str_for_log}{denoise_str_for_log}) "
                f"{'and GFPGAN ' if enhance_face else ''}loaded for target {output_scale_factor_ui:.2f}x output."
            )
            progress(0.1, desc=f"Initializing {output_scale_factor_ui:.2f}x upscaling{face_enhance_str_for_log}{denoise_str_for_log} process...")
            
            resolved_video_path = str(Path(video_path).resolve())
            upscaled_frames = []
            
            progress(0.12, desc="Reading video info...")
            reader = imageio.get_reader(resolved_video_path)
            meta_data = reader.get_meta_data(); original_fps = meta_data.get('fps', 30.0)
            
            n_frames = meta_data.get('nframes')
            if n_frames is None or n_frames == float('inf'):
                try: n_frames = reader.count_frames()
                except: n_frames = None 
            if n_frames == float('inf'): n_frames = None

            n_frames_display = str(int(n_frames)) if n_frames is not None else "Unknown"
            self.message_manager.add_message(f"Original FPS: {original_fps:.2f}. Total frames: {n_frames_display}.")

            progress_desc = (
                f"Upscaling Frames to {output_scale_factor_ui:.2f}x (Model: {model_key}{face_enhance_str_for_log}{denoise_str_for_log}, "
                f"Native: {model_native_scale}x, Tile: {tile_size_str_for_log})"
            )
            frame_iterator = enumerate(reader)
            if n_frames is not None: frame_iterator = progress.tqdm(enumerate(reader), total=int(n_frames), desc=progress_desc)
            else: self.message_manager.add_message(f"Total frames unknown, progress per batch ({progress_desc}).")

            for i, frame_np in frame_iterator:
                if n_frames is None and i % 10 == 0: 
                    current_progress_val = 0.15 + ( (i/(i+500.0)) * 0.65 )
                    progress(current_progress_val , desc=f"Upscaling frame {i+1} to {output_scale_factor_ui:.2f}x (Tile: {tile_size_str_for_log})...")
                
                upscaled_frame_np = self.esrgan_upscaler.upscale_frame( # DNI is handled by loaded model
                    frame_np_array=frame_np, 
                    model_key=model_key,
                    target_outscale_factor=float(output_scale_factor_ui), 
                    enhance_face=enhance_face
                )
                if upscaled_frame_np is not None: upscaled_frames.append(upscaled_frame_np)
                else: # Error handling for frame upscale
                    self.message_manager.add_error(f"Failed to upscale frame {i+1}. Skipping.")
                    if "out of memory" in self.message_manager.get_recent_errors_as_str(count=1).lower():
                        self.message_manager.add_error("CUDA OOM likely. Aborting video upscale."); return None 
                if (i+1) % 20 == 0: devicetorch.empty_cache(torch); gc.collect()
            
            if reader: reader.close(); reader = None 
            if not upscaled_frames: self.message_manager.add_error("No frames upscaled."); return None 
            
            self.message_manager.add_message(f"Successfully upscaled {len(upscaled_frames)} frames to {output_scale_factor_ui:.2f}x.")
            progress(0.80, desc="Saving upscaled video stream...")

            temp_video_suffix_base = (
                f"upscaled_{model_key}"
                f"{face_enhance_str_for_log.replace('+','_')}"
            )
            if model_key == "RealESR-general-x4v3" and denoise_strength_ui is not None:
                 temp_video_suffix_base += f"_dni{denoise_strength_ui:.2f}"
            temp_video_suffix = temp_video_suffix_base.replace(".","p") + "_temp_video"
            
            video_stream_output_path = self._tb_generate_output_path(resolved_video_path, temp_video_suffix, self.toolbox_video_output_dir)
            final_muxed_output_path = video_stream_output_path.replace("_temp_video", "")

            imageio.mimwrite(video_stream_output_path, upscaled_frames, fps=original_fps, quality=VIDEO_QUALITY, macro_block_size=None)
            del upscaled_frames; devicetorch.empty_cache(torch); gc.collect()

            final_output_path = final_muxed_output_path
            can_process_audio = self.has_ffmpeg
            original_video_has_audio = self._tb_has_audio_stream(resolved_video_path) if can_process_audio else False

            if can_process_audio and original_video_has_audio:
                progress(0.90, desc="Muxing audio...")
                self.message_manager.add_message("Original video has audio. Muxing audio with FFmpeg...")
                ffmpeg_mux_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", video_stream_output_path, "-i", resolved_video_path,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_muxed_output_path
                ]
                try:
                    subprocess.run(ffmpeg_mux_cmd, check=True, capture_output=True, text=True)
                    self.message_manager.add_success(f"Upscaled video saved with audio: {final_muxed_output_path}")
                except subprocess.CalledProcessError as e_mux:
                    self._tb_log_ffmpeg_error(e_mux, "audio muxing for upscaled video")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path)
                    os.rename(video_stream_output_path, final_muxed_output_path)
                except FileNotFoundError: 
                    self.message_manager.add_error(f"FFmpeg not found during muxing. Unexpected.")
                    if os.path.exists(final_muxed_output_path): os.remove(final_muxed_output_path)
                    os.rename(video_stream_output_path, final_muxed_output_path)
            else:
                if original_video_has_audio and not can_process_audio:
                     self.message_manager.add_warning("Original video has audio, but FFmpeg is not available to process it. Upscaled output will be silent.")
                elif not original_video_has_audio :
                     self.message_manager.add_message("No audio in original or detection issue. Saving upscaled video-only.")
                if os.path.exists(final_muxed_output_path) and final_muxed_output_path != video_stream_output_path:
                    os.remove(final_muxed_output_path)
                os.rename(video_stream_output_path, final_muxed_output_path)

            if os.path.exists(video_stream_output_path) and video_stream_output_path != final_muxed_output_path:
                try: os.remove(video_stream_output_path)
                except Exception as e_clean: self.message_manager.add_warning(f"Could not remove temp upscaled video: {e_clean}")
            
            progress(1.0, desc="Upscaling complete.")
            self.message_manager.add_success(f"Video upscaling to {output_scale_factor_ui:.2f}x complete: {final_output_path}")
            return final_output_path

        except Exception as e:
            self.message_manager.add_error(f"Error during video upscaling: {e}")
            import traceback; self.message_manager.add_error(traceback.format_exc())
            progress(1.0, desc="Error during upscaling."); return None 
        finally:
            if reader: 
                try: 
                    if hasattr(reader, 'closed') and not reader.closed: reader.close()
                except: pass
            if model_key and self.esrgan_upscaler: # Ensure model is unloaded
                 self.esrgan_upscaler.unload_model(model_key) 
            if enhance_face and self.esrgan_upscaler and self.esrgan_upscaler.face_enhancer:
                self.esrgan_upscaler._unload_face_enhancer()
            devicetorch.empty_cache(torch); gc.collect()

    def tb_open_output_folder(self):
        folder_path = os.path.abspath(self.postprocessed_output_root_dir)
        try:
            os.makedirs(folder_path, exist_ok=True) 
            if sys.platform == 'win32': subprocess.run(['explorer', folder_path])
            elif sys.platform == 'darwin': subprocess.run(['open', folder_path])
            else: subprocess.run(['xdg-open', folder_path])
            self.message_manager.add_success(f"Opened postprocessed output folder: {folder_path}")
        except Exception as e:
            self.message_manager.add_error(f"Error opening folder {folder_path}: {e}")

    def tb_clear_temporary_files(self):
        temp_dir_path_str = str(self._base_temp_output_dir)
        self.message_manager.add_message(f"Attempting to clear temporary files in: {temp_dir_path_str}", "INFO")
        
        cleared_successfully = False
        if os.path.exists(temp_dir_path_str):
            try:
                # Count items for logging
                items = os.listdir(temp_dir_path_str)
                file_count = sum(1 for item in items if os.path.isfile(os.path.join(temp_dir_path_str, item)))
                dir_count = sum(1 for item in items if os.path.isdir(os.path.join(temp_dir_path_str, item)))
                
                shutil.rmtree(temp_dir_path_str)
                self.message_manager.add_success(
                    f"Successfully removed temporary directory and its contents ({file_count} files, {dir_count} subdirectories)."
                )
                cleared_successfully = True
            except Exception as e:
                self.message_manager.add_error(f"Error deleting temporary directory '{temp_dir_path_str}': {e}")
                self.message_manager.add_error(traceback.format_exc())
        else:
            self.message_manager.add_message("Temporary directory does not exist. Nothing to clear.", "INFO")
            cleared_successfully = True

        try:
            os.makedirs(temp_dir_path_str, exist_ok=True) # Always recreate
            if cleared_successfully: self.message_manager.add_message(f"Recreated temporary directory: {temp_dir_path_str}", "INFO")
        except Exception as e_recreate:
            self.message_manager.add_error(f"CRITICAL: Failed to recreate temporary directory '{temp_dir_path_str}': {e_recreate}. Processing may fail.")
            self.message_manager.add_error(traceback.format_exc())
            cleared_successfully = False
        return cleared_successfully