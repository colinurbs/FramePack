import threading
import time
import uuid
import json
import os
import zipfile
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import queue as queue_module  # Renamed to avoid conflicts
import io
import base64
from PIL import Image
import numpy as np

from diffusers_helper.thread_utils import AsyncStream
from modules.pipelines.metadata_utils import create_metadata
from modules.settings import Settings
from diffusers_helper.gradio.progress_bar import make_progress_bar_html


# Simple LIFO queue implementation to avoid dependency on queue.LifoQueue
class SimpleLifoQueue:
    def __init__(self):
        self._queue = []
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
    
    def put(self, item):
        with self._mutex:
            self._queue.append(item)
            self._not_empty.notify()
    
    def get(self):
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            return self._queue.pop()
    
    def task_done(self):
        pass  # For compatibility with queue.Queue


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    SINGLE = "single"
    GRID = "grid"


@dataclass
class Job:
    id: str
    params: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    job_type: JobType = JobType.SINGLE
    child_job_ids: List[str] = field(default_factory=list)
    parent_job_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[str] = None
    progress_data: Optional[Dict] = None
    queue_position: Optional[int] = None
    stream: Optional[Any] = None
    input_image: Optional[np.ndarray] = None
    latent_type: Optional[str] = None
    thumbnail: Optional[str] = None
    generation_type: Optional[str] = None # Added generation_type
    input_image_saved: bool = False  # Flag to track if input image has been saved
    end_frame_image_saved: bool = False  # Flag to track if end frame image has been saved

    def __post_init__(self):
        # Store generation type
        self.generation_type = self.params.get('model_type', 'Original') # Initialize generation_type

        # Store input image or latent type
        if 'input_image' in self.params and self.params['input_image'] is not None:
            self.input_image = self.params['input_image']
            # Create thumbnail
            if isinstance(self.input_image, np.ndarray):
                # Handle numpy array (image)
                img = Image.fromarray(self.input_image)
                img.thumbnail((100, 100))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            elif isinstance(self.input_image, str):
                # Handle string (video path)
                try:
                    print(f"Attempting to extract thumbnail from video: {self.input_image}")
                    # Try to extract frames from the video using imageio
                    import imageio
                    
                    # Check if the file exists
                    if not os.path.exists(self.input_image):
                        print(f"Video file not found: {self.input_image}")
                        raise FileNotFoundError(f"Video file not found: {self.input_image}")
                    
                    # Create outputs directory if it doesn't exist
                    os.makedirs("outputs", exist_ok=True)
                    
                    # Try to open the video file
                    try:
                        reader = imageio.get_reader(self.input_image)
                        print(f"Successfully opened video file with imageio")
                    except Exception as e:
                        print(f"Failed to open video with imageio: {e}")
                        raise
                    
                    # Get the total number of frames
                    num_frames = None
                    try:
                        # Try to get the number of frames from metadata
                        meta_data = reader.get_meta_data()
                        print(f"Video metadata: {meta_data}")
                        num_frames = meta_data.get('nframes')
                        if num_frames is None or num_frames == float('inf'):
                            print("Number of frames not available in metadata")
                            # If not available, try to count frames
                            if hasattr(reader, 'count_frames'):
                                print("Trying to count frames...")
                                num_frames = reader.count_frames()
                                print(f"Counted {num_frames} frames")
                    except Exception as e:
                        print(f"Error getting frame count: {e}")
                        num_frames = None
                    
                    # If we couldn't determine the number of frames, read the last frame by iterating
                    if num_frames is None or num_frames == float('inf'):
                        print("Reading frames by iteration to find the last one")
                        # Read frames until we reach the end
                        frame_count = 0
                        first_frame = None
                        last_frame = None
                        try:
                            for frame in reader:
                                if frame_count == 0:
                                    first_frame = frame
                                last_frame = frame
                                frame_count += 1
                                # Print progress every 100 frames
                                if frame_count % 100 == 0:
                                    print(f"Read {frame_count} frames...")
                            print(f"Finished reading {frame_count} frames")
                            
                            # Save the first frame if available
                            if first_frame is not None:
                                print(f"Found first frame with shape: {first_frame.shape}")
                                # DEBUG IMAGE SAVING REMOVED
                        except Exception as e:
                            print(f"Error reading frames: {e}")
                        
                        if last_frame is not None:
                            print(f"Found last frame with shape: {last_frame.shape}")
                            
                            # DEBUG IMAGE SAVING REMOVED
                            # Use the last frame for the thumbnail
                            img = Image.fromarray(last_frame)
                            img.thumbnail((100, 100))
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                            print("Successfully created thumbnail from last frame")
                        else:
                            print("No frames were read, using red thumbnail")
                            # Fallback to red thumbnail if no frames were read - more visible for debugging
                            img = Image.new('RGB', (100, 100), (255, 0, 0))  # Red for video
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                    else:
                        # If we know the number of frames, try to get multiple frames for debugging
                        try:
                            # Try to get the first frame
                            first_frame = None
                            try:
                                first_frame = reader.get_data(0)
                                print(f"Got first frame with shape: {first_frame.shape}")
                                
                                # DEBUG IMAGE SAVING REMOVED
                            except Exception as e:
                                print(f"Error getting first frame: {e}")
                            
                            # Try to get a middle frame
                            middle_frame = None
                            try:
                                middle_frame_idx = int(num_frames / 2)
                                middle_frame = reader.get_data(middle_frame_idx)
                                print(f"Got middle frame (frame {middle_frame_idx}) with shape: {middle_frame.shape}")
                                
                                # DEBUG IMAGE SAVING REMOVED
                            except Exception as e:
                                print(f"Error getting middle frame: {e}")
                            
                            # Try to get the last frame
                            last_frame = None
                            try:
                                last_frame_idx = int(num_frames) - 1
                                last_frame = reader.get_data(last_frame_idx)
                                print(f"Got last frame (frame {last_frame_idx}) with shape: {last_frame.shape}")
                                
                                # DEBUG IMAGE SAVING REMOVED
                            except Exception as e:
                                print(f"Error getting last frame: {e}")
                            
                            # If we couldn't get the last frame directly, try to get it by iterating
                            if last_frame is None:
                                print("Trying to get last frame by iterating through all frames")
                                try:
                                    for frame in reader:
                                        last_frame = frame
                                    
                                    if last_frame is not None:
                                        print(f"Got last frame by iteration with shape: {last_frame.shape}")
                                        
                                        # DEBUG IMAGE SAVING REMOVED
                                except Exception as e:
                                    print(f"Error getting last frame by iteration: {e}")
                            
                            # Use the last frame for the thumbnail if available, otherwise use the middle or first frame
                            frame_for_thumbnail = last_frame if last_frame is not None else (middle_frame if middle_frame is not None else first_frame)
                            
                            if frame_for_thumbnail is not None:
                                # Convert to PIL Image and create a thumbnail
                                img = Image.fromarray(frame_for_thumbnail)
                                img.thumbnail((100, 100))
                                buffered = io.BytesIO()
                                img.save(buffered, format="PNG")
                                self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                                print("Successfully created thumbnail from frame")
                            else:
                                print("No frames were extracted, using blue thumbnail")
                                # Fallback to blue thumbnail if no frames were extracted
                                img = Image.new('RGB', (100, 100), (0, 0, 255))  # Blue for video
                                buffered = io.BytesIO()
                                img.save(buffered, format="PNG")
                                self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                        except Exception as e:
                            # Fallback to blue thumbnail on error
                            img = Image.new('RGB', (100, 100), (0, 0, 255))  # Blue for video
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                    
                    # Close the reader
                    try:
                        reader.close()
                        print("Successfully closed video reader")
                    except Exception as e:
                        print(f"Error closing reader: {e}")
                    
                except Exception as e:
                    print(f"Error extracting thumbnail from video: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to bright green thumbnail on error to make it more visible
                    img = Image.new('RGB', (100, 100), (0, 255, 0))  # Bright green for error
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                    print("Created bright green fallback thumbnail")
            else:
                # Handle other types
                self.thumbnail = None
        elif 'latent_type' in self.params:
            self.latent_type = self.params['latent_type']
            # Create a colored square based on latent type
            color_map = {
                "Black": (0, 0, 0),
                "White": (255, 255, 255),
                "Noise": (128, 128, 128),
                "Green Screen": (0, 177, 64)
            }
            color = color_map.get(self.latent_type, (0, 0, 0))
            img = Image.new('RGB', (100, 100), color)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            self.thumbnail = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


class VideoJobQueue:
    def __init__(self):
        self.queue = queue_module.Queue()  # Using standard Queue instead of LifoQueue
        self.jobs = {}
        self.current_job = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.worker_function = None  # Will be set from outside
        self.is_processing = False  # Flag to track if we're currently processing a job
    
    def set_worker_function(self, worker_function):
        """Set the worker function to use for processing jobs"""
        self.worker_function = worker_function
    
    def serialize_job(self, job):
        """Serialize a job to a JSON-compatible format"""
        try:
            # Create a simplified representation of the job
            serialized = {
                "id": job.id,
                "status": job.status.value,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
                "result": job.result,
                "queue_position": job.queue_position,
                "generation_type": job.generation_type,
            }
            
            # Add simplified params (excluding complex objects)
            serialized_params = {}
            for k, v in job.params.items():
                if k not in ["input_image", "end_frame_image", "stream"]:
                    # Try to include only JSON-serializable values
                    try:
                        # Test if value is JSON serializable
                        json.dumps({k: v})
                        serialized_params[k] = v
                    except (TypeError, OverflowError):
                        # Skip non-serializable values
                        pass
            
            # Handle LoRA information specifically
            # Only include selected LoRAs for the generation
            if "selected_loras" in job.params and job.params["selected_loras"]:
                selected_loras = job.params["selected_loras"]
                # Ensure it's a list
                if not isinstance(selected_loras, list):
                    selected_loras = [selected_loras] if selected_loras is not None else []
                
                # Get LoRA values if available
                lora_values = job.params.get("lora_values", [])
                if not isinstance(lora_values, list):
                    lora_values = [lora_values] if lora_values is not None else []
                
                # Get loaded LoRA names
                lora_loaded_names = job.params.get("lora_loaded_names", [])
                if not isinstance(lora_loaded_names, list):
                    lora_loaded_names = [lora_loaded_names] if lora_loaded_names is not None else []
                
                # Create LoRA data dictionary
                lora_data = {}
                for lora_name in selected_loras:
                    try:
                        # Find the index of the LoRA in loaded names
                        idx = lora_loaded_names.index(lora_name) if lora_loaded_names else -1
                        # Get the weight value
                        weight = lora_values[idx] if lora_values and idx >= 0 and idx < len(lora_values) else 1.0
                        # Handle weight as list
                        if isinstance(weight, list):
                            weight_value = weight[0] if weight and len(weight) > 0 else 1.0
                        else:
                            weight_value = weight
                        # Store as float
                        lora_data[lora_name] = float(weight_value)
                    except (ValueError, IndexError):
                        # Default weight if not found
                        lora_data[lora_name] = 1.0
                    except Exception as e:
                        print(f"Error processing LoRA {lora_name}: {e}")
                        lora_data[lora_name] = 1.0
                
                # Add to serialized params
                serialized_params["loras"] = lora_data
            
            serialized["params"] = serialized_params
            
            # Don't include the thumbnail as it can be very large and cause issues
            # if job.thumbnail:
            #     serialized["thumbnail"] = job.thumbnail
                
            return serialized
        except Exception as e:
            print(f"Error serializing job {job.id}: {e}")
            # Return minimal information that should always be serializable
            return {
                "id": job.id,
                "status": job.status.value,
                "error": f"Error serializing: {str(e)}"
            }
    
    def save_queue_to_json(self):
        """Save the current queue to queue.json using the central metadata utility"""
        try:
            # Make a copy of job IDs to avoid holding the lock while serializing
            with self.lock:
                job_ids = list(self.jobs.keys())
            
            # Create a settings instance
            settings = Settings()
            
            # Create a directory to store queue images if it doesn't exist
            queue_images_dir = "queue_images"
            os.makedirs(queue_images_dir, exist_ok=True)
            
            # First, ensure all images are saved
            for job_id in job_ids:
                job = self.get_job(job_id)
                if job:
                    # Save input image to disk if it exists and hasn't been saved yet
                    if 'input_image' in job.params and isinstance(job.params['input_image'], np.ndarray) and not job.input_image_saved:
                        input_image_path = os.path.join(queue_images_dir, f"{job_id}_input.png")
                        try:
                            Image.fromarray(job.params['input_image']).save(input_image_path)
                            print(f"Saved input image for job {job_id} to {input_image_path}")
                            # Mark the image as saved
                            job.input_image_saved = True
                        except Exception as e:
                            print(f"Error saving input image for job {job_id}: {e}")
                    
                    # Save end frame image to disk if it exists and hasn't been saved yet
                    if 'end_frame_image' in job.params and isinstance(job.params['end_frame_image'], np.ndarray) and not job.end_frame_image_saved:
                        end_frame_image_path = os.path.join(queue_images_dir, f"{job_id}_end_frame.png")
                        try:
                            Image.fromarray(job.params['end_frame_image']).save(end_frame_image_path)
                            print(f"Saved end frame image for job {job_id} to {end_frame_image_path}")
                            # Mark the end frame image as saved
                            job.end_frame_image_saved = True
                        except Exception as e:
                            print(f"Error saving end frame image for job {job_id}: {e}")
            
            # Now serialize jobs with the updated image saved flags
            serialized_jobs = {}
            for job_id in job_ids:
                job = self.get_job(job_id)
                if job:
                    # Try to use metadata_utils.create_metadata if possible
                    try:
                        # Create metadata using the central utility
                        metadata = create_metadata(job.params, job.id, settings.settings)
                        
                        # Add job status and other fields not included in metadata
                        metadata.update({
                            "id": job.id,
                            "status": job.status.value,
                            "created_at": job.created_at,
                            "started_at": job.started_at,
                            "completed_at": job.completed_at,
                            "error": job.error,
                            "result": job.result,
                            "queue_position": job.queue_position,
                        })
                        
                        # Add image paths to metadata if they've been saved
                        if job.input_image_saved:
                            input_image_path = os.path.join(queue_images_dir, f"{job_id}_input.png")
                            if os.path.exists(input_image_path):
                                metadata["saved_input_image_path"] = input_image_path
                        
                        if job.end_frame_image_saved:
                            end_frame_image_path = os.path.join(queue_images_dir, f"{job_id}_end_frame.png")
                            if os.path.exists(end_frame_image_path):
                                metadata["saved_end_frame_image_path"] = end_frame_image_path
                        
                        serialized_jobs[job_id] = metadata
                    except Exception as e:
                        print(f"Error using metadata_utils for job {job_id}: {e}")
                        # Fall back to the old serialization method
                        serialized_jobs[job_id] = self.serialize_job(job)
            
            # Save to file
            with open("queue.json", "w") as f:
                json.dump(serialized_jobs, f, indent=2)
            
            # Clean up images for jobs that no longer exist
            self.cleanup_orphaned_images(job_ids)
            self.cleanup_orphaned_videos(job_ids)
                
            print(f"Saved {len(serialized_jobs)} jobs to queue.json")
        except Exception as e:
            print(f"Error saving queue to JSON: {e}")
    
    def cleanup_orphaned_videos(self, current_job_ids_uuids): # Renamed arg for clarity
        """
        Remove video files from input_files_dir for jobs that no longer exist
        or whose input_image_path does not point to them.
        
        Args:
            current_job_ids_uuids: List of job UUIDs currently in self.jobs
        """
        try:
            # Get the input_files_dir from settings to be robust
            settings = Settings()
            input_files_dir = settings.get("input_files_dir", "input_files")
            if not os.path.exists(input_files_dir):
                return

            referenced_video_paths = set()
            with self.lock: # Access self.jobs safely
                for job_id_uuid in current_job_ids_uuids: # Iterate using the provided UUIDs
                    job = self.jobs.get(job_id_uuid)
                    if job and job.params:
                        # For Video models, job.params['input_image'] is the path after studio.py copy.
                        # This is the path that decord will try to open.
                        video_path_from_job_params = job.params.get('input_image')
                        if isinstance(video_path_from_job_params, str) and os.path.isabs(video_path_from_job_params) and input_files_dir in os.path.normpath(video_path_from_job_params):
                            referenced_video_paths.add(os.path.normpath(video_path_from_job_params))
                        
                        # Also consider 'input_image_path' which might be used by other logic or older jobs.
                        input_image_path_val = job.params.get('input_image_path')
                        if isinstance(input_image_path_val, str) and os.path.isabs(input_image_path_val) and input_files_dir in os.path.normpath(input_image_path_val):
                             referenced_video_paths.add(os.path.normpath(input_image_path_val))

            removed_count = 0
            for filename in os.listdir(input_files_dir):
                if filename.endswith(".mp4"): # Only process MP4 files
                    file_path_to_check = os.path.normpath(os.path.join(input_files_dir, filename))
                    
                    if file_path_to_check not in referenced_video_paths:
                        try:
                            os.remove(file_path_to_check)
                            removed_count += 1
                            print(f"Removed orphaned video: {filename} (path: {file_path_to_check})")
                        except Exception as e:
                            print(f"Error removing orphaned video {filename}: {e}")
            if removed_count > 0:
                print(f"Cleaned up {removed_count} orphaned videos from {input_files_dir}")
        except Exception as e:
            print(f"Error cleaning up orphaned videos: {e}")
            import traceback
            traceback.print_exc()

    def cleanup_orphaned_images(self, current_job_ids):
        """
        Remove image files for jobs that no longer exist in the queue.
        
        Args:
            current_job_ids: List of job IDs currently in the queue
        """
        try:
            queue_images_dir = "queue_images"
            if not os.path.exists(queue_images_dir):
                return
            
            # Convert to set for faster lookups
            current_job_ids = set(current_job_ids)
            
            # Check all files in the queue_images directory
            removed_count = 0
            for filename in os.listdir(queue_images_dir):
                # Only process PNG files with our naming pattern
                if filename.endswith(".png") and ("_input.png" in filename or "_end_frame.png" in filename):
                    # Extract job ID from filename
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        job_id = parts[0]
                        
                        # If job ID is not in current jobs, remove the file
                        if job_id not in current_job_ids:
                            file_path = os.path.join(queue_images_dir, filename)
                            try:
                                os.remove(file_path)
                                removed_count += 1
                                print(f"Removed orphaned image: {filename}")
                            except Exception as e:
                                print(f"Error removing orphaned image {filename}: {e}")
            
            if removed_count > 0:
                print(f"Cleaned up {removed_count} orphaned images")
        except Exception as e:
            print(f"Error cleaning up orphaned images: {e}")

    
    def synchronize_queue_images(self):
        """
        Synchronize the queue_images directory with the current jobs in the queue.
        This ensures all necessary images are saved and only images for removed jobs are deleted.
        """
        try:
            queue_images_dir = "queue_images"
            os.makedirs(queue_images_dir, exist_ok=True)
            
            # Get all current job IDs
            with self.lock:
                current_job_ids = set(self.jobs.keys())
            
            # Get all image files in the queue_images directory
            existing_image_files = set()
            if os.path.exists(queue_images_dir):
                for filename in os.listdir(queue_images_dir):
                    if filename.endswith(".png") and ("_input.png" in filename or "_end_frame.png" in filename):
                        existing_image_files.add(filename)
            
            # Extract job IDs from filenames
            file_job_ids = set()
            for filename in existing_image_files:
                # Extract job ID from filename (format: "{job_id}_input.png" or "{job_id}_end_frame.png")
                parts = filename.split("_")
                if len(parts) >= 2:
                    job_id = parts[0]
                    file_job_ids.add(job_id)
            
            # Find job IDs in files that are no longer in the queue
            removed_job_ids = file_job_ids - current_job_ids
            
            # Delete images for jobs that have been removed from the queue
            removed_count = 0
            for job_id in removed_job_ids:
                input_image_path = os.path.join(queue_images_dir, f"{job_id}_input.png")
                end_frame_image_path = os.path.join(queue_images_dir, f"{job_id}_end_frame.png")
                
                if os.path.exists(input_image_path):
                    try:
                        os.remove(input_image_path)
                        removed_count += 1
                        print(f"Removed image for deleted job: {input_image_path}")
                    except Exception as e:
                        print(f"Error removing image {input_image_path}: {e}")
                
                if os.path.exists(end_frame_image_path):
                    try:
                        os.remove(end_frame_image_path)
                        removed_count += 1
                        print(f"Removed image for deleted job: {end_frame_image_path}")
                    except Exception as e:
                        print(f"Error removing image {end_frame_image_path}: {e}")
            
            # Now ensure all current jobs have their images saved
            saved_count = 0
            with self.lock:
                for job_id, job in self.jobs.items():
                    # Only save images for running or completed jobs
                    if job.status in [JobStatus.RUNNING, JobStatus.COMPLETED]:
                        # Save input image if it exists and hasn't been saved yet
                        if 'input_image' in job.params and isinstance(job.params['input_image'], np.ndarray) and not job.input_image_saved:
                            input_image_path = os.path.join(queue_images_dir, f"{job_id}_input.png")
                            try:
                                Image.fromarray(job.params['input_image']).save(input_image_path)
                                job.input_image_saved = True
                                saved_count += 1
                                print(f"Saved input image for job {job_id}")
                            except Exception as e:
                                print(f"Error saving input image for job {job_id}: {e}")
                        
                        # Save end frame image if it exists and hasn't been saved yet
                        if 'end_frame_image' in job.params and isinstance(job.params['end_frame_image'], np.ndarray) and not job.end_frame_image_saved:
                            end_frame_image_path = os.path.join(queue_images_dir, f"{job_id}_end_frame.png")
                            try:
                                Image.fromarray(job.params['end_frame_image']).save(end_frame_image_path)
                                job.end_frame_image_saved = True
                                saved_count += 1
                                print(f"Saved end frame image for job {job_id}")
                            except Exception as e:
                                print(f"Error saving end frame image for job {job_id}: {e}")
            
            # Save the queue to ensure the image paths are properly referenced
            self.save_queue_to_json()
            
            if removed_count > 0 or saved_count > 0:
                print(f"Queue image synchronization: removed {removed_count} images, saved {saved_count} images")
            
        except Exception as e:
            print(f"Error synchronizing queue images: {e}")

    
    def add_job(self, params, job_type=JobType.SINGLE, child_job_params_list=None, parent_job_id=None):
        """Add a job to the queue and return its ID"""
        job_id = str(uuid.uuid4())
        
        # For grid jobs, create child jobs first
        child_job_ids = []
        if job_type == JobType.GRID and child_job_params_list:
            with self.lock:
                for child_params in child_job_params_list:
                    child_job_id = str(uuid.uuid4())
                    child_job_ids.append(child_job_id)
                    child_job = Job(
                        id=child_job_id,
                        params=child_params,
                        status=JobStatus.PENDING,
                        job_type=JobType.SINGLE, # Children are single jobs
                        parent_job_id=job_id,
                        created_at=time.time(),
                        progress_data={},
                        stream=AsyncStream(),
                        input_image_saved=False,
                        end_frame_image_saved=False
                    )
                    self.jobs[child_job_id] = child_job
                    print(f"  - Created child job {child_job_id} for grid job {job_id}")

        job = Job(
            id=job_id,
            params=params,
            status=JobStatus.PENDING,
            job_type=job_type,
            child_job_ids=child_job_ids,
            parent_job_id=parent_job_id,
            created_at=time.time(),
            progress_data={},
            stream=AsyncStream(),
            input_image_saved=False,
            end_frame_image_saved=False
        )

        with self.lock:
            print(f"Adding job {job_id} (type: {job_type.value}) to queue.")
            self.jobs[job_id] = job
            self.queue.put(job_id) # Only the parent (or single) job is added to the queue initially
        
        # Save the queue to JSON after adding a new job (outside the lock)
        try:
            self.save_queue_to_json()
        except Exception as e:
            print(f"Error saving queue to JSON after adding job: {e}")
        
        return job_id
    
    def get_job(self, job_id):
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self):
        """Get all jobs"""
        with self.lock:
            return list(self.jobs.values())
    
    def cancel_job(self, job_id):
        """Cancel a pending job"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
                
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                result = True
            elif job.status == JobStatus.RUNNING:
                # Send cancel signal to the job's stream
                if hasattr(job, 'stream') and job.stream:
                    job.stream.input_queue.push('end')
                    
                # Mark job as cancelled (this will be confirmed when the worker processes the end signal)
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()  # Mark completion time
                
                # Let the worker loop handle the transition to the next job
                # This ensures the current job is fully processed before switching
                # DEBUG PRINT REMOVED
                result = True
            else:
                result = False
        
        # Save the queue to JSON after cancelling a job (outside the lock)
        if result:
            try:
                self.save_queue_to_json()
            except Exception as e:
                print(f"Error saving queue to JSON after cancelling job: {e}")
        
        return result
    
    def clear_queue(self):
        """Cancel all pending jobs in the queue"""
        cancelled_count = 0
        try:
            # First, make a copy of all pending job IDs to avoid modifying the dictionary during iteration
            with self.lock:
                # Get all pending job IDs
                pending_job_ids = [job_id for job_id, job in self.jobs.items() 
                                if job.status == JobStatus.PENDING]
            
            # Cancel each pending job individually
            for job_id in pending_job_ids:
                try:
                    with self.lock:
                        job = self.jobs.get(job_id)
                        if job and job.status == JobStatus.PENDING:
                            job.status = JobStatus.CANCELLED
                            job.completed_at = time.time()
                            cancelled_count += 1
                except Exception as e:
                    print(f"Error cancelling job {job_id}: {e}")
            
            # Now clear the queue
            with self.lock:
                # Clear the queue (this doesn't affect running jobs)
                queue_items_cleared = 0
                try:
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                            self.queue.task_done()
                            queue_items_cleared += 1
                        except queue_module.Empty:
                            break
                except Exception as e:
                    print(f"Error clearing queue: {e}")
            
            # Save the updated queue state
            try:
                self.save_queue_to_json()
            except Exception as e:
                print(f"Error saving queue state: {e}")
            
            # Synchronize queue images after clearing the queue
            if cancelled_count > 0:
                self.synchronize_queue_images()
            
            print(f"Cleared {cancelled_count} jobs from the queue")
            return cancelled_count
        except Exception as e:
            import traceback
            print(f"Error in clear_queue: {e}")
            traceback.print_exc()
            return 0
            
    def clear_completed_jobs(self):
        """Remove cancelled or completed jobs from the queue"""
        removed_count = 0
        try:
            # First, make a copy of all completed/cancelled job IDs to avoid modifying the dictionary during iteration
            with self.lock:
                # Get all completed or cancelled job IDs
                completed_job_ids = [job_id for job_id, job in self.jobs.items() 
                                  if job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED]]
            
            # Remove each completed/cancelled job individually
            for job_id in completed_job_ids:
                try:
                    with self.lock:
                        if job_id in self.jobs:
                            del self.jobs[job_id]
                            removed_count += 1
                except Exception as e:
                    print(f"Error removing job {job_id}: {e}")
            
            # Save the updated queue state
            try:
                self.save_queue_to_json()
            except Exception as e:
                print(f"Error saving queue state: {e}")
            
            # Synchronize queue images after removing completed jobs
            if removed_count > 0:
                self.synchronize_queue_images()
            
            print(f"Removed {removed_count} completed/cancelled jobs from the queue")
            return removed_count
        except Exception as e:
            import traceback
            print(f"Error in clear_completed_jobs: {e}")
            traceback.print_exc()
            return 0
    
    def get_queue_position(self, job_id):
        """Get position in queue (0 = currently running)"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
                
            if job.status == JobStatus.RUNNING:
                return 0
                
            if job.status != JobStatus.PENDING:
                return None
                
            # Count pending jobs ahead in queue
            position = 1  # Start at 1 because 0 means running
            for j in self.jobs.values():
                if (j.status == JobStatus.PENDING and 
                    j.created_at < job.created_at):
                    position += 1
            return position
    
    def update_job_progress(self, job_id, progress_data):
        """Update job progress data"""
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.progress_data = progress_data
    
    def export_queue_to_zip(self, output_path=None):
        """Export the current queue to a zip file containing queue.json and queue_images directory
        
        Args:
            output_path: Path to save the zip file. If None, uses 'queue_export.zip' in the configured output directory.
            
        Returns:
            str: Path to the created zip file
        """
        try:
            # Get the output directory from settings
            settings = Settings()
            output_dir = settings.get("output_dir", "outputs")
            os.makedirs(output_dir, exist_ok=True)

            # Use default path if none provided
            if output_path is None:
                output_path = os.path.join(output_dir, "queue_export.zip")
            
            # Make sure queue.json is up to date
            self.save_queue_to_json()
            
            # Create a zip file
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add queue.json to the zip file
                if os.path.exists("queue.json"):
                    zipf.write("queue.json")
                    print(f"Added queue.json to {output_path}")
                else:
                    print("Warning: queue.json not found, creating an empty one")
                    with open("queue.json", "w") as f:
                        json.dump({}, f)
                    zipf.write("queue.json")
                
                # Add queue_images directory to the zip file if it exists
                queue_images_dir = "queue_images"
                if os.path.exists(queue_images_dir) and os.path.isdir(queue_images_dir):
                    for root, _, files in os.walk(queue_images_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Add file to zip with path relative to queue_images_dir
                            arcname = os.path.join(os.path.basename(queue_images_dir), file)
                            zipf.write(file_path, arcname)
                            print(f"Added {file_path} to {output_path}")
                else:
                    print(f"Warning: {queue_images_dir} directory not found or empty")
                    # Create the directory if it doesn't exist
                    os.makedirs(queue_images_dir, exist_ok=True)

                # Add input_files directory to the zip file if it exists
                input_files_dir = "input_files"
                if os.path.exists(input_files_dir) and os.path.isdir(input_files_dir):
                    for root, _, files in os.walk(input_files_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Add file to zip with path relative to input_files_dir
                            arcname = os.path.join(os.path.basename(input_files_dir), file)
                            zipf.write(file_path, arcname)
                            print(f"Added {file_path} to {output_path}")
                else:
                    print(f"Warning: {input_files_dir} directory not found or empty")
                    # Create the directory if it doesn't exist
                    os.makedirs(input_files_dir, exist_ok=True)
            
            print(f"Queue exported to {output_path}")
            return output_path
            
        except Exception as e:
            import traceback
            print(f"Error exporting queue to zip: {e}")
            traceback.print_exc()
            return None
    
    def load_queue_from_json(self, file_path=None):
        """Load queue from a JSON file or zip file
        
        Args:
            file_path: Path to the JSON or ZIP file. If None, uses 'queue.json' in the current directory.
            
        Returns:
            int: Number of jobs loaded
        """
        try:
            # Import required modules
            import os
            import json
            from pathlib import PurePath
            
            # Use default path if none provided
            if file_path is None:
                file_path = "queue.json"
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Queue file not found: {file_path}")
                return 0
            
            # Check if it's a zip file
            if file_path.lower().endswith('.zip'):
                return self._load_queue_from_zip(file_path)
            
            # Load the JSON data
            with open(file_path, 'r') as f:
                serialized_jobs = json.load(f)
            
            # Count of jobs loaded
            loaded_count = 0
            
            # Process each job
            with self.lock:
                for job_id, job_data in serialized_jobs.items():
                    # Skip if job already exists
                    if job_id in self.jobs:
                        print(f"Job {job_id} already exists, skipping")
                        continue
                    
                    # Skip completed, failed, or cancelled jobs
                    status = job_data.get('status')
                    if status in ['completed', 'failed', 'cancelled']:
                        print(f"Skipping job {job_id} with status {status}")
                        continue
                    
                    # If the job was running when saved, we'll need to set it as the current job
                    was_running = (status == 'running')
                    
                    # Extract relevant fields to construct params
                    params = {
                        # Basic parameters
                        'model_type': job_data.get('model_type', 'Original'),
                        'prompt_text': job_data.get('prompt', ''),
                        'n_prompt': job_data.get('negative_prompt', ''),
                        'seed': job_data.get('seed', 0),
                        'steps': job_data.get('steps', 25),
                        'cfg': job_data.get('cfg', 1.0),
                        'gs': job_data.get('gs', 10.0),
                        'rs': job_data.get('rs', 0.0),
                        'latent_type': job_data.get('latent_type', 'Black'),
                        'total_second_length': job_data.get('total_second_length', 6),
                        'blend_sections': job_data.get('blend_sections', 4),
                        'latent_window_size': job_data.get('latent_window_size', 9),
                        'resolutionW': job_data.get('resolutionW', 640),
                        'resolutionH': job_data.get('resolutionH', 640),
                        
                        # Initialize image parameters
                        'input_image': None,
                        'end_frame_image': None,
                        'end_frame_strength': job_data.get('end_frame_strength', 1.0),
                        'use_teacache': job_data.get('use_teacache', True),
                        'teacache_num_steps': job_data.get('teacache_num_steps', 25),
                        'teacache_rel_l1_thresh': job_data.get('teacache_rel_l1_thresh', 0.15),
                        'has_input_image': job_data.get('has_input_image', True),
                    }
                    
                    # Load input image from disk if saved path exists
                    if "saved_input_image_path" in job_data and os.path.exists(job_data["saved_input_image_path"]):
                        try:
                            input_image_path = job_data["saved_input_image_path"]
                            print(f"Loading input image from {input_image_path}")
                            input_image = np.array(Image.open(input_image_path))
                            params['input_image'] = input_image
                            params['input_image_path'] = input_image_path  # Store the path for reference
                            params['has_input_image'] = True
                        except Exception as e:
                            print(f"Error loading input image for job {job_id}: {e}")
                    
                    # Load video from disk if saved path exists
                    if "input_video" in job_data and os.path.exists(job_data["input_video"]):
                        try:
                            video_path = job_data["input_video"]
                            print(f"Loading video from {video_path}")
                            params['input_image'] = video_path
                            params['input_image_path'] = video_path
                            params['has_input_image'] = True
                        except Exception as e:
                            print(f"Error loading video for job {job_id}: {e}")
                    
                    # Load end frame image from disk if saved path exists
                    if "saved_end_frame_image_path" in job_data and os.path.exists(job_data["saved_end_frame_image_path"]):
                        try:
                            end_frame_image_path = job_data["saved_end_frame_image_path"]
                            print(f"Loading end frame image from {end_frame_image_path}")
                            end_frame_image = np.array(Image.open(end_frame_image_path))
                            params['end_frame_image'] = end_frame_image
                            params['end_frame_image_path'] = end_frame_image_path  # Store the path for reference
                            # Make sure end_frame_strength is set if this is an endframe model
                            if params['model_type'] == "Original with Endframe" or params['model_type'] == "F1 with Endframe":
                                if 'end_frame_strength' not in params or params['end_frame_strength'] is None:
                                    params['end_frame_strength'] = job_data.get('end_frame_strength', 1.0)
                                    print(f"Set end_frame_strength to {params['end_frame_strength']} for job {job_id}")
                        except Exception as e:
                            print(f"Error loading end frame image for job {job_id}: {e}")
                    
                    # Add LoRA information if present
                    if 'loras' in job_data:
                        lora_data = job_data.get('loras', {})
                        selected_loras = list(lora_data.keys())
                        lora_values = list(lora_data.values())
                        params['selected_loras'] = selected_loras
                        params['lora_values'] = lora_values
                        
                        # Ensure the selected LoRAs are also in lora_loaded_names
                        # This is critical for metadata_utils.create_metadata to find the LoRAs
                        from modules.settings import Settings
                        settings = Settings()
                        lora_dir = settings.get("lora_dir", "loras")
                        
                        # Get the current lora_loaded_names from the system
                        import os
                        from pathlib import PurePath
                        current_lora_names = []
                        if os.path.isdir(lora_dir):
                            for root, _, files in os.walk(lora_dir):
                                for file in files:
                                    if file.endswith('.safetensors') or file.endswith('.pt'):
                                        lora_relative_path = os.path.relpath(os.path.join(root, file), lora_dir)
                                        lora_name = str(PurePath(lora_relative_path).with_suffix(''))
                                        current_lora_names.append(lora_name)
                        
                        # Combine the selected LoRAs with the current lora_loaded_names
                        # This ensures that all selected LoRAs are in lora_loaded_names
                        combined_lora_names = list(set(current_lora_names + selected_loras))
                        params['lora_loaded_names'] = combined_lora_names
                        
                        print(f"Loaded LoRA data for job {job_id}: {lora_data}")
                        print(f"Combined lora_loaded_names: {combined_lora_names}")
                    
                    # Get settings for output_dir and metadata_dir
                    settings = Settings()
                    output_dir = settings.get("output_dir")
                    metadata_dir = settings.get("metadata_dir")
                    input_files_dir = settings.get("input_files_dir")
                    
                    # Add these directories to the params
                    params['output_dir'] = output_dir
                    params['metadata_dir'] = metadata_dir
                    params['input_files_dir'] = input_files_dir
                    
                    # Create a dummy preview image for the job
                    dummy_preview = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    # Create progress data with the dummy preview
                    from diffusers_helper.gradio.progress_bar import make_progress_bar_html
                    initial_progress_data = {
                        'preview': dummy_preview,
                        'desc': 'Imported job...',
                        'html': make_progress_bar_html(0, 'Imported job...')
                    }
                    
                    # Create a dummy preview image for the job
                    dummy_preview = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    # Create progress data with the dummy preview
                    from diffusers_helper.gradio.progress_bar import make_progress_bar_html
                    initial_progress_data = {
                        'preview': dummy_preview,
                        'desc': 'Imported job...',
                        'html': make_progress_bar_html(0, 'Imported job...')
                    }
                    
                    # Create a new job
                    job = Job(
                        id=job_id,
                        params=params,
                        status=JobStatus(job_data.get('status', 'pending')),
                        created_at=job_data.get('created_at', time.time()),
                        progress_data={},
                        stream=AsyncStream(),
                        # Mark images as saved if their paths exist in the job data
                        input_image_saved="saved_input_image_path" in job_data and os.path.exists(job_data["saved_input_image_path"]),
                        end_frame_image_saved="saved_end_frame_image_path" in job_data and os.path.exists(job_data["saved_end_frame_image_path"])
                    )
                    
                    # Add job to the internal jobs dictionary
                    self.jobs[job_id] = job
                    
                    # If a job was marked "running" in the JSON, reset it to "pending"
                    # and add it to the processing queue.
                    if was_running:
                        print(f"Job {job_id} was 'running', resetting to 'pending' and adding to queue.")
                        job.status = JobStatus.PENDING
                        job.started_at = None # Clear started_at for re-queued job
                        job.progress_data = {} # Reset progress
                    
                    # Add all non-completed/failed/cancelled jobs (now including reset 'running' ones) to the processing queue
                    if job.status == JobStatus.PENDING:
                        self.queue.put(job_id)
                        loaded_count += 1
            
            # Synchronize queue images after loading the queue
            self.synchronize_queue_images()
            
            print(f"Loaded {loaded_count} pending jobs from {file_path}")
            return loaded_count
            
        except Exception as e:
            import traceback
            print(f"Error loading queue from JSON: {e}")
            traceback.print_exc()
            return 0
    
    def _load_queue_from_zip(self, zip_path):
        """Load queue from a zip file
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            int: Number of jobs loaded
        """
        try:
            # Create a temporary directory to extract the zip file
            temp_dir = "temp_queue_import"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Check if queue.json exists in the extracted files
            queue_json_path = os.path.join(temp_dir, "queue.json")
            if not os.path.exists(queue_json_path):
                print(f"queue.json not found in {zip_path}")
                shutil.rmtree(temp_dir)
                return 0
            
            # Check if queue_images directory exists in the extracted files
            queue_images_dir = os.path.join(temp_dir, "queue_images")
            if os.path.exists(queue_images_dir) and os.path.isdir(queue_images_dir):
                # Copy the queue_images directory to the current directory
                target_queue_images_dir = "queue_images"
                os.makedirs(target_queue_images_dir, exist_ok=True)
                
                # Copy all files from the extracted queue_images directory to the target directory
                for file in os.listdir(queue_images_dir):
                    src_path = os.path.join(queue_images_dir, file)
                    dst_path = os.path.join(target_queue_images_dir, file)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {src_path} to {dst_path}")
            
            # Check if input_files directory exists in the extracted files
            input_files_dir = os.path.join(temp_dir, "input_files")
            if os.path.exists(input_files_dir) and os.path.isdir(input_files_dir):
                # Copy the input_files directory to the current directory
                target_input_files_dir = "input_files"
                os.makedirs(target_input_files_dir, exist_ok=True)
                
                # Copy all files from the extracted input_files directory to the target directory
                for file in os.listdir(input_files_dir):
                    src_path = os.path.join(input_files_dir, file)
                    dst_path = os.path.join(target_input_files_dir, file)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {src_path} to {dst_path}")
                
            # Update paths in the queue.json file to reflect the new location of the images
            try:
                with open(queue_json_path, 'r') as f:
                    queue_data = json.load(f)
                
                # Update paths for each job
                for job_id, job_data in queue_data.items():
                    # Check for files with job_id in the name to identify input and end frame images
                    input_image_filename = f"{job_id}_input.png"
                    end_frame_image_filename = f"{job_id}_end_frame.png"
                    
                    # Check if these files exist in the target directory
                    input_image_path = os.path.join(target_queue_images_dir, input_image_filename)
                    end_frame_image_path = os.path.join(target_queue_images_dir, end_frame_image_filename)
                    
                    # Update paths in job_data
                    if os.path.exists(input_image_path):
                        job_data["saved_input_image_path"] = input_image_path
                        print(f"Updated input image path for job {job_id}: {input_image_path}")
                    elif "saved_input_image_path" in job_data:
                        # Fallback to updating the existing path
                        job_data["saved_input_image_path"] = os.path.join(target_queue_images_dir, os.path.basename(job_data["saved_input_image_path"]))
                        print(f"Updated existing input image path for job {job_id}")
                    
                    if os.path.exists(end_frame_image_path):
                        job_data["saved_end_frame_image_path"] = end_frame_image_path
                        print(f"Updated end frame image path for job {job_id}: {end_frame_image_path}")
                    elif "saved_end_frame_image_path" in job_data:
                        # Fallback to updating the existing path
                        job_data["saved_end_frame_image_path"] = os.path.join(target_queue_images_dir, os.path.basename(job_data["saved_end_frame_image_path"]))
                        print(f"Updated existing end frame image path for job {job_id}")

                    if "input_video" in job_data:
                        job_data["input_video"] = os.path.join("input_files", os.path.basename(job_data["input_video"]))
                        print(f"Updated video path for job {job_id}: {job_data['input_video']}")

                # Write the updated queue.json back to the file
                with open(queue_json_path, 'w') as f:
                    json.dump(queue_data, f, indent=2)
                
                print(f"Updated image paths in queue.json to reflect new location")
            except Exception as e:
                print(f"Error updating paths in queue.json: {e}")
            
            # Load the queue from the extracted queue.json
            loaded_count = self.load_queue_from_json(queue_json_path)
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
            
            return loaded_count
            
        except Exception as e:
            import traceback
            print(f"Error loading queue from zip: {e}")
            traceback.print_exc()
            # Clean up the temporary directory if it exists
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return 0
    
    def _worker_loop(self):
        """Worker thread that processes jobs from the queue"""
        while True:
            try:
                # Get the next job ID from the queue
                try:
                    job_id = self.queue.get(block=True, timeout=1.0)
                except queue_module.Empty:
                    self._check_and_process_completed_grids()
                    continue

                with self.lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        self.queue.task_done()
                        continue
                    
                    # Skip cancelled jobs
                    if job.status == JobStatus.CANCELLED:
                        self.queue.task_done()
                        continue
                    
                    # If it's a grid job, queue its children and mark it as running
                    if job.job_type == JobType.GRID:
                        print(f"Processing grid job {job.id}, adding {len(job.child_job_ids)} child jobs to queue.")
                        job.status = JobStatus.RUNNING # Mark the grid job as running
                        job.started_at = time.time()
                        # Add child jobs to the front of the queue
                        temp_queue = []
                        while not self.queue.empty():
                            temp_queue.append(self.queue.get())
                        for child_id in reversed(job.child_job_ids): # Add in reverse to maintain order
                            self.queue.put(child_id)
                        for item in temp_queue:
                            self.queue.put(item)
                        
                        self.queue.task_done()
                        continue # Continue to the next iteration to process the first child job

                    # If we're already processing a job, wait for it to complete
                    if self.is_processing:
                        # Check if this is the job that's already marked as running
                        # This can happen if the job was marked as running but not yet processed
                        if job.status == JobStatus.RUNNING and self.current_job and self.current_job.id == job_id:
                            print(f"Job {job_id} is already marked as running, processing it now")
                            # We'll process this job now
                            pass
                        else:
                            # Put the job back in the queue
                            self.queue.put(job_id)
                            self.queue.task_done()
                            time.sleep(0.1)  # Small delay to prevent busy waiting
                            continue
                    
                    # Check if there's a previously running job that was interrupted
                    previously_running_job = None
                    for j in self.jobs.values():
                        if j.status == JobStatus.RUNNING and j.id != job_id:
                            previously_running_job = j
                            break
                    
                    # If there's a previously running job, process it first
                    if previously_running_job:
                        print(f"Found previously running job {previously_running_job.id}, processing it first")
                        # Put the current job back in the queue
                        self.queue.put(job_id)
                        self.queue.task_done()
                        # Process the previously running job
                        job = previously_running_job
                        job_id = previously_running_job.id
                        
                        # Create a new stream for the resumed job and initialize progress_data
                        job.stream = AsyncStream()
                        job.progress_data = {}
                        
                        # Push an initial progress update to the stream
                        from diffusers_helper.gradio.progress_bar import make_progress_bar_html
                        job.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Resuming job...'))))
                    
                    print(f"Starting job {job_id}, current job was {self.current_job.id if self.current_job else 'None'}")
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    self.current_job = job
                    self.is_processing = True
                
                job_completed = False
                
                try:
                    if self.worker_function is None:
                        raise ValueError("Worker function not set. Call set_worker_function() first.")
                    
                    # Start the worker function with the job parameters
                    from diffusers_helper.thread_utils import async_run
                    print(f"Starting worker function for job {job_id}")
                    
                    # Clean up params for the worker function
                    worker_params = job.params.copy()
                    if 'end_frame_image_original' in worker_params:
                        del worker_params['end_frame_image_original']
                    if 'end_frame_strength_original' in worker_params:
                        del worker_params['end_frame_strength_original']

                    async_run(
                        self.worker_function,
                        **worker_params,
                        job_stream=job.stream
                    )
                    print(f"Worker function started for job {job_id}")
                    
                    # Process the results from the stream
                    output_filename = None
                    
                    # Track activity time for logging purposes
                    last_activity_time = time.time()
                    
                    while True:
                        # Check if job has been cancelled before processing next output
                        with self.lock:
                            if job.status == JobStatus.CANCELLED:
                                print(f"Job {job_id} was cancelled, breaking out of processing loop")
                                job_completed = True
                                break
                        
                        # Get current time for activity checks
                        current_time = time.time()
                        
                        # Check for inactivity (no output for a while)
                        if current_time - last_activity_time > 60:  # 1 minute of inactivity
                            print(f"Checking if job {job_id} is still active...")
                            # Just a periodic check, don't break yet
                        
                        try:
                            # Try to get data from the queue with a non-blocking approach
                            flag, data = job.stream.output_queue.next()
                            
                            # Update activity time since we got some data
                            last_activity_time = time.time()
                            
                            if flag == 'file':
                                output_filename = data
                                with self.lock:
                                    job.result = output_filename
                            
                            elif flag == 'progress':
                                preview, desc, html = data
                                with self.lock:
                                    job.progress_data = {
                                        'preview': preview,
                                        'desc': desc,
                                        'html': html
                                    }
                            
                            elif flag == 'end':
                                print(f"Received end signal for job {job_id}")
                                job_completed = True
                                break
                                
                        except IndexError:
                            # Queue is empty, wait a bit and try again
                            time.sleep(0.1)
                            continue
                        except Exception as e:
                            print(f"Error processing job output: {e}")
                            # Wait a bit before trying again
                            time.sleep(0.1)
                            continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error processing job {job_id}: {e}")
                    with self.lock:
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.completed_at = time.time()
                    job_completed = True
                
                finally:
                    with self.lock:
                        # Make sure we properly clean up the job state
                        if job.status == JobStatus.RUNNING:
                            if job_completed:
                                job.status = JobStatus.COMPLETED
                            else:
                                # Something went wrong but we didn't mark it as completed
                                job.status = JobStatus.FAILED
                                job.error = "Job processing was interrupted"
                            
                            job.completed_at = time.time()
                    
                    print(f"Finishing job {job_id} with status {job.status}")
                    self.is_processing = False
                    
                    # Check if there's another job in the queue before setting current_job to None
                    # This helps prevent UI flashing when a job is cancelled
                    next_job_id = None
                    try:
                        # Peek at the next job without removing it from the queue
                        if not self.queue.empty():
                            # We can't peek with the standard Queue, so we'll have to get creative
                            # Store the queue items temporarily
                            temp_queue = []
                            while not self.queue.empty():
                                item = self.queue.get()
                                temp_queue.append(item)
                                if next_job_id is None:
                                    next_job_id = item
                            
                            # Put everything back
                            for item in temp_queue:
                                self.queue.put(item)
                    except Exception as e:
                        print(f"Error checking for next job: {e}")
                    
                    # After a job completes or is cancelled, always set current_job to None
                    self.current_job = None
                    
                    # The main loop's self.queue.get() will pick up the next available job.
                    # No need to explicitly find and start the next job here.
                    
                    self.queue.task_done()
                    
                    # Save the queue to JSON after job completion (outside the lock)
                    try:
                        self.save_queue_to_json()
                    except Exception as e:
                        print(f"Error saving queue to JSON after job completion: {e}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in worker loop: {e}")
                
                # Make sure we reset processing state if there was an error
                with self.lock:
                    self.is_processing = False
                    if self.current_job:
                        self.current_job.status = JobStatus.FAILED
                        self.current_job.error = f"Worker loop error: {str(e)}"
                        self.current_job.completed_at = time.time()
                        self.current_job = None
                
                time.sleep(0.5)  # Prevent tight loop on error

    def _check_and_process_completed_grids(self):
        """Check for completed grid jobs and process them."""
        with self.lock:
            # Find all running grid jobs
            running_grid_jobs = [job for job in self.jobs.values() if job.job_type == JobType.GRID and job.status == JobStatus.RUNNING]
            
            for grid_job in running_grid_jobs:
                # Check if all child jobs are completed
                child_jobs = [self.jobs.get(child_id) for child_id in grid_job.child_job_ids]
                
                if not all(child_jobs):
                    print(f"Warning: Some child jobs for grid {grid_job.id} not found.")
                    continue

                all_children_done = all(job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] for job in child_jobs)

                if all_children_done:
                    print(f"All child jobs for grid {grid_job.id} are done. Assembling grid.")
                    # Logic to assemble the grid
                    # This is a placeholder for the actual grid assembly logic
                    # For now, we'll just mark the grid job as completed.
                    
                    # Collect results from child jobs
                    child_results = [child.result for child in child_jobs if child.status == JobStatus.COMPLETED and child.result]
                    
                    if not child_results:
                        print(f"Grid job {grid_job.id} failed because no child jobs completed successfully.")
                        grid_job.status = JobStatus.FAILED
                        grid_job.error = "No child jobs completed successfully."
                        grid_job.completed_at = time.time()
                        continue

                    # Placeholder for grid assembly.
                    # In a real implementation, you would use a tool like FFmpeg or MoviePy to stitch the videos.
                    # For this example, we'll just create a text file with the paths of the child videos.
                    try:
                        output_dir = grid_job.params.get("output_dir", "outputs")
                        grid_filename = os.path.join(output_dir, f"grid_{grid_job.id}.txt")
                        with open(grid_filename, "w") as f:
                            f.write(f"Grid for job: {grid_job.id}\n")
                            f.write("Child video paths:\n")
                            for result_path in child_results:
                                f.write(f"{result_path}\n")
                        
                        grid_job.result = grid_filename
                        grid_job.status = JobStatus.COMPLETED
                        print(f"Grid assembly for job {grid_job.id} complete. Result saved to {grid_filename}")

                    except Exception as e:
                        print(f"Error during grid assembly for job {grid_job.id}: {e}")
                        grid_job.status = JobStatus.FAILED
                        grid_job.error = f"Grid assembly failed: {e}"

                    grid_job.completed_at = time.time()
                    self.save_queue_to_json()
