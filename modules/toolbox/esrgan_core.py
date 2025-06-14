import os
import torch
import gc
import devicetorch
import warnings 
import traceback 

from pathlib import Path
from huggingface_hub import snapshot_download
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url # Import for direct downloads

# Conditional import for GFPGAN
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

from .message_manager import MessageManager

_MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_ESRGAN_PATH = _MODULE_DIR / "model_esrgan"
# Define a path for GFPGAN models, can be within MODEL_ESRGAN_PATH or separate
MODEL_GFPGAN_PATH = _MODULE_DIR / "model_gfpgan" 

class ESRGANUpscaler:
    def __init__(self, message_manager: MessageManager, device: torch.device):
        self.message_manager = message_manager
        self.device = device
        self.model_dir = Path(MODEL_ESRGAN_PATH)
        self.gfpgan_model_dir = Path(MODEL_GFPGAN_PATH) # GFPGAN model directory
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.gfpgan_model_dir, exist_ok=True) # Ensure GFPGAN model dir exists
        
        self.supported_models = {

            "RealESRGAN_x2plus": {
                "filename": "RealESRGAN_x2plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                "hf_repo_id": None,
                "scale": 2,
                "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                "description": "General purpose. Faster than x4 models due to smaller native output. Good for moderate upscaling."
            },
            "RealESRGAN_x4plus": {
                "filename": "RealESRGAN_x4plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "hf_repo_id": None,
                "scale": 4,
                "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                "description": "General purpose. Prioritizes sharpness & detail. Good default for most videos."
            },
            "RealESRNet_x4plus": {
                "filename": "RealESRNet_x4plus.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
                "hf_repo_id": None,
                "scale": 4,
                "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                "description": "Similar to RealESRGAN_x4plus, but trained for higher fidelity, often yielding smoother results."
            },
            "RealESR-general-x4v3": {
                "filename": "realesr-general-x4v3.pth", # Main model
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                "wdn_filename": "realesr-general-wdn-x4v3.pth", # Companion WDN model
                "wdn_file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "scale": 4, "model_class": SRVGGNetCompact,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
                "description": "Versatile SRVGG-based. Balances detail & naturalness. Has adjustable denoise strength." # Updated description
            },
            "RealESRGAN_x4plus_anime_6B": {
                "filename": "RealESRGAN_x4plus_anime_6B.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "hf_repo_id": None,
                "scale": 4,
                "model_class": RRDBNet,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                "description": "Optimized for anime. Lighter 6-block version of x4plus for faster anime upscaling."
            },
            "RealESR_AnimeVideo_v3": {
                "filename": "realesr-animevideov3.pth",
                "file_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                "hf_repo_id": None,
                "scale": 4,
                "model_class": SRVGGNetCompact,
                "model_params": dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
                "description": "Specialized SRVGG-based model for anime. Often excels with animated content."
            }
        }
        
        self.upsamplers: dict[str, dict[str, RealESRGANer | int | None]] = {} 
        self.face_enhancer: GFPGANer | None = None # For GFPGAN

    def _ensure_model_downloaded(self, model_key: str, target_dir: Path | None = None, is_gfpgan: bool = False, is_wdn_companion: bool = False) -> bool:
        # Modified to handle WDN companion model download for RealESR-general-x4v3
        if target_dir is None:
            current_model_dir = self.gfpgan_model_dir if is_gfpgan else self.model_dir
        else:
            current_model_dir = target_dir

        model_info_source = {}
        actual_model_filename = ""

        if is_gfpgan:
            model_info_source = {
                "filename": "GFPGANv1.4.pth",
                "file_url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                "hf_repo_id": None
            }
            actual_model_filename = model_info_source["filename"]
        else:
            if model_key not in self.supported_models:
                self.message_manager.add_error(f"ESRGAN model key '{model_key}' not supported.")
                return False
            
            model_details = self.supported_models[model_key]
            if is_wdn_companion:
                if "wdn_filename" not in model_details or "wdn_file_url" not in model_details:
                    self.message_manager.add_error(f"WDN companion model details missing for '{model_key}'.")
                    return False
                model_info_source = {
                    "filename": model_details["wdn_filename"],
                    "file_url": model_details["wdn_file_url"],
                    "hf_repo_id": None # Assuming direct URL for WDN for now
                }
                actual_model_filename = model_details["wdn_filename"]
            else: # Regular ESRGAN model
                model_info_source = model_details
                actual_model_filename = model_details["filename"]

        model_path = current_model_dir / actual_model_filename

        if not model_path.exists():
            log_prefix = "WDN " if is_wdn_companion else ""
            self.message_manager.add_message(f"{log_prefix}Model '{actual_model_filename}' not found. Downloading...")
            try:
                downloaded_successfully = False
                if "file_url" in model_info_source and model_info_source["file_url"]:
                    urls_to_try = model_info_source["file_url"]
                    if isinstance(urls_to_try, str): urls_to_try = [urls_to_try]
                    
                    for url in urls_to_try:
                        self.message_manager.add_message(f"Attempting download from URL: {url}")
                        try:
                            load_file_from_url(
                                url=url, model_dir=str(current_model_dir),
                                progress=True, file_name=actual_model_filename
                            )
                            if model_path.exists():
                                downloaded_successfully = True
                                self.message_manager.add_success(f"{log_prefix}Model '{actual_model_filename}' downloaded from URL.")
                                break
                        except Exception as e_url:
                            self.message_manager.add_warning(f"Failed to download from {url}: {e_url}. Trying next source.")
                            continue
                
                if not downloaded_successfully and "hf_repo_id" in model_info_source and model_info_source["hf_repo_id"]:
                    self.message_manager.add_message(f"Attempting download from Hugging Face Hub: {model_info_source['hf_repo_id']}")
                    snapshot_download(
                        repo_id=model_info_source["hf_repo_id"], allow_patterns=[actual_model_filename], 
                        local_dir=current_model_dir, local_dir_use_symlinks=False
                    )
                    if model_path.exists():
                        downloaded_successfully = True
                        self.message_manager.add_success(f"{log_prefix}Model '{actual_model_filename}' downloaded from Hugging Face Hub.")

                if not downloaded_successfully:
                    self.message_manager.add_error(f"All download attempts failed for '{actual_model_filename}'.")
                    return False
            except Exception as e:
                self.message_manager.add_error(f"Failed to download {log_prefix}model '{actual_model_filename}': {e}")
                self.message_manager.add_error(traceback.format_exc())
                return False
        return True

    def load_model(self, model_key: str, tile_size: int = 0, denoise_strength: float | None = None) -> RealESRGANer | None:
        if model_key not in self.supported_models:
            self.message_manager.add_error(f"ESRGAN model key '{model_key}' not supported.")
            return None

        # Check if model is already loaded with the same configuration
        current_config_signature = (tile_size, denoise_strength if model_key == "RealESR-general-x4v3" else None)
        
        if model_key in self.upsamplers:
            existing_config = self.upsamplers[model_key]
            existing_config_signature = (
                existing_config.get('tile_size', 0),
                existing_config.get('denoise_strength') if model_key == "RealESR-general-x4v3" else None
            )

            if existing_config.get("upsampler") is not None and existing_config_signature == current_config_signature:
                log_tile = f"Tile: {str(tile_size) if tile_size > 0 else 'Auto'}"
                log_dni = f", DNI: {denoise_strength:.2f}" if denoise_strength is not None and model_key == "RealESR-general-x4v3" else ""
                self.message_manager.add_message(f"ESRGAN model '{model_key}' ({log_tile}{log_dni}) already loaded.")
                return existing_config["upsampler"]
            elif existing_config.get("upsampler") is not None and existing_config_signature != current_config_signature:
                self.message_manager.add_message(
                    f"ESRGAN model '{model_key}' config changed. Unloading to reload with new settings."
                )
                self.unload_model(model_key)

        # Ensure main model is downloaded
        if not self._ensure_model_downloaded(model_key):
            return None

        model_info = self.supported_models[model_key]
        model_path_for_upsampler = str(self.model_dir / model_info["filename"])
        dni_weight_for_upsampler = None
        
        log_msg_parts = [
            f"Loading ESRGAN model '{model_info['filename']}' (Key: {model_key}, Scale: {model_info['scale']}x",
            f"Tile: {str(tile_size) if tile_size > 0 else 'Auto'}"
        ]

        # Specific handling for RealESR-general-x4v3 with denoise_strength
        if model_key == "RealESR-general-x4v3" and denoise_strength is not None and 0.0 <= denoise_strength < 1.0:
            # Denoise strength 1.0 means use only the main model, so no DNI.
            # Denoise strength < 0.0 is invalid.
            if "wdn_filename" not in model_info or "wdn_file_url" not in model_info:
                self.message_manager.add_error(f"WDN companion model details missing for '{model_key}'. Cannot apply denoise strength.")
                return None # Or fallback to no DNI? For now, error.
            
            # Ensure WDN companion model is downloaded
            if not self._ensure_model_downloaded(model_key, is_wdn_companion=True):
                self.message_manager.add_error(f"Failed to download WDN companion for '{model_key}'. Cannot apply denoise strength.")
                return None

            wdn_model_path_str = str(self.model_dir / model_info["wdn_filename"])
            model_path_for_upsampler = [model_path_for_upsampler, wdn_model_path_str] # Pass list of paths
            dni_weight_for_upsampler = [denoise_strength, 1.0 - denoise_strength] # [main_model_strength, wdn_model_strength]
            log_msg_parts.append(f"DNI Strength: {denoise_strength:.2f}")
        
        log_msg_parts.append(f") to device: {self.device}...")
        self.message_manager.add_message(" ".join(log_msg_parts))

        try:
            model_params_with_correct_scale = model_info["model_params"].copy()
            if "scale" in model_params_with_correct_scale: model_params_with_correct_scale["scale"] = model_info["scale"]
            elif "upscale" in model_params_with_correct_scale: model_params_with_correct_scale["upscale"] = model_info["scale"]
            else: model_params_with_correct_scale["scale"] = model_info["scale"]
            
            model_arch = model_info["model_class"](**model_params_with_correct_scale)
            
            gpu_id_for_realesrgan = self.device.index if self.device.type == 'cuda' and self.device.index is not None else None
            use_half_precision = True if self.device.type == 'cuda' else False
            
            with warnings.catch_warnings():
                # Suppress the TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD warning from RealESRGANer/basicsr
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*"
                )
                # Suppress torchvision pretrained/weights warnings potentially triggered by basicsr
                warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated.*")
                warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated.*")
                
                upsampler = RealESRGANer(
                    scale=model_info["scale"],
                    model_path=model_path_for_upsampler,
                    dni_weight=dni_weight_for_upsampler,
                    model=model_arch,
                    tile=tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    half=use_half_precision,
                    gpu_id=gpu_id_for_realesrgan
                )
            
            self.upsamplers[model_key] = {
                "upsampler": upsampler, 
                "tile_size": tile_size, 
                "native_scale": model_info["scale"],
                "denoise_strength": denoise_strength if model_key == "RealESR-general-x4v3" else None
            }
            self.message_manager.add_success(f"ESRGAN model '{model_info['filename']}' (Key: {model_key}) loaded successfully.")
            return upsampler
        except Exception as e:
            self.message_manager.add_error(f"Failed to load ESRGAN model '{model_info['filename']}' (Key: {model_key}): {e}")
            self.message_manager.add_error(traceback.format_exc())
            if model_key in self.upsamplers: del self.upsamplers[model_key]
            return None

    def _load_face_enhancer(self, model_name="GFPGANv1.4.pth", bg_upsampler=None) -> bool:
        if not GFPGAN_AVAILABLE:
            self.message_manager.add_warning("GFPGAN library not available. Cannot load face enhancer.")
            return False
        if self.face_enhancer is not None:
            # If bg_upsampler changed, we might need to re-init. For now, assume if loaded, it's fine or will be handled by caller.
            if bg_upsampler is not None and hasattr(self.face_enhancer, 'bg_upsampler') and self.face_enhancer.bg_upsampler != bg_upsampler:
                self.message_manager.add_message("GFPGAN face enhancer already loaded, but with a different background upsampler. Re-initializing GFPGAN...")
                self._unload_face_enhancer() # Unload to reload with new bg_upsampler
            else:
                self.message_manager.add_message("GFPGAN face enhancer already loaded.")
                return True


        if not self._ensure_model_downloaded(model_key=model_name, is_gfpgan=True): 
            self.message_manager.add_error(f"Failed to download GFPGAN model '{model_name}'.")
            return False

        gfpgan_model_path = str(self.gfpgan_model_dir / model_name)
        self.message_manager.add_message(f"Loading GFPGAN face enhancer from {gfpgan_model_path}...")
        try:
            # --- ADDED: warnings.catch_warnings() context manager ---
            with warnings.catch_warnings():
                # Suppress warnings from GFPGANer and its dependencies (facexlib)
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*"
                )
                warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated.*")
                warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated.*")

                self.face_enhancer = GFPGANer(
                    model_path=gfpgan_model_path,
                    upscale=1, 
                    arch='clean', 
                    channel_multiplier=2,
                    bg_upsampler=bg_upsampler, 
                    device=self.device
                )
            self.message_manager.add_success("GFPGAN face enhancer loaded.")
            return True
        except Exception as e:
            self.message_manager.add_error(f"Failed to load GFPGAN face enhancer: {e}")
            self.message_manager.add_error(traceback.format_exc())
            self.face_enhancer = None
            return False

    def _unload_face_enhancer(self):
        if self.face_enhancer is not None:
            self.message_manager.add_message("Unloading GFPGAN face enhancer...")
            del self.face_enhancer
            self.face_enhancer = None
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.message_manager.add_success("GFPGAN face enhancer unloaded.")
        else:
            self.message_manager.add_message("GFPGAN face enhancer not loaded.")


    def unload_model(self, model_key: str):
        if model_key in self.upsamplers and self.upsamplers[model_key].get("upsampler") is not None:
            config = self.upsamplers.pop(model_key)
            upsampler_instance = config["upsampler"]
            tile_s = config.get("tile_size", 0)
            native_scale = config.get("native_scale", "N/A") # Get native_scale for logging
            log_tile_size = str(tile_s) if tile_s > 0 else "Auto"
            self.message_manager.add_message(f"Unloading ESRGAN model '{model_key}' (Scale: {native_scale}x, Tile: {log_tile_size})...")
            
            if self.face_enhancer and hasattr(self.face_enhancer, 'bg_upsampler') and self.face_enhancer.bg_upsampler == upsampler_instance:
                self.message_manager.add_message("Unloading associated GFPGAN as its BG upsampler is being removed.")
                self._unload_face_enhancer()

            del upsampler_instance 
            devicetorch.empty_cache(torch) 
            gc.collect()
            self.message_manager.add_success(f"ESRGAN model '{model_key}' unloaded and memory cleared.")
        else:
            self.message_manager.add_message(f"ESRGAN model '{model_key}' not loaded, no need to unload.")

    def unload_all_models(self): 
        if not self.upsamplers and not self.face_enhancer:
            self.message_manager.add_message("No ESRGAN or GFPGAN models currently loaded.")
            return
            
        self.message_manager.add_message("Unloading all ESRGAN models...")
        model_keys_to_unload = list(self.upsamplers.keys()) 
        for key in model_keys_to_unload: 
            if key in self.upsamplers: 
                config = self.upsamplers.pop(key)
                upsampler_instance = config["upsampler"]
                del upsampler_instance # type: ignore
        
        self._unload_face_enhancer() 

        devicetorch.empty_cache(torch) 
        gc.collect()
        self.message_manager.add_success("All ESRGAN and GFPGAN models unloaded and memory cleared.")

    def upscale_frame(self, frame_np_array, model_key: str, target_outscale_factor: float, enhance_face: bool = False):
        """
        Upscales a single frame using the specified model and target output scale.
        """
        config = self.upsamplers.get(model_key)
        upsampler: RealESRGANer | None = None
        current_tile_size = 0
        model_native_scale = 0 

        if config and config.get("upsampler"):
            upsampler = config["upsampler"] # type: ignore
            current_tile_size = config.get("tile_size", 0) # type: ignore
            model_native_scale = config.get("native_scale", 0) # type: ignore
            if model_native_scale == 0: 
                self.message_manager.add_error(f"Error: Native scale for model '{model_key}' is 0 or not found in config.")
                return None
        
        if upsampler is None:
            self.message_manager.add_warning(
                f"ESRGAN model '{model_key}' not pre-loaded. Attempting to load now (with default Tile: Auto)..."
            )
            tile_to_load_with = config.get("tile_size", 0) if config else 0
            upsampler = self.load_model(model_key, tile_size=tile_to_load_with)
            if upsampler is None:
                self.message_manager.add_error(f"Failed to auto-load ESRGAN model '{model_key}'. Cannot upscale.")
                return None
            
            loaded_config = self.upsamplers.get(model_key) # Re-fetch config after load
            if loaded_config:
                current_tile_size = loaded_config.get("tile_size", 0) # type: ignore
                model_native_scale = loaded_config.get("native_scale", 0) # type: ignore
                if model_native_scale == 0:
                    self.message_manager.add_error(f"Error: Native scale for auto-loaded model '{model_key}' is 0.")
                    return None
            else: 
                self.message_manager.add_error(f"Error: Config for auto-loaded model '{model_key}' not found.")
                return None

        # Validate target_outscale_factor against model's native scale.
        # Allow outscale from a small factor up to the model's native scale.
        # You could allow slightly more (e.g., model_native_scale * 1.1) if you want to permit minor bicubic post-upscale.
        # For now, strictly <= native_scale.
        if not (0.25 <= target_outscale_factor <= model_native_scale): 
             self.message_manager.add_warning(
                f"Target outscale factor {target_outscale_factor:.2f}x is outside the recommended range "
                f"(0.25x to {model_native_scale:.2f}x) for model '{model_key}' (native {model_native_scale}x). "
                f"Adjusting to model's native scale {model_native_scale:.2f}x."
            )
             target_outscale_factor = float(model_native_scale)


        if enhance_face:
            if not self.face_enhancer or (hasattr(self.face_enhancer, 'bg_upsampler') and self.face_enhancer.bg_upsampler != upsampler):
                self.message_manager.add_message("Face enhancement requested, loading/re-configuring GFPGAN...")
                self._load_face_enhancer(bg_upsampler=upsampler) 
            
            if not self.face_enhancer: 
                self.message_manager.add_warning("GFPGAN could not be loaded. Proceeding without face enhancement.")
                enhance_face = False 

        try:
            img_bgr = frame_np_array[:, :, ::-1] 
            
            outscale_for_enhance = float(target_outscale_factor)

            if enhance_face and self.face_enhancer:
                if self.face_enhancer.upscale != 1: # Ensure GFPGAN is only cleaning, not upscaling itself in this pipeline path
                     self.message_manager.add_warning(
                         f"GFPGANer's internal upscale is {self.face_enhancer.upscale}, but for the 'Clean Face -> ESRGAN Upscale' pipeline, "
                         f"it should be 1. RealESRGAN will handle the main scaling to {target_outscale_factor:.2f}x."
                     )
                
                _, _, cleaned_img_bgr = self.face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                output_bgr, _ = upsampler.enhance(cleaned_img_bgr, outscale=outscale_for_enhance)
            else:
                output_bgr, _ = upsampler.enhance(img_bgr, outscale=outscale_for_enhance) 
            
            output_rgb = output_bgr[:, :, ::-1] 
            return output_rgb
        except Exception as e:
            tile_size_msg_part = str(current_tile_size) if current_tile_size > 0 else 'Auto'
            face_msg_part = " + Face Enhance" if enhance_face else ""
            self.message_manager.add_error(
                f"Error during ESRGAN frame upscaling (Model: {model_key}{face_msg_part}, "
                f"Target Scale: {target_outscale_factor:.2f}x, Native: {model_native_scale}x, Tile: {tile_size_msg_part}): {e}"
            )
            self.message_manager.add_error(traceback.format_exc()) 
            if "out of memory" in str(e).lower() and self.device.type == 'cuda':
                self.message_manager.add_warning(
                    "CUDA OOM during upscaling. Emptying cache. "
                    f"Current model (Model: {model_key}, Tile: {tile_size_msg_part}) may need reloading. "
                    "Consider using a smaller tile size or a smaller input video if issues persist."
                )
                devicetorch.empty_cache(torch)
            return None