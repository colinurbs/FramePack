import gradio as gr
import os
import sys
import torch
import devicetorch
import traceback
import gc
import psutil
import json # for preset loading/saving

# patch fix for basicsr
from torchvision.transforms.functional import rgb_to_grayscale
import types
functional_tensor_mod = types.ModuleType('functional_tensor')
functional_tensor_mod.rgb_to_grayscale = rgb_to_grayscale
sys.modules.setdefault('torchvision.transforms.functional_tensor', functional_tensor_mod)

from modules.toolbox.toolbox_processor import VideoProcessor
from modules.toolbox.message_manager import MessageManager
from modules.toolbox.system_monitor import SystemMonitor
from modules.settings import Settings

try:
    from diffusers_helper.memory import cpu
except ImportError:
    print("WARNING: Could not import cpu from diffusers_helper.memory. Falling back to torch.device('cpu')")
    cpu = torch.device('cpu')


tb_message_mgr = MessageManager()
settings_instance = Settings()
tb_processor = VideoProcessor(tb_message_mgr, settings_instance) # Pass settings to VideoProcessor

# --- Default Filter Values ---
TB_DEFAULT_FILTER_SETTINGS = {
    "brightness": 0, "contrast": 1, "saturation": 1, "temperature": 0,
    "sharpen": 0, "blur": 0, "denoise": 0, "vignette": 0,
    "s_curve_contrast": 0, "film_grain_strength": 0
}

# --- Filter Presets Handling ---
TB_BUILT_IN_PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toolbox", "data", "filter_presets.json")
tb_filter_presets_data = {} # Will be populated by _initialize_presets

def _initialize_presets():
    global tb_filter_presets_data
    default_preset_map_for_creation = {
        "none": TB_DEFAULT_FILTER_SETTINGS.copy(),
        "cinematic": {"brightness": -5, "contrast": 1.3, "saturation": 0.9, "temperature": 20, "vignette": 10, "sharpen": 1.2, "blur": 0, "denoise": 0, "s_curve_contrast": 15, "film_grain_strength": 5},
        "vintage": {"brightness": 5, "contrast": 1.1, "saturation": 0.7, "temperature": 15, "vignette": 30, "sharpen": 0, "blur": 0.5, "denoise": 0, "s_curve_contrast": 10, "film_grain_strength": 10},
        "cool": {"brightness": 0, "contrast": 1.2, "saturation": 1.1, "temperature": -15, "vignette": 0, "sharpen": 1.0, "blur": 0, "denoise": 0, "s_curve_contrast": 5, "film_grain_strength": 0},
        "warm": {"brightness": 5, "contrast": 1.1, "saturation": 1.2, "temperature": 20, "vignette": 0, "sharpen": 0, "blur": 0, "denoise": 0, "s_curve_contrast": 5, "film_grain_strength": 0},
        "dramatic": {"brightness": -5, "contrast": 1.2, "saturation": 0.9, "temperature": -10, "vignette": 20, "sharpen": 1.2, "blur": 0, "denoise": 0, "s_curve_contrast": 20, "film_grain_strength": 8}
    }
    try:
        os.makedirs(os.path.dirname(TB_BUILT_IN_PRESETS_FILE), exist_ok=True)
        if not os.path.exists(TB_BUILT_IN_PRESETS_FILE):
            tb_message_mgr.add_message(f"Presets file not found. Creating with default presets: {TB_BUILT_IN_PRESETS_FILE}", "INFO")
            with open(TB_BUILT_IN_PRESETS_FILE, 'w') as f:
                json.dump(default_preset_map_for_creation, f, indent=4)
            tb_filter_presets_data = default_preset_map_for_creation
            tb_message_mgr.add_success("Default presets file created.")
        else:
            with open(TB_BUILT_IN_PRESETS_FILE, 'r') as f:
                tb_filter_presets_data = json.load(f)
            # Ensure "none" preset always exists and uses TB_DEFAULT_FILTER_SETTINGS
            if "none" not in tb_filter_presets_data or tb_filter_presets_data["none"] != TB_DEFAULT_FILTER_SETTINGS:
                 tb_filter_presets_data["none"] = TB_DEFAULT_FILTER_SETTINGS.copy()
                 # Optionally re-save if "none" was missing or incorrect, or just use in-memory fix
                 # with open(TB_BUILT_IN_PRESETS_FILE, 'w') as f:
                 #    json.dump(tb_filter_presets_data, f, indent=4)

            tb_message_mgr.add_message(f"Filter presets loaded from {TB_BUILT_IN_PRESETS_FILE}.", "INFO")
    except Exception as e:
        tb_message_mgr.add_error(f"Error with filter presets file {TB_BUILT_IN_PRESETS_FILE}: {e}. Using in-memory defaults.")
        tb_filter_presets_data = default_preset_map_for_creation
_initialize_presets() # Call once when the script module is loaded

def tb_update_messages():
    return tb_message_mgr.get_messages()

def tb_handle_update_monitor(monitor_enabled): # This updates the TOOLBOX TAB's monitor
    if not monitor_enabled:
        return gr.update() # Do nothing if disabled to save resources.
    return SystemMonitor.get_system_info()

def tb_handle_analyze_video(video_path):
    tb_message_mgr.clear()
    analysis = tb_processor.tb_analyze_video_input(video_path)
    return tb_update_messages(), analysis

def tb_handle_process_frames(video_path, fps_mode, speed_factor, progress=gr.Progress()):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_process_frames(video_path, fps_mode, speed_factor, progress)
    return output_video, tb_update_messages()

def tb_handle_create_loop(video_path, loop_type, num_loops):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_create_loop(video_path, loop_type, num_loops)
    return output_video, tb_update_messages()

def tb_update_filter_sliders_from_preset(preset_name):
    preset_settings = tb_filter_presets_data.get(preset_name)
    if not preset_settings:
        tb_message_mgr.add_warning(f"Preset '{preset_name}' not found. Using 'none' settings.")
        preset_settings = tb_filter_presets_data.get("none", TB_DEFAULT_FILTER_SETTINGS.copy())
    
    # Ensure all keys from default are present, using default value if missing in loaded preset
    # and also ensuring that the loaded preset values take precedence if they exist.
    final_settings = TB_DEFAULT_FILTER_SETTINGS.copy() # Start with defaults
    final_settings.update(preset_settings) # Override with preset values

    # Ensure only known keys are passed through, to prevent errors if preset has extra keys
    # And ensure the order matches TB_DEFAULT_FILTER_SETTINGS keys
    ordered_values = []
    for key in TB_DEFAULT_FILTER_SETTINGS.keys():
        ordered_values.append(final_settings.get(key, TB_DEFAULT_FILTER_SETTINGS[key]))
        
    return tuple(ordered_values)

def tb_handle_reset_all_filters():
    tb_message_mgr.add_message("Filter sliders reset to default 'none' values.")
    none_settings_values = tb_update_filter_sliders_from_preset("none")
    # Returns: dropdown_value, preset_name_input_value, *slider_values, messages_value
    return "none", "", *none_settings_values, tb_update_messages()

def tb_handle_save_user_preset(new_preset_name_str, *slider_values):
    global tb_filter_presets_data; tb_message_mgr.clear()
    if not new_preset_name_str or not new_preset_name_str.strip():
        tb_message_mgr.add_warning("Preset name cannot be empty."); return gr.update(), tb_update_messages(), gr.update()
    
    clean_preset_name = new_preset_name_str.strip()

    new_preset_values = dict(zip(TB_DEFAULT_FILTER_SETTINGS.keys(), slider_values))
    preset_existed = clean_preset_name in tb_filter_presets_data
    tb_filter_presets_data[clean_preset_name] = new_preset_values
    try:
        with open(TB_BUILT_IN_PRESETS_FILE, 'w') as f: json.dump(tb_filter_presets_data, f, indent=4)
        tb_message_mgr.add_success(f"Preset '{clean_preset_name}' {'updated' if preset_existed else 'saved'} successfully!")
        
        updated_choices = list(tb_filter_presets_data.keys())
        if "none" in updated_choices: updated_choices.remove("none"); updated_choices.sort(); updated_choices.insert(0, "none")
        else: updated_choices.sort()
        
        return gr.update(choices=updated_choices, value=clean_preset_name), tb_update_messages(), ""
    except Exception as e:
        tb_message_mgr.add_error(f"Error saving preset '{clean_preset_name}': {e}")
        _initialize_presets()
        return gr.update(), tb_update_messages(), gr.update(value=new_preset_name_str)

def tb_handle_delete_user_preset(preset_name_to_delete):
    global tb_filter_presets_data; tb_message_mgr.clear()
    if not preset_name_to_delete or not preset_name_to_delete.strip():
        tb_message_mgr.add_warning("No preset name to delete (select from dropdown or type)."); return gr.update(), tb_update_messages(), gr.update(), *tb_update_filter_sliders_from_preset("none")
    
    clean_preset_name = preset_name_to_delete.strip()
    if clean_preset_name.lower() == "none":
        tb_message_mgr.add_warning("'none' preset cannot be deleted."); return gr.update(), tb_update_messages(), gr.update(value="none"), *tb_update_filter_sliders_from_preset("none")
    if clean_preset_name not in tb_filter_presets_data:
        tb_message_mgr.add_warning(f"Preset '{clean_preset_name}' not found."); return gr.update(), tb_update_messages(), gr.update(), *tb_update_filter_sliders_from_preset("none")

    del tb_filter_presets_data[clean_preset_name]
    try:
        with open(TB_BUILT_IN_PRESETS_FILE, 'w') as f: json.dump(tb_filter_presets_data, f, indent=4)
        tb_message_mgr.add_success(f"Preset '{clean_preset_name}' deleted.")
        
        updated_choices = list(tb_filter_presets_data.keys())
        if "none" in updated_choices: updated_choices.remove("none"); updated_choices.sort(); updated_choices.insert(0, "none")
        else: updated_choices.sort()
        
        sliders_reset_values = tb_update_filter_sliders_from_preset("none")
        return gr.update(choices=updated_choices, value="none"), tb_update_messages(), "", *sliders_reset_values
    except Exception as e:
        tb_message_mgr.add_error(f"Error deleting preset '{clean_preset_name}' from file: {e}")
        _initialize_presets();
        current_choices = list(tb_filter_presets_data.keys())
        if "none" in current_choices: current_choices.remove("none"); current_choices.sort(); current_choices.insert(0, "none")
        else: current_choices.sort()
        selected_val_after_error = clean_preset_name if clean_preset_name in current_choices else "none"
        sliders_after_error_values = tb_update_filter_sliders_from_preset(selected_val_after_error)
        return gr.update(choices=current_choices, value=selected_val_after_error), tb_update_messages(), gr.update(value=selected_val_after_error), *sliders_after_error_values

def tb_handle_apply_filters(video_path, brightness, contrast, saturation, temperature,
                         sharpen, blur, denoise, vignette,
                         s_curve_contrast, film_grain_strength,
                         progress=gr.Progress()):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_apply_filters(video_path, brightness, contrast, saturation, temperature, 
                                          sharpen, blur, denoise, vignette,
                                          s_curve_contrast, film_grain_strength, progress)
    return output_video, tb_update_messages()
    
def tb_handle_reassemble_frames(
    selected_extracted_folder, # New input from dropdown
    uploaded_frames_dir_info,  # Existing input from gr.File
    output_fps,
    output_video_name, # New input from Textbox
    progress=gr.Progress()
):
    tb_message_mgr.clear()

    frames_source_to_use = None
    source_description = ""

    if selected_extracted_folder and selected_extracted_folder.strip():
        # Dropdown selection takes precedence
        frames_source_to_use = os.path.join(tb_processor.extracted_frames_target_path, selected_extracted_folder)
        source_description = f"selected folder '{selected_extracted_folder}'"
        if not os.path.isdir(frames_source_to_use):
            tb_message_mgr.add_error(f"Selected folder '{selected_extracted_folder}' not found at expected path: {frames_source_to_use}")
            return None, tb_update_messages()
    elif uploaded_frames_dir_info and (isinstance(uploaded_frames_dir_info, list) and uploaded_frames_dir_info):
        # Fallback to gr.File upload if dropdown is not used
        frames_source_to_use = uploaded_frames_dir_info
        source_description = "uploaded files/folder"
    else:
        tb_message_mgr.add_warning("No frame source selected or provided (neither dropdown nor file upload).")
        return None, tb_update_messages()

    tb_message_mgr.add_message(f"Attempting to reassemble frames from {source_description}.")
    output_video = tb_processor.tb_reassemble_frames_to_video(
        frames_source_to_use,
        output_fps,
        output_base_name_override=output_video_name, # Pass the desired name
        progress=progress
    )
    return output_video, tb_update_messages()

# NEW HANDLERS for extracted frames folder management
def tb_handle_extract_frames(video_path, extraction_rate, progress=gr.Progress()):
    tb_message_mgr.clear()
    tb_processor.tb_extract_frames(video_path, int(extraction_rate), progress)
    return tb_update_messages()

def tb_handle_refresh_extracted_folders():
    # tb_message_mgr.clear() # Optional: clear messages before refresh
    folders = tb_processor.tb_get_extracted_frame_folders()
    # Disable clear button if no folders, enable if folders exist and one might be selected
    clear_btn_update = gr.update(interactive=False) # Default to disabled
    if folders:
        # We don't know if one is selected yet, so keep it disabled until selection.
        # Or, if you want to enable it if *any* folder exists, change logic here.
        pass # Keep it disabled for now. It will be enabled on dropdown.change
    return gr.update(choices=folders, value=None), tb_update_messages(), clear_btn_update

def tb_handle_clear_selected_folder(selected_folder_to_delete):
    tb_message_mgr.clear()
    if not selected_folder_to_delete:
        tb_message_mgr.add_warning("No folder selected from the dropdown to delete.")
        return tb_update_messages(), gr.update() # Return current dropdown state

    success = tb_processor.tb_delete_extracted_frames_folder(selected_folder_to_delete)

    # After deletion, refresh the folder list
    updated_folders = tb_processor.tb_get_extracted_frame_folders()
    # If the deleted folder was the one selected, the value should clear.
    # The dropdown will clear itself if its current value is no longer in choices.
    return tb_update_messages(), gr.update(choices=updated_folders, value=None)

def tb_handle_upscale_video(video_path, model_key_selected, output_scale_factor_from_slider, tile_size, enhance_face_ui, denoise_strength_from_slider, progress=gr.Progress()):
    tb_message_mgr.clear()
    if video_path is None:
        tb_message_mgr.add_warning("No input video selected for upscaling.")
        return None, tb_update_messages()
    if not model_key_selected:
        tb_message_mgr.add_warning("No upscale model selected.")
        return None, tb_update_messages()

    try:
        tile_size_int = int(tile_size)
    except ValueError:
        tb_message_mgr.add_error(f"Invalid tile size value: {tile_size}. Using Auto (0).")
        tile_size_int = 0
    
    try:
        output_scale_factor_float = float(output_scale_factor_from_slider)
        # Add a more robust check based on what ESRGAN core expects, e.g., > 0.1 or similar
        if not (output_scale_factor_float >= 0.25): # Matches the lower bound in esrgan_core's check
             tb_message_mgr.add_error(f"Invalid output scale factor: {output_scale_factor_from_slider:.2f}. Must be >= 0.25.")
             return None, tb_update_messages()
    except ValueError:
        tb_message_mgr.add_error(f"Invalid output scale factor: {output_scale_factor_from_slider}. Not a valid number.")
        return None, tb_update_messages()

    output_video = tb_processor.tb_upscale_video(
        video_path, 
        model_key_selected, 
        output_scale_factor_float, # Pass the new scale factor
        tile_size_int, 
        enhance_face_ui, 
        denoise_strength_from_slider,
        progress=progress 
    )
    return output_video, tb_update_messages()
    
def tb_get_model_info_and_update_scale_slider(model_key_selected: str):
    """
    Fetches model info and returns Gradio updates for the model info textbox,
    the outscale factor slider, and the denoise strength slider.
    """
    native_scale = 2.0  # Default native scale if model not found or scale not specified
    slider_min = 1.0
    slider_max = 2.0
    slider_step = 0.05
    slider_default_value = 2.0 # Will be overridden by native_scale
    model_info_text = "Info: Select a model."
    slider_label = "Target Upscale Factor"
    
    denoise_slider_visible = False
    denoise_slider_value = 0.5

    if model_key_selected and model_key_selected in tb_processor.esrgan_upscaler.supported_models:
        model_details = tb_processor.esrgan_upscaler.supported_models[model_key_selected]
        fetched_native_scale = model_details.get('scale')
        description = model_details.get('description', 'No description available.')
        
        if isinstance(fetched_native_scale, (int, float)) and fetched_native_scale > 0:
            native_scale = float(fetched_native_scale)
            slider_max = native_scale 
            slider_default_value = native_scale
            slider_min = max(1.0, native_scale / 4.0) # Example: Allow downscaling to 1/4th of native, but not below 1.0
            slider_min = 1.0 # Cap minimum at 1.0x output relative to original input

            if native_scale >= 4.0:
                slider_step = 0.1
            elif native_scale >= 2.0:
                slider_step = 0.05

        model_info_text = f"{description}"
        slider_label = f"Target Upscale Factor (Native {native_scale}x)"

        if model_key_selected == "RealESR-general-x4v3":
            denoise_slider_visible = True

    model_info_update = gr.update(value=model_info_text)
    outscale_slider_update = gr.update(
        minimum=slider_min, # This will now be 1.0 for typical ESRGAN models
        maximum=slider_max,
        step=slider_step,
        value=slider_default_value,
        label=slider_label
    )
    denoise_slider_update = gr.update(
        visible=denoise_slider_visible,
        value=denoise_slider_value
    )
    
    return model_info_update, outscale_slider_update, denoise_slider_update

def tb_get_selected_model_scale_info(model_key_selected):
    if model_key_selected and model_key_selected in tb_processor.esrgan_upscaler.supported_models:
        model_details = tb_processor.esrgan_upscaler.supported_models[model_key_selected]
        scale = model_details.get('N/A')
        description = model_details.get('description', 'No description available.')
        return f"{description}"
    return "Info: Select a model."
    
def tb_handle_delete_studio_transformer():
    tb_message_mgr.clear()
    tb_message_mgr.add_message("Attempting to directly access and delete Studio transformer...")
    print("Attempting to directly access and delete Studio transformer...")
    log_messages_from_action = [] # This will store detailed steps for the UI log

    studio_module_instance = None
    # Check __main__ first
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'current_generator'):
        studio_module_instance = sys.modules['__main__']
        print("Found studio context in __main__.") # Console log only
    elif 'studio' in sys.modules and hasattr(sys.modules['studio'], 'current_generator'):
        studio_module_instance = sys.modules['studio']
        print("Found studio context in sys.modules['studio'].") # Console log only

    if studio_module_instance is None:
        print("ERROR: Could not find the 'studio' module's active context.")
        tb_message_mgr.add_message("ERROR: Could not find the 'studio' module's active context in sys.modules.")
        tb_message_mgr.add_error("Deletion Failed: Studio module context not found.")
        return tb_update_messages()

    # --- CHECK IF A JOB IS CURRENTLY RUNNING ---
    job_queue_instance = getattr(studio_module_instance, 'job_queue', None)
    JobStatus_enum = getattr(studio_module_instance, 'JobStatus', None) # JobStatus is an enum in modules.video_queue

    if job_queue_instance and JobStatus_enum:
        current_job_in_queue = getattr(job_queue_instance, 'current_job', None)
        if current_job_in_queue and hasattr(current_job_in_queue, 'status') and current_job_in_queue.status == JobStatus_enum.RUNNING:
            tb_message_mgr.add_warning("Cannot unload model: A video generation job is currently running.")
            tb_message_mgr.add_message("Please wait for the current job to complete or cancel it first using the main interface.")
            print("Cannot unload model: A job is currently running in the queue.")
            return tb_update_messages()
    # else: # Optional: for debugging if job_queue or JobStatus is not found on a valid studio_module_instance
        # print("Debug: Could not access job_queue or JobStatus from studio module to check for running jobs.")

    generator_object_to_delete = getattr(studio_module_instance, 'current_generator', None)
    print(f"Direct access: generator_object_to_delete is {type(generator_object_to_delete)}, id: {id(generator_object_to_delete)}")

    if generator_object_to_delete is not None:
        model_name_str = "Unknown Model"
        try:
            if hasattr(generator_object_to_delete, 'get_model_name') and callable(generator_object_to_delete.get_model_name):
                model_name_str = generator_object_to_delete.get_model_name()
            elif hasattr(generator_object_to_delete, 'transformer') and generator_object_to_delete.transformer is not None:
                model_name_str = generator_object_to_delete.transformer.__class__.__name__
            else:
                model_name_str = generator_object_to_delete.__class__.__name__
        except Exception:
            pass # model_name_str remains "Unknown Model"

        tb_message_mgr.add_message(f" Deletion of '{model_name_str}' initiated.")

        log_messages_from_action.append(f" Found active generator: {model_name_str}. Preparing for deletion.")
        print(f"Found active generator: {model_name_str}. Preparing for deletion.")

        try:
            # Step 1: Unload LoRAs
            if hasattr(generator_object_to_delete, 'unload_loras') and callable(generator_object_to_delete.unload_loras):
                print("   - LoRAs: Unloading from transformer...")
                generator_object_to_delete.unload_loras()
            else:
                log_messages_from_action.append("    - LoRAs: No unload method found or not applicable.")

            # Step 2: Operate on the transformer object
            if hasattr(generator_object_to_delete, 'transformer') and generator_object_to_delete.transformer is not None:
                transformer_object_ref = generator_object_to_delete.transformer
                transformer_name_for_log = transformer_object_ref.__class__.__name__ # Still useful for detailed prints/errors
                print(f"   - Transformer ({transformer_name_for_log}): Preparing for memory operations.")

                moved_to_cpu_successfully = False
                if hasattr(transformer_object_ref, 'device') and transformer_object_ref.device != cpu:
                    if hasattr(transformer_object_ref, 'to') and callable(transformer_object_ref.to):
                        try:
                            print(f"   - Transformer ({transformer_name_for_log}): Moving to CPU...")
                            transformer_object_ref.to(cpu)
                            # ℹ️    - Transformer moved to CPU.
                            log_messages_from_action.append("    - Transformer moved to CPU.")
                            print(f"   - Transformer ({transformer_name_for_log}): Moved to CPU.")
                            moved_to_cpu_successfully = True
                        except Exception as e_cpu:
                            error_msg_cpu = f"    - Transformer ({transformer_name_for_log}): Move to CPU FAILED: {e_cpu}"
                            log_messages_from_action.append(error_msg_cpu)
                            print(error_msg_cpu) # Keep detailed error for console
                    else:
                        log_messages_from_action.append(f"    - Transformer ({transformer_name_for_log}): Cannot move to CPU, 'to' method not found.")
                        print(f"   - Transformer ({transformer_name_for_log}): Cannot move to CPU, 'to' method not found.")
                elif hasattr(transformer_object_ref, 'device') and transformer_object_ref.device == cpu:
                     log_messages_from_action.append("    - Transformer already on CPU.")
                     print(f"   - Transformer ({transformer_name_for_log}): Already on CPU.")
                     moved_to_cpu_successfully = True # Considered successful for deletion purposes
                else: # No device attribute or other case
                    log_messages_from_action.append("    - Transformer: Could not determine device or move to CPU.")
                    print(f"   - Transformer ({transformer_name_for_log}): Could not determine device or move to CPU.")

                print(f"   - Transformer ({transformer_name_for_log}): Removing attribute from generator...")
                generator_object_to_delete.transformer = None
                print(f"   - Transformer ({transformer_name_for_log}): Deleting Python reference...")
                del transformer_object_ref
                # ℹ️    - Transformer reference deleted.
                log_messages_from_action.append("    - Transformer reference deleted.")
                print(f"   - Transformer ({transformer_name_for_log}): Reference deleted.")
            else:
                log_messages_from_action.append("    - Transformer: Not found or already unloaded.")
                print("   - Transformer: Not found or already unloaded.")

            # Step 3: Nullify the global reference in studio module
            generator_class_name_for_log = generator_object_to_delete.__class__.__name__
            print(f"   - Model Generator ({generator_class_name_for_log}): Setting global reference to None...")
            setattr(studio_module_instance, 'current_generator', None)
            log_messages_from_action.append("    - 'current_generator' in studio module set to None.")
            print("   - Global 'current_generator' in studio module successfully set to None.")

            # Step 4: Delete our Python reference to the generator object itself
            print(f"   - Model Generator ({generator_class_name_for_log}): Deleting local Python reference...")
            del generator_object_to_delete # Actual deletion
            print(f"   - Model Generator ({generator_class_name_for_log}): Python reference deleted.")

            # Step 5: System Cleanup
            print("   - System: Performing garbage collection and CUDA cache clearing.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_messages_from_action.append("    - GC and CUDA cache cleared.")
            print("   - System: GC and CUDA cache clear completed.")

            log_messages_from_action.append(f"✅ Deletion of '{model_name_str}' completed successfully from toolbox.")
            tb_message_mgr.add_success(f"Deletion of '{model_name_str}' initiated from toolbox.")

        except Exception as e_del:
            error_msg_del = f"Error during deletion process: {e_del}"
            log_messages_from_action.append(f"    - {error_msg_del}") # Add error to detailed log
            print(f"   - {error_msg_del}")
            traceback.print_exc()
            tb_message_mgr.add_error(f"Deletion Error: {e_del}")
    else:
        tb_message_mgr.add_message("ℹ️ No active generator found. Nothing to delete.") # Direct UI message
        print("No active generator found via direct access. Nothing to delete.")

    # Add all collected detailed log messages to the UI
    for msg_item in log_messages_from_action:
        tb_message_mgr.add_message(msg_item)

    return tb_update_messages()

def tb_handle_manually_save_video(temp_video_path_from_component):
    tb_message_mgr.clear()
    if not temp_video_path_from_component:
        tb_message_mgr.add_warning("No video in the output player to save.")
        # Return the current value of the component (which is None or empty)
        return temp_video_path_from_component, tb_update_messages()

    # tb_copy_video_to_permanent_storage now returns the path in the permanent store if successful,
    # or the original temp_video_path if copy failed or source was invalid.
    # The key is that the source temp_video_path_from_component is NOT deleted by this operation.
    copied_path = tb_processor.tb_copy_video_to_permanent_storage(temp_video_path_from_component)

    # We don't need to change the tb_processed_video_output component's value,
    # as the temp file is still there and playable.
    # Just update the messages.
    if copied_path and os.path.abspath(copied_path) != os.path.abspath(temp_video_path_from_component):
        # This means copy was successful and a new file exists at copied_path
        tb_message_mgr.add_success(f"Video successfully copied to permanent storage.") # tb_copy already adds detailed msg
    # If copy failed, tb_copy_video_to_permanent_storage would have added an error message.

    return temp_video_path_from_component, tb_update_messages() # Always return original temp path to keep video in player

def tb_handle_clear_temp_files():
    tb_message_mgr.clear() # Clear messages for this new operation
    success = tb_processor.tb_clear_temporary_files()

    # Regardless of success/failure of deletion, we clear the output video component
    # as its source file (if any) would be gone or the dir structure changed.
    if success:
        tb_message_mgr.add_success("Temporary files cleared.") # General success message
    else:
        tb_message_mgr.add_warning("Issue during temporary file cleanup. Check messages.")

    return None, tb_update_messages() # Clear tb_processed_video_output, update messages

# NEW: Handler for "Use Processed as Input" button's main logic
def tb_handle_use_processed_as_input(processed_video_path):
    if not processed_video_path:
        tb_message_mgr.add_warning("No processed video available to use as input.")
        # tb_input_video_component should not change: gr.update()
        # tb_message_output should be updated: tb_update_messages()
        return gr.update(), tb_update_messages()
    else:
        tb_message_mgr.add_message("Moved processed video to input.")
        # tb_input_video_component gets the new video path
        # tb_message_output should be updated
        return processed_video_path, tb_update_messages()

# NEW: Handler for the .then() part of "Use Processed as Input"
def tb_clear_processed_on_successful_move(original_processed_video_path_from_click_input):
    # This function receives the value of tb_processed_video_output *at the time the button was clicked*.
    if original_processed_video_path_from_click_input:
        # If there was a video that was supposedly moved, clear the processed output and analysis.
        return None, None  # Clears tb_processed_video_output and tb_video_analysis_output
    else:
        # If there was no video to begin with, don't change these components.
        return gr.update(), gr.update()

# This function IS EXPORTED and used by interface.py
def tb_get_formatted_toolbar_stats():
    """Fetches and formats System RAM, NVIDIA VRAM, Temp, and Load for the main toolbar textboxes."""
    vram_full_str = "VRAM: N/A"
    gpu_full_str = "GPU: N/A"
    ram_full_str = "RAM: N/A" 
    
    vram_component_visible = False 
    gpu_component_visible = False

    try:
        ram_info_psutil = psutil.virtual_memory() 
        ram_used_gb = ram_info_psutil.used / (1024**3)
        ram_total_gb = ram_info_psutil.total / (1024**3)
        ram_full_str = f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB ({ram_info_psutil.percent}%)"

        if torch.cuda.is_available(): 
            _, nvidia_metrics, _ = SystemMonitor.get_nvidia_gpu_info()
            if nvidia_metrics:
                vram_used = nvidia_metrics.get('memory_used_gb', 0.0)
                vram_total = nvidia_metrics.get('memory_total_gb', 0.0)
                vram_full_str = f"VRAM: {vram_used:.1f}/{vram_total:.1f}GB"
                vram_component_visible = True
                
                temp = nvidia_metrics.get('temperature', 0.0)
                load = nvidia_metrics.get('utilization', 0.0)
                gpu_full_str = f"GPU: {temp:.0f}°C {load:.0f}%"
                gpu_component_visible = True
        
    except Exception as e:
        print(f"Error getting system stats values for toolbar (from toolbox_app.py): {e}")
        ram_full_str = "RAM: Error"
        is_nvidia_expected = torch.cuda.is_available()
        if is_nvidia_expected:
            vram_full_str = "VRAM: Error"
            gpu_full_str = "GPU: Error"
            vram_component_visible = True 
            gpu_component_visible = True
        else:
            vram_full_str = "VRAM: N/A" 
            gpu_full_str = "GPU: N/A"
            vram_component_visible = False
            gpu_component_visible = False
        
    # If a component is not visible, we can send an empty string or its "N/A" string,
    # as the `visible` flag in gr.update will handle hiding it.
    # For consistency, let's send the appropriate string even if it's about to be hidden.
    
    return (
        gr.update(value=ram_full_str), 
        gr.update(value=vram_full_str, visible=vram_component_visible),
        gr.update(value=gpu_full_str, visible=gpu_component_visible)
    )

# --- Gradio Interface ---

def tb_create_video_toolbox_ui():
    initial_autosave_state = settings_instance.get("toolbox_autosave_enabled", True)
    tb_processor.set_autosave_mode(initial_autosave_state)

    with gr.Column() as tb_toolbox_ui_main_container:
        with gr.Row():
            with gr.Column(scale=1):
                tb_input_video_component = gr.Video(
                    label="Upload Video for post-processing",
                    autoplay=True,
                    elem_classes="video-size",
                    elem_id="toolbox-video-player"
                )
                tb_analyze_button = gr.Button("📊 Analyze Video")

            with gr.Column(scale=1):
                tb_processed_video_output = gr.Video(
                    label="Processed Video",
                    autoplay=True,
                    interactive=False,
                    elem_classes="video-size"
                )
                with gr.Row():
                    tb_use_processed_as_input_btn = gr.Button("🔄 Use Processed as Input", scale=3)
                    tb_manual_save_btn = gr.Button(
                        "💾 Save to Permanent Folder",
                        variant="secondary",
                        scale=3,
                        visible=not initial_autosave_state
                    )
                    tb_autosave_checkbox = gr.Checkbox(
                        label="Autosave",
                        value=initial_autosave_state,
                        scale=1
                    )
    with gr.Accordion("💡 Video Analysis and System Monitor", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                tb_video_analysis_output = gr.Textbox(
                    # label="Video Analysis",
                    container=False,
                    lines=12,
                    show_label=False,
                    interactive=False,
                    elem_classes="analysis-box",
                )
            with gr.Column(scale=1): 
                with gr.Row(scale=4):                
                    tb_monitor_toggle_checkbox = gr.Checkbox(label="Live System Monitoring", scale=1, value=False)
                    tb_delete_studio_transformer_btn = gr.Button("📤 Unload Studio Model", scale=3, variant="stop")
                with gr.Row():
                    tb_resource_monitor_output = gr.Textbox(
                        show_label=False,
                        container=False,
                        lines=9,
                        interactive=False,
                        visible=False, # Initially hidden
                    )

    with gr.Accordion("Operations", open=True):
        with gr.Tabs():
            with gr.TabItem("📈 Upscale Video (ESRGAN)"):
                gr.Markdown("Upscale video resolution using Real-ESRGAN.")
                with gr.Row():
                    with gr.Column(scale=2):
                        tb_upscale_model_select = gr.Dropdown(
                            choices=list(tb_processor.esrgan_upscaler.supported_models.keys()),
                            value=list(tb_processor.esrgan_upscaler.supported_models.keys())[0] if tb_processor.esrgan_upscaler.supported_models else None,
                            label="ESRGAN Model",
                            info="Select the Real-ESRGAN model."
                        )
                        # Determine initial states for model info and sliders
                        default_model_key_init = list(tb_processor.esrgan_upscaler.supported_models.keys())[0] if tb_processor.esrgan_upscaler.supported_models else None
                        initial_model_info_gr_val, initial_slider_gr_val, initial_denoise_gr_val = tb_get_model_info_and_update_scale_slider(default_model_key_init)

                        tb_selected_model_scale_display = gr.Textbox(
                            label="Selected Model Info", 
                            value=initial_model_info_gr_val.get('value', "Info: Select a model."),
                            interactive=False,
                            lines=2 
                        )                 
                        
                        # RE-ADDED and DYNAMIC Upscale Factor Slider
                        tb_upscale_factor_slider = gr.Slider(
                            minimum=initial_slider_gr_val.get('minimum', 1.0), 
                            maximum=initial_slider_gr_val.get('maximum', 2.0), 
                            step=initial_slider_gr_val.get('step', 0.05), 
                            value=initial_slider_gr_val.get('value', 2.0), 
                            label=initial_slider_gr_val.get('label', "Target Upscale Factor"),
                            info="Desired output scale (e.g., 2.0 for 2x). Video is upscaled by the model, then resized if this differs from native scale."
                        )
                    with gr.Column(scale=2):
                        # Tile Size Radio (no changes needed here)
                        tb_upscale_tile_size_radio = gr.Radio(
                            choices=[
                                ("Auto (Recommended)", 0),
                                ("512px", 512),
                                ("256px", 256)
                            ],
                            value=0,
                            label="Tile Size for Upscaling",
                            info="Splits video frames into tiles for processing. 'Auto' (0) disables tiling. Smaller values (e.g., 256, 512) use less VRAM but are slower and can show seams on some videos. Use if 'Auto' causes Out-Of-Memory."
                        )
                        with gr.Row():
                            # Face Enhancement Checkbox
                            tb_upscale_enhance_face_checkbox = gr.Checkbox(
                                label="Enhance Faces (GFPGAN)",
                                value=False,
                                info="Uses GFPGAN to restore (human-like) faces. Increases processing time."
                            )
                        with gr.Row():
                            # NEW Denoise Strength Slider
                            tb_denoise_strength_slider = gr.Slider(
                                label="Denoise Strength (for RealESR-general-x4v3)",
                                minimum=0.0,
                                maximum=1.0, # Max 1.0, where 1.0 means no WDN model contribution
                                step=0.01,
                                value=initial_denoise_gr_val.get('value', 0.5),
                                info="Adjusts denoising for RealESR-general-x4v3. 0.0=Max WDN, <1.0=Blend, 1.0=No WDN.",
                                visible=initial_denoise_gr_val.get('visible', False), # Initial visibility
                                interactive=True
                            )                    
                with gr.Row():
                    tb_upscale_video_btn = gr.Button("🚀 Upscale Video", variant="primary")


            with gr.TabItem("🎨 Video Filters (FFmpeg)"):
                gr.Markdown("Apply visual enhancements using FFmpeg filters. Ensure FFmpeg is fully installed for best results.")

                # Filter Sliders...
                with gr.Row():
                    tb_filter_brightness = gr.Slider(-100, 100, value=TB_DEFAULT_FILTER_SETTINGS["brightness"], step=1, label="Brightness (%)", info="Adjusts overall image brightness.")
                    tb_filter_contrast = gr.Slider(0, 3, value=TB_DEFAULT_FILTER_SETTINGS["contrast"], step=0.05, label="Contrast (Linear)", info="Increases/decreases difference between light/dark areas.")
                with gr.Row():
                    tb_filter_saturation = gr.Slider(0, 3, value=TB_DEFAULT_FILTER_SETTINGS["saturation"], step=0.05, label="Saturation", info="Adjusts color intensity. 0=grayscale, 1=original.")
                    tb_filter_temperature = gr.Slider(-100, 100, value=TB_DEFAULT_FILTER_SETTINGS["temperature"], step=1, label="Color Temperature Adjust", info="Shifts colors towards orange (warm) or blue (cool).")
                with gr.Row():
                    tb_filter_sharpen = gr.Slider(0, 5, value=TB_DEFAULT_FILTER_SETTINGS["sharpen"], step=0.1, label="Sharpen Strength", info="Enhances edge details. Use sparingly.")
                    tb_filter_blur = gr.Slider(0, 5, value=TB_DEFAULT_FILTER_SETTINGS["blur"], step=0.1, label="Blur Strength", info="Softens the image.")
                with gr.Row():
                    tb_filter_denoise = gr.Slider(0, 10, value=TB_DEFAULT_FILTER_SETTINGS["denoise"], step=0.1, label="Denoise Strength", info="Reduces video noise/grain.")
                    tb_filter_vignette = gr.Slider(0, 100, value=TB_DEFAULT_FILTER_SETTINGS["vignette"], step=1, label="Vignette Strength (%)", info="Darkens corners, drawing focus to center.")
                with gr.Row():
                    tb_filter_s_curve_contrast = gr.Slider(0, 100, value=TB_DEFAULT_FILTER_SETTINGS["s_curve_contrast"], step=1, label="S-Curve Contrast", info="Non-linear contrast, boosting highlights/shadows subtly.")
                    tb_filter_film_grain_strength = gr.Slider(0, 50, value=TB_DEFAULT_FILTER_SETTINGS["film_grain_strength"], step=1, label="Film Grain Strength", info="Adds artificial film grain.")
                
                tb_apply_filters_btn = gr.Button("✨ Apply Filters to Video", variant="primary")
                
                with gr.Row(equal_height=False): # Use equal_height=False if components have different natural heights
                    with gr.Column(scale=2):
                        with gr.Row():                        
                            preset_choices = list(tb_filter_presets_data.keys()) if tb_filter_presets_data else ["none"]
                            if "none" not in preset_choices and preset_choices:
                                preset_choices.insert(0,"none")
                            elif not preset_choices:
                                preset_choices = ["none"]

                            tb_filter_preset_select = gr.Dropdown(
                                choices=preset_choices,
                                value="none",
                                label="Load Preset",
                                scale=2
                            )
                            tb_new_preset_name_input = gr.Textbox(
                                label="Preset Name (for saving/editing)",
                                placeholder="Select preset or type new name...",
                                scale=2
                            )
                    with gr.Column(scale=1):          
                        with gr.Row():
                            tb_save_preset_btn = gr.Button(
                                "💾 Save/Update",
                                variant="primary",
                                scale=1
                            )
                            tb_delete_preset_btn = gr.Button(
                                "🗑️ Delete",
                                variant="stop",
                                scale=1
                            )
                        with gr.Row():
                            tb_reset_filters_btn = gr.Button("🔄 Reset All Sliders to 'None' Preset")    
                            
            with gr.TabItem("🎞️ Frame Adjust (Speed & Interpolation)"):
                gr.Markdown("Adjust video speed and interpolate frames using RIFE AI.")
                tb_process_fps_mode = gr.Radio(
                    choices=["No Interpolation", "2x RIFE Interpolation"],
                    value="No Interpolation",
                    label="RIFE Frame Interpolation",
                    info="Select '2x RIFE Interpolation' to double the frame rate, creating smoother motion."
                )
                tb_process_speed_factor = gr.Slider(
                    minimum=0.25, maximum=4.0, step=0.05, value=1.0, label="Adjust Video Speed Factor",
                    info="Values < 1.0 slow down the video, values > 1.0 speed it up. Affects video and audio (if present & FFmpeg is available)."
                )
                tb_process_frames_btn = gr.Button("🚀 Process Frames", variant="primary")

            with gr.TabItem("🔄 Video Loop"):
                gr.Markdown("Create looped or ping-pong versions of the video.")
                tb_loop_type_select = gr.Radio(
                    choices=["loop", "ping-pong"], value="loop", label="Loop Type"
                )
                tb_num_loops_slider = gr.Slider(
                    minimum=1, maximum=10, step=1, value=1, label="Number of Loops/Repeats",
                    info="The video will play its original content, then repeat this many additional times. E.g., 1 loop = 2 total plays of the segment."
                )
                tb_create_loop_btn = gr.Button("🔁 Create Loop", variant="primary")

            with gr.TabItem("🖼️ Frames I/O"):
                with gr.Row():
                    with gr.Column(): # Column for extraction
                        gr.Markdown("### Extract Frames from Video")
                        gr.Markdown("Extract frames from the **uploaded video (top-left)** as images.")
                        tb_extract_rate_slider = gr.Number(
                            label="Extract Every Nth Frame", value=1, minimum=1, step=1,
                            info="1 = all frames. N = 1st, (N+1)th... (i.e., frame 0, frame N, frame 2N, etc.)"
                        )
                        tb_extract_frames_btn = gr.Button("🔨 Extract Frames", variant="primary")

                    with gr.Column(): # Column for reassembly
                        gr.Markdown("### Reassemble Frames to Video")

                        tb_extracted_folders_dropdown = gr.Dropdown(
                            label="Select Previously Extracted Folder",
                            info="Select a folder from your 'extracted_frames' directory. This takes precedence over uploaded files below."
                        )
                        with gr.Row():
                            tb_refresh_extracted_folders_btn = gr.Button("🔄 Refresh List")
                            tb_clear_selected_folder_btn = gr.Button(
                                "🗑️ Clear Selected Folder", variant="stop", interactive=False
                            )

                        gr.Markdown("Alternatively, drag individual frames or Click to upload a folder containing frame images:")
                        tb_reassemble_frames_input_files = gr.File(
                            label="Upload Frame Images Folder (or individual image files)",
                            file_count="directory",
                            # info="If a folder is selected in the dropdown above, this upload will be ignored."
                        )
                        tb_reassemble_output_fps = gr.Number(
                            label="Output Video FPS", value=30, minimum=1, step=1
                        )
                        tb_reassemble_video_name_input = gr.Textbox(
                            label="Output Video Name (optional, .mp4 added automatically)"
                        )
                        tb_reassemble_frames_btn = gr.Button("🧩 Reassemble Video", variant="primary")

        # NEW: Toolbox Guide & Tips Accordion
        with gr.Accordion("💡 Post-processing Guide & Tips", open=False): # Initially closed, title updated
            gr.Markdown(value="""### This set of tools is designed to help you post-process your generated videos.


**Core Workflow:**
*   **Input & Output:** Most operations use the video in the **'Upload Video' ⬅️ (top-left)** player as their input.
*   Processed videos will appear in the **'Processed Video' ➡️ (top-right)** player.
*   **Analysis First:** It's often helpful to upload a video and click **'📊 Analyze Video'** first. This provides details like resolution, frame rate, and duration, which can inform your choices for processing.


**Chaining Operations (Applying Multiple Effects):**
*   To apply several effects one after another (e.g., first upscale, then change speed, then apply filters, etc):
    1.  Perform the first operation (e.g., apply upscale).
    2.  Once the processed video appears, click the **'🔄 Use Processed as Input'** button. This moves the result from the 'Processed Video' player to the 'Upload Video' player.
    3.  Now, the output of the first operation is ready to be the input for your next operation.
    4.  Repeat as needed.


**Saving Your Work:**
*   By default, all processed videos are auto-saved to the 'saved_videos' folder.

*   **To save outputs manually:**
    *   Disable the **'Autosave' checkbox**. When unchecked, all processed videos will save to the 'temp_processing' folder.
    *   Use the **'💾 Save to Permanent Folder'** button (visible if Autosave is off). This saves the current video from the 'Processed Video' player to the 'saved_videos' folder.

*   You can open the permanent output folder using the **'📁 Open Output Folder'** button.
*   You can empty the 'temp_processing' folder by pressing the **`🗑️ Clear Temporary Files`** button


**Working with Video Filters & Presets:**
*   Adjust visual effects like brightness, contrast, and color using the **Filter Sliders**.
*   **Load Preset Dropdown:** Select a pre-defined or saved custom look.
*   **Preset Name Textbox:**
    *   Shows the loaded preset's name.
    *   Type a new name here to save current slider settings as a new preset.
*   **💾 Save/Update Button:** Saves the current slider settings using the name in the 'Preset Name' textbox. Adds new presets to the dropdown or updates existing ones.
*   **🗑️ Delete Button:** Deletes the preset whose name is currently in the 'Preset Name' textbox from your saved presets.
*   **🔄 Reset All Sliders Button:** Clears all filter effects, sets sliders to default ("none" preset values).
*   **✨ Apply Filters to Video Button:** Processes the input video with the current filter slider settings.


**Understanding Frames I/O:**
*   **Extracting:** You can extract frames from the input video. These are saved into a new subfolder within `postprocessed_output/toolbox_frames/extracted_frames/`.
        **This defaults to extracting _every_ frame. If you're after fewer frames, change the '1' to a higher number - i.e. `5` will extract every 5th frame (~30 frames from a typical 5s FramePack video)**
*   **Reassembling:**
    *   **Dropdown:** You can select one of these previously extracted folders from the **'Select Previously Extracted Folder'** dropdown menu.
    *   **Upload:** Alternatively, you can upload your own folder of frames or individual frame images using the **'Upload Frame Images Folder'** component.
    *   **Precedence:** If a folder is selected in the dropdown, any files/folder provided to the 'Upload Frame Images Folder' component will be **ignored**. The dropdown selection takes priority.
    *   **Refresh:** Use '🔄 Refresh List' on first use to populate the dropdown and/or after an extraction or if you've manually added/removed folders in the `extracted_frames` directory.


**Unloading the Main Studio Model:**
*   The **'📤 Unload Studio Model'** button attempts to remove the main video generation model from your computer's memory (VRAM).
*   **Why use this?**
    *   To free up VRAM if you plan to run memory-heavy tasks in this toolbox (like '📈 Upscale Video') and are not actively using the main video generation tab.

*   The main Studio interface will automatically reload this model when you start a new generation task there.


**Important Dependency: FFmpeg**
*   Audio functions in this toolbox and reliable video analysis depend on **FFmpeg**.
*   The application will use a "lite" version of FFmpeg bundled with `imageio-ffmpeg`.
*   **For best results and full functionality (especially audio handling):** It's highly recommended to have a full, up-to-date version of FFmpeg (with `ffprobe`) installed on your system and accessible in your system's PATH.
*   If FFmpeg or ffprobe is not found or not fully functional, audio handling operations might be limited and produce silent video. Check the 'Console Messages' for status.
*   Adding ffprobe and/or full ffmpeg is likely in a future update!


**Check Console Messages:**
*   The **'Console Messages' box** at the bottom of the tab provides important feedback, status updates, warnings, and error messages for all operations. Always check it if something doesn't seem right!
            """)

        with gr.Row():
            tb_message_output = gr.Textbox(label="Console Messages", lines=10, interactive=False, elem_classes="message-box", value=tb_update_messages)
        with gr.Row():
            tb_open_folder_button = gr.Button("📁 Open Output Folder", scale=4)
            tb_clear_temp_button = gr.Button("🗑️ Clear Temporary Files", variant="stop", scale=1)

        # --- Event Handlers ---
        
        _ORDERED_FILTER_SLIDERS_ = [
            tb_filter_brightness,
            tb_filter_contrast,
            tb_filter_saturation,
            tb_filter_temperature,
            tb_filter_sharpen,
            tb_filter_blur,
            tb_filter_denoise,
            tb_filter_vignette, 
            tb_filter_s_curve_contrast,
            tb_filter_film_grain_strength
        ]     
        
        tb_input_video_component.upload(fn=lambda: (tb_message_mgr.clear() or tb_update_messages(), None), outputs=[tb_message_output, tb_video_analysis_output])
        tb_input_video_component.clear(fn=lambda: (tb_message_mgr.clear() or tb_update_messages(), None, None), outputs=[tb_message_output, tb_video_analysis_output, tb_processed_video_output])

        tb_analyze_button.click(
            fn=tb_handle_analyze_video,
            inputs=[tb_input_video_component],
            outputs=[tb_message_output, tb_video_analysis_output]
        )

        tb_process_frames_btn.click(
            fn=tb_handle_process_frames,
            inputs=[tb_input_video_component, tb_process_fps_mode, tb_process_speed_factor],
            outputs=[tb_processed_video_output, tb_message_output]
        )

        tb_create_loop_btn.click(
            fn=tb_handle_create_loop,
            inputs=[tb_input_video_component, tb_loop_type_select, tb_num_loops_slider],
            outputs=[tb_processed_video_output, tb_message_output]
        )

        # When a preset is selected from the dropdown, update sliders AND the preset name textbox
        tb_filter_preset_select.change(
            fn=lambda preset_name_from_dropdown: (preset_name_from_dropdown, *tb_update_filter_sliders_from_preset(preset_name_from_dropdown)),
            inputs=[tb_filter_preset_select],
            outputs=[tb_new_preset_name_input] + _ORDERED_FILTER_SLIDERS_
        )

        tb_apply_filters_btn.click(
            fn=tb_handle_apply_filters,
            inputs=[tb_input_video_component] + _ORDERED_FILTER_SLIDERS_,
            outputs=[tb_processed_video_output, tb_message_output]
        )

        tb_save_preset_btn.click(
            fn=tb_handle_save_user_preset,
            inputs=[tb_new_preset_name_input] + _ORDERED_FILTER_SLIDERS_,
            outputs=[tb_filter_preset_select, tb_message_output, tb_new_preset_name_input]
        )

        tb_delete_preset_btn.click(
            fn=tb_handle_delete_user_preset,
            inputs=[tb_new_preset_name_input],
            outputs=[tb_filter_preset_select, tb_message_output, tb_new_preset_name_input] + _ORDERED_FILTER_SLIDERS_
        )

        tb_reset_filters_btn.click(
            fn=tb_handle_reset_all_filters,
            inputs=None,
            outputs=[
                tb_filter_preset_select,      # "none"
                tb_new_preset_name_input,     # ""
                *_ORDERED_FILTER_SLIDERS_,    # *none_settings_values
                tb_message_output             # tb_update_messages()
            ]
        )

        tb_use_processed_as_input_btn.click(
            fn=tb_handle_use_processed_as_input,
            inputs=[tb_processed_video_output],
            outputs=[tb_input_video_component, tb_message_output]
        ).then(
            fn=tb_clear_processed_on_successful_move,
            inputs=[tb_processed_video_output], # This input is the state of tb_processed_video_output *before* the .click's first fn ran
            outputs=[tb_processed_video_output, tb_video_analysis_output]
        )

        # UPDATED event handler for tb_upscale_video_btn
        tb_upscale_video_btn.click(
            fn=tb_handle_upscale_video,
            inputs=[
                tb_input_video_component, 
                tb_upscale_model_select,      
                tb_upscale_factor_slider,
                tb_upscale_tile_size_radio,
                tb_upscale_enhance_face_checkbox,
                tb_denoise_strength_slider # NEW input
            ], 
            outputs=[tb_processed_video_output, tb_message_output]
        )
        
        # MODIFIED Event handler for model selection changing
        tb_upscale_model_select.change(
            fn=tb_get_model_info_and_update_scale_slider,
            inputs=[tb_upscale_model_select],
            outputs=[
                tb_selected_model_scale_display,
                tb_upscale_factor_slider,
                tb_denoise_strength_slider # NEW output to control visibility/value
            ]
        )
        
        # Frames I/O Event Handlers
        tb_extract_frames_btn.click(
            fn=tb_handle_extract_frames,
            inputs=[tb_input_video_component, tb_extract_rate_slider],
            outputs=[tb_message_output]
        ).then( # Automatically refresh folder list after extraction
            fn=tb_handle_refresh_extracted_folders,
            inputs=None,
            outputs=[tb_extracted_folders_dropdown, tb_message_output, tb_clear_selected_folder_btn]
        )

        tb_refresh_extracted_folders_btn.click(
            fn=tb_handle_refresh_extracted_folders,
            inputs=None,
            outputs=[tb_extracted_folders_dropdown, tb_message_output, tb_clear_selected_folder_btn]
        )

        tb_extracted_folders_dropdown.change(
            fn=lambda selection: gr.update(interactive=bool(selection)), # Enable/disable clear button
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_clear_selected_folder_btn]
        )

        tb_clear_selected_folder_btn.click(
            fn=tb_handle_clear_selected_folder,
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_message_output, tb_extracted_folders_dropdown] # Dropdown updates from handler
        ).then( # After clearing, re-evaluate clear button interactivity
            fn=lambda selection: gr.update(interactive=bool(selection)),
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_clear_selected_folder_btn]
        )

        tb_reassemble_frames_btn.click(
            fn=tb_handle_reassemble_frames,
            inputs=[
                tb_extracted_folders_dropdown,
                tb_reassemble_frames_input_files,
                tb_reassemble_output_fps,
                tb_reassemble_video_name_input # New input for name
            ],
            outputs=[tb_processed_video_output, tb_message_output]
        )

        # Clear messages on new uploads to gr.File
        tb_reassemble_frames_input_files.upload(fn=lambda: tb_message_mgr.clear() or tb_update_messages(), outputs=tb_message_output)
        tb_reassemble_frames_input_files.clear(fn=lambda: tb_message_mgr.clear() or tb_update_messages(), outputs=tb_message_output)
        tb_open_folder_button.click(
            fn=lambda: tb_processor.tb_open_output_folder() or tb_update_messages(),
            outputs=[tb_message_output]
        )

        # This will show/hide the monitor output textbox.
        tb_monitor_toggle_checkbox.change(
            fn=lambda is_enabled: gr.update(visible=is_enabled),
            inputs=[tb_monitor_toggle_checkbox],
            outputs=[tb_resource_monitor_output]
        )

        tb_monitor_timer = gr.Timer(2, active=True)
        tb_monitor_timer.tick(
            fn=tb_handle_update_monitor,
            inputs=[tb_monitor_toggle_checkbox], # Pass the checkbox state to the handler
            outputs=[tb_resource_monitor_output],
        )
        # ADDED: Event handler for the new unload button
        tb_delete_studio_transformer_btn.click(
            fn=tb_handle_delete_studio_transformer,
            inputs=[], # No Gradio inputs for this action
            outputs=[tb_message_output] # Update the toolbox's message console
        )
        # New event handler for the manual save button
        tb_manual_save_btn.click(
            fn=tb_handle_manually_save_video,
            inputs=[tb_processed_video_output],
            outputs=[tb_processed_video_output, tb_message_output]
        )
        # Handler for the autosave checkbox
        def tb_handle_autosave_toggle(autosave_is_on_ui_value):
            tb_processor.set_autosave_mode(autosave_is_on_ui_value) # Update the processor's internal switch
            return {
                tb_manual_save_btn: gr.update(visible=not autosave_is_on_ui_value),
                tb_message_output: gr.update(value=tb_update_messages()) # Fetch new messages
            }

        tb_autosave_checkbox.change(
            fn=tb_handle_autosave_toggle,
            inputs=[tb_autosave_checkbox],
            # Outputs list refined to only what the handler updates
            outputs=[tb_manual_save_btn, tb_message_output]
        )
        # Event handler for the new clear temporary files button
        tb_clear_temp_button.click(
            fn=tb_handle_clear_temp_files,
            inputs=None, # No inputs needed
            outputs=[tb_processed_video_output, tb_message_output] # Clear video output, update messages
        )
    # MODIFIED RETURN SIGNATURE: Only return what's needed by interface.py
    return tb_toolbox_ui_main_container, tb_input_video_component
