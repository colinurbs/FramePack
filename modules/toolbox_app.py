import gradio as gr
import os
import sys
import torch
import devicetorch
import traceback
import gc
import psutil
import json # for preset loading/saving
import imageio # Added for reading frame dimensions

# --- Standalone Startup & Path Fix ---
# This block allows the script to be run directly, without needing the main app.
# It adjusts the Python path to include the parent 'app' directory,
# so that imports like 'from modules.settings import Settings' work correctly.
if __name__ == '__main__':
    # Get the absolute path of the 'modules' directory
    modules_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the absolute path of the parent 'app' directory
    project_root = os.path.dirname(modules_dir)
    # Add the project root to the Python path if it's not already there
    if project_root not in sys.path:
        print(f"--- Running Toolbox in Standalone Mode ---")
        print(f"Adding project root to sys.path: {project_root}")
        sys.path.insert(0, project_root)
# --- End Standalone Startup & Path Fix ---

# patch fix for basicsr
from torchvision.transforms.functional import rgb_to_grayscale
import types
functional_tensor_mod = types.ModuleType('functional_tensor')
functional_tensor_mod.rgb_to_grayscale = rgb_to_grayscale
sys.modules.setdefault('torchvision.transforms.functional_tensor', functional_tensor_mod)

# --- additional instance of this for standalone mode ---
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
            

from modules.toolbox.toolbox_processor import VideoProcessor
from modules.toolbox.message_manager import MessageManager
from modules.toolbox.system_monitor import SystemMonitor
from modules.settings import Settings
from modules.toolbox.setup_ffmpeg import setup_ffmpeg

try:
    from diffusers_helper.memory import cpu
except ImportError:
    print("WARNING: Could not import cpu from diffusers_helper.memory. Falling back to torch.device('cpu')")
    cpu = torch.device('cpu')


# Check if FFmpeg is set up, if not, run the setup
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the correct path to the target bin directory.
bin_dir = os.path.join(script_dir, 'toolbox', 'bin')

ffmpeg_exe_name = 'ffmpeg.exe' if sys.platform == "win32" else 'ffmpeg'
ffmpeg_full_path = os.path.join(bin_dir, ffmpeg_exe_name)

# Check if the executable exists at the correct location.
if not os.path.exists(ffmpeg_full_path):
    print(f"Bundled FFmpeg not found in '{bin_dir}'. Running one-time setup...")
    setup_ffmpeg()


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
    # Return a third value to control the accordion's 'open' state
    return tb_update_messages(), analysis, gr.update(open=True)

def tb_handle_process_frames(video_path, fps_mode, speed_factor, progress=gr.Progress()):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_process_frames(video_path, fps_mode, speed_factor, progress)
    return output_video, tb_update_messages()

def tb_handle_create_loop(video_path, loop_type, num_loops, progress=gr.Progress()):
    tb_message_mgr.clear()
    output_video = tb_processor.tb_create_loop(video_path, loop_type, num_loops, progress)
    return output_video, tb_update_messages()

def tb_update_filter_sliders_from_preset(preset_name):
    preset_settings = tb_filter_presets_data.get(preset_name)
    if not preset_settings:
        tb_message_mgr.add_warning(f"Preset '{preset_name}' not found. Using 'none' settings.")
        preset_settings = tb_filter_presets_data.get("none", TB_DEFAULT_FILTER_SETTINGS.copy())

    final_settings = TB_DEFAULT_FILTER_SETTINGS.copy()
    final_settings.update(preset_settings)

    ordered_values = []
    for key in TB_DEFAULT_FILTER_SETTINGS.keys():
        ordered_values.append(final_settings.get(key, TB_DEFAULT_FILTER_SETTINGS[key]))

    return tuple(ordered_values)

def tb_handle_reset_all_filters():
    tb_message_mgr.add_message("Filter sliders reset to default 'none' values.")
    none_settings_values = tb_update_filter_sliders_from_preset("none")
    return "none", "", *none_settings_values, tb_update_messages()

def tb_handle_save_user_preset(new_preset_name_str, *slider_values):
    global tb_filter_presets_data; tb_message_mgr.clear()
    if not new_preset_name_str or not new_preset_name_str.strip():
        tb_message_mgr.add_warning("Preset name cannot be empty."); return gr.update(), tb_update_messages(), gr.update()

    clean_preset_name = new_preset_name_str.strip()

    if clean_preset_name.lower() == "none":
        tb_message_mgr.add_warning("'none' is a protected preset and cannot be overwritten.")
        return gr.update(), tb_update_messages(), gr.update(value="") # Clear input box

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

# --- Workflow Presets Handling ---
TB_WORKFLOW_PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toolbox", "data", "workflow_presets.json")
tb_workflow_presets_data = {} # Will be populated by _initialize_workflow_presets

# This helper function creates a dictionary of all default parameter values
def _get_default_workflow_params():
    # Gets default values from filter settings and adds other op defaults
    params = TB_DEFAULT_FILTER_SETTINGS.copy()
    params.update({
        "upscale_model": list(tb_processor.esrgan_upscaler.supported_models.keys())[0] if tb_processor.esrgan_upscaler.supported_models else None,
        "upscale_factor": 2.0, # A sensible default
        "tile_size": 0,
        "enhance_face": False,
        "denoise_strength": 0.5,
        "fps_mode": "No Interpolation",
        "speed_factor": 1.0,
        "loop_type": "loop",
        "num_loops": 1,
        "export_format": "MP4",
        "export_quality": 85,
        "export_max_width": 1024,
    })
    return params

def _initialize_workflow_presets():
    global tb_workflow_presets_data
    # The 'None' preset stores default values for all controls and no active steps
    default_workflow_map = {
        "None": {
            "active_steps": [],
            "params": _get_default_workflow_params()
        }
    }
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(TB_WORKFLOW_PRESETS_FILE), exist_ok=True)
        if not os.path.exists(TB_WORKFLOW_PRESETS_FILE):
            tb_message_mgr.add_message(f"Workflow presets file not found. Creating with a default 'None' preset: {TB_WORKFLOW_PRESETS_FILE}", "INFO")
            with open(TB_WORKFLOW_PRESETS_FILE, 'w') as f:
                json.dump(default_workflow_map, f, indent=4)
            tb_workflow_presets_data = default_workflow_map
        else:
            with open(TB_WORKFLOW_PRESETS_FILE, 'r') as f:
                tb_workflow_presets_data = json.load(f)
            # Ensure "None" preset always exists and is up-to-date
            tb_workflow_presets_data["None"] = default_workflow_map["None"]
            tb_message_mgr.add_message(f"Workflow presets loaded from {TB_WORKFLOW_PRESETS_FILE}.", "INFO")
    except Exception as e:
        tb_message_mgr.add_error(f"Error with workflow presets file {TB_WORKFLOW_PRESETS_FILE}: {e}. Using in-memory defaults.")
        tb_workflow_presets_data = default_workflow_map

_initialize_workflow_presets() # Call once when the script module is loaded

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
    frames_source_folder,
    output_fps,
    output_video_name,
    progress=gr.Progress()
):
    tb_message_mgr.clear()
    tb_message_mgr.add_message("Preparing to reassemble from Frames Studio...")
    if not frames_source_folder:
        tb_message_mgr.add_warning("No source folder selected in the Frames Studio dropdown.")
        return None, tb_update_messages()

    frames_path_to_use = os.path.join(tb_processor.extracted_frames_target_path, frames_source_folder)
    source_description = f"Frames Studio folder '{frames_source_folder}'"

    if not os.path.isdir(frames_path_to_use):
        tb_message_mgr.add_error(f"Selected folder not found: {frames_path_to_use}")
        return None, tb_update_messages()

    tb_message_mgr.add_message(f"Attempting to reassemble frames from {source_description}.")
    output_video = tb_processor.tb_reassemble_frames_to_video(
        frames_path_to_use,
        output_fps,
        output_base_name_override=output_video_name,
        progress=progress
    )
    return output_video, tb_update_messages()

def tb_handle_extract_frames(video_path, extraction_rate, progress=gr.Progress()):
    tb_message_mgr.clear()
    tb_processor.tb_extract_frames(video_path, int(extraction_rate), progress)
    return tb_update_messages()

def tb_handle_refresh_extracted_folders():
    folders = tb_processor.tb_get_extracted_frame_folders()
    clear_btn_update = gr.update(interactive=False)
    # When refreshing, clear the gallery and info box
    return gr.update(choices=folders, value=None), tb_update_messages(), clear_btn_update, None, "Select a folder and click 'Load'."

def tb_handle_clear_selected_folder(selected_folder_to_delete):
    tb_message_mgr.clear()
    if not selected_folder_to_delete:
        tb_message_mgr.add_warning("No folder selected from the dropdown to delete.")
        return tb_update_messages(), gr.update()

    success = tb_processor.tb_delete_extracted_frames_folder(selected_folder_to_delete)
    updated_folders = tb_processor.tb_get_extracted_frame_folders()
    return tb_update_messages(), gr.update(choices=updated_folders, value=None)

def tb_handle_load_frames_to_studio(selected_folder):
    tb_message_mgr.clear()
    if not selected_folder:
        tb_message_mgr.add_warning("No folder selected to load into the studio.")
        return tb_update_messages(), None, "Select a folder and click 'Load'."

    frame_files_list = tb_processor.tb_get_frames_from_folder(selected_folder)

    if not frame_files_list:
        tb_message_mgr.add_warning(f"No image files found in '{selected_folder}'.")
        return tb_update_messages(), None, "No frames found in this folder."

    tb_message_mgr.add_success(f"Loaded {len(frame_files_list)} frames from '{selected_folder}' into the studio.")
    return tb_update_messages(), frame_files_list, "Select a frame from the gallery."

def tb_handle_frame_select(evt: gr.SelectData):
    """Handles frame selection in the gallery, providing more detailed info."""
    # The gallery's select event (evt.value) is a dictionary like: 
    # {'image': {'path': '...', 'url': '...'}, 'caption': None}
    if evt.value and 'image' in evt.value and 'path' in evt.value['image']:
        # CORRECT WAY to access the path string
        selected_image_path = evt.value['image']['path'] 
        
        # Now 'selected_image_path' is a string, so os.path.basename will work correctly.
        filename = os.path.basename(selected_image_path)
        
        info_text = f"File: {filename}"
        try:
            # Add image dimensions to the info box for better context
            img = imageio.imread(selected_image_path)
            h, w, *_ = img.shape
            info_text += f"\nDimensions: {w}x{h}"
        except Exception as e:
            tb_message_mgr.add_warning(f"Could not read dimensions for {filename}: {e}")

        # 1. Return the info string to the gr.Textbox component.
        # 2 & 3. Return updates to enable the buttons.
        return info_text, gr.update(interactive=True), gr.update(interactive=True)
    else:
        # This part handles deselection or malformed event data
        return "Select a frame...", gr.update(interactive=False), gr.update(interactive=False)

def _get_frame_path_from_ui(selected_folder, frame_info_str):
    """Helper to safely parse UI components to get a full file path."""
    if not selected_folder or not frame_info_str:
        return None, "Missing folder or frame selection."
    
    # Extract filename from the first line of the info string "File: frame_0000.png"
    first_line = frame_info_str.splitlines()[0]
    if not first_line.startswith("File: "):
        return None, "Invalid frame info format."
        
    filename_to_process = first_line.replace("File: ", "").strip()
    full_path = os.path.join(tb_processor.extracted_frames_target_path, selected_folder, filename_to_process)
    return full_path, None


def tb_handle_delete_selected_frame(selected_folder, frame_info_str):
    """
    Handler for the delete button. Calls backend and returns the status message.
    The message log is NOT cleared here.
    """
    # NO tb_message_mgr.clear()
    
    full_path, error = _get_frame_path_from_ui(selected_folder, frame_info_str)
    if error:
        # We can still add a main log message for critical errors if we want
        tb_message_mgr.add_error(error)
        return error, tb_update_messages() # Return error to both places

    # Call the processor, which now returns a message string
    status_message = tb_processor.tb_delete_single_frame(full_path)
    
    # Return the status message for the info box, and any other messages for the main console
    return status_message, tb_update_messages()

def tb_handle_save_selected_frame(selected_folder, frame_info_str):
    """Handler for the save button. Calls the backend processor."""
    tb_message_mgr.clear()
    full_path, error = _get_frame_path_from_ui(selected_folder, frame_info_str)
    if error:
        tb_message_mgr.add_error(error)
        return error, tb_update_messages()
    
    status_message = tb_processor.tb_save_single_frame(full_path)
    # The message is generated inside the processor, we just need to return it.
    return status_message, tb_update_messages()

# --- End of Frames Studio Handlers ---

def tb_handle_input_tab_change(evt: gr.SelectData):
    """
    Handles the user switching between the 'Single Video' and 'Batch Video' input tabs.
    Shows or hides all batch-related UI components based on the selected tab.
    """
    # The .select() event for gr.Tabs passes an event data object
    # where evt.index is the index of the selected tab.
    # 'Single Video' is at index 0, 'Batch Video' is at index 1.
    is_batch_mode_active = (evt.index == 1)
    
    visibility_update = gr.update(visible=is_batch_mode_active)
    
    # Return an update for each component we want to toggle.
    # The order must match the 'outputs' list in the .select() event listener.
    return (
        visibility_update, # tb_start_batch_btn
        visibility_update, # tb_batch_include_upscale
        visibility_update, # tb_batch_include_filters
        visibility_update, # tb_batch_include_frame_adjust
        visibility_update  # tb_batch_include_loop
    )

# --- START: New Workflow Preset Handlers ---

def tb_handle_save_workflow_preset(preset_name, active_steps, *params):
    global tb_workflow_presets_data
    tb_message_mgr.clear()

    if not preset_name or not preset_name.strip():
        tb_message_mgr.add_warning("Workflow Preset name cannot be empty.")
        return gr.update(), gr.update(value=preset_name), tb_update_messages()

    clean_name = preset_name.strip()
    param_keys = [
        # Upscale
        "upscale_model", "upscale_factor", "tile_size",
        "enhance_face", "denoise_strength",
        # Frame Adjust
        "fps_mode", "speed_factor",
        # Loop
        "loop_type", "num_loops",
        # Filters (using the ordered keys from the constant)
        *list(TB_DEFAULT_FILTER_SETTINGS.keys()),
        # Export
        "export_format", "export_quality", "export_max_width"
    ]
    
    # Pack the parameters into a dictionary
    params_dict = dict(zip(param_keys, params))

    new_preset_data = {
        "active_steps": active_steps,
        "params": params_dict
    }

    preset_existed = clean_name in tb_workflow_presets_data
    tb_workflow_presets_data[clean_name] = new_preset_data

    try:
        with open(TB_WORKFLOW_PRESETS_FILE, 'w') as f:
            json.dump(tb_workflow_presets_data, f, indent=4)

        tb_message_mgr.add_success(f"Workflow Preset '{clean_name}' {'updated' if preset_existed else 'saved'} successfully!")
        
        # Update dropdown choices
        updated_choices = sorted([k for k in tb_workflow_presets_data.keys() if k != "None"])
        updated_choices.insert(0, "None")
        
        return gr.update(choices=updated_choices, value=clean_name), "", tb_update_messages()

    except Exception as e:
        tb_message_mgr.add_error(f"Error saving workflow preset '{clean_name}': {e}")
        # Revert to last known good state
        _initialize_workflow_presets()
        return gr.update(), gr.update(value=preset_name), tb_update_messages()

def tb_handle_load_workflow_preset(preset_name):
    tb_message_mgr.clear()
    preset_data = tb_workflow_presets_data.get(preset_name)

    if not preset_data:
        tb_message_mgr.add_warning(f"Workflow preset '{preset_name}' not found. Loading 'None' state.")
        preset_data = tb_workflow_presets_data.get("None")

    # Get the default parameter structure to ensure all keys are present
    final_params = _get_default_workflow_params()
    # Update with the loaded preset's parameters
    final_params.update(preset_data.get("params", {}))
    
    active_steps = preset_data.get("active_steps", [])

    # The order of values returned MUST match the order of components in the event handler's output list
    ordered_values = [
        # Checkbox
        active_steps,
        # Upscale
        final_params["upscale_model"], final_params["upscale_factor"], final_params["tile_size"],
        final_params["enhance_face"], final_params["denoise_strength"],
        # Frame Adjust
        final_params["fps_mode"], final_params["speed_factor"],
        # Loop
        final_params["loop_type"], final_params["num_loops"],
        # Filters (must be in the same order as _ORDERED_FILTER_SLIDERS_)
        final_params["brightness"], final_params["contrast"], final_params["saturation"],
        final_params["temperature"], final_params["sharpen"], final_params["blur"],
        final_params["denoise"], final_params["vignette"], final_params["s_curve_contrast"],
        final_params["film_grain_strength"],
        # Export
        final_params["export_format"], final_params["export_quality"], final_params["export_max_width"]
    ]

    tb_message_mgr.add_message(f"Loaded workflow preset: '{preset_name}'")
    # Also return the preset name to the input box, and the updated messages
    return preset_name, *ordered_values, tb_update_messages()

def tb_handle_delete_workflow_preset(preset_name):
    global tb_workflow_presets_data
    tb_message_mgr.clear()

    if not preset_name or not preset_name.strip():
        tb_message_mgr.add_warning("No workflow preset name provided to delete.")
        # Return updates for all components, but change nothing
        return gr.update(), gr.update(), *([gr.update()] * 23), tb_update_messages()


    clean_name = preset_name.strip()
    if clean_name == "None":
        tb_message_mgr.add_warning("'None' preset cannot be deleted.")
        return gr.update(value="None"), gr.update(), *([gr.update()] * 23), tb_update_messages()

    if clean_name not in tb_workflow_presets_data:
        tb_message_mgr.add_warning(f"Workflow preset '{clean_name}' not found.")
        return gr.update(), gr.update(), *([gr.update()] * 23), tb_update_messages()

    del tb_workflow_presets_data[clean_name]

    try:
        with open(TB_WORKFLOW_PRESETS_FILE, 'w') as f:
            json.dump(tb_workflow_presets_data, f, indent=4)
        tb_message_mgr.add_success(f"Workflow preset '{clean_name}' deleted.")

        updated_choices = sorted([k for k in tb_workflow_presets_data.keys() if k != "None"])
        updated_choices.insert(0, "None")
        
        # After deleting, load the "None" state to get the reset values
        none_state_outputs = tb_handle_load_workflow_preset("None")
        
        # The rest of the values come from the 'load' function, but we skip its first value (which was also for the textbox)
        return gr.update(choices=updated_choices, value="None"), "", *none_state_outputs[1:]

    except Exception as e:
        tb_message_mgr.add_error(f"Error deleting workflow preset '{clean_name}': {e}")
        _initialize_workflow_presets() # Revert
        # On error, we don't know the state, so just update the messages
        return gr.update(), gr.update(value=clean_name), *([gr.update()] * 23), tb_update_messages()

def tb_handle_reset_workflow_to_defaults():
    # This function loads the 'None' preset to get the reset values for most components...
    load_outputs = tb_handle_load_workflow_preset("None")
    # ...then it PREPENDS an update specifically for the dropdown menu.
    # The first value is for the dropdown, the rest are for the components in _WORKFLOW_LOAD_OUTPUTS_
    return gr.update(value="None"), *load_outputs

# --- END: New Workflow Preset Handlers ---


def tb_handle_start_pipeline(
    # 1. Active Tab Index
    active_tab_index,
    # 2. Selected Operations
    selected_ops,
    # Inputs
    single_video_path, batch_video_paths,
    # Upscale
    model_key, output_scale_factor, tile_size, enhance_face, denoise_strength,
    # Frame Adjust
    fps_mode, speed_factor,
    # Loop
    loop_type, num_loops,
    # Filters
    brightness, contrast, saturation, temperature, sharpen, blur, denoise, vignette, s_curve_contrast, film_grain_strength,
    # Export
    export_format, export_quality, export_max_width,
    progress=gr.Progress()
):
    tb_message_mgr.clear()
    input_paths_to_process = []
    
    if active_tab_index == 1 and batch_video_paths and len(batch_video_paths) > 0:
        # Process batch only if the batch tab is active and it has files
        input_paths_to_process = batch_video_paths
        tb_message_mgr.add_message(f"Starting BATCH pipeline for {len(input_paths_to_process)} videos (from active Batch tab).")
    elif active_tab_index == 0 and single_video_path:
        # Process single video only if the single tab is active and it has a video
        input_paths_to_process = [single_video_path]
        tb_message_mgr.add_message(f"Starting SINGLE video pipeline for {os.path.basename(single_video_path)} (from active Single tab).")
    else:
        # Handle cases where the active tab is empty
        if active_tab_index == 1:
            tb_message_mgr.add_warning("Batch Input tab is active, but no files were provided.")
        else: # active_tab_index == 0 or default
            tb_message_mgr.add_warning("Single Video Input tab is active, but no video was provided.")
        return None, tb_update_messages()

    if not selected_ops:
        tb_message_mgr.add_warning("No operations selected for the pipeline. Please check at least one box in 'Pipeline Steps'.")
        return None, tb_update_messages()

    # Map checkbox labels to operation keys
    op_map = {
        "upscale": "upscale",
        "frames": "frame_adjust",
        "loop": "loop",
        "filters": "filters",
        "export": "export"
    }
    
    # Define the execution order
    execution_order = ["upscale", "frame_adjust", "filters", "loop", "export"]
    
    pipeline_config = {"operations": []}

    # Build the pipeline configuration based on user selections in the correct order
    for op_key in execution_order:
        # Find the display name from the op_map that corresponds to the current key
        display_name = next((d_name for d_name, k_name in op_map.items() if k_name == op_key), None)
        
        if display_name and display_name in selected_ops:
            if op_key == "upscale":
                pipeline_config["operations"].append({
                    "name": "upscale",
                    "params": {
                        "model_key": model_key,
                        "output_scale_factor_ui": float(output_scale_factor),
                        "tile_size": int(tile_size),
                        "enhance_face": enhance_face,
                        "denoise_strength_ui": denoise_strength
                    }
                })
            elif op_key == "frame_adjust":
                pipeline_config["operations"].append({
                    "name": "frame_adjust",
                    "params": { "target_fps_mode": fps_mode, "speed_factor": speed_factor }
                })
            elif op_key == "loop":
                pipeline_config["operations"].append({
                    "name": "loop",
                    "params": { "loop_type": loop_type, "num_loops": num_loops }
                })
            elif op_key == "filters":
                pipeline_config["operations"].append({
                    "name": "filters",
                    "params": {
                        "brightness": brightness, "contrast": contrast, "saturation": saturation, "temperature": temperature,
                        "sharpen": sharpen, "blur": blur, "denoise": denoise, "vignette": vignette,
                        "s_curve_contrast": s_curve_contrast, "film_grain_strength": film_grain_strength
                    }
                })
            elif op_key == "export":
                pipeline_config["operations"].append({
                    "name": "export",
                    "params": {
                        "export_format": export_format,
                        "quality_slider": int(export_quality),
                        "max_width": int(export_max_width)
                    }
                })

    # Call the batch processor, which now handles both single and batch jobs
    final_video_path = tb_processor.tb_process_video_batch(input_paths_to_process, pipeline_config, progress)

    # Return the final video path to the player (will be None for batch, path for single)
    return final_video_path, tb_update_messages()

def tb_handle_tab_change_debug(evt: gr.SelectData):
    """Debug handler to confirm tab selection is being registered."""
    if not evt: # Should not happen in practice but good for safety
        return 0, tb_update_messages()
        
    index = evt.index
    tab_name = "Single Video" if index == 0 else "Batch Video"
    tb_message_mgr.add_message(f"DEBUG: Active tab changed to -> {tab_name} (Index: {index})")
    
    # Return the new index for the state and the updated messages
    return index, tb_update_messages()
    
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
        tb_message_mgr.add_error(f"Invalid tile size value: {tile_size}. Using None (0).")
        tile_size_int = 0

    try:
        output_scale_factor_float = float(output_scale_factor_from_slider)
        if not (output_scale_factor_float >= 0.25):
             tb_message_mgr.add_error(f"Invalid output scale factor: {output_scale_factor_from_slider:.2f}. Must be >= 0.25.")
             return None, tb_update_messages()
    except ValueError:
        tb_message_mgr.add_error(f"Invalid output scale factor: {output_scale_factor_from_slider}. Not a valid number.")
        return None, tb_update_messages()

    output_video = tb_processor.tb_upscale_video(
        video_path,
        model_key_selected,
        output_scale_factor_float,
        tile_size_int,
        enhance_face_ui,
        denoise_strength_from_slider,
        progress=progress
    )
    return output_video, tb_update_messages()

def tb_get_model_info_and_update_scale_slider(model_key_selected: str):
    native_scale = 2.0
    slider_min = 1.0
    slider_max = 2.0
    slider_step = 0.05
    slider_default_value = 2.0
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
            slider_min = 1.0

            if native_scale >= 4.0: slider_step = 0.1
            elif native_scale >= 2.0: slider_step = 0.05

        model_info_text = f"{description}"
        slider_label = f"Target Upscale Factor (Native {native_scale}x)"

        if model_key_selected == "RealESR-general-x4v3":
            denoise_slider_visible = True

    model_info_update = gr.update(value=model_info_text)
    outscale_slider_update = gr.update(
        minimum=slider_min, maximum=slider_max, step=slider_step,
        value=slider_default_value, label=slider_label
    )
    denoise_slider_update = gr.update(
        visible=denoise_slider_visible, value=denoise_slider_value
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
    log_messages_from_action = []

    studio_module_instance = None
    if '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'current_generator'):
        studio_module_instance = sys.modules['__main__']
        print("Found studio context in __main__.")
    elif 'studio' in sys.modules and hasattr(sys.modules['studio'], 'current_generator'):
        studio_module_instance = sys.modules['studio']
        print("Found studio context in sys.modules['studio'].")

    if studio_module_instance is None:
        print("ERROR: Could not find the 'studio' module's active context.")
        tb_message_mgr.add_message("ERROR: Could not find the 'studio' module's active context in sys.modules.")
        tb_message_mgr.add_error("Deletion Failed: Studio module context not found.")
        return tb_update_messages()

    job_queue_instance = getattr(studio_module_instance, 'job_queue', None)
    JobStatus_enum = getattr(studio_module_instance, 'JobStatus', None)

    if job_queue_instance and JobStatus_enum:
        current_job_in_queue = getattr(job_queue_instance, 'current_job', None)
        if current_job_in_queue and hasattr(current_job_in_queue, 'status') and current_job_in_queue.status == JobStatus_enum.RUNNING:
            tb_message_mgr.add_warning("Cannot unload model: A video generation job is currently running.")
            tb_message_mgr.add_message("Please wait for the current job to complete or cancel it first using the main interface.")
            print("Cannot unload model: A job is currently running in the queue.")
            return tb_update_messages()

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
            pass

        tb_message_mgr.add_message(f" Deletion of '{model_name_str}' initiated.")
        log_messages_from_action.append(f" Found active generator: {model_name_str}. Preparing for deletion.")
        print(f"Found active generator: {model_name_str}. Preparing for deletion.")

        try:
            if hasattr(generator_object_to_delete, 'unload_loras') and callable(generator_object_to_delete.unload_loras):
                print("   - LoRAs: Unloading from transformer...")
                generator_object_to_delete.unload_loras()
            else:
                log_messages_from_action.append("    - LoRAs: No unload method found or not applicable.")

            if hasattr(generator_object_to_delete, 'transformer') and generator_object_to_delete.transformer is not None:
                transformer_object_ref = generator_object_to_delete.transformer
                transformer_name_for_log = transformer_object_ref.__class__.__name__
                print(f"   - Transformer ({transformer_name_for_log}): Preparing for memory operations.")

                if hasattr(transformer_object_ref, 'device') and transformer_object_ref.device != cpu:
                    if hasattr(transformer_object_ref, 'to') and callable(transformer_object_ref.to):
                        try:
                            print(f"   - Transformer ({transformer_name_for_log}): Moving to CPU...")
                            transformer_object_ref.to(cpu)
                            log_messages_from_action.append("    - Transformer moved to CPU.")
                            print(f"   - Transformer ({transformer_name_for_log}): Moved to CPU.")
                        except Exception as e_cpu:
                            error_msg_cpu = f"    - Transformer ({transformer_name_for_log}): Move to CPU FAILED: {e_cpu}"
                            log_messages_from_action.append(error_msg_cpu)
                            print(error_msg_cpu)
                    else:
                        log_messages_from_action.append(f"    - Transformer ({transformer_name_for_log}): Cannot move to CPU, 'to' method not found.")
                        print(f"   - Transformer ({transformer_name_for_log}): Cannot move to CPU, 'to' method not found.")
                elif hasattr(transformer_object_ref, 'device') and transformer_object_ref.device == cpu:
                     log_messages_from_action.append("    - Transformer already on CPU.")
                     print(f"   - Transformer ({transformer_name_for_log}): Already on CPU.")
                else:
                    log_messages_from_action.append("    - Transformer: Could not determine device or move to CPU.")
                    print(f"   - Transformer ({transformer_name_for_log}): Could not determine device or move to CPU.")

                print(f"   - Transformer ({transformer_name_for_log}): Removing attribute from generator...")
                generator_object_to_delete.transformer = None
                print(f"   - Transformer ({transformer_name_for_log}): Deleting Python reference...")
                del transformer_object_ref
                log_messages_from_action.append("    - Transformer reference deleted.")
                print(f"   - Transformer ({transformer_name_for_log}): Reference deleted.")
            else:
                log_messages_from_action.append("    - Transformer: Not found or already unloaded.")
                print("   - Transformer: Not found or already unloaded.")

            generator_class_name_for_log = generator_object_to_delete.__class__.__name__
            print(f"   - Model Generator ({generator_class_name_for_log}): Setting global reference to None...")
            setattr(studio_module_instance, 'current_generator', None)
            log_messages_from_action.append("    - 'current_generator' in studio module set to None.")
            print("   - Global 'current_generator' in studio module successfully set to None.")

            print(f"   - Model Generator ({generator_class_name_for_log}): Deleting local Python reference...")
            del generator_object_to_delete
            print(f"   - Model Generator ({generator_class_name_for_log}): Python reference deleted.")

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
            log_messages_from_action.append(f"    - {error_msg_del}")
            print(f"   - {error_msg_del}")
            traceback.print_exc()
            tb_message_mgr.add_error(f"Deletion Error: {e_del}")
    else:
        tb_message_mgr.add_message("ℹ️ No active generator found. Nothing to delete.")
        print("No active generator found via direct access. Nothing to delete.")

    for msg_item in log_messages_from_action:
        tb_message_mgr.add_message(msg_item)

    return tb_update_messages()

def tb_handle_manually_save_video(temp_video_path_from_component):
    tb_message_mgr.clear()
    if not temp_video_path_from_component:
        tb_message_mgr.add_warning("No video in the output player to save.")
        return temp_video_path_from_component, tb_update_messages()

    copied_path = tb_processor.tb_copy_video_to_permanent_storage(temp_video_path_from_component)

    if copied_path and os.path.abspath(copied_path) != os.path.abspath(temp_video_path_from_component):
        tb_message_mgr.add_success(f"Video successfully copied to permanent storage.")

    return temp_video_path_from_component, tb_update_messages()

def tb_handle_clear_temp_files():
    tb_message_mgr.clear()
    success = tb_processor.tb_clear_temporary_files()

    if success:
        tb_message_mgr.add_success("Temporary files cleared.")
    else:
        tb_message_mgr.add_warning("Issue during temporary file cleanup. Check messages.")

    return None, tb_update_messages()

def tb_handle_use_processed_as_input(processed_video_path):
    if not processed_video_path:
        tb_message_mgr.add_warning("No processed video available to use as input.")
        # Return updates for all 3 outputs, changing nothing.
        return gr.update(), tb_update_messages(), gr.update()
    else:
        tb_message_mgr.add_message("Moved processed video to input.")
        # Return new value for input, messages, and None to clear analysis.
        return processed_video_path, tb_update_messages(), None

def tb_handle_join_videos(video_files_list, custom_output_name, progress=gr.Progress()): # Add new parameter
    tb_message_mgr.clear()
    
    if not video_files_list:
        tb_message_mgr.add_warning("No video files were uploaded to join.")
        return None, tb_update_messages()
        
    video_paths = [file.name for file in video_files_list]
    
    # Pass the custom name to the processor
    output_video = tb_processor.tb_join_videos(video_paths, custom_output_name, progress)
    return output_video, tb_update_messages()

def tb_handle_export_video(video_path, export_format, quality, max_width, custom_name, progress=gr.Progress()):
    tb_message_mgr.clear()
    if not video_path:
        tb_message_mgr.add_warning("No input video in the top-left player to export.")
        # Return None for the video player and the message update
        return None, tb_update_messages()

    # The input video for this operation is ALWAYS the one in the main input player.
    output_file = tb_processor.tb_export_video(
        video_path,
        export_format,
        quality,
        max_width,
        custom_name,
        progress
    )
    
    # Return the path to the new video and the updated messages.
    return output_file, tb_update_messages()
    
def tb_get_formatted_toolbar_stats():
    vram_full_str = "VRAM: N/A"
    gpu_full_str = "GPU: N/A"
    ram_full_str = "RAM: N/A"

    vram_component_visible = False
    gpu_component_visible = False

    try:
        ram_info_psutil = psutil.virtual_memory()
        ram_used_gb = ram_info_psutil.used / (1024**3)
        ram_total_gb = ram_info_psutil.total / (1024**3)
        ram_full_str = f"RAM: {ram_used_gb:.1f}/{round(ram_total_gb)}GB ({round(ram_info_psutil.percent)}%)"

        if torch.cuda.is_available():
            _, nvidia_metrics, _ = SystemMonitor.get_nvidia_gpu_info()
            if nvidia_metrics:
                vram_used = nvidia_metrics.get('memory_used_gb', 0.0)
                vram_total = nvidia_metrics.get('memory_total_gb', 0.0)
                vram_full_str = f"VRAM: {vram_used:.1f}/{round(vram_total)}GB"
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
                # Replace gr.State with a hidden gr.Number for robust state passing
                tb_active_tab_index_storage = gr.Number(value=0, visible=False)

                with gr.Tabs(elem_id="toolbox_input_tabs") as tb_input_tabs:
                    with gr.TabItem("Single Video Input", id=0):
                        tb_input_video_component = gr.Video(
                            label="Upload Video for processing",
                            autoplay=True,
                            elem_classes="video-size",
                            elem_id="toolbox-video-player"
                        )
                    with gr.TabItem("Batch Video Input", id=1):
                        tb_batch_input_files = gr.File(
                            label="Upload Multiple Videos for Batch Processing",
                            file_count="multiple",
                            type="filepath"
                        )
                tb_start_pipeline_btn = gr.Button("🚀 Start Pipeline Processing", variant="primary", size="sm", elem_id="toolbox-start-pipeline-btn")
                    
            with gr.Column(scale=1):
                with gr.Tabs(elem_id="toolbox_output_tabs"):
                    with gr.TabItem("Video Output"):
                        tb_processed_video_output = gr.Video(
                            label="Processed Video",
                            autoplay=True,
                            interactive=False,
                            elem_classes="video-size"
                        )
                        with gr.Row():
                            tb_use_processed_as_input_btn = gr.Button("Use as Input", size="sm", scale=4)
                            tb_manual_save_btn = gr.Button("Manual Save", variant="secondary", size="sm", scale=4, visible=not initial_autosave_state)

        with gr.Row():                  
            with gr.Column(scale=1):
                with gr.Accordion("Processing Pipeline", open=True):
                    gr.Markdown("Required for batch processing and recommended for single video. ", elem_classes="small-text-info")
                    with gr.Row(equal_height=False):
                        with gr.Group():
                            tb_pipeline_steps_chkbox = gr.CheckboxGroup(
                                label="Pipeline Steps:",
                                choices=["upscale", "frames", "filters", "loop", "export"],
                                value=[],
                                info="Select which pre-configured operations to run. Executed in order."
                            )
                                 
            # --- Right Column: Workflow Presets ---
            with gr.Column(scale=1):
                with gr.Accordion("Workflow Presets", open=True):
                    gr.Markdown("Save/load all operation settings and active steps.", elem_classes="small-text-info")
                    with gr.Row():
                        workflow_choices = sorted([k for k in tb_workflow_presets_data.keys() if k != "None"])
                        workflow_choices.insert(0, "None")
                        with gr.Column(scale=1):
                            tb_workflow_preset_select = gr.Dropdown(
                                choices=workflow_choices, value="None", label="Load Workflow"
                            )
                        with gr.Column(scale=1):    
                            tb_workflow_preset_name_input = gr.Textbox(
                            label="Preset Name (for saving)", placeholder="e.g., My Favorite Upscale"
                        )
                    with gr.Group():  
                        with gr.Row():
                            tb_workflow_save_btn = gr.Button("💾 Save/Update", size="sm", variant="primary")
                            tb_workflow_delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                            tb_workflow_reset_btn = gr.Button("🔄 Reset All to Defaults", size="sm")
                                    
        with gr.Row():                  
            with gr.Column():
                with gr.Group():                        
                    tb_analyze_button = gr.Button("Click to Analyze Input Video", size="sm", variant="huggingface")
                    with gr.Accordion("Video Analysis Results", open=False) as tb_analysis_accordion:
                        tb_video_analysis_output = gr.Textbox(
                            container=False, lines=10, show_label=False,
                            interactive=False, elem_classes="analysis-box",
                        )
                        
            with gr.Column():                        
                with gr.Group():
                    with gr.Row():
                        tb_monitor_toggle_checkbox = gr.Checkbox(label="Live System Monitoring", value=False)
                        tb_autosave_checkbox = gr.Checkbox(label="Autosave", value=initial_autosave_state)
                    tb_resource_monitor_output = gr.Textbox(
                        show_label=False, container=False, max_lines=8,
                        interactive=False, visible=False,
                    )
                    with gr.Row():                          
                        tb_delete_studio_transformer_btn = gr.Button("Click to Unload Studio Model", size="sm", scale=3, variant="stop")

    with gr.Accordion("Operations", open=True):
        with gr.Tabs():
            with gr.TabItem("📈 Upscale Video (ESRGAN)"):
                with gr.Row():
                    gr.Markdown("Upscale video resolution using Real-ESRGAN.")
                with gr.Row():
                    with gr.Column(scale=2):
                        tb_upscale_model_select = gr.Dropdown(
                            choices=list(tb_processor.esrgan_upscaler.supported_models.keys()),
                            value=list(tb_processor.esrgan_upscaler.supported_models.keys())[0] if tb_processor.esrgan_upscaler.supported_models else None,
                            label="ESRGAN Model",
                            info="Select the Real-ESRGAN model."
                        )
                        default_model_key_init = list(tb_processor.esrgan_upscaler.supported_models.keys())[0] if tb_processor.esrgan_upscaler.supported_models else None
                        initial_model_info_gr_val, initial_slider_gr_val, initial_denoise_gr_val = tb_get_model_info_and_update_scale_slider(default_model_key_init)

                        tb_selected_model_scale_display = gr.Textbox(
                            label="Selected Model Info",
                            value=initial_model_info_gr_val.get('value', "Info: Select a model."),
                            interactive=False,
                            lines=2
                        )

                        tb_upscale_factor_slider = gr.Slider(
                            minimum=initial_slider_gr_val.get('minimum', 1.0),
                            maximum=initial_slider_gr_val.get('maximum', 2.0),
                            step=initial_slider_gr_val.get('step', 0.05),
                            value=initial_slider_gr_val.get('value', 2.0),
                            label=initial_slider_gr_val.get('label', "Target Upscale Factor"),
                            info="Desired output scale (e.g., 2.0 for 2x). Video is upscaled by the model, then resized if this differs from native scale."
                        )
                    with gr.Column(scale=2):
                        tb_upscale_tile_size_radio = gr.Radio(
                            choices=[("None (Recommended)", 0), ("512px", 512), ("256px", 256)],
                            value=0, label="Tile Size for Upscaling",
                            info="Splits video frames into tiles for processing. 'None' disables tiling. Smaller values (e.g., 512, 256) use less VRAM but are slower and can potentially show seams on some videos. Use if 'None' causes Out-Of-Memory."
                        )
                        with gr.Row():
                            tb_upscale_enhance_face_checkbox = gr.Checkbox(
                                label="Enhance Faces (GFPGAN)", value=False,
                                info="Uses GFPGAN to restore (human-like) faces. Increases processing time."
                            )
                        with gr.Row():
                            tb_denoise_strength_slider = gr.Slider(
                                label="Denoise Strength (for RealESR-general-x4v3)",
                                minimum=0.0, maximum=1.0, step=0.01,
                                value=initial_denoise_gr_val.get('value', 0.5),
                                info="Adjusts denoising for RealESR-general-x4v3. 0.0=Max WDN, <1.0=Blend, 1.0=No WDN.",
                                visible=initial_denoise_gr_val.get('visible', False),
                                interactive=True
                            )
                with gr.Row():
                    tb_upscale_video_btn = gr.Button("🚀 Upscale Video", variant="primary")

            with gr.TabItem("🎞️ Frame Adjust (Speed & Interpolation)"):
                with gr.Row():
                    gr.Markdown("Adjust video speed and interpolate frames using RIFE AI.")
                tb_process_fps_mode = gr.Radio(
                    choices=["No Interpolation", "2x Frames", "4x Frames"], value="No Interpolation", label="RIFE Frame Interpolation",
                    info="Select '2x' or '4x' RIFE Interpolation to double or quadruple the frame rate, creating smoother motion. 4x is more intensive and runs the 2x process twice."
                )
                tb_process_speed_factor = gr.Slider(
                    minimum=0.25, maximum=4.0, step=0.05, value=1.0, label="Adjust Video Speed Factor",
                    info="Values < 1.0 slow down the video, values > 1.0 speed it up. Affects video and audio."
                )
                tb_process_frames_btn = gr.Button("🚀 Process Frames", variant="primary")

            with gr.TabItem("🎨 Video Filters (FFmpeg)"):
                with gr.Row():
                    gr.Markdown("Apply visual enhancements using FFmpeg filters.")
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

                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        with gr.Row():
                            preset_choices = list(tb_filter_presets_data.keys()) if tb_filter_presets_data else ["none"]
                            if "none" not in preset_choices and preset_choices:
                                preset_choices.insert(0,"none")
                            elif not preset_choices:
                                preset_choices = ["none"]

                            tb_filter_preset_select = gr.Dropdown(choices=preset_choices, value="none", label="Load Preset", scale=2)
                            tb_new_preset_name_input = gr.Textbox(label="Preset Name (for saving/editing)", placeholder="Select preset or type new name...", scale=2)
                    with gr.Column(scale=1):
                        with gr.Row():
                            tb_save_preset_btn = gr.Button("💾 Save/Update", variant="primary", scale=1)
                            tb_delete_preset_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                        with gr.Row():
                            tb_reset_filters_btn = gr.Button("🔄 Reset All Sliders to 'None' Preset")
                            
            with gr.TabItem("🔄 Video Loop"):
                with gr.Row():
                    gr.Markdown("Create looped or ping-pong versions of the video.")

                tb_loop_type_select = gr.Radio(choices=["loop", "ping-pong"], value="loop", label="Loop Type")
                tb_num_loops_slider = gr.Slider(
                    minimum=1, maximum=10, step=1, value=1, label="Number of Loops/Repeats",
                    info="The video will play its original content, then repeat this many additional times. E.g., 1 loop = 2 total plays of the segment."
                )
                tb_create_loop_btn = gr.Button("🔁 Create Loop", variant="primary")

            with gr.TabItem("🖼️ Frames Studio"):
                with gr.Column():
                    gr.Markdown("### 1. Extract Frames from Video")
                    gr.Markdown(
                        "⚠️ **Warning:** Extracting frames from high-resolution (e.g., 4K+) or long videos can consume a significant amount of disk space (many gigabytes) and may cause the Frames Studio gallery to load slowly. Proceed with caution."
                    )
                    gr.Markdown("Extract frames from the **uploaded video (top-left)** as images. These folders can then be loaded into the Frames Studio below.")
                    with gr.Row():
                        tb_extract_rate_slider = gr.Number(
                            label="Extract Every Nth Frame", value=1, minimum=1, step=1,
                            info="1 = all frames. N = 1st, (N+1)th... (i.e., frame 0, frame N, frame 2N, etc.)",
                            scale=1
                        )
                        tb_extract_frames_btn = gr.Button("🔨 Extract Frames", variant="primary", scale=2)

                gr.Markdown("---")

                with gr.Column():
                    gr.Markdown("### 2. Frames Studio")
                    gr.Markdown("Load an extracted frames folder to view, delete, and manage individual frames before reassembling.")
                    with gr.Row():
                        tb_extracted_folders_dropdown = gr.Dropdown(
                            label="Select Extracted Folder to Load",
                            info="Select a folder from your 'extracted_frames' directory.",
                            scale=3
                        )
                        tb_refresh_extracted_folders_btn = gr.Button("🔄 Refresh List", scale=1)
                        tb_clear_selected_folder_btn = gr.Button(
                            "🗑️ Delete ENTIRE Folder", variant="stop", interactive=False, scale=1
                        )
                    tb_load_frames_to_studio_btn = gr.Button("🖼️ Load Frames to Studio", variant="secondary")

                    # Redesigned Studio Area
                    with gr.Column(variant="panel"):
                        with gr.Column(elem_id="gallery-scroll-wrapper"):
                            tb_frames_gallery = gr.Gallery(
                                label="Extracted Frames", show_label=False, elem_id="toolbox_frames_gallery",
                                columns=8, # height is now controlled by the wrapper's CSS
                                object_fit="contain", preview=True
                            )
                        with gr.Row():
                            with gr.Column(scale=1, min_width=220):
                                tb_save_selected_frame_btn = gr.Button("💾 Save Selected Frame", interactive=False)
                                tb_delete_selected_frame_btn = gr.Button("🗑️ Delete Selected Frame", variant="stop", interactive=False)
                            with gr.Column(scale=3):
                                tb_frame_info_box = gr.Textbox(label="Selected Frame Info", interactive=False, placeholder="Click a frame in the gallery above to select it.", lines=2)

                gr.Markdown("---")

                with gr.Column():
                    gr.Markdown("### 3. Reassemble Frames to Video")
                    gr.Markdown("After you are satisfied with the frames in the studio, reassemble them into a new video.")
                    with gr.Row():
                        tb_reassemble_output_fps = gr.Number(label="Output Video FPS", value=30, minimum=1, step=1)
                        tb_reassemble_video_name_input = gr.Textbox(label="Output Video Name (optional, .mp4 added)", placeholder="e.g., my_edited_video")
                    tb_reassemble_frames_btn = gr.Button("🧩 Reassemble From Studio", variant="primary")

            with gr.TabItem("🧩 Join Videos (Concatenate)"):
                with gr.Accordion("Select two or more videos to join them together into a single file", open=True):
                    gr.Markdown(
                        """
                        *   **Input:** The Input accepts multiple videos dragged in or ctrl+clicked via `Click to Upload`**.
                        *   **Output:** The result will appear in the **'Processed Video' player (top-right)** for you to review.
                        *   **Saving:** The output is saved to your 'saved_videos' folder if 'Autosave' is enabled. Otherwise, you must click the 'Manual Save' button.
                        """                    
                    )
                tb_join_videos_input = gr.File(
                    label="Upload Videos to Join",
                    file_count="multiple",
                    file_types=["video", "file"]
                )

                tb_join_video_name_input = gr.Textbox(
                    label="Output Video Name (optional, .mp4 and timestamp added)", 
                    placeholder="e.g., my_awesome_compilation"
                )
                
                tb_join_videos_btn = gr.Button("🤝 Join Videos", variant="primary")
                
            with gr.TabItem("📦 Export & Compress"):
                with gr.Accordion("Compress your final video and/or convert it into a shareable format", open=True):
                    gr.Markdown(
                        """
                        *   **Input:** This operation always uses the video in the **'Upload Video' player (top-left)**.
                        *   **Output:** The result will appear in the **'Processed Video' player (top-right)** for you to review.
                        *   **Saving:** The output is saved to your 'saved_videos' folder if 'Autosave' is enabled. Otherwise, you must click the 'Manual Save' button. Note: GIFs will _always_ be saved!
                        *   **Note:** WebM and GIF encoding can be slow for long or high-resolution videos. Please be patient!
                        """
                    )
                with gr.Row():
                    with gr.Column(scale=2):
                        tb_export_format_radio = gr.Radio(
                            ["MP4", "WebM", "GIF"], value="MP4", label="Output Format",
                            info="MP4 is best for general use. WebM is great for web/Discord (smaller size). GIF is a widely-supported format for short, silent, looping clips. GIF output will always be saved."
                        )
                        tb_export_quality_slider = gr.Slider(
                            0, 100, value=85, step=1, label="Quality",
                            info="Higher quality means a larger file size. 80-90 is a good balance for MP4/WebM."
                        )
                    with gr.Column(scale=2):
                        tb_export_resize_slider = gr.Slider(
                            256, 2048, value=1024, step=64, label="Max Width (pixels)",
                            info="Resizes the video to this maximum width while maintaining aspect ratio. A powerful way to reduce file size."
                        )
                        tb_export_name_input = gr.Textbox(
                            label="Output Filename (optional)",
                            placeholder="e.g., my_final_video_for_discord"
                        )
                
                tb_export_video_btn = gr.Button("🚀 Export Video", variant="primary")
                
    with gr.Accordion("💡 Post-processing Guide & Tips", open=False):
        with gr.Tabs():
            with gr.TabItem("🚀 Getting Started"):
                gr.Markdown("""
                ### Welcome to the Toolbox!
                
                **1. Input & Output**
                *   Most tools use the video in the **Upload Video player ⬅️ (top-left)** as their input.
                *   Your results will appear in the **Processed Video player ➡️ (top-right)**.

                **2. Chaining Operations (Applying multiple effects)**
                *   To use a result as the input for your next step:
                    1.  Run an operation (like Upscale).
                    2.  When the result appears, click the **'Use as Input'** button.
                    3.  Your result is now in the input player, ready for the next operation!

                **3. Saving Your Work**
                *   **Autosave:** When the `Autosave` checkbox is on, all results are automatically saved to the `saved_videos` folder.
                *   **Manual Save:** If `Autosave` is off, results go to a temporary folder. Use the **'💾 Manual Save'** button to save the video from the output player permanently.

                **4. Analyze First!**
                *   It's a good idea to click the **'Analyze Video'** button after uploading. It gives you helpful info like resolution and frame rate.
                """)

            with gr.TabItem("⛓️ The Processing Pipeline"):
                gr.Markdown("""
                ### Run Multiple Operations at Once
                The pipeline lets you set up a series of operations and run them with a single click. This is the main way to process videos.

                **How to Use the Pipeline:**
                1.  **Configure:** Go to the operation tabs (📈 Upscale, 🎨 Filters, etc.) and set the sliders and options exactly how you want them.
                2.  **Select:** In the **'Processing Pipeline'** section, check the boxes for the steps you want to run.
                3.  **Input:** Make sure your video is in the 'Single Video' tab, or your files are in the 'Batch Video' tab.
                4.  **Execute:** Click the **'🚀 Start Pipeline Processing'** button.

                **Execution Order:**
                The pipeline always runs in this fixed order, no matter when you check the boxes:
                `Upscale` ➡️ `Frame Adjust` ➡️ `Filters` ➡️ `Loop` ➡️ `Export`

                **Single vs. Batch Video:**
                *   **Single Video:** Processes one video. The final result will appear in the output player.
                *   **Batch Video:** Processes multiple videos. Each video will go through the entire pipeline. Outputs are saved directly to a new, timestamped folder inside `saved_videos`. The output player will only show the very last video processed.

                **Workflow Presets:**
                *   Use presets to **save and load your entire pipeline setup**, including all slider values and selected steps.
                """)

            with gr.TabItem("🖼️ Frames Studio Workflow"):
                gr.Markdown("""
                ### Edit Your Video Frame-by-Frame
                The Frames Studio lets you break a video into images, edit them, and put them back together.

                **Step 1: Extract Frames**
                *   Upload a video and use the **'🔨 Extract Frames'** button.
                *   This creates a new folder of images in `postprocessed_output/frames/extracted_frames/`.
                *   ⚠️ **Warning:** Extracting from long or high-res videos can use a lot of disk space!

                **Step 2: Edit in the Studio**
                *   Click **'🔄 Refresh List'** to find your new folder, then click **'🖼️ Load Frames to Studio'**.
                *   The frames will appear in the gallery. Click a frame to select it.
                *   **Delete Frames:** Use the **'🗑️ Delete Selected Frame'** button to remove bad frames or glitches.
                *   **Save Frames:** Use the **'💾 Save Selected Frame'** button to save a high-quality copy of a single frame. Perfect for use as an image prompt!

                **Step 3: Reassemble Video**
                *   Once you're done editing, use the **'🧩 Reassemble From Studio'** button.
                *   This creates a new video using only the frames that are left in the folder.
                """)
            
            with gr.TabItem("⚙️ Other Tools & Tips"):
                 gr.Markdown("""
                ### Individual Operations
                *   **🧩 Join Videos:** Combine multiple video clips into a single video file. The tool will automatically handle different resolutions and audio.
                *   **📦 Export & Compress:** A powerful tool to make your final video smaller. You can lower the quality, resize the video, or convert it to `MP4`, `WebM`, or a silent `GIF`.

                ### Memory Management
                *   The **'📤 Unload Studio Model'** button can free up VRAM by removing the main video generation model from memory.
                *   This is useful before running a heavy task here, like a 4K video upscale. The main app will reload the model automatically when you need it again.

                ### 👇 Check Console Messages!
                *   The text box at the very bottom of the page shows important status updates, warnings, and error messages. If something isn't working, the answer is probably there!
                """)

    with gr.Row():
        tb_message_output = gr.Textbox(label="Console Messages", lines=10, interactive=False, elem_classes="message-box", value=tb_update_messages)
    with gr.Row():
        tb_open_folder_button = gr.Button("📁 Open Output Folder", scale=4)
        tb_clear_temp_button = gr.Button("🗑️ Clear Temporary Files", variant="stop", scale=1)

        # --- Event Handlers ---

        _ORDERED_FILTER_SLIDERS_ = [
            tb_filter_brightness, tb_filter_contrast, tb_filter_saturation, tb_filter_temperature,
            tb_filter_sharpen, tb_filter_blur, tb_filter_denoise, tb_filter_vignette,
            tb_filter_s_curve_contrast, tb_filter_film_grain_strength
        ]
        
        # A list of all operation parameter components in the correct order for workflow presets
        _ALL_PIPELINE_PARAMS_COMPONENTS_ = [
            # Upscale
            tb_upscale_model_select, tb_upscale_factor_slider, tb_upscale_tile_size_radio,
            tb_upscale_enhance_face_checkbox, tb_denoise_strength_slider,
            # Frame Adjust
            tb_process_fps_mode, tb_process_speed_factor,
            # Loop
            tb_loop_type_select, tb_num_loops_slider,
            # Filters
            *_ORDERED_FILTER_SLIDERS_,
            # Export
            tb_export_format_radio, tb_export_quality_slider, tb_export_resize_slider
        ]
        
        # The list of all inputs for the main pipeline execution
        _ALL_PIPELINE_INPUTS_ = [
            tb_active_tab_index_storage,
            tb_pipeline_steps_chkbox,
            # Inputs
            tb_input_video_component, tb_batch_input_files,
            # Parameters
            *_ALL_PIPELINE_PARAMS_COMPONENTS_
        ]

        # --- NEW: Workflow Preset Event Handlers ---
        tb_workflow_save_btn.click(
            fn=tb_handle_save_workflow_preset,
            inputs=[tb_workflow_preset_name_input, tb_pipeline_steps_chkbox, *_ALL_PIPELINE_PARAMS_COMPONENTS_],
            outputs=[tb_workflow_preset_select, tb_workflow_preset_name_input, tb_message_output]
        )

        # The list of outputs for loading must include the name box, then ALL controls in the correct order
        _WORKFLOW_LOAD_OUTPUTS_ = [
            tb_workflow_preset_name_input, # First output is the name box
            tb_pipeline_steps_chkbox,      # Second is the checkbox group
            *_ALL_PIPELINE_PARAMS_COMPONENTS_, # Then all the parameter controls
            tb_message_output              # Finally, the message box
        ]
        tb_workflow_preset_select.change(
            fn=tb_handle_load_workflow_preset,
            inputs=[tb_workflow_preset_select],
            outputs=_WORKFLOW_LOAD_OUTPUTS_
        )
        tb_workflow_delete_btn.click(
            fn=tb_handle_delete_workflow_preset,
            inputs=[tb_workflow_preset_name_input],
            # This list now also needs the dropdown prepended
            outputs=[tb_workflow_preset_select, *_WORKFLOW_LOAD_OUTPUTS_]
        )
        tb_workflow_reset_btn.click(
            fn=tb_handle_reset_workflow_to_defaults,
            inputs=None,
            # The outputs list now starts with the dropdown, followed by the standard load outputs
            outputs=[tb_workflow_preset_select, *_WORKFLOW_LOAD_OUTPUTS_]
        )
        # --- End Workflow Preset Handlers ---

        tb_start_pipeline_btn.click(
            fn=tb_handle_start_pipeline,
            inputs=_ALL_PIPELINE_INPUTS_,
            outputs=[tb_processed_video_output, tb_message_output]
        )
        # Listen for tab changes and update the state component
        tb_input_tabs.select(
            fn=tb_handle_tab_change_debug,
            inputs=None, # evt is passed implicitly
            outputs=[tb_active_tab_index_storage, tb_message_output]
        )

        # --- SINGLE VIDEO HANDLERS ---
        tb_input_video_component.upload(fn=lambda: (tb_message_mgr.clear() or tb_update_messages(), None), outputs=[tb_message_output, tb_video_analysis_output])
        tb_input_video_component.clear(fn=lambda: (tb_message_mgr.clear() or tb_update_messages(), None, None), outputs=[tb_message_output, tb_video_analysis_output, tb_processed_video_output])

        tb_analyze_button.click(
            fn=tb_handle_analyze_video,
            inputs=[tb_input_video_component],
            outputs=[tb_message_output, tb_video_analysis_output, tb_analysis_accordion]
        )
        tb_process_frames_btn.click(fn=tb_handle_process_frames, inputs=[tb_input_video_component, tb_process_fps_mode, tb_process_speed_factor], outputs=[tb_processed_video_output, tb_message_output])
        tb_create_loop_btn.click(fn=tb_handle_create_loop, inputs=[tb_input_video_component, tb_loop_type_select, tb_num_loops_slider], outputs=[tb_processed_video_output, tb_message_output])

        tb_filter_preset_select.change(
            fn=lambda preset_name_from_dropdown: (preset_name_from_dropdown, *tb_update_filter_sliders_from_preset(preset_name_from_dropdown)),
            inputs=[tb_filter_preset_select], outputs=[tb_new_preset_name_input] + _ORDERED_FILTER_SLIDERS_
        )
        tb_apply_filters_btn.click(fn=tb_handle_apply_filters, inputs=[tb_input_video_component] + _ORDERED_FILTER_SLIDERS_, outputs=[tb_processed_video_output, tb_message_output])
        tb_save_preset_btn.click(fn=tb_handle_save_user_preset, inputs=[tb_new_preset_name_input] + _ORDERED_FILTER_SLIDERS_, outputs=[tb_filter_preset_select, tb_message_output, tb_new_preset_name_input])
        tb_delete_preset_btn.click(fn=tb_handle_delete_user_preset, inputs=[tb_new_preset_name_input], outputs=[tb_filter_preset_select, tb_message_output, tb_new_preset_name_input] + _ORDERED_FILTER_SLIDERS_)
        tb_reset_filters_btn.click(fn=tb_handle_reset_all_filters, inputs=None, outputs=[tb_filter_preset_select, tb_new_preset_name_input, *_ORDERED_FILTER_SLIDERS_, tb_message_output])

        tb_use_processed_as_input_btn.click(
            fn=tb_handle_use_processed_as_input,
            inputs=[tb_processed_video_output],
            outputs=[tb_input_video_component, tb_message_output, tb_video_analysis_output]
        )

        tb_upscale_video_btn.click(
            fn=tb_handle_upscale_video,
            inputs=[tb_input_video_component, tb_upscale_model_select, tb_upscale_factor_slider, tb_upscale_tile_size_radio, tb_upscale_enhance_face_checkbox, tb_denoise_strength_slider],
            outputs=[tb_processed_video_output, tb_message_output]
        )
        tb_upscale_model_select.change(
            fn=tb_get_model_info_and_update_scale_slider, inputs=[tb_upscale_model_select],
            outputs=[tb_selected_model_scale_display, tb_upscale_factor_slider, tb_denoise_strength_slider]
        )

        # --- Frames Studio Event Handlers ---
        tb_extract_frames_btn.click(
            fn=tb_handle_extract_frames, inputs=[tb_input_video_component, tb_extract_rate_slider], outputs=[tb_message_output]
        ).then(
            fn=tb_handle_refresh_extracted_folders, inputs=None,
            outputs=[tb_extracted_folders_dropdown, tb_message_output, tb_clear_selected_folder_btn, tb_frames_gallery, tb_frame_info_box]
        )
        tb_refresh_extracted_folders_btn.click(
            fn=tb_handle_refresh_extracted_folders, inputs=None,
            outputs=[tb_extracted_folders_dropdown, tb_message_output, tb_clear_selected_folder_btn, tb_frames_gallery, tb_frame_info_box]
        )
        tb_extracted_folders_dropdown.change(
            fn=lambda selection: gr.update(interactive=bool(selection)),
            inputs=[tb_extracted_folders_dropdown], outputs=[tb_clear_selected_folder_btn]
        )
        tb_clear_selected_folder_btn.click(
            fn=tb_handle_clear_selected_folder, inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_message_output, tb_extracted_folders_dropdown]
        ).then(
            fn=lambda selection: gr.update(interactive=bool(selection)),
            inputs=[tb_extracted_folders_dropdown], outputs=[tb_clear_selected_folder_btn]
        )

        tb_load_frames_to_studio_btn.click(
            fn=tb_handle_load_frames_to_studio,
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_message_output, tb_frames_gallery, tb_frame_info_box]
        )
        tb_frames_gallery.select(
            fn=tb_handle_frame_select,
            inputs=None, # evt_data is passed implicitly
            outputs=[tb_frame_info_box, tb_save_selected_frame_btn, tb_delete_selected_frame_btn]
        )
        tb_delete_selected_frame_btn.click(
            fn=tb_handle_delete_selected_frame,
            inputs=[tb_extracted_folders_dropdown, tb_frame_info_box],
            outputs=[tb_frame_info_box, tb_message_output] # Step 1: Delete and update logs. Info box gets delete status.
        ).then(
            # Step 2: Refresh the gallery content ONLY. Don't touch messages or the info box.
            fn=tb_processor.tb_get_frames_from_folder,
            inputs=[tb_extracted_folders_dropdown],
            outputs=[tb_frames_gallery]
        ).then(
            # Step 3: Disable buttons, as the selection is now invalid.
            fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
            inputs=None,
            outputs=[tb_save_selected_frame_btn, tb_delete_selected_frame_btn]
        )
        tb_save_selected_frame_btn.click(
            fn=tb_handle_save_selected_frame,
            inputs=[tb_extracted_folders_dropdown, tb_frame_info_box],
            outputs=[tb_message_output, tb_frame_info_box]
        )
        tb_reassemble_frames_btn.click(
            fn=tb_handle_reassemble_frames,
            inputs=[tb_extracted_folders_dropdown, tb_reassemble_output_fps, tb_reassemble_video_name_input],
            outputs=[tb_processed_video_output, tb_message_output]
        )
        
        tb_join_videos_btn.click(
            fn=tb_handle_join_videos,
            # Add the new textbox to the inputs list
            inputs=[tb_join_videos_input, tb_join_video_name_input],
            outputs=[tb_processed_video_output, tb_message_output]
        )
        
        tb_export_video_btn.click(
            fn=tb_handle_export_video,
            inputs=[
                tb_input_video_component, # The video to process
                tb_export_format_radio,
                tb_export_quality_slider,
                tb_export_resize_slider,
                tb_export_name_input
            ],
            # The outputs now include the video player.
            outputs=[tb_processed_video_output, tb_message_output]
        )
        
        # --- Other System Handlers ---
        tb_open_folder_button.click(fn=lambda: tb_processor.tb_open_output_folder() or tb_update_messages(), outputs=[tb_message_output])
        tb_monitor_toggle_checkbox.change(fn=lambda is_enabled: gr.update(visible=is_enabled), inputs=[tb_monitor_toggle_checkbox], outputs=[tb_resource_monitor_output])
        tb_monitor_timer = gr.Timer(2, active=True)
        tb_monitor_timer.tick(fn=tb_handle_update_monitor, inputs=[tb_monitor_toggle_checkbox], outputs=[tb_resource_monitor_output])
        tb_delete_studio_transformer_btn.click(fn=tb_handle_delete_studio_transformer, inputs=[], outputs=[tb_message_output])
        tb_manual_save_btn.click(fn=tb_handle_manually_save_video, inputs=[tb_processed_video_output], outputs=[tb_processed_video_output, tb_message_output])

        def tb_handle_autosave_toggle(autosave_is_on_ui_value):
            settings_instance.set("toolbox_autosave_enabled", autosave_is_on_ui_value)
            tb_processor.set_autosave_mode(autosave_is_on_ui_value)
            return {
                tb_manual_save_btn: gr.update(visible=not autosave_is_on_ui_value),
                tb_message_output: gr.update(value=tb_update_messages())
            }
        tb_autosave_checkbox.change(fn=tb_handle_autosave_toggle, inputs=[tb_autosave_checkbox], outputs=[tb_manual_save_btn, tb_message_output])
        tb_clear_temp_button.click(fn=tb_handle_clear_temp_files, inputs=None, outputs=[tb_processed_video_output, tb_message_output])

    return tb_toolbox_ui_main_container, tb_input_video_component
 
 
# --- Main execution block for standalone mode ---

if __name__ == "__main__":
    import argparse

    def launch_standalone():
        """Creates and launches the Gradio interface for the toolbox when run as a script."""
        
        # 1. Setup and parse command-line arguments, similar to studio.py
        parser = argparse.ArgumentParser(description="Run FramePack Toolbox in Standalone Mode")
        parser.add_argument('--share', action='store_true', help="Enable Gradio sharing link")
        parser.add_argument("--server", type=str, default='127.0.0.1', help="Server name to launch on (default: 127.0.0.1)")
        parser.add_argument("--port", type=int, required=False, help="Server port to launch on (default: 7860)")
        parser.add_argument("--inbrowser", action='store_true', help="Automatically open in browser")
        args = parser.parse_args()

        # 2. Define custom CSS
        css = """
        /* hide the gr.Video source selection bar for tb_input_video_component */
        #toolbox-video-player .source-selection {
            display: none !important;
        }
        /* control sizing for gr.Video components */    
        .video-size video {
            max-height: 60vh;
            min-height: 300px !important;
            object-fit: contain;
        }
        /* NEW: Closes the gap between input tabs and the pipeline accordion below them */
        #pipeline-controls-wrapper {
            margin-top: -15px !important; /* Adjust this value to get the perfect "snug" fit */
        }
        /* --- NEW CSS RULE FOR GALLERY SCROLLING --- */
        #gallery-scroll-wrapper {
            max-height: 600px; /* Set your desired fixed height */
            overflow-y: auto;   /* Add a scrollbar only when needed */
        }
        /* ---  --- */
        #toolbox-start-pipeline-btn {
            margin-top: -14px !important; /* Adjust this value to get the perfect alignment */
        }
        .small-text-info {
            font-size: 0.6rem !important; /* Start with a more reasonable size */
        }
        """

        # 3. Get the output directory path from the existing settings instance
        output_dir_from_settings = settings_instance.get("output_dir")
        allowed_paths = [output_dir_from_settings]
        print(f"Gradio server will be allowed to access path: {output_dir_from_settings}")

        # 4. Create the Gradio interface
        with gr.Blocks(title="FramePack Toolbox (Standalone)", css=css) as block:
            gr.Markdown("#  FramePack Post-processing Toolbox (Standalone Mode)")
            gr.Markdown(
                "This is the standalone version of the toolbox. "
                "Upload a video to the 'Upload Video' component to begin. "
                "The 'Unload Studio Model' button will have no effect in this mode."
            )
            
            tb_create_video_toolbox_ui()
        print(f"Launching Toolbox server. Access it at http://{args.server}:{args.port if args.port else 7860}")
        block.launch(
            server_name=args.server,
            
        # 5. Launch the Gradio app with all the configured arguments
            server_port=args.port,
            share=args.share,
            inbrowser=args.inbrowser,
            allowed_paths=allowed_paths
        )

    # Call the launch function
    launch_standalone()