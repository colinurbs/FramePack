import gradio as gr
import time
import datetime
import random
import json
import os
import shutil
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import io

from modules.version import APP_VERSION, APP_VERSION_DISPLAY

import subprocess
import itertools
import re
from collections import defaultdict
import imageio
import imageio.plugins.ffmpeg
import ffmpeg
from diffusers_helper.utils import generate_timestamp

from modules.video_queue import JobStatus, Job, JobType
from modules.prompt_handler import get_section_boundaries, get_quick_prompts, parse_timestamped_prompt
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.bucket_tools import find_nearest_bucket
from modules.pipelines.metadata_utils import create_metadata

from modules.toolbox_app import tb_create_video_toolbox_ui, tb_get_formatted_toolbar_stats

def create_interface(
    process_fn,
    monitor_fn,
    end_process_fn,
    update_queue_status_fn,
    load_lora_file_fn,
    job_queue,
    settings,
    default_prompt: str = '[1s: The person waves hello] [3s: The person jumps up and down] [5s: The person does a dance]',
    lora_names: list = [],
    lora_values: list = []
):
    """
    Create the Gradio interface for the video generation application

    Args:
        process_fn: Function to process a new job
        monitor_fn: Function to monitor an existing job
        end_process_fn: Function to cancel the current job
        update_queue_status_fn: Function to update the queue status display
        default_prompt: Default prompt text
        lora_names: List of loaded LoRA names

    Returns:
        Gradio Blocks interface
    """
    def is_video_model(model_type_value):
        return model_type_value in ["Video", "Video with Endframe", "Video F1"]

    # Get section boundaries and quick prompts
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()

    # --- Function to update queue stats (Moved earlier to resolve UnboundLocalError) ---
    def update_stats(*args): # Accept any arguments and ignore them
        # Get queue status data
        queue_status_data = update_queue_status_fn()
        
        # Get queue statistics for the toolbar display
        jobs = job_queue.get_all_jobs()
        
        # Count jobs by status
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
        
        # Format the queue stats display text
        queue_stats_text = f"<p style='margin:0;color:white;' class='toolbar-text'>Queue: {pending_count} | Running: {running_count} | Completed: {completed_count}</p>"
        
        return queue_status_data, queue_stats_text

    # Create the interface
    css = make_progress_bar_css()
    css += """
    .short-import-box, .short-import-box > div {
        min-height: 40px !important;
        height: 40px !important;
    }
    /* Image container styling - more aggressive approach */
    .contain-image, .contain-image > div, .contain-image > div > img {
        object-fit: contain !important;
    }

    #non-mirrored-video {
        transform: scaleX(-1) !important;
    }
    
    /* Target all images in the contain-image class and its children */
    .contain-image img,
    .contain-image > div > img,
    .contain-image * img {
        object-fit: contain !important;
        width: 100% !important;
        height: 60vh !important;
        max-height: 100% !important;
        max-width: 100% !important;
    }
    
    /* Additional selectors to override Gradio defaults */
    .gradio-container img,
    .gradio-container .svelte-1b5oq5x,
    .gradio-container [data-testid="image"] img {
        object-fit: contain !important;
    }
    
    /* Toolbar styling */
    #fixed-toolbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 1000;
        background: #333;
        color: #fff;
        padding: 0px 10px; /* Reduced top/bottom padding */
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive toolbar title */
    .toolbar-title {
        font-size: 1.4rem;
        margin: 0;
        color: white;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Toolbar Patreon link */
    .toolbar-patreon {
        margin: 0 0 0 20px;
        color: white;
        font-size: 0.9rem;
        white-space: nowrap;
        display: inline-block;
    }
    .toolbar-patreon a {
        color: white;
        text-decoration: none;
    }
    .toolbar-patreon a:hover {
        text-decoration: underline;
    }

    /* Toolbar Version number */
    .toolbar-version {
        margin: 0 15px; /* Space around version */
        color: white;
        font-size: 0.8rem;
        white-space: nowrap;
        display: inline-block;
    }
    
    /* Responsive design for screens */
    @media (max-width: 1147px) {
        .toolbar-patreon, .toolbar-version { /* Hide both on smaller screens */
            display: none;
        }
        .footer-patreon, .footer-version { /* Show both in footer on smaller screens */
            display: inline-block !important; /* Ensure they are shown */
        }
        #fixed-toolbar {
            gap: 4px !important; /* Reduce gap for screens <= 1024px */
        }
        #fixed-toolbar > div:first-child { /* Target the first gr.Column (Title) */
            min-width: fit-content !important; /* Override Python-set min-width */
            flex-shrink: 0 !important; /* Prevent title column from shrinking too much */
        }
    }
    
    @media (min-width: 1148px) {
        .footer-patreon, .footer-version { /* Hide both in footer on larger screens */
            display: none !important;
        }
    }
    
    @media (max-width: 768px) {
        .toolbar-title {
            font-size: 1.1rem;
            max-width: 150px;
        }
        #fixed-toolbar {
            padding: 3px 6px;
            gap: 4px;
        }
        .toolbar-text {
            font-size: 0.75rem;
        }
    }
    
    @media (max-width: 510px) {
        #toolbar-ram-col, #toolbar-vram-col, #toolbar-gpu-col {
            display: none !important;
        }
    }

    @media (max-width: 480px) {
        .toolbar-title {
            font-size: 1rem;
            max-width: 120px;
        }
        #fixed-toolbar {
            padding: 2px 4px;
            gap: 2px;
        }
        .toolbar-text {
            font-size: 0.7rem;
        }
    }
    
    /* Button styling */
    #toolbar-add-to-queue-btn button {
        font-size: 14px !important;
        padding: 4px 16px !important;
        height: 32px !important;
        min-width: 80px !important;
    }
    .narrow-button {
        min-width: 40px !important;
        width: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gr-button-primary {
        color: white;
    }
    
    /* Layout adjustments */
    body, .gradio-container {
        padding-top: 42px !important; /* Adjusted for new toolbar height (36px - 10px) */
    }
    
    @media (max-width: 848px) {
        body, .gradio-container {
            padding-top: 48px !important;
        }
    }
    
    @media (max-width: 768px) {
        body, .gradio-container {
            padding-top: 22px !important; /* Adjusted for new toolbar height (32px - 10px) */
        }
    }
    
    @media (max-width: 480px) {
        body, .gradio-container {
            padding-top: 18px !important; /* Adjusted for new toolbar height (28px - 10px) */
        }
    }
    
    /* control sizing for tb_input_video_component */    
    .video-size video {
        max-height: 60vh;
        min-height: 300px !important;
        object-fit: contain;
    }

    /* hide the gr.Video source selection bar for tb_input_video_component */
    #toolbox-video-player .source-selection {
        display: none !important;
    }

    /* Styling for Textboxes used as stat displays in the toolbar */
    .toolbar-stat-textbox {
        /* Reset Gradio Textbox defaults */
        border: none !important;
        background-color: transparent !important; /* Will be overridden by stat bar bg */
        box-shadow: none !important;
        /* padding: 0px 4px !important; /* Padding now on textarea for text, not container */
        min-height: 25px !important; 
        height: 25px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative; /* Needed for absolute positioning of the stat-bar div */
        overflow: hidden; /* To clip the stat-bar div */
        /* border-radius: 3px; /* Rounded corners for the container - REMOVED */
    }

    .toolbar-stat-textbox textarea { /* Target the actual textarea element */
        color: white !important;
        font-family: inherit !important; /* Inherit from parent */
        font-size: 0.75rem !important; /* Match .toolbar-text or desired */
        line-height: normal !important;   /* Allow natural line height for the font size */
        padding: 0px 4px !important;      /* Minimal horizontal padding for text */
        white-space: nowrap !important;
        overflow: hidden !important; /* Hide scrollbars if any */
        resize: none !important; /* Disable textarea resizing */
        text-align: center !important;  /* Ensure text centered*/
        /* min-height and height are removed, flex parent controls this */
        background-color: transparent !important; /* Make textarea background transparent */
        position: relative; /* Ensure text is on top of the bar */
        z-index: 2; /* Text on top */
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7); /* Add shadow for better readability */
        width: 100%; /* Ensure textarea takes full width for text-align to work as expected */
    }

    /* Styling for Stat Bar Graphs (background color moved to the main .toolbar-stat-textbox) */
    /* .toolbar-stat-textbox {
        position: relative; 
        background-color: #555 !important; 
        border-radius: 3px; /* This was a duplicate and also removed if present */
    } */

    .stat-bar {
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        background-color: #4CAF50; /* Default bar color (greenish) */
        width: var(--stat-percentage, 0%); /* Controlled by JS */
        z-index: 1; /* Bar is behind the text */
        transition: width 0.3s ease-out;
        border-radius: 3px; /* Apply to all corners of the bar itself */
    }

    /* Specific bar colors (optional) */
    #toolbar-ram-stat .stat-bar { background-color: #2196F3; } /* Blue for RAM */
    #toolbar-vram-stat .stat-bar { background-color: #FF9800; } /* Orange for VRAM */
    #toolbar-gpu-stat .stat-bar { background-color: #E91E63; } /* Pink for GPU */
    """

    # Get the theme from settings
    current_theme = settings.get("gradio_theme", "default") # Use default if not found
    block = gr.Blocks(css=css, title="FramePack Studio", theme=current_theme).queue()

    with block:
        with gr.Row(elem_id="fixed-toolbar"):
            with gr.Column(scale=0, min_width=400): # Title/Version/Patreon
                gr.HTML(f"""
                <div style="display: flex; align-items: center;">
                    <h1 class='toolbar-title'>FP Studio</h1>
                    <p class='toolbar-version'>{APP_VERSION_DISPLAY}</p>
                    <p class='toolbar-patreon'><a href='https://patreon.com/Colinu' target='_blank'>Support on Patreon</a></p>
                </div>
                """)
            # REMOVED: refresh_stats_btn - Toolbar refresh button is no longer needed
            # with gr.Column(scale=0, min_width=40):
            #     refresh_stats_btn = gr.Button("⟳", elem_id="refresh-stats-btn", elem_classes="narrow-button")  
            with gr.Column(scale=1, min_width=180): # Queue Stats
                queue_stats_display = gr.Markdown("<p style='margin:0;color:white;' class='toolbar-text'>Queue: 0 | Running: 0 | Completed: 0</p>")
                
            # --- System Stats Display - Single gr.Textbox per stat ---
            with gr.Column(scale=0, min_width=205, elem_id="toolbar-ram-col"): # RAM Column
                toolbar_ram_display_component = gr.Textbox(
                    value="RAM: N/A",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    elem_id="toolbar-ram-stat",
                    elem_classes="toolbar-stat-textbox"
                )
            with gr.Column(scale=0, min_width=160, elem_id="toolbar-vram-col"): # VRAM Column
                toolbar_vram_display_component = gr.Textbox(
                    value="VRAM: N/A",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    elem_id="toolbar-vram-stat",
                    elem_classes="toolbar-stat-textbox"
                    # Visibility controlled by tb_get_formatted_toolbar_stats
                )
            with gr.Column(scale=0, min_width=140, elem_id="toolbar-gpu-col"): # GPU Column
                toolbar_gpu_display_component = gr.Textbox(
                    value="GPU: N/A",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    elem_id="toolbar-gpu-stat",
                    elem_classes="toolbar-stat-textbox"
                    # Visibility controlled by tb_get_formatted_toolbar_stats
                )
            # --- End of System Stats Display ---
            
            # Removed old version_display column
            # --- End of Toolbar ---
            
        # Essential to capture main_tabs_component for later use by send_to_toolbox_btn
        with gr.Tabs(elem_id="main_tabs") as main_tabs_component:
            with gr.Tab("Generate", id="generate_tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_type = gr.Radio(
                            choices=[("Original", "Original"), ("Original with Endframe", "Original with Endframe"), ("F1", "F1"), ("Video", "Video"), ("Video with Endframe", "Video with Endframe"), ("Video F1", "Video F1"), ("Grid", "XY Plot")],
                            value="Original",
                            label="Generation Type"
                        )
                        with gr.Group():
                            with gr.Row("Resolution"):
                                resolutionW = gr.Slider(
                                    label="Width", minimum=128, maximum=768, value=640, step=32, 
                                    info="Nearest valid width will be used."
                                )
                                resolutionH = gr.Slider(
                                    label="Height", minimum=128, maximum=768, value=640, step=32, 
                                    info="Nearest valid height will be used."
                                )
                            resolution_text = gr.Markdown(value="<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: 640 x 640</div>", label="", show_label=False)
                        with gr.Group(visible=False) as xy_group:  # Default visibility: True
                            xy_plot_axis_options = {
                                # "type": [
                                #     "dropdown(checkboxGroup), textbox or number", 
                                #     "empty if textbox, dtype if number, [] if dropdown", 
                                #     "standard values", 
                                #     "True if multi axis - like prompt replace, False is only on one axis - like steps"
                                # ],
                                "Nothing": ["nothing", "", "", True],
                                "Model type": ["dropdown", ["Original", "F1"], ["Original", "F1"], False],
                                "End frame influence": ["number", "float", "0.05-0.95[3]", False],
                                "Latent type": ["dropdown", ["Black", "White", "Noise", "Green Screen"], ["Black", "Noise"], False],
                                "Prompt add": ["textbox", "", "", True],
                                "Prompt replace": ["textbox", "", "", True],
                                "Blend sections": ["number", "int", "3-7 [3]", False],
                                "Steps": ["number", "int", "15-30 [3]", False],
                                "Seed": ["number", "int", "1000-10000 [3]", False],
                                "Use teacache": ["dropdown", [True, False], [True, False], False],
                                "TeaCache steps": ["number", "int", "5-25 [3]", False],
                                "TeaCache rel_l1_thresh": ["number", "float", "0.01-0.3 [3]", False],
                                # "CFG": ["number", "float", "", False],
                                "Distilled CFG Scale": ["number", "float", "5-15 [3]", False],
                                # "RS": ["number", "float", "", False],
                                # "Use weighted embeddings": ["dropdown", [True, False], [True, False], False],
                            }
                            text_to_base_keys = {
                                "Model type": "model_type",
                                "End frame influence": "end_frame_strength_original",
                                "Latent type": "latent_type",
                                "Prompt add": "prompt",
                                "Prompt replace": "prompt",
                                "Blend sections": "blend_sections",
                                "Steps": "steps",
                                "Seed": "seed",
                                "Use teacache": "use_teacache",
                                "TeaCache steps":"teacache_num_steps",
                                "TeaCache rel_l1_thresh":"teacache_rel_l1_thresh",
                                "Latent window size": "latent_window_size",
                                # "CFG": "",
                                "Distilled CFG Scale": "gs",
                                # "RS": "",
                                # "Use weighted embeddings": "",
                            }

                            def xy_plot_parse_input(text):
                                text = text.strip()
                                if ',' in text:
                                    return [x.strip() for x in text.split(",")]
                                match = re.match(r'^\s*(-?\d*\.?\d*)\s*-\s*(-?\d*\.?\d*)\s*\[\s*(\d+)\s*\]$', text)
                                if match:
                                    start, end, count = map(float, match.groups())
                                    result = np.linspace(start, end, int(count))
                                    if np.allclose(result, np.round(result)):
                                        result = np.round(result).astype(int)
                                    return result.tolist()
                                return []
                            def xy_plot_axis_change(updated_value_type):
                                if xy_plot_axis_options[updated_value_type][0] == "textbox" or xy_plot_axis_options[updated_value_type][0] == "number":
                                    return gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2]), gr.update(visible=False, value=[], choices=[])
                                elif xy_plot_axis_options[updated_value_type][0] == "dropdown":
                                    return gr.update(visible=False), gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2], choices=xy_plot_axis_options[updated_value_type][1])
                                else:
                                    return gr.update(visible=False), gr.update(visible=False, value=[], choices=[])
                            def xy_plot_process(
                                    model_type, input_image, end_frame_image_original, 
                                    end_frame_strength_original, latent_type, 
                                    prompt, blend_sections, steps, total_second_length, 
                                    resolutionW, resolutionH, seed, randomize_seed, use_teacache, 
                                    teacache_num_steps, teacache_rel_l1_thresh, latent_window_size, 
                                    cfg, gs, rs, gpu_memory_preservation, mp4_crf, 
                                    axis_x_switch, axis_x_value_text, axis_x_value_dropdown, 
                                    axis_y_switch, axis_y_value_text, axis_y_value_dropdown, 
                                    axis_z_switch, axis_z_value_text, axis_z_value_dropdown,
                                    selected_loras,
                                    *lora_slider_values
                                    ):
                                # print(model_type, input_image, latent_type, 
                                #     prompt, blend_sections, steps, total_second_length, 
                                #     resolutionW, resolutionH, seed, randomize_seed, use_teacache, 
                                #     latent_window_size, cfg, gs, rs, gpu_memory_preservation, 
                                #     mp4_crf, 
                                #     axis_x_switch, axis_x_value_text, axis_x_value_dropdown, 
                                #     axis_y_switch, axis_y_value_text, axis_y_value_dropdown, 
                                #     axis_z_switch, axis_z_value_text, axis_z_value_dropdown, sep=", ")
                                if axis_x_switch == "Nothing" and axis_y_switch == "Nothing" and axis_z_switch == "Nothing":
                                    return "Not selected any axis for plot", gr.update()
                                if (axis_x_switch == "Nothing" or axis_y_switch == "Nothing") and axis_z_switch != "Nothing":
                                    return "For using Z axis, first use X and Y axis", gr.update()
                                if axis_x_switch == "Nothing" and axis_y_switch != "Nothing":
                                    return "For using Y axis, first use X axis", gr.update()
                                if xy_plot_axis_options[axis_x_switch][0] == "dropdown" and len(axis_x_value_dropdown) < 1:
                                    return "No values for axis X", gr.update()
                                if xy_plot_axis_options[axis_y_switch][0] == "dropdown" and len(axis_y_value_dropdown) < 1:
                                    return "No values for axis Y", gr.update()
                                if xy_plot_axis_options[axis_z_switch][0] == "dropdown" and len(axis_z_value_dropdown) < 1:
                                    return "No values for axis Z", gr.update()
                                if not xy_plot_axis_options[axis_x_switch][3]:
                                    if axis_x_switch == axis_y_switch: 
                                        return "Axis type on X and Y axis are same, you can't do that generation.<br>Multi axis supported only for \"Prompt add\" and \"Prompt replace\".", gr.update()
                                    if axis_x_switch == axis_z_switch: 
                                        return "Axis type on X and Z axis are same, you can't do that generation.<br>Multi axis supported only for \"Prompt add\" and \"Prompt replace\".", gr.update()
                                if not xy_plot_axis_options[axis_y_switch][3]:
                                    if axis_y_switch == axis_z_switch: 
                                        return "Axis type on Y and Z axis are same, you can't do that generation.<br>Multi axis supported only for \"Prompt add\" and \"Prompt replace\".", gr.update()

                                base_generator_vars = {
                                    "model_type": model_type,
                                    "input_image": input_image,
                                    "end_frame_image": None,
                                    "end_frame_strength": 1.0,
                                    "input_video": None,
                                    "end_frame_image_original": end_frame_image_original,
                                    "end_frame_strength_original": end_frame_strength_original,
                                    "prompt_text": prompt,
                                    "n_prompt": "",
                                    "seed": seed,
                                    "total_second_length": total_second_length,
                                    "latent_window_size": latent_window_size,
                                    "steps": steps,
                                    "cfg": cfg,
                                    "gs": gs,
                                    "rs": rs,
                                    "use_teacache": use_teacache,
                                    "teacache_num_steps": teacache_num_steps,
                                    "teacache_rel_l1_thresh": teacache_rel_l1_thresh,
                                    "has_input_image": True if input_image is not None else False,
                                    "save_metadata_checked": True,
                                    "blend_sections": blend_sections,
                                    "latent_type": latent_type,
                                    "selected_loras": selected_loras,
                                    "resolutionW": resolutionW,
                                    "resolutionH": resolutionH,
                                    "lora_loaded_names": lora_names,
                                    "lora_values": lora_slider_values
                                }

                                def xy_plot_convert_values(type, value_textbox, value_dropdown):
                                    retVal = []
                                    if type[0] == "dropdown":
                                        retVal = value_dropdown
                                    elif type[0] == "textbox":
                                        retVal = xy_plot_parse_input(value_textbox)
                                    elif type[0] == "number":
                                        if type[1] == "int":
                                            retVal = [int(float(x)) for x in xy_plot_parse_input(value_textbox)]
                                        else:
                                            retVal = [float(x) for x in xy_plot_parse_input(value_textbox)]
                                    return retVal
                                prompt_replace_initial_values = {}
                                all_axis_values = {
                                    axis_x_switch+" -> X": xy_plot_convert_values(xy_plot_axis_options[axis_x_switch], axis_x_value_text, axis_x_value_dropdown)
                                }
                                if axis_x_switch == "Prompt replace":
                                    prompt_replace_initial_values["X"] = all_axis_values[axis_x_switch+" -> X"][0]
                                    if prompt_replace_initial_values["X"] not in base_generator_vars["prompt"]:
                                        return "Prompt for replacing in X axis not present in generation prompt", gr.update()
                                if axis_y_switch != "Nothing":
                                    all_axis_values[axis_y_switch+" -> Y"] = xy_plot_convert_values(xy_plot_axis_options[axis_y_switch], axis_y_value_text, axis_y_value_dropdown)
                                    if axis_y_switch == "Prompt replace":
                                        prompt_replace_initial_values["Y"] = all_axis_values[axis_y_switch+" -> Y"][0]
                                        if prompt_replace_initial_values["Y"] not in base_generator_vars["prompt"]:
                                            return "Prompt for replacing in Y axis not present in generation prompt", gr.update()
                                if axis_z_switch != "Nothing":
                                    all_axis_values[axis_z_switch+" -> Z"] = xy_plot_convert_values(xy_plot_axis_options[axis_z_switch], axis_z_value_text, axis_z_value_dropdown)
                                    if axis_z_switch == "Prompt replace":
                                        prompt_replace_initial_values["Z"] = all_axis_values[axis_z_switch+" -> Z"][0]
                                        if prompt_replace_initial_values["Z"] not in base_generator_vars["prompt"]:
                                            return "Prompt for replacing in Z axis not present in generation prompt", gr.update()

                                active_axes = list(all_axis_values.keys())
                                value_lists = [all_axis_values[axis] for axis in active_axes]
                                output_generator_vars = []

                                combintion_plot = itertools.product(*value_lists)
                                for combo in combintion_plot:
                                    vars_copy = base_generator_vars.copy()
                                    for axis, value in zip(active_axes, combo):
                                        splitted_axis_name = axis.split(" -> ")
                                        if splitted_axis_name[0] == "Prompt add":
                                            vars_copy["prompt_text"] = vars_copy["prompt_text"] + " " + str(value)
                                        elif splitted_axis_name[0] == "Prompt replace":
                                            orig_copy_prompt_text = vars_copy["prompt_text"]
                                            vars_copy["prompt_text"] = orig_copy_prompt_text.replace(prompt_replace_initial_values[splitted_axis_name[1]], str(value))
                                        else:
                                            vars_copy[text_to_base_keys[splitted_axis_name[0]]] = value
                                        vars_copy[splitted_axis_name[1]+"_axis_on_plot"] = str(value)
                                    
                                    worker_params = {k: v for k, v in vars_copy.items() if k not in ["X_axis_on_plot", "Y_axis_on_plot", "Z_axis_on_plot"]}
                                    output_generator_vars.append(worker_params)
                                # print("----- BEFORE GENERATED VIDS VARS START -----")
                                # for v in output_generator_vars:
                                #     print(v)
                                # print("------ BEFORE GENERATED VIDS VARS END ------")

                                from modules.video_queue import JobType
                                job_queue.add_job(
                                    params=base_generator_vars,
                                    job_type=JobType.GRID,
                                    child_job_params_list=output_generator_vars
                                )
                                return "Grid job added to the queue.", gr.update(visible=False)
                                # print("----- GENERATED VIDS VARS START -----")
                                # for v in output_generator_vars:
                                #     print(v)
                                # print("------ GENERATED VIDS VARS END ------")

                                # -------------------------- connect with settings --------------------------
                                # Ensure settings is available in this scope or passed in.
                                # Assuming 'settings' object is available from create_interface's scope.
                                output_dir_setting = settings.get("output_dir", "outputs")
                                mp4_crf_setting = settings.get("mp4_crf", 16) # Default CRF if not in settings
                                # -------------------------- connect with settings --------------------------

                                font_path = None
                                common_font_names = ["DejaVuSans-Bold.ttf", "arial.ttf", "LiberationSans-Bold.ttf"]
                                for fp_name in common_font_names:
                                    try:
                                        ImageFont.truetype(fp_name, 10) # Try loading with a small size
                                        font_path = fp_name
                                        print(f"XY Plot: Using font '{font_path}' for labels.")
                                        break
                                    except OSError:
                                        pass # Font not found, try next
                                
                                if not font_path:
                                    print("XY Plot Warning: Could not find DejaVuSans-Bold, Arial, or LiberationSans-Bold. Text labels might use a basic Pillow font or not render optimally.")
                                    # Pillow might use a default bitmap font if path is invalid.

                                timestamp_generation = generate_timestamp()
                                output_path = os.path.join(output_dir_setting, f"{timestamp_generation}_grid_XY.mp4")
                                
                                has_y_axis = any('Y_axis_on_plot' in v for v in output_generator_vars)

                                x_labels_unique = []
                                y_labels_unique = []
                                video_grid_data = {}

                                for item in output_generator_vars:
                                    x_val = str(item['X_axis_on_plot'])
                                    y_val = str(item.get('Y_axis_on_plot', 'single_row'))
                                    if x_val not in x_labels_unique:
                                        x_labels_unique.append(x_val)
                                    if y_val not in y_labels_unique:
                                        y_labels_unique.append(y_val)
                                    video_grid_data[(y_val, x_val)] = item['result']

                                if not output_generator_vars or not video_grid_data:
                                    return "No videos generated for XY plot.", gr.update(visible=False)

                                first_video_path = next(iter(video_grid_data.values()), None)
                                if not first_video_path or not os.path.exists(first_video_path):
                                    return f"First video for grid base parameters not found: {first_video_path}", gr.update(visible=False)

                                default_fps = 10 # Fallback FPS
                                frame_height, frame_width = resolutionH, resolutionW # Use UI resolution as fallback
                                try:
                                    with imageio.get_reader(first_video_path, 'ffmpeg') as reader:
                                        meta_data = reader.get_meta_data()
                                        fps = meta_data.get('fps', default_fps)
                                        if reader.count_frames() > 0 : # Check if count_frames is valid
                                            first_frame_shape = reader.get_data(0).shape
                                            frame_height, frame_width = first_frame_shape[0], first_frame_shape[1]
                                        else: # Fallback if count_frames is 0 or not available
                                            print(f"Warning: Could not get frame dimensions from {first_video_path}, using UI resolution {frame_width}x{frame_height}")

                                except Exception as e:
                                    print(f"XY Plot Error: Could not read first video {first_video_path} for metadata: {e}. Using UI defaults.")
                                    fps = default_fps


                                num_cols = len(x_labels_unique)
                                num_rows = len(y_labels_unique)
                                if num_rows == 1 and y_labels_unique[0] == 'single_row':
                                    has_y_axis = False

                                video_readers = {}
                                min_total_frames = float('inf')
                                video_paths_for_readers = {}

                                for y_label_val in y_labels_unique:
                                    for x_label_val in x_labels_unique:
                                        path = video_grid_data.get((y_label_val, x_label_val))
                                        video_paths_for_readers[(y_label_val, x_label_val)] = path # Store for deferred opening

                                # Determine min_total_frames by checking all videos first
                                for (y_label_val, x_label_val), path in video_paths_for_readers.items():
                                    if not path or not os.path.exists(path):
                                        print(f"XY Plot Warning: Missing video for X={x_label_val}, Y={y_label_val}. Will use black frames.")
                                        # Assume it contributes 0 frames to min_total_frames effectively, or handle later
                                        continue
                                    try:
                                        with imageio.get_reader(path, 'ffmpeg') as r:
                                            n_frames = r.count_frames()
                                            if n_frames == float('inf') or n_frames == 0: # Handle streaming or unknown length
                                                duration = r.get_meta_data().get('duration')
                                                if duration: n_frames = int(duration * fps)
                                                else: n_frames = int(total_second_length * fps) # Fallback to UI length
                                            if n_frames < min_total_frames:
                                                min_total_frames = n_frames
                                    except Exception as e:
                                        print(f"XY Plot Error: Could not get frame count for {path}: {e}")
                                
                                if min_total_frames == float('inf') or min_total_frames == 0:
                                    min_total_frames = int(total_second_length * fps) # Fallback if no video had countable frames
                                    print(f"XY Plot Warning: Could not determine minimum frame count. Defaulting to {min_total_frames} based on UI video length.")
                                min_total_frames = int(min_total_frames)


                                font_size = 20 # Increased font size
                                text_color = (255, 255, 255)
                                box_color = (0, 0, 0, 180) # Slightly more opaque box
                                pil_font = None
                                if font_path: # font_path is the name found earlier e.g. "arial.ttf"
                                    try:
                                        pil_font = ImageFont.truetype(font_path, font_size)
                                    except Exception as e:
                                        print(f"XY Plot Warning: Error loading font '{font_path}' with Pillow: {e}. Text may not render well.")
                                if not pil_font: # Fallback if specific font failed or wasn't found
                                    try: pil_font = ImageFont.load_default() # Pillow's built-in basic font
                                    except: pass # Should not fail, but just in case
                                    print("XY Plot Warning: Using Pillow's default bitmap font for labels.")


                                def draw_text_on_frame(numpy_frame_array, text_str, is_x_label):
                                    if not pil_font or not text_str: return numpy_frame_array
                                    img_pil = Image.fromarray(numpy_frame_array)
                                    draw = ImageDraw.Draw(img_pil, "RGBA")
                                    
                                    try:
                                        text_bbox = draw.textbbox((0,0), text_str, font=pil_font)
                                        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                                    except Exception as e:
                                        print(f"XY Plot Warning: Could not get text dimensions for '{text_str}': {e}")
                                        return numpy_frame_array

                                    padding = 5
                                    if is_x_label: # Top-center for X labels
                                        x = (img_pil.width - text_w) // 2
                                        y = padding - text_bbox[1] # Adjust for precise vertical alignment from bbox
                                        rect_coords = [x - padding, padding, x + text_w + padding, padding + text_h + padding]
                                    else: # Center-left for Y labels
                                        x = padding
                                        y = (img_pil.height - text_h) // 2 - text_bbox[1] # Adjust for precise vertical alignment
                                        rect_coords = [padding, (img_pil.height - text_h) // 2 - padding, padding + text_w + padding, (img_pil.height + text_h) // 2 + padding]
                                    
                                    draw.rectangle(rect_coords, fill=box_color)
                                    draw.text((x, y), text_str, font=pil_font, fill=text_color)
                                    return np.array(img_pil.convert("RGB"))

                                black_frame_template = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                                try:
                                    with imageio.get_writer(output_path, 'ffmpeg', fps=fps, quality=1, macro_block_size=1) as writer: # Set quality to 1 (highest for imageio-ffmpeg)
                                        for frame_num in range(min_total_frames):
                                            grid_row_strips = []
                                            for y_idx, y_label_val in enumerate(y_labels_unique):
                                                current_row_frames = []
                                                for x_idx, x_label_val in enumerate(x_labels_unique):
                                                    reader_key = (y_label_val, x_label_val)
                                                    current_frame = None
                                                    
                                                    if reader_key not in video_readers: # Open reader on demand
                                                        path_to_open = video_paths_for_readers.get(reader_key)
                                                        if path_to_open and os.path.exists(path_to_open):
                                                            try:
                                                                video_readers[reader_key] = imageio.get_reader(path_to_open, 'ffmpeg')
                                                            except Exception as e:
                                                                print(f"XY Plot Error: Failed to open reader for {path_to_open}: {e}")
                                                                video_readers[reader_key] = None # Mark as failed
                                                        else:
                                                            video_readers[reader_key] = None # No path or not exists

                                                    active_reader = video_readers.get(reader_key)
                                                    if active_reader:
                                                        try:
                                                            current_frame = active_reader.get_data(frame_num)
                                                            # Ensure frame is correct shape/type
                                                            if current_frame.shape[0] != frame_height or current_frame.shape[1] != frame_width:
                                                                # Basic resize if needed, though ideally all inputs are consistent
                                                                pil_img_temp = Image.fromarray(current_frame).resize((frame_width, frame_height))
                                                                current_frame = np.array(pil_img_temp)
                                                            if current_frame.dtype != np.uint8:
                                                                current_frame = current_frame.astype(np.uint8)

                                                        except IndexError: # Frame out of bounds for this specific video
                                                            current_frame = black_frame_template.copy()
                                                        except Exception as e:
                                                            print(f"XY Plot Error: Reading frame {frame_num} from video for X={x_label_val}, Y={y_label_val}: {e}")
                                                            current_frame = black_frame_template.copy()
                                                    else: # Reader failed or no video
                                                        current_frame = black_frame_template.copy()

                                                    # Apply labels
                                                    if y_idx == 0: # Top row videos get X-axis labels
                                                        current_frame = draw_text_on_frame(current_frame, x_label_val, is_x_label=True)
                                                    if x_idx == 0 and has_y_axis: # First column videos get Y-axis labels
                                                        current_frame = draw_text_on_frame(current_frame, y_label_val, is_x_label=False)
                                                    
                                                    current_row_frames.append(current_frame)
                                                
                                                if current_row_frames:
                                                    grid_row_strips.append(np.hstack(current_row_frames))
                                            
                                            if grid_row_strips:
                                                final_frame_for_video = np.vstack(grid_row_strips)
                                                writer.append_data(final_frame_for_video)
                                            
                                            if frame_num % int(fps * 5) == 0: # Log every 5 seconds of video
                                                 print(f"XY Plot Grid: Assembled frame {frame_num + 1}/{min_total_frames}")
                                
                                except Exception as e:
                                    import traceback
                                    tb_str = traceback.format_exc()
                                    err_msg = f"XY Plot Error: Failed during imageio grid creation: {e}\nTraceback:\n{tb_str}"
                                    print(err_msg)
                                    return err_msg, gr.update(visible=False)
                                finally:
                                    for r in video_readers.values():
                                        if r and hasattr(r, 'close'):
                                            r.close()
                                
                                print(f"XY Plot grid video successfully saved to: {output_path}")
                                return "", gr.update(value=output_path, visible=True)

                            with gr.Row():
                                xy_plot_model_type = gr.Radio(
                                    ["Original", "F1"], 
                                    label="Model Type", 
                                    value="F1",
                                    info="Select which model to use for generation"
                                )
                            with gr.Group():
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        xy_plot_input_image = gr.Image(
                                            sources='upload',
                                            type="numpy",
                                            label="Image (optional)",
                                            height=420,
                                            image_mode="RGB",
                                            elem_classes="contain-image"
                                        )
                                    with gr.Column(scale=1):
                                        xy_plot_end_frame_image_original = gr.Image(
                                            sources='upload',
                                            type="numpy",
                                            label="End Frame (Optional)", 
                                            height=420, 
                                            elem_classes="contain-image",
                                            image_mode="RGB",
                                            show_download_button=False,
                                            show_label=True,
                                            container=True
                                        )
                                with gr.Group():
                                    xy_plot_end_frame_strength_original = gr.Slider(
                                        label="End Frame Influence",
                                        minimum=0.05,
                                        maximum=1.0,
                                        value=1.0,
                                        step=0.05,
                                        info="Controls how strongly the end frame guides the generation. 1.0 is full influence."
                                    )
                            with gr.Accordion("Latent Image Options", open=False):
                                xy_plot_latent_type = gr.Dropdown(
                                    ["Black", "White", "Noise", "Green Screen"], 
                                    label="Latent Image", 
                                    value="Black", 
                                    info="Used as a starting point if no image is provided"
                                )
                            xy_plot_prompt = gr.Textbox(label="Prompt", value=default_prompt)
                            with gr.Accordion("Prompt Parameters", open=False):
                                xy_plot_blend_sections = gr.Slider(
                                    minimum=0, maximum=10, value=4, step=1,
                                    label="Number of sections to blend between prompts"
                                )
                            with gr.Accordion("Generation Parameters", open=True):
                                with gr.Row():
                                    xy_plot_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=5, step=1)
                                    xy_plot_total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=0.1, maximum=120, value=1, step=0.1)
                                with gr.Row():
                                    xy_plot_seed = gr.Number(label="Seed", value=31337, precision=0)
                                    xy_plot_randomize_seed = gr.Checkbox(label="Randomize", value=False, info="Generate a new random seed for each job")
                                with gr.Row("LoRAs"):
                                    xy_plot_lora_selector = gr.Dropdown(
                                        choices=lora_names,
                                        label="Select LoRAs to Load",
                                        multiselect=True,
                                        value=[],
                                        info="Select one or more LoRAs to use for this job"
                                    )
                                    xy_plot_lora_names_states = gr.State(lora_names)
                                    xy_plot_lora_sliders = {}
                                    for lora in lora_names:
                                        xy_plot_lora_sliders[lora] = gr.Slider(
                                            minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                            label=f"{lora} Weight", visible=False, interactive=True
                                        )
                            with gr.Accordion("Advanced Parameters", open=False):
                                with gr.Row("TeaCache"):
                                    xy_plot_use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                                    xy_plot_teacache_num_steps = gr.Slider(label="TeaCache steps", minimum=1, maximum=50, step=1, value=25, visible=True, info='How many intermediate sections to keep in the cache')
                                    xy_plot_teacache_rel_l1_thresh = gr.Slider(label="TeaCache rel_l1_thresh", minimum=0.01, maximum=1.0, step=0.01, value=0.15, visible=True, info='Relative L1 Threshold')
                                    xy_plot_use_teacache.change(lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)), inputs=xy_plot_use_teacache, outputs=[xy_plot_teacache_num_steps, xy_plot_teacache_rel_l1_thresh])

                                xy_plot_latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')  # Should not change
                                xy_plot_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                                xy_plot_gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                                xy_plot_rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change
                                xy_plot_gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=1, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                            with gr.Accordion("Output Parameters", open=False):
                                xy_plot_mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                            with gr.Accordion("Plot Parameters", open=True):
                                def xy_plot_axis_change(updated_value_type):
                                    if xy_plot_axis_options[updated_value_type][0] == "textbox" or xy_plot_axis_options[updated_value_type][0] == "number":
                                        return gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2]), gr.update(visible=False, value=[], choices=[])
                                    elif xy_plot_axis_options[updated_value_type][0] == "dropdown":
                                        return gr.update(visible=False), gr.update(visible=True, value=xy_plot_axis_options[updated_value_type][2], choices=xy_plot_axis_options[updated_value_type][1])
                                    else:
                                        return gr.update(visible=False), gr.update(visible=False, value=[], choices=[])
                                with gr.Row():
                                    xy_plot_axis_x_switch = gr.Dropdown(label="X axis type for plotting", choices=list(xy_plot_axis_options.keys()))
                                    xy_plot_axis_x_value_text = gr.Textbox(label="X axis comma separated text", visible=False)
                                    xy_plot_axis_x_value_dropdown = gr.CheckboxGroup(label="X axis values", visible=False) #, multiselect=True)
                                    xy_plot_axis_x_switch.change(fn=xy_plot_axis_change, inputs=[xy_plot_axis_x_switch], outputs=[xy_plot_axis_x_value_text, xy_plot_axis_x_value_dropdown])
                                with gr.Row():
                                    xy_plot_axis_y_switch = gr.Dropdown(label="Y axis type for plotting", choices=list(xy_plot_axis_options.keys()))
                                    xy_plot_axis_y_value_text = gr.Textbox(label="Y axis comma separated text", visible=False)
                                    xy_plot_axis_y_value_dropdown = gr.CheckboxGroup(label="Y axis values", visible=False) #, multiselect=True)
                                    xy_plot_axis_y_switch.change(fn=xy_plot_axis_change, inputs=[xy_plot_axis_y_switch], outputs=[xy_plot_axis_y_value_text, xy_plot_axis_y_value_dropdown])
                                with gr.Row(visible=False): # not implemented Z axis
                                    xy_plot_axis_z_switch = gr.Dropdown(label="Z axis type for plotting", choices=list(xy_plot_axis_options.keys()))
                                    xy_plot_axis_z_value_text = gr.Textbox(label="Z axis comma separated text", visible=False)
                                    xy_plot_axis_z_value_dropdown = gr.CheckboxGroup(label="Z axis values", visible=False) #, multiselect=True)
                                    xy_plot_axis_z_switch.change(fn=xy_plot_axis_change, inputs=[xy_plot_axis_z_switch], outputs=[xy_plot_axis_z_value_text, xy_plot_axis_z_value_dropdown])

                            # xy_plot_result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                            xy_plot_status = gr.HTML("")
                            xy_plot_output = gr.Video(autoplay=True, loop=True, sources=[], height=256, visible=False) # or Gallery, but return need value=[paths] instead of value=video
                        with gr.Group(visible=True) as standard_generation_group:    # Default visibility: True because "Original" model is not "Video"
                            with gr.Group(visible=True) as image_input_group: # This group now only contains the start frame image
                                with gr.Row():
                                    with gr.Column(scale=1): # Start Frame Image Column
                                        input_image = gr.Image(
                                            sources='upload',
                                            type="numpy",
                                            label="Start Frame (optional)",
                                            elem_classes="contain-image",
                                            image_mode="RGB",
                                            show_download_button=False,
                                            show_label=True, # Keep label for clarity
                                            container=True
                                        )
                            
                            with gr.Group(visible=False) as video_input_group:
                                input_video = gr.Video(
                                    sources='upload',
                                    label="Video Input",
                                    height=420,
                                    show_label=True
                                )
                                combine_with_source = gr.Checkbox(
                                    label="Combine with source video",
                                    value=True,
                                    info="If checked, the source video will be combined with the generated video",
                                    interactive=True
                                )
                                num_cleaned_frames = gr.Slider(label="Number of Context Frames (Adherence to Video)", minimum=2, maximum=10, value=5, step=1, interactive=True, info="Expensive. Retain more video details. Reduce if memory issues or motion too restricted (jumpcut, ignoring prompt, still).")

                            
                            # End Frame Image Input
                            # Initial visibility is False, controlled by update_input_visibility
                            with gr.Column(scale=1, visible=False) as end_frame_group_original:
                                end_frame_image_original = gr.Image(
                                    sources='upload',
                                    type="numpy",
                                    label="End Frame (Optional)", 
                                    elem_classes="contain-image",
                                    image_mode="RGB",
                                    show_download_button=False,
                                    show_label=True,
                                    container=True
                                )
                            
                            # End Frame Influence slider
                            # Initial visibility is False, controlled by update_input_visibility
                            with gr.Group(visible=False) as end_frame_slider_group:
                                end_frame_strength_original = gr.Slider(
                                    label="End Frame Influence",
                                    minimum=0.05,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.05,
                                    info="Controls how strongly the end frame guides the generation. 1.0 is full influence."
                                )

                            with gr.Accordion("Latent Image Options", open=False):
                                latent_type = gr.Dropdown(
                                    ["Black", "White", "Noise", "Green Screen"], label="Latent Image", value="Black", info="Used as a starting point if no image is provided"
                                )

                            prompt = gr.Textbox(label="Prompt", value=default_prompt)

                            with gr.Accordion("Prompt Parameters", open=False):
                                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True)  # Make visible for both models

                                blend_sections = gr.Slider(
                                    minimum=0, maximum=10, value=4, step=1,
                                    label="Number of sections to blend between prompts"
                                )
                            with gr.Accordion("Generation Parameters", open=True):
                                with gr.Accordion("Presets", open=True):
                                    with gr.Row():
                                        preset_dropdown = gr.Dropdown(label="Select Preset", choices=[], interactive=True)
                                        load_preset_button = gr.Button("Load")
                                        delete_preset_button = gr.Button("Delete", variant="stop")
                                    with gr.Row():
                                        preset_name_textbox = gr.Textbox(label="Preset Name", placeholder="Enter a name for your preset")
                                        save_preset_button = gr.Button("Save", variant="primary")
                                    with gr.Row(visible=False) as confirm_delete_row:
                                        gr.Markdown("### Are you sure you want to delete this preset?")
                                        confirm_delete_yes_btn = gr.Button("Yes, Delete", variant="stop")
                                        confirm_delete_no_btn = gr.Button("No, Go Back")
                                with gr.Row():
                                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                                    total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=1, maximum=120, value=6, step=0.1)
                                def on_input_image_change(img):
                                    if img is not None:
                                        return gr.update(info="Nearest valid bucket size will be used. Height will be adjusted automatically."), gr.update(visible=False)
                                    else:
                                        return gr.update(info="Nearest valid width will be used."), gr.update(visible=True)
                                input_image.change(fn=on_input_image_change, inputs=[input_image], outputs=[resolutionW, resolutionH])
                                def on_resolution_change(img, resolutionW, resolutionH):
                                    out_bucket_resH, out_bucket_resW = [640, 640]
                                    if img is not None:
                                        H, W, _ = img.shape
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(H, W, resolution=resolutionW)
                                    else:
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2) # if resolutionW > resolutionH else resolutionH
                                    return gr.update(value=f"<div style='text-align:right; padding:5px 15px 5px 5px;'>Selected bucket for resolution: {out_bucket_resW} x {out_bucket_resH}</div>")
                                resolutionW.change(fn=on_resolution_change, inputs=[input_image, resolutionW, resolutionH], outputs=[resolution_text], show_progress="hidden")
                                resolutionH.change(fn=on_resolution_change, inputs=[input_image, resolutionW, resolutionH], outputs=[resolution_text], show_progress="hidden")
                                with gr.Row("LoRAs"):
                                    lora_selector = gr.Dropdown(
                                        choices=lora_names,
                                        label="Select LoRAs to Load",
                                        multiselect=True,
                                        value=[],
                                        info="Select one or more LoRAs to use for this job"
                                    )
                                    lora_names_states = gr.State(lora_names)
                                    lora_sliders = {}
                                    for lora in lora_names:
                                        lora_sliders[lora] = gr.Slider(
                                            minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                            label=f"{lora} Weight", visible=False, interactive=True
                                        )

                                with gr.Row("Metadata"):
                                    json_upload = gr.File(
                                        label="Upload Metadata JSON (optional)",
                                        file_types=[".json"],
                                        type="filepath",
                                        height=100,
                                    )
                                with gr.Row("TeaCache"):
                                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                                    teacache_num_steps = gr.Slider(label="TeaCache steps", minimum=1, maximum=50, step=1, value=25, visible=True, info='How many intermediate sections to keep in the cache')
                                    teacache_rel_l1_thresh = gr.Slider(label="TeaCache rel_l1_thresh", minimum=0.01, maximum=1.0, step=0.01, value=0.15, visible=True, info='Relative L1 Threshold')
                                    use_teacache.change(lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)), inputs=use_teacache, outputs=[teacache_num_steps, teacache_rel_l1_thresh])

                                with gr.Row():
                                    seed = gr.Number(label="Seed", value=2500, precision=0)
                                    randomize_seed = gr.Checkbox(label="Randomize", value=True, info="Generate a new random seed for each job")

                            with gr.Accordion("Advanced Parameters", open=False):
                                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')  # Should not change
                                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                    with gr.Column():
                        preview_image = gr.Image(
                            label="Next Latents", 
                            height=150, 
                            visible=True, 
                            type="numpy", 
                            interactive=False,
                            elem_classes="contain-image",
                            image_mode="RGB"
                        )
                        result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                        with gr.Row():
                            current_job_id = gr.Textbox(label="Current Job ID", value="", visible=True, interactive=True)
                            start_button = gr.Button(value="Add to Queue", variant="primary", elem_id="toolbar-add-to-queue-btn")
                            xy_plot_process_btn = gr.Button("Submit", visible=False)
                            video_input_required_message = gr.Markdown(
                                "<p style='color: red; text-align: center;'>Input video required</p>", visible=False
                            )
                            end_button = gr.Button(value="Cancel Current Job", interactive=True, visible=False)

           

            with gr.Tab("Queue"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row() as queue_controls_row:
                            refresh_button = gr.Button("Refresh Queue")
                            clear_queue_button = gr.Button("Cancel Queue", variant="stop")
                            clear_complete_button = gr.Button("Clear Complete", variant="secondary")
                            load_queue_button = gr.Button("Load Queue")
                            queue_export_button = gr.Button("Export Queue")
                            import_queue_file = gr.File(
                                label="Import Queue",
                                file_types=[".json", ".zip"],
                                type="filepath",
                                visible=True,
                                elem_classes="short-import-box"
                            )
                        
                        with gr.Row(visible=False) as confirm_cancel_row:
                            gr.Markdown("### Are you sure you want to cancel all pending jobs?")
                            confirm_cancel_yes_btn = gr.Button("Yes, Cancel All", variant="stop")
                            confirm_cancel_no_btn = gr.Button("No, Go Back")

                        with gr.Row():
                            queue_status = gr.DataFrame(
                                headers=["Job ID", "Type", "Status", "Created", "Started", "Completed", "Elapsed", "Preview"], 
                                datatype=["str", "str", "str", "str", "str", "str", "str", "html"], 
                                label="Job Queue"
                            )

                        with gr.Accordion("Queue Documentation", open=False):
                            gr.Markdown("""
                            ## Queue Tab Guide
                            
                            This tab is for managing your generation jobs.
                            
                            - **Refresh Queue**: Update the job list.
                            - **Cancel Queue**: Stop all pending jobs.
                            - **Clear Complete**: Remove finished, failed, or cancelled jobs from the list.
                            - **Load Queue**: Load jobs from the default `queue.json`.
                            - **Export Queue**: Save the current job list and its images to a zip file.
                            - **Import Queue**: Load a queue from a `.json` or `.zip` file.
                            """)
                        
                        # --- Event Handlers for Queue Tab ---

                        # Function to clear all jobs in the queue
                        def clear_all_jobs():
                            try:
                                cancelled_count = job_queue.clear_queue()
                                print(f"Cleared {cancelled_count} jobs from the queue")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error in clear_all_jobs: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to clear completed and cancelled jobs
                        def clear_completed_jobs():
                            try:
                                removed_count = job_queue.clear_completed_jobs()
                                print(f"Removed {removed_count} completed/cancelled jobs from the queue")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error in clear_completed_jobs: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to load queue from queue.json
                        def load_queue_from_json():
                            try:
                                loaded_count = job_queue.load_queue_from_json()
                                print(f"Loaded {loaded_count} jobs from queue.json")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error loading queue from JSON: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to import queue from a custom JSON file
                        def import_queue_from_file(file_path):
                            if not file_path:
                                return update_stats()
                            try:
                                loaded_count = job_queue.load_queue_from_json(file_path)
                                print(f"Loaded {loaded_count} jobs from {file_path}")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error importing queue from file: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to export queue to a zip file
                        def export_queue_to_zip():
                            try:
                                zip_path = job_queue.export_queue_to_zip()
                                if zip_path and os.path.exists(zip_path):
                                    print(f"Queue exported to {zip_path}")
                                else:
                                    print("Failed to export queue to zip")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error exporting queue to zip: {e}")
                                traceback.print_exc()
                                return [], ""

                        # --- Connect Buttons ---
                        refresh_button.click(fn=update_stats, inputs=[], outputs=[queue_status, queue_stats_display])
                        
                        # Confirmation logic for Cancel Queue
                        def show_cancel_confirmation():
                            return gr.update(visible=False), gr.update(visible=True)

                        def hide_cancel_confirmation():
                            return gr.update(visible=True), gr.update(visible=False)

                        def confirmed_clear_all_jobs():
                            qs_data, qs_text = clear_all_jobs()
                            return qs_data, qs_text, gr.update(visible=True), gr.update(visible=False)

                        clear_queue_button.click(fn=show_cancel_confirmation, inputs=None, outputs=[queue_controls_row, confirm_cancel_row])
                        confirm_cancel_no_btn.click(fn=hide_cancel_confirmation, inputs=None, outputs=[queue_controls_row, confirm_cancel_row])
                        confirm_cancel_yes_btn.click(fn=confirmed_clear_all_jobs, inputs=None, outputs=[queue_status, queue_stats_display, queue_controls_row, confirm_cancel_row])

                        clear_complete_button.click(fn=clear_completed_jobs, inputs=[], outputs=[queue_status, queue_stats_display])
                        queue_export_button.click(fn=export_queue_to_zip, inputs=[], outputs=[queue_status, queue_stats_display])

                        # Create a container for thumbnails (kept for potential future use, though not displayed in DataFrame)
                        with gr.Row():
                            thumbnail_container = gr.Column()
                            thumbnail_container.elem_classes = ["thumbnail-container"]

                        # Add CSS for thumbnails
            with gr.Tab("Outputs", id="outputs_tab"): # Ensure 'id' is present for tab switching
                outputDirectory_video = settings.get("output_dir", settings.default_settings['output_dir'])
                outputDirectory_metadata = settings.get("metadata_dir", settings.default_settings['metadata_dir'])
                def get_gallery_items():
                    items = []
                    for f in os.listdir(outputDirectory_metadata):
                        if f.endswith(".png"):
                            prefix = os.path.splitext(f)[0]
                            latest_video = get_latest_video_version(prefix)
                            if latest_video:
                                video_path = os.path.join(outputDirectory_video, latest_video)
                                mtime = os.path.getmtime(video_path)
                                preview_path = os.path.join(outputDirectory_metadata, f)
                                items.append((preview_path, prefix, mtime))
                    items.sort(key=lambda x: x[2], reverse=True)
                    return [(i[0], i[1]) for i in items]
                def get_latest_video_version(prefix):
                    max_number = -1
                    selected_file = None
                    for f in os.listdir(outputDirectory_video):
                        if f.startswith(prefix + "_") and f.endswith(".mp4"):
                            num = int(f.replace(prefix + "_", '').replace(".mp4", ''))
                            if num > max_number:
                                max_number = num
                                selected_file = f
                    return selected_file
                # load_video_and_info_from_prefix now also returns button visibility
                def load_video_and_info_from_prefix(prefix):
                    video_file = get_latest_video_version(prefix)
                    json_path = os.path.join(outputDirectory_metadata, prefix) + ".json"
                    
                    if not video_file or not os.path.exists(os.path.join(outputDirectory_video, video_file)) or not os.path.exists(json_path):
                        # If video or info not found, button should be hidden
                        return None, "Video or JSON not found.", gr.update(visible=False) 

                    video_path = os.path.join(outputDirectory_video, video_file)
                    info_content = {"description": "no info"}
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            info_content = json.load(f)
                    # If video and info found, button should be visible
                    return video_path, json.dumps(info_content, indent=2, ensure_ascii=False), gr.update(visible=True)

                gallery_items_state = gr.State(get_gallery_items())
                selected_original_video_path_state = gr.State(None) # Holds the ORIGINAL, UNPROCESSED path
                with gr.Row():
                    with gr.Column(scale=2):
                        thumbs = gr.Gallery(
                            # value=[i[0] for i in get_gallery_items()],
                            columns=[4],
                            allow_preview=False,
                            object_fit="cover",
                            height="auto"
                        )
                        refresh_button = gr.Button("Update")
                    with gr.Column(scale=5):
                        video_out = gr.Video(sources=[], autoplay=True, loop=True, visible=False)
                    with gr.Column(scale=1):
                        info_out = gr.Textbox(label="Generation info", visible=False)
                        send_to_toolbox_btn = gr.Button("➡️ Send to Post-processing", visible=False)  # Added new send_to_toolbox_btn
                    def refresh_gallery():
                        new_items = get_gallery_items()
                        return gr.update(value=[i[0] for i in new_items]), new_items
                    refresh_button.click(fn=refresh_gallery, outputs=[thumbs, gallery_items_state])
                    
                    # MODIFIED: on_select now handles visibility of the new button
                    def on_select(evt: gr.SelectData, gallery_items):
                        if evt.index is None or not gallery_items or evt.index >= len(gallery_items):
                            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None

                        prefix = gallery_items[evt.index][1]
                        # original_video_path is e.g., "outputs/my_actual_video.mp4"
                        original_video_path, info_string, button_visibility_update = load_video_and_info_from_prefix(prefix)

                        # Determine visibility for video and info based on whether video_path was found
                        video_out_update = gr.update(value=original_video_path, visible=bool(original_video_path))
                        info_out_update = gr.update(value=info_string, visible=bool(original_video_path))

                        # IMPORTANT: Store the ORIGINAL, UNPROCESSED path in the gr.State
                        return video_out_update, info_out_update, button_visibility_update, original_video_path

                    thumbs.select(
                        fn=on_select,
                        inputs=[gallery_items_state],
                        outputs=[video_out, info_out, send_to_toolbox_btn, selected_original_video_path_state] # Output original path to State
                    )
            with gr.Tab("Post-processing", id="toolbox_tab"):          
                # Call the function from toolbox_app.py to build the Toolbox UI
                # The toolbox_ui_layout (e.g., a gr.Column) is automatically placed here.                
                toolbox_ui_layout, tb_target_video_input = tb_create_video_toolbox_ui()
                
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        save_metadata = gr.Checkbox(
                            label="Save Metadata", 
                            info="Save to JSON file", 
                            value=settings.get("save_metadata", 6),
                        )
                        gpu_memory_preservation = gr.Slider(
                            label="GPU Inference Preserved Memory (GB) (larger means slower)",
                            minimum=1,
                            maximum=128,
                            step=0.1,
                            value=settings.get("gpu_memory_preservation", 6),
                            info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed."
                        )
                        mp4_crf = gr.Slider(
                            label="MP4 Compression",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=settings.get("mp4_crf", 16),
                            info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs."
                        )
                        clean_up_videos = gr.Checkbox(
                            label="Clean up video files",
                            value=settings.get("clean_up_videos", True),
                            info="If checked, only the final video will be kept after generation."
                        )
                        cleanup_temp_folder = gr.Checkbox(
                            label="Clean up temp folder after generation",
                            visible=False,
                            value=settings.get("cleanup_temp_folder", True),
                            info="If checked, temporary files will be cleaned up after each generation."
                        )
                        
                        with gr.Accordion("System Prompt", open=False):
                            override_system_prompt = gr.Checkbox(
                                label="Override System Prompt",
                                value=settings.get("override_system_prompt", False),
                                info="If checked, the system prompt template below will be used instead of the default one."
                            )
                            system_prompt_template = gr.Textbox(
                                label="System Prompt Template",
                                value=settings.get("system_prompt_template", "{\"template\": \"<|start_header_id|>system<|end_header_id|>\\n\\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{}<|eot_id|>\", \"crop_start\": 95}"),
                                lines=10,
                                info="System prompt template used for video generation. Must be a valid JSON or Python dictionary string with 'template' and 'crop_start' keys. Example: {\"template\": \"your template here\", \"crop_start\": 95}"
                            )
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            value=settings.get("output_dir"),
                            placeholder="Path to save generated videos"
                        )
                        metadata_dir = gr.Textbox(
                            label="Metadata Directory",
                            value=settings.get("metadata_dir"),
                            placeholder="Path to save metadata files"
                        )
                        lora_dir = gr.Textbox(
                            label="LoRA Directory",
                            value=settings.get("lora_dir"),
                            placeholder="Path to LoRA models"
                        )
                        gradio_temp_dir = gr.Textbox(label="Gradio Temporary Directory", value=settings.get("gradio_temp_dir"))
                        auto_save = gr.Checkbox(
                            label="Auto-save settings",
                            value=settings.get("auto_save_settings", True)
                        )
                        # Add Gradio Theme Dropdown
                        gradio_themes = ["default", "base", "soft", "glass", "mono", "huggingface"]
                        theme_dropdown = gr.Dropdown(
                            label="Theme",
                            choices=gradio_themes,
                            value=settings.get("gradio_theme", "soft"),
                            info="Select the Gradio UI theme. Requires restart."
                        )
                        save_btn = gr.Button("Save Settings")
                        cleanup_btn = gr.Button("Clean Up Temporary Files")
                        status = gr.HTML("")
                        cleanup_output = gr.Textbox(label="Cleanup Status", interactive=False)

                        def save_settings(save_metadata, gpu_memory_preservation, mp4_crf, clean_up_videos, cleanup_temp_folder, override_system_prompt_value, system_prompt_template_value, output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, selected_theme):
                            try:
                                # Save the system prompt template as is, without trying to parse it
                                # The hunyuan.py file will handle parsing it when needed
                                processed_template = system_prompt_template_value
                                
                                settings.save_settings(
                                    save_metadata=save_metadata,
                                    gpu_memory_preservation=gpu_memory_preservation,
                                    mp4_crf=mp4_crf,
                                    clean_up_videos=clean_up_videos,
                                    cleanup_temp_folder=cleanup_temp_folder,
                                    override_system_prompt=override_system_prompt_value,
                                    system_prompt_template=processed_template,
                                    output_dir=output_dir,
                                    metadata_dir=metadata_dir,
                                    lora_dir=lora_dir,
                                    gradio_temp_dir=gradio_temp_dir,
                                    auto_save_settings=auto_save,
                                    gradio_theme=selected_theme
                                )
                                return "<p style='color:green;'>Settings saved successfully! Restart required for theme change.</p>"
                            except Exception as e:
                                return f"<p style='color:red;'>Error saving settings: {str(e)}</p>"

                        save_btn.click(
                            fn=save_settings,
                            inputs=[save_metadata, gpu_memory_preservation, mp4_crf, clean_up_videos, cleanup_temp_folder, override_system_prompt, system_prompt_template, output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, theme_dropdown],
                            outputs=[status]
                        )

                        def cleanup_temp_files():
                            """Clean up temporary files and folders in the Gradio temp directory"""
                            temp_dir = settings.get("gradio_temp_dir")
                            if not temp_dir or not os.path.exists(temp_dir):
                                return "No temporary directory found or directory does not exist."
                            
                            try:
                                # Get all items in the temp directory
                                items = os.listdir(temp_dir)
                                removed_count = 0
                                print(f"Finding items in {temp_dir}")
                                for item in items:
                                    item_path = os.path.join(temp_dir, item)
                                    try:
                                        if os.path.isfile(item_path) or os.path.islink(item_path):
                                            print(f"Removing {item_path}")
                                            os.remove(item_path)
                                            removed_count += 1
                                        elif os.path.isdir(item_path):
                                            print(f"Removing directory {item_path}")
                                            shutil.rmtree(item_path)
                                            removed_count += 1
                                    except Exception as e:
                                        print(f"Error removing {item_path}: {e}")
                                
                                return f"Cleaned up {removed_count} temporary files/folders."
                            except Exception as e:
                                return f"Error cleaning up temporary files: {str(e)}"

        # --- Event Handlers and Connections (Now correctly indented) ---

        # --- Connect Monitoring ---
        # Auto-check for current job on page load and job change
        def check_for_current_job():
            # This function will be called when the interface loads
            # It will check if there's a current job in the queue and update the UI
            with job_queue.lock:
                current_job = job_queue.current_job
                if current_job:
                    # Return all the necessary information to update the preview windows
                    job_id = current_job.id
                    result = current_job.result
                    preview = current_job.progress_data.get('preview') if current_job.progress_data else None
                    desc = current_job.progress_data.get('desc', '') if current_job.progress_data else ''
                    html = current_job.progress_data.get('html', '') if current_job.progress_data else ''
                    
                    # Also trigger the monitor_job function to start monitoring this job
                    print(f"Auto-check found current job {job_id}, triggering monitor_job")
                    return job_id, result, preview, desc, html
                return None, None, None, '', ''
                
        # Auto-check for current job on page load and handle handoff between jobs.
        def check_for_current_job_and_monitor():
            # This function is now the key to the handoff.
            # It finds the current job and returns its ID, which will trigger the monitor.
            job_id, result, preview, desc, html = check_for_current_job()
            # We also need to get fresh stats at the same time.
            queue_status_data, queue_stats_text = update_stats()
            # Return everything needed to update the UI atomically.
            return job_id, result, preview, desc, html, queue_status_data, queue_stats_text

        # Connect the main process function (wrapper for adding to queue)
        def process_with_queue_update(model_type_arg, *args):
            # Call update_stats to get both queue_status_data and queue_stats_text
            queue_status_data, queue_stats_text = update_stats() # MODIFIED

            # Extract all arguments (ensure order matches inputs lists)
            # The order here MUST match the order in the `ips` list.
            # RT_BORG: Global settings gpu_memory_preservation, mp4_crf, save_metadata removed from direct args.
            (input_image_arg,
             input_video_arg,
             end_frame_image_original_arg,
             end_frame_strength_original_arg,
             prompt_text_arg,
             n_prompt_arg,
             seed_arg, # the seed value
             randomize_seed_arg, # the boolean value of the checkbox
             total_second_length_arg,
             latent_window_size_arg,
             steps_arg,
             cfg_arg, 
             gs_arg,
             rs_arg,
             use_teacache_arg,
             teacache_num_steps_arg,
             teacache_rel_l1_thresh_arg,
             blend_sections_arg,
             latent_type_arg,
             clean_up_videos_arg, # UI checkbox from Generate tab
             selected_loras_arg,
             resolutionW_arg, resolutionH_arg,
             combine_with_source_arg, 
             num_cleaned_frames_arg,
             lora_names_states_arg,   # This is from lora_names_states (gr.State)
             *lora_slider_values_tuple # Remaining args are LoRA slider values
            ) = args
            # DO NOT parse the prompt here. Parsing happens once in the worker.

            # Determine the model type to send to the backend
            backend_model_type = model_type_arg # model_type_arg is the UI selection
            if model_type_arg == "Video with Endframe":
                backend_model_type = "Video" # The backend "Video" model_type handles with and without endframe

            # Use the appropriate input based on model type
            is_ui_video_model = is_video_model(model_type_arg)
            input_data = input_video_arg if is_ui_video_model else input_image_arg

            # Define actual end_frame params to pass to backend
            actual_end_frame_image_for_backend = None
            actual_end_frame_strength_for_backend = 1.0  # Default strength

            if model_type_arg == "Original with Endframe" or model_type_arg == "F1 with Endframe" or model_type_arg == "Video with Endframe":
                actual_end_frame_image_for_backend = end_frame_image_original_arg
                actual_end_frame_strength_for_backend = end_frame_strength_original_arg

            # Get the input video path for Video model
            input_image_path = None
            if is_ui_video_model and input_video_arg is not None:
                # For Video models, input_video contains the path to the video file
                input_image_path = input_video_arg

            # Use the current seed value as is for this job
            # Call the process function with all arguments
            # Pass the backend_model_type and the ORIGINAL prompt_text string to the backend process function
            result = process_fn(backend_model_type, input_data, actual_end_frame_image_for_backend, actual_end_frame_strength_for_backend,
                                prompt_text_arg, n_prompt_arg, seed_arg, total_second_length_arg,
                                latent_window_size_arg, steps_arg, cfg_arg, gs_arg, rs_arg,
                                use_teacache_arg, teacache_num_steps_arg, teacache_rel_l1_thresh_arg,
                                blend_sections_arg, latent_type_arg, clean_up_videos_arg, # clean_up_videos_arg is from UI
                                selected_loras_arg, resolutionW_arg, resolutionH_arg, 
                                input_image_path, 
                                combine_with_source_arg,
                                num_cleaned_frames_arg,
                                lora_names_states_arg,
                                *lora_slider_values_tuple
                               )
            # If randomize_seed is checked, generate a new random seed for the next job
            new_seed_value = None
            if randomize_seed_arg:
                new_seed_value = random.randint(0, 21474)
                print(f"Generated new seed for next job: {new_seed_value}")

            # Create the button update for start_button WITHOUT interactive=True.
            # The interactivity will be set by update_start_button_state later in the chain.
            start_button_update_after_add = gr.update(value="Add to Queue")
            
            # If a job ID was created, automatically start monitoring it and update queue
            if result and result[1]:  # Check if job_id exists in results
                job_id = result[1]
                # queue_status_data = update_queue_status_fn() # OLD: update_stats now called earlier
                # Call update_stats again AFTER the job is added to get the freshest stats
                queue_status_data, queue_stats_text = update_stats()


                # Add the new seed value to the results if randomize is checked
                if new_seed_value is not None:
                    # Use result[6] directly for end_button to preserve its value. Add gr.update() for video_input_required_message.
                    return [result[0], job_id, result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, new_seed_value, gr.update()]
                else:
                    # Use result[6] directly for end_button to preserve its value. Add gr.update() for video_input_required_message.
                    return [result[0], job_id, result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, gr.update(), gr.update()]

            # If no job ID was created, still return the new seed if randomize is checked
            # Also, ensure we return the latest stats even if no job was created (e.g., error during param validation)
            queue_status_data, queue_stats_text = update_stats()
            if new_seed_value is not None:
                # Make sure to preserve the end_button update from result[6]
                return [result[0], result[1], result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, new_seed_value, gr.update()]
            else:
                # Make sure to preserve the end_button update from result[6]
                return [result[0], result[1], result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, gr.update(), gr.update()]

        # Custom end process function that ensures the queue is updated and changes button text
        def end_process_with_update():
            _ = end_process_fn() # Call the original end_process_fn
            # Now, get fresh stats for both queue table and toolbar
            queue_status_data, queue_stats_text = update_stats()
            
            # Don't try to get the new job ID immediately after cancellation
            # The monitor_job function will handle the transition to the next job
            
            # Change the cancel button text to "Cancelling..." and make it non-interactive
            # This ensures the button stays in this state until the job is fully cancelled
            return queue_status_data, queue_stats_text, gr.update(value="Cancelling...", interactive=False), gr.update(value=None)

        # MODIFIED handle_send_video_to_toolbox:
        def handle_send_video_to_toolbox(original_path_from_state): # Input is now the original path from gr.State
            print(f"Button clicked. Sending ORIGINAL video path (from State) to Post-processing: {original_path_from_state}")

            if original_path_from_state and isinstance(original_path_from_state, str) and os.path.exists(original_path_from_state):
                # tb_target_video_input will now process the ORIGINAL path (e.g., "outputs/my_actual_video.mp4").
                return gr.update(value=original_path_from_state), gr.update(selected="toolbox_tab")
            else:
                print(f"No valid original video path from State to send. Path: {original_path_from_state}")
                return gr.update(), gr.update()

        send_to_toolbox_btn.click(
            fn=handle_send_video_to_toolbox,
            inputs=[selected_original_video_path_state], # INPUT IS NOW THE gr.State holding the ORIGINAL path
            outputs=[
                tb_target_video_input, # This is tb_input_video_component from toolbox_app.py
                main_tabs_component
            ]
        )
        
        # --- Inputs Lists ---
        # --- Inputs for all models ---
        ips = [
            input_image,                # Corresponds to input_image_arg
            input_video,                # Corresponds to input_video_arg
            end_frame_image_original,   # Corresponds to end_frame_image_original_arg
            end_frame_strength_original,# Corresponds to end_frame_strength_original_arg
            prompt,                     # Corresponds to prompt_text_arg
            n_prompt,                   # Corresponds to n_prompt_arg
            seed,                       # Corresponds to seed_arg
            randomize_seed,             # Corresponds to randomize_seed_arg
            total_second_length,        # Corresponds to total_second_length_arg
            latent_window_size,         # Corresponds to latent_window_size_arg
            steps,                      # Corresponds to steps_arg
            cfg,                        # Corresponds to cfg_arg
            gs,                         # Corresponds to gs_arg
            rs,                         # Corresponds to rs_arg
            use_teacache,               # Corresponds to use_teacache_arg
            teacache_num_steps,         # Corresponds to teacache_num_steps_arg
            teacache_rel_l1_thresh,     # Corresponds to teacache_rel_l1_thresh_arg
            blend_sections,             # Corresponds to blend_sections_arg
            latent_type,                # Corresponds to latent_type_arg
            clean_up_videos,            # Corresponds to clean_up_videos_arg (UI checkbox)
            lora_selector,              # Corresponds to selected_loras_arg
            resolutionW,                # Corresponds to resolutionW_arg
            resolutionH,                # Corresponds to resolutionH_arg
            combine_with_source,        # Corresponds to combine_with_source_arg
            num_cleaned_frames,         # Corresponds to num_cleaned_frames_arg
            lora_names_states           # Corresponds to lora_names_states_arg
        ]
        # Add LoRA sliders to the input list
        ips.extend([lora_sliders[lora] for lora in lora_names])


        # --- Connect Buttons ---
        def handle_start_button(selected_model, *args):
            # For other model types, use the regular process function
            return process_with_queue_update(selected_model, *args)
                
        # Validation ensures the start button is only enabled when appropriate
        def update_start_button_state(*args):
            """
            Validation fails if a video model is selected and no input video is provided.
            Updates the start button interactivity and validation message visibility.
            Handles variable inputs from different Gradio event chains.
            """
            # The required values are the last two arguments provided by the Gradio event
            if len(args) >= 2:
                selected_model = args[-2]
                input_video_value = args[-1]
            else:
                # Fallback or error handling if not enough arguments are received
                # This might happen if the event is triggered in an unexpected way
                print(f"Warning: update_start_button_state received {len(args)} args, expected at least 2.")
                # Default to a safe state (button disabled)
                return gr.Button(value="Error", interactive=False), gr.update(visible=True)

            video_provided = input_video_value is not None
            
            if is_video_model(selected_model) and not video_provided:
                # Video model selected, but no video provided
                return gr.Button(value="Missing Video", interactive=False), gr.update(visible=True)
            else:
                # Either not a video model, or video model selected and video provided
                return gr.update(value="Add to Queue", interactive=True), gr.update(visible=False)
        # Function to update button state before processing
        def update_button_before_processing(selected_model, *args):
            # First update the button to show "Adding..." and disable it
            # Also return current stats so they don't get blanked out during the "Adding..." phase
            qs_data, qs_text = update_stats()
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value="Adding...", interactive=False), gr.update(), qs_data, qs_text, gr.update(), gr.update() # Added update for video_input_required_message
        
        # Connect the start button to first update its state
        start_button.click(
            fn=update_button_before_processing,
            inputs=[model_type] + ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, queue_stats_display, seed, video_input_required_message]
        ).then(
            # Then process the job
            fn=handle_start_button,
            inputs=[model_type] + ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, queue_stats_display, seed, video_input_required_message] # Added video_input_required_message
        ).then( # Ensure validation is re-checked after job processing completes
            fn=update_start_button_state,
            inputs=[model_type, input_video], # Current values of model_type and input_video
            outputs=[start_button, video_input_required_message]
        )

        xy_plot_inputs = [xy_plot_model_type, xy_plot_input_image, xy_plot_end_frame_image_original,
            xy_plot_end_frame_strength_original, xy_plot_latent_type, 
            xy_plot_prompt, xy_plot_blend_sections, xy_plot_steps, xy_plot_total_second_length, 
            resolutionW, resolutionH, xy_plot_seed, xy_plot_randomize_seed, 
            xy_plot_use_teacache, xy_plot_teacache_num_steps, xy_plot_teacache_rel_l1_thresh, 
            xy_plot_latent_window_size, xy_plot_cfg, xy_plot_gs, xy_plot_rs, 
            xy_plot_gpu_memory_preservation, xy_plot_mp4_crf, 
            xy_plot_axis_x_switch, xy_plot_axis_x_value_text, xy_plot_axis_x_value_dropdown, 
            xy_plot_axis_y_switch, xy_plot_axis_y_value_text, xy_plot_axis_y_value_dropdown, 
            xy_plot_axis_z_switch, xy_plot_axis_z_value_text, xy_plot_axis_z_value_dropdown,
            xy_plot_lora_selector
        ]
        xy_plot_inputs.extend(xy_plot_lora_sliders.values())
        xy_plot_process_btn.click(fn=xy_plot_process, inputs=xy_plot_inputs, outputs=[xy_plot_status, xy_plot_output]).then(
            fn=update_stats,
            outputs=[queue_status, queue_stats_display]
        ).then(
            fn=check_for_current_job,
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar]
        )
        
        def xy_plot_update_lora_sliders(selected_loras):
            updates = []
            for lora in lora_names:
                 updates.append(gr.update(visible=(lora in selected_loras)))
            num_sliders = len(xy_plot_lora_sliders)
            return updates[:num_sliders]

        xy_plot_lora_selector.change(
            fn=xy_plot_update_lora_sliders,
            inputs=[xy_plot_lora_selector],
            outputs=[xy_plot_lora_sliders[lora] for lora in lora_names]
        )

        #putting this here for now because this file is way too big
        def on_model_type_change(selected_model):
            is_xy_plot = selected_model == "XY Plot"
            shows_end_frame = selected_model in ["Original with Endframe", "Video with Endframe"] # F1 with Endframe is not a direct option

            return (
                gr.update(visible=not is_xy_plot),  # standard_generation_group
                gr.update(visible=is_xy_plot),      # xy_group
                gr.update(visible=not is_xy_plot and not is_video_model(selected_model)),  # image_input_group
                gr.update(visible=not is_xy_plot and is_video_model(selected_model)),      # video_input_group
                gr.update(visible=not is_xy_plot and shows_end_frame),     # end_frame_group_original
                gr.update(visible=not is_xy_plot and shows_end_frame),      # end_frame_slider_group
                gr.update(visible=not is_xy_plot),   # start_button
                gr.update(visible=is_xy_plot)    # xy_plot_process_btn
            )

        # Model change listener
        model_type.change(
            fn=on_model_type_change,
            inputs=model_type,
            outputs=[
                standard_generation_group, 
                xy_group,
                image_input_group,
                video_input_group,
                end_frame_group_original,
                end_frame_slider_group,
                start_button,
                xy_plot_process_btn
            ]
        ).then( # Also trigger validation after model type changes
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )
        
        # Connect input_video change to the validation function
        input_video.change(
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )
        # Also trigger validation when video is cleared
        input_video.clear(
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )

        

        # Auto-monitor the current job when job_id changes
        current_job_id.change(
            fn=monitor_fn,
            inputs=[current_job_id],
            outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
        ).then(
            fn=update_stats, # When a monitor finishes, always update the stats.
            inputs=None,
            outputs=[queue_status, queue_stats_display]
        ).then( # re-validate button state
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )
        
        # Connect the auto-check function to the interface load event
        block.load(
            fn=check_for_current_job_and_monitor, # Use the new combined function
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar, queue_status, queue_stats_display]
        )

        cleanup_btn.click(
            fn=cleanup_temp_files,
            outputs=[cleanup_output]
        )
        
        # The "end_button" (Cancel Job) is the trigger for the next job's monitor.
        # When a job is cancelled, we check for the next one.
        end_button.click(
            fn=end_process_with_update,
            outputs=[queue_status, queue_stats_display, end_button, current_job_id]
        ).then(
            fn=check_for_current_job_and_monitor,
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar, queue_status, queue_stats_display]
        )
        
        load_queue_button.click(
            fn=load_queue_from_json,
            inputs=[],
            outputs=[queue_status, queue_stats_display]
        ).then( # ADD THIS .then() CLAUSE
            fn=check_for_current_job,
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar]
        )
        
        import_queue_file.change(
            fn=import_queue_from_file,
            inputs=[import_queue_file],
            outputs=[queue_status, queue_stats_display]
        ).then( # ADD THIS .then() CLAUSE
            fn=check_for_current_job,
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar]
        )

                        
        # --- Connect Queue Refresh ---
        # The update_stats function is now defined much earlier.
        
        # REMOVED: refresh_stats_btn.click - Toolbar refresh button is no longer needed
        # refresh_stats_btn.click(
        #     fn=update_stats,
        #     inputs=None,
        #     outputs=[queue_status, queue_stats_display]
        # )

        # Set up auto-refresh for queue status
        # Instead of using a timer with 'every' parameter, we'll use the queue refresh button
        # and rely on manual refreshes. The user can click the refresh button in the toolbar
        # to update the stats.

        # --- Connect LoRA UI ---
        # Function to update slider visibility based on selection
        def update_lora_sliders(selected_loras):
            updates = []
            # Need to handle potential missing keys if lora_names changes dynamically
            # For now, assume lora_names passed to create_interface is static
            for lora in lora_names:
                 updates.append(gr.update(visible=(lora in selected_loras)))
            # Ensure the output list matches the number of sliders defined
            num_sliders = len(lora_sliders)
            return updates[:num_sliders] # Return only updates for existing sliders

        # Connect the dropdown to the sliders
        lora_selector.change(
            fn=update_lora_sliders,
            inputs=[lora_selector],
            outputs=[lora_sliders[lora] for lora in lora_names] # Assumes lora_sliders keys match lora_names
        )

        # --- Preset System Functions ---
        PRESET_FILE = "generation_presets.json"

        def load_presets(model_type):
            if not os.path.exists(PRESET_FILE):
                return []
            with open(PRESET_FILE, 'r') as f:
                data = json.load(f)
            return list(data.get(model_type, {}).keys())

        def apply_preset(preset_name, model_type):
            if not preset_name:
                return [gr.update()] * len(ui_components)

            with open(PRESET_FILE, 'r') as f:
                data = json.load(f)
            preset = data.get(model_type, {}).get(preset_name, {})

            updates = []
            for key, component in ui_components.items():
                if key in preset:
                    updates.append(gr.update(value=preset[key]))
                else:
                    updates.append(gr.update())
            
            if 'lora_values' in preset and isinstance(preset['lora_values'], dict):
                lora_values_dict = preset['lora_values']
                for lora_name, slider_component in lora_sliders.items():
                    if lora_name in lora_values_dict:
                        slider_index = list(ui_components.keys()).index(lora_name)
                        updates[slider_index] = gr.update(value=lora_values_dict[lora_name])

            return updates

        def save_preset(preset_name, model_type, *args):
            if not preset_name:
                return gr.update()

            if not os.path.exists(PRESET_FILE):
                with open(PRESET_FILE, 'w') as f:
                    json.dump({}, f)

            with open(PRESET_FILE, 'r') as f:
                data = json.load(f)

            if model_type not in data:
                data[model_type] = {}

            keys = list(ui_components.keys())
            preset_data = {keys[i]: args[i] for i in range(len(args))}
            
            selected_loras = preset_data.get("lora_selector", [])
            lora_values = {}
            for lora_name in selected_loras:
                if lora_name in lora_sliders:
                    slider_index = keys.index(lora_name)
                    lora_values[lora_name] = args[slider_index]

            preset_data['lora_values'] = lora_values

            data[model_type][preset_name] = preset_data

            with open(PRESET_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            return gr.update(choices=load_presets(model_type), value=preset_name)

        def delete_preset(preset_name, model_type):
            if not preset_name:
                return gr.update(), gr.update(visible=True), gr.update(visible=False)
                
            with open(PRESET_FILE, 'r') as f:
                data = json.load(f)

            if model_type in data and preset_name in data[model_type]:
                del data[model_type][preset_name]

            with open(PRESET_FILE, 'w') as f:
                json.dump(data, f, indent=2)

            return gr.update(choices=load_presets(model_type), value=None), gr.update(visible=True), gr.update(visible=False)

        # --- Connect Preset UI ---
        ui_components = {
            "steps": steps, "total_second_length": total_second_length, "resolutionW": resolutionW,
            "resolutionH": resolutionH, "seed": seed, "randomize_seed": randomize_seed,
            "use_teacache": use_teacache, "teacache_num_steps": teacache_num_steps,
            "teacache_rel_l1_thresh": teacache_rel_l1_thresh, "latent_window_size": latent_window_size,
            "gs": gs, "lora_selector": lora_selector, **lora_sliders
        }
        
        model_type.change(
            fn=lambda mt: gr.update(choices=load_presets(mt)),
            inputs=[model_type],
            outputs=[preset_dropdown]
        )
        
        preset_dropdown.select(
            fn=lambda name: name,
            inputs=[preset_dropdown],
            outputs=[preset_name_textbox]
        )

        load_preset_button.click(
            fn=apply_preset,
            inputs=[preset_dropdown, model_type],
            outputs=list(ui_components.values())
        )

        save_preset_button.click(
            fn=save_preset,
            inputs=[preset_name_textbox, model_type, *list(ui_components.values())],
            outputs=[preset_dropdown]
        )
        
        def show_delete_confirmation():
            return gr.update(visible=False), gr.update(visible=True)

        def hide_delete_confirmation():
            return gr.update(visible=True), gr.update(visible=False)

        delete_preset_button.click(
            fn=show_delete_confirmation,
            outputs=[save_preset_button, confirm_delete_row]
        )
        
        confirm_delete_no_btn.click(
            fn=hide_delete_confirmation,
            outputs=[save_preset_button, confirm_delete_row]
        )

        confirm_delete_yes_btn.click(
            fn=delete_preset,
            inputs=[preset_dropdown, model_type],
            outputs=[preset_dropdown, save_preset_button, confirm_delete_row]
        )


        # --- Auto-refresh for Toolbar System Stats Monitor (Timer) ---
        main_toolbar_system_stats_timer = gr.Timer(2, active=True) 
        
        main_toolbar_system_stats_timer.tick(
            fn=tb_get_formatted_toolbar_stats, # Function imported from toolbox_app.py
            inputs=None, 
            outputs=[ # Target the Textbox components
                toolbar_ram_display_component,
                toolbar_vram_display_component,
                toolbar_gpu_display_component 
            ]
        )
        
        # --- Connect Metadata Loading ---
        # Function to load metadata from JSON file
        def load_metadata_from_json(json_path):
            # Define the total number of output components to handle errors gracefully
            num_outputs = 6 + len(lora_sliders)

            if not json_path:
                # Return empty updates for all components if no file is provided
                return [gr.update()] * num_outputs

            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                # Extract values from metadata with defaults
                prompt_val = metadata.get('prompt')
                seed_val = metadata.get('seed')
                total_second_length_val = metadata.get('total_second_length')
                end_frame_strength_val = metadata.get('end_frame_strength')
                model_type_val = metadata.get('model_type')
                lora_weights = metadata.get('loras', {})
                
                # Get the names of the selected LoRAs from the metadata
                selected_lora_names = list(lora_weights.keys())

                print(f"Loaded metadata from JSON: {json_path}")
                print(f"Model Type: {model_type_val}, Prompt: {prompt_val}, Seed: {seed_val}, LoRAs: {selected_lora_names}")

                # Create a list of UI updates
                updates = [
                    gr.update(value=prompt_val) if prompt_val is not None else gr.update(),
                    gr.update(value=seed_val) if seed_val is not None else gr.update(),
                    gr.update(value=total_second_length_val) if total_second_length_val is not None else gr.update(),
                    gr.update(value=end_frame_strength_val) if end_frame_strength_val is not None else gr.update(),
                    gr.update(value=model_type_val) if model_type_val else gr.update(),
                    gr.update(value=selected_lora_names) if selected_lora_names else gr.update()
                ]

                # Update LoRA sliders based on loaded weights
                for lora in lora_names:
                    if lora in lora_weights:
                        updates.append(gr.update(value=lora_weights[lora], visible=True))
                    else:
                        # Hide sliders for LoRAs not in the metadata
                        updates.append(gr.update(visible=False))

                return updates

            except Exception as e:
                print(f"Error loading metadata: {e}")
                import traceback
                traceback.print_exc()
                # Return empty updates for all components on error
                return [gr.update()] * num_outputs


        # Connect JSON metadata loader for Original tab
        json_upload.change(
            fn=load_metadata_from_json,
            inputs=[json_upload],
            outputs=[
                prompt, 
                seed, 
                total_second_length, 
                end_frame_strength_original,
                model_type,
                lora_selector
            ] + [lora_sliders[lora] for lora in lora_names]
        )


        # --- Helper Functions (defined within create_interface scope if needed by handlers) ---
        # Function to get queue statistics
        def get_queue_stats():
            try:
                # Get all jobs from the queue
                jobs = job_queue.get_all_jobs()

                # Count jobs by status
                status_counts = {
                    "QUEUED": 0,
                    "RUNNING": 0,
                    "COMPLETED": 0,
                    "FAILED": 0,
                    "CANCELLED": 0
                }

                for job in jobs:
                    if hasattr(job, 'status'):
                        status = str(job.status) # Use str() for safety
                        if status in status_counts:
                            status_counts[status] += 1

                # Format the display text
                stats_text = f"Queue: {status_counts['QUEUED']} | Running: {status_counts['RUNNING']} | Completed: {status_counts['COMPLETED']} | Failed: {status_counts['FAILED']} | Cancelled: {status_counts['CANCELLED']}"

                return f"<p style='margin:0;color:white;'>{stats_text}</p>"

            except Exception as e:
                print(f"Error getting queue stats: {e}")
                return "<p style='margin:0;color:white;'>Error loading queue stats</p>"

        # Add footer with social links
        with gr.Row(elem_id="footer"):
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div style="text-align: center; padding: 20px; color: #666;">
                    <div style="margin-top: 10px;">
                        <span class="footer-version" style="margin: 0 10px; color: #666;">{APP_VERSION_DISPLAY}</span>
                        <a href="https://patreon.com/Colinu" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;" class="footer-patreon">
                            <i class="fab fa-patreon"></i>Support on Patreon
                        </a>
                        <a href="https://discord.gg/MtuM7gFJ3V" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">
                            <i class="fab fa-discord"></i> Discord
                        </a>
                        <a href="https://github.com/colinurbs/FramePack-Studio" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">
                            <i class="fab fa-github"></i> GitHub
                        </a>
                    </div>
                </div>
                """)

        # Add CSS for footer

        gr.HTML("""
            <script>
            (function() {
                "use strict";
                console.log("Stat Bar Script: Initializing");

                const statConfig = {
                    ram: { selector: '#toolbar-ram-stat', regex: /\((\d+)%\)/, valueIndex: 1, isRawPercentage: true },
                    vram: { selector: '#toolbar-vram-stat', regex: /VRAM: (\d+\.?\d+)\/(\d+\.?\d+)GB/, usedIndex: 1, totalIndex: 2, isRawPercentage: false },
                    gpu: { selector: '#toolbar-gpu-stat', regex: /GPU: \d+°C (\d+)%/, valueIndex: 1, isRawPercentage: true }
                };

                function setBarPercentage(statElement, percentage) {
                    if (!statElement) {
                        console.warn("Stat Bar Script: setBarPercentage called with no element.");
                        return;
                    }
                    let bar = statElement.querySelector('.stat-bar');
                    if (!bar) {
                        console.log("Stat Bar Script: Creating .stat-bar for", statElement.id);
                        bar = document.createElement('div');
                        bar.className = 'stat-bar';
                        statElement.insertBefore(bar, statElement.firstChild);
                    }
                    const clampedPercentage = Math.min(100, Math.max(0, parseFloat(percentage)));
                    statElement.style.setProperty('--stat-percentage', clampedPercentage + '%');
                    // console.log("Stat Bar Script: Updated", statElement.id, "to", clampedPercentage + "%");
                }

                function updateSingleStatVisual(key, config) {
                    try {
                        const container = document.querySelector(config.selector);
                        if (!container) {
                            // console.warn("Stat Bar Script: Container not found for", key, config.selector);
                            return false; // Element not ready
                        }
                        const textarea = container.querySelector('textarea');
                        if (!textarea) {
                            // console.warn("Stat Bar Script: Textarea not found for", key);
                            return false; // Element not ready
                        }

                        const textValue = textarea.value;
                        if (textValue === "RAM: N/A" || textValue === "VRAM: N/A" || textValue === "GPU: N/A") {
                             setBarPercentage(container, 0); // Set to 0 if N/A
                             return true;
                        }

                        const match = textValue.match(config.regex);
                        if (match) {
                            let percentage = 0;
                            if (config.isRawPercentage) {
                                percentage = parseInt(match[config.valueIndex]);
                            } else { // VRAM case
                                const used = parseFloat(match[config.usedIndex]);
                                const total = parseFloat(match[config.totalIndex]);
                                percentage = (total > 0) ? (used / total) * 100 : 0;
                            }
                            setBarPercentage(container, percentage);
                        } else {
                            // console.warn("Stat Bar Script: Regex mismatch for", key, "-", textValue);
                             setBarPercentage(container, 0); // Default to 0 on mismatch after initial load
                        }
                        return true; // Processed or N/A
                    } catch (error) {
                        console.error("Stat Bar Script: Error updating visual for", key, error);
                        return true; // Assume processed to avoid retry loops on error
                    }
                }
                
                function updateAllStatVisuals() {
                    let allReady = true;
                    for (const key in statConfig) {
                        if (!updateSingleStatVisual(key, statConfig[key])) {
                            allReady = false;
                        }
                    }
                    return allReady;
                }

                function initStatBars() {
                    console.log("Stat Bar Script: initStatBars called");
                    if (updateAllStatVisuals()) {
                        console.log("Stat Bar Script: All stats initialized. Setting up MutationObserver.");
                        setupMutationObservers();
                    } else {
                        console.log("Stat Bar Script: Elements not ready, retrying init in 250ms.");
                        setTimeout(initStatBars, 250); // Retry if not all elements were ready
                    }
                }

                function setupMutationObservers() {
                    const observer = new MutationObserver((mutationsList) => {
                        // Use a Set to avoid redundant updates if multiple mutations point to the same stat
                        const changedStats = new Set();

                        for (const mutation of mutationsList) {
                            let targetElement = mutation.target;
                            // Traverse up to find the .toolbar-stat-textbox parent if mutation is deep
                            while(targetElement && !targetElement.matches('.toolbar-stat-textbox')) {
                                targetElement = targetElement.parentElement;
                            }

                            if (targetElement && targetElement.matches('.toolbar-stat-textbox')) {
                                for (const key in statConfig) {
                                    if (targetElement.id === statConfig[key].selector.substring(1)) {
                                        changedStats.add(key);
                                        break;
                                    }
                                }
                            }
                        }
                        if (changedStats.size > 0) {
                           // console.log("Stat Bar Script: MutationObserver detected changes for:", Array.from(changedStats));
                           changedStats.forEach(key => updateSingleStatVisual(key, statConfig[key]));
                        }
                    });

                    for (const key in statConfig) {
                        const container = document.querySelector(statConfig[key].selector);
                        if (container) {
                            // Observe the container for changes to its children (like textarea value)
                            // and the textarea itself if it exists.
                            observer.observe(container, { childList: true, subtree: true, characterData: true });
                            console.log("Stat Bar Script: Observer attached to", container.id);
                        } else {
                            console.warn("Stat Bar Script: Could not attach observer, container not found for", key);
                        }
                    }
                }

                // More robust DOM ready check
                if (document.readyState === "complete" || (document.readyState !== "loading" && !document.documentElement.doScroll)) {
                    console.log("Stat Bar Script: DOM already ready.");
                    initStatBars();
                } else {
                    document.addEventListener("DOMContentLoaded", () => {
                        console.log("Stat Bar Script: DOMContentLoaded event.");
                        initStatBars();
                    });
                }
                 // Fallback for Gradio's dynamic loading, if DOMContentLoaded isn't enough
                 window.addEventListener('gradio.rendered', () => {
                    console.log('Stat Bar Script: Gradio rendered event detected.');
                    initStatBars();
                });

            })();
            </script>
        """)

        return block

# --- Top-level Helper Functions (Used by Gradio callbacks, must be defined outside create_interface) ---

def format_queue_status(jobs):
    """Format job data for display in the queue status table"""
    rows = []
    for job in jobs:
        created = time.strftime('%H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
        started = time.strftime('%H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
        completed = time.strftime('%H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

        # Calculate elapsed time
        elapsed_time = ""
        if job.started_at:
            if job.completed_at:
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                complete_datetime = datetime.datetime.fromtimestamp(job.completed_at)
                elapsed_seconds = (complete_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s"
            else:
                # For running jobs, calculate elapsed time from now
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                current_datetime = datetime.datetime.now()
                elapsed_seconds = (current_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}s (running)"

        # Get generation type from job data
        generation_type = getattr(job, 'generation_type', 'Original')

        # Get thumbnail from job data and format it as HTML for display
        thumbnail = getattr(job, 'thumbnail', None)
        thumbnail_html = f'<img src="{thumbnail}" width="64" height="64" style="object-fit: contain;">' if thumbnail else ""

        rows.append([
            job.id[:6] + '...',
            generation_type,
            job.status.value,
            created,
            started,
            completed,
            elapsed_time,
            thumbnail_html  # Add formatted thumbnail HTML to row data
        ])
    return rows

# Create the queue status update function (wrapper around format_queue_status)
def update_queue_status_with_thumbnails(): # Function name is now slightly misleading, but keep for now to avoid breaking clicks
    # This function is likely called by the refresh button and potentially the timer
    # It needs access to the job_queue object
    # Assuming job_queue is accessible globally or passed appropriately
    # For now, let's assume it's globally accessible as defined in studio.py
    # If not, this needs adjustment based on how job_queue is managed.
    try:
        # Need access to the global job_queue instance from studio.py
        # This might require restructuring or passing job_queue differently.
        # For now, assuming it's accessible (this might fail if run standalone)
        from __main__ import job_queue # Attempt to import from main script scope

        jobs = job_queue.get_all_jobs()
        for job in jobs:
            if job.status == JobStatus.PENDING:
                job.queue_position = job_queue.get_queue_position(job.id)

        if job_queue.current_job:
            job_queue.current_job.status = JobStatus.RUNNING

        return format_queue_status(jobs)
    except ImportError:
        print("Error: Could not import job_queue. Queue status update might fail.")
        return [] # Return empty list on error
    except Exception as e:
        print(f"Error updating queue status: {e}")
        return []
