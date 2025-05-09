import gradio as gr
import time
import datetime
import random
import json
import os
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import base64
import io
import json

from modules.video_queue import JobStatus, Job
from modules.prompt_handler import get_section_boundaries, get_quick_prompts, parse_timestamped_prompt, format_prompt_segments, parse_prompt_segments
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.bucket_tools import find_nearest_bucket

def create_prompt_interface(default_prompt="[1s: The person waves hello] [3s: The person jumps up and down] [5s: The person does a dance]", max_segments=10):
    """Create a reusable prompt interface component"""
    
    # Container for the interface
    interface = {}
    
    # Parse initial prompt
    initial_segments = parse_prompt_segments(default_prompt)
    
    # Hidden state to store segments
    interface['prompt_segments_state'] = gr.State(initial_segments)
    
    # Main UI container
    with gr.Column():
        gr.Markdown("### Prompt Timeline")
        
        # Create rows for each segment
        interface['segment_rows'] = []
        interface['segment_visibility'] = []
        interface['segment_time_inputs'] = []
        interface['segment_prompt_inputs'] = []
        interface['segment_delete_buttons'] = []
        
        for i in range(max_segments):
            visible = (i < len(initial_segments))
            
            with gr.Row(visible=visible) as row:
                with gr.Column(scale=10):
                    with gr.Row():
                        time_input = gr.Number(
                            label=f"Segment {i + 1} - Start Time (seconds)",
                            value=initial_segments[i].get('start_time', 0) if i < len(initial_segments) else 0,
                            minimum=0,
                            maximum=120,
                            step=0.1
                        )
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            value=initial_segments[i].get('prompt', '') if i < len(initial_segments) else '',
                            placeholder="Enter your prompt for this segment"
                        )
                with gr.Column(scale=1):
                    delete_btn = gr.Button("❌", variant="stop", size="sm")
            
            interface['segment_rows'].append(row)
            interface['segment_time_inputs'].append(time_input)
            interface['segment_prompt_inputs'].append(prompt_input)
            interface['segment_delete_buttons'].append(delete_btn)
        
        # Hidden components for state management
        interface['hidden_prompt'] = gr.Textbox(value=default_prompt, visible=False)
        interface['segment_count'] = gr.Number(value=len(initial_segments), visible=False)
        
        # Add segment button
        with gr.Row():
            interface['add_segment_button'] = gr.Button("+ Add Prompt Segment", variant="primary")
    
    return interface


def connect_prompt_interface_events(interface, max_segments=10):
    """Connect event handlers for the prompt interface"""
    
    # Helper functions for event handling
    def update_segments(segment_count, *inputs):
        """Update segments when time or prompt changes"""
        segments = []
        
        for i in range(0, len(inputs), 2):
            if i < segment_count * 2:
                time_val = inputs[i]
                prompt_val = inputs[i + 1]
                if prompt_val:  # Only include segments with content
                    segments.append({"start_time": time_val, "prompt": prompt_val})
        
        segments.sort(key=lambda x: x['start_time'])
        formatted_prompt = format_prompt_segments(segments)
        
        return segments, formatted_prompt
    
    def add_segment(segment_count):
        """Add a new segment"""
        new_count = min(segment_count + 1, max_segments)
        updates = []
        
        # Update visibility of rows
        for i in range(max_segments):
            updates.append(gr.update(visible=(i < new_count)))
        
        return [new_count] + updates
    
    def delete_segment(segment_index, segment_count, *inputs):
        """Delete a segment"""
        if segment_count <= 1:  # Keep at least one segment
            return [gr.update()] * (max_segments * 3 + 1)
        
        segments = []
        
        # Collect all segments except the deleted one
        for i in range(0, len(inputs), 2):
            if i < segment_count * 2 and i // 2 != segment_index:
                time_val = inputs[i]
                prompt_val = inputs[i + 1]
                if prompt_val:
                    segments.append({"start_time": time_val, "prompt": prompt_val})
        
        segments.sort(key=lambda x: x['start_time'])
        new_count = len(segments)
        
        # Prepare updates for all components
        updates = []
        
        # Update segment count
        updates.append(new_count)
        
        # Update row visibility
        for i in range(max_segments):
            updates.append(gr.update(visible=(i < new_count)))
        
        # Update time inputs
        for i in range(max_segments):
            if i < new_count:
                updates.append(gr.update(value=segments[i]['start_time']))
            else:
                updates.append(gr.update(value=0))
        
        # Update prompt inputs
        for i in range(max_segments):
            if i < new_count:
                updates.append(gr.update(value=segments[i]['prompt']))
            else:
                updates.append(gr.update(value=''))
        
        return updates
    
    # Get all inputs for the update function
    all_inputs = []
    for i in range(max_segments):
        all_inputs.extend([
            interface['segment_time_inputs'][i],
            interface['segment_prompt_inputs'][i]
        ])
    
    # Connect change handlers for all time and prompt inputs
    for i in range(max_segments):
        # Time input changes
        interface['segment_time_inputs'][i].change(
            fn=update_segments,
            inputs=[interface['segment_count']] + all_inputs,
            outputs=[interface['prompt_segments_state'], interface['hidden_prompt']]
        )
        
        # Prompt input changes
        interface['segment_prompt_inputs'][i].change(
            fn=update_segments,
            inputs=[interface['segment_count']] + all_inputs,
            outputs=[interface['prompt_segments_state'], interface['hidden_prompt']]
        )
        
        # Delete button clicks
        interface['segment_delete_buttons'][i].click(
            fn=delete_segment,
            inputs=[gr.Number(i, visible=False), interface['segment_count']] + all_inputs,
            outputs=[interface['segment_count']] + 
                    interface['segment_rows'] + 
                    interface['segment_time_inputs'] + 
                    interface['segment_prompt_inputs']
        )
    
    # Add segment button click
    interface['add_segment_button'].click(
        fn=add_segment,
        inputs=[interface['segment_count']],
        outputs=[interface['segment_count']] + interface['segment_rows']
    )
    
    return interface


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
    # Get section boundaries and quick prompts
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()

    # Create the interface
    css = make_progress_bar_css()
    css += """
    .contain-image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
        background: #222;
    }
    
    .prompt-segment {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background: #1a1a1a;
    }
    
    .segment-controls {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 10px;
    }
    
    .time-input {
        width: 100px !important;
    }
    
    #fixed-toolbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 1000;
        background: rgb(11, 15, 25);
        color: #fff;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-bottom: 1px solid #4f46e5;
    }
    #toolbar-add-to-queue-btn button {
        font-size: 14px !important;
        padding: 4px 16px !important;
        height: 32px !important;
        min-width: 80px !important;
    }

    .gr-button-primary{
        color:white;
    }
    body, .gradio-container {
        padding-top: 40px !important;
    }
    .narrow-button {
        min-width: 40px !important;
        width: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .thumbnail-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px;
    }
    .thumbnail-item {
        width: 100px;
        height: 100px;
        border: 1px solid #444;
        border-radius: 4px;
        overflow: hidden;
    }
    .thumbnail-item img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    #footer {
        margin-top: 20px;
        padding: 20px;
        border-top: 1px solid #eee;
    }
    #footer a:hover {
        color: #4f46e5 !important;
    }
    """

    # Get the theme from settings
    current_theme = settings.get("gradio_theme", "default") # Use default if not found
    block = gr.Blocks(css=css, title="FramePack Studio", theme=current_theme).queue()

    with block:

        with gr.Row(elem_id="fixed-toolbar"):
            gr.Markdown("<h1 style='margin:0;color:white;'>FramePack Studio</h1>")
            # with gr.Column(scale=1):
            #     queue_stats_display = gr.Markdown("<p style='margin:0;color:white;'>Queue: 0 | Completed: 0</p>")
            with gr.Column(scale=0):
                refresh_stats_btn = gr.Button("⟳", elem_id="refresh-stats-btn")


        

        with gr.Tabs():
            with gr.Tab("Generate", id="generate_tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_type = gr.Radio(
                            choices=["Original", "F1"],
                            value="Original",
                            label="Model Type",
                            info="Select which model to use for generation"
                        )
                        input_image = gr.Image(
                            sources='upload',
                            type="numpy",
                            label="Image (optional)",
                            height=420,
                            elem_classes="contain-image"
                        )
                        

                        with gr.Accordion("Latent Image Options", open=False):
                            latent_type = gr.Dropdown(
                                ["Black", "White", "Noise", "Green Screen"], label="Latent Image", value="Black", info="Used as a starting point if no image is provided"
                            )

                        # Create prompt interface for Original model
                        prompt_interface = create_prompt_interface(default_prompt)
                        prompt_segments_state = prompt_interface['prompt_segments_state']
                        hidden_prompt = prompt_interface['hidden_prompt']
                        segment_count = prompt_interface['segment_count']
                        
                        # Connect events
                        connect_prompt_interface_events(prompt_interface)
                        
                        with gr.Accordion("Prompt Parameters", open=False):
                            blend_sections = gr.Slider(
                                minimum=0, maximum=10, value=4, step=1,
                                label="Number of sections to blend between prompts"
                            )
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                                total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=1, maximum=120, value=6, step=0.1)
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
                                save_metadata = gr.Checkbox(label="Save Metadata", value=True, info="Save to JSON file")
                            with gr.Row("TeaCache"):
                                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True)  # Make visible for both models

                            with gr.Row():
                                seed = gr.Number(label="Seed", value=31337, precision=0)
                                randomize_seed = gr.Checkbox(label="Randomize", value=False, info="Generate a new random seed for each job")

                        with gr.Accordion("Advanced Parameters", open=False):
                            latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info='Change at your own risk, very experimental')  # Should not change
                            cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                            gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                            rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change
                            gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=1, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                        with gr.Accordion("Output Parameters", open=False):
                            mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                            clean_up_videos = gr.Checkbox(
                                label="Clean up video files",
                                value=True,
                                info="If checked, only the final video will be kept after generation."
                            )

                    with gr.Column():
                        preview_image = gr.Image(label="Next Latents", height=150, visible=True, type="numpy", interactive=False)
                        result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=256, loop=True)
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')

                        with gr.Row():
                            current_job_id = gr.Textbox(label="Current Job ID", visible=True, interactive=True)
                            end_button = gr.Button(value="Cancel Current Job", interactive=True)
                            start_button = gr.Button(value="Add to Queue", elem_id="toolbar-add-to-queue-btn")

            with gr.Tab("Queue"):
                with gr.Row():
                    with gr.Column():
                        # Create a container for the queue status
                        with gr.Row():
                            queue_status = gr.DataFrame(
                                headers=["Job ID", "Type", "Status", "Created", "Started", "Completed", "Elapsed"], # Removed Preview header
                                datatype=["str", "str", "str", "str", "str", "str", "str"], # Removed image datatype
                                label="Job Queue"
                            )
                        with gr.Row():
                            refresh_button = gr.Button("Refresh Queue")
                            # Connect the refresh button (Moved inside 'with block')
                            refresh_button.click(
                                fn=update_queue_status_fn, # Use the function passed in
                                inputs=[],
                                outputs=[queue_status]
                            )
                        # Create a container for thumbnails (kept for potential future use, though not displayed in DataFrame)
                        with gr.Row():
                            thumbnail_container = gr.Column()
                            thumbnail_container.elem_classes = ["thumbnail-container"]

            with gr.TabItem("Outputs"):
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
                def load_video_and_info_from_prefix(prefix):
                    video_file = get_latest_video_version(prefix)
                    if not video_file:
                        return None, "JSON not found."
                    video_path = os.path.join(outputDirectory_video, video_file)
                    json_path = os.path.join(outputDirectory_metadata, prefix) + ".json"
                    info = {"description": "no info"}
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            info = json.load(f)
                    return video_path, json.dumps(info, indent=2, ensure_ascii=False)
                gallery_items_state = gr.State(get_gallery_items())
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
                    def refresh_gallery():
                        new_items = get_gallery_items()
                        return gr.update(value=[i[0] for i in new_items]), new_items
                    refresh_button.click(fn=refresh_gallery, outputs=[thumbs, gallery_items_state])
                    def on_select(evt: gr.SelectData, gallery_items):
                        prefix = gallery_items[evt.index][1]
                        video, info = load_video_and_info_from_prefix(prefix)
                        return gr.update(value=video, visible=True), gr.update(value=info, visible=True)
                    thumbs.select(fn=on_select, inputs=[gallery_items_state], outputs=[video_out, info_out])
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
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

                        def save_settings(output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, selected_theme):
                            try:
                                settings.save_settings(
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
                            inputs=[output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, theme_dropdown],
                            outputs=[status]
                        )

                        def cleanup_temp_files():
                            """Clean up temporary files in the Gradio temp directory"""
                            temp_dir = settings.get("gradio_temp_dir")
                            if not temp_dir or not os.path.exists(temp_dir):
                                return "No temporary directory found or directory does not exist."
                            
                            try:
                                # Get all files in the temp directory
                                files = os.listdir(temp_dir)
                                removed_count = 0
                                
                                for file in files:
                                    file_path = os.path.join(temp_dir, file)
                                    try:
                                        if os.path.isfile(file_path):
                                            os.remove(file_path)
                                            removed_count += 1
                                    except Exception as e:
                                        print(f"Error removing {file_path}: {e}")
                                
                                return f"Cleaned up {removed_count} temporary files."
                            except Exception as e:
                                return f"Error cleaning up temporary files: {str(e)}"

                        cleanup_btn.click(
                            fn=cleanup_temp_files,
                            outputs=[cleanup_output]
                        )

        # --- Event Handlers and Connections (Now correctly indented) ---

        # Connect the main process function (wrapper for adding to queue)
        def process_with_queue_update(model_type, *args):
            # Extract all arguments (ensure order matches inputs lists)
            input_image, prompt_segments, hidden_prompt_text, n_prompt, seed_value, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, randomize_seed_checked, save_metadata_checked, blend_sections, latent_type, clean_up_videos, selected_loras, resolutionW, resolutionH, *lora_args = args

            # Use the formatted prompt text
            prompt_text = hidden_prompt_text

            # Call the process function with all arguments
            result = process_fn(model_type, input_image, prompt_text, n_prompt, seed_value, total_second_length,
                            latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                            use_teacache, mp4_crf, save_metadata_checked, blend_sections, latent_type, clean_up_videos, selected_loras, resolutionW, resolutionH, *lora_args)

            # If randomize_seed is checked, generate a new random seed for the next job
            new_seed_value = None
            if randomize_seed_checked:
                new_seed_value = random.randint(0, 21474)
                print(f"Generated new seed for next job: {new_seed_value}")

            # If a job ID was created, automatically start monitoring it and update queue
            if result and result[1]:  # Check if job_id exists in results
                job_id = result[1]
                queue_status_data = update_queue_status_fn()

                # Add the new seed value to the results if randomize is checked
                if new_seed_value is not None:
                    return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data, new_seed_value]
                else:
                    return [result[0], job_id, result[2], result[3], result[4], result[5], result[6], queue_status_data, gr.update()]

            # If no job ID was created, still return the new seed if randomize is checked
            if new_seed_value is not None:
                return result + [update_queue_status_fn(), new_seed_value]
            else:
                return result + [update_queue_status_fn(), gr.update()]

        # Custom end process function that ensures the queue is updated
        def end_process_with_update():
            queue_status_data = end_process_fn()
            # Make sure to return the queue status data
            return queue_status_data

        # --- Inputs Lists ---
        # --- Inputs for Original Model ---
        ips = [
            input_image,
            prompt_segments_state,
            hidden_prompt,
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
            mp4_crf,
            randomize_seed,
            save_metadata,
            blend_sections,
            latent_type,
            clean_up_videos,
            lora_selector,
            resolutionW,
            resolutionH,
            lora_names_states
        ]
        # Add LoRA sliders to the input list
        ips.extend([lora_sliders[lora] for lora in lora_names])


        # --- Connect Buttons ---
        start_button.click(
            # Pass the selected model type from the radio buttons
            fn=lambda selected_model, *args: process_with_queue_update(selected_model, *args),
            inputs=[model_type] + ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, seed]
        )

        # Connect the end button to cancel the current job and update the queue
        end_button.click(
            fn=end_process_with_update,
            outputs=[queue_status]
        )

        # --- Connect Monitoring ---
        # Auto-monitor the current job when job_id changes
        # Monitor original tab
        current_job_id.change(
            fn=monitor_fn,
            inputs=[current_job_id],
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button]
        )


        # --- Connect Queue Refresh ---
        refresh_stats_btn.click(
            fn=lambda: update_queue_status_fn(), # Use update_queue_status_fn passed in
            inputs=None,
            outputs=[queue_status]  # Removed queue_stats_display from outputs
        )

        # Set up auto-refresh for queue status (using a timer)
        refresh_timer = gr.Number(value=0, visible=False)
        def refresh_timer_fn():
            """Updates the timer value periodically to trigger queue refresh"""
            return int(time.time())
        # This timer seems unused, maybe intended for block.load()? Keeping definition for now.
        # refresh_timer.change(
        #     fn=update_queue_status_fn, # Use the function passed in
        #     outputs=[queue_status] # Update shared queue status display
        # )

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


        # --- Connect Metadata Loading ---
        def load_metadata_from_json(json_path, max_segments=10):
            if not json_path:
                return [gr.update()] * (3 + len(lora_names) + max_segments * 3 + 1)

            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                prompt_val = metadata.get('prompt')
                seed_val = metadata.get('seed')

                # Parse the prompt into segments
                segments = parse_prompt_segments(prompt_val) if prompt_val else [{"start_time": 0, "prompt": ""}]
                segment_count = len(segments)

                # Check for LoRA values in metadata
                lora_weights = metadata.get('loras', {})

                print(f"Loaded metadata from JSON: {json_path}")
                print(f"Prompt: {prompt_val}, Seed: {seed_val}")

                # Update the UI components
                updates = []
                
                # prompt_segments_state
                updates.append(segments)
                
                # hidden_prompt
                updates.append(gr.update(value=prompt_val) if prompt_val else gr.update())
                
                # seed
                updates.append(gr.update(value=seed_val) if seed_val is not None else gr.update())
                
                # LoRA sliders
                for lora in lora_names:
                    if lora in lora_weights:
                        updates.append(gr.update(value=lora_weights[lora]))
                    else:
                        updates.append(gr.update())

                # segment_count
                updates.append(segment_count)

                # Update visibility of rows
                for i in range(max_segments):
                    updates.append(gr.update(visible=(i < segment_count)))

                # Update time inputs
                for i in range(max_segments):
                    if i < segment_count:
                        updates.append(gr.update(value=segments[i]['start_time']))
                    else:
                        updates.append(gr.update(value=0))

                # Update prompt inputs  
                for i in range(max_segments):
                    if i < segment_count:
                        updates.append(gr.update(value=segments[i]['prompt']))
                    else:
                        updates.append(gr.update(value=''))

                return updates

            except Exception as e:
                print(f"Error loading metadata: {e}")
                return [gr.update()] * (3 + len(lora_names) + max_segments * 3 + 1)

        # Connect JSON metadata loader for Original tab
        json_upload.change(
            fn=load_metadata_from_json,
            inputs=[json_upload],
            outputs=[prompt_segments_state, hidden_prompt, seed] + 
                    [lora_sliders[lora] for lora in lora_names] + 
                    [segment_count] + 
                    prompt_interface['segment_rows'] + 
                    prompt_interface['segment_time_inputs'] + 
                    prompt_interface['segment_prompt_inputs']
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
                gr.HTML("""
                <div style="text-align: center; padding: 20px; color: #666;">
                    <div style="margin-top: 10px;">
                        <a href="https://patreon.com/Colinu" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">
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

    return block


# --- Top-level Helper Functions (Used by Gradio callbacks, must be defined outside create_interface) ---

def format_queue_status(jobs):
    """Format job data for display in the queue status table"""
    rows = []
    for job in jobs:
        created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
        started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
        completed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

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

        # Removed thumbnail processing

        rows.append([
            job.id[:6] + '...',
            generation_type,
            job.status.value,
            created,
            started,
            completed,
            elapsed_time
            # Removed thumbnail from row data
        ])
    return rows