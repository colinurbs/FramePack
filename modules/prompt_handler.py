import re
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class PromptSection:
    """Represents a section of the prompt with specific timing information"""
    prompt: str
    start_time: float = 0  # in seconds
    end_time: Optional[float] = None  # in seconds, None means until the end


def snap_to_section_boundaries(prompt_sections: List[PromptSection], latent_window_size: int, fps: int = 30) -> List[PromptSection]:
    """
    Adjust timestamps to align with model's internal section boundaries
    
    Args:
        prompt_sections: List of PromptSection objects
        latent_window_size: Size of the latent window used in the model
        fps: Frames per second (default: 30)
        
    Returns:
        List of PromptSection objects with aligned timestamps
    """
    section_duration = (latent_window_size * 4 - 3) / fps  # Duration of one section in seconds
    
    aligned_sections = []
    for section in prompt_sections:
        # Snap start time to nearest section boundary
        aligned_start = round(section.start_time / section_duration) * section_duration
        
        # Snap end time to nearest section boundary
        aligned_end = None
        if section.end_time is not None:
            aligned_end = round(section.end_time / section_duration) * section_duration
        
        # Ensure minimum section length
        if aligned_end is not None and aligned_end <= aligned_start:
            aligned_end = aligned_start + section_duration
            
        aligned_sections.append(PromptSection(
            prompt=section.prompt,
            start_time=aligned_start,
            end_time=aligned_end
        ))
    
    return aligned_sections


def parse_timestamped_prompt(prompt_text: str, total_duration: float, latent_window_size: int = 9, generation_type: str = "Original") -> List[PromptSection]:
    """
    Parse a prompt with timestamps in the format [0s-2s: text] or [3s: text]
    
    Args:
        prompt_text: The input prompt text with optional timestamp sections
        total_duration: Total duration of the video in seconds
        latent_window_size: Size of the latent window used in the model
        generation_type: Type of generation ("Original" or "F1")
        
    Returns:
        List of PromptSection objects with timestamps aligned to section boundaries
        and reversed to account for reverse generation (only for Original type)
    """
    # Default prompt for the entire duration if no timestamps are found
    if "[" not in prompt_text or "]" not in prompt_text:
        return [PromptSection(prompt=prompt_text.strip())]
    
    sections = []
    # Find all timestamp sections [time: text]
    timestamp_pattern = r'\[(\d+(?:\.\d+)?s)(?:-(\d+(?:\.\d+)?s))?\s*:\s*(.*?)\]'
    regular_text = prompt_text
    
    for match in re.finditer(timestamp_pattern, prompt_text):
        start_time_str = match.group(1)
        end_time_str = match.group(2)
        section_text = match.group(3).strip()
        
        # Convert time strings to seconds
        start_time = float(start_time_str.rstrip('s'))
        end_time = float(end_time_str.rstrip('s')) if end_time_str else None
        
        sections.append(PromptSection(
            prompt=section_text,
            start_time=start_time,
            end_time=end_time
        ))
        
        # Remove the processed section from regular_text
        regular_text = regular_text.replace(match.group(0), "")
    
    # If there's any text outside of timestamp sections, use it as a default for the entire duration
    regular_text = regular_text.strip()
    if regular_text:
        sections.append(PromptSection(
            prompt=regular_text,
            start_time=0,
            end_time=None
        ))
    
    # Sort sections by start time
    sections.sort(key=lambda x: x.start_time)
    
    # Fill in end times if not specified
    for i in range(len(sections) - 1):
        if sections[i].end_time is None:
            sections[i].end_time = sections[i+1].start_time
    
    # Set the last section's end time to the total duration if not specified
    if sections and sections[-1].end_time is None:
        sections[-1].end_time = total_duration
    
    # Snap timestamps to section boundaries
    sections = snap_to_section_boundaries(sections, latent_window_size)
    
    # Only reverse timestamps for Original generation type
    if generation_type == "Original":
        # Now reverse the timestamps to account for reverse generation
        reversed_sections = []
        for section in sections:
            reversed_start = total_duration - section.end_time if section.end_time is not None else 0
            reversed_end = total_duration - section.start_time
            reversed_sections.append(PromptSection(
                prompt=section.prompt,
                start_time=reversed_start,
                end_time=reversed_end
            ))
        
        # Sort the reversed sections by start time
        reversed_sections.sort(key=lambda x: x.start_time)
        return reversed_sections
    
    return sections


def get_section_boundaries(latent_window_size: int = 9, count: int = 10) -> str:
    """
    Calculate and format section boundaries for UI display
    
    Args:
        latent_window_size: Size of the latent window used in the model
        count: Number of boundaries to display
        
    Returns:
        Formatted string of section boundaries
    """
    section_duration = (latent_window_size * 4 - 3) / 30
    return ", ".join([f"{i*section_duration:.1f}s" for i in range(count)])


def get_quick_prompts() -> List[List[str]]:
    """
    Get a list of example timestamped prompts
    
    Returns:
        List of example prompts formatted for Gradio Dataset
    """
    prompts = [
        '[0s: The person waves hello] [2s: The person jumps up and down] [4s: The person does a spin]',
        '[0s: The person raises both arms slowly] [2s: The person claps hands enthusiastically]',
        '[0s: Person gives thumbs up] [1.1s: Person smiles and winks] [2.2s: Person shows two thumbs down]',
        '[0s: Person looks surprised] [1.1s: Person raises arms above head] [2.2s-3.3s: Person puts hands on hips]'
    ]
    return [[x] for x in prompts]


def parse_prompt_segments(prompt_text: str) -> List[Dict[str, float | str]]:
    """
    Parse existing prompt text to segments for editing in the UI
    
    Args:
        prompt_text: The formatted prompt text with timestamps
    
    Returns:
        List of dictionaries containing start_time and prompt for each segment
    """
    if not prompt_text or "[" not in prompt_text:
        return [{"start_time": 0, "prompt": prompt_text}]
    
    segments = []
    pattern = r'\[(\d+(?:\.\d+)?s)(?:-(\d+(?:\.\d+)?s))?\s*:\s*(.*?)\]'
    
    for match in re.finditer(pattern, prompt_text):
        start_time_str = match.group(1)
        section_text = match.group(3).strip()
        start_time = float(start_time_str.rstrip('s'))
        segments.append({"start_time": start_time, "prompt": section_text})
    
    # Sort by start time
    segments.sort(key=lambda x: x['start_time'])
    return segments if segments else [{"start_time": 0, "prompt": ""}]


def format_prompt_segments(segments: List[Dict[str, float | str]]) -> str:
    """
    Convert prompt segments from UI format to the format expected by the backend
    
    Args:
        segments: List of segment dictionaries with start_time and prompt
    
    Returns:
        Formatted prompt string with timestamp notation
    """
    formatted_parts = []
    for segment in segments:
        start_time = segment.get('start_time', 0)
        prompt = segment.get('prompt', '')
        if prompt:
            formatted_parts.append(f"[{start_time}s: {prompt}]")
    return " ".join(formatted_parts)


def validate_segments(segments: List[Dict[str, float | str]], total_duration: float) -> List[str]:
    """
    Validate prompt segments for potential issues
    
    Args:
        segments: List of segment dictionaries
        total_duration: Total video duration in seconds
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not segments:
        errors.append("At least one prompt segment is required")
        return errors
    
    # Check for empty prompts
    for i, segment in enumerate(segments):
        if not segment.get('prompt', '').strip():
            errors.append(f"Segment {i + 1} has an empty prompt")
    
    # Check for out-of-range times
    for i, segment in enumerate(segments):
        start_time = segment.get('start_time', 0)
        if start_time < 0:
            errors.append(f"Segment {i + 1} has negative start time")
        elif start_time > total_duration:
            errors.append(f"Segment {i + 1} starts after video ends")
    
    # Check for overlapping segments (optional - could be intentional for blending)
    sorted_segments = sorted(segments, key=lambda x: x.get('start_time', 0))
    for i in range(len(sorted_segments) - 1):
        current_end = sorted_segments[i].get('start_time', 0) + 0.1  # Minimal duration
        next_start = sorted_segments[i + 1].get('start_time', 0)
        if current_end > next_start:
            errors.append(f"Segments {i + 1} and {i + 2} have overlapping times")
    
    return errors