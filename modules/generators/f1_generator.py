import torch
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import DynamicSwapInstaller
from .base_generator import BaseModelGenerator

class F1ModelGenerator(BaseModelGenerator):
    """
    Model generator for the F1 HunyuanVideo model.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the F1 model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "F1"
        self.model_path = 'lllyasviel/FramePack_F1_I2V_HY_20250503'
    
    def get_model_name(self):
        """
        Get the name of the model.
        """
        return self.model_name
    
    def load_model(self):
        """
        Load the F1 transformer model.
        """
        print(f"Loading {self.model_name} Transformer...")
        
        # Create the transformer model
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            self.model_path, 
            torch_dtype=torch.bfloat16
        ).cpu()
        
        # Configure the model
        self.transformer.eval()
        self.transformer.to(dtype=torch.bfloat16)
        self.transformer.requires_grad_(False)
        
        # Set up dynamic swap if not in high VRAM mode
        if not self.high_vram:
            DynamicSwapInstaller.install_model(self.transformer, device=self.gpu)
        
        print(f"{self.model_name} Transformer Loaded.")
        return self.transformer
    
    def prepare_history_latents(self, height, width):
        """
        Prepare the history latents tensor for the F1 model.
        
        Args:
            height: The height of the image
            width: The width of the image
            
        Returns:
            The initialized history latents tensor
        """
        return torch.zeros(
            size=(1, 16, 16 + 2 + 1, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
    
    def initialize_with_start_latent(self, history_latents, start_latent):
        """
        Initialize the history latents with the start latent for the F1 model.
        
        Args:
            history_latents: The history latents
            start_latent: The start latent
            
        Returns:
            The initialized history latents
        """
        # Add the start frame to history_latents
        return torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
    
    def get_latent_paddings(self, total_latent_sections):
        """
        Get the latent paddings for the F1 model.
        
        Args:
            total_latent_sections: The total number of latent sections
            
        Returns:
            A list of latent paddings
        """
        # F1 model uses a fixed approach with just 0 for last section and 1 for others
        return [1] * (total_latent_sections - 1) + [0]
    
    def prepare_indices(self, latent_padding_size, latent_window_size):
        """
        Prepare the indices for the F1 model.
        
        Args:
            latent_padding_size: The size of the latent padding
            latent_window_size: The size of the latent window
            
        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices)
        """
        # F1 model uses a different indices approach
        # latent_window_sizeが4.5の場合は特別に5を使用
        effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
        indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
        clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
        
        return clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices
    
    def prepare_clean_latents(self, start_latent, history_latents):
        """
        Prepare the clean latents for the F1 model.
        
        Args:
            start_latent: The start latent
            history_latents: The history latents
            
        Returns:
            A tuple of (clean_latents, clean_latents_2x, clean_latents_4x)
        """
        # For F1, we take the last frames for clean latents
        clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
        # For F1, we prepend the start latent to clean_latents_1x
        clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
        
        return clean_latents, clean_latents_2x, clean_latents_4x
    
    def update_history_latents(self, history_latents, generated_latents):
        """
        Update the history latents with the generated latents for the F1 model.
        
        Args:
            history_latents: The history latents
            generated_latents: The generated latents
            
        Returns:
            The updated history latents
        """
        # For F1, we append new frames to the end
        return torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
    
    def get_real_history_latents(self, history_latents, total_generated_latent_frames):
        """
        Get the real history latents for the F1 model.
        
        Args:
            history_latents: The history latents
            total_generated_latent_frames: The total number of generated latent frames
            
        Returns:
            The real history latents
        """
        # For F1, we take frames from the end
        return history_latents[:, :, -total_generated_latent_frames:, :, :]
    
    def update_history_pixels(self, history_pixels, current_pixels, overlapped_frames):
        """
        Update the history pixels with the current pixels for the F1 model.
        
        Args:
            history_pixels: The history pixels
            current_pixels: The current pixels
            overlapped_frames: The number of overlapped frames
            
        Returns:
            The updated history pixels
        """
        from diffusers_helper.utils import soft_append_bcthw
        # For F1 model, history_pixels is first, current_pixels is second
        return soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
    
    def get_section_latent_frames(self, latent_window_size, is_last_section):
        """
        Get the number of section latent frames for the F1 model.
        
        Args:
            latent_window_size: The size of the latent window
            is_last_section: Whether this is the last section
            
        Returns:
            The number of section latent frames
        """
        return (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
    
    def get_current_pixels(self, real_history_latents, section_latent_frames, vae):
        """
        Get the current pixels for the F1 model.
        
        Args:
            real_history_latents: The real history latents
            section_latent_frames: The number of section latent frames
            vae: The VAE model
            
        Returns:
            The current pixels
        """
        from diffusers_helper.hunyuan import vae_decode
        # For F1, we take frames from the end
        return vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
    
    def format_position_description(self, total_generated_latent_frames, current_pos, original_pos, current_prompt):
        """
        Format the position description for the F1 model.
        
        Args:
            total_generated_latent_frames: The total number of generated latent frames
            current_pos: The current position in seconds
            original_pos: The original position in seconds
            current_prompt: The current prompt
            
        Returns:
            The formatted position description
        """
        return (f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, '
                f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). '
                f'Current position: {current_pos:.2f}s. '
                f'using prompt: {current_prompt[:256]}...')
