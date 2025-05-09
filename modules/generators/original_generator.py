import torch
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.memory import DynamicSwapInstaller
from .base_generator import BaseModelGenerator

class OriginalModelGenerator(BaseModelGenerator):
    """
    Model generator for the Original HunyuanVideo model.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Original model generator.
        """
        super().__init__(**kwargs)
        self.model_name = "Original"
        self.model_path = 'lllyasviel/FramePackI2V_HY'
    
    def get_model_name(self):
        """
        Get the name of the model.
        """
        return self.model_name
    
    def load_model(self):
        """
        Load the Original transformer model.
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
        Prepare the history latents tensor for the Original model.
        
        Args:
            height: The height of the image
            width: The width of the image
            
        Returns:
            The initialized history latents tensor
        """
        return torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), 
            dtype=torch.float32
        ).cpu()
    
    def get_latent_paddings(self, total_latent_sections):
        """
        Get the latent paddings for the Original model.
        
        Args:
            total_latent_sections: The total number of latent sections
            
        Returns:
            A list of latent paddings
        """
        # Original model uses reversed latent paddings
        if total_latent_sections > 4:
            return [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            return list(reversed(range(total_latent_sections)))
    
    def prepare_indices(self, latent_padding_size, latent_window_size):
        """
        Prepare the indices for the Original model.
        
        Args:
            latent_padding_size: The size of the latent padding
            latent_window_size: The size of the latent window
            
        Returns:
            A tuple of (clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices)
        """
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
        
        return clean_latent_indices, latent_indices, clean_latent_2x_indices, clean_latent_4x_indices
    
    def prepare_clean_latents(self, start_latent, history_latents):
        """
        Prepare the clean latents for the Original model.
        
        Args:
            start_latent: The start latent
            history_latents: The history latents
            
        Returns:
            A tuple of (clean_latents, clean_latents_2x, clean_latents_4x)
        """
        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
        
        return clean_latents, clean_latents_2x, clean_latents_4x
    
    def update_history_latents(self, history_latents, generated_latents):
        """
        Update the history latents with the generated latents for the Original model.
        
        Args:
            history_latents: The history latents
            generated_latents: The generated latents
            
        Returns:
            The updated history latents
        """
        # For Original model, we prepend the generated latents
        return torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
    
    def get_real_history_latents(self, history_latents, total_generated_latent_frames):
        """
        Get the real history latents for the Original model.
        
        Args:
            history_latents: The history latents
            total_generated_latent_frames: The total number of generated latent frames
            
        Returns:
            The real history latents
        """
        return history_latents[:, :, :total_generated_latent_frames, :, :]
    
    def update_history_pixels(self, history_pixels, current_pixels, overlapped_frames):
        """
        Update the history pixels with the current pixels for the Original model.
        
        Args:
            history_pixels: The history pixels
            current_pixels: The current pixels
            overlapped_frames: The number of overlapped frames
            
        Returns:
            The updated history pixels
        """
        from diffusers_helper.utils import soft_append_bcthw
        # For Original model, current_pixels is first, history_pixels is second
        return soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
    
    def get_section_latent_frames(self, latent_window_size, is_last_section):
        """
        Get the number of section latent frames for the Original model.
        
        Args:
            latent_window_size: The size of the latent window
            is_last_section: Whether this is the last section
            
        Returns:
            The number of section latent frames
        """
        return (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
    
    def get_current_pixels(self, real_history_latents, section_latent_frames, vae):
        """
        Get the current pixels for the Original model.
        
        Args:
            real_history_latents: The real history latents
            section_latent_frames: The number of section latent frames
            vae: The VAE model
            
        Returns:
            The current pixels
        """
        from diffusers_helper.hunyuan import vae_decode
        return vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
    
    def format_position_description(self, total_generated_latent_frames, current_pos, original_pos, current_prompt):
        """
        Format the position description for the Original model.
        
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
                f'Current position: {current_pos:.2f}s (original: {original_pos:.2f}s). '
                f'using prompt: {current_prompt[:256]}...')
