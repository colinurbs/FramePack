import torch
import os # required for os.path
from abc import ABC, abstractmethod
from diffusers_helper import lora_utils

class BaseModelGenerator(ABC):
    """
    Base class for model generators.
    This defines the common interface that all model generators must implement.
    """
    
    def __init__(self, 
                 text_encoder, 
                 text_encoder_2, 
                 tokenizer, 
                 tokenizer_2, 
                 vae, 
                 image_encoder, 
                 feature_extractor, 
                 high_vram=False,
                 prompt_embedding_cache=None,
                 settings=None,
                 offline=False): # NEW: offline flag
        """
        Initialize the base model generator.
        
        Args:
            text_encoder: The text encoder model
            text_encoder_2: The second text encoder model
            tokenizer: The tokenizer for the first text encoder
            tokenizer_2: The tokenizer for the second text encoder
            vae: The VAE model
            image_encoder: The image encoder model
            feature_extractor: The feature extractor
            high_vram: Whether high VRAM mode is enabled
            prompt_embedding_cache: Cache for prompt embeddings
            settings: Application settings
            offline: Whether to run in offline mode for model loading
        """
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.vae = vae
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.high_vram = high_vram
        self.prompt_embedding_cache = prompt_embedding_cache or {}
        self.settings = settings
        self.offline = offline 
        self.transformer = None
        self.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

            
    @abstractmethod
    def load_model(self):
        """
        Load the transformer model.
        This method should be implemented by each specific model generator.
        """
        pass
    
    @abstractmethod
    def get_model_name(self):
        """
        Get the name of the model.
        This method should be implemented by each specific model generator.
        """
        pass

    @staticmethod
    def _get_snapshot_hash_from_refs(model_repo_id_for_cache: str) -> str | None:
        """
        Reads the commit hash from the refs/main file for a given model in the HF cache.
        Args:
            model_repo_id_for_cache (str): The model ID formatted for cache directory names
                                           (e.g., "models--lllyasviel--FramePackI2V_HY").
        Returns:
            str: The commit hash if found, otherwise None.
        """
        hf_home_dir = os.environ.get('HF_HOME')
        if not hf_home_dir:
            print("Warning: HF_HOME environment variable not set. Cannot determine snapshot hash.")
            return None
            
        refs_main_path = os.path.join(hf_home_dir, 'hub', model_repo_id_for_cache, 'refs', 'main')
        if os.path.exists(refs_main_path):
            try:
                with open(refs_main_path, 'r') as f:
                    print(f"Offline mode: Reading snapshot hash from: {refs_main_path}")
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read snapshot hash from {refs_main_path}: {e}")
                return None
        else:
            print(f"Warning: refs/main file not found at {refs_main_path}. Cannot determine snapshot hash.")
            return None

    def _get_offline_load_path(self) -> str:
        """
        Returns the local snapshot path for offline loading if available.
        Falls back to the default self.model_path if local snapshot can't be found.
        Relies on self.model_repo_id_for_cache and self.model_path being set by subclasses.
        """
        # Ensure necessary attributes are set by the subclass
        if not hasattr(self, 'model_repo_id_for_cache') or not self.model_repo_id_for_cache:
            print(f"Warning: model_repo_id_for_cache not set in {self.__class__.__name__}. Cannot determine offline path.")
            # Fallback to model_path if it exists, otherwise None
            return getattr(self, 'model_path', None) 

        if not hasattr(self, 'model_path') or not self.model_path:
            print(f"Warning: model_path not set in {self.__class__.__name__}. Cannot determine fallback for offline path.")
            return None

        snapshot_hash = self._get_snapshot_hash_from_refs(self.model_repo_id_for_cache)
        hf_home = os.environ.get('HF_HOME')

        if snapshot_hash and hf_home:
            specific_snapshot_path = os.path.join(
                hf_home, 'hub', self.model_repo_id_for_cache, 'snapshots', snapshot_hash
            )
            if os.path.isdir(specific_snapshot_path):
                return specific_snapshot_path
                
        # If snapshot logic fails or path is not a dir, fallback to the default model path
        return self.model_path
        
    def unload_loras(self):
        """
        Unload all LoRAs from the transformer model.
        """
        if self.transformer is not None:
            print(f"Unloading all LoRAs from {self.get_model_name()} model")
            self.transformer = lora_utils.unload_all_loras(self.transformer)
            self.verify_lora_state("After unloading LoRAs")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def verify_lora_state(self, label=""):
        """
        Debug function to verify the state of LoRAs in the transformer model.
        """
        if self.transformer is None:
            print(f"[{label}] Transformer is None, cannot verify LoRA state")
            return
            
        has_loras = False
        if hasattr(self.transformer, 'peft_config'):
            adapter_names = list(self.transformer.peft_config.keys()) if self.transformer.peft_config else []
            if adapter_names:
                has_loras = True
                print(f"[{label}] Transformer has LoRAs: {', '.join(adapter_names)}")
            else:
                print(f"[{label}] Transformer has no LoRAs in peft_config")
        else:
            print(f"[{label}] Transformer has no peft_config attribute")
            
        # Check for any LoRA modules
        for name, module in self.transformer.named_modules():
            if hasattr(module, 'lora_A') and module.lora_A:
                has_loras = True
                # print(f"[{label}] Found lora_A in module {name}")
            if hasattr(module, 'lora_B') and module.lora_B:
                has_loras = True
                # print(f"[{label}] Found lora_B in module {name}")
                
        if not has_loras:
            print(f"[{label}] No LoRA components found in transformer")
    
    def move_lora_adapters_to_device(self, target_device):
        """
        Move all LoRA adapters in the transformer model to the specified device.
        This handles the PEFT implementation of LoRA.
        """
        if self.transformer is None:
            return
            
        print(f"Moving all LoRA adapters to {target_device}")
        
        # First, find all modules with LoRA adapters
        lora_modules = []
        for name, module in self.transformer.named_modules():
            if hasattr(module, 'active_adapter') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules.append((name, module))
        
        # Now move all LoRA components to the target device
        for name, module in lora_modules:
            # Get the active adapter name
            active_adapter = module.active_adapter
            
            # Move the LoRA layers to the target device
            if active_adapter is not None:
                if isinstance(module.lora_A, torch.nn.ModuleDict):
                    # Handle ModuleDict case (PEFT implementation)
                    for adapter_name in list(module.lora_A.keys()):
                        # Move lora_A
                        if adapter_name in module.lora_A:
                            module.lora_A[adapter_name] = module.lora_A[adapter_name].to(target_device)
                        
                        # Move lora_B
                        if adapter_name in module.lora_B:
                            module.lora_B[adapter_name] = module.lora_B[adapter_name].to(target_device)
                        
                        # Move scaling
                        if hasattr(module, 'scaling') and isinstance(module.scaling, dict) and adapter_name in module.scaling:
                            if isinstance(module.scaling[adapter_name], torch.Tensor):
                                module.scaling[adapter_name] = module.scaling[adapter_name].to(target_device)
                else:
                    # Handle direct attribute case
                    if hasattr(module, 'lora_A') and module.lora_A is not None:
                        module.lora_A = module.lora_A.to(target_device)
                    if hasattr(module, 'lora_B') and module.lora_B is not None:
                        module.lora_B = module.lora_B.to(target_device)
                    if hasattr(module, 'scaling') and module.scaling is not None:
                        if isinstance(module.scaling, torch.Tensor):
                            module.scaling = module.scaling.to(target_device)
        
        print(f"Moved all LoRA adapters to {target_device}")
    
    def load_loras(self, selected_loras, lora_folder, lora_loaded_names, lora_values=None):
        """
        Load LoRAs into the transformer model.
        
        Args:
            selected_loras: List of LoRA names to load
            lora_folder: Folder containing the LoRA files
            lora_loaded_names: List of loaded LoRA names
            lora_values: Optional list of LoRA strength values
        """
        if self.transformer is None:
            print("Cannot load LoRAs: Transformer model is not loaded")
            return
            
        import os
        
        # Ensure all LoRAs are unloaded first
        self.unload_loras()
        
        # Load each selected LoRA
        for lora_name in selected_loras:
            try:
                idx = lora_loaded_names.index(lora_name)
                lora_file = None
                for ext in [".safetensors", ".pt"]:
                    candidate_path_relative = f"{lora_name}{ext}"
                    candidate_path_full = os.path.join(lora_folder, candidate_path_relative)
                    if os.path.isfile(candidate_path_full):
                        lora_file = candidate_path_relative
                        break
                        
                if lora_file:
                    print(f"Loading LoRA '{lora_file}' to {self.get_model_name()} model")
                    self.transformer = lora_utils.load_lora(self.transformer, lora_folder, lora_file)
                    
                    # Set LoRA strength if provided
                    if lora_values and idx < len(lora_values):
                        lora_strength = float(lora_values[idx])
                        print(f"Setting LoRA '{lora_name}' strength to {lora_strength}")
                        
                        # Set scaling for this LoRA by iterating through modules
                        for name, module in self.transformer.named_modules():
                            if hasattr(module, 'scaling'):
                                if isinstance(module.scaling, dict):
                                    # Handle ModuleDict case (PEFT implementation)
                                    if lora_name in module.scaling:
                                        if isinstance(module.scaling[lora_name], torch.Tensor):
                                            module.scaling[lora_name] = torch.tensor(
                                                lora_strength, device=module.scaling[lora_name].device
                                            )
                                        else:
                                            module.scaling[lora_name] = lora_strength
                                else:
                                    # Handle direct attribute case for scaling if needed
                                    if isinstance(module.scaling, torch.Tensor):
                                        module.scaling = torch.tensor(
                                            lora_strength, device=module.scaling.device
                                        )
                                    else:
                                        module.scaling = lora_strength
                else:
                    print(f"LoRA file for {lora_name} not found!")
            except Exception as e:
                print(f"Error loading LoRA {lora_name}: {e}")
        
        # Verify LoRA state after loading
        self.verify_lora_state("After loading LoRAs")
