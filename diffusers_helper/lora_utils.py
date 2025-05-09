from pathlib import Path, PurePath
from typing import Dict, List, Optional, Union
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
import torch

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors"):
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.

    """
    
    state_dict = _fetch_state_dict(
        lora_path,
        weight_name,
        True,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None)

    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    
    # should weight_name even be Optional[str] or just str?
    # For now, we assume it is never None
    # The module name in the state_dict must not include a . in the name
    # See https://github.com/pytorch/pytorch/pull/6639/files#diff-4be56271f7bfe650e3521c81fd363da58f109cd23ee80d243156d2d6ccda6263R133-R134
    adapter_name = PurePath(str(weight_name).replace('_DOT_', '.')).stem.replace('.', '_DOT_')
    if '_DOT_' in adapter_name:
        print(
            f"LoRA file '{weight_name}' contains a '.' in the name. " +
            'This may cause issues. Consider renaming the file.' +
            f" Using '{adapter_name}' as the adapter name to be safe."
        )
    
    # Check if adapter already exists and delete it if it does
    if hasattr(transformer, 'peft_config') and adapter_name in transformer.peft_config:
        print(f"Adapter '{adapter_name}' already exists. Removing it before loading again.")
        # Use delete_adapters (plural) instead of delete_adapter
        transformer.delete_adapters([adapter_name])
    
    # Load the adapter with the original name
    transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=adapter_name)
    print(f"LoRA weights '{adapter_name}' loaded successfully.")
    
    return transformer

def unload_all_loras(transformer):
    """
    Completely unload all LoRA adapters from the transformer model.
    """
    if hasattr(transformer, 'peft_config') and transformer.peft_config:
        # Get all adapter names
        adapter_names = list(transformer.peft_config.keys())
        
        if adapter_names:
            print(f"Removing all LoRA adapters: {', '.join(adapter_names)}")
            # Delete all adapters
            transformer.delete_adapters(adapter_names)
            
            # Force cleanup of any remaining adapter references
            if hasattr(transformer, 'active_adapter'):
                transformer.active_adapter = None
                
            # Clear any cached states
            for module in transformer.modules():
                if hasattr(module, 'lora_A'):
                    if isinstance(module.lora_A, dict):
                        module.lora_A.clear()
                if hasattr(module, 'lora_B'):
                    if isinstance(module.lora_B, dict):
                        module.lora_B.clear()
                if hasattr(module, 'scaling'):
                    if isinstance(module.scaling, dict):
                        module.scaling.clear()
            
            print("All LoRA adapters have been completely removed.")
        else:
            print("No LoRA adapters found to remove.")
    else:
        print("Model doesn't have any LoRA adapters or peft_config.")
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return transformer

    
# TODO(neph1): remove when HunyuanVideoTransformer3DModelPacked is in _SET_ADAPTER_SCALE_FN_MAPPING
def set_adapters(
        transformer,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):

    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

    # Expand weights into a list, one entry per adapter
    # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)

    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
        )

    # Set None values to default of 1.0
    # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
    weights = [w if w is not None else 1.0 for w in weights]

    # e.g. [{...}, 7] -> [{expanded dict...}, 7]
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING["HunyuanVideoTransformer3DModel"]
    weights = scale_expansion_fn(transformer, weights)

    set_weights_and_activate_adapters(transformer, adapter_names, weights)
