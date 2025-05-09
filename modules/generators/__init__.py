from .original_generator import OriginalModelGenerator
from .f1_generator import F1ModelGenerator

def create_model_generator(model_type, **kwargs):
    """
    Create a model generator based on the model type.
    
    Args:
        model_type: The type of model to create ("Original" or "F1")
        **kwargs: Additional arguments to pass to the model generator constructor
        
    Returns:
        A model generator instance
        
    Raises:
        ValueError: If the model type is not supported
    """
    if model_type == "Original":
        return OriginalModelGenerator(**kwargs)
    elif model_type == "F1":
        return F1ModelGenerator(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
