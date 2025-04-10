"""
Utility functions for the TensorFlow model analyzer
"""
import os
import sys
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple

def human_readable_size(size_bytes: int) -> str:
    """
    Convert a size in bytes to a human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Human-readable size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f}{size_names[i]}"

def get_model_memory_size(model: tf.keras.Model) -> int:
    """
    Estimate the memory size of a Keras model.
    
    Args:
        model (tf.keras.Model): Keras model
        
    Returns:
        int: Estimated size in bytes
    """
    # Calculate size in bytes (approximation)
    size_bytes = 0
    for weight in model.weights:
        size_bytes += weight.numpy().nbytes
    
    return size_bytes

def check_tensorflow_gpu() -> Tuple[bool, str]:
    """
    Check if TensorFlow is using GPU.
    
    Returns:
        Tuple[bool, str]: (is_gpu_available, device_info)
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_info = []
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            if details and 'device_name' in details:
                gpu_info.append(details['device_name'])
            else:
                gpu_info.append(gpu.name)
        return True, f"GPU(s) available: {', '.join(gpu_info)}"
    else:
        return False, "No GPU detected. Using CPU."

def get_layer_info(model: tf.keras.Model) -> List[Dict[str, Any]]:
    """
    Get detailed information about each layer in the model.
    
    Args:
        model (tf.keras.Model): Keras model
        
    Returns:
        List[Dict[str, Any]]: List of layer information dictionaries
    """
    layers_info = []
    
    for i, layer in enumerate(model.layers):
        layer_info = {
            "index": i,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "parameters": layer.count_params(),
            "trainable": layer.trainable,
        }
        
        # Get input/output shapes if available
        try:
            if hasattr(layer, "input_shape"):
                layer_info["input_shape"] = str(layer.input_shape)
            if hasattr(layer, "output_shape"):
                layer_info["output_shape"] = str(layer.output_shape)
        except AttributeError:
            pass
        
        layers_info.append(layer_info)
    
    return layers_info

def is_tensorflow_installed() -> bool:
    """
    Check if TensorFlow is installed.
    
    Returns:
        bool: True if TensorFlow is installed, False otherwise
    """
    try:
        import tensorflow
        return True
    except ImportError:
        return False
