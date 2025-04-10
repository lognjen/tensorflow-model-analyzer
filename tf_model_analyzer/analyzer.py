"""
TensorFlow model analyzer module
"""
import os
import json
import datetime
import tensorflow as tf
from collections import Counter
from typing import Dict, Any, Optional, Union, Tuple

class ModelAnalyzer:
    """
    Analyzes TensorFlow models and extracts key information.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the analyzer with a model path.
        
        Args:
            model_path (str): Path to the TensorFlow model (.h5, .pb, or SavedModel directory)
        """
        self.model_path = model_path
        self.model = None
        self.analysis_results = {}
        
    def load_model(self) -> None:
        """
        Load the TensorFlow model from the specified path.
        
        Raises:
            ValueError: If the model cannot be loaded
        """
        try:
            if os.path.isdir(self.model_path):
                # Load as SavedModel
                self.model = tf.keras.models.load_model(self.model_path)
            elif self.model_path.endswith('.h5'):
                # Load as H5 file
                self.model = tf.keras.models.load_model(self.model_path)
            elif self.model_path.endswith('.pb'):
                # Load as frozen graph
                self.model = tf.saved_model.load(os.path.dirname(self.model_path))
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}. "
                                 f"Supported formats: .h5, .pb, or SavedModel directory.")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the model and return results as a dictionary.
        
        Returns:
            Dict[str, Any]: Analysis results
        
        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            self.load_model()
            
        # Get basic model info
        if isinstance(self.model, tf.keras.Model):
            self._analyze_keras_model()
        else:
            self._analyze_saved_model()
            
        return self.analysis_results
    
    def _analyze_keras_model(self) -> None:
        """
        Analyze a Keras model and populate analysis_results.
        """
        # Get current date and time
        analysis_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count layers by type
        layer_counter = Counter()
        for layer in self.model.layers:
            layer_counter[layer.__class__.__name__] += 1
            
        # Get input and output shapes
        input_shapes = [str(inp.shape) for inp in self.model.inputs]
        output_shapes = [str(out.shape) for out in self.model.outputs]
            
        # Get trainable and non-trainable parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        # Compile results
        self.analysis_results = {
            "model_info": {
                "filename": os.path.basename(self.model_path),
                "analysis_date": analysis_date
            },
            "summary": {
                "total_layers": len(self.model.layers),
                "trainable_parameters": int(trainable_params),
                "non_trainable_parameters": int(non_trainable_params)
            },
            "layer_types": dict(layer_counter),
            "shapes": {
                "input": input_shapes[0] if len(input_shapes) == 1 else input_shapes,
                "output": output_shapes[0] if len(output_shapes) == 1 else output_shapes
            }
        }
    
    def _analyze_saved_model(self) -> None:
        """
        Analyze a SavedModel and populate analysis_results.
        This is a basic implementation as SavedModel offers less introspection.
        """
        analysis_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get limited information from saved model
        concrete_functions = list(self.model.signatures.values())
        
        # Get input and output specs
        if concrete_functions:
            first_function = concrete_functions[0]
            input_specs = {name: spec.shape.as_list() for name, spec in first_function.inputs.items()}
            output_specs = {name: spec.shape.as_list() for name, spec in first_function.outputs.items()}
        else:
            input_specs = {}
            output_specs = {}
        
        self.analysis_results = {
            "model_info": {
                "filename": os.path.basename(self.model_path),
                "analysis_date": analysis_date,
                "format": "SavedModel"
            },
            "shapes": {
                "input": input_specs,
                "output": output_specs
            },
            "note": "Limited information available for SavedModel format"
        }
    
    def export_txt(self, output_path: Optional[str] = None) -> str:
        """
        Export the analysis results as formatted text.
        
        Args:
            output_path (str, optional): Path to save the text output
            
        Returns:
            str: Formatted text output
        """
        if not self.analysis_results:
            self.analyze()
            
        txt_output = []
        txt_output.append("TensorFlow Model Analysis")
        txt_output.append("========================")
        txt_output.append(f"Model: {self.analysis_results['model_info']['filename']}")
        txt_output.append(f"Date: {self.analysis_results['model_info']['analysis_date']}")
        txt_output.append("")
        
        # Add summary section if available
        if "summary" in self.analysis_results:
            txt_output.append("Summary:")
            txt_output.append(f"- Total layers: {self.analysis_results['summary']['total_layers']}")
            txt_output.append(f"- Trainable parameters: {self.analysis_results['summary']['trainable_parameters']:,}")
            txt_output.append(f"- Non-trainable parameters: {self.analysis_results['summary']['non_trainable_parameters']:,}")
            txt_output.append("")
        
        # Add layer types if available
        if "layer_types" in self.analysis_results:
            txt_output.append("Layer Types:")
            for layer_type, count in self.analysis_results["layer_types"].items():
                txt_output.append(f"- {layer_type}: {count}")
            txt_output.append("")
        
        # Add shapes
        if "shapes" in self.analysis_results:
            shapes = self.analysis_results["shapes"]
            txt_output.append(f"Input Shape: {shapes['input']}")
            txt_output.append(f"Output Shape: {shapes['output']}")
        
        # Add any notes
        if "note" in self.analysis_results:
            txt_output.append("")
            txt_output.append(f"Note: {self.analysis_results['note']}")
        
        formatted_text = "\n".join(txt_output)
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(formatted_text)
                
        return formatted_text
    
    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        Export the analysis results as JSON.
        
        Args:
            output_path (str, optional): Path to save the JSON output
            
        Returns:
            str: JSON string
        """
        if not self.analysis_results:
            self.analyze()
            
        json_output = json.dumps(self.analysis_results, indent=2)
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_output)
                
        return json_output
    
    def export(self, output_path: Optional[str] = None, format_type: str = 'txt') -> str:
        """
        Export the analysis results in the specified format.
        
        Args:
            output_path (str, optional): Path to save the output
            format_type (str): Output format - 'txt' or 'json'
            
        Returns:
            str: Formatted output
            
        Raises:
            ValueError: If format_type is not supported
        """
        # If no format specified but output_path has extension, infer format
        if output_path and format_type == 'txt':
            if output_path.endswith('.json'):
                format_type = 'json'
        
        if format_type.lower() == 'txt':
            return self.export_txt(output_path)
        elif format_type.lower() == 'json':
            return self.export_json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}. Use 'txt' or 'json'.")
