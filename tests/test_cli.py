"""
Tests for the CLI module
"""
import os
import sys
import json
import unittest
import tempfile
import tensorflow as tf
from unittest.mock import patch
from io import StringIO
from tf_model_analyzer.cli import main, parse_arguments

class TestCLI(unittest.TestCase):
    """Test cases for the CLI functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Create a simple model for testing"""
        # Build a simple sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        
        # Compile the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        
        # Save the model to a temporary file
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, "test_model.h5")
        model.save(cls.model_path)
        cls.model = model

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)
    
    def test_parse_arguments_defaults(self):
        """Test parsing command line arguments with defaults"""
        with patch('sys.argv', ['tf-analyzer', self.model_path]):
            args = parse_arguments()
            self.assertEqual(args.model_path, self.model_path)
            self.assertIsNone(args.output)
            self.assertEqual(args.format, 'txt')
            self.assertFalse(args.verbose)
    
    def test_parse_arguments_all_options(self):
        """Test parsing command line arguments with all options specified"""
        with patch('sys.argv', [
            'tf-analyzer', 
            self.model_path, 
            '--output', 'output.json', 
            '--format', 'json', 
            '--verbose'
        ]):
            args = parse_arguments()
            self.assertEqual(args.model_path, self.model_path)
            self.assertEqual(args.output, 'output.json')
            self.assertEqual(args.format, 'json')
            self.assertTrue(args.verbose)
    
    def test_main_console_output(self):
        """Test main function with output to console"""
        output_file = os.path.join(self.temp_dir, "output.txt")
        
        # Test with output to console (captured)
        with patch('sys.argv', ['tf-analyzer', self.model_path]), \
             patch('sys.stdout', new=StringIO()) as fake_out:
            main()
            console_output = fake_out.getvalue()
            self.assertIn("TensorFlow Model Analysis", console_output)
            self.assertIn("Conv2D", console_output)
    
    def test_main_file_output_txt(self):
        """Test main function with output to txt file"""
        output_file = os.path.join(self.temp_dir, "output.txt")
        
        # Test with output to file
        with patch('sys.argv', ['tf-analyzer', self.model_path, '--output', output_file]), \
             patch('sys.stdout', new=StringIO()) as fake_out:
            main()
            # Should print success message
            console_output = fake_out.getvalue()
            self.assertIn(f"Analysis saved to {output_file}", console_output)
            
            # File should exist and contain analysis
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn("TensorFlow Model Analysis", content)
        
        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)
    
    def test_main_file_output_json(self):
        """Test main function with output to json file"""
        output_file = os.path.join(self.temp_dir, "output.json")
        
        # Test with output to JSON file
        with patch('sys.argv', [
            'tf-analyzer', self.model_path, '--output', output_file, '--format', 'json'
        ]), patch('sys.stdout', new=StringIO()) as fake_out:
            main()
            
            # File should exist and contain valid JSON
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, 'r') as f:
                content = json.load(f)
                self.assertIn("model_info", content)
                self.assertIn("summary", content)
        
        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)
    
    def test_main_format_inference(self):
        """Test format inference from output file extension"""
        output_file = os.path.join(self.temp_dir, "output.json")
        
        # Test format inference from file extension
        with patch('sys.argv', ['tf-analyzer', self.model_path, '--output', output_file]):
            main()
            
            # File should exist and contain valid JSON
            self.assertTrue(os.path.exists(output_file))
            with open(output_file, 'r') as f:
                try:
                    content = json.load(f)
                    self.assertIn("model_info", content)
                    json_format = True
                except json.JSONDecodeError:
                    json_format = False
                    
            self.assertTrue(json_format, "Output should be in JSON format")
        
        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)
    
    def test_main_nonexistent_model(self):
        """Test handling of nonexistent model path"""
        with patch('sys.argv', ['tf-analyzer', 'nonexistent_model.h5']), \
             patch('sys.stdout', new=StringIO()) as fake_out, \
             patch('sys.exit') as mock_exit:
            main()
            console_output = fake_out.getvalue()
            self.assertIn("Error: Model path 'nonexistent_model.h5' does not exist", console_output)
            mock_exit.assert_called_once_with(1)

if __name__ == "__main__":
    unittest.main()
