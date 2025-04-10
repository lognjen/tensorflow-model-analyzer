"""
Tests for the analyzer module
"""
import os
import json
import unittest
import tempfile
import tensorflow as tf
from tf_model_analyzer.analyzer import ModelAnalyzer

class TestModelAnalyzer(unittest.TestCase):
    """Test cases for the ModelAnalyzer class"""
    
    @classmethod
    def setUpClass(cls):
        """Create a simple model for testing"""
        # Build a simple sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
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
    
    def test_load_model(self):
        """Test loading a model"""
        analyzer = ModelAnalyzer(self.model_path)
        analyzer.load_model()
        self.assertIsNotNone(analyzer.model)
        self.assertTrue(isinstance(analyzer.model, tf.keras.Model))
    
    def test_analyze(self):
        """Test analyzing a model"""
        analyzer = ModelAnalyzer(self.model_path)
        results = analyzer.analyze()
        
        # Check that the results dictionary contains expected keys
        self.assertIn("model_info", results)
        self.assertIn("summary", results)
        self.assertIn("layer_types", results)
        self.assertIn("shapes", results)
        
        # Verify some of the model attributes
        self.assertEqual(results["summary"]["total_layers"], 8)
        self.assertIn("Conv2D", results["layer_types"])
        self.assertIn("Dense", results["layer_types"])
        self.assertEqual(results["layer_types"]["Conv2D"], 2)
        self.assertEqual(results["layer_types"]["Dense"], 2)
    
    def test_export_txt(self):
        """Test exporting results to TXT"""
        analyzer = ModelAnalyzer(self.model_path)
        analyzer.analyze()
        
        # Test export to string
        txt_output = analyzer.export_txt()
        self.assertIsInstance(txt_output, str)
        self.assertIn("TensorFlow Model Analysis", txt_output)
        self.assertIn("Conv2D", txt_output)
        
        # Test export to file
        temp_txt = os.path.join(self.temp_dir, "output.txt")
        analyzer.export_txt(temp_txt)
        self.assertTrue(os.path.exists(temp_txt))
        
        # Clean up
        if os.path.exists(temp_txt):
            os.remove(temp_txt)
    
    def test_export_json(self):
        """Test exporting results to JSON"""
        analyzer = ModelAnalyzer(self.model_path)
        analyzer.analyze()
        
        # Test export to string
        json_output = analyzer.export_json()
        self.assertIsInstance(json_output, str)
        
        # Verify JSON can be parsed
        json_data = json.loads(json_output)
        self.assertIn("model_info", json_data)
        self.assertIn("summary", json_data)
        
        # Test export to file
        temp_json = os.path.join(self.temp_dir, "output.json")
        analyzer.export_json(temp_json)
        self.assertTrue(os.path.exists(temp_json))
        
        # Clean up
        if os.path.exists(temp_json):
            os.remove(temp_json)
    
    def test_export_format_detection(self):
        """Test that format is correctly detected from file extension"""
        analyzer = ModelAnalyzer(self.model_path)
        analyzer.analyze()
        
        # Test with explicit format
        temp_txt = os.path.join(self.temp_dir, "output.foo")
        analyzer.export(temp_txt, format_type="txt")
        with open(temp_txt, 'r') as f:
            content = f.read()
        self.assertIn("TensorFlow Model Analysis", content)
        
        # Test with format from file extension
        temp_json = os.path.join(self.temp_dir, "output.json")
        analyzer.export(temp_json)
        with open(temp_json, 'r') as f:
            content = f.read()
        # Verify it's valid JSON
        json_data = json.loads(content)
        self.assertIn("model_info", json_data)
        
        # Clean up
        if os.path.exists(temp_txt):
            os.remove(temp_txt)
        if os.path.exists(temp_json):
            os.remove(temp_json)
    
    def test_invalid_model_path(self):
        """Test handling of invalid model path"""
        analyzer = ModelAnalyzer("nonexistent_model.h5")
        with self.assertRaises(ValueError):
            analyzer.load_model()
    
    def test_invalid_export_format(self):
        """Test handling of invalid export format"""
        analyzer = ModelAnalyzer(self.model_path)
        analyzer.analyze()
        with self.assertRaises(ValueError):
            analyzer.export(format_type="invalid")

if __name__ == "__main__":
    unittest.main()
