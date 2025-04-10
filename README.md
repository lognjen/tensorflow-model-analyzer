# TensorFlow Model Analyzer

A command-line tool for analyzing TensorFlow models and extracting basic information about their architecture.

## Features

- Analyze TensorFlow model files (.h5, .pb, SavedModel)
- Extract key information:
  - Number of layers
  - Layer types
  - Total parameter count
  - Input/output shapes
- Export analysis to TXT or JSON format
- Simple command-line interface

## Installation

### From Source

```bash
git clone https://github.com/lognjen/tensorflow-model-analyzer.git
cd tensorflow-model-analyzer
pip install -e .
```

## Requirements

- Python 3.7+
- TensorFlow 2.x

## Usage

### Basic Usage

```bash
# Analyze a model and print to console
tf-analyzer path/to/model.h5

# Analyze a SavedModel directory
tf-analyzer path/to/saved_model_dir

# Save output to TXT file
tf-analyzer path/to/model.h5 --output model_info.txt

# Save output to JSON file
tf-analyzer path/to/model.h5 --output model_info.json --format json
```

### Options

```
--output, -o     Output file path (default: print to console)
--format, -f     Output format: 'txt' or 'json' (default: determined by file extension)
--verbose, -v    Include additional model details
--help, -h       Show help message
```

## Examples

### Example Output (TXT format)

```
TensorFlow Model Analysis
========================
Model: my_model.h5
Date: 2025-04-10 14:30:22

Summary:
- Total layers: 15
- Trainable parameters: 1,435,788
- Non-trainable parameters: 256

Layer Types:
- Conv2D: 8
- BatchNormalization: 3
- MaxPooling2D: 2
- Dense: 2

Input Shape: (None, 224, 224, 3)
Output Shape: (None, 10)
```

### Example Output (JSON format)

```json
{
  "model_info": {
    "filename": "my_model.h5",
    "analysis_date": "2025-04-10 14:30:22"
  },
  "summary": {
    "total_layers": 15,
    "trainable_parameters": 1435788,
    "non_trainable_parameters": 256
  },
  "layer_types": {
    "Conv2D": 8,
    "BatchNormalization": 3,
    "MaxPooling2D": 2,
    "Dense": 2
  },
  "shapes": {
    "input": "(None, 224, 224, 3)",
    "output": "(None, 10)"
  }
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
