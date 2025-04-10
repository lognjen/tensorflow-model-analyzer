"""
Command-line interface for the TensorFlow model analyzer
"""
import os
import sys
import argparse
from .analyzer import ModelAnalyzer

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="TensorFlow Model Analyzer - Extract basic information from TensorFlow models"
    )
    
    parser.add_argument(
        "model_path",
        help="Path to the TensorFlow model (.h5, .pb, or SavedModel directory)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: print to console)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "json"],
        default="txt",
        help="Output format (default: txt, or determined by file extension)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include additional model details"
    )
    
    return parser.parse_args()

def main():
    """
    Main entry point for the CLI.
    """
    args = parse_arguments()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        sys.exit(1)
    
    # Determine output format from file extension if not specified
    output_format = args.format
    if args.output:
        if args.output.lower().endswith('.json'):
            output_format = 'json'
        elif args.output.lower().endswith('.txt'):
            output_format = 'txt'
    
    try:
        # Create analyzer and analyze model
        analyzer = ModelAnalyzer(args.model_path)
        
        # Export results
        output = analyzer.export(args.output, output_format)
        
        # If no output file specified, print to console
        if not args.output:
            print(output)
            
        # Print success message if output was saved to file
        else:
            print(f"Analysis saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
