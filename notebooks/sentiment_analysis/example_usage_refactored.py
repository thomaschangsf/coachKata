#!/usr/bin/env python3
"""
Example usage of the refactored onnx_memory_tracking.py

This script shows how to use the memory tracking functions with your own
ONNX model and runtime session.
"""

import sys
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Add the notebooks directory to the path
notebooks_dir = Path(__file__).parent
sys.path.insert(0, str(notebooks_dir))

# Import the refactored memory tracking functions
from onnx_memory_tracking import (
    track_with_psutil,
    track_with_tracemalloc,
    track_with_onnx_providers,
    track_disk_io,
    track_with_system_monitor,
    get_onnx_model_info,
    create_onnx_session_with_profiling,
    run_comprehensive_tracking
)

def example_basic_usage():
    """Example 1: Basic usage with your own ONNX model and tokenizer."""
    print("=== Example 1: Basic Usage ===")
    
    # Load your model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find your ONNX model
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create your ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Your test data
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Run basic memory tracking
    metrics = track_with_psutil(onnx_session, tokenizer, test_texts)
    print(f"Basic tracking completed. Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")

def example_with_preprocessed_data():
    """Example 2: Using pre-processed input data."""
    print("\n=== Example 2: Pre-processed Data ===")
    
    # Load your model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find your ONNX model
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create your ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Pre-process your data
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    tokenized = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    input_data = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    }
    
    # Run tracking with pre-processed data
    metrics = track_with_psutil(onnx_session, tokenizer, input_data=input_data)
    print(f"Tracking with pre-processed data completed. Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")

def example_with_profiling():
    """Example 3: Using ONNX profiling."""
    print("\n=== Example 3: ONNX Profiling ===")
    
    # Load your model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find your ONNX model
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create ONNX session with profiling enabled
    profiling_session = create_onnx_session_with_profiling(onnx_path, "my_profile")
    
    # Your test data
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Run profiling
    profile_results = track_with_onnx_providers(profiling_session, tokenizer, test_texts, num_runs=10)
    print(f"Profiling completed. Profile file: {profile_results['profile_file']}")

def example_comprehensive_tracking():
    """Example 4: Comprehensive tracking with all methods."""
    print("\n=== Example 4: Comprehensive Tracking ===")
    
    # Load your model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find your ONNX model
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create your ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Your test data
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Run comprehensive tracking
    results = run_comprehensive_tracking(
        onnx_session=onnx_session,
        tokenizer=tokenizer,
        test_texts=test_texts,
        enable_debug=True,      # Enable tracemalloc
        enable_optimization=False,  # Disable ONNX profiling (would need separate session)
        enable_disk_io=True,    # Enable disk I/O tracking
        enable_real_time=False  # Disable real-time monitoring
    )
    
    print("\nComprehensive tracking results:")
    for method, result in results.items():
        print(f"  {method}: {type(result).__name__}")

def example_custom_model():
    """Example 5: Using with a custom ONNX model."""
    print("\n=== Example 5: Custom Model ===")
    
    # This example shows how to use with any ONNX model
    # Replace this path with your own ONNX model
    custom_onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / "cardiffnlp_twitter-roberta-base-sentiment-latest" / "model.onnx"
    
    if not custom_onnx_path.exists():
        print(f"Custom ONNX model not found at {custom_onnx_path}")
        return
    
    # Load your custom tokenizer (replace with your own)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # Create your ONNX session
    onnx_session = ort.InferenceSession(str(custom_onnx_path))
    
    # Get model information
    model_info = get_onnx_model_info(onnx_session)
    print(f"Model has {len(model_info['inputs'])} inputs and {len(model_info['outputs'])} outputs")
    
    # Your custom test data
    test_texts = ["Custom text 1", "Custom text 2", "Custom text 3"]
    
    # Run tracking
    metrics = track_with_psutil(onnx_session, tokenizer, test_texts)
    print(f"Custom model tracking completed. Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")

def example_jupyter_usage():
    """Example 6: Code you can copy-paste into Jupyter notebooks."""
    print("\n=== Example 6: Jupyter Notebook Usage ===")
    
    # Copy this code into a Jupyter cell:
    jupyter_code = '''
# Jupyter Notebook Cell - Copy and paste this

import sys
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer

# Add notebooks directory to path
sys.path.insert(0, str(Path.cwd() / "notebooks"))

# Import memory tracking functions
from onnx_memory_tracking import track_with_psutil, get_onnx_model_info

# Load your model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Find your ONNX model
onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"

# Create ONNX session
onnx_session = ort.InferenceSession(str(onnx_path))

# Your test data
test_texts = ["I love this product!", "This is terrible!", "It's okay."]

# Run memory tracking
metrics = track_with_psutil(onnx_session, tokenizer, test_texts)
print(f"Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")

# Get model info
model_info = get_onnx_model_info(onnx_session)
'''
    
    print("Copy this code into a Jupyter notebook cell:")
    print(jupyter_code)

if __name__ == "__main__":
    print("Refactored ONNX Memory Tracking Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_with_preprocessed_data()
    example_with_profiling()
    example_comprehensive_tracking()
    example_custom_model()
    example_jupyter_usage()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nKey benefits of the refactored version:")
    print("1. Pass your own ONNX session and tokenizer")
    print("2. Use pre-processed input data")
    print("3. Flexible parameter configuration")
    print("4. Return structured results")
    print("5. Easy integration into existing workflows") 