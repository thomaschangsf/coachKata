#!/usr/bin/env python3
"""
Example script demonstrating model unloading memory tracking.
"""

import sys
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer

# Add the sentiment_analysis directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from onnx_memory_tracking import (
    track_with_psutil,
    track_model_unloading,
    unload_model_with_tracking,
    track_onnx_runtime_memory_release,
    demonstrate_actual_onnx_memory_release,
    track_memory_leaks_during_unloading
)

def example_basic_unloading():
    """Example 1: Basic model unloading tracking."""
    print("=== Example 1: Basic Model Unloading ===")
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Test texts
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Track model unloading
    unloading_metrics = track_model_unloading(
        onnx_session=onnx_session,
        tokenizer=tokenizer,
        test_texts=test_texts
    )
    
    print(f"\nUnloading Summary:")
    print(f"Success: {unloading_metrics['unloading_successful']}")
    print(f"Memory freed: {unloading_metrics['memory_freed_mb']:.2f} MB")
    print(f"Memory freed percentage: {unloading_metrics['memory_freed_percent']:.2f}%")

def example_actual_unloading():
    """Example 1b: Actual model unloading with tracking."""
    print("\n=== Example 1b: Actual Model Unloading ===")
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Test texts
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Actually unload the model with tracking
    unloading_metrics = unload_model_with_tracking(
        onnx_session=onnx_session,
        tokenizer=tokenizer,
        test_texts=test_texts
    )
    
    print(f"\nActual Unloading Summary:")
    print(f"Success: {unloading_metrics['unloading_successful']}")
    print(f"Memory freed: {unloading_metrics['memory_freed_mb']:.2f} MB")
    print(f"Memory freed percentage: {unloading_metrics['memory_freed_percent']:.2f}%")
    
    # Note: After this function, the model is actually unloaded
    print("Note: Model has been unloaded and is no longer available for use")

def example_onnx_runtime_memory_release():
    """Example 1c: Track actual ONNX Runtime memory release."""
    print("\n=== Example 1c: ONNX Runtime Memory Release ===")
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Test texts
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Track ONNX Runtime memory release
    release_metrics = track_onnx_runtime_memory_release(
        onnx_session=onnx_session,
        tokenizer=tokenizer,
        test_texts=test_texts
    )
    
    print(f"\nONNX Runtime Release Summary:")
    print(f"Success: {release_metrics['unloading_successful']}")
    print(f"Process memory freed: {release_metrics['memory_freed_mb']:.2f} MB")
    print(f"System memory freed: {release_metrics['system_memory_freed_mb']:.2f} MB")
    print(f"Session providers: {release_metrics['session_providers']}")

def example_actual_memory_release_demonstration():
    """Example 1d: Demonstrate actual memory release using separate process."""
    print("\n=== Example 1d: Actual Memory Release Demonstration ===")
    
    # Find ONNX model path
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Test texts
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Demonstrate actual memory release
    demonstration_results = demonstrate_actual_onnx_memory_release(
        onnx_path=onnx_path,
        model_name=model_name,
        test_texts=test_texts
    )
    
    print(f"\nMemory Release Demonstration Summary:")
    print(f"Memory released: {demonstration_results['memory_released']}")
    print(f"System memory change: {demonstration_results['system_memory_change_mb']:+.2f} MB")
    print(f"Process memory change: {demonstration_results['process_memory_change_mb']:+.2f} MB")

def example_memory_leak_detection():
    """Example 2: Memory leak detection during multiple cycles."""
    print("\n=== Example 2: Memory Leak Detection ===")
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Track memory leaks over multiple cycles
    leak_analysis = track_memory_leaks_during_unloading(
        onnx_session=onnx_session,
        tokenizer=tokenizer,
        num_cycles=5
    )
    
    print(f"\nLeak Analysis Summary:")
    print(f"Cycles completed: {leak_analysis['cycles_completed']}")
    print(f"Has memory leak: {leak_analysis['has_memory_leak']}")
    print(f"Total memory leak: {leak_analysis['memory_leak_mb']:.2f} MB")
    print(f"Memory leak per cycle: {leak_analysis['memory_leak_per_cycle_mb']:.2f} MB")

def example_comparison_loading_vs_unloading():
    """Example 3: Compare loading vs unloading memory patterns."""
    print("\n=== Example 3: Loading vs Unloading Comparison ===")
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Test texts
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Track loading/inference
    print("--- Loading and Inference ---")
    loading_metrics = track_with_psutil(
        onnx_session=onnx_session,
        tokenizer=tokenizer,
        test_texts=test_texts
    )
    
    # Track unloading
    print("\n--- Unloading ---")
    unloading_metrics = track_model_unloading(
        onnx_session=onnx_session,
        tokenizer=tokenizer,
        test_texts=test_texts
    )
    
    # Compare results
    print(f"\n--- Comparison ---")
    print(f"Loading memory used: {loading_metrics['memory_used'] / 1024 / 1024:.2f} MB")
    print(f"Unloading memory freed: {unloading_metrics['memory_freed_mb']:.2f} MB")
    
    efficiency = abs(unloading_metrics['memory_freed_mb'] / (loading_metrics['memory_used'] / 1024 / 1024)) * 100
    print(f"Memory cleanup efficiency: {efficiency:.1f}%")

def example_manual_unloading_tracking():
    """Example 4: Manual unloading tracking with custom logic."""
    print("\n=== Example 4: Manual Unloading Tracking ===")
    
    from onnx_memory_tracking import ONNXMemoryTracker
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Create ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Create tracker
    tracker = ONNXMemoryTracker()
    
    # Track before unloading
    tracker.start_tracking()
    
    # Your custom unloading logic here
    print("Performing custom unloading steps...")
    
    # Step 1: Clear any cached data
    if hasattr(onnx_session, 'clear_cache'):
        onnx_session.clear_cache()
    
    # Step 2: Delete session
    del onnx_session
    
    # Step 3: Delete tokenizer
    del tokenizer
    
    # Step 4: Force garbage collection
    import gc
    gc.collect()
    
    # Get metrics
    metrics = tracker.end_tracking()
    
    print(f"Custom unloading completed:")
    print(f"Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")
    print(f"Current memory: {metrics['memory_current'] / 1024 / 1024:.2f} MB")
    print(f"Memory percentage change: {metrics['memory_percent_change']:+.2f}%")

if __name__ == "__main__":
    print("Model Unloading Memory Tracking Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_unloading()
    example_actual_unloading()
    example_onnx_runtime_memory_release()
    example_actual_memory_release_demonstration()
    example_memory_leak_detection()
    example_comparison_loading_vs_unloading()
    example_manual_unloading_tracking()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nKey insights for model unloading:")
    print("1. Focus on 'memory_used' (should be negative)")
    print("2. Monitor 'memory_current' for final footprint")
    print("3. Use 'memory_percent_change' for relative impact")
    print("4. Run multiple cycles to detect memory leaks")
    print("5. Compare loading vs unloading for efficiency") 