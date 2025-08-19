#!/usr/bin/env python3
"""
Simple Memory Tracking for ONNX Runtime Inference

This script shows how to track memory usage for the specific ONNX inference code:
    onnx_session = ort.InferenceSession(str(onnx_path))
    outputs = onnx_session.run(...)
"""

import time
import psutil
import tracemalloc
import onnxruntime as ort
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

def track_memory_simple():
    """Simple memory tracking for ONNX inference."""
    
    # Get process info
    process = psutil.Process()
    
    print("=== Simple Memory Tracking ===")
    
    # Record initial state
    print("1. Initial memory state:")
    initial_memory = process.memory_info()
    print(f"   RSS (RAM): {initial_memory.rss / 1024 / 1024:.2f} MB")
    print(f"   VMS (Virtual): {initial_memory.vms / 1024 / 1024:.2f} MB")
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Record memory after loading tokenizer
    print("\n2. After loading tokenizer:")
    tokenizer_memory = process.memory_info()
    print(f"   RSS (RAM): {tokenizer_memory.rss / 1024 / 1024:.2f} MB")
    print(f"   VMS (Virtual): {tokenizer_memory.vms / 1024 / 1024:.2f} MB")
    print(f"   Delta: {(tokenizer_memory.rss - initial_memory.rss) / 1024 / 1024:.2f} MB")
    
    # Start timing and memory tracking for ONNX session creation
    start_time = time.time()
    
    # YOUR CODE: Load ONNX session
    print("\n3. Loading ONNX session...")
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    session_load_time = time.time() - start_time
    session_memory = process.memory_info()
    
    print(f"   Load time: {session_load_time:.4f} seconds")
    print(f"   RSS (RAM): {session_memory.rss / 1024 / 1024:.2f} MB")
    print(f"   VMS (Virtual): {session_memory.vms / 1024 / 1024:.2f} MB")
    print(f"   Delta from initial: {(session_memory.rss - initial_memory.rss) / 1024 / 1024:.2f} MB")
    
    # Prepare test data
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    tokenized = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    # Record memory after tokenization
    print("\n4. After tokenization:")
    tokenized_memory = process.memory_info()
    print(f"   RSS (RAM): {tokenized_memory.rss / 1024 / 1024:.2f} MB")
    print(f"   VMS (Virtual): {tokenized_memory.vms / 1024 / 1024:.2f} MB")
    
    # Start timing and memory tracking for inference
    inference_start_time = time.time()
    
    # YOUR CODE: Run inference
    print("\n5. Running inference...")
    outputs = onnx_session.run(
        None,
        {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    )
    
    inference_time = time.time() - inference_start_time
    inference_memory = process.memory_info()
    
    print(f"   Inference time: {inference_time:.4f} seconds")
    print(f"   RSS (RAM): {inference_memory.rss / 1024 / 1024:.2f} MB")
    print(f"   VMS (Virtual): {inference_memory.vms / 1024 / 1024:.2f} MB")
    print(f"   Delta from session load: {(inference_memory.rss - session_memory.rss) / 1024 / 1024:.2f} MB")
    
    # Process results
    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Final memory state
    print("\n6. Final memory state:")
    final_memory = process.memory_info()
    print(f"   RSS (RAM): {final_memory.rss / 1024 / 1024:.2f} MB")
    print(f"   VMS (Virtual): {final_memory.vms / 1024 / 1024:.2f} MB")
    print(f"   Total delta: {(final_memory.rss - initial_memory.rss) / 1024 / 1024:.2f} MB")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"ONNX Session Load: {session_load_time:.4f}s, {(session_memory.rss - tokenizer_memory.rss) / 1024 / 1024:.2f} MB")
    print(f"Inference: {inference_time:.4f}s, {(inference_memory.rss - session_memory.rss) / 1024 / 1024:.2f} MB")
    print(f"Total: {session_load_time + inference_time:.4f}s, {(final_memory.rss - initial_memory.rss) / 1024 / 1024:.2f} MB")

def track_with_tracemalloc_simple():
    """Using tracemalloc for detailed memory tracking."""
    
    print("\n=== Tracemalloc Memory Tracking ===")
    
    # Start tracemalloc
    tracemalloc.start()
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Take snapshot before ONNX session creation
    snapshot1 = tracemalloc.take_snapshot()
    
    # Load ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Take snapshot after ONNX session creation
    snapshot2 = tracemalloc.take_snapshot()
    
    # Prepare test data
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    tokenized = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    # Take snapshot before inference
    snapshot3 = tracemalloc.take_snapshot()
    
    # Run inference
    outputs = onnx_session.run(
        None,
        {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    )
    
    # Take snapshot after inference
    snapshot4 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    print("Memory allocation for ONNX session creation:")
    stats1 = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in stats1[:3]:
        print(f"  {stat.size_diff / 1024 / 1024:.1f} MB: {stat.traceback.format()}")
    
    print("\nMemory allocation for inference:")
    stats2 = snapshot4.compare_to(snapshot3, 'lineno')
    for stat in stats2[:3]:
        print(f"  {stat.size_diff / 1024 / 1024:.1f} MB: {stat.traceback.format()}")
    
    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent memory: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()

def track_disk_io_simple():
    """Track disk I/O for ONNX operations."""
    
    print("\n=== Disk I/O Tracking ===")
    
    # Get initial disk I/O stats
    initial_io = psutil.disk_io_counters()
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        return
    
    # Load ONNX session (this will read from disk)
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Get disk I/O stats after loading
    after_load_io = psutil.disk_io_counters()
    
    # Prepare test data
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    tokenized = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    # Run inference
    outputs = onnx_session.run(
        None,
        {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    )
    
    # Get final disk I/O stats
    final_io = psutil.disk_io_counters()
    
    print(f"Disk reads for model loading: {(after_load_io.read_bytes - initial_io.read_bytes) / 1024 / 1024:.2f} MB")
    print(f"Disk reads for inference: {(final_io.read_bytes - after_load_io.read_bytes) / 1024 / 1024:.2f} MB")
    print(f"Total disk reads: {(final_io.read_bytes - initial_io.read_bytes) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    import torch  # Import here to avoid issues
    
    print("Simple Memory Tracking for ONNX Runtime")
    print("=" * 40)
    
    # Primary method: psutil (90% of use cases)
    print("Running primary tracking with psutil...")
    track_memory_simple()
    
    # Optional: Enable additional tracking methods when needed
    DEBUG_MEMORY_LEAKS = False  # Set to True when debugging Python memory issues
    MONITOR_DISK_IO = False     # Set to True when analyzing I/O bottlenecks
    OPTIMIZE_MODEL = False      # Set to True when optimizing model architecture (not available in simple version)
    REAL_TIME_MONITORING = False # Set to True for live monitoring (not available in simple version)
    
    if DEBUG_MEMORY_LEAKS:
        print("\n" + "="*40)
        print("DEBUG_MEMORY_LEAKS enabled - Running tracemalloc analysis...")
        track_with_tracemalloc_simple()
    
    if MONITOR_DISK_IO:
        print("\n" + "="*40)
        print("MONITOR_DISK_IO enabled - Running disk I/O analysis...")
        track_disk_io_simple()
    
    if OPTIMIZE_MODEL:
        print("\n" + "="*40)
        print("OPTIMIZE_MODEL enabled - Note: ONNX profiling not available in simple version")
        print("Use onnx_memory_tracking.py for detailed ONNX profiling")
    
    if REAL_TIME_MONITORING:
        print("\n" + "="*40)
        print("REAL_TIME_MONITORING enabled - Note: Real-time monitoring not available in simple version")
        print("Use onnx_memory_tracking.py for real-time system monitoring")
    
    # Always show basic model info
    print("\n" + "="*40)
    print("Basic Model Information:")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    if onnx_path.exists():
        file_size = onnx_path.stat().st_size
        print(f"  Model file size: {file_size / 1024 / 1024:.2f} MB")
        print(f"  Model path: {onnx_path}")
    else:
        print(f"  ONNX model not found at {onnx_path}")
    
    print("\n" + "="*40)
    print("Tracking completed!")
    print("To enable additional tracking methods, set the flags to True:")
    print("  DEBUG_MEMORY_LEAKS = True    # For Python memory debugging")
    print("  MONITOR_DISK_IO = True       # For I/O bottleneck analysis")
    print("  OPTIMIZE_MODEL = True        # For model optimization (use onnx_memory_tracking.py)")
    print("  REAL_TIME_MONITORING = True  # For live monitoring (use onnx_memory_tracking.py)") 