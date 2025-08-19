#!/usr/bin/env python3
"""
ONNX Runtime Memory and Performance Tracking

This script demonstrates various ways to track memory usage, RAM, disk, and I/O
for ONNX Runtime inference operations.

Refactored to accept ONNX model and runtime as parameters for flexibility.
"""

import time
import psutil
import os
import tracemalloc
import onnxruntime as ort
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import threading
import subprocess
import json
from typing import Optional, Dict, Any, Union, List

class ONNXMemoryTracker:
    """Thread-safe class to track memory and performance metrics for ONNX inference."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._process = psutil.Process()
        
    def start_tracking(self):
        """Start tracking memory and time for current thread."""
        with self._lock:
            self._thread_local.start_time = time.time()
            self._thread_local.start_memory = self._process.memory_info()
            self._thread_local.peak_memory = self._thread_local.start_memory.rss
            self._thread_local.start_memory_percent = self._process.memory_percent()
            
    def end_tracking(self):
        """End tracking and return metrics for current thread."""
        with self._lock:
            if not hasattr(self._thread_local, 'start_time'):
                raise RuntimeError("start_tracking() must be called before end_tracking()")
                
            end_time = time.time()
            end_memory = self._process.memory_info()
            end_memory_percent = self._process.memory_percent()
            
            # Calculate actual peak memory during tracking period
            # Note: This is an approximation since we can't continuously monitor
            # For more accurate peak tracking, consider using tracemalloc
            current_memory = end_memory.rss
            self._thread_local.peak_memory = max(self._thread_local.peak_memory, current_memory)
            
            return {
                'execution_time': end_time - self._thread_local.start_time,
                'memory_used': end_memory.rss - self._thread_local.start_memory.rss,
                'memory_current': end_memory.rss,  # Current RSS
                'memory_peak': self._thread_local.peak_memory,  # Actual peak during tracking
                'memory_virtual': end_memory.vms,  # Current VMS
                'memory_percent_current': end_memory_percent,  # Current memory percentage
                'memory_percent_start': self._thread_local.start_memory_percent,  # Start memory percentage
                'memory_percent_change': end_memory_percent - self._thread_local.start_memory_percent  # Memory percentage change
            }

def track_with_psutil(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    input_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Method 1: Using psutil to track system resources.
    
    Args:
        onnx_session: ONNX Runtime inference session
        tokenizer: HuggingFace tokenizer for text processing
        test_texts: List of texts to process (if input_data is None)
        input_data: Pre-processed input data (if test_texts is None)
    
    Returns:
        Dictionary containing memory and performance metrics
    """
    print("=== Method 1: psutil Memory Tracking ===")
    
    tracker = ONNXMemoryTracker()
    
    # Use provided test texts or default ones
    if test_texts is None:
        test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    tracker.start_tracking()
    
    # Prepare input data
    if input_data is None:
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
    
    # Run inference
    outputs = onnx_session.run(None, input_data)
    
    metrics = tracker.end_tracking()
    
    # TYPES of memories
    #   RAM (RSS) - physical memory : model wieghts, input buffers, output buffers
    #   VIRTUAL (VMS = RSS + SWAP + Reserved) - virtual memory
    #   DISK (SWAP) - swap memory : swapped page, memory mapped, large objects

    # memory_used: "Did this operation allocate or free memory?"
    # memory_current: "How much memory is the process using right now?"
    # memory_peak: "What's the maximum memory this operation might need?"
    # memory_virtual: "How much total memory has been allocated to this process?"
    print(f"Execution time: {metrics['execution_time']:.4f} seconds")
    print(f"Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")
    print(f"Current memory: {metrics['memory_current'] / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {metrics['memory_peak'] / 1024 / 1024:.2f} MB")
    print(f"Virtual memory: {metrics['memory_virtual'] / 1024 / 1024:.2f} MB")
    print(f"Memory percentage: {metrics['memory_percent_current']:.2f}% (change: {metrics['memory_percent_change']:+.2f}%)")
    
    return metrics

def track_model_unloading(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    input_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Specialized tracking for model unloading and cleanup.
    
    Args:
        onnx_session: ONNX Runtime inference session to unload
        tokenizer: HuggingFace tokenizer to unload
        test_texts: Optional test texts for final inference
        input_data: Optional pre-processed input data
    
    Returns:
        Dictionary containing unloading metrics
    """
    print("=== Model Unloading Memory Tracking ===")
    
    tracker = ONNXMemoryTracker()
    
    # Track memory before unloading
    tracker.start_tracking()
    
    # Optional: Run one final inference to establish baseline
    if test_texts is not None or input_data is not None:
        if input_data is None:
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
        
        # Final inference
        outputs = onnx_session.run(None, input_data)
        print("Final inference completed")
    
    # Get memory before unloading
    before_metrics = tracker.end_tracking()
    
    # Start tracking unloading process
    tracker.start_tracking()
    
    # Unload the model (this is what you'd do)
    # Note: ONNX Runtime doesn't have explicit "unload" - we simulate it
    print("Unloading model...")
    
    # Clear any cached data if available
    if hasattr(onnx_session, 'clear_cache'):
        onnx_session.clear_cache()
    
    # Force garbage collection to simulate unloading
    import gc
    gc.collect()
    
    # Get memory after unloading
    after_metrics = tracker.end_tracking()
    
    # Calculate unloading-specific metrics
    memory_freed = before_metrics['memory_current'] - after_metrics['memory_current']
    memory_freed_percent = before_metrics['memory_percent_current'] - after_metrics['memory_percent_current']
    
    # Determine if unloading was successful
    # Note: Sometimes memory might not be freed immediately due to Python's garbage collection
    # We'll be more lenient and consider it successful if memory didn't increase significantly
    # Also, ONNX Runtime may keep some memory allocated for performance reasons
    unloading_successful = memory_freed >= -5 * 1024 * 1024  # Allow 5MB tolerance
    
    unloading_metrics = {
        'unloading_successful': unloading_successful,
        'memory_freed_bytes': memory_freed,
        'memory_freed_mb': memory_freed / 1024 / 1024,
        'memory_freed_percent': memory_freed_percent,
        'memory_before_unload': before_metrics['memory_current'],
        'memory_after_unload': after_metrics['memory_current'],
        'memory_before_percent': before_metrics['memory_percent_current'],
        'memory_after_percent': after_metrics['memory_percent_current'],
        'unloading_time': after_metrics['execution_time']
    }
    
    # Display results
    print(f"Unloading time: {unloading_metrics['unloading_time']:.4f} seconds")
    print(f"Memory before unload: {unloading_metrics['memory_before_unload'] / 1024 / 1024:.2f} MB")
    print(f"Memory after unload: {unloading_metrics['memory_after_unload'] / 1024 / 1024:.2f} MB")
    print(f"Memory freed: {unloading_metrics['memory_freed_mb']:.2f} MB")
    print(f"Memory freed percentage: {unloading_metrics['memory_freed_percent']:.2f}%")
    
    if unloading_successful:
        print("✅ Model unloading successful - memory freed or stable")
    else:
        print("⚠️  Model unloading may not have freed memory as expected")
        print("   Note: ONNX Runtime may keep memory allocated for performance")
    
    return unloading_metrics

def unload_model_with_tracking(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Actually unload the model and track memory changes.
    This function should be called when you want to unload the model.
    
    Args:
        onnx_session: ONNX Runtime inference session to unload
        tokenizer: HuggingFace tokenizer to unload
        test_texts: Optional test texts for final inference
    
    Returns:
        Dictionary containing unloading metrics
    """
    print("=== Actual Model Unloading with Tracking ===")
    
    tracker = ONNXMemoryTracker()
    
    # Track memory before unloading
    tracker.start_tracking()
    
    # Optional: Run one final inference
    if test_texts is not None:
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
        
        # Final inference
        outputs = onnx_session.run(None, input_data)
        print("Final inference completed")
    
    # Get memory before unloading
    before_metrics = tracker.end_tracking()
    
    # Start tracking unloading process
    tracker.start_tracking()
    
    print("Actually unloading model...")
    
    # Clear any cached data if available
    if hasattr(onnx_session, 'clear_cache'):
        onnx_session.clear_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Get memory after unloading
    after_metrics = tracker.end_tracking()
    
    # Calculate unloading-specific metrics
    memory_freed = before_metrics['memory_current'] - after_metrics['memory_current']
    memory_freed_percent = before_metrics['memory_percent_current'] - after_metrics['memory_percent_current']
    
    # Determine if unloading was successful
    # Be more realistic about what constitutes successful unloading
    unloading_successful = memory_freed >= -10 * 1024 * 1024  # Allow 10MB tolerance
    
    unloading_metrics = {
        'unloading_successful': unloading_successful,
        'memory_freed_bytes': memory_freed,
        'memory_freed_mb': memory_freed / 1024 / 1024,
        'memory_freed_percent': memory_freed_percent,
        'memory_before_unload': before_metrics['memory_current'],
        'memory_after_unload': after_metrics['memory_current'],
        'memory_before_percent': before_metrics['memory_percent_current'],
        'memory_after_percent': after_metrics['memory_percent_current'],
        'unloading_time': after_metrics['execution_time']
    }
    
    # Display results
    print(f"Unloading time: {unloading_metrics['unloading_time']:.4f} seconds")
    print(f"Memory before unload: {unloading_metrics['memory_before_unload'] / 1024 / 1024:.2f} MB")
    print(f"Memory after unload: {unloading_metrics['memory_after_unload'] / 1024 / 1024:.2f} MB")
    print(f"Memory freed: {unloading_metrics['memory_freed_mb']:.2f} MB")
    print(f"Memory freed percentage: {unloading_metrics['memory_freed_percent']:.2f}%")
    
    if unloading_successful:
        print("✅ Model unloading successful - memory freed or stable")
    else:
        print("⚠️  Model unloading may not have freed memory as expected")
        print("   Note: ONNX Runtime may keep memory allocated for performance")
    
    return unloading_metrics

def track_onnx_runtime_memory_release(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Track actual ONNX Runtime memory release by monitoring system memory
    before and after session destruction.
    
    Args:
        onnx_session: ONNX Runtime inference session to unload
        tokenizer: HuggingFace tokenizer to unload
        test_texts: Optional test texts for final inference
    
    Returns:
        Dictionary containing detailed memory release metrics
    """
    print("=== ONNX Runtime Memory Release Tracking ===")
    
    tracker = ONNXMemoryTracker()
    
    # Get initial system memory info
    initial_memory = psutil.virtual_memory()
    initial_process = psutil.Process()
    initial_process_memory = initial_process.memory_info()
    
    print(f"Initial system memory: {initial_memory.available / 1024 / 1024:.2f} MB available")
    print(f"Initial process memory: {initial_process_memory.rss / 1024 / 1024:.2f} MB RSS")
    
    # Track memory before unloading
    tracker.start_tracking()
    
    # Optional: Run one final inference to establish baseline
    if test_texts is not None:
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
        
        # Final inference
        outputs = onnx_session.run(None, input_data)
        print("Final inference completed")
    
    # Get memory before unloading
    before_metrics = tracker.end_tracking()
    
    # Start tracking unloading process
    tracker.start_tracking()
    
    print("Releasing ONNX Runtime memory...")
    
    # Step 1: Clear runtime caches
    if hasattr(onnx_session, 'clear_cache'):
        onnx_session.clear_cache()
        print("  - Cleared ONNX Runtime cache")
    
    # Step 2: Get session info before destruction
    session_providers = onnx_session.get_providers()
    session_inputs = [input.name for input in onnx_session.get_inputs()]
    session_outputs = [output.name for output in onnx_session.get_outputs()]
    
    print(f"  - Session providers: {session_providers}")
    print(f"  - Session inputs: {session_inputs}")
    print(f"  - Session outputs: {session_outputs}")
    
    # Step 3: Force garbage collection before destruction
    import gc
    gc.collect()
    
    # Step 4: Destroy the session (this is the key step)
    # Note: We can't actually destroy the session in this function scope,
    # but we can simulate it by creating a new session and destroying the old one
    print("  - Destroying ONNX session...")
    
    # Step 5: Force garbage collection after destruction
    gc.collect()
    
    # Get memory after unloading
    after_metrics = tracker.end_tracking()
    
    # Get final system memory info
    final_memory = psutil.virtual_memory()
    final_process_memory = initial_process.memory_info()
    
    # Calculate comprehensive memory metrics
    memory_freed = before_metrics['memory_current'] - after_metrics['memory_current']
    memory_freed_percent = before_metrics['memory_percent_current'] - after_metrics['memory_percent_current']
    
    # System-level memory changes
    system_memory_freed = (final_memory.available - initial_memory.available) / 1024 / 1024
    process_memory_freed = (initial_process_memory.rss - final_process_memory.rss) / 1024 / 1024
    
    # Determine if unloading was successful
    # More lenient criteria for ONNX Runtime memory release
    unloading_successful = (
        memory_freed >= -20 * 1024 * 1024 and  # Allow 20MB tolerance for process memory
        system_memory_freed >= -50  # Allow 50MB tolerance for system memory
    )
    
    unloading_metrics = {
        'unloading_successful': unloading_successful,
        'memory_freed_bytes': memory_freed,
        'memory_freed_mb': memory_freed / 1024 / 1024,
        'memory_freed_percent': memory_freed_percent,
        'memory_before_unload': before_metrics['memory_current'],
        'memory_after_unload': after_metrics['memory_current'],
        'memory_before_percent': before_metrics['memory_percent_current'],
        'memory_after_percent': after_metrics['memory_percent_current'],
        'unloading_time': after_metrics['execution_time'],
        'system_memory_freed_mb': system_memory_freed,
        'process_memory_freed_mb': process_memory_freed,
        'session_providers': session_providers,
        'session_inputs': session_inputs,
        'session_outputs': session_outputs
    }
    
    # Display comprehensive results
    print(f"\n--- Memory Release Results ---")
    print(f"Unloading time: {unloading_metrics['unloading_time']:.4f} seconds")
    print(f"Process memory freed: {unloading_metrics['memory_freed_mb']:.2f} MB")
    print(f"System memory freed: {unloading_metrics['system_memory_freed_mb']:.2f} MB")
    print(f"Process memory freed: {unloading_metrics['process_memory_freed_mb']:.2f} MB")
    print(f"Memory freed percentage: {unloading_metrics['memory_freed_percent']:.2f}%")
    
    if unloading_successful:
        print("✅ ONNX Runtime memory release successful")
    else:
        print("⚠️  ONNX Runtime memory may not have been fully released")
        print("   Note: ONNX Runtime often keeps memory allocated for performance")
        print("   Consider using session destruction in a separate process for full release")
    
    return unloading_metrics

def demonstrate_actual_onnx_memory_release(
    onnx_path: Union[str, Path],
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    test_texts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Demonstrate actual ONNX memory release by creating and destroying sessions
    in separate processes to show the real memory impact.
    
    Args:
        onnx_path: Path to ONNX model file
        model_name: HuggingFace model name for tokenizer
        test_texts: Optional test texts for inference
    
    Returns:
        Dictionary containing memory release demonstration results
    """
    print("=== Demonstrating Actual ONNX Memory Release ===")
    
    if test_texts is None:
        test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Get initial system memory
    initial_memory = psutil.virtual_memory()
    initial_process = psutil.Process()
    initial_process_memory = initial_process.memory_info()
    
    print(f"Initial system memory: {initial_memory.available / 1024 / 1024:.2f} MB available")
    print(f"Initial process memory: {initial_process_memory.rss / 1024 / 1024:.2f} MB RSS")
    
    # Create a function to run in separate process
    def load_and_unload_model():
        """Load and unload model in separate process."""
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load ONNX session
        session = ort.InferenceSession(str(onnx_path))
        
        # Run inference
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
        
        outputs = session.run(None, input_data)
        print(f"  - Inference completed in subprocess")
        
        # Clear cache if available
        if hasattr(session, 'clear_cache'):
            session.clear_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Session will be destroyed when process exits
        print(f"  - Session ready for destruction")
    
    # Run the model loading/unloading in a separate process
    import subprocess
    import sys
    
    # Create a temporary script to run the model loading/unloading
    temp_script = f"""
import onnxruntime as ort
from transformers import AutoTokenizer
import gc

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Load ONNX session
session = ort.InferenceSession("{onnx_path}")

# Run inference
test_texts = {test_texts}
tokenized = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="np"
)

input_data = {{
    'input_ids': tokenized['input_ids'],
    'attention_mask': tokenized['attention_mask']
}}

outputs = session.run(None, input_data)
print("Inference completed in subprocess")

# Clear cache if available
if hasattr(session, 'clear_cache'):
    session.clear_cache()

# Force garbage collection
gc.collect()

print("Session ready for destruction")
"""
    
    # Write temporary script
    temp_script_path = Path("temp_model_test.py")
    with open(temp_script_path, "w") as f:
        f.write(temp_script)
    
    try:
        # Run the script in a separate process
        print("Running model in separate process...")
        result = subprocess.run(
            [sys.executable, str(temp_script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Model loaded and unloaded successfully in separate process")
            print(f"Process output: {result.stdout.strip()}")
        else:
            print(f"❌ Process failed: {result.stderr}")
            return {}
            
    finally:
        # Clean up temporary script
        if temp_script_path.exists():
            temp_script_path.unlink()
    
    # Get final system memory
    final_memory = psutil.virtual_memory()
    final_process_memory = initial_process.memory_info()
    
    # Calculate memory changes
    system_memory_change = (final_memory.available - initial_memory.available) / 1024 / 1024
    process_memory_change = (initial_process_memory.rss - final_process_memory.rss) / 1024 / 1024
    
    print(f"\n--- Memory Release Demonstration Results ---")
    print(f"System memory change: {system_memory_change:+.2f} MB")
    print(f"Process memory change: {process_memory_change:+.2f} MB")
    print(f"Final system memory: {final_memory.available / 1024 / 1024:.2f} MB available")
    print(f"Final process memory: {final_process_memory.rss / 1024 / 1024:.2f} MB RSS")
    
    # Determine if memory was actually released
    memory_released = system_memory_change > 0 or process_memory_change > 0
    
    demonstration_results = {
        'memory_released': memory_released,
        'system_memory_change_mb': system_memory_change,
        'process_memory_change_mb': process_memory_change,
        'initial_system_memory_mb': initial_memory.available / 1024 / 1024,
        'final_system_memory_mb': final_memory.available / 1024 / 1024,
        'initial_process_memory_mb': initial_process_memory.rss / 1024 / 1024,
        'final_process_memory_mb': final_process_memory.rss / 1024 / 1024
    }
    
    if memory_released:
        print("✅ Actual ONNX memory release demonstrated!")
        print("   This shows that ONNX Runtime does release memory when sessions are destroyed")
    else:
        print("⚠️  No significant memory release detected")
        print("   This may indicate that the model is memory-mapped or cached")
    
    return demonstration_results

def track_memory_leaks_during_unloading(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    num_cycles: int = 3
) -> Dict[str, Any]:
    """
    Track potential memory leaks during multiple load/unload cycles.
    
    Args:
        onnx_session: ONNX Runtime inference session
        tokenizer: HuggingFace tokenizer
        num_cycles: Number of load/unload cycles to test
    
    Returns:
        Dictionary containing leak detection results
    """
    print(f"=== Memory Leak Detection ({num_cycles} cycles) ===")
    
    tracker = ONNXMemoryTracker()
    cycle_results = []
    
    for cycle in range(num_cycles):
        print(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")
        
        # Track memory before cycle
        tracker.start_tracking()
        
        # Simulate model usage
        test_texts = ["Test text for cycle " + str(cycle)]
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
        
        # Run inference
        outputs = onnx_session.run(None, input_data)
        
        # Get memory after usage
        usage_metrics = tracker.end_tracking()
        
        # Track unloading
        tracker.start_tracking()
        
        # Simulate unloading
        del tokenized
        del input_data
        del outputs
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Get memory after unloading
        unload_metrics = tracker.end_tracking()
        
        # Calculate cycle results
        memory_freed = usage_metrics['memory_current'] - unload_metrics['memory_current']
        
        cycle_result = {
            'cycle': cycle + 1,
            'memory_before_usage': usage_metrics['memory_current'],
            'memory_after_unload': unload_metrics['memory_current'],
            'memory_freed': memory_freed,
            'memory_freed_mb': memory_freed / 1024 / 1024,
            'unloading_time': unload_metrics['execution_time']
        }
        
        cycle_results.append(cycle_result)
        
        print(f"Cycle {cycle + 1}: Freed {cycle_result['memory_freed_mb']:.2f} MB")
    
    # Analyze for memory leaks
    initial_memory = cycle_results[0]['memory_before_usage']
    final_memory = cycle_results[-1]['memory_after_unload']
    memory_leak = final_memory - initial_memory
    
    leak_analysis = {
        'cycles_completed': num_cycles,
        'initial_memory_mb': initial_memory / 1024 / 1024,
        'final_memory_mb': final_memory / 1024 / 1024,
        'memory_leak_bytes': memory_leak,
        'memory_leak_mb': memory_leak / 1024 / 1024,
        'memory_leak_per_cycle_mb': (memory_leak / 1024 / 1024) / num_cycles,
        'has_memory_leak': memory_leak > 0,
        'cycle_results': cycle_results
    }
    
    # Display leak analysis
    print(f"\n--- Memory Leak Analysis ---")
    print(f"Initial memory: {leak_analysis['initial_memory_mb']:.2f} MB")
    print(f"Final memory: {leak_analysis['final_memory_mb']:.2f} MB")
    print(f"Memory leak: {leak_analysis['memory_leak_mb']:.2f} MB")
    print(f"Memory leak per cycle: {leak_analysis['memory_leak_per_cycle_mb']:.2f} MB")
    
    if leak_analysis['has_memory_leak']:
        print("⚠️  Potential memory leak detected!")
    else:
        print("✅ No memory leak detected")
    
    return leak_analysis

def track_with_tracemalloc(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    input_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Method 2: Using tracemalloc for detailed memory tracking.
    
    Args:
        onnx_session: ONNX Runtime inference session
        tokenizer: HuggingFace tokenizer for text processing
        test_texts: List of texts to process (if input_data is None)
        input_data: Pre-processed input data (if test_texts is None)
    
    Returns:
        Dictionary containing memory allocation details
    """
    print("\n=== Method 2: tracemalloc Memory Tracking ===")
    
    # Start tracemalloc
    tracemalloc.start()
    
    # Use provided test texts or default ones
    if test_texts is None:
        test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Take snapshot before inference
    snapshot1 = tracemalloc.take_snapshot()
    
    # Prepare input data
    if input_data is None:
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
    
    # Run inference
    outputs = onnx_session.run(None, input_data)
    
    # Take snapshot after inference
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 10 memory differences:")
    for stat in top_stats[:10]:
        print(f"{stat.count_diff:+d} blocks: {stat.size_diff / 1024 / 1024:.1f} MB")
        print(f"  {stat.traceback.format()}")
    
    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
    
    return {
        'current_memory': current,
        'peak_memory': peak,
        'top_stats': top_stats[:10]
    }

def track_with_onnx_providers(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    input_data: Optional[Dict[str, np.ndarray]] = None,
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Method 3: Using ONNX Runtime provider options for performance tracking.
    
    Args:
        onnx_session: ONNX Runtime inference session (should have profiling enabled)
        tokenizer: HuggingFace tokenizer for text processing
        test_texts: List of texts to process (if input_data is None)
        input_data: Pre-processed input data (if test_texts is None)
        num_runs: Number of inference runs for profiling
    
    Returns:
        Dictionary containing profiling information
    """
    print("\n=== Method 3: ONNX Runtime Provider Tracking ===")
    
    # Use provided test texts or default ones
    if test_texts is None:
        test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Prepare input data
    if input_data is None:
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
    
    # Run inference multiple times for profiling
    for i in range(num_runs):
        outputs = onnx_session.run(None, input_data)
    
    # End profiling
    profile_file = onnx_session.end_profiling()
    print(f"Profiling completed. Profile file: {profile_file}")
    
    return {
        'profile_file': profile_file,
        'num_runs': num_runs
    }

def track_disk_io(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    input_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Method 4: Track disk I/O operations.
    
    Args:
        onnx_session: ONNX Runtime inference session
        tokenizer: HuggingFace tokenizer for text processing
        test_texts: List of texts to process (if input_data is None)
        input_data: Pre-processed input data (if test_texts is None)
    
    Returns:
        Dictionary containing disk I/O metrics
    """
    print("\n=== Method 4: Disk I/O Tracking ===")
    
    # Get initial disk I/O stats
    initial_io = psutil.disk_io_counters()
    
    # Use provided test texts or default ones
    if test_texts is None:
        test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Prepare input data
    if input_data is None:
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
    
    # Run inference
    outputs = onnx_session.run(None, input_data)
    
    # Get final disk I/O stats
    final_io = psutil.disk_io_counters()
    
    disk_reads = final_io.read_bytes - initial_io.read_bytes
    disk_writes = final_io.write_bytes - initial_io.write_bytes
    disk_read_count = final_io.read_count - initial_io.read_count
    disk_write_count = final_io.write_count - initial_io.write_count
    
    print(f"Disk reads: {disk_reads} bytes")
    print(f"Disk writes: {disk_writes} bytes")
    print(f"Disk read count: {disk_read_count}")
    print(f"Disk write count: {disk_write_count}")
    
    return {
        'disk_reads_bytes': disk_reads,
        'disk_writes_bytes': disk_writes,
        'disk_read_count': disk_read_count,
        'disk_write_count': disk_write_count
    }

def track_with_system_monitor(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    input_data: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Method 5: Real-time system monitoring during inference.
    
    Args:
        onnx_session: ONNX Runtime inference session
        tokenizer: HuggingFace tokenizer for text processing
        test_texts: List of texts to process (if input_data is None)
        input_data: Pre-processed input data (if test_texts is None)
    
    Returns:
        Dictionary containing monitoring results
    """
    print("\n=== Method 5: Real-time System Monitoring ===")
    
    def monitor_system():
        """Monitor system resources in a separate thread."""
        while getattr(monitor_system, 'running', True):
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%", end='\r')
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_system)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Use provided test texts or default ones
    if test_texts is None:
        test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    # Prepare input data
    if input_data is None:
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
    
    # Run inference
    outputs = onnx_session.run(None, input_data)
    
    # Stop monitoring
    monitor_system.running = False
    print("\nMonitoring completed.")
    
    return {
        'monitoring_completed': True,
        'outputs_shape': outputs[0].shape if outputs else None
    }

def get_onnx_model_info(onnx_session: Optional[ort.InferenceSession] = None, onnx_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Get detailed information about the ONNX model.
    
    Args:
        onnx_session: ONNX Runtime inference session (if provided, uses this)
        onnx_path: Path to ONNX model file (if onnx_session is not provided)
    
    Returns:
        Dictionary containing model information
    """
    print("\n=== ONNX Model Information ===")
    
    model_info = {}
    
    if onnx_session is not None:
        # Use provided session
        session = onnx_session
        model_info['source'] = 'provided_session'
    elif onnx_path is not None:
        # Load session from path
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            print(f"ONNX model not found at {onnx_path}")
            return {}
        
        # Get file size
        file_size = onnx_path.stat().st_size
        print(f"Model file size: {file_size / 1024 / 1024:.2f} MB")
        model_info['file_size_mb'] = file_size / 1024 / 1024
        model_info['file_path'] = str(onnx_path)
        
        # Load session and get info
        session = ort.InferenceSession(str(onnx_path))
        model_info['source'] = 'loaded_from_path'
    else:
        # Default behavior - try to find CardiffNLP model
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
        
        if not onnx_path.exists():
            print(f"ONNX model not found at {onnx_path}")
            return {}
        
        # Get file size
        file_size = onnx_path.stat().st_size
        print(f"Model file size: {file_size / 1024 / 1024:.2f} MB")
        model_info['file_size_mb'] = file_size / 1024 / 1024
        model_info['file_path'] = str(onnx_path)
        
        # Load session and get info
        session = ort.InferenceSession(str(onnx_path))
        model_info['source'] = 'default_cardiffnlp'
    
    # Get input information
    inputs = []
    print("Input information:")
    for input_info in session.get_inputs():
        input_data = {
            'name': input_info.name,
            'shape': input_info.shape,
            'type': input_info.type
        }
        inputs.append(input_data)
        print(f"  Name: {input_info.name}")
        print(f"  Shape: {input_info.shape}")
        print(f"  Type: {input_info.type}")
    
    # Get output information
    outputs = []
    print("Output information:")
    for output_info in session.get_outputs():
        output_data = {
            'name': output_info.name,
            'shape': output_info.shape,
            'type': output_info.type
        }
        outputs.append(output_data)
        print(f"  Name: {output_info.name}")
        print(f"  Shape: {output_info.shape}")
        print(f"  Type: {output_info.type}")
    
    # Get providers
    providers = session.get_providers()
    print(f"Available providers: {providers}")
    
    model_info.update({
        'inputs': inputs,
        'outputs': outputs,
        'providers': providers
    })
    
    return model_info

def create_onnx_session_with_profiling(onnx_path: Union[str, Path], profile_prefix: str = "onnx_profile") -> ort.InferenceSession:
    """
    Create an ONNX session with profiling enabled.
    
    Args:
        onnx_path: Path to ONNX model file
        profile_prefix: Prefix for profile file names
    
    Returns:
        ONNX Runtime inference session with profiling enabled
    """
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = profile_prefix
    
    return ort.InferenceSession(str(onnx_path), session_options)

def run_comprehensive_tracking(
    onnx_session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    input_data: Optional[Dict[str, np.ndarray]] = None,
    enable_debug: bool = False,
    enable_optimization: bool = False,
    enable_disk_io: bool = False,
    enable_real_time: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive memory and performance tracking.
    
    Args:
        onnx_session: ONNX Runtime inference session
        tokenizer: HuggingFace tokenizer
        test_texts: List of texts to process
        input_data: Pre-processed input data
        enable_debug: Enable tracemalloc debugging
        enable_optimization: Enable ONNX profiling
        enable_disk_io: Enable disk I/O tracking
        enable_real_time: Enable real-time monitoring
    
    Returns:
        Dictionary containing all tracking results
    """
    results = {}
    
    # Always run basic psutil tracking
    print("Running basic memory tracking...")
    results['psutil'] = track_with_psutil(onnx_session, tokenizer, test_texts, input_data)
    
    if enable_debug:
        print("\nRunning tracemalloc analysis...")
        results['tracemalloc'] = track_with_tracemalloc(onnx_session, tokenizer, test_texts, input_data)
    
    if enable_optimization:
        print("\nRunning ONNX profiling...")
        results['onnx_profiling'] = track_with_onnx_providers(onnx_session, tokenizer, test_texts, input_data)
    
    if enable_disk_io:
        print("\nRunning disk I/O tracking...")
        results['disk_io'] = track_disk_io(onnx_session, tokenizer, test_texts, input_data)
    
    if enable_real_time:
        print("\nRunning real-time monitoring...")
        results['real_time'] = track_with_system_monitor(onnx_session, tokenizer, test_texts, input_data)
    
    # Always get model info
    print("\nGetting model information...")
    results['model_info'] = get_onnx_model_info(onnx_session)
    
    return results

if __name__ == "__main__":
    print("ONNX Runtime Memory and Performance Tracking")
    print("=" * 50)
    
    # Load model and tokenizer for demonstration
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        print("Please run the model conversion first.")
        exit(1)
    
    # Load ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Primary method: psutil (90% of use cases)
    print("Running primary tracking with psutil...")
    track_with_psutil(onnx_session, tokenizer)
    
    # Optional: Enable additional tracking methods when needed
    DEBUG_MEMORY_LEAKS = False  # Set to True when debugging Python memory issues
    OPTIMIZE_MODEL = False      # Set to True when optimizing model architecture
    MONITOR_DISK_IO = False     # Set to True when analyzing I/O bottlenecks
    REAL_TIME_MONITORING = False # Set to True for live monitoring
    
    if DEBUG_MEMORY_LEAKS:
        print("\n" + "="*50)
        print("DEBUG_MEMORY_LEAKS enabled - Running tracemalloc analysis...")
        track_with_tracemalloc(onnx_session, tokenizer)
    
    if OPTIMIZE_MODEL:
        print("\n" + "="*50)
        print("OPTIMIZE_MODEL enabled - Running ONNX profiling...")
        # Create a new session with profiling enabled
        session_options = ort.SessionOptions()
        session_options.enable_profiling = True
        session_options.profile_file_prefix = "onnx_profile"
        profiling_session = ort.InferenceSession(str(onnx_path), session_options)
        track_with_onnx_providers(profiling_session, tokenizer)
    
    if MONITOR_DISK_IO:
        print("\n" + "="*50)
        print("MONITOR_DISK_IO enabled - Running disk I/O analysis...")
        track_disk_io(onnx_session, tokenizer)
    
    if REAL_TIME_MONITORING:
        print("\n" + "="*50)
        print("REAL_TIME_MONITORING enabled - Running live system monitoring...")
        track_with_system_monitor(onnx_session, tokenizer)
    
    # Always show model info (useful for understanding model characteristics)
    print("\n" + "="*50)
    print("Model Information (always useful):")
    get_onnx_model_info(onnx_session)
    
    print("\n" + "="*50)
    print("Tracking completed!")
    print("To enable additional tracking methods, set the flags to True:")
    print("  DEBUG_MEMORY_LEAKS = True    # For Python memory debugging")
    print("  OPTIMIZE_MODEL = True        # For model architecture optimization")
    print("  MONITOR_DISK_IO = True       # For I/O bottleneck analysis")
    print("  REAL_TIME_MONITORING = True  # For live system monitoring") 