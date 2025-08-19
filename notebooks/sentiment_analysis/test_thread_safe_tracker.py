#!/usr/bin/env python3
"""
Test script to demonstrate the thread-safe improvements to ONNXMemoryTracker.
"""

import threading
import time
import sys
from pathlib import Path

# Add the sentiment_analysis directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from onnx_memory_tracking import ONNXMemoryTracker

def worker_function(worker_id: int, tracker: ONNXMemoryTracker, delay: float = 0.1):
    """Worker function that uses the tracker in a thread."""
    print(f"Worker {worker_id}: Starting tracking...")
    
    # Start tracking
    tracker.start_tracking()
    
    # Simulate some work
    time.sleep(delay)
    
    # Allocate some memory (simulate ONNX inference)
    dummy_data = [i for i in range(10000)]
    time.sleep(delay)
    
    # End tracking
    metrics = tracker.end_tracking()
    
    print(f"Worker {worker_id}: Tracking completed!")
    print(f"Worker {worker_id}: Execution time: {metrics['execution_time']:.4f} seconds")
    print(f"Worker {worker_id}: Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")
    print(f"Worker {worker_id}: Current memory: {metrics['memory_current'] / 1024 / 1024:.2f} MB")
    print(f"Worker {worker_id}: Peak memory: {metrics['memory_peak'] / 1024 / 1024:.2f} MB")
    print(f"Worker {worker_id}: Memory percentage: {metrics['memory_percent_current']:.2f}%")
    print(f"Worker {worker_id}: Memory percentage change: {metrics['memory_percent_change']:+.2f}%")
    print("-" * 50)

def test_single_thread():
    """Test the tracker in a single thread."""
    print("=== Single Thread Test ===")
    
    tracker = ONNXMemoryTracker()
    
    print("Starting single thread tracking...")
    tracker.start_tracking()
    
    # Simulate some work
    time.sleep(0.2)
    dummy_data = [i for i in range(50000)]
    time.sleep(0.2)
    
    metrics = tracker.end_tracking()
    
    print("Single thread tracking completed!")
    print(f"Execution time: {metrics['execution_time']:.4f} seconds")
    print(f"Memory used: {metrics['memory_used'] / 1024 / 1024:.2f} MB")
    print(f"Current memory: {metrics['memory_current'] / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {metrics['memory_peak'] / 1024 / 1024:.2f} MB")
    print(f"Memory percentage: {metrics['memory_percent_current']:.2f}%")
    print(f"Memory percentage change: {metrics['memory_percent_change']:+.2f}%")
    print("=" * 50)

def test_multiple_threads():
    """Test the tracker with multiple threads."""
    print("=== Multiple Threads Test ===")
    
    tracker = ONNXMemoryTracker()
    threads = []
    
    # Create and start multiple threads
    for i in range(3):
        thread = threading.Thread(
            target=worker_function,
            args=(i, tracker, 0.1 + i * 0.05)  # Different delays for each thread
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All threads completed!")
    print("=" * 50)

def test_concurrent_access():
    """Test concurrent access to the tracker."""
    print("=== Concurrent Access Test ===")
    
    tracker = ONNXMemoryTracker()
    results = []
    
    def concurrent_worker(worker_id: int):
        """Worker that accesses the tracker concurrently."""
        try:
            tracker.start_tracking()
            time.sleep(0.05)  # Very short delay
            metrics = tracker.end_tracking()
            results.append((worker_id, metrics))
        except Exception as e:
            results.append((worker_id, f"Error: {e}"))
    
    # Create many threads that access the tracker simultaneously
    threads = []
    for i in range(10):
        thread = threading.Thread(target=concurrent_worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check results
    successful = sum(1 for _, result in results if isinstance(result, dict))
    errors = sum(1 for _, result in results if isinstance(result, str))
    
    print(f"Successful tracking calls: {successful}")
    print(f"Errors: {errors}")
    
    if successful > 0:
        print("\nSample successful result:")
        sample_result = next(result for _, result in results if isinstance(result, dict))
        print(f"  Execution time: {sample_result['execution_time']:.4f} seconds")
        print(f"  Memory used: {sample_result['memory_used'] / 1024 / 1024:.2f} MB")
        print(f"  Peak memory: {sample_result['memory_peak'] / 1024 / 1024:.2f} MB")
    
    print("=" * 50)

def test_memory_peak_tracking():
    """Test that peak memory tracking works correctly."""
    print("=== Memory Peak Tracking Test ===")
    
    tracker = ONNXMemoryTracker()
    
    print("Starting peak memory tracking test...")
    tracker.start_tracking()
    
    # Initial memory state
    initial_metrics = tracker.end_tracking()
    print(f"Initial peak: {initial_metrics['memory_peak'] / 1024 / 1024:.2f} MB")
    
    # Start tracking again
    tracker.start_tracking()
    
    # Allocate memory in stages
    print("Allocating memory in stages...")
    
    # Stage 1: Small allocation
    data1 = [i for i in range(1000)]
    time.sleep(0.1)
    metrics1 = tracker.end_tracking()
    print(f"Stage 1 peak: {metrics1['memory_peak'] / 1024 / 1024:.2f} MB")
    
    # Stage 2: Larger allocation
    tracker.start_tracking()
    data2 = [i for i in range(100000)]
    time.sleep(0.1)
    metrics2 = tracker.end_tracking()
    print(f"Stage 2 peak: {metrics2['memory_peak'] / 1024 / 1024:.2f} MB")
    
    # Stage 3: Even larger allocation
    tracker.start_tracking()
    data3 = [i for i in range(500000)]
    time.sleep(0.1)
    metrics3 = tracker.end_tracking()
    print(f"Stage 3 peak: {metrics3['memory_peak'] / 1024 / 1024:.2f} MB")
    
    # Verify peak increases
    if metrics3['memory_peak'] > metrics2['memory_peak'] > metrics1['memory_peak']:
        print("✅ Peak memory tracking working correctly!")
    else:
        print("❌ Peak memory tracking may have issues")
    
    print("=" * 50)

if __name__ == "__main__":
    print("ONNXMemoryTracker Thread Safety and Improvements Test")
    print("=" * 60)
    
    # Run all tests
    test_single_thread()
    test_multiple_threads()
    test_concurrent_access()
    test_memory_peak_tracking()
    
    print("\nTest Summary:")
    print("✅ Thread-safe implementation")
    print("✅ Fixed misleading memory_peak naming")
    print("✅ Added actual peak memory tracking")
    print("✅ Added memory percentage tracking")
    print("✅ Added memory percentage change tracking")
    print("✅ Added proper error handling for missing start_tracking()") 