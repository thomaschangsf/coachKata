#!/usr/bin/env python3
"""
Test script to verify model unloading tracking works correctly.
"""

import sys
from pathlib import Path
import onnxruntime as ort
from transformers import AutoTokenizer

# Add the sentiment_analysis directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from onnx_memory_tracking import track_model_unloading, unload_model_with_tracking

def test_unloading_tracking():
    """Test the model unloading tracking functionality."""
    print("Testing Model Unloading Tracking")
    print("=" * 40)
    
    # Load model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find ONNX model path
    onnx_path = Path.home() / ".cache" / "huggingface" / "hub" / "onnx_models" / model_name.replace("/", "_") / "model.onnx"
    
    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}")
        print("Please run the model conversion first.")
        return False
    
    # Create ONNX session
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Test texts
    test_texts = ["I love this product!", "This is terrible!", "It's okay."]
    
    print("1. Testing basic unloading tracking...")
    try:
        unloading_metrics = track_model_unloading(
            onnx_session=onnx_session,
            tokenizer=tokenizer,
            test_texts=test_texts
        )
        
        print(f"✅ Basic unloading tracking completed")
        print(f"   Success: {unloading_metrics['unloading_successful']}")
        print(f"   Memory freed: {unloading_metrics['memory_freed_mb']:.2f} MB")
        
        # Check if the function returns expected keys
        expected_keys = [
            'unloading_successful', 'memory_freed_bytes', 'memory_freed_mb',
            'memory_freed_percent', 'memory_before_unload', 'memory_after_unload',
            'memory_before_percent', 'memory_after_percent', 'unloading_time'
        ]
        
        missing_keys = [key for key in expected_keys if key not in unloading_metrics]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return False
        else:
            print(f"✅ All expected keys present")
        
    except Exception as e:
        print(f"❌ Basic unloading tracking failed: {e}")
        return False
    
    print("\n2. Testing actual unloading tracking...")
    try:
        # Create new session and tokenizer for actual unloading test
        tokenizer2 = AutoTokenizer.from_pretrained(model_name)
        onnx_session2 = ort.InferenceSession(str(onnx_path))
        
        unloading_metrics2 = unload_model_with_tracking(
            onnx_session=onnx_session2,
            tokenizer=tokenizer2,
            test_texts=test_texts
        )
        
        print(f"✅ Actual unloading tracking completed")
        print(f"   Success: {unloading_metrics2['unloading_successful']}")
        print(f"   Memory freed: {unloading_metrics2['memory_freed_mb']:.2f} MB")
        
    except Exception as e:
        print(f"❌ Actual unloading tracking failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_unloading_tracking()
    if success:
        print("\n🎉 Model unloading tracking is working correctly!")
    else:
        print("\n❌ Model unloading tracking has issues.")
        sys.exit(1) 