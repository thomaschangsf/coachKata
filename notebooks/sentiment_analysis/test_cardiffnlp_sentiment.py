#!/usr/bin/env python3
"""
Test script for CardiffNLP Sentiment Analysis ONNX implementation.
"""

import sys
import os

# Add the notebooks directory to the path so we can import our script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from cardiffnlp_sentiment_onnx import CardiffNLPSentimentONNX, quick_sentiment_analysis
    import numpy as np
    
    print("✅ Successfully imported CardiffNLP sentiment analysis modules")
    
    # Test basic functionality
    print("\n🧪 Testing basic sentiment analysis...")
    
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special."
    ]
    
    # Test quick analysis
    results = quick_sentiment_analysis(test_texts)
    
    print("Quick Analysis Results:")
    for i, text in enumerate(test_texts):
        print(f"  Text: {text}")
        print(f"  Prediction: {results['predicted_labels'][i]}")
        print(f"  Probabilities: {results['predictions'][i]}")
        print()
    
    # Test main class
    print("🧪 Testing main class...")
    model = CardiffNLPSentimentONNX()
    
    # Get model info
    model_info = model.get_model_info()
    print(f"Model loaded: {model_info['model_name']}")
    print(f"Providers: {model_info['providers']}")
    
    # Test prediction
    predictions = model.predict(test_texts, return_labels=True)
    
    print("Main Class Results:")
    for i, text in enumerate(test_texts):
        print(f"  Text: {text}")
        print(f"  Prediction: {predictions['predicted_labels'][i]}")
        print(f"  Probabilities: {predictions['predictions'][i]}")
        print()
    
    # Test numpy array output
    print("🧪 Testing numpy array output...")
    raw_predictions = model.predict(test_texts, return_labels=False)
    print(f"Raw predictions type: {type(raw_predictions)}")
    print(f"Raw predictions shape: {raw_predictions.shape}")
    print(f"Raw predictions dtype: {raw_predictions.dtype}")
    
    print("\n✅ All tests passed! The CardiffNLP sentiment analysis ONNX script is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install the required dependencies:")
    print("  pip install onnxruntime transformers torch")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 