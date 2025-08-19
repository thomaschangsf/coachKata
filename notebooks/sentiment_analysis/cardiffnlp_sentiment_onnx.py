"""
CardiffNLP Twitter Sentiment Analysis ONNX Runtime Script

This script provides a wrapper for the CardiffNLP Twitter Sentiment Analysis model
using ONNX Runtime for efficient inference with numpy input/output support.

Features:
- Loads the CardiffNLP Twitter Sentiment Analysis model into ONNX Runtime
- Accepts numpy arrays as input
- Provides pre-processing and post-processing extension points
- Returns numpy arrays as output
- Designed for easy import into Jupyter notebooks
"""

import os
import logging
from typing import Optional, Callable, Dict, Any, Union, List
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CardiffNLPSentimentONNX:
    """
    CardiffNLP Twitter Sentiment Analysis model wrapper using ONNX Runtime.
    
    This class provides a convenient interface for sentiment analysis with
    pre-processing and post-processing extension points.
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        cache_dir: Optional[str] = None,
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None
    ):
        """
        Initialize the CardiffNLP Sentiment Analysis ONNX model.
        
        Args:
            model_name: HuggingFace model name for CardiffNLP sentiment model
            cache_dir: Directory to cache the model and tokenizer
            providers: ONNX Runtime execution providers (e.g., ['CPUExecutionProvider'])
            session_options: ONNX Runtime session options
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.providers = providers or ['CPUExecutionProvider']
        self.session_options = session_options or ort.SessionOptions()
        
        # Initialize components
        self.tokenizer = None
        self.onnx_session = None
        self.model_path = None
        
        # Extension points
        self.pre_process_fn: Optional[Callable] = None
        self.post_process_fn: Optional[Callable] = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and convert/load the ONNX model."""
        try:
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Check if ONNX model already exists
            model_dir = Path(self.cache_dir) if self.cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
            onnx_model_path = model_dir / "onnx_models" / self.model_name.replace("/", "_") / "model.onnx"
            
            if onnx_model_path.exists():
                logger.info(f"Loading existing ONNX model from {onnx_model_path}")
                self.model_path = str(onnx_model_path)
            else:
                logger.info("Converting PyTorch model to ONNX format")
                self._convert_to_onnx()
            
            # Load ONNX session
            self.onnx_session = ort.InferenceSession(
                self.model_path,
                sess_options=self.session_options,
                providers=self.providers
            )
            
            logger.info(f"Model loaded successfully with providers: {self.providers}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _convert_to_onnx(self):
        """Convert the PyTorch model to ONNX format."""
        try:
            # Load the PyTorch model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            model.eval()
            
            # Create dummy input for ONNX conversion
            dummy_input = self.tokenizer(
                "This is a test sentence.",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Create output directory
            model_dir = Path(self.cache_dir) if self.cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
            onnx_dir = model_dir / "onnx_models" / self.model_name.replace("/", "_")
            onnx_dir.mkdir(parents=True, exist_ok=True)
            
            self.model_path = str(onnx_dir / "model.onnx")
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                self.model_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Model converted and saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error converting model to ONNX: {e}")
            raise
    
    def set_pre_process(self, pre_process_fn: Callable):
        """
        Set a custom pre-processing function.
        
        Args:
            pre_process_fn: Function that takes input data and returns processed input
                          for the model. Should return a dict with 'input_ids' and 
                          'attention_mask' as numpy arrays.
        """
        self.pre_process_fn = pre_process_fn
        logger.info("Pre-processing function set")
    
    def set_post_process(self, post_process_fn: Callable):
        """
        Set a custom post-processing function.
        
        Args:
            post_process_fn: Function that takes model output (numpy array) and 
                           returns processed output.
        """
        self.post_process_fn = post_process_fn
        logger.info("Post-processing function set")
    
    def _default_pre_process(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Default pre-processing function.
        
        Args:
            texts: Input text(s) to process
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' as numpy arrays
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize the texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
    def _default_post_process(self, logits: np.ndarray) -> np.ndarray:
        """
        Default post-processing function.
        
        Args:
            logits: Raw model output logits
            
        Returns:
            Processed output (probabilities)
        """
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probabilities
    
    def predict(
        self, 
        texts: Union[str, List[str]], 
        return_labels: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Perform sentiment analysis prediction.
        
        Args:
            texts: Input text(s) for sentiment analysis
            return_labels: Whether to return label mapping along with predictions
            
        Returns:
            Numpy array of predictions or dict with predictions and labels
        """
        try:
            # Pre-processing
            if self.pre_process_fn:
                model_input = self.pre_process_fn(texts)
            else:
                model_input = self._default_pre_process(texts)
            
            # Run inference
            outputs = self.onnx_session.run(
                None,
                {
                    'input_ids': model_input['input_ids'],
                    'attention_mask': model_input['attention_mask']
                }
            )
            
            # Post-processing
            if self.post_process_fn:
                predictions = self.post_process_fn(outputs[0])
            else:
                predictions = self._default_post_process(outputs[0])
            
            if return_labels:
                # Get label mapping
                labels = ['negative', 'neutral', 'positive']
                return {
                    'predictions': predictions,
                    'labels': labels,
                    'predicted_labels': [labels[np.argmax(pred)] for pred in predictions]
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'providers': self.providers,
            'input_names': [input.name for input in self.onnx_session.get_inputs()],
            'output_names': [output.name for output in self.onnx_session.get_outputs()],
            'has_pre_process': self.pre_process_fn is not None,
            'has_post_process': self.post_process_fn is not None
        }


# Convenience function for quick sentiment analysis
def quick_sentiment_analysis(
    texts: Union[str, List[str]], 
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
) -> Dict[str, Any]:
    """
    Quick sentiment analysis function for simple use cases.
    
    Args:
        texts: Input text(s) for sentiment analysis
        model_name: HuggingFace model name
        
    Returns:
        Dictionary with predictions, labels, and predicted labels
    """
    model = CardiffNLPSentimentONNX(model_name=model_name)
    return model.predict(texts, return_labels=True)


# Example usage and extension point examples
def example_pre_process(texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Example custom pre-processing function.
    
    This function could be used to:
    - Clean and normalize text
    - Apply custom tokenization rules
    - Add domain-specific preprocessing
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Example: Convert to lowercase and remove extra whitespace
    cleaned_texts = [text.lower().strip() for text in texts]
    
    # Use the default tokenizer but with cleaned text
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    tokenized = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    }


def example_post_process(logits: np.ndarray) -> np.ndarray:
    """
    Example custom post-processing function.
    
    This function could be used to:
    - Apply custom thresholding
    - Combine multiple model outputs
    - Add confidence scores
    """
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Example: Add confidence threshold
    confidence_threshold = 0.6
    max_probs = np.max(probabilities, axis=-1)
    
    # Zero out predictions below confidence threshold
    low_confidence_mask = max_probs < confidence_threshold
    probabilities[low_confidence_mask] = 0
    
    return probabilities


if __name__ == "__main__":
    # Example usage
    print("CardiffNLP Sentiment Analysis ONNX Script")
    print("=" * 50)
    
    # Test the model
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special."
    ]
    
    # Quick analysis
    results = quick_sentiment_analysis(test_texts)
    
    print("Quick Analysis Results:")
    for i, text in enumerate(test_texts):
        print(f"Text: {text}")
        print(f"Prediction: {results['predicted_labels'][i]}")
        print(f"Probabilities: {results['predictions'][i]}")
        print("-" * 30)
    
    # Custom preprocessing example
    print("\nCustom Preprocessing Example:")
    model = CardiffNLPSentimentONNX()
    model.set_pre_process(example_pre_process)
    model.set_post_process(example_post_process)
    
    custom_results = model.predict(test_texts, return_labels=True)
    
    for i, text in enumerate(test_texts):
        print(f"Text: {text}")
        print(f"Prediction: {custom_results['predicted_labels'][i]}")
        print(f"Probabilities: {custom_results['predictions'][i]}")
        print("-" * 30) 