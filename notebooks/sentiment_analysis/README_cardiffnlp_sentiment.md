# CardiffNLP Twitter Sentiment Analysis with ONNX Runtime

This directory contains a Python script that loads the CardiffNLP Twitter Sentiment Analysis model into ONNX Runtime for efficient inference with numpy input/output support.

## Features

- ✅ Loads the CardiffNLP Twitter Sentiment Analysis model into ONNX Runtime
- ✅ Accepts numpy arrays as input
- ✅ Provides pre-processing and post-processing extension points
- ✅ Returns numpy arrays as output
- ✅ Designed for easy import into Jupyter notebooks
- ✅ Automatic model conversion from PyTorch to ONNX format
- ✅ Caching of converted models for faster subsequent loads

## Files

- `cardiffnlp_sentiment_onnx.py` - Main script with the CardiffNLPSentimentONNX class
- `cardiffnlp_sentiment_example.ipynb` - Jupyter notebook with usage examples
- `test_cardiffnlp_sentiment.py` - Test script to verify functionality
- `README_cardiffnlp_sentiment.md` - This documentation file

## Installation

The required dependencies are already included in the project's `pyproject.toml` file. To install them:

```bash
# From the project root
uv sync --group dev
```

Or if using pip:

```bash
pip install onnxruntime>=1.15.0 transformers>=4.30.0 torch>=2.0.0
```

## Quick Start

### Basic Usage

```python
from cardiffnlp_sentiment_onnx import quick_sentiment_analysis

# Analyze a single text
result = quick_sentiment_analysis("I love this product!")
print(result['predicted_labels'][0])  # 'positive'

# Analyze multiple texts
texts = ["I love this!", "I hate this!", "It's okay."]
results = quick_sentiment_analysis(texts)
print(results['predicted_labels'])  # ['positive', 'negative', 'neutral']
```

### Advanced Usage

```python
from cardiffnlp_sentiment_onnx import CardiffNLPSentimentONNX

# Initialize the model
model = CardiffNLPSentimentONNX(
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
    providers=['CPUExecutionProvider']  # or ['CUDAExecutionProvider'] for GPU
)

# Make predictions
texts = ["I love this product!", "This is terrible!"]
predictions = model.predict(texts, return_labels=True)

print(predictions['predicted_labels'])  # ['positive', 'negative']
print(predictions['predictions'])  # numpy array of probabilities
```

## Extension Points

### Custom Pre-processing

```python
def my_pre_process(texts):
    """Custom pre-processing function."""
    if isinstance(texts, str):
        texts = [texts]
    
    # Your custom preprocessing logic here
    cleaned_texts = [text.lower().strip() for text in texts]
    
    # Return the required format
    tokenized = model.tokenizer(
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

# Set the custom pre-processing function
model.set_pre_process(my_pre_process)
```

### Custom Post-processing

```python
def my_post_process(logits):
    """Custom post-processing function."""
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Your custom post-processing logic here
    # For example, apply confidence thresholding
    confidence_threshold = 0.8
    max_probs = np.max(probabilities, axis=-1)
    low_confidence_mask = max_probs < confidence_threshold
    probabilities[low_confidence_mask] = 0
    
    return probabilities

# Set the custom post-processing function
model.set_post_process(my_post_process)
```

## Model Information

The script uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model by default, which:

- **Input**: Text strings (single or batch)
- **Output**: Sentiment probabilities for 3 classes: negative, neutral, positive
- **Model Type**: RoBERTa-based transformer
- **Training Data**: Twitter data
- **Performance**: Optimized for social media text sentiment analysis

## Performance

The ONNX Runtime provides significant performance improvements over the original PyTorch model:

- **Faster inference**: 2-5x speedup depending on batch size
- **Lower memory usage**: Optimized model representation
- **Cross-platform**: Works on CPU, GPU, and mobile devices
- **Production-ready**: Stable and reliable for production deployments

## Usage in Jupyter

1. Start Jupyter from the project root:
   ```bash
   uv run jupyter notebook
   ```

2. Navigate to the `notebooks` directory

3. Open `cardiffnlp_sentiment_example.ipynb` for comprehensive examples

4. Or import the script directly in your notebook:
   ```python
   import sys
   sys.path.append('./notebooks')
   from cardiffnlp_sentiment_onnx import CardiffNLPSentimentONNX
   ```

## Testing

Run the test script to verify everything works:

```bash
cd notebooks
python test_cardiffnlp_sentiment.py
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you've installed the required dependencies
2. **Model Download**: The first run will download the model (~500MB) - ensure you have internet connection
3. **Memory Issues**: For large batches, consider processing in smaller chunks
4. **GPU Issues**: If using GPU, ensure CUDA is properly installed

### Model Caching

The converted ONNX model is cached in:
- `~/.cache/huggingface/hub/onnx_models/cardiffnlp_twitter-roberta-base-sentiment-latest/model.onnx`

You can delete this file to force re-conversion if needed.

## API Reference

### CardiffNLPSentimentONNX Class

#### Constructor
```python
CardiffNLPSentimentONNX(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    cache_dir: Optional[str] = None,
    providers: Optional[List[str]] = None,
    session_options: Optional[ort.SessionOptions] = None
)
```

#### Methods

- `predict(texts, return_labels=False)` - Make predictions
- `set_pre_process(pre_process_fn)` - Set custom pre-processing function
- `set_post_process(post_process_fn)` - Set custom post-processing function
- `get_model_info()` - Get model information

### Functions

- `quick_sentiment_analysis(texts, model_name)` - Quick sentiment analysis

## License

This script is part of the Coach Kata project. The underlying CardiffNLP model is subject to its own license terms. 