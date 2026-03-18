
# NLP Toolkit

A Python-based natural language processing toolkit with common NLP tasks like tokenization, stemming, and sentiment analysis.

## Features

- **Tokenization**: Split text into words or sentences.
- **Stemming**: Reduce words to their root form.
- **Sentiment Analysis**: Determine the emotional tone of text.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.nlp_utils import tokenize, stem, analyze_sentiment

text = "The quick brown fox jumps over the lazy dog. This is a great library!"

# Tokenization
tokens = tokenize(text)
print(f"Tokens: {tokens}")

# Stemming
stemmed_tokens = [stem(token) for token in tokens]
print(f"Stemmed: {stemmed_tokens}")

# Sentiment Analysis
sentiment = analyze_sentiment(text)
print(f"Sentiment: {sentiment}")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
