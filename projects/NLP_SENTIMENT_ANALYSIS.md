# NLP Sentiment Analysis — LSTM & DistilBERT

**Area:** Natural Language Processing / Deep Learning  
**Language:** Python  
**Core stack:** pandas, NumPy, NLTK, TensorFlow/Keras, PyTorch, Hugging Face Transformers, scikit-learn

## Project overview
This project builds a three-class sentiment analysis workflow for customer reviews: **negative, neutral and positive**.

The implementation compares two neural NLP approaches:

1. **Bidirectional LSTM** using tokenised and padded text sequences.
2. **DistilBERT Transformer** fine-tuned for three-class sequence classification.

## What I implemented
- Dataset loading and target-label creation from review ratings.
- Text cleaning, stop-word removal and lemmatisation with NLTK.
- Stratified train/test splitting.
- Tokenisation and sequence padding for the LSTM pipeline.
- Bidirectional LSTM with embedding, dropout and dense layers.
- DistilBERT tokenisation and fine-tuning with Hugging Face Transformers.
- Evaluation using classification reports and confusion matrices.
- Side-by-side exploration of recurrent neural networks and transformer-based NLP.

## Skills demonstrated
`Python` · `NLP` · `Deep Learning` · `LSTM` · `Transformers` · `DistilBERT` · `TensorFlow` · `PyTorch` · `scikit-learn`

## Source code
[View nlp.py](../nlp.py)

---
This project was developed as part of my Master's Degree in Applied Artificial Intelligence.