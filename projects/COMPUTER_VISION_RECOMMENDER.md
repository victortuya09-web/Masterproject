# Visual Product Recommendation System — ResNet50

**Area:** Computer Vision / Recommendation Systems  
**Language:** Python  
**Core stack:** TensorFlow/Keras, ResNet50, NumPy, pandas, scikit-learn, Pillow, Matplotlib

## Project overview
A content-based visual recommendation system that represents catalogue images as deep-learning feature vectors and retrieves the most visually similar products to a query image.

## How it works
1. Loads a pretrained **ResNet50** model with ImageNet weights as a feature extractor.
2. Preprocesses catalogue images to the model's expected input format.
3. Extracts and L2-normalises image embeddings.
4. Stores catalogue features for reuse.
5. Computes **cosine similarity** between the query image and catalogue embeddings.
6. Ranks and returns the top-K most similar products.
7. Visualises the query and recommended products with similarity scores.

## Engineering elements
- `pathlib`-based project paths.
- Catalogue validation and missing-image handling.
- Persistent NumPy feature files.
- Reusable functions for feature extraction, similarity calculation and ranking.
- Separation between data preparation, model inference and visualisation.

## Skills demonstrated
`Python` · `Computer Vision` · `Deep Learning` · `ResNet50` · `Transfer Learning` · `Embeddings` · `Cosine Similarity` · `Recommendation Systems`

## Source code
[View cv_proyecto_final.py](../cv_proyecto_final.py)

---
This project was developed as part of my Master's Degree in Applied Artificial Intelligence.