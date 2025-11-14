#  Toxic-comment-classifier

A small project to classify text comments as "Toxic" or "Not Toxic" using various machine learning models. This app is built in Python and deployed using Streamlit.

## 🚀 Live App

You can try the live, deployed model here:

**https://sarvagya-sharma-toxic-classification-app-vtvkex.streamlit.app/**

## 🤖 Models & Methodology

This project compares several different classification models to find the best-performing one.

### 1. Feature Engineering

1.  **Text Preprocessing:** Comments are cleaned by making them lowercase, removing punctuation, and stripping out non-alphabetic characters.
2.  **Embeddings:** A TF-IDF-weighted document embedding is created for each comment.
    * A `TfidfVectorizer` is fit on the training text.
    * A custom `embedding_matrix` is built using pre-trained GloVe word vectors.
    * The final feature for each comment is a 100-dimension vector created by multiplying its TF-IDF vector by the embedding matrix.

### 2. Models Compared

Four different models were trained and evaluated on these embeddings:

* **Logistic Regression (LR)**
* **Decision Tree (DT)**
* **Random Forest (RF)**
* **A basic Multi-Layer Perceptron (Neural Network)**

### 3. Deployed Model

The **Random Forest (RF)** model was chosen for the final deployment. It provided a strong balance of precision (not wrongly accusing non-toxic comments) and recall (finding actual toxic comments).
