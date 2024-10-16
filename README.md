# Overview
The Fake News Prediction project aims to classify news articles as either fake or real using a machine learning approach. By leveraging natural language processing (NLP) techniques like TF-IDF vectorization and logistic regression, the model is trained to recognize patterns in news data that distinguish fake news from real news.

# Features
Text Preprocessing: Tokenization, stopword removal, and vectorization using TF-IDF (Term Frequency-Inverse Document Frequency).

Classification Model: Logistic Regression is used for binary classification (fake/real).

Performance Metrics: Evaluates model performance using accuracy, precision, recall, and F1 score.

# Dataset
The dataset used for this project consists of news articles labeled as either real or fake. It is split into two parts:

Training Set: Used to train the machine learning model.

Test Set: Used to evaluate the performance of the model.

# Workflow
1. Data Loading: Load the dataset of news articles with their labels.
2. Text Preprocessing:
   
   -Convert text to lowercase.
   
   -Remove punctuation and special characters.
   
   -Tokenize the text and remove stopwords.

4. Feature Extraction:
   
   Use TF-IDF to transform the text data into numerical features.

6. Model Building:
   
   Implement Logistic Regression for classification.

8. Evaluation:
   
   Test the model using the test set and evaluate using performance metrics.

# Installation

1. CLone the repository
   ```bash
   git clone https://github.com/yourusername/Fake-News-Prediction.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd Fake-News-Prediction
   ```
   
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
# Usage

1. Train the model: Run the script to train the model on the provided dataset.
   ```bash
   python train_model.py
   ```

2. Evaluate the model: Evaluate the model's performance on the test data.
   ```bash
   python evaluate_model.py
   ```


