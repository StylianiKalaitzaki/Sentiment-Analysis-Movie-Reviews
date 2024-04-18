# Sentiment Analysis of Movie Reviews

## Overview
A sentiment analysis on movie reviews using two different feature representations—Word Embeddings (Word2Vec) and TF-IDF (Term Frequency-Inverse Document Frequency)—and compare the performance of two classifiers—Naive Bayes and Support Vector Machines (SVM). Sentiment analysis, a subset of natural language processing, involves determining the sentiment polarity of a given text, in this case, movie reviews.

## Dataset
The dataset used in this project is the `movie_reviews` dataset available in the NLTK library. It contains a collection of movie reviews labeled as positive or negative.

## Tools and Technologies Used
- **Python**
- **NLTK**: Natural Language Toolkit library used for accessing the `movie_reviews` dataset.
- **Gensim**: Library used for Word2Vec model training.
- **Scikit-learn**: Library used for machine learning tasks such as classification and evaluation.
- **NumPy**

## Feature Representations
### Word2Vec
Word2Vec is used to learn continuous vector representations of words. Each movie review is represented as the mean vector of its word embeddings.

### TF-IDF
TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique is applied to convert text into numerical feature vectors. TF-IDF reflects the importance of words in documents relative to the entire corpus.

## Classifiers
### Naive Bayes
A Gaussian Naive Bayes classifier is employed for sentiment analysis.

### Support Vector Machines (SVM)
Linear SVM classifier is utilized as an alternative approach for sentiment analysis.

## Evaluation Metrics
The performance of each classifier and feature representation combination is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Results
The results of the evaluation are as follows:
- **Word2Vec:**
  - Naive Bayes Metrics:
    - Accuracy: 0.609
    - Precision: 0.610
    - Recall: 0.609
    - F1 Score: 0.608
  - SVM Metrics:
    - Accuracy: 0.689
    - Precision: 0.689
    - Recall: 0.689
    - F1 Score: 0.689
- **TF-IDF:**
  - Naive Bayes Metrics:
    - Accuracy: 0.657
    - Precision: 0.659
    - Recall: 0.657
    - F1 Score: 0.655
  - SVM Metrics:
    - Accuracy: 0.852
    - Precision: 0.852
    - Recall: 0.852
    - F1 Score: 0.852

## Conclusion
The SVM classifier, particularly when used with TF-IDF features, demonstrated the highest performance across all metrics. These findings suggest that TF-IDF is more effective than Word2Vec for sentiment analysis of movie reviews, and SVM outperforms Naive Bayes in classifying sentiment polarity.
