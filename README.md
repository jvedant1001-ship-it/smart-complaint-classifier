# Smart Complaint Classification

AI-powered customer complaint classification and routing using
TF-IDF, Logistic Regression and Linear SVM.

## 1. Problem

Organizations receive large numbers of customer complaints through
emails, forms and support systems.

Manually determining which product/category a complaint belongs to
is repetitive and difficult to scale.

This project demonstrates an offline NLP pipeline that predicts the
financial product category associated with a customer complaint.

## 2. Solution

The system:

1. Loads public CFPB complaint narratives.
2. Cleans and normalizes complaint text.
3. Removes duplicates and unusable records.
4. Uses stratified train/validation/test splits.
5. Converts text to TF-IDF unigram/bigram features.
6. Trains Logistic Regression and Linear SVM.
7. Compares the models.
8. Selects a model using validation Macro F1.
9. Evaluates it on an untouched test set.
10. Provides a Streamlit interface for predictions and insights.

## 3. Features

- Complaint classification
- TF-IDF NLP pipeline
- Logistic Regression
- Linear SVM
- Model comparison
- Accuracy
- Macro F1
- Weighted F1
- Confusion matrix
- Category-level metrics
- Representative learned terms
- Interactive category explorer
- Demo complaints
- Streamlit dashboard
- Offline prediction after training

## 4. Architecture

```mermaid
flowchart TD
    A[Complaint Dataset] --> B[Cleaning & Preprocessing]
    B --> C[Train / Validation / Test Split]
    C --> D[TF-IDF]
    D --> E[Logistic Regression]
    D --> F[Linear SVM]
    E --> G[Validation Comparison]
    F --> G
    G --> H[Selected Model]
    H --> I[Held-out Test Evaluation]
    H --> J[Streamlit Application]
    J --> K[Prediction]
    J --> L[Insights]
    J --> M[Explainability]
