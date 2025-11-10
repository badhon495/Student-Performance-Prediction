# Student Performance Prediction

A machine learning project that predicts student final grades using classical ML algorithms on the UCI Student Performance dataset. The dataset includes student performance data from two Portuguese schools, covering Mathematics and Portuguese language courses with demographic, social, and school-related features.

**Course**: CSE427 - Machine Learning | BRAC University  
**Note**: For the most current implementation details, please refer to the Jupyter Notebook. The research paper included in this repository may not reflect the latest code updates.

## Overview

This project compares five machine learning models for multi-class classification of student performance:
- Decision Tree
- SGD Classifier
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes

Each model is evaluated with and without PCA dimensionality reduction to assess performance trade-offs.

## Dataset

**Source**: [Kaggle](https://www.kaggle.com/datasets/whenamancodes/student-performance)
**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Instances**: 1,044 students (merged from Mathematics and Portuguese courses)  
**Features**: 30 attributes including demographic, family background, school-related, and social features  
**Target**: G3 (final grade 0-20), binned into 3 classes for classification

Key feature categories:
- **Demographic**: school, sex, age, address, family size
- **Family**: parental education, parental occupation, guardian
- **Academic**: study time, past failures, extra support, grades (G1, G2)
- **Social**: going out frequency, alcohol consumption, free time, relationships

## Methodology

### Preprocessing
- Merged Mathematics and Portuguese datasets
- Encoded categorical features with ordinal encoding
- Scaled continuous features using StandardScaler
- Applied 70-30 train-test split to avoid data leakage

### Class Imbalance Handling
- Dataset is imbalanced: Class 0 (83), Class 1 (671), Class 2 (294)
- Applied balanced class weights to penalize minority class misclassifications

### Dimensionality Reduction
- Applied PCA retaining 95% variance (32 features → 10 components)
- Fitted on training data only

### Evaluation Metrics
- Accuracy
- Weighted Precision and Recall
- Confusion Matrices

## Results

### Model Accuracies

**Without PCA:**
- Random Forest: 0.88
- Decision Tree: 0.86
- SGD Classifier: 0.84
- SVM: 0.81
- Naive Bayes: 0.75

**With PCA:**
- Random Forest: 0.81
- Naive Bayes: 0.81
- SGD Classifier: 0.80
- SVM: 0.79
- Decision Tree: 0.77

### Key Findings

- **Random Forest** achieved the best overall performance (0.88 without PCA)
- **PCA impact varies by model**: improved Naive Bayes (+0.06), degraded Decision Tree (-0.09)
- All models struggled with minority class (Class 0) despite class weighting
- Weighted precision and recall aligned closely with accuracy across all models

## Future Improvements

- Implement SMOTE or ADASYN for synthetic minority oversampling
- Apply cross-validation for robust evaluation
- Experiment with ensemble methods like Balanced Random Forest
- Perform hyperparameter tuning for optimal performance

## How to Run

1. Clone the repository
2. Open `Student_Performance_Prediction.ipynb` in Google Colab or Jupyter Notebook
3. Run all cells sequentially