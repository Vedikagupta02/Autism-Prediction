# Autism Prediction using Machine Learning 🧠

## Overview
This project implements multiple machine learning models to predict Autism Spectrum Disorder (ASD) using behavioral attributes. Comparing the performance of various classifiers, the project aims to provide accurate autism screening predictions to assist healthcare professionals.

## Models Implemented
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

## Dataset
The project uses the Autism Screening Adult Dataset from UCI Machine Learning Repository, consisting of:
- 704 instances
- 21 attributes including:
  - Age
  - Gender
  - Ethnicity
  - Jaundice
  - Family history of autism
  - 10 behavioral features (A1-A10)
  - Screening score

## Features
- Comprehensive data preprocessing and cleaning
- Feature scaling and encoding
- Multiple model implementation and comparison
- Performance evaluation using various metrics
- Cross-validation for robust results

## Model Performance
                  Train accuracy validation accuracy 
Logistic Regression	0.88	0.86
Random Forest	1.00	0.84
XGBoost	1.00	0.83
Support Vector Machine	0.90	0.85
K-Nearest Neighbors	0.89	0.87
Naive Bayes	0.83	0.84
 

## Technologies Used
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Project Structure
```
autism-prediction/
├── Autism_Prediction_using_Machine_Learning.ipynb   # Main notebook
├── dataset/
│   └── Autism-Adult-Data.arff    # Dataset file
└── README.md
```

## How to Run
1. Open the notebook in Google Colab
2. Mount your Google Drive
3. Upload the dataset
4. Run all cells sequentially

## Key Findings
- KNN showed the best performance


## Future Improvements
- Implement deep learning models
- Collect more diverse dataset
- Add feature importance visualization
- Deploy model as a web application


## Contact
For any queries regarding this project, feel free to reach out:
- GitHub: [Vedikagupta02](https://github.com/Vedikagupta02)
