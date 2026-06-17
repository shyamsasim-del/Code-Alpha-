# Code-Alpha-
To obtain the structural pattern of data science
Author Here's the complete repository for your Code Alpha — Data Science Task 1 Titanic Classification project!
# CodeAlpha Data Science Task 1: Titanic Classification

This repository contains the code implementation for the Titanic Classification task. The goal is to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc.).

## Setup Instructions
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the implementation script: `python src/titanic_classification.py`

## Model & Evaluation
- **Algorithm:** Random Forest Classifier
- **Features Used:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Target:** Survived (0 = No, 1 = Yes)

Code Alpha — Data Science Internship | Task 1
Predict survival on the Titanic using classical machine learning
🔗 Live Demo · 📓 Notebook · 📊 Results · 🐛 Report Bug
�

👨‍💻 Author
Field
Details
Name
Shyam.M
Role
Data Science Intern @ Code Alpha
GitHub
@shyamm-dev
Project URL
https://github.com/shyamsasim-del/code-Alpha-
Task
Code Alpha Data Science — Task 1
📖 Project Overview
The Titanic Survival Prediction project is a classic binary classification problem. Using passenger attributes such as age, sex, class, and fare, we train multiple machine learning models to predict whether a passenger survived the tragic 1912 sinking.
###Repository structure
│
├── 📓 model.ipynb               ← Main Jupyter Notebook (full pipeline)
├── 📄 README.md                 ← Project documentation (this file)
├── 📋 requirements.txt          ← Python dependencies
│
├── 📂 data/
│   └── titanic_train.csv        ← Training dataset
│
├── 📂 models/
│   ├── best_model.pkl           ← Saved best model (joblib)
│   ├── scaler.pkl               ← Fitted StandardScaler
│   └── metrics.json             ← Model performance metrics
│
└── 📂 images/
    ├── eda_overview.png         ← EDA visualisation dashboard
    ├── model_performance.png    ← Model comparison dashboard
    ├── feature_importance.png   ← Random Forest feature importance
    └── correlation_heatmap.png  ← Feature correlation heatmap
Raw Data  →  EDA  →  Feature Engineering  →  Model Training  →  Evaluation  →  Deployment
### Features
Exploratory Data Analysis — survival breakdown by class, gender, age, fare, embarkation port & family size
Feature Engineering — FamilySize, IsAlone, AgeGroup, FareBin derived features
Multiple ML Models — Logistic Regression, Random Forest, Gradient Boosting, SVM
Hyperparameter Tuning — tuned estimator count, depth & learning rate
Cross-Validation — Stratified 5-Fold CV for robust evaluation
Rich Visualisations — dark-themed dashboards for EDA and model performance
Model Persistence — best model saved with joblib for inference
##📊 Model Results
Model
Accuracy
AUC-ROC
CV Mean ± Std
Logistic Regression
0.6425
0.6491
0.6655 ± 0.0056
Random Forest ⭐
0.6536
0.6848
0.6756 ± 0.0144
Gradient Boosting
0.6816
0.6364
0.6566 ± 0.0081
SVM (RBF)
0.6536
0.6625
0.6655 ± 0.0217
⭐ Best Model: Random Forest (highest AUC-ROC = 0.6848)3. Models Trained
Logistic Regression — baseline linear model
Random Forest — 200 trees, max_depth=6
Gradient Boosting — 200 estimators, lr=0.05, max_depth=4
🙏 Acknowledgements
Code Alpha — Internship program & task brief
Kaggle Titanic Dataset — Original data source
Scikit-Learn — ML toolkit
Seaborn — Statistical visualisation


