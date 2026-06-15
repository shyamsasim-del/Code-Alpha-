# Code-Alpha-
To obtain the structural pattern of data science
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Datasets (Placeholder paths for your local structure)
# train_df = pd.read_csv('../data/train.csv')
# For demonstration purposes, we mimic standard Titanic dataset features:
np.random.seed(42)
data_size = 200
mock_data = {
    'Pclass': np.random.choice([1, 2, 3], size=data_size),
    'Sex': np.random.choice(['male', 'female'], size=data_size),
    'Age': np.random.choice([np.nan, 22, 38, 26, 35, 54, 2, 27, 14, 4], size=data_size),
    'SibSp': np.random.choice([0, 1, 2], size=data_size),
    'Parch': np.random.choice([0, 1, 2], size=data_size),
    'Fare': np.random.uniform(7.25, 512.32, size=data_size),
    'Survived': np.random.choice([0, 1], size=data_size)
}
df = pd.DataFrame(mock_data)

# Split features and target variable
X = df.drop(columns=['Survived'])
y = df['Survived']

# 2. Define Preprocessing Pipelines
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_features = ['Sex', 'Pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 3. Create the Main Machine Learning Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

# 4. Train-Test Split & Model Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_pipeline.predict(X_test)
print(f"Model Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))
