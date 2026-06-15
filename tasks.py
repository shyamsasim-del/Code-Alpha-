import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    """
    Loads the Titanic training dataset from the data directory.
    If the file isn't present, it generates a fallback mock dataset for safety.
    """
    train_path = os.path.join(os.path.dirname(__file__), '../data/train.csv')
    
    if os.path.exists(train_path):
        print(f"--> Loading dataset from {train_path}")
        return pd.read_csv(train_path)
    else:
        print("--> data/train.csv not found. Generating mock Titanic dataset for testing...")
        np.random.seed(42)
        mock_size = 200
        mock_data = {
            'Pclass': np.random.choice([1, 2, 3], size=mock_size),
            'Sex': np.random.choice(['male', 'female'], size=mock_size),
            'Age': np.random.choice([np.nan, 22, 38, 26, 35, 54, 2, 27, 14, 4], size=mock_size),
            'SibSp': np.random.choice([0, 1, 2], size=mock_size),
            'Parch': np.random.choice([0, 1, 2], size=mock_size),
            'Fare': np.random.uniform(7.25, 512.32, size=mock_size),
            'Survived': np.random.choice([0, 1], size=mock_size)
        }
        return pd.DataFrame(mock_data)

def main():
    # 1. Load Data
    df = load_data()
    
    # Separate features and target
    X = df.drop(columns=['Survived']) if 'Survived' in df.columns else df
    y = df['Survived'] if 'Survived' in df.columns else None
    
    if y is None:
        print("Error: Target column 'Survived' not found in dataset.")
        return

    # 2. Define Feature Sub-pipelines
    # Numerical data needs missing values filled with the median
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical data needs missing values filled and text converted to binary flags
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

    # 3. Create the Complete Pipeline (Preprocessing + Classifier)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    # 4. Train-Test Split (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Training
    print("--> Initializing model training pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("--> Model trained successfully.")

    # 6. Evaluation
    y_pred = model_pipeline.predict(X_test)
    print(f"\n======================================")
    print(f"Model Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(f"======================================\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
