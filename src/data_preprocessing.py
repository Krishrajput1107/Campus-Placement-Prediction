# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handle categorical columns with binary encoding
    df['ExtracurricularActivities'] = df['ExtracurricularActivities'].map({'Yes': 1, 'No': 0})
    df['PlacementTraining'] = df['PlacementTraining'].map({'Yes': 1, 'No': 0})
    
    # Feature and target separation
    X = df.drop(['StudentID', 'PlacementStatus'], axis=1)
    y = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
