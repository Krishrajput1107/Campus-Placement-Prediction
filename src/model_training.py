# src/model_training.py
from sklearn.linear_model import LogisticRegression
import pickle

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
