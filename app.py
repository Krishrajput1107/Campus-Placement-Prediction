from flask import Flask, request, render_template
import pickle
import numpy as np
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model_path = 'campus_placement_model.pkl'

# Load and preprocess data for training
file_path = 'data/placementdata.csv'
X, y = load_and_preprocess_data(file_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
print(f"Logistic Regression:\nAccuracy: {metrics['Accuracy']:.2f}, Precision: {metrics['Precision']:.2f}, Recall: {metrics['Recall']:.2f}, F1 Score: {metrics['F1 Score']:.2f}, ROC AUC: {metrics['ROC AUC']:.2f}")

# Save model
save_model(model, model_path)

# Load the trained model for prediction
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    
    # Standard scaling
    scaler = StandardScaler()
    final_features_scaled = scaler.fit_transform(final_features)
    
    prediction = model.predict(final_features_scaled)
    output = 'placed' if prediction[0] == 1 else 'not placed'
    
    return render_template('result.html', prediction_text=f'Student will be {output}')

if __name__ == "__main__":
    app.run(debug=True)
