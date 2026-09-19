import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    print("🚀 Starting Model Training & Logging to MLflow...")
    
    # Baseline Reference Data
    X_train = pd.DataFrame({"feat1": [1, 2, 3, 4, 1, 2], "feat2": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2]})
    y_train = [0, 1, 1, 0, 0, 1]
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Enterprise_Fraud_Detection")

    with mlflow.start_run():
        mlflow.log_params(params)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_train, model.predict(X_train))
        mlflow.log_metric("accuracy", acc)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/latest_model.pkl")
        mlflow.sklearn.log_model(model, "model", registered_model_name="FraudClassifier")
        print(f"✅ Model Trained & Logged Successfully. Accuracy: {acc * 100}%")

if __name__ == "__main__":
    train()
  
