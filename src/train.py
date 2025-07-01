import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import evaluate_model

# Load data
df = pd.read_csv("data/processed/features_with_target.csv")
X = df.drop(columns=["is_high_risk", "customer_id"])
y = df["is_high_risk"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "logistic_regression": {
        "model": LogisticRegression(),
        "params": {"C": [0.1, 1, 10]}
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
    }
}

# Train and log to MLflow
for name, cfg in models.items():
    with mlflow.start_run(run_name=name):
        grid = GridSearchCV(cfg["model"], cfg["params"], cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)

        # Log params, metrics, and model
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, name + "_model")

        print(f"\n✅ Finished training {name} with metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
