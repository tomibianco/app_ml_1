import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow

mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("Experiment_1")

with mlflow.start_run(run_name = "Test 1"):

    ruta = './data/data.csv'
    df = pd.read_csv(ruta)

    df_cleaned = df.dropna(subset=['linea_sf', 'deuda_sf', 'exp_sf'])
    df_cat = pd.get_dummies(df_cleaned, columns = ['zona', 'nivel_educ', 'vivienda'])
    X1 = df_cat.drop(columns=['mora'])
    y1 = df_cat['mora']
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X1.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X1.columns)

    params = {
        "n_estimators": 10,
        "random_state": 42
    }
    mlflow.log_params(params)

    model = RandomForestClassifier(**params).fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")