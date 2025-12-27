import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from data import load_data, clean_data
from features import build_preprocessor


def train_and_save():
    # --- LOAD & CLEAN ---
    df = clean_data(load_data())

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # --- PREPROCESSING ---
    preprocessor, _, _ = build_preprocessor(X)
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    # --- FIRST MODEL (ALL FEATURES) ---
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_pre, y_train)

    y_proba = model.predict_proba(X_test_pre)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC (all features): {auc:.4f}")

    # --- FEATURE IMPORTANCE ---
    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    top_idx = np.argsort(importances)[-5:]
    top_features = [feature_names[i] for i in top_idx]

    print("Top 5 features used for final model:")
    for f in top_features:
        print(f" - {f}")

    # --- 5 features finales ---
    features_finales = ["InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges"]
    X = df[features_finales]
    y = df['Churn']

    # --- Séparer train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # --- Préprocesseur pour ces 5 features ---
    categorical_features = ["InternetService", "Contract"]
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ])

    # --- Pipeline modèle + préprocessing ---
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # --- Entraînement ---
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"NEW ROC AUC test: {auc:.4f}")

    # --- SAVE MODEL ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "churn_model.pkl"))
    joblib.dump(features_finales, os.path.join(MODEL_DIR, "features.pkl"))

    print(f"✅ Model saved in {MODEL_DIR}")

if __name__ == "__main__":
    train_and_save()