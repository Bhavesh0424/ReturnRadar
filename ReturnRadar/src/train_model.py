"""Train and evaluate the ReturnRadar demonstration model."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_generator import make_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    data_path = ROOT / "data" / "synthetic_orders.csv"
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        data = make_dataset()
        data.to_csv(data_path, index=False)

    target = "returned"
    X = data.drop(columns=[target])
    y = data[target]

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numerical),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n=== ReturnRadar evaluation (synthetic demo data) ===")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

    model_path = MODEL_DIR / "return_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to: {model_path}")

if __name__ == "__main__":
    main()
