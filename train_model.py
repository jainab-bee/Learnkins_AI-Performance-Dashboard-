import os
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import load_data, MODEL_FEATURES

_HERE      = os.path.dirname(__file__)
DATA_PATH  = os.path.join(_HERE, "data", "learnkins_ai_analytics.csv")
MODEL_PATH = os.path.join(_HERE, "models", "student_risk_model.pkl")


def train():
    print("Loading data...")
    df = load_data(DATA_PATH)

    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features]
    y = df["risk_level"]

    print(f"Training on {len(df)} students with {len(features)} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            max_depth=10,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
