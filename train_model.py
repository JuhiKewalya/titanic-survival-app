"""Train a simple Titanic survival classifier and save it with joblib.

Run once:  python train_model.py
It creates model.joblib, which app.py loads.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# IMPORTANT: this order must be identical in app.py
FEATURES = ["pclass", "sex", "age", "fare"]

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"


def main():
    df = pd.read_csv(DATA_URL)

    # Keep only the columns we need and clean them
    df = df[["survived", "pclass", "sex", "age", "fare"]].copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["sex"] = df["sex"].map({"female": 1, "male": 0})  # same encoding as app.py
    df = df.dropna()

    X = df[FEATURES]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.2%}")

    joblib.dump(model, "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
