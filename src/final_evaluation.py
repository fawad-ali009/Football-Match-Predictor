
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_final_model(train_data, test_data, features):

    print("\n========== FINAL MODEL EVALUATION ==========")

    train_data = train_data.sort_values(["date", "gameID"])
    test_data = test_data.sort_values(["date", "gameID"])

    # Verify that the test period is later.
    last_train_date = pd.to_datetime(train_data["date"]).max()
    first_test_date = pd.to_datetime(test_data["date"]).min()

    if last_train_date >= first_test_date:
        raise ValueError("Training and test dates overlap!")

    if set(train_data["gameID"]) & set(test_data["gameID"]):
        raise ValueError("The same game appears in both datasets!")

    X_train = train_data[features]
    y_train = train_data["target"]

    X_test = test_data[features]
    y_test = test_data["target"]

    print("Training matches:", len(X_train))
    print("Testing matches:", len(X_test))

    # Best parameters selected during validation.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=1000
        ))
    ])

    # Train on the combined training + validation set.
    model.fit(X_train, y_train)

    # Test on the reserved matches.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print("\nFINAL ACCURACY:", round(accuracy * 100, 2), "%")
    print("FINAL MACRO F1:", round(macro_f1, 4))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        predictions,
        labels=["W", "D", "L"],
        zero_division=0
    ))

    print("\nCONFUSION MATRIX:")
    print("Labels: W, D, L")

    print(confusion_matrix(
        y_test,
        predictions,
        labels=["W", "D", "L"]
    ))

    # Save the fitted model and feature names.
    os.makedirs("models", exist_ok=True)

    model_data = {
        "model": model,
        "feature_columns": features,
        "model_name": "Logistic Regression",
        "trained_until": str(last_train_date),
        "stage": "final_evaluated"
    }

    joblib.dump(
        model_data,
        "models/final_football_predictor.joblib"
    )

    print("\nFinal model saved successfully!")

    return model