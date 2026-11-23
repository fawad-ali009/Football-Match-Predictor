
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def save_candidate_models(train_data, validation_data, feature_columns):

    os.makedirs("models", exist_ok=True)

    train_data = train_data.sort_values("date")

    X_train = train_data[feature_columns]
    y_train = train_data["target"]

    X_validation = validation_data[feature_columns]
    y_validation = validation_data["target"]

    models = {
        "logistic_regression": LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=1000
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_validation)

        accuracy = accuracy_score(
            y_validation,
            predictions
        )

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro"
        )

        model_data = {
            "model": pipeline,
            "feature_columns": feature_columns,
            "trained_until": str(train_data["date"].max()),
            "stage": "development_candidate"
        }

        file_path = "models/" + name + "_candidate.joblib"

        joblib.dump(model_data, file_path)

        print("\nModel:", name)
        print("Accuracy:", round(accuracy * 100, 2), "%")
        print("Macro F1:", round(macro_f1, 4))
        print("Saved to:", file_path)
