
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def tune_all_models(train_data, validation_data, features):

    # Use original, unscaled features.
    # GridSearchCV will handle preprocessing itself.

    train_data = train_data.sort_values("date").copy()

    X_train = train_data[features]
    y_train = train_data["target"]

    X_validation = validation_data[features]
    y_validation = validation_data["target"]

    # Chronological cross-validation.
    # Matches on the same date stay in the same fold.

    unique_dates = train_data["date"].unique()
    time_split = TimeSeriesSplit(n_splits=3)

    cv_splits = []

    for train_dates, validation_dates in time_split.split(unique_dates):

        train_rows = np.where(
            train_data["date"].isin(unique_dates[train_dates])
        )[0]

        validation_rows = np.where(
            train_data["date"].isin(unique_dates[validation_dates])
        )[0]

        cv_splits.append((train_rows, validation_rows))

    draw_weights = {
        "W": 1.0,
        "D": 1.4,
        "L": 1.0
    }

    # ----------------------------------
    # MODELS AND THEIR PARAMETER GRIDS
    # ----------------------------------

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [
                    None,
                    draw_weights,
                    "balanced"
                ]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            {
                "model__n_estimators": [100, 150],
                "model__max_depth": [8, 12],
                "model__min_samples_leaf": [3, 5],
                "model__class_weight": [
                    None,
                    "balanced"
                ]
            }
        ),

        "Gradient Boosting": (
            HistGradientBoostingClassifier(
                random_state=42,
                early_stopping=False
            ),
            {
                "model__max_iter": [100, 150],
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__class_weight": [
                    None,
                    "balanced"
                ]
            }
        )
    }

    results = {}

    # ----------------------------------
    # TUNE EACH MODEL
    # ----------------------------------

    for name, (classifier, parameters) in models.items():

        print("\n====================================")
        print("TUNING:", name)
        print("====================================")

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", classifier)
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring="f1_macro",
            cv=cv_splits,
            n_jobs=1,
            refit=True,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        predictions = search.predict(X_validation)

        accuracy = accuracy_score(
            y_validation,
            predictions
        )

        macro_f1 = f1_score(
            y_validation,
            predictions,
            average="macro"
        )

        actual_draws = y_validation == "D"
        predicted_draws = predictions == "D"

        draw_precision = precision_score(
            actual_draws,
            predicted_draws,
            zero_division=0
        )

        draw_recall = recall_score(
            actual_draws,
            predicted_draws,
            zero_division=0
        )

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nCross-validation Macro F1:")
        print(round(search.best_score_, 4))

        print("\nValidation results:")
        print("Accuracy:", round(accuracy * 100, 2), "%")
        print("Macro F1:", round(macro_f1, 4))
        print("Draw Precision:", round(draw_precision * 100, 2), "%")
        print("Draw Recall:", round(draw_recall * 100, 2), "%")

        results[name] = {
            "model": search.best_estimator_,
            "parameters": search.best_params_,
            "cv_f1": search.best_score_,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "draw_precision": draw_precision,
            "draw_recall": draw_recall
        }

    # ----------------------------------
    # FINAL COMPARISON
    # ----------------------------------

    print("\n========== FINAL MODEL COMPARISON ==========")

    for name, result in results.items():

        print("\nModel:", name)
        print("Accuracy:", round(result["accuracy"] * 100, 2), "%")
        print("Macro F1:", round(result["macro_f1"], 4))
        print("Draw Recall:", round(result["draw_recall"] * 100, 2), "%")

    best_name = max(
        results,
        key=lambda name: results[name]["macro_f1"]
    )

    print("\nBest validation Macro F1:", best_name)

    return results
