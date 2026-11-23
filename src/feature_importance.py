import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


def analyze_feature_importance(
    model,
    validation_data,
    feature_columns,
    model_name
):

    print("\n====================================")
    print("FEATURE IMPORTANCE:", model_name)
    print("====================================")

    # Use the original validation features.
    # The tuned model already contains an imputer and scaler.

    X_validation = validation_data[feature_columns]
    y_validation = validation_data["target"]

    # Calculate the original model performance.

    predictions = model.predict(X_validation)

    baseline_f1 = f1_score(
        y_validation,
        predictions,
        average="macro"
    )

    print("Original Macro F1:", round(baseline_f1, 4))

    # Shuffle each feature 5 times and measure
    # how much the model performance decreases.

    importance = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=1
    )

    # Organize the results into a DataFrame.

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance.importances_mean,
        "Std": importance.importances_std
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Print the 10 most important features.

    print("\nTOP 10 IMPORTANT FEATURES")

    print(
        importance_df.head(10).to_string(index=False)
    )

    # Create output directories.

    os.makedirs(
        "graphs/feature_importance",
        exist_ok=True
    )

    os.makedirs(
        "data/processed/feature_importance",
        exist_ok=True
    )

    file_name = model_name.lower().replace(" ", "_")

    # Save all 32 feature importance scores.

    csv_path = (
        "data/processed/feature_importance/"
        + file_name
        + ".csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    # Create a graph for the top 10 features.

    top_features = importance_df.head(10)

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Decrease in Macro F1")

    plt.ylabel("Features")

    plt.title(
        "Feature Importance - " + model_name
    )

    plt.tight_layout()

    graph_path = (
        "graphs/feature_importance/"
        + file_name
        + ".png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print("\nCSV saved:", csv_path)

    print("Graph saved:", graph_path)

    return importance_df
