
import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_2021_season():

    print("\n========== 2021/22 SEASON EVALUATION ==========")

    # Load actual recorded results.
    actual = pd.read_csv(
        "data/evaluation/actual_results_2021_22.csv"
    )

    # Load our model predictions.
    predicted = pd.read_csv(
        "data/evaluation/predictions_2021_22.csv"
    )

    # Ensure both files contain the same fixture IDs.
    actual_ids = set(actual["match_id"])
    predicted_ids = set(predicted["match_id"])

    if actual_ids != predicted_ids:
        raise ValueError(
            "Actual and predicted fixture IDs do not match."
        )

    if (
        actual["match_id"].duplicated().any()
        or predicted["match_id"].duplicated().any()
    ):
        raise ValueError("Duplicate match IDs found.")

    # Join prediction with corresponding actual result.
    comparison = actual.merge(
        predicted[
            ["match_id", "predicted_result"]
        ],
        on="match_id",
        how="inner",
        validate="one_to_one"
    )

    # Identify correct predictions.
    comparison["correct"] = (
        comparison["actual_result"]
        == comparison["predicted_result"]
    )

    y_true = comparison["actual_result"]
    y_pred = comparison["predicted_result"]


    # =====================================
    # BASELINE MODEL COMPARISON
    # =====================================

    # Find the most common actual result.
    most_common = "W"

    # Simple baseline: always predict that result.
    baseline_predictions = [most_common] * len(y_true)

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_predictions
    )

    baseline_f1 = f1_score(
        y_true,
        baseline_predictions,
        labels=["W", "D", "L"],
        average="macro",
        zero_division=0
    )

    print("\n========== BASELINE COMPARISON ==========")

    print("Baseline strategy: Always predict", most_common)

    print(
        "Baseline accuracy:",
        round(baseline_accuracy * 100, 2), "%"
    )

    print("Baseline Macro F1:", round(baseline_f1, 4))

    # Overall performance.
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=["W", "D", "L"],
        average="macro",
        zero_division=0
    )

    correct = comparison["correct"].sum()

    print("\nTotal matches:", len(comparison))
    print("Correct predictions:", correct)
    print("Incorrect predictions:", len(comparison) - correct)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["W", "D", "L"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print("Order: W, D, L")

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["W", "D", "L"]
        )
    )

    # Evaluate each league separately.
    print("\n========== LEAGUE-WISE RESULTS ==========")

    for league, group in comparison.groupby("league"):

        league_accuracy = accuracy_score(
            group["actual_result"],
            group["predicted_result"]
        )

        league_f1 = f1_score(
            group["actual_result"],
            group["predicted_result"],
            labels=["W", "D", "L"],
            average="macro",
            zero_division=0
        )

        print("\nLeague:", league)
        print("Matches:", len(group))
        print("Accuracy:", round(league_accuracy * 100, 2), "%")
        print("Macro F1:", round(league_f1, 4))

    # Save individual comparison results.
    os.makedirs("data/evaluation", exist_ok=True)

    comparison.to_csv(
        "data/evaluation/comparison_2021_22.csv",
        index=False
    )

    print("\nComparison file saved!")
    print("2021/22 EVALUATION COMPLETED!")

    return comparison
