from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from src.model_training import train_logistic_regression


def compare_logistic_draw_weights(X_train, y_train, X_validation, y_validation):

    draw_weights = [
        1.0,
        1.2,
        1.4,
        1.6,
        2.0
    ]

    for draw_weight in draw_weights:

        class_weights = {
            "W": 1.0,
            "D": draw_weight,
            "L": 1.0
        }

        model = train_logistic_regression(
            X_train,
            y_train,
            class_weight=class_weights
        )

        predictions = model.predict(X_validation)

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

        draw_recall = recall_score(
            actual_draws,
            predicted_draws,
            zero_division=0
        )

        draw_precision = precision_score(
            actual_draws,
            predicted_draws,
            zero_division=0
        )

        print("\n========== EXPERIMENT ==========")

        print("Draw weight:", draw_weight)

        print("Accuracy:", round(accuracy * 100, 2), "%")

        print("Macro F1:", round(macro_f1, 3))

        print("Draw Recall:", round(draw_recall * 100, 2), "%")

        print("Draw Precision:", round(draw_precision * 100, 2), "%")