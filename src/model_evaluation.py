from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")

    print("\nAccuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["W", "D", "L"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print("Label order: W, D, L")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["W", "D", "L"]
        )
    )

    return predictions