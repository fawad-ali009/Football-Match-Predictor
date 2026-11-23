
import os
import joblib
import pandas as pd

from src.feature_generator import generate_match_features


def predict_2021_season(team_history):

    print("\n========== 2021/22 SEASON PREDICTIONS ==========")

    # Load fixtures WITHOUT actual scores.
    fixtures = pd.read_csv(
        "data/evaluation/fixtures_2021_22.csv"
    )

    fixtures["date"] = pd.to_datetime(fixtures["date"])

    # Load the final trained model.
    saved = joblib.load(
        "models/final_football_predictor.joblib"
    )

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    # 2021/22 season begins after this cutoff.
    cutoff_date = pd.Timestamp("2021-07-01")

    # Check that training happened before the season.
    trained_until = pd.to_datetime(saved["trained_until"])

    if trained_until >= cutoff_date:
        raise ValueError(
            "Model was trained on data from the evaluation period!"
        )

    # Prepare historical records available before the season.
    history = team_history.copy()
    history["date"] = pd.to_datetime(history["date"])

    history = history[
        history["date"] < cutoff_date
    ].copy()

    if history.empty:
        raise ValueError("No pre-season history available.")

    if (fixtures["date"] < cutoff_date).any():
        raise ValueError("Fixture is before the evaluation season.")

    if fixtures["match_id"].duplicated().any():
        raise ValueError("Duplicate fixture IDs found.")

    print("Fixtures to predict:", len(fixtures))
    print("Model trained until:", trained_until)
    print("Latest historical match:", history["date"].max())

    predictions = []

    # Predict each fixture independently.
    for _, fixture in fixtures.iterrows():

        home_id = int(fixture["home_team_id"])
        away_id = int(fixture["away_team_id"])

        # Generate 32 features automatically.
        X = generate_match_features(
            history,
            home_id,
            away_id
        )

        # Ensure exact training feature order.
        X = X[feature_columns]

        # Predict W, D or L.
        predicted_result = model.predict(X)[0]

        # Get model probability estimates.
        probabilities = model.predict_proba(X)[0]

        probability_map = dict(
            zip(model.classes_, probabilities)
        )

        predictions.append({
            "match_id": fixture["match_id"],
            "league": fixture["league"],
            "date": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "predicted_result": predicted_result,
            "home_win_probability": probability_map["W"],
            "draw_probability": probability_map["D"],
            "away_win_probability": probability_map["L"]
        })

    # Save all predictions.
    results = pd.DataFrame(predictions)

    os.makedirs("data/evaluation", exist_ok=True)

    output_path = (
        "data/evaluation/predictions_2021_22.csv"
    )

    results.to_csv(output_path, index=False)

    print("\nPredictions generated:", len(results))
    print("Saved to:", output_path)

    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))

    return results
