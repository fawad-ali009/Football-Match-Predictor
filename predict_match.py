
import os
import joblib
import pandas as pd

from difflib import get_close_matches

from src.data_loader import load_teams

from src.feature_engineering import (
    create_match_features,
    create_team_history,
    add_recent_features
)

from src.feature_generator import generate_match_features


# =====================================
# 1. LOAD MODEL AND HISTORICAL DATA
# =====================================

def load_prediction_data():

    # Work from the project's main folder
    project_folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_folder)

    # Load the teams and historical statistics
    teams = load_teams()

    stats = pd.read_csv(
        "data/processed/teamsStats_cleaned.csv"
    )

    # Generate team history
    matches = create_match_features(stats)

    history = create_team_history(matches)

    history = add_recent_features(history)

    # Load our already-trained ML model
    saved = joblib.load(
        "models/final_football_predictor.joblib"
    )

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    return teams, history, model, feature_columns


# =====================================
# 2. FIND TEAM ID FROM TEAM NAME
# =====================================

def find_team_id(teams, team_name):

    names = teams["name"].astype(str).str.strip()

    matched = teams[
        names.str.casefold() == team_name.strip().casefold()
    ]

    if not matched.empty:
        return int(matched.iloc[0]["teamID"])

    print("\nTeam not found:", team_name)

    suggestions = get_close_matches(
        team_name.lower(),
        [name.lower() for name in names],
        n=5,
        cutoff=0.4
    )

    if suggestions:

        print("Did you mean:")

        for suggestion in suggestions:
            print(" -", suggestion)

    return None


# =====================================
# 3. PREDICT A FOOTBALL MATCH
# =====================================

def predict_match(
    home_team,
    away_team,
    teams,
    history,
    model,
    feature_columns
):

    # Find corresponding team IDs
    home_id = find_team_id(teams, home_team)
    away_id = find_team_id(teams, away_team)

    if home_id is None or away_id is None:
        return

    if home_id == away_id:
        print("\nHome and away teams cannot be the same.")
        return

    # Automatically generate 32 features
    X = generate_match_features(
        history,
        home_id,
        away_id
    )

    # Arrange features in correct training order
    X = X[feature_columns]

    # Predict match result
    prediction = model.predict(X)[0]

    # Get probability estimates
    probabilities = model.predict_proba(X)[0]

    probability_map = dict(
        zip(model.classes_, probabilities)
    )

    # Convert W, D, L into readable results
    if prediction == "W":
        result = home_team + " Wins"

    elif prediction == "L":
        result = away_team + " Wins"

    else:
        result = "Draw"

    # Display results
    print("\n====================================")
    print("       MATCH PREDICTION RESULT")
    print("====================================")

    print("\nHome Team:", home_team)
    print("Away Team:", away_team)

    print("\nPredicted Result:", result)

    print("\nModel Probability Estimates:")

    print(
        "Home Win:",
        round(probability_map["W"] * 100, 2), "%"
    )

    print(
        "Draw:",
        round(probability_map["D"] * 100, 2), "%"
    )

    print(
        "Away Win:",
        round(probability_map["L"] * 100, 2), "%"
    )

    print("\nNote: These estimates use historical data.")
    print("Probabilities are not calibrated.")

    print("====================================")


# =====================================
# 4. MAIN PROGRAM
# =====================================

def main():

    print("\n====================================")
    print("       FOOTBALL MATCH PREDICTOR")
    print("====================================")

    print("\nLoading saved model and team statistics...")

    teams, history, model, feature_columns = (
        load_prediction_data()
    )

    print("\nPredict a match by entering two teams.")

    while True:

        print("\n------------------------------------")

        home_team = input("Enter Home Team: ").strip()
        away_team = input("Enter Away Team: ").strip()

        if not home_team or not away_team:
            print("Please enter both team names.")
            continue

        try:
            predict_match(
                home_team,
                away_team,
                teams,
                history,
                model,
                feature_columns
            )

        except (ValueError, KeyError) as error:
            print("\nPrediction error:", error)

        again = input(
            "\nPredict another match? (y/n): "
        ).strip().lower()

        if again != "y":
            print("\nThank you for using Football Match Predictor!")
            break


if __name__ == "__main__":
    main()
