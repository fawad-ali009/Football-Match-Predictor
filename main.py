from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import clean_dataset
from src.feature_engineering import (
    create_match_features,
    create_target,
    create_team_history,
    add_historical_averages,
    add_match_historical_features
)
from src.preprocessing import split_by_date, prepare_features
from src.model_training import train_logistic_regression
from src.model_evaluation import evaluate_model


# Step 1: Load datasets
games = load_games()
teams = load_teams()
stats = load_teamStats()


# Step 2: Clean datasets
games = clean_dataset(
    games, "data/processed/games_cleaned.csv"
)

teams = clean_dataset(
    teams, "data/processed/teams_cleaned.csv"
)

stats = clean_dataset(
    stats, "data/processed/teamsStats_cleaned.csv"
)


# Step 3: Feature engineering
match_features = create_match_features(stats)

match_features = create_target(match_features)

team_history = create_team_history(match_features)

team_history = add_historical_averages(team_history)

match_features = add_match_historical_features(
    match_features,
    team_history
)


# =========================
# DATA PREPROCESSING
# =========================

train_data, test_data = split_by_date(match_features)

X_train, X_test, y_train, y_test, imputer, scaler = prepare_features(
    train_data,
    test_data
)

print("Data preprocessing completed successfully!")


# Step 4: Final confirmation
print("Feature engineering completed successfully!")
print("Total matches:", len(match_features))
print("Total features:", len(match_features.columns))

print("\n========== MODEL TRAINING ==========")

print("Training Logistic Regression...")

logistic_model = train_logistic_regression(
    X_train,
    y_train
)

print("Logistic Regression training completed!")


print("\nTesting Logistic Regression...")

predictions = evaluate_model(
    logistic_model,
    X_test,
    y_test
)

print("\n========== SAMPLE PREDICTIONS ==========")

print("Predicted Results:")
print(predictions[:10])

print("\nActual Results:")
print(y_test.iloc[:10].values)


