from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import clean_dataset
from src.feature_engineering import (
    create_match_features,
    create_target,
    create_team_history,
    add_historical_averages,
    add_match_historical_features,
    add_match_recent_features,
    add_recent_features
)
from src.preprocessing import split_by_date, prepare_features
from src.model_training import train_logistic_regression, train_random_forest, train_gradient_boosting
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


# =====================================
# FEATURE ENGINEERING
# =====================================

match_features = create_match_features(stats)

match_features = create_target(match_features)

team_history = create_team_history(match_features)

team_history = add_historical_averages(team_history)

team_history = add_recent_features(team_history)

match_features = add_match_historical_features(
    match_features,
    team_history
)

match_features = add_match_recent_features(
    match_features,
    team_history
)

print("\nFeature engineering completed!")

print("Total matches:", len(match_features))

print("Total columns:", len(match_features.columns))

# =========================
# DATA PREPROCESSING
# =========================

train_data, test_data = split_by_date(match_features)

train_data, test_data = split_by_date(match_features)

model_train_data, validation_data = split_by_date(train_data)

X_train, X_validation, y_train, y_validation, imputer, scaler = prepare_features(
    model_train_data,
    validation_data
)


# Step 4: Final confirmation
print("Feature engineering completed successfully!")
print("Total matches:", len(match_features))
print("Total features:", len(match_features.columns))

# =====================================
# LOGISTIC REGRESSION
# =====================================

print("\n========== LOGISTIC REGRESSION ==========")

logistic_model = train_logistic_regression(
    X_train,
    y_train
)

print("Logistic Regression training completed!")

logistic_predictions = evaluate_model(
    logistic_model,
    X_validation,
    y_validation
)

# =====================================
# RANDOM FOREST
# =====================================

print("\n========== RANDOM FOREST ==========")

random_forest_model = train_random_forest(
    X_train,
    y_train
)

print("Random Forest training completed!")

random_forest_predictions = evaluate_model(
    random_forest_model,
    X_validation,
    y_validation
)

# =====================================
# GRADIENT BOOSTING
# =====================================

print("\n========== GRADIENT BOOSTING ==========")

gradient_model = train_gradient_boosting(
    X_train,
    y_train
)

print("Gradient Boosting training completed!")

gradient_predictions = evaluate_model(
    gradient_model,
    X_validation,
    y_validation
)