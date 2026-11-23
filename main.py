
import matplotlib
matplotlib.use("Agg")

from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import clean_dataset

from src.feature_engineering import (
    create_match_features,
    create_target,
    create_team_history,
    add_historical_averages,
    add_recent_features,
    add_match_historical_features,
    add_match_recent_features
)

from src.preprocessing import split_by_date, prepare_features

from src.model_training import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting
)

from src.model_evaluation import evaluate_model
from src.model_optimization import tune_all_models
from src.feature_importance import analyze_feature_importance
from src.model_storage import save_candidate_models


# =====================================
# SELECT THE MODE
# =====================================

MODE = "save"   # Options: "train" or "tune"


# =====================================
# 1. LOAD DATASETS
# =====================================

games = load_games()
teams = load_teams()
stats = load_teamStats()


# =====================================
# 2. CLEAN DATASETS
# =====================================

games = clean_dataset(
    games,
    "data/processed/games_cleaned.csv"
)

teams = clean_dataset(
    teams,
    "data/processed/teams_cleaned.csv"
)

stats = clean_dataset(
    stats,
    "data/processed/teamsStats_cleaned.csv"
)


# =====================================
# 3. FEATURE ENGINEERING
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


# =====================================
# 4. CHRONOLOGICAL SPLIT
# =====================================

train_data, test_data = split_by_date(match_features)

model_train_data, validation_data = split_by_date(train_data)

print("\nTraining records:", len(model_train_data))
print("Validation records:", len(validation_data))
print("Reserved testing records:", len(test_data))


# =====================================
# 5. PREPARE FEATURE LIST
# =====================================

X_train, X_validation, y_train, y_validation, imputer, scaler = prepare_features(
    model_train_data,
    validation_data
)

feature_columns = X_train.columns.tolist()

print("\nTotal model features:", len(feature_columns))

if len(feature_columns) != 32:
    raise ValueError("Expected 32 model features")


# =====================================
# 6. RUN SELECTED MODE
# =====================================


if MODE == "tune":

    print("\n========== HYPERPARAMETER TUNING ==========")

    results = tune_all_models(
        model_train_data,
        validation_data,
        feature_columns
    )

    # =====================================
    # FEATURE IMPORTANCE ANALYSIS
    # =====================================

    print("\n========== FEATURE IMPORTANCE ==========")

    logistic_importance = analyze_feature_importance(
        results["Logistic Regression"]["model"],
        validation_data,
        feature_columns,
        "Logistic Regression"
    )

    forest_importance = analyze_feature_importance(
        results["Random Forest"]["model"],
        validation_data,
        feature_columns,
        "Random Forest"
    )


elif MODE == "train":

    print("\n========== LOGISTIC REGRESSION ==========")

    logistic_model = train_logistic_regression(
        X_train,
        y_train
    )

    evaluate_model(
        logistic_model,
        X_validation,
        y_validation
    )

    print("\n========== RANDOM FOREST ==========")

    forest_model = train_random_forest(
        X_train,
        y_train
    )

    evaluate_model(
        forest_model,
        X_validation,
        y_validation
    )

    print("\n========== GRADIENT BOOSTING ==========")

    boosting_model = train_gradient_boosting(
        X_train,
        y_train
    )

    evaluate_model(
        boosting_model,
        X_validation,
        y_validation
    )

elif MODE == "save":

    print("\n========== SAVING CANDIDATE MODELS ==========")

    save_candidate_models(
        model_train_data,
        validation_data,
        feature_columns
    )

else:

    raise ValueError("MODE must be 'train' or 'tune'")

print("\nPROJECT EXECUTION COMPLETED!")
