from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import clean_dataset
from src.feature_engineering import (
    create_match_features,
    create_target,
    create_team_history,
    add_historical_averages,
    add_match_historical_features
)


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


# Step 4: Final confirmation
print("Feature engineering completed successfully!")
print("Total matches:", len(match_features))
print("Total features:", len(match_features.columns))