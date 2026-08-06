from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import dataset_info, save_dataset,remove_duplicates,remove_missing_values,check_duplicates,check_missingValues

games = load_games()
teams = load_teams()
stats = load_teamStats()

dataset_info(games)
check_missingValues(games)
check_duplicates(games)
games = remove_duplicates(games)
games = remove_missing_values(games)
save_dataset(games,"data/processed/games_cleaned.csv")

print("\n\nCleaning the teams dataset...\n")
dataset_info(teams)
check_missingValues(teams)
check_duplicates(teams)
teams = remove_duplicates(teams)
teams = remove_missing_values(teams)
save_dataset(teams,"data/processed/teams_cleaned.csv")

print("\n\nCleaning the teams Stats dataset...\n")
dataset_info(stats)
check_missingValues(stats)
check_duplicates(stats)
stats = remove_duplicates(stats)
stats = remove_missing_values(stats)
save_dataset(stats,"data/processed/teamsStats_cleaned.csv")