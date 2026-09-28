from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import clean_dataset
from src.statistics import dataset_report
from src.visualization import histogram, boxplot, countplot, scatterplot, heatmap, barplot
from src.feature_engineering import create_match_features,create_target,create_team_history,add_historical_averages


games = load_games()
teams = load_teams()
stats = load_teamStats()

games = clean_dataset(
    games, "data/processed/games_cleaned.csv"
)

teams = clean_dataset(
    teams, "data/processed/teams_cleaned.csv"
)

stats = clean_dataset(
    stats, "data/processed/teamsStats_cleaned.csv"
)

dataset_report(games)

game_histogram_columns = [
    "homeGoals",
    "awayGoals",
    "homeProbability",
    "drawProbability",
    "awayProbability"
]

teamstats_histogram_columns = [
    "goals",
    "xGoals",
    "shots",
    "shotsOnTarget",
    "deep",
    "ppda",
    "fouls",
    "corners",
    "yellowCards",
    "redCards"
]

for column in game_histogram_columns:
    histogram(
        games,
        column,
        "graphs/Histograms/" +column+ ".png"
    )

for column in teamstats_histogram_columns:
    histogram(
        stats,
        column,
        "graphs/Histograms/" +column+ ".png"
    )

teamstats_boxplot_columns = [
    "goals",
    "xGoals",
    "shots",
    "shotsOnTarget",
    "deep",
    "ppda",
    "fouls",
    "corners",
    "yellowCards",
    "redCards"
]

for column in teamstats_boxplot_columns:

    boxplot(
        stats,
        column,
        "graphs/Boxplots/" + column + ".png"
    )

stats_countplot_columns = [
    "result",
    "location"
]
games_countplot_columns = [
    "season",
    "leagueID"
]

for column in stats_countplot_columns:
    countplot(
        stats,
        column,
        "graphs/counterplots/"+column+".png"
    )

for column in games_countplot_columns:
    countplot(
        games,
        column,
        "graphs/counterplots/"+column+".png"
    )

scatter_plots_columns = [
    ("shots","goals"),
    ("xGoals","goals"),
    ("shotsOnTarget","goals"),
    ("corners","goals")
]
for x_column, y_column in scatter_plots_columns:
    scatterplot(
        stats,
        x_column,
        y_column,
        "graphs/scatterplots/"
        +x_column + "_vs_"
        +y_column + ".png"

    )

heatmap(
    stats,
    "graphs/heatmaps/stats_correlation.png"
)

barplot(
    stats,
    "location",
    "goals",
    "graphs/barplots/average_goals_by_location.png"
)

match_features = create_match_features(stats)

print(match_features.head())
print(match_features.shape)
print(match_features.columns)

match_features = create_target(match_features)

print(match_features[["home_goals","away_goals","target"]].head(10))
print(match_features["target"].value_counts())

team_history = create_team_history(match_features)
print(team_history.head(10))
print(team_history.shape)
print(team_history.columns)
team_history = add_historical_averages(team_history)
print(
    team_history[
        [
            "teamID",
            "date",
            "goals",
            "previous_avg_goals",
            "xGoals",
            "previous_avg_xGoals",
            "shots",
            "previous_avg_shots"
        ]
    ].head(20)
)
team_history = team_history[
    team_history["teamID"] == 177
]
