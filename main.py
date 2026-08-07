from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import clean_dataset
from src.statistics import dataset_report
from src.visualization import histogram, boxplot, countplot, scatterplot


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
        "graphs/Histograms" +column+ ".png"
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

