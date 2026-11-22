from src.data_loader import load_games, load_teams, load_teamStats
from src.data_cleaning import clean_dataset
from src.statistics import dataset_report

from src.visualization import (
    histogram,
    boxplot,
    countplot,
    scatterplot,
    heatmap,
    barplot
)

from src.feature_engineering import (
    create_match_features,
    create_target,
    create_team_history,
    add_historical_averages,
    add_match_historical_features,
    add_recent_features,
    add_match_recent_features
)

from src.preprocessing import prepare_features, split_by_date


# =====================================
# GRAPH GENERATION
# =====================================

def generate_graphs(games, stats):

    game_histogram_columns = [
        "homeGoals",
        "awayGoals",
        "homeProbability",
        "drawProbability",
        "awayProbability"
    ]

    teamstats_numeric_columns = [
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

    # Histograms
    for column in game_histogram_columns:

        histogram(
            games,
            column,
            "graphs/Histograms/" + column + ".png"
        )

    for column in teamstats_numeric_columns:

        histogram(
            stats,
            column,
            "graphs/Histograms/" + column + ".png"
        )

    # Boxplots
    for column in teamstats_numeric_columns:

        boxplot(
            stats,
            column,
            "graphs/Boxplots/" + column + ".png"
        )

    # Countplots
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
            "graphs/counterplots/" + column + ".png"
        )

    for column in games_countplot_columns:

        countplot(
            games,
            column,
            "graphs/counterplots/" + column + ".png"
        )

    # Scatterplots
    scatter_plots_columns = [
        ("shots", "goals"),
        ("xGoals", "goals"),
        ("shotsOnTarget", "goals"),
        ("corners", "goals")
    ]

    for x_column, y_column in scatter_plots_columns:

        scatterplot(
            stats,
            x_column,
            y_column,
            "graphs/scatterplots/"
            + x_column + "_vs_"
            + y_column + ".png"
        )

    # Heatmap
    heatmap(
        stats,
        "graphs/heatmaps/stats_correlation.png"
    )

    # Barplot
    barplot(
        stats,
        "location",
        "goals",
        "graphs/barplots/average_goals_by_location.png"
    )


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
# 3. DATASET STATISTICS
# =====================================

print("\n========== DATASET STATISTICS ==========")

dataset_report(games)


# =====================================
# 4. DATA VISUALIZATION
# =====================================

RUN_GRAPHS = False

if RUN_GRAPHS:

    generate_graphs(games, stats)


# =====================================
# 5. CREATE MATCH FEATURES
# =====================================

print("\n========== MATCH FEATURES ==========")

match_features = create_match_features(stats)

print("Total matches:", len(match_features))

print("Total columns:", len(match_features.columns))


# =====================================
# 6. CREATE TARGET
# =====================================

match_features = create_target(match_features)

print("\nTarget distribution:")

print(match_features["target"].value_counts())


# =====================================
# 7. CREATE TEAM HISTORY
# =====================================

team_history = create_team_history(match_features)

print("\n========== TEAM HISTORY ==========")

print("Total team history records:", len(team_history))

print("Unique teams:", team_history["teamID"].nunique())


# =====================================
# 8. ADD HISTORICAL AVERAGES
# =====================================

team_history = add_historical_averages(team_history)

print("\nHistorical averages calculated!")

print(
    team_history[
        [
            "teamID",
            "goals",
            "previous_avg_goals",
            "shots",
            "previous_avg_shots"
        ]
    ].head(10)
)


# =====================================
# 9. ADD RECENT FEATURES
# =====================================

team_history = add_recent_features(team_history)

print("\n========== RECENT FEATURES ==========")

print("Recent features calculated!")

print(
    team_history[
        [
            "teamID",
            "goals",
            "conceded_goals",
            "points",
            "previous_form_points_5",
            "previous_avg_goals_5",
            "previous_avg_conceded_5"
        ]
    ].head(10)
)


# =====================================
# 10. MERGE HISTORICAL FEATURES
# =====================================

match_features = add_match_historical_features(
    match_features,
    team_history
)

print("\nHistorical features merged!")


# =====================================
# 11. MERGE RECENT FEATURES
# =====================================

match_features = add_match_recent_features(
    match_features,
    team_history
)

print("\nRecent features merged!")


# =====================================
# 12. VERIFY FEATURE ENGINEERING
# =====================================

print("\n========== FEATURE VERIFICATION ==========")

print("Total matches:", len(match_features))

print("Total columns:", len(match_features.columns))

print("\nAll columns:")

print(match_features.columns.tolist())

print("\nNew features sample:")

print(
    match_features[
        [
            "gameID",
            "home_previous_form_points_5",
            "away_previous_form_points_5",
            "home_previous_avg_goals_5",
            "away_previous_avg_goals_5"
        ]
    ].head(10)
)


# =====================================
# 13. VERIFY REQUIRED FEATURES
# =====================================

required_features = [
    "home_previous_avg_goals",
    "away_previous_avg_goals",
    "home_previous_form_points_5",
    "away_previous_form_points_5",
    "home_previous_avg_goals_5",
    "away_previous_avg_goals_5"
]

print("\n========== REQUIRED FEATURES ==========")

for column in required_features:

    if column in match_features.columns:

        print(column + " : Available")

    else:

        print(column + " : MISSING")

        raise ValueError("Missing feature: " + column)


# =====================================
# 14. SPLIT DATASET
# =====================================

train_data, test_data = split_by_date(match_features)

model_train_data, validation_data = split_by_date(train_data)


# =====================================
# 15. PREPROCESS DATA
# =====================================

X_train, X_validation, y_train, y_validation, imputer, scaler = prepare_features(
    model_train_data,
    validation_data
)


# =====================================
# 16. VERIFY PREPROCESSING
# =====================================

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))

print("Validation records:", len(X_validation))

print("Reserved testing records:", len(test_data))


print("\n========== FEATURE COUNT ==========")

print("Training features:", X_train.shape[1])

print("Validation features:", X_validation.shape[1])


print("\n========== MISSING VALUES ==========")

print("Training missing values:")

print(X_train.isnull().sum().sum())

print("Validation missing values:")

print(X_validation.isnull().sum().sum())


print("\n========== DATE CHECK ==========")

print("Last training date:")

print(model_train_data["date"].max())

print("First validation date:")

print(validation_data["date"].min())

print("First reserved testing date:")

print(test_data["date"].min())


print("\n========== TARGET DISTRIBUTION ==========")

print("Training targets:")

print(y_train.value_counts())

print("\nValidation targets:")

print(y_validation.value_counts())


print("\n========== TRAINING FEATURES ==========")

print(X_train.columns.tolist())


print("\n========== TRAINING SAMPLE ==========")

print(X_train.head())


# =====================================
# 17. CHECK ONE TEAM'S HISTORY
# =====================================

print("\n========== TEAM HISTORY VERIFICATION ==========")

print(
    team_history[
        team_history["teamID"] == 177
    ][
        [
            "date",
            "goals",
            "conceded_goals",
            "points",
            "previous_form_points_5",
            "previous_avg_goals_5"
        ]
    ].head(10)
)


print("\n====================================")
print("PROJECT INSPECTION COMPLETED!")
print("====================================")