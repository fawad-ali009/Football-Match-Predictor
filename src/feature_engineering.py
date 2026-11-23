import pandas as pd


def create_match_features(stats):

    home_stats = stats[stats["location"] == "h"].copy()
    away_stats = stats[stats["location"] == "a"].copy()

    home_stats = home_stats.rename(
        columns={
            "teamID": "homeTeamID",
            "goals": "home_goals",
            "xGoals": "home_xGoals",
            "shots": "home_shots",
            "shotsOnTarget": "home_shotsOnTarget",
            "deep": "home_deep",
            "ppda": "home_ppda",
            "fouls": "home_fouls",
            "corners": "home_corners",
            "yellowCards": "home_yellowCards",
            "redCards": "home_redCards",
            "result": "home_result"
        }
    )

    away_stats = away_stats.rename(
        columns={
            "teamID": "awayTeamID",
            "goals": "away_goals",
            "xGoals": "away_xGoals",
            "shots": "away_shots",
            "shotsOnTarget": "away_shotsOnTarget",
            "deep": "away_deep",
            "ppda": "away_ppda",
            "fouls": "away_fouls",
            "corners": "away_corners",
            "yellowCards": "away_yellowCards",
            "redCards": "away_redCards",
            "result": "away_result"
        }
    )

    home_stats = home_stats.drop(columns=["location"])
    away_stats = away_stats.drop(columns=["location"])

    match_features = pd.merge(
    home_stats,
    away_stats,
    on=["gameID", "season", "date"],
    how="inner",
    validate="one_to_one"
)

    return match_features

def create_target(match_features):
    match_features["target"] = "D"

    match_features.loc[
        match_features["home_goals"] > match_features["away_goals"],
        "target"
    ] = "W"

    match_features.loc[
            match_features["home_goals"] < match_features["away_goals"],
            "target"
        ] = "L"

    return match_features

def create_team_history(match_features):

    home = match_features[
        [
            "gameID",
            "homeTeamID",
            "season",
            "date",
            "home_goals",
            "home_xGoals",
            "home_shots",
            "home_shotsOnTarget",
            "home_deep",
            "home_ppda",
            "home_fouls",
            "home_corners",
            "home_yellowCards",
            "home_redCards"
        ]
    ].copy()

    away = match_features[
        [
            "gameID",
            "awayTeamID",
            "season",
            "date",
            "away_goals",
            "away_xGoals",
            "away_shots",
            "away_shotsOnTarget",
            "away_deep",
            "away_ppda",
            "away_fouls",
            "away_corners",
            "away_yellowCards",
            "away_redCards"
        ]
    ].copy()

    home = home.rename(
        columns={
            "homeTeamID": "teamID",
            "home_goals": "goals",
            "home_xGoals": "xGoals",
            "home_shots": "shots",
            "home_shotsOnTarget": "shotsOnTarget",
            "home_deep": "deep",
            "home_ppda": "ppda",
            "home_fouls": "fouls",
            "home_corners": "corners",
            "home_yellowCards": "yellowCards",
            "home_redCards": "redCards"
        }
    )

    away = away.rename(
        columns={
            "awayTeamID": "teamID",
            "away_goals": "goals",
            "away_xGoals": "xGoals",
            "away_shots": "shots",
            "away_shotsOnTarget": "shotsOnTarget",
            "away_deep": "deep",
            "away_ppda": "ppda",
            "away_fouls": "fouls",
            "away_corners": "corners",
            "away_yellowCards": "yellowCards",
            "away_redCards": "redCards"
        }
    )

    home["location"] = "h"
    away["location"] = "a"

    team_history = pd.concat(
        [home, away],
        ignore_index=True
    )

    
    team_history["date"] = pd.to_datetime(team_history["date"])

    
    # Convert dates
    team_history["date"] = pd.to_datetime(
        team_history["date"]
    )

    # Sort matches chronologically
    team_history = team_history.sort_values(
        ["date", "gameID", "teamID"]
    ).reset_index(drop=True)

    # Check for duplicate kickoff times
    duplicate_times = team_history.duplicated(
        subset=["teamID", "date"],
        keep=False
    )

    print(
        "Conflicting kickoff records:",
        duplicate_times.sum()
    )

    # Raise an error ONLY if duplicates exist
    if duplicate_times.any():

        print(
            team_history.loc[
                duplicate_times,
                ["gameID", "teamID", "date"]
            ]
        )

        raise ValueError(
            "A team has multiple matches at the same time."
        )

    return team_history



def add_historical_averages(team_history):

    team_history = team_history.copy()

    features = [
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

    for feature in features:

        team_history["previous_avg_" + feature] = (
            team_history
            .groupby("teamID")[feature]
            .transform(
                lambda x: x.shift().expanding().mean()
            )
        )

    return team_history


def add_match_historical_features(match_features, team_history):

    historical_features = [
        "previous_avg_goals",
        "previous_avg_xGoals",
        "previous_avg_shots",
        "previous_avg_shotsOnTarget",
        "previous_avg_deep",
        "previous_avg_ppda",
        "previous_avg_fouls",
        "previous_avg_corners",
        "previous_avg_yellowCards",
        "previous_avg_redCards"
    ]

    # Get historical statistics of home teams
    home_history = team_history[
        team_history["location"] == "h"
    ][
        ["gameID", "teamID"] + historical_features
    ].copy()

    # Get historical statistics of away teams
    away_history = team_history[
        team_history["location"] == "a"
    ][
        ["gameID", "teamID"] + historical_features
    ].copy()

    # Rename home team ID and historical features
    home_history = home_history.rename(
        columns={
            "teamID": "homeTeamID",
            **{
                feature: "home_" + feature
                for feature in historical_features
            }
        }
    )

    # Rename away team ID and historical features
    away_history = away_history.rename(
        columns={
            "teamID": "awayTeamID",
            **{
                feature: "away_" + feature
                for feature in historical_features
            }
        }
    )

    # Merge home historical features
    match_features = pd.merge(
        match_features,
        home_history,
        on=["gameID", "homeTeamID"],
        how="left",
        validate="one_to_one"
    )

    # Merge away historical features
    match_features = pd.merge(
        match_features,
        away_history,
        on=["gameID", "awayTeamID"],
        how="left",
        validate="one_to_one"
    )

    return match_features


# New feature added
def previous_five_average(values):

    previous_values = values.shift(1)

    averages = previous_values.rolling(
        window=5,
        min_periods=1
    ).mean()

    return averages

def previous_five_points(values):

    previous_values = values.shift(1)

    total_points = previous_values.rolling(
        window=5,
        min_periods=1
    ).sum()

    return total_points

def add_recent_features(team_history):

    history = team_history.copy()

    history["date"] = pd.to_datetime(history["date"])

    history = history.sort_values(
        ["teamID", "date", "gameID"]
    )

    # Every match must have two team records
    match_counts = history.groupby("gameID")["teamID"].transform("size")

    if (match_counts != 2).any():
        raise ValueError("Each match must have exactly two team records.")

    # Calculate goals conceded
    history["conceded_goals"] = (
        history.groupby("gameID")["goals"].transform("sum")
        - history["goals"]
    )

    # Calculate goal difference
    history["goal_difference"] = (
        history["goals"] - history["conceded_goals"]
    )

    # Calculate match points
    history["points"] = 0

    history.loc[
        history["goals"] > history["conceded_goals"],
        "points"
    ] = 3

    history.loc[
        history["goals"] == history["conceded_goals"],
        "points"
    ] = 1

    # Calculate recent form
    history["previous_form_points_5"] = (
        history.groupby("teamID")["points"]
        .transform(previous_five_points)
    )

    # Calculate recent averages
    average_features = {
        "goals": "previous_avg_goals_5",
        "conceded_goals": "previous_avg_conceded_5",
        "goal_difference": "previous_goal_difference_5"
    }

    for column, new_column in average_features.items():

        history[new_column] = (
            history.groupby("teamID")[column]
            .transform(previous_five_average)
        )

    # Identify home and away matches
    history["home_match"] = (
        history["location"] == "h"
    ).astype(int)

    history["away_match"] = (
        history["location"] == "a"
    ).astype(int)

    # Identify home and away wins
    history["home_win"] = (
        (history["location"] == "h") &
        (history["points"] == 3)
    ).astype(int)

    history["away_win"] = (
        (history["location"] == "a") &
        (history["points"] == 3)
    ).astype(int)

    # Calculate historical home and away win rates
    for location in ["home", "away"]:

        match_column = location + "_match"
        win_column = location + "_win"

        previous_matches = (
            history.groupby("teamID")[match_column].cumsum()
            - history[match_column]
        )

        previous_wins = (
            history.groupby("teamID")[win_column].cumsum()
            - history[win_column]
        )

        history["previous_" + location + "_win_rate"] = (
            previous_wins /
            previous_matches.replace(0, float("nan"))
        )

    # Remove temporary calculation columns
    history = history.drop(
        columns=[
            "home_match",
            "away_match",
            "home_win",
            "away_win"
        ]
    )

    return history

def add_match_recent_features(match_features, team_history):

    feature_columns = [
        "previous_form_points_5",
        "previous_avg_goals_5",
        "previous_avg_conceded_5",
        "previous_goal_difference_5",
        "previous_home_win_rate",
        "previous_away_win_rate"
    ]

    columns = ["gameID", "teamID"] + feature_columns

    # Home team records
    home = team_history[
        team_history["location"] == "h"
    ][columns].copy()

    # Away team records
    away = team_history[
        team_history["location"] == "a"
    ][columns].copy()

    home = home.rename(
        columns={"teamID": "homeTeamID"}
    )

    away = away.rename(
        columns={"teamID": "awayTeamID"}
    )

    # Rename historical features
    for column in feature_columns:

        home = home.rename(
            columns={column: "home_" + column}
        )

        away = away.rename(
            columns={column: "away_" + column}
        )

    # Merge home features
    matches = match_features.merge(
        home,
        on=["gameID", "homeTeamID"],
        how="left",
        validate="one_to_one"
    )

    # Merge away features
    matches = matches.merge(
        away,
        on=["gameID", "awayTeamID"],
        how="left",
        validate="one_to_one"
    )

    return matches