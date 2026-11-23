
import pandas as pd


def get_team_features(team_history, team_id, prefix):

    # Get matches played by the selected team
    history = team_history[
        team_history["teamID"] == team_id
    ].copy()

    history["date"] = pd.to_datetime(history["date"])

    history = history.sort_values(
        ["date", "gameID"]
    )

    if history.empty:
        raise ValueError(
            "No historical matches for team ID: " + str(team_id)
        )

    features = {}

    # =====================================
    # 1. HISTORICAL AVERAGES (10 FEATURES)
    # =====================================

    columns = [
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

    for column in columns:

        feature_name = prefix + "_previous_avg_" + column

        features[feature_name] = history[column].mean()

    # =====================================
    # 2. RECENT FORM (4 FEATURES)
    # =====================================

    recent = history.tail(5)

    features[prefix + "_previous_form_points_5"] = (
        recent["points"].sum()
    )

    features[prefix + "_previous_avg_goals_5"] = (
        recent["goals"].mean()
    )

    features[prefix + "_previous_avg_conceded_5"] = (
        recent["conceded_goals"].mean()
    )

    features[prefix + "_previous_goal_difference_5"] = (
        recent["goal_difference"].mean()
    )

    # =====================================
    # 3. HOME AND AWAY WIN RATES
    # =====================================

    home_matches = history[
        history["location"] == "h"
    ]

    away_matches = history[
        history["location"] == "a"
    ]

    # Previous home win rate
    if len(home_matches) > 0:

        home_wins = (
            home_matches["points"] == 3
        ).sum()

        home_win_rate = home_wins / len(home_matches)

    else:
        home_win_rate = float("nan")

    # Previous away win rate
    if len(away_matches) > 0:

        away_wins = (
            away_matches["points"] == 3
        ).sum()

        away_win_rate = away_wins / len(away_matches)

    else:
        away_win_rate = float("nan")

    features[prefix + "_previous_home_win_rate"] = (
        home_win_rate
    )

    features[prefix + "_previous_away_win_rate"] = (
        away_win_rate
    )

    return features


def generate_match_features(
    team_history,
    home_team_id,
    away_team_id
):

    if home_team_id == away_team_id:
        raise ValueError(
            "Home and away teams cannot be the same."
        )

    # Generate 16 historical features for home team
    home_features = get_team_features(
        team_history,
        home_team_id,
        "home"
    )

    # Generate 16 historical features for away team
    away_features = get_team_features(
        team_history,
        away_team_id,
        "away"
    )

    # Combine all 32 features
    all_features = {
        **home_features,
        **away_features
    }

    X = pd.DataFrame([all_features])

    if len(X.columns) != 32:
        raise ValueError(
            "Expected 32 features, found "
            + str(len(X.columns))
        )

    return X
