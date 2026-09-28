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
        how="inner"
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

    team_history = team_history.sort_values("date")

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
