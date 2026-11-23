
import pandas as pd
import numpy as np

from src.feature_engineering import (
    create_match_features,
    create_target,
    create_team_history,
    add_historical_averages,
    add_recent_features,
    add_match_historical_features,
    add_match_recent_features
)


# =====================================
# REBUILD ALL HISTORICAL FEATURES
# =====================================

def build_features(stats):

    matches = create_match_features(stats)
    matches = create_target(matches)

    history = create_team_history(matches)
    history = add_historical_averages(history)
    history = add_recent_features(history)

    matches = add_match_historical_features(
        matches, history
    )

    matches = add_match_recent_features(
        matches, history
    )

    return matches


# =====================================
# LOAD CLEANED DATA
# =====================================

stats = pd.read_csv(
    "data/processed/teamsStats_cleaned.csv"
)

stats["date"] = pd.to_datetime(stats["date"])


# =====================================
# DEBUG: CHECK MATCH MERGING
# =====================================

print("\n========== MERGE DIAGNOSTIC ==========")

print("Original stats rows:", len(stats))

matches_check = create_match_features(stats)

print("Merged match rows:", len(matches_check))

print(
    "Repeated game IDs:",
    matches_check.duplicated("gameID", keep=False).sum()
)

# Reconstruct team/date pairs from merged matches

home_check = matches_check[
    ["gameID", "date", "homeTeamID"]
].rename(
    columns={"homeTeamID": "teamID"}
)

away_check = matches_check[
    ["gameID", "date", "awayTeamID"]
].rename(
    columns={"awayTeamID": "teamID"}
)

history_check = pd.concat(
    [home_check, away_check],
    ignore_index=True
)

history_check["date"] = pd.to_datetime(
    history_check["date"]
)

conflicts = history_check[
    history_check.duplicated(
        ["teamID", "date"],
        keep=False
    )
]

print("Reconstructed history rows:", len(history_check))
print("Conflicting history records:", len(conflicts))

if not conflicts.empty:
    print(
        conflicts.sort_values(
            ["teamID", "date"]
        ).to_string(index=False)
    )


full_data = build_features(stats)

features = [
    column for column in full_data.columns
    if column.startswith("home_previous_")
    or column.startswith("away_previous_")
]

print("\nTotal historical features:", len(features))

if len(features) != 32:
    raise ValueError("Expected exactly 32 historical features")


# =====================================
# SELECT SAMPLE MATCHES
# =====================================

ordered = full_data.sort_values("date").reset_index(drop=True)

sample_indices = [1000, 4000, 7500]


# =====================================
# TEST HISTORICAL FEATURES
# =====================================

for index in sample_indices:

    match = ordered.iloc[index]

    game_id = match["gameID"]
    match_date = pd.to_datetime(match["date"])

    original = full_data[
        full_data["gameID"] == game_id
    ][features].iloc[0].to_numpy(dtype=float)

    # TEST 1:
    # Remove all matches after this match's date.

    old_stats = stats[
        stats["date"] <= match_date
    ].copy()

    old_data = build_features(old_stats)

    without_future = old_data[
        old_data["gameID"] == game_id
    ][features].iloc[0].to_numpy(dtype=float)

    future_test = np.allclose(
        original,
        without_future,
        equal_nan=True
    )

    # TEST 2:
    # Change the current match's statistics.
    # Its PREVIOUS features should stay unchanged.

    changed_stats = stats.copy()

    columns_to_change = [
        "goals",
        "xGoals",
        "shots",
        "shotsOnTarget",
        "corners"
    ]

    changed_stats.loc[
        changed_stats["gameID"] == game_id,
        columns_to_change
    ] += 3

    changed_data = build_features(changed_stats)

    after_change = changed_data[
        changed_data["gameID"] == game_id
    ][features].iloc[0].to_numpy(dtype=float)

    current_match_test = np.allclose(
        original,
        after_change,
        equal_nan=True
    )

    print("\n====================================")
    print("Game ID:", game_id)
    print("Date:", match_date)
    print("====================================")

    print("Future data test:", "PASS" if future_test else "FAIL")
    print("Current match test:", "PASS" if current_match_test else "FAIL")

    if not future_test or not current_match_test:
        raise AssertionError(
            "Potential data leakage detected for game "
            + str(game_id)
        )

print("\nALL SAMPLE TESTS PASSED")
