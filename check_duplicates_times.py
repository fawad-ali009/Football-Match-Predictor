
import pandas as pd

# Load our cleaned team statistics
stats = pd.read_csv(
    "data/processed/teamsStats_cleaned.csv"
)

stats["date"] = pd.to_datetime(stats["date"])

# Find teams having multiple matches
# at the exact same date and time
duplicates = stats[
    stats.duplicated(
        subset=["teamID", "date"],
        keep=False
    )
].copy()

duplicates = duplicates.sort_values(
    ["teamID", "date", "gameID"]
)

print("\n========== DUPLICATE MATCH CHECK ==========")

print("Total conflicting records:", len(duplicates))

print(
    "Affected teams:",
    duplicates["teamID"].nunique()
)

print(
    "Affected games:",
    duplicates["gameID"].nunique()
)

columns = [
    "gameID",
    "teamID",
    "season",
    "date",
    "location",
    "goals",
    "result"
]

print("\nConflicting match records:\n")

print(
    duplicates[columns].head(40).to_string(index=False)
)

# Save all records for inspection
duplicates[columns].to_csv(
    "data/processed/duplicate_match_times.csv",
    index=False
)

print("\nResults saved successfully!")
