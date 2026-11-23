
import os
import pandas as pd
import unicodedata


LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1"
}

# Some clubs have different names in different datasets.
ALIASES = {
    "mancity": "manchestercity",
    "manunited": "manchesterunited",
    "manutd": "manchesterunited",
    "wolves": "wolverhamptonwanderers",
    "leicester": "leicestercity",
    "newcastle": "newcastleunited",
    "westham": "westhamunited",
    "brighton": "brightonandhovealbion",
    "tottenhamhotspur": "tottenham",
    "norwich": "norwichcity",

    "athmadrid": "atleticomadrid",
    "atletico": "atleticomadrid",
    "athbilbao": "athleticbilbao",
    "athleticclub": "athleticbilbao",
    "betis": "realbetis",
    "sociedad": "realsociedad",
    "celta": "celtavigo",
    "espanyol": "espanol",

    "inter": "intermilan",
    "internazionale": "intermilan",
    "acmilan": "milan",
    "asroma": "roma",
    "hellasverona": "verona",

    "bayernmunchen": "bayernmunich",
    "dortmund": "borussiadortmund",
    "leverkusen": "bayerleverkusen",
    "mgladbach": "borussiamgladbach",
    "bmonchengladbach": "borussiamonchengladbach",
    "frankfurt": "eintrachtfrankfurt",
    "cologne": "fcologne",
    "fckoln": "fccologne",

    "psg": "parissaintgermain",
    "parissg": "parissaintgermain",
    "lyon": "olympiquelyonnais",
    "marseille": "olympiquemarseille",
    "stetienne": "saintetienne",


    # Additional verified 2021/22 team-name mappings

    "vallecano": "rayovallecano",

    "bielefeld": "arminiabielefeld",
    "stuttgart": "vfbstuttgart",
    "mainz": "mainz05",

    "rbleipzig": "rasenballsportleipzig",
    "einfrankfurt": "eintrachtfrankfurt",
    "hertha": "herthaberlin",

}


def normalize_name(name):

    name = unicodedata.normalize("NFKD", str(name))

    name = "".join(
        character for character in name
        if not unicodedata.combining(character)
    )

    name = "".join(
        character for character in name.lower()
        if character.isalnum()
    )

    return ALIASES.get(name, name)


def prepare_2021_results(teams):

    print("\n========== 2021/22 DATA COLLECTION ==========")

    folder = "data/evaluation"

    os.makedirs(folder, exist_ok=True)

    # Map historical team names to their existing IDs.
    team_lookup = {}

    for _, row in teams.iterrows():

        key = normalize_name(row["name"])

        if key in team_lookup:
            if team_lookup[key] != row["teamID"]:
                raise ValueError(
                    "Ambiguous historical team name: " + key
                )

        team_lookup[key] = row["teamID"]

    selected_matches = []
    unmatched_names = []

    for code, league in LEAGUES.items():

        file_path = os.path.join(folder, code + ".csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                "Missing source file: " + file_path
            )

        data = pd.read_csv(
            file_path,
            encoding="latin-1"
        )

        required = [
            "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR"
        ]

        if not all(col in data.columns for col in required):
            raise ValueError(
                "Missing required columns in " + file_path
            )

        # Keep complete match-result records.
        data = data.dropna(subset=required).copy()
        data = data.reset_index(drop=True)

        # Preserve a reference to each source row.
        data["match_id"] = [
            code + "-" + str(i + 1).zfill(4)
            for i in range(len(data))
        ]

        data["date"] = pd.to_datetime(
            data["Date"],
            dayfirst=True,
            errors="coerce"
        )

        if data["date"].isna().any():
            raise ValueError("Invalid dates in " + file_path)

        # Verify this is the 2021/22 season.
        if not data["date"].between(
            "2021-07-01",
            "2022-06-30",
            inclusive="both"
        ).all():
            raise ValueError(
                "File contains dates outside 2021/22: "
                + file_path
            )

        data["home_goals"] = pd.to_numeric(
            data["FTHG"],
            errors="raise"
        ).astype(int)

        data["away_goals"] = pd.to_numeric(
            data["FTAG"],
            errors="raise"
        ).astype(int)

        if (
            (data["home_goals"] < 0) |
            (data["away_goals"] < 0)
        ).any():
            raise ValueError("Negative goal counts found.")

        # Convert source outcomes to our W/D/L labels.
        data["actual_result"] = data["FTR"].map({
            "H": "W",
            "D": "D",
            "A": "L"
        })

        # Independently derive results from final scores.
        calculated = []

        for home, away in zip(
            data["home_goals"],
            data["away_goals"]
        ):

            if home > away:
                calculated.append("W")
            elif home < away:
                calculated.append("L")
            else:
                calculated.append("D")

        # Reject inconsistent records.
        if (
            data["actual_result"].isna().any()
            or not (
                data["actual_result"]
                == calculated
            ).all()
        ):
            raise ValueError(
                "Results do not match scores in " + file_path
            )

        if data.duplicated(
            ["date", "HomeTeam", "AwayTeam"]
        ).any():
            raise ValueError(
                "Duplicate fixtures in " + file_path
            )

        data["league"] = league

        # Match club names with the historical dataset.
        data["home_key"] = data["HomeTeam"].apply(
            normalize_name
        )

        data["away_key"] = data["AwayTeam"].apply(
            normalize_name
        )

        # Record unknown names for later inspection.
        for column, key_column in [
            ("HomeTeam", "home_key"),
            ("AwayTeam", "away_key")
        ]:

            unknown = data[
                ~data[key_column].isin(team_lookup)
            ][column].unique()

            for name in unknown:
                unmatched_names.append({
                    "league": league,
                    "source_name": name,
                    "normalized_name": normalize_name(name)
                })

        # Only use fixtures involving known historical teams.
        eligible = data[
            data["home_key"].isin(team_lookup)
            &
            data["away_key"].isin(team_lookup)
        ].copy()

        print("\nLeague:", league)
        print("Source matches:", len(data))
        print("Eligible matches:", len(eligible))

        if len(eligible) < 20:
            print("WARNING: Fewer than 20 eligible matches.")
            continue

        # Reproducible random selection.
        selected = eligible.sample(
            n=20,
            random_state=42
        ).copy()

        selected["home_team_id"] = (
            selected["home_key"].map(team_lookup)
        )

        selected["away_team_id"] = (
            selected["away_key"].map(team_lookup)
        )

        selected = selected.rename(columns={
            "HomeTeam": "home_team",
            "AwayTeam": "away_team"
        })

        columns = [
            "match_id",
            "league",
            "date",
            "home_team",
            "away_team",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "actual_result"
        ]

        selected_matches.append(selected[columns])

    # Save unmatched names for verification.
    unmatched = pd.DataFrame(
        unmatched_names,
        columns=["league", "source_name", "normalized_name"]
    ).drop_duplicates()

    unmatched.to_csv(
        os.path.join(folder, "unmatched_teams_2021_22.csv"),
        index=False
    )

    if len(selected_matches) != 5:
        raise ValueError(
            "Not all five leagues have 20 eligible matches. "
            "Check unmatched_teams_2021_22.csv."
        )

    actual = pd.concat(
        selected_matches,
        ignore_index=True
    )

    actual = actual.sort_values(
        ["league", "date", "match_id"]
    )

    # Save actual results from the historical source.
    actual_path = os.path.join(
        folder,
        "actual_results_2021_22.csv"
    )

    actual.to_csv(actual_path, index=False)

    # Also save fixtures WITHOUT actual outcomes.
    fixtures = actual.drop(
        columns=["home_goals", "away_goals", "actual_result"]
    )

    fixtures_path = os.path.join(
        folder,
        "fixtures_2021_22.csv"
    )

    fixtures.to_csv(fixtures_path, index=False)

    print("\n========== COLLECTION COMPLETED ==========")
    print("Selected matches:", len(actual))
    print("Actual results:", actual_path)
    print("Fixtures without results:", fixtures_path)

    return actual
