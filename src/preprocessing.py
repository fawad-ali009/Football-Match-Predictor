import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def split_by_date(match_features):

    df = match_features.copy()

    # Convert date column to real dates
    df["date"] = pd.to_datetime(df["date"])

    # Arrange matches from oldest to newest
    df = df.sort_values(["date", "gameID"])

    # Get unique match dates
    dates = df["date"].drop_duplicates()
    dates = dates.sort_values()
    dates = dates.reset_index(drop=True)

    # Find the 80% cutoff
    split_index = int(len(dates) * 0.80)
    cutoff_date = dates.iloc[split_index]

    # Older matches for training
    train_data = df[df["date"] < cutoff_date].copy()

    # Newer matches for testing
    test_data = df[df["date"] >= cutoff_date].copy()

    return train_data, test_data



def prepare_features(train_data, test_data):

    # Select our 32 historical features
    features = [
        "home_previous_avg_goals",
        "away_previous_avg_goals",

        "home_previous_avg_xGoals",
        "away_previous_avg_xGoals",

        "home_previous_avg_shots",
        "away_previous_avg_shots",

        "home_previous_avg_shotsOnTarget",
        "away_previous_avg_shotsOnTarget",

        "home_previous_avg_deep",
        "away_previous_avg_deep",

        "home_previous_avg_ppda",
        "away_previous_avg_ppda",

        "home_previous_avg_fouls",
        "away_previous_avg_fouls",

        "home_previous_avg_corners",
        "away_previous_avg_corners",

        "home_previous_avg_yellowCards",
        "away_previous_avg_yellowCards",

        "home_previous_avg_redCards",
        "away_previous_avg_redCards"
    ]

    features.extend([
    "home_previous_form_points_5",
    "away_previous_form_points_5",

    "home_previous_avg_goals_5",
    "away_previous_avg_goals_5",

    "home_previous_avg_conceded_5",
    "away_previous_avg_conceded_5",

    "home_previous_goal_difference_5",
    "away_previous_goal_difference_5",

    "home_previous_home_win_rate",
    "away_previous_home_win_rate",

    "home_previous_away_win_rate",
    "away_previous_away_win_rate"
])

    # X = inputs
    X_train = train_data[features].copy()
    X_test = test_data[features].copy()

    # y = correct results
    y_train = train_data["target"].copy()
    y_test = test_data["target"].copy()

    # Fill missing values using training medians
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=features,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=features,
        index=X_test.index
    )

    # Scale the numerical features
    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=features,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=features,
        index=X_test.index
    )

    return X_train, X_test, y_train, y_test, imputer, scaler
