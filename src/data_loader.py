import pandas as pd    

def load_games():
    print("Loading games dataset...")
    games_df = pd.read_csv("data/raw/games.csv")
    return games_df

def load_teams():
    print("Loading teams dataset...")
    teams_df = pd.read_csv("data/raw/teams.csv")
    return teams_df

def load_teamStats():
    print("Loading teams STATS dataset...")
    teamsStat_df = pd.read_csv("data/raw/teamstats.csv")
    return teamsStat_df