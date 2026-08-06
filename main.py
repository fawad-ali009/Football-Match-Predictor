from src.data_loader import load_games, load_teams, load_teamStats

games = load_games()
print(games.shape)

teams = load_teams()
print(teams.shape)

teams_stats = load_teamStats()
print(teams_stats.shape)

print(teams_stats.columns)