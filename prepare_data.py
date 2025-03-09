import pandas as pd
import sqlite3

# File paths to your CSV files
regular_season_csv_men = 'csv/MRegularSeasonDetailedResults.csv'
regular_season_csv_women = 'csv/WRegularSeasonDetailedResults.csv'
tournament_csv_men = 'csv/MNCAATourneyDetailedResults.csv'
tournament_csv_women = 'csv/WNCAATourneyDetailedResults.csv'
sqlite_db_path = 'sql/madness2025.db'

# Load CSV files into pandas DataFrames
regular_season_df_men = pd.read_csv(regular_season_csv_men)
tournament_df_men = pd.read_csv(tournament_csv_men)
regular_season_df_women = pd.read_csv(regular_season_csv_women)
tournament_df_women = pd.read_csv(tournament_csv_women)

# Add game type
regular_season_df_men['GameType'] = 'RegularSeason'
tournament_df_men['GameType'] = 'Tournament'
regular_season_df_women['GameType'] = 'RegularSeason'
tournament_df_women['GameType'] = 'Tournament'

# Combine dataframes
games_df = pd.concat([regular_season_df_men, tournament_df_men, regular_season_df_women, tournament_df_women], ignore_index=True)

# Transformation function
def transform_games(df):
    team_stats = []
    for _, row in df.iterrows():
        # Winner row
        team_stats.append({
            "Season": row["Season"],
            "DayNum": row["DayNum"],
            "TeamID": row["WTeamID"],
            "OppTeamID": row["LTeamID"],
            "GameType": row["GameType"],
            "Score": row["WScore"],
            "OppScore": row["LScore"],
            "FGM": row["WFGM"],
            "FGA": row["WFGA"],
            "FGM3": row["WFGM3"],
            "FGA3": row["WFGA3"],
            "FTM": row["WFTM"],
            "FTA": row["WFTA"],
            "OR": row["WOR"],
            "DR": row["WDR"],
            "Ast": row["WAst"],
            "TO": row["WTO"],
            "Stl": row["WStl"],
            "Blk": row["WBlk"],
            "PF": row["WPF"],
            "OppFGM": row["LFGM"],
            "OppFGA": row["LFGA"],
            "OppFGM3": row["LFGM3"],
            "OppFGA3": row["LFGA3"],
            "OppFTM": row["LFTM"],
            "OppFTA": row["LFTA"],
            "OppOR": row["LOR"],
            "OppDR": row["LDR"],
            "OppAst": row["LAst"],
            "OppTO": row["LTO"],
            "OppStl": row["LStl"],
            "OppBlk": row["LBlk"],
            "OppPF": row["LPF"],
            "NumOT": row["NumOT"],
        })

        # Loser row
        team_stats.append({
            "Season": row["Season"],
            "DayNum": row["DayNum"],
            "TeamID": row["LTeamID"],
            "OppTeamID": row["WTeamID"],
            "GameType": row["GameType"],
            "Score": row["LScore"],
            "OppScore": row["WScore"],
            "FGM": row["LFGM"],
            "FGA": row["LFGA"],
            "FGM3": row["LFGM3"],
            "FGA3": row["LFGA3"],
            "FTM": row["LFTM"],
            "FTA": row["LFTA"],
            "OR": row["LOR"],
            "DR": row["LDR"],
            "Ast": row["LAst"],
            "TO": row["LTO"],
            "Stl": row["LStl"],
            "Blk": row["LBlk"],
            "PF": row["LPF"],
            "OppFGM": row["WFGM"],
            "OppFGA": row["WFGA"],
            "OppFGM3": row["WFGM3"],
            "OppFGA3": row["WFGA3"],
            "OppFTM": row["WFTM"],
            "OppFTA": row["WFTA"],
            "OppOR": row["WOR"],
            "OppDR": row["WDR"],
            "OppAst": row["WAst"],
            "OppTO": row["WTO"],
            "OppStl": row["WStl"],
            "OppBlk": row["WBlk"],
            "OppPF": row["WPF"],
            "NumOT": row["NumOT"],
        })

    return pd.DataFrame(team_stats)

# Transform csv
transformed_df = transform_games(games_df)

# Connect to SQLite
conn = sqlite3.connect(sqlite_db_path)

# Insert csv into table
transformed_df.to_sql('TeamGameStats', conn, if_exists='replace', index=False)

# Commit and close
conn.commit()
conn.close()

print("✅ Data successfully transformed and inserted into SQLite.")