import numpy as np
import torch
from torch.utils.data import Dataset
import sqlite3
import pandas as pd

class MarchMadnessDataset(Dataset):
    def __init__(self, conn, seasons, num_games=5):
        self.conn = conn
        self.seasons = seasons
        self.num_games = num_games

        placeholders = ','.join(['?'] * len(seasons))

        # Only include games with at least num_games past games available
        self.id_query = f"""
            SELECT Season, DayNum, TeamID, Score, OppScore
            FROM TeamGameStats g
            WHERE Season IN ({placeholders})
              AND (SELECT COUNT(*) FROM TeamGameStats past
                   WHERE past.Season = g.Season 
                     AND past.TeamID = g.TeamID 
                     AND past.DayNum < g.DayNum) >= {num_games}
            ORDER BY Season, DayNum, TeamID
            LIMIT 1 OFFSET ?
        """.replace("{placeholders}", placeholders)

        count_query = f"""
            SELECT COUNT(*)
            FROM TeamGameStats g
            WHERE Season IN ({placeholders})
              AND (SELECT COUNT(*) FROM TeamGameStats past
                   WHERE past.Season = g.Season 
                     AND past.TeamID = g.TeamID 
                     AND past.DayNum < g.DayNum) >= {num_games}
        """
        self.length = conn.execute(count_query, seasons).fetchone()[0]

    def __len__(self):
        return self.length

    @staticmethod
    def compute_derived_stats(game):
        derived = {}
        # Shooting Efficiency
        derived['FG%'] = game['FGM'] / game['FGA'] if game['FGA'] else 0
        derived['3PT%'] = game['FGM3'] / game['FGA3'] if game['FGA3'] else 0

        # Turnover & Ball Control
        derived['TO_rate'] = game['TO'] / (game['FGA'] + 0.44 * game['FTA'] + game['TO'])  # approx. possessions
        derived['AST_TO_ratio'] = game['Ast'] / game['TO'] if game['TO'] else game['Ast']

        # Rebounding Percentages
        derived['ORB%'] = game['OR'] / (game['OR'] + game['OppDR']) if (game['OR'] + game['OppDR']) else 0
        derived['DRB%'] = game['DR'] / (game['DR'] + game['OppOR']) if (game['DR'] + game['OppOR']) else 0

        # Defensive metrics
        derived['Stl'] = game['Stl']
        derived['Blk'] = game['Blk']
        opponent_possessions = game['OppFGA'] + 0.44 * game['OppFTA'] + game['OppTO']
        derived['DefensiveRating'] = game['OppScore'] / opponent_possessions if opponent_possessions else 0

        # Free Throws and Fouls
        derived['FT%'] = game['FTM'] / game['FTA'] if game['FTA'] else 0
        derived['FTA_rate'] = game['FTA'] / game['FGA'] if game['FGA'] else 0
        derived['OppPF'] = game['OppPF']

        # Overall Efficiency
        team_possessions = game['FGA'] + 0.44 * game['FTA'] + game['TO']
        derived['OffEff'] = game['Score'] / team_possessions if team_possessions else 0
        derived['DefEff'] = derived['DefensiveRating']
        derived['NetRating'] = derived['OffEff'] - derived['DefEff']
        derived['PossessionAdv'] = (game['OR'] + game['OppTO']) - (game['TO'] + game['OppOR'])

        return derived

    def __getitem__(self, idx):
        row = self.conn.execute(self.id_query, (*self.seasons, idx)).fetchone()
        if row is None:
            raise IndexError(f"Index {idx} is out of bounds.")

        season, daynum, team_id, score, opp_score = row

        past_query = f"""
            SELECT * FROM TeamGameStats
            WHERE Season=? AND TeamID=? AND DayNum < ?
            ORDER BY DayNum DESC LIMIT {self.num_games}
        """
        past_games_df = pd.read_sql_query(past_query, self.conn, params=(season, team_id, daynum))
        derived_stats = past_games_df.apply(self.compute_derived_stats, axis=1, result_type='expand')

        # Padding if necessary (should rarely be needed now)
        if len(past_games_df) < self.num_games:
            padding_needed = self.num_games - len(past_games_df)
            last_row = past_games_df.iloc[-1].to_dict()
            padding_rows = pd.DataFrame([last_row]*padding_needed, columns=past_games_df.columns)
            past_games_df = pd.concat([past_games_df, padding_rows], ignore_index=True)

        inputs = {
            'shooting': torch.tensor(derived_stats[['FG%', '3PT%']].values, dtype=torch.float32),
            'turnover': torch.tensor(derived_stats[['TO_rate', 'AST_TO_ratio']].values, dtype=torch.float32),
            'rebounding': torch.tensor(derived_stats[['ORB%', 'DRB%']].values, dtype=torch.float32),
            'defense': torch.tensor(derived_stats[['Stl', 'Blk', 'DefensiveRating']].values, dtype=torch.float32),
            'ft_foul': torch.tensor(derived_stats[['FT%', 'FTA_rate', 'OppPF']].values, dtype=torch.float32),
            'game_control': torch.tensor(derived_stats[['OffEff', 'DefEff', 'NetRating', 'PossessionAdv']].values,
                                         dtype=torch.float32),
        }

        # Compute the label
        label = torch.tensor(int(score > opp_score), dtype=torch.float32)

        return {"inputs": inputs, "label": label}

class SyntheticMarchMadnessDataset(Dataset):
    def __init__(self, num_games=5, num_samples=100000, seed=42):
        self.num_games = num_games
        self.num_samples = num_samples
        np.random.seed(seed)

        self.stat_distributions = {
            "TeamID": {
                "mean": 1285.762939645765,
                "std": 105.09567208748034
            },
            "OppTeamID": {
                "mean": 1285.762939645765,
                "std": 105.09567208748034
            },
            "Score": {
                "mean": 69.88024007386888,
                "std": 12.45656837152892
            },
            "OppScore": {
                "mean": 69.88024007386888,
                "std": 12.45656837152892
            },
            "FGM": {
                "mean": 24.60363468479812,
                "std": 4.864164142798951
            },
            "FGA": {
                "mean": 56.28052547637035,
                "std": 7.5178622058838664
            },
            "FGM3": {
                "mean": 6.7776336774951735,
                "std": 3.019369273587391
            },
            "FGA3": {
                "mean": 19.743876437505246,
                "std": 6.05678832384197
            },
            "FTM": {
                "mean": 13.89533702677747,
                "std": 6.07981068579007
            },
            "FTA": {
                "mean": 19.85345001259129,
                "std": 7.8904949193969
            },
            "OR": {
                "mean": 10.41581465625787,
                "std": 4.175341348825279
            },
            "DR": {
                "mean": 23.63131872744061,
                "std": 5.111401637737804
            },
            "Ast": {
                "mean": 13.05568706455133,
                "std": 4.404983588614779
            },
            "TO": {
                "mean": 13.21708637622765,
                "std": 4.253489308591596
            },
            "Stl": {
                "mean": 6.484882061613363,
                "std": 2.964810357963124
            },
            "Blk": {
                "mean": 3.3260471753546548,
                "std": 2.2739408950051736
            },
            "PF": {
                "mean": 18.22531688071854,
                "std": 4.4753689094991635
            },
            "OppFGM": {
                "mean": 24.60363468479812,
                "std": 4.864164142798951
            },
            "OppFGA": {
                "mean": 56.28052547637035,
                "std": 7.5178622058838664
            },
            "OppFGM3": {
                "mean": 6.7776336774951735,
                "std": 3.019369273587391
            },
            "OppFGA3": {
                "mean": 19.743876437505246,
                "std": 6.05678832384197
            },
            "OppFTM": {
                "mean": 13.89533702677747,
                "std": 6.07981068579007
            },
            "OppFTA": {
                "mean": 19.85345001259129,
                "std": 7.8904949193969
            },
            "OppOR": {
                "mean": 10.41581465625787,
                "std": 4.175341348825279
            },
            "OppDR": {
                "mean": 23.63131872744061,
                "std": 5.111401637737804
            },
            "OppAst": {
                "mean": 13.05568706455133,
                "std": 4.404983588614779
            },
            "OppTO": {
                "mean": 13.21708637622765,
                "std": 4.253489308591596
            },
            "OppStl": {
                "mean": 6.484882061613363,
                "std": 2.964810357963124
            },
            "OppBlk": {
                "mean": 3.3260471753546548,
                "std": 2.2739408950051736
            },
            "OppPF": {
                "mean": 18.22531688071854,
                "std": 4.4753689094991635
            },
            "NumOT": {
                "mean": 0.06866448417694955,
                "std": 0.30486652711571444
            }
        }

    def __len__(self):
        return self.num_samples

    def generate_synthetic_stat(self, stat_name):
        params = self.stat_distributions.get(stat_name, {'mean':10, 'std':2})
        return np.random.normal(params['mean'], params['std'])

    def __getitem__(self, idx):
        num_games = self.num_games

        # Dynamically generate a DataFrame to resemble the real dataset
        past_games_data = {}
        for col in self.stat_distributions.keys():
            past_games_df_col = np.clip(
                np.random.normal(
                    self.stat_distributions[col]['mean'],
                    self.stat_distributions[col]['std'],
                    size=num_games
                ),
                a_min=0, a_max=None  # ensure no negative values
            )
            past_games_data[col] = past_games_df_col

        past_games_df = pd.DataFrame(past_games_data)
        derived_stats = past_games_df.apply(MarchMadnessDataset.compute_derived_stats, axis=1, result_type='expand')

        # Assemble input tensors per FIN dynamically
        inputs = {
            'shooting': torch.tensor(derived_stats[['FG%', '3PT%']].values, dtype=torch.float32),
            'turnover': torch.tensor(derived_stats[['TO_rate', 'AST_TO_ratio']].values, dtype=torch.float32),
            'rebounding': torch.tensor(derived_stats[['ORB%', 'DRB%']].values, dtype=torch.float32),
            'defense': torch.tensor(derived_stats[['Stl', 'Blk', 'DefensiveRating']].values, dtype=torch.float32),
            'ft_foul': torch.tensor(derived_stats[['FT%', 'FTA_rate', 'OppPF']].values, dtype=torch.float32),
            'game_control': torch.tensor(derived_stats[['OffEff', 'DefEff', 'NetRating', 'PossessionAdv']].values,
                                         dtype=torch.float32),
        }

        # Simple synthetic rule for outcome: (FG% > threshold)
        fg_percentage = past_games_df['FGM'].sum() / past_games_df['FGA'].sum()
        label = torch.tensor(int(fg_percentage > 0.45), dtype=torch.float32)  # synthetic rule

        return {"inputs": inputs, "label": label}