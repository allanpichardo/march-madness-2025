import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MarchMadnessDataset(Dataset):
    def __init__(self, conn, seasons, num_games=5, matchup=False):
        self.conn = conn
        self.seasons = seasons
        self.num_games = num_games
        self.matchup = matchup

        # Preload valid games from DB
        placeholders = ','.join(['?'] * len(seasons))
        valid_games_query = f"""
            SELECT Season, DayNum, TeamID, OppTeamID, Score, OppScore
            FROM TeamGameStats
            WHERE Season IN ({placeholders})
        """
        df_valid = pd.read_sql_query(valid_games_query, conn, params=seasons)
        df_valid["game_count"] = df_valid.groupby(["Season", "TeamID"])["DayNum"].rank(method="first")
        df_valid = df_valid[df_valid["game_count"] > num_games].drop(columns=["game_count"])
        self.data = df_valid.to_records(index=False)
        self.length = len(self.data)

        # Preload past games and compute derived stats once
        past_games_query = f"""
            SELECT *
            FROM TeamGameStats
            WHERE Season IN ({placeholders})
            ORDER BY Season, TeamID, DayNum
        """
        df_past = pd.read_sql_query(past_games_query, conn, params=seasons)
        # Compute derived stats for each row
        derived_stats_df = df_past.apply(self.compute_derived_stats, axis=1, result_type='expand')
        # Keep only the necessary raw columns (for grouping/filtering) and join with derived stats
        base_cols = ['Season', 'DayNum', 'TeamID']
        self.past_games_df = df_past[base_cols].join(derived_stats_df)

        # Build a dictionary: keys are (Season, TeamID) and values are DataFrames of past games
        self.past_games_dict = {}
        for (season, team_id), group in self.past_games_df.groupby(["Season", "TeamID"]):
            group_sorted = group.sort_values("DayNum")
            self.past_games_dict[(season, team_id)] = group_sorted

        # Cache for get_inputs results to avoid recomputation
        self.input_cache = {}

    @staticmethod
    def compute_derived_stats(game):
        return {
            "FG%": game["FGM"] / game["FGA"] if game["FGA"] else 0,
            "3PT%": game["FGM3"] / game["FGA3"] if game["FGA3"] else 0,
            "TO_rate": game["TO"] / (game["FGA"] + 0.44 * game["FTA"] + game["TO"]) if game["TO"] else 0,
            "AST_TO_ratio": game["Ast"] / game["TO"] if game["TO"] else game["Ast"],
            "ORB%": game["OR"] / (game["OR"] + game["OppDR"]) if (game["OR"] + game["OppDR"]) else 0,
            "DRB%": game["DR"] / (game["DR"] + game["OppOR"]) if (game["DR"] + game["OppOR"]) else 0,
            "Stl": game["Stl"],
            "Blk": game["Blk"],
            "DefensiveRating": game["OppScore"] / (game["OppFGA"] + 0.44 * game["OppFTA"] + game["OppTO"])
            if (game["OppFGA"] + 0.44 * game["OppFTA"] + game["OppTO"]) else 0,
            "FT%": game["FTM"] / game["FTA"] if game["FTA"] else 0,
            "FTA_rate": game["FTA"] / game["FGA"] if game["FGA"] else 0,
            "OppPF": game["OppPF"],
            "OffEff": game["Score"] / (game["FGA"] + 0.44 * game["FTA"] + game["TO"])
            if (game["FGA"] + 0.44 * game["FTA"] + game["TO"]) else 0,
            "DefEff": game["OppScore"] / (game["OppFGA"] + 0.44 * game["OppFTA"] + game["OppTO"])
            if (game["OppFGA"] + 0.44 * game["OppFTA"] + game["OppTO"]) else 0,
            "NetRating": game["Score"] / (game["FGA"] + 0.44 * game["FTA"] + game["TO"]) -
                         game["OppScore"] / (game["OppFGA"] + 0.44 * game["OppFTA"] + game["OppTO"])
            if (game["FGA"] + 0.44 * game["FTA"] + game["TO"]) and (
                        game["OppFGA"] + 0.44 * game["OppFTA"] + game["OppTO"]) else 0,
            "PossessionAdv": (game["OR"] + game["OppTO"]) - (game["TO"] + game["OppOR"]),
        }

    def get_inputs(self, season, team_id, daynum):
        key = (season, team_id, daynum)
        if key in self.input_cache:
            return self.input_cache[key]

        # Retrieve preloaded past games for this team
        df_team = self.past_games_dict.get((season, team_id), pd.DataFrame())

        # Check if df_team is empty before filtering
        if df_team.empty:
            print(f"Warning: No past games found for team {team_id} in season {season}, returning zeros.")
            inputs = {
                'shooting': torch.zeros((self.num_games, 2), dtype=torch.float32),
                'turnover': torch.zeros((self.num_games, 2), dtype=torch.float32),
                'rebounding': torch.zeros((self.num_games, 2), dtype=torch.float32),
                'defense': torch.zeros((self.num_games, 3), dtype=torch.float32),
                'ft_foul': torch.zeros((self.num_games, 3), dtype=torch.float32),
                'game_control': torch.zeros((self.num_games, 4), dtype=torch.float32),
            }
            self.input_cache[key] = inputs
            return inputs

        # Filter games up to the specified daynum
        df_filtered = df_team[df_team["DayNum"] <= daynum]
        # Select the most recent num_games rows
        df_selected = df_filtered.tail(self.num_games)

        if df_selected.empty:
            print(
                f"Warning: No past games found for team {team_id} in season {season} up to day {daynum}, returning zeros.")
            inputs = {
                'shooting': torch.zeros((self.num_games, 2), dtype=torch.float32),
                'turnover': torch.zeros((self.num_games, 2), dtype=torch.float32),
                'rebounding': torch.zeros((self.num_games, 2), dtype=torch.float32),
                'defense': torch.zeros((self.num_games, 3), dtype=torch.float32),
                'ft_foul': torch.zeros((self.num_games, 3), dtype=torch.float32),
                'game_control': torch.zeros((self.num_games, 4), dtype=torch.float32),
            }
            self.input_cache[key] = inputs
            return inputs

        # Now extract the derived stats for each aspect
        shooting_stats = df_selected[['FG%', '3PT%']]
        turnover_stats = df_selected[['TO_rate', 'AST_TO_ratio']]
        rebounding_stats = df_selected[['ORB%', 'DRB%']]
        defense_stats = df_selected[['Stl', 'Blk', 'DefensiveRating']]
        ft_foul_stats = df_selected[['FT%', 'FTA_rate', 'OppPF']]
        game_control_stats = df_selected[['OffEff', 'DefEff', 'NetRating', 'PossessionAdv']]

        # If fewer than num_games rows, pad with the last row
        if len(df_selected) < self.num_games:
            last_row = df_selected.iloc[-1]
            num_padding = self.num_games - len(df_selected)
            last_row_df = pd.DataFrame([last_row] * num_padding)
            shooting_stats = pd.concat([shooting_stats, last_row_df[['FG%', '3PT%']]], ignore_index=True)
            turnover_stats = pd.concat([turnover_stats, last_row_df[['TO_rate', 'AST_TO_ratio']]], ignore_index=True)
            rebounding_stats = pd.concat([rebounding_stats, last_row_df[['ORB%', 'DRB%']]], ignore_index=True)
            defense_stats = pd.concat([defense_stats, last_row_df[['Stl', 'Blk', 'DefensiveRating']]],
                                      ignore_index=True)
            ft_foul_stats = pd.concat([ft_foul_stats, last_row_df[['FT%', 'FTA_rate', 'OppPF']]], ignore_index=True)
            game_control_stats = pd.concat(
                [game_control_stats, last_row_df[['OffEff', 'DefEff', 'NetRating', 'PossessionAdv']]],
                ignore_index=True)

        inputs = {
            'shooting': torch.tensor(shooting_stats.values, dtype=torch.float32),
            'turnover': torch.tensor(turnover_stats.values, dtype=torch.float32),
            'rebounding': torch.tensor(rebounding_stats.values, dtype=torch.float32),
            'defense': torch.tensor(defense_stats.values, dtype=torch.float32),
            'ft_foul': torch.tensor(ft_foul_stats.values, dtype=torch.float32),
            'game_control': torch.tensor(game_control_stats.values, dtype=torch.float32),
        }
        self.input_cache[key] = inputs
        return inputs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.data[idx]
        season, daynum, team_id, opp_team_id, score, opp_score = row
        inputs_team_a = self.get_inputs(season, team_id, daynum)
        if self.matchup:
            inputs_team_b = self.get_inputs(season, opp_team_id, daynum)
            label = torch.tensor(int(score > opp_score), dtype=torch.float32)
            return {"inputs_team_a": inputs_team_a, "inputs_team_b": inputs_team_b, "label": label}
        label = torch.tensor(int(score > opp_score), dtype=torch.float32)
        return {"inputs": inputs_team_a, "label": label}

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