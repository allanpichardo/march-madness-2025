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
    def __init__(self, num_games=5, num_samples=100000, seed=1984):
        self.num_games = num_games
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)

        # Define the synthetic stat distributions (same as before)
        self.stat_distributions = {
            "Season": {"mean": 2015.5272010018166, "std": 6.054642012405052},
            "DayNum": {"mean": 70.92618037647316, "std": 36.53394680394814},
            "TeamID": {"mean": 2099.59360599506, "std": 988.1592657466399},
            "OppTeamID": {"mean": 2099.59360599506, "std": 988.1592657466399},
            "Score": {"mean": 67.7105101641811, "std": 13.110915758448435},
            "OppScore": {"mean": 67.7105101641811, "std": 13.110915758448435},
            "FGM": {"mean": 24.113120133839523, "std": 5.09842382452348},
            "FGA": {"mean": 57.197207881880686, "std": 7.789520204931382},
            "FGM3": {"mean": 6.3125528997738, "std": 3.056215036006145},
            "FGA3": {"mean": 19.022919523048216, "std": 6.327135615735982},
            "FTM": {"mean": 13.171716996728257, "std": 5.950441973725968},
            "FTA": {"mean": 18.845403968658587, "std": 7.766409525871011},
            "OR": {"mean": 10.935431340424584, "std": 4.448085957317066},
            "DR": {"mean": 23.99391683536848, "std": 5.319523102970879},
            "Ast": {"mean": 13.017358550335837, "std": 4.5329674237415745},
            "TO": {"mean": 14.365608588695906, "std": 4.818496098572084},
            "Stl": {"mean": 7.046814134324591, "std": 3.2990604036141113},
            "Blk": {"mean": 3.29672132770389, "std": 2.2845533553189243},
            "PF": {"mean": 17.7647042809838, "std": 4.532525172508227},
            "OppFGM": {"mean": 24.113120133839523, "std": 5.09842382452348},
            "OppFGA": {"mean": 57.197207881880686, "std": 7.789520204931382},
            "OppFGM3": {"mean": 6.3125528997738, "std": 3.056215036006145},
            "OppFGA3": {"mean": 19.022919523048216, "std": 6.327135615735982},
            "OppFTM": {"mean": 13.171716996728257, "std": 5.950441973725968},
            "OppFTA": {"mean": 18.845403968658587, "std": 7.766409525871011},
            "OppOR": {"mean": 10.935431340424584, "std": 4.448085957317066},
            "OppDR": {"mean": 23.99391683536848, "std": 5.319523102970879},
            "OppAst": {"mean": 13.017358550335837, "std": 4.5329674237415745},
            "OppTO": {"mean": 14.365608588695906, "std": 4.818496098572084},
            "OppStl": {"mean": 7.046814134324591, "std": 3.2990604036141113},
            "OppBlk": {"mean": 3.29672132770389, "std": 2.2845533553189243},
            "OppPF": {"mean": 17.7647042809838, "std": 4.532525172508227},
            "NumOT": {"mean": 0.061673093009557846, "std": 0.2868821684002519}
        }
        # Precompute the list of columns and corresponding means and stds
        self.columns = list(self.stat_distributions.keys())
        self.means = np.array([self.stat_distributions[k]['mean'] for k in self.columns])
        self.stds = np.array([self.stat_distributions[k]['std'] for k in self.columns])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        num_games = self.num_games
        # Generate synthetic data for all stats at once.
        # Shape: (num_games, num_columns)
        data = np.random.normal(loc=self.means, scale=self.stds, size=(num_games, len(self.columns)))
        data = np.clip(data, a_min=0, a_max=None)

        # Create a dictionary mapping each column to its data array
        data_dict = {col: data[:, i] for i, col in enumerate(self.columns)}

        # Vectorized computation of derived statistics
        FGM      = data_dict["FGM"]
        FGA      = data_dict["FGA"]
        FGM3     = data_dict["FGM3"]
        FGA3     = data_dict["FGA3"]
        FTM      = data_dict["FTM"]
        FTA      = data_dict["FTA"]
        OR_      = data_dict["OR"]
        DR       = data_dict["DR"]
        Ast      = data_dict["Ast"]
        TO       = data_dict["TO"]
        Stl      = data_dict["Stl"]
        Blk      = data_dict["Blk"]
        OppFGM   = data_dict["OppFGM"]
        OppFGA   = data_dict["OppFGA"]
        OppFGM3  = data_dict["OppFGM3"]
        OppFGA3  = data_dict["OppFGA3"]
        OppFTM   = data_dict["OppFTM"]
        OppFTA   = data_dict["OppFTA"]
        OppOR    = data_dict["OppOR"]
        OppDR    = data_dict["OppDR"]
        OppAst   = data_dict["OppAst"]
        OppTO    = data_dict["OppTO"]
        OppStl   = data_dict["OppStl"]
        OppBlk   = data_dict["OppBlk"]
        OppPF    = data_dict["OppPF"]
        Score    = data_dict["Score"]
        OppScore = data_dict["OppScore"]

        FG_pct       = np.where(FGA != 0, FGM / FGA, 0)
        ThreePT_pct  = np.where(FGA3 != 0, FGM3 / FGA3, 0)
        TO_rate      = np.where((FGA + 0.44 * FTA + TO) != 0, TO / (FGA + 0.44 * FTA + TO), 0)
        AST_TO_ratio = np.where(TO != 0, Ast / TO, Ast)
        ORB_pct      = np.where((OR_ + OppDR) != 0, OR_ / (OR_ + OppDR), 0)
        DRB_pct      = np.where((DR + OppOR) != 0, DR / (DR + OppOR), 0)
        DefensiveRating = np.where((OppFGA + 0.44 * OppFTA + OppTO) != 0,
                                   OppScore / (OppFGA + 0.44 * OppFTA + OppTO), 0)
        FT_pct       = np.where(FTA != 0, FTM / FTA, 0)
        FTA_rate     = np.where(FGA != 0, FTA / FGA, 0)
        OffEff       = np.where((FGA + 0.44 * FTA + TO) != 0, Score / (FGA + 0.44 * FTA + TO), 0)
        DefEff       = np.where((OppFGA + 0.44 * OppFTA + OppTO) != 0, OppScore / (OppFGA + 0.44 * OppFTA + OppTO), 0)
        NetRating    = OffEff - DefEff
        PossessionAdv= (OR_ + OppTO) - (TO + data_dict["OppOR"])

        # Assemble input tensors for each FIN
        shooting    = torch.tensor(np.stack([FG_pct, ThreePT_pct], axis=1), dtype=torch.float32)
        turnover    = torch.tensor(np.stack([TO_rate, AST_TO_ratio], axis=1), dtype=torch.float32)
        rebounding  = torch.tensor(np.stack([ORB_pct, DRB_pct], axis=1), dtype=torch.float32)
        defense     = torch.tensor(np.stack([Stl, Blk, DefensiveRating], axis=1), dtype=torch.float32)
        ft_foul     = torch.tensor(np.stack([FT_pct, FTA_rate, OppPF], axis=1), dtype=torch.float32)
        game_control= torch.tensor(np.stack([OffEff, DefEff, NetRating, PossessionAdv], axis=1), dtype=torch.float32)

        inputs = {
            'shooting': shooting,
            'turnover': turnover,
            'rebounding': rebounding,
            'defense': defense,
            'ft_foul': ft_foul,
            'game_control': game_control
        }

        # Use a simple synthetic rule for the outcome
        overall_fg_pct = np.sum(FGM) / np.sum(FGA) if np.sum(FGA) != 0 else 0
        label = torch.tensor(int(overall_fg_pct > 0.45), dtype=torch.float32)

        return {"inputs": inputs, "label": label}