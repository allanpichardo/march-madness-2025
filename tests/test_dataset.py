import unittest
import sqlite3
import pandas as pd
import torch

from data.dataset import MarchMadnessDataset, SyntheticMarchMadnessDataset


class TestMarchMadnessDataset(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database
        self.conn = sqlite3.connect(':memory:')
        self.create_test_db(self.conn)

        # Dataset instances
        self.dataset_single = MarchMadnessDataset(
            conn=self.conn,
            seasons=[2023],
            num_games=5,
            matchup=False  # Default behavior
        )

        self.dataset_matchup = MarchMadnessDataset(
            conn=self.conn,
            seasons=[2023],
            num_games=5,
            matchup=True  # Pair team matchups
        )

    def create_test_db(self, conn):
        schema = """
        CREATE TABLE TeamGameStats (
            Season INTEGER, DayNum INTEGER, TeamID INTEGER, OppTeamID INTEGER,
            GameType TEXT, Score INTEGER, OppScore INTEGER, FGM INTEGER, FGA INTEGER,
            FGM3 INTEGER, FGA3 INTEGER, FTM INTEGER, FTA INTEGER, "OR" INTEGER,
            DR INTEGER, Ast INTEGER, "TO" INTEGER, Stl INTEGER, Blk INTEGER, PF INTEGER,
            OppFGM INTEGER, OppFGA INTEGER, OppFGM3 INTEGER, OppFGA3 INTEGER, OppFTM INTEGER,
            OppFTA INTEGER, OppOR INTEGER, OppDR INTEGER, OppAst INTEGER, OppTO INTEGER,
            OppStl INTEGER, OppBlk INTEGER, OppPF INTEGER, NumOT INTEGER,
            PRIMARY KEY (Season, DayNum, TeamID)
        );"""
        conn.execute(schema)

        mock_data = [
            # Game 1: Team 1 vs Team 2
            (2023, 10, 1, 2, 'RegularSeason', 80, 70, 30, 60, 8, 20, 12, 15, 5, 20, 15, 10, 5, 2, 18, 25, 50, 7, 18, 13,
             18, 4, 18, 12, 12, 6, 1, 17, 0),
            (2023, 10, 2, 1, 'RegularSeason', 70, 80, 25, 50, 7, 18, 13, 18, 4, 18, 12, 12, 6, 1, 17, 30, 60, 8, 20, 12,
             15, 5, 20, 15, 10, 5, 2, 18, 0),

            # Game 2: Team 1 vs Team 2
            (2023, 12, 1, 2, 'RegularSeason', 75, 65, 28, 58, 9, 22, 10, 14, 7, 21, 13, 11, 6, 3, 17, 23, 54, 6, 17, 13,
             19, 6, 17, 10, 11, 5, 3, 19, 0),
            (2023, 12, 2, 1, 'RegularSeason', 65, 75, 24, 55, 8, 21, 9, 13, 6, 18, 12, 12, 5, 4, 16, 28, 58, 9, 22, 10,
             14, 7, 21, 13, 11, 6, 3, 17, 0),

            # Game 3: Team 1 vs Team 2
            (2023, 15, 1, 2, 'RegularSeason', 70, 60, 27, 55, 7, 19, 9, 13, 6, 18, 12, 12, 5, 4, 16, 22, 52, 8, 21, 8,
             11, 5, 20, 9, 13, 4, 2, 15, 0),
            (2023, 15, 2, 1, 'RegularSeason', 60, 70, 23, 50, 6, 18, 7, 12, 5, 17, 11, 11, 4, 3, 14, 27, 55, 7, 19, 9,
             13, 6, 18, 12, 12, 5, 4, 16, 0),

            # Game 4: Team 1 vs Team 2
            (2023, 20, 1, 2, 'RegularSeason', 85, 75, 32, 65, 10, 25, 11, 16, 8, 22, 18, 9, 7, 3, 15, 26, 60, 9, 24, 14,
             18, 7, 21, 11, 14, 7, 4, 20, 0),
            (
            2023, 20, 2, 1, 'RegularSeason', 75, 85, 28, 60, 8, 22, 10, 14, 6, 20, 14, 11, 5, 2, 13, 32, 65, 10, 25, 11,
            16, 8, 22, 18, 9, 7, 3, 15, 0),

            # Game 5: Team 1 vs Team 2
            (2023, 30, 1, 2, 'RegularSeason', 80, 70, 30, 60, 8, 20, 12, 15, 5, 20, 15, 10, 5, 2, 18, 25, 50, 7, 18, 13,
             18, 4, 18, 12, 12, 6, 1, 17, 0),
            (2023, 30, 2, 1, 'RegularSeason', 70, 80, 26, 55, 7, 18, 11, 13, 4, 19, 13, 9, 5, 3, 14, 30, 60, 8, 20, 12,
             15, 5, 20, 15, 10, 5, 2, 18, 0),

            # Game 6: Team 1 vs Team 2
            (2023, 32, 1, 2, 'RegularSeason', 75, 65, 28, 58, 9, 22, 10, 14, 7, 21, 13, 11, 6, 3, 17, 23, 54, 6, 17, 13,
             19, 6, 17, 10, 11, 5, 3, 19, 0),
            (2023, 32, 2, 1, 'RegularSeason', 65, 75, 24, 55, 8, 21, 9, 13, 6, 18, 12, 12, 5, 4, 16, 28, 58, 9, 22, 10,
             14, 7, 21, 13, 11, 6, 3, 17, 0),

            # Game 7: Team 1 vs Team 2
            (2023, 35, 1, 2, 'RegularSeason', 70, 60, 27, 55, 7, 19, 9, 13, 6, 18, 12, 12, 5, 4, 16, 22, 52, 8, 21, 8,
             11, 5, 20, 9, 13, 4, 2, 15, 0),
            (2023, 35, 2, 1, 'RegularSeason', 60, 70, 23, 50, 6, 18, 7, 12, 5, 17, 11, 11, 4, 3, 14, 27, 55, 7, 19, 9,
             13, 6, 18, 12, 12, 5, 4, 16, 0),

            # Game 8: Team 1 vs Team 2
            (2023, 40, 1, 2, 'RegularSeason', 85, 75, 32, 65, 10, 25, 11, 16, 8, 22, 18, 9, 7, 3, 15, 26, 60, 9, 24, 14,
             18, 7, 21, 11, 14, 7, 4, 20, 0),
            (
            2023, 40, 2, 1, 'RegularSeason', 75, 85, 28, 60, 8, 22, 10, 14, 6, 20, 14, 11, 5, 2, 13, 32, 65, 10, 25, 11,
            16, 8, 22, 18, 9, 7, 3, 15, 0),
        ]

        conn.executemany("""
            INSERT INTO TeamGameStats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, mock_data)
        conn.commit()

    def test_length(self):
        self.assertGreater(len(self.dataset_single), 0)
        self.assertGreater(len(self.dataset_matchup), 0)

    def test_get_item_single(self):
        """ Test fetching a single team's history """
        item = self.dataset_single[0]
        self.assertIn("inputs", item)
        self.assertIn("label", item)
        self.assertEqual(item["inputs"]['shooting'].shape[0], 5)  # Ensure 5 games history
        self.assertTrue(0 <= item["label"].item() <= 1)  # Label should be binary

    def test_get_item_matchup(self):
        """ Test fetching a team matchup with histories for both teams """
        item = self.dataset_matchup[0]
        self.assertIn("inputs_team_a", item)
        self.assertIn("inputs_team_b", item)
        self.assertIn("label", item)

        self.assertEqual(item["inputs_team_a"]['shooting'].shape[0], 5)
        self.assertEqual(item["inputs_team_b"]['shooting'].shape[0], 5)

        # Check label is valid binary output
        self.assertTrue(0 <= item["label"].item() <= 1)

    def tearDown(self):
        self.conn.close()

class TestSyntheticMarchMadnessDataset(unittest.TestCase):
    def setUp(self):
        self.fin_columns = {
            'shooting': ['FG%', '3PT%'],
        }
        self.dataset = SyntheticMarchMadnessDataset(num_games=5)

    def test_len(self):
        self.assertEqual(len(self.dataset), 100000)

    def test_item_structure(self):
        item = self.dataset[0]
        self.assertIn('inputs', item)
        self.assertIn('label', item)

        print("Synthetic data for shooting FIN no padding:", item['inputs']['shooting'])

        for fin in self.fin_columns:
            self.assertIn(fin, item['inputs'])
            tensor = item['inputs'][fin]
            self.assertEqual(tensor.shape, (5, len(self.fin_columns[fin])))
            self.assertIsInstance(tensor, torch.Tensor)

        self.assertIn(item['label'].item(), [0.0, 1.0])

if __name__ == '__main__':
    unittest.main()