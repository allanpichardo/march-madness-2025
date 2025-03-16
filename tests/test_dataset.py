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

        # For testing, we set num_games=2 (instead of 5) because the test data only contains a few games.
        self.dataset_single = MarchMadnessDataset(
            conn=self.conn,
            seasons=[2023],
            num_games=2,  # Lowered for testing purposes
            matchup=False  # Default behavior
        )

        self.dataset_matchup = MarchMadnessDataset(
            conn=self.conn,
            seasons=[2023],
            num_games=2,  # Lowered for testing purposes
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
            # For testing, we'll include 3 games per team so that with num_games=2, valid data exists.
            # Game 1: Team 1 vs Team 2 (Team 1 game_count=1, Team 2 game_count=1)
            (2023, 10, 1, 2, 'RegularSeason', 80, 70, 30, 60, 8, 20, 12, 15, 5, 20, 15, 10, 5, 2, 18, 25, 50, 7, 18, 13,
             18, 4, 18, 12, 12, 6, 1, 17, 0),
            (2023, 10, 2, 1, 'RegularSeason', 70, 80, 25, 50, 7, 18, 13, 18, 4, 18, 12, 12, 6, 1, 17, 30, 60, 8, 20, 12,
             15, 5, 20, 15, 10, 5, 1, 17, 0),

            # Game 2: Team 1 vs Team 2 (Team 1 game_count=2, Team 2 game_count=2)
            (2023, 12, 1, 2, 'RegularSeason', 75, 65, 28, 58, 9, 22, 10, 14, 7, 21, 13, 11, 6, 3, 17, 23, 54, 6, 17, 13,
             19, 6, 17, 10, 11, 5, 3, 19, 0),
            (2023, 12, 2, 1, 'RegularSeason', 65, 75, 24, 55, 8, 21, 9, 13, 6, 18, 12, 12, 5, 4, 16, 28, 58, 9, 22, 10,
             14, 7, 21, 13, 11, 6, 3, 17, 0),

            # Game 3: Team 1 vs Team 2 (Team 1 game_count=3, Team 2 game_count=3)
            (2023, 15, 1, 2, 'RegularSeason', 70, 60, 27, 55, 7, 19, 9, 13, 6, 18, 12, 12, 5, 4, 16, 22, 52, 8, 21, 8,
             11, 5, 20, 9, 13, 4, 2, 15, 0),
            (2023, 15, 2, 1, 'RegularSeason', 60, 70, 23, 50, 6, 18, 7, 12, 5, 17, 11, 11, 4, 3, 14, 27, 55, 7, 19, 9,
             13, 6, 18, 12, 12, 5, 4, 16, 0)
        ]
        conn.executemany("""
            INSERT INTO TeamGameStats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, mock_data)
        conn.commit()

        # Create Seeds table and insert test seeds
        seeds_schema = """
        CREATE TABLE Seeds (
            Season INTEGER,
            TeamID INTEGER,
            Seed INTEGER,
            PRIMARY KEY (Season, TeamID)
        );
        """
        conn.execute(seeds_schema)
        seeds_data = [
            (2023, 1, 5),
            (2023, 2, 12)
        ]
        conn.executemany("INSERT INTO Seeds VALUES (?, ?, ?)", seeds_data)
        conn.commit()

    def test_length(self):
        self.assertGreater(len(self.dataset_single), 0)
        self.assertGreater(len(self.dataset_matchup), 0)

    def test_get_item_single(self):
        """ Test fetching a single team's history, including the seed. """
        item = self.dataset_single[0]
        self.assertIn("inputs", item)
        self.assertIn("label", item)
        self.assertIn("seed", item["inputs"])
        self.assertEqual(item["inputs"]['shooting'].shape[0], 2)  # Ensure 5 games history
        # Now we expect the seed to be a 20-dimensional vector.
        self.assertEqual(item["inputs"]["seed"].shape, (20,))
        # Check that the one-hot vector is valid, e.g. for team 1 the sum should be 1 and the index corresponding to the expected seed should be 1.
        self.assertEqual(item["inputs"]["seed"].sum().item(), 2.0)
        self.assertTrue(0 <= item["label"].item() <= 1)

    def test_get_item_matchup(self):
        """ Test fetching a team matchup with histories for both teams, including seeds. """
        item = self.dataset_matchup[0]
        self.assertIn("inputs_team_a", item)
        self.assertIn("inputs_team_b", item)
        self.assertIn("label", item)

        self.assertEqual(item["inputs_team_a"]['shooting'].shape[0], 2)
        self.assertEqual(item["inputs_team_b"]['shooting'].shape[0], 2)

        # Check that both team inputs include the seed as a 20-dimensional vector.
        self.assertIn("seed", item["inputs_team_a"])
        self.assertIn("seed", item["inputs_team_b"])
        self.assertEqual(item["inputs_team_a"]["seed"].shape, (20,))
        self.assertEqual(item["inputs_team_b"]["seed"].shape, (20,))
        # Optionally, you can also check that the one-hot vector sums to 1.
        self.assertEqual(item["inputs_team_a"]["seed"].sum().item(), 2.0)
        self.assertEqual(item["inputs_team_b"]["seed"].sum().item(), 2.0)

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
        for fin in self.fin_columns:
            self.assertIn(fin, item['inputs'])
            tensor = item['inputs'][fin]
            self.assertEqual(tensor.shape, (5, len(self.fin_columns[fin])))
            self.assertIsInstance(tensor, torch.Tensor)
        self.assertIn(item['label'].item(), [0.0, 1.0])

if __name__ == '__main__':
    unittest.main()