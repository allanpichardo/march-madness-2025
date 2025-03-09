import unittest
import sqlite3
import pandas as pd
import torch

from data.dataset import MarchMadnessDataset, SyntheticMarchMadnessDataset


class TestMarchMadnessDataset(unittest.TestCase):
    def setUp(self):
        # Use the same in-memory database connection
        self.conn = sqlite3.connect(':memory:')
        self.create_test_db(self.conn)

        # Pass existing connection to the dataset
        self.dataset = MarchMadnessDataset(
            seasons=[2023],
            conn=self.conn
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
            (2023, 10, 1, 2, 'RegularSeason', 80, 70, 30, 60, 8, 20, 12, 15, 5, 20, 15, 10, 5, 2, 18, 25, 50, 7, 18, 13, 18, 4, 18, 12, 12, 6, 1, 17, 0),
            (2023, 12, 1, 3, 'RegularSeason', 75, 65, 28, 58, 9, 22, 10, 14, 7, 21, 13, 11, 6, 3, 17, 23, 54, 6, 17, 13, 19, 6, 17, 10, 11, 5, 3, 19, 0),
            (2023, 15, 1, 4, 'RegularSeason', 70, 60, 27, 55, 7, 19, 9, 13, 6, 18, 12, 12, 5, 4, 16, 22, 52, 8, 21, 8, 11, 5, 20, 9, 13, 4, 2, 15, 0),
            (2023, 20, 1, 5, 'RegularSeason', 85, 75, 32, 65, 10, 25, 11, 16, 8, 22, 18, 9, 7, 3, 15, 26, 60, 9, 24, 14, 18, 7, 21, 11, 14, 7, 4, 20, 0),
            (2023, 30, 1, 6, 'RegularSeason', 80, 70, 30, 60, 8, 20, 12, 15, 5, 20, 15, 10, 5, 2, 18, 25, 50, 7, 18, 13,18, 4, 18, 12, 12, 6, 1, 17, 0),
            (2023, 32, 1, 7, 'RegularSeason', 75, 65, 28, 58, 9, 22, 10, 14, 7, 21, 13, 11, 6, 3, 17, 23, 54, 6, 17, 13,19, 6, 17, 10, 11, 5, 3, 19, 0),
            (2023, 35, 1, 8, 'RegularSeason', 70, 60, 27, 55, 7, 19, 9, 13, 6, 18, 12, 12, 5, 4, 16, 22, 52, 8, 21, 8,11, 5, 20, 9, 13, 4, 2, 15, 0),
            (2023, 40, 1, 9, 'RegularSeason', 85, 75, 32, 65, 10, 25, 11, 16, 8, 22, 18, 9, 7, 3, 15, 26, 60, 9, 24, 14,18, 7, 21, 11, 14, 7, 4, 20, 0),
        ]
        conn.executemany("""
            INSERT INTO TeamGameStats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, mock_data)
        conn.commit()

    def test_length(self):
        self.assertEqual(len(self.dataset), 3)

    def test_get_item_no_padding_needed(self):
        # Last game (DayNum=15) should have at least 2 past games
        item = self.dataset[2]
        shooting_data = item['inputs']['shooting']

        print("Data for shooting FIN no padding:", shooting_data)

        # Validate non-zero values
        self.assertTrue(torch.any(shooting_data != 0))

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