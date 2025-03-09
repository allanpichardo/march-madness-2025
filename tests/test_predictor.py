import unittest
import torch

from models.predictor import MatchOutcomePredictor


class TestMatchOutcomePredictor(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_games = 5
        self.fin_output_dim = 16

        self.inputs_team_a = {
            'shooting': torch.rand(self.batch_size, 5, 2),
            'turnover': torch.rand(self.batch_size, 5, 2),
            'rebounding': torch.rand(self.batch_size, 5, 2),
            'defense': torch.rand(self.batch_size, 5, 3),
            'ft_foul': torch.rand(self.batch_size, 5, 3),
            'game_control': torch.rand(self.batch_size, 5, 4),
        }

        self.inputs_team_b = {
            key: torch.rand_like(value) for key, value in self.inputs_team_a.items()
        }

        self.model = MatchOutcomePredictor()

    def test_forward_pass_shape(self):
        output = self.model(self.inputs_team_a, self.inputs_team_b)
        self.assertEqual(output.shape, (self.batch_size, 1), "Output shape mismatch.")

    def test_forward_pass_values(self):
        output = self.model(self.inputs_team_a, self.inputs_team_b)
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1),
                        "Output probabilities not in [0, 1] range.")

    def test_invalid_input_shapes(self):
        invalid_inputs_team_a = {key: torch.rand(3, 4) for key in self.inputs_team_a}
        with self.assertRaises(RuntimeError):
            self.model(invalid_inputs_team_a, self.inputs_team_b)


if __name__ == '__main__':
    unittest.main()
