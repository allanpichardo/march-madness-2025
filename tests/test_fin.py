import unittest
import torch
from torch import nn

from models.fin import FIN


class TestFINCNN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.num_games = 5
        self.num_features = 3  # Example: Defense FIN has 3 features
        self.hidden_dim = 32
        self.output_dim = 16

        self.fin_cnn = FIN(
            num_features=self.num_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )

    def test_output_shape(self):
        # Random synthetic input tensor
        x = torch.randn(self.batch_size, self.num_games, self.num_features)

        # Forward pass
        output, _ = self.fin_cnn(x=x)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim),
                         msg="Output shape mismatch.")

    def test_invalid_input_shapes(self):
        invalid_input = torch.randn(self.batch_size, self.num_games)  # Missing feature dimension
        with self.assertRaises(RuntimeError):
            self.fin_cnn(invalid_input)

if __name__ == '__main__':
    unittest.main()