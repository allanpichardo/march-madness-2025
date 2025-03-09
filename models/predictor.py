import torch
import torch.nn as nn
from models.fin import FIN


class MatchOutcomePredictor(nn.Module):
    def __init__(self, fin_output_dim=16, hidden_dims=[128, 64]):
        super().__init__()

        # Define FINs for each aspect
        self.team_fins = nn.ModuleDict({
            'shooting': FIN(num_features=2, output_dim=fin_output_dim),
            'turnover': FIN(num_features=2, output_dim=fin_output_dim),
            'rebounding': FIN(num_features=2, output_dim=fin_output_dim),
            'defense': FIN(num_features=3, output_dim=fin_output_dim),
            'ft_foul': FIN(num_features=3, output_dim=fin_output_dim),
            'game_control': FIN(num_features=4, output_dim=fin_output_dim),
        })

        combined_input_dim = fin_output_dim * 6 * 2  # 6 FINs per team × 2 teams

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )

    def forward(self, inputs_team_a, inputs_team_b):
        # Run FINs for Team A and extract only embeddings (features at index 0)
        team_a_embeddings = [
            self.team_fins[key](inputs_team_a[key])[0] for key in self.team_fins
        ]
        team_a_combined = torch.cat(team_a_embeddings, dim=-1)

        # Run FINs for Team B and extract embeddings
        team_b_embeddings = [
            self.team_fins[key](inputs_team_b[key])[0] for key in self.team_fins
        ]
        team_b_combined = torch.cat(team_b_embeddings, dim=-1)

        # Concatenate both team embeddings
        combined_features = torch.cat([team_a_combined, team_b_combined], dim=-1)

        # Classify the outcome (probability that Team A wins)
        prob_team_a_wins = torch.sigmoid(self.classifier(combined_features))

        return prob_team_a_wins