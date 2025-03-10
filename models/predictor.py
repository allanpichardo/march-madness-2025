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
        prob_team_a_wins = self.classifier(combined_features)

        return prob_team_a_wins

import torch.nn as nn

class MatchOutcomeTransformer(nn.Module):
    def __init__(self, fin_output_dim=16, hidden_dims=[128, 64], num_heads=4, transformer_layers=2):
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

        combined_input_dim = fin_output_dim * 6  # per team FIN outputs

        # Transformer encoder to capture inter-game interactions
        encoder_layer = nn.TransformerEncoderLayer(d_model=combined_input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Final classifier, now after a transformer layer
        self.classifier = nn.Sequential(
            nn.Linear(combined_input_dim * 2, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, inputs_team_a, inputs_team_b):
        # Process each aspect for Team A and Team B
        team_a_embeddings = [self.team_fins[key](inputs_team_a[key])[0] for key in self.team_fins]
        team_a_combined = torch.cat(team_a_embeddings, dim=-1)  # shape: (batch, combined_input_dim)

        team_b_embeddings = [self.team_fins[key](inputs_team_b[key])[0] for key in self.team_fins]
        team_b_combined = torch.cat(team_b_embeddings, dim=-1)

        # Suppose you want to capture interactions across aspects via transformer:
        # Reshape to (sequence_length, batch, feature_dim); here sequence_length=1 is trivial,
        # so consider stacking if you have multiple time steps
        team_a_encoded = self.transformer_encoder(team_a_combined.unsqueeze(0)).squeeze(0)
        team_b_encoded = self.transformer_encoder(team_b_combined.unsqueeze(0)).squeeze(0)

        # Concatenate both team embeddings and classify
        combined_features = torch.cat([team_a_encoded, team_b_encoded], dim=-1)
        prob_team_a_wins = self.classifier(combined_features)
        return prob_team_a_wins