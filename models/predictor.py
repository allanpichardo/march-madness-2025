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

import torch
import torch.nn as nn
from models.fin import FIN

def ensure_batch(input_dict):
    return {k: (v if v.dim() == 3 else v.unsqueeze(0)) for k, v in input_dict.items()}


class MatchOutcomeTransformer(nn.Module):
    def __init__(self, fin_output_dim=16, hidden_dims=[128, 64], num_heads=4, transformer_layers=2):
        super().__init__()

        # Define FINs for each aspect.
        self.team_fins = nn.ModuleDict({
            'shooting': FIN(num_features=2, output_dim=fin_output_dim),
            'turnover': FIN(num_features=2, output_dim=fin_output_dim),
            'rebounding': FIN(num_features=2, output_dim=fin_output_dim),
            'defense': FIN(num_features=3, output_dim=fin_output_dim),
            'ft_foul': FIN(num_features=3, output_dim=fin_output_dim),
            'game_control': FIN(num_features=4, output_dim=fin_output_dim),
        })

        combined_input_dim = fin_output_dim * 6  # per team FIN outputs

        # Transformer encoder to capture inter-game interactions.
        # (Note: It expects input shape (S, N, E) by default.)
        encoder_layer = nn.TransformerEncoderLayer(d_model=combined_input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Final classifier, now after a transformer layer.
        self.classifier = nn.Sequential(
            nn.Linear(combined_input_dim * 2, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, inputs_team_a, inputs_team_b):
        # Ensure inputs have a batch dimension.
        inputs_team_a = ensure_batch(inputs_team_a)
        inputs_team_b = ensure_batch(inputs_team_b)

        # Process each aspect for Team A.
        team_a_embeddings = [self.team_fins[key](inputs_team_a[key])[0] for key in self.team_fins]
        team_a_combined = torch.cat(team_a_embeddings, dim=-1)  # shape: (batch, E)

        # Process each aspect for Team B.
        team_b_embeddings = [self.team_fins[key](inputs_team_b[key])[0] for key in self.team_fins]
        team_b_combined = torch.cat(team_b_embeddings, dim=-1)  # shape: (batch, E)

        # Add a sequence dimension to each sample.
        # If we simply unsqueeze at dimension 1, we get shape (batch, 1, E)
        team_a_seq = team_a_combined.unsqueeze(1)  # (batch, 1, E)
        team_b_seq = team_b_combined.unsqueeze(1)  # (batch, 1, E)

        # Repeat each sample along the sequence dimension to simulate a sequence length of 2.
        # This repetition is done per sample so data across different samples remain separate.
        team_a_seq = team_a_seq.repeat(1, 2, 1)  # (batch, 2, E)
        team_b_seq = team_b_seq.repeat(1, 2, 1)  # (batch, 2, E)

        # Since the transformer expects shape (S, N, E) (sequence-first),
        # transpose the inputs: (batch, 2, E) -> (2, batch, E)
        team_a_seq = team_a_seq.transpose(0, 1)  # (2, batch, E)
        team_b_seq = team_b_seq.transpose(0, 1)  # (2, batch, E)

        # Run the transformer encoder.
        # Now each sample is processed with a sequence length of 2.
        team_a_encoded = self.transformer_encoder(team_a_seq)  # (2, batch, E)
        team_b_encoded = self.transformer_encoder(team_b_seq)  # (2, batch, E)

        # Take the output corresponding to the first token from each sample.
        team_a_encoded = team_a_encoded[0]  # (batch, E)
        team_b_encoded = team_b_encoded[0]  # (batch, E)

        # Concatenate both team embeddings and classify.
        combined_features = torch.cat([team_a_encoded, team_b_encoded], dim=-1)
        prob_team_a_wins = self.classifier(combined_features)
        return prob_team_a_wins