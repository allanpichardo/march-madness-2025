import torch
import torch.nn as nn

class FIN(nn.Module):
    def __init__(self, num_features, hidden_dim=32, output_dim=16):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # pool along the sequence length dimension
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.outcome_head = nn.Linear(output_dim, 1)

    def forward(self, x):
        # Input shape: (batch_size, num_games, num_features)
        # Rearrange to (batch_size, num_features, num_games)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)  # shape: (batch_size, hidden_dim, L_out)
        features = self.fc(x)

        outcome = self.outcome_head(features)
        outcome = torch.sigmoid(outcome)
        return features, outcome