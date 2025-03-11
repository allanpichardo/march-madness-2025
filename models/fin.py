import torch
import torch.nn as nn

class FIN(nn.Module):
    def __init__(self, num_features, hidden_dim=64, output_dim=16, dropout_rate=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1)  # Reduce the sequence length dimension
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.outcome_head = nn.Linear(output_dim, 1)

    def forward(self, x):
        # Expected input shape: (batch_size, num_games, num_features)
        # Rearrange to (batch_size, num_features, num_games) for conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)  # Output shape: (batch_size, hidden_dim, 1)
        features = self.fc(x)      # Shape: (batch_size, output_dim)
        outcome = self.outcome_head(features)  # Shape: (batch_size, 1)
        outcome = torch.sigmoid(outcome)
        return features, outcome