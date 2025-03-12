import torch
import torch.nn as nn
from models.fin import FIN

def ensure_batch(input_dict):
    # Ensure each tensor in the input dictionary is 3D: (batch, num_games, num_features)
    return {k: (v if v.dim() == 3 else v.unsqueeze(0)) for k, v in input_dict.items()}

class MatchOutcomePredictor(nn.Module):
    def __init__(self, fin_output_dim=16, conv1_out_channels=16, conv2_out_channels=16, hidden_dims=[128, 64]):
        super().__init__()
        # Define FIN modules for each aspect
        self.team_fins = nn.ModuleDict({
            'shooting': FIN(num_features=2, output_dim=fin_output_dim),
            'turnover': FIN(num_features=2, output_dim=fin_output_dim),
            'rebounding': FIN(num_features=2, output_dim=fin_output_dim),
            'defense': FIN(num_features=3, output_dim=fin_output_dim),
            'ft_foul': FIN(num_features=3, output_dim=fin_output_dim),
            'game_control': FIN(num_features=4, output_dim=fin_output_dim),
        })
        # Each FIN outputs a vector of dimension fin_output_dim.
        # We treat the outputs of the 6 FINs as 6 channels.
        # So for each team, after stacking, the tensor has shape (batch, 6, fin_output_dim).
        self.team_feature_dim = fin_output_dim  # e.g. 16

        # First convolution: takes 6 channels and produces conv1_out_channels.
        # We use kernel size 3 with padding 1 to preserve the spatial dimension.
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=conv1_out_channels, kernel_size=3, padding=1)
        # Second convolution: takes conv1_out_channels and outputs conv2_out_channels,
        # with stride=2 to reduce the spatial (FIN output) dimension by half.
        self.conv2 = nn.Conv1d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=3, stride=2, padding=1)
        # After the FINs, the spatial dimension is fin_output_dim.
        # After conv2, if fin_output_dim == 16, the output spatial dimension will be 8.
        # We'll flatten this feature map.
        self.team_fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # For each team, the final feature vector is conv2_out_channels * (fin_output_dim/2)
        team_vector_dim = conv2_out_channels * (fin_output_dim // 2)
        # Concatenate the two teams' vectors (size = 2 * team_vector_dim) then classify.
        classifier_input_dim = team_vector_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward_team(self, inputs_team):
        # Ensure each input has a batch dimension.
        inputs_team = ensure_batch(inputs_team)
        # Run each FIN and get its embedding.
        # Assume each FIN returns a tensor of shape (batch, fin_output_dim)
        fin_outputs = [self.team_fins[key](inputs_team[key])[0] for key in self.team_fins]
        # Convert each output from (batch, fin_output_dim) to (batch, 1, fin_output_dim)
        fin_outputs = [out.unsqueeze(1) for out in fin_outputs]
        # Stack along channel dimension: (batch, 6, fin_output_dim)
        x = torch.cat(fin_outputs, dim=1)
        # Pass through first conv. Input: (batch, 6, fin_output_dim); output: (batch, conv1_out_channels, fin_output_dim)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        # Second conv: (batch, conv1_out_channels, fin_output_dim) -> (batch, conv2_out_channels, fin_output_dim//2)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # Flatten the spatial dimension: (batch, conv2_out_channels * (fin_output_dim//2))
        x = x.view(x.size(0), -1)
        x = self.team_fc(x)
        return x

    def forward(self, inputs_team_a, inputs_team_b):
        team_a_vector = self.forward_team(inputs_team_a)  # (batch, team_vector_dim)
        team_b_vector = self.forward_team(inputs_team_b)  # (batch, team_vector_dim)
        combined = torch.cat([team_a_vector, team_b_vector], dim=-1)  # (batch, team_vector_dim * 2)
        prob_team_a_wins = self.classifier(combined)
        return prob_team_a_wins