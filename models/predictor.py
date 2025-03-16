import torch
import torch.nn as nn
from models.fin import FIN

def ensure_batch(input_dict):
    """
    Ensure that for keys other than 'seed', each tensor is 3D: (batch, num_games, num_features).
    For the 'seed' key, we expect a 2D tensor: (batch, 20). If it is 1D, unsqueeze it.
    """
    new_dict = {}
    for k, v in input_dict.items():
        if k == "seed":
            # If seed is 1D, add a batch dimension; if 2D already, leave it.
            new_dict[k] = v if v.dim() == 2 else v.unsqueeze(0)
        else:
            # For all other keys, ensure a 3D tensor.
            new_dict[k] = v if v.dim() == 3 else v.unsqueeze(0)
    return new_dict

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
        # After stacking, shape: (batch, 6, fin_output_dim)
        self.team_feature_dim = fin_output_dim  # e.g., 16

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=conv1_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv1_out_channels, out_channels=conv2_out_channels,
                               kernel_size=3, stride=2, padding=1)
        self.team_fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # After conv2, spatial dimension becomes fin_output_dim//2.
        team_vector_dim = conv2_out_channels * (fin_output_dim // 2)
        self.seed_dim = 20  # New seed dimension
        # Each team's final vector will be (team_vector_dim + seed_dim)
        classifier_input_dim = 2 * (team_vector_dim + self.seed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward_team(self, inputs_team):
        # Ensure inputs have proper dimensions.
        inputs_team = ensure_batch(inputs_team)
        # Process each FIN; each output shape: (batch, fin_output_dim)
        fin_outputs = [self.team_fins[key](inputs_team[key])[0] for key in self.team_fins]
        # Unsqueeze and stack along channel dimension to get (batch, 6, fin_output_dim)
        fin_outputs = [out.unsqueeze(1) for out in fin_outputs]
        x = torch.cat(fin_outputs, dim=1)
        # Apply conv layers:
        x = self.conv1(x)           # (batch, conv1_out_channels, fin_output_dim)
        x = nn.ReLU()(x)
        x = self.conv2(x)           # (batch, conv2_out_channels, fin_output_dim//2)
        x = nn.ReLU()(x)
        # Flatten: (batch, conv2_out_channels*(fin_output_dim//2))
        x = x.view(x.size(0), -1)
        x = self.team_fc(x)         # (batch, team_vector_dim)

        # Process the seed: it should have shape (batch, 20).
        if "seed" in inputs_team:
            seed = inputs_team["seed"]
            # If seed is already 2D, leave it; if 1D, unsqueeze.
            if seed.dim() == 1:
                seed = seed.unsqueeze(0)
        else:
            batch_size = x.size(0)
            seed = torch.zeros(batch_size, self.seed_dim, device=x.device, dtype=x.dtype)
        # Concatenate along feature dimension.
        x = torch.cat([x, seed], dim=1)  # (batch, team_vector_dim + 20)
        return x

    def forward(self, inputs_team_a, inputs_team_b):
        team_a_vector = self.forward_team(inputs_team_a)  # (batch, team_vector_dim+20)
        team_b_vector = self.forward_team(inputs_team_b)  # (batch, team_vector_dim+20)
        combined = torch.cat([team_a_vector, team_b_vector], dim=1)  # (batch, 2*(team_vector_dim+20))
        prob_team_a_wins = self.classifier(combined)
        return prob_team_a_wins