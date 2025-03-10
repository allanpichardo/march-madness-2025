import pandas as pd
import sqlite3
import torch
from data.dataset import MarchMadnessDataset
from models.predictor import MatchOutcomeTransformer

# Set up device and load the trained model
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")  # Use Metal Performance Shaders (Apple Silicon)
model = MatchOutcomeTransformer()  # Change to your model class if needed
checkpoint_path = "weights/predictor.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Connect to the database (adjust path if needed)
conn = sqlite3.connect("sql/madness2025.db")
cursor = conn.cursor()

# Load the CSV file containing sample submissions
df = pd.read_csv("csv/SampleSubmissionStage1.csv")

# Prepare a list to hold the new predictions
new_predictions = []

# Iterate over each row in the CSV file
for idx, row in df.iterrows():
    matchup_id = row["ID"]  # Format: season_teama_teamb, e.g. "2021_1101_1102"
    parts = matchup_id.split("_")
    season = int(parts[0])
    team_a = int(parts[1])
    team_b = int(parts[2])

    # Query the database to get the latest day for this season
    cursor.execute("SELECT MAX(DayNum) FROM TeamGameStats WHERE Season = ?", (season,))
    latest_day = cursor.fetchone()[0]

    # Create a dataset instance for this season (preloads necessary data)
    dataset = MarchMadnessDataset(conn, seasons=[season], num_games=5, matchup=True)

    # Get the most recent inputs for each team using the latest day
    inputs_team_a = dataset.get_inputs(season, team_a, latest_day)
    inputs_team_b = dataset.get_inputs(season, team_b, latest_day)

    # Move inputs to the proper device and add batch dimension if necessary.
    inputs_team_a = {
        key: (tensor.unsqueeze(0) if tensor.dim() == 2 else tensor).to(device)
        for key, tensor in inputs_team_a.items()
    }
    inputs_team_b = {
        key: (tensor.unsqueeze(0) if tensor.dim() == 2 else tensor).to(device)
        for key, tensor in inputs_team_b.items()
    }

    # Run inference
    with torch.no_grad():
        logits = model(inputs_team_a=inputs_team_a, inputs_team_b=inputs_team_b)
        # Convert raw logits to probability (using sigmoid)
        prob_team_a_wins = torch.sigmoid(logits).item()

    new_predictions.append(prob_team_a_wins)
    print(f"Matchup {matchup_id}: Probability Team {team_a} wins = {prob_team_a_wins:.4f}")

# Update the DataFrame with the new predictions and save to CSV
df["Pred"] = new_predictions
df.to_csv("predictions/SampleSubmissionStage1.csv", index=False)
print("New predictions saved to predictions/SampleSubmissionStage1.csv")