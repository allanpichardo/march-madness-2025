import argparse
import pandas as pd
import sqlite3
import torch
from data.dataset import MarchMadnessDataset
from models.predictor import MatchOutcomeTransformer
import torch.nn as nn

def enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

def mc_predict(model, inputs_team_a, inputs_team_b, mc_runs=10):
    """
    Perform MC dropout inference: run multiple forward passes with dropout active,
    then average the probabilities.
    """
    model.eval()
    model.apply(enable_dropout)  # Activate dropout layers during inference
    preds_list = []
    with torch.no_grad():
        for _ in range(mc_runs):
            logits = model(inputs_team_a=inputs_team_a, inputs_team_b=inputs_team_b)
            probs = torch.sigmoid(logits)
            preds_list.append(probs)
    preds_stack = torch.stack(preds_list, dim=0)
    avg_prob = preds_stack.mean(dim=0)
    return avg_prob

def main(args):
    # Set up device and load the trained model
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    model = MatchOutcomeTransformer()
    checkpoint_path = "weights/predictor.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Connect to the database (adjust path if needed)
    conn = sqlite3.connect("sql/madness2025.db")
    cursor = conn.cursor()

    # Load the CSV file from the csv directory
    csv_input_path = f"csv/{args.csv_filename}"
    df = pd.read_csv(csv_input_path)

    # Cache latest day and dataset per season
    unique_seasons = set()
    for idx, row in df.iterrows():
        matchup_id = row["ID"]  # Format: season_teama_teamb
        parts = matchup_id.split("_")
        season = int(parts[0])
        unique_seasons.add(season)
    season_to_latest_day = {}
    season_to_dataset = {}
    for season in unique_seasons:
        cursor.execute("SELECT MAX(DayNum) FROM TeamGameStats WHERE Season = ?", (season,))
        latest_day = cursor.fetchone()[0]
        season_to_latest_day[season] = latest_day
        season_to_dataset[season] = MarchMadnessDataset(conn, seasons=[season], num_games=5, matchup=True)

    new_predictions = []
    # Iterate over each row in the CSV file
    for idx, row in df.iterrows():
        matchup_id = row["ID"]
        parts = matchup_id.split("_")
        season = int(parts[0])
        team_a = int(parts[1])
        team_b = int(parts[2])

        latest_day = season_to_latest_day[season]
        dataset = season_to_dataset[season]
        inputs_team_a = dataset.get_inputs(season, team_a, latest_day)
        inputs_team_b = dataset.get_inputs(season, team_b, latest_day)
        # Add batch dimension if necessary and move to device
        inputs_team_a = {key: (tensor.unsqueeze(0) if tensor.dim() == 2 else tensor).to(device)
                         for key, tensor in inputs_team_a.items()}
        inputs_team_b = {key: (tensor.unsqueeze(0) if tensor.dim() == 2 else tensor).to(device)
                         for key, tensor in inputs_team_b.items()}

        with torch.no_grad():
            avg_prob = mc_predict(model, inputs_team_a, inputs_team_b, mc_runs=20)
            prob_team_a_wins = avg_prob.item()

        new_predictions.append(prob_team_a_wins)
        print(f"Matchup {matchup_id}: Probability Team {team_a} wins = {prob_team_a_wins:.4f}")

    df["Pred"] = new_predictions
    csv_output_path = f"predictions/{args.csv_filename}"
    df.to_csv(csv_output_path, index=False)
    print(f"New predictions saved to {csv_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_filename", type=str, required=True,
                        help="Name of the CSV file in the 'csv' directory (e.g., SampleSubmissionStage1.csv)")
    args = parser.parse_args()
    main(args)