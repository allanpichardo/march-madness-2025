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
    Perform MC dropout inference on batched inputs: run multiple forward passes with dropout active,
    then average the probabilities.
    Assumes that inputs_team_a and inputs_team_b are dictionaries with tensors of shape
    (batch_size, num_games, num_features) for each key.
    """
    print("Starting MC dropout inference with {} runs...".format(mc_runs))
    model.eval()
    model.apply(enable_dropout)  # Activate dropout layers during inference

    # Determine batch size from one of the input tensors
    example_tensor = next(iter(inputs_team_a.values()))
    batch_size = example_tensor.shape[0]

    preds_list = []
    with torch.no_grad():
        for run in range(mc_runs):
            # Only use autocast if CUDA is available and batch size > 1 to avoid kernel config errors
            if torch.cuda.is_available() and batch_size > 1:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(inputs_team_a=inputs_team_a, inputs_team_b=inputs_team_b)
            else:
                logits = model(inputs_team_a=inputs_team_a, inputs_team_b=inputs_team_b)
            probs = torch.sigmoid(logits)
            preds_list.append(probs)
            print("MC run {} complete.".format(run + 1))
    preds_stack = torch.stack(preds_list, dim=0)  # (mc_runs, batch_size, 1)
    avg_prob = preds_stack.mean(dim=0)  # (batch_size, 1)
    print("MC dropout inference complete.")
    return avg_prob


def main(args):
    print("Starting prediction script...")
    # Set up device and load the trained model
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    print("Using device: {}".format(device))

    model = MatchOutcomeTransformer().to(device)
    checkpoint_path = "weights/predictor.pth"
    print("Loading model checkpoint from '{}'...".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Connect to the database
    conn = sqlite3.connect("sql/madness2025.db")
    cursor = conn.cursor()

    # Load the CSV file
    csv_input_path = f"csv/{args.csv_filename}"
    print("Loading CSV file from '{}'...".format(csv_input_path))
    df = pd.read_csv(csv_input_path)
    print("CSV loaded. Total matchups: {}".format(len(df)))

    # Cache the latest day per season
    season_latest_day = {}
    for idx, row in df.iterrows():
        matchup_id = row["ID"]  # Format: season_teama_teamb
        season = int(matchup_id.split("_")[0])
        if season not in season_latest_day:
            cursor.execute("SELECT MAX(DayNum) FROM TeamGameStats WHERE Season = ?", (season,))
            season_latest_day[season] = cursor.fetchone()[0]
    print("Cached latest day per season: {}".format(season_latest_day))

    # Instantiate one dataset covering all seasons in the CSV
    seasons_in_csv = sorted(list(season_latest_day.keys()))
    print("Instantiating dataset for seasons: {}...".format(seasons_in_csv))
    dataset = MarchMadnessDataset(conn, seasons=seasons_in_csv, num_games=5, matchup=True)
    print("Dataset instantiated.")

    # Cache team inputs so we don't recompute the same ones multiple times.
    # Key: (season, team) -> computed inputs.
    team_input_cache = {}

    # --- First Loop: Gather and cache inputs for all matchups ---
    print("Starting data gathering loop for matchups...")
    matchup_ids = []
    team_a_ids = []
    team_b_ids = []
    # We'll store the already prepared inputs in lists, one per matchup.
    sample_inputs_team_a = []
    sample_inputs_team_b = []

    for i, row in df.iterrows():
        matchup_id = row["ID"]
        parts = matchup_id.split("_")
        season = int(parts[0])
        team_a = int(parts[1])
        team_b = int(parts[2])
        matchup_ids.append(matchup_id)
        team_a_ids.append(team_a)
        team_b_ids.append(team_b)
        latest_day = season_latest_day[season]

        # Retrieve or compute inputs for team A
        key_a = (season, team_a)
        if key_a not in team_input_cache:
            team_input_cache[key_a] = dataset.get_inputs(season, team_a, latest_day)
            print("Computed inputs for Season {}, Team {}.".format(season, team_a))
        inputs_a = team_input_cache[key_a]

        # Retrieve or compute inputs for team B
        key_b = (season, team_b)
        if key_b not in team_input_cache:
            team_input_cache[key_b] = dataset.get_inputs(season, team_b, latest_day)
            print("Computed inputs for Season {}, Team {}.".format(season, team_b))
        inputs_b = team_input_cache[key_b]

        # Add batch dimension if needed and move to device.
        sample_a = {k: (v if v.dim() == 3 else v.unsqueeze(0)).to(device) for k, v in inputs_a.items()}
        sample_b = {k: (v if v.dim() == 3 else v.unsqueeze(0)).to(device) for k, v in inputs_b.items()}
        sample_inputs_team_a.append(sample_a)
        sample_inputs_team_b.append(sample_b)

        if (i + 1) % 100 == 0:
            print("Processed {} / {} matchups.".format(i + 1, len(df)))
    print("Data gathering complete. Total matchups processed: {}".format(len(df)))

    # --- Second Loop: Run inference on the batched data ---
    print("Stacking inputs into batches...")
    # For each key, stack the corresponding tensors across matchups.
    batch_inputs_team_a = {}
    batch_inputs_team_b = {}
    for key in sample_inputs_team_a[0].keys():
        batch_inputs_team_a[key] = torch.cat([sample[key] for sample in sample_inputs_team_a], dim=0)
    for key in sample_inputs_team_b[0].keys():
        batch_inputs_team_b[key] = torch.cat([sample[key] for sample in sample_inputs_team_b], dim=0)
    print("Batched inputs created for Team A and Team B.")

    print("Starting batched inference using MC dropout...")
    with torch.no_grad():
        avg_probs = mc_predict(model, batch_inputs_team_a, batch_inputs_team_b, mc_runs=args.mc_runs)
    avg_probs = avg_probs.squeeze(1)  # (batch_size,)
    print("Inference complete on {} matchups.".format(avg_probs.size(0)))

    # Create predictions with only ID and Pred columns.
    predictions = []
    for i in range(avg_probs.size(0)):
        predictions.append({
            "ID": matchup_ids[i],
            "Pred": avg_probs[i].item()
        })
        print(f"Matchup {matchup_ids[i]}: Probability = {avg_probs[i].item():.4f}")

    pred_df = pd.DataFrame(predictions)
    csv_output_path = f"predictions/{args.csv_filename}"
    pred_df.to_csv(csv_output_path, index=False)
    print(f"New predictions saved to {csv_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_filename", type=str, required=True,
                        help="Name of the CSV file in the 'csv' directory (e.g., SampleSubmissionStage1.csv)")
    parser.add_argument("--mc_runs", type=int, default=10,
                        help="Number of MC dropout runs (default: 10)")
    args = parser.parse_args()
    main(args)