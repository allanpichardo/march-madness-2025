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
    model.apply(enable_dropout)
    preds_list = []
    with torch.no_grad():
        for run in range(mc_runs):
            if torch.cuda.is_available():
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(inputs_team_a=inputs_team_a, inputs_team_b=inputs_team_b)
            else:
                logits = model(inputs_team_a=inputs_team_a, inputs_team_b=inputs_team_b)
            probs = torch.sigmoid(logits)
            preds_list.append(probs)
            print("MC run {} complete.".format(run + 1))
    preds_stack = torch.stack(preds_list, dim=0)
    avg_prob = preds_stack.mean(dim=0)
    print("MC dropout inference complete.")
    return avg_prob


def main(args):
    print("Starting prediction script...")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    print("Using device: {}".format(device))

    model = MatchOutcomeTransformer()
    checkpoint_path = "weights/predictor.pth"
    print("Loading model checkpoint from '{}'...".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    conn = sqlite3.connect("sql/madness2025.db")
    cursor = conn.cursor()

    csv_input_path = f"csv/{args.csv_filename}"
    print("Loading CSV file from '{}'...".format(csv_input_path))
    df = pd.read_csv(csv_input_path)
    print("CSV loaded. Total matchups: {}".format(len(df)))

    # Cache latest day per season and create one dataset instance for all seasons
    seasons_in_csv = set()
    for idx, row in df.iterrows():
        matchup_id = row["ID"]
        season = int(matchup_id.split("_")[0])
        seasons_in_csv.add(season)
    seasons_in_csv = sorted(list(seasons_in_csv))
    season_latest_day = {}
    for season in seasons_in_csv:
        cursor.execute("SELECT MAX(DayNum) FROM TeamGameStats WHERE Season = ?", (season,))
        season_latest_day[season] = cursor.fetchone()[0]
    print("Cached latest day per season: {}".format(season_latest_day))

    print("Instantiating dataset for seasons: {}...".format(seasons_in_csv))
    dataset = MarchMadnessDataset(conn, seasons=seasons_in_csv, num_games=5, matchup=True)
    print("Dataset instantiated.")

    team_input_cache = {}

    # --- Gather inputs for all matchups (batch processing) ---
    print("Starting data gathering loop for matchups...")
    batch_inputs_team_a = {}
    batch_inputs_team_b = {}
    matchup_ids = []
    team_a_ids = []
    team_b_ids = []

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

        key_a = (season, team_a)
        if key_a not in team_input_cache:
            team_input_cache[key_a] = dataset.get_inputs(season, team_a, latest_day)
            print("Computed inputs for Season {}, Team {}.".format(season, team_a))
        inputs_a = team_input_cache[key_a]

        key_b = (season, team_b)
        if key_b not in team_input_cache:
            team_input_cache[key_b] = dataset.get_inputs(season, team_b, latest_day)
            print("Computed inputs for Season {}, Team {}.".format(season, team_b))
        inputs_b = team_input_cache[key_b]

        # The predictor now handles both batched and single-sample inputs.
        # Convert each input tensor to batch form if necessary.
        inputs_a = {k: (v if v.dim() == 3 else v.unsqueeze(0)).to(device) for k, v in inputs_a.items()}
        inputs_b = {k: (v if v.dim() == 3 else v.unsqueeze(0)).to(device) for k, v in inputs_b.items()}

        # Append to batch lists per key.
        for key, tensor in inputs_a.items():
            if key not in batch_inputs_team_a:
                batch_inputs_team_a[key] = []
            batch_inputs_team_a[key].append(tensor)
        for key, tensor in inputs_b.items():
            if key not in batch_inputs_team_b:
                batch_inputs_team_b[key] = []
            batch_inputs_team_b[key].append(tensor)

        if (i + 1) % 100 == 0:
            print("Processed {} / {} matchups.".format(i + 1, len(df)))
    print("Data gathering complete. Total matchups processed: {}".format(len(df)))

    # Stack the lists into batched tensors.
    for key in batch_inputs_team_a:
        batch_inputs_team_a[key] = torch.cat(batch_inputs_team_a[key], dim=0)
    for key in batch_inputs_team_b:
        batch_inputs_team_b[key] = torch.cat(batch_inputs_team_b[key], dim=0)
    print("Batched inputs created for Team A and Team B.")

    # --- Run inference on the batched data ---
    print("Starting batched inference using MC dropout...")
    with torch.no_grad():
        avg_probs = mc_predict(model, batch_inputs_team_a, batch_inputs_team_b, mc_runs=args.mc_runs)
    avg_probs = avg_probs.squeeze(1)
    print("Inference complete on {} matchups.".format(avg_probs.size(0)))

    predictions = []
    for i in range(avg_probs.size(0)):
        predictions.append({
            "ID": matchup_ids[i],
            "TeamA": team_a_ids[i],
            "TeamB": team_b_ids[i],
            "Pred": avg_probs[i].item()
        })
        print(f"Matchup {matchup_ids[i]}: Probability Team {team_a_ids[i]} wins = {avg_probs[i].item():.4f}")

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