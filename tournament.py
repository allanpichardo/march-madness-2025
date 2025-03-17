#!/usr/bin/env python
import argparse
import csv
import sqlite3
import pandas as pd

def load_team_names(db_path):
    """Load team names from the TeamNames table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT TeamID, TeamName FROM TeamNames")
    team_names = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return team_names

def load_team_seeds(db_path, season_filter="2025"):
    """
    Load seeds for a given season.
    Returns a dictionary mapping TeamID -> Seed.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT TeamID, Seed FROM Seeds WHERE Season = ?", (season_filter,))
    seeds = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return seeds

def parse_submission_file(csv_path, season_filter="2025", seeds_dict=None):
    """
    Parse the submission CSV file.
    Expects a header with columns: ID,Pred.
    The ID is in the format season_teamA_teamB.
    Only keeps rows where the season matches and, if seeds_dict is provided,
    both teams are seeded.
    Returns a list of dictionaries with keys: season, team_a, team_b, outcome.
    """
    submissions = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id_str = row["ID"]
            parts = id_str.split("_")
            if len(parts) != 3:
                continue  # Skip malformed rows.
            season, team_a_str, team_b_str = parts
            if season != season_filter:
                continue  # Only process rows for the specified season.
            team_a = int(team_a_str)
            team_b = int(team_b_str)
            # If seeds_dict is provided, only include matchups where both teams are seeded.
            if seeds_dict is not None:
                if team_a not in seeds_dict or team_b not in seeds_dict:
                    continue
            outcome = float(row["Pred"])
            submissions.append({
                "season": season,
                "team_a": team_a,
                "team_b": team_b,
                "outcome": outcome
            })
    return submissions

def write_bracket_csv(submissions, team_names, output_csv):
    """
    Write a new CSV file with columns: "Team A", "Team B", "Outcome", "League".
    League is determined as "men" if team A's ID is less than 3000, "women" otherwise.
    """
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Team A", "Team B", "Outcome", "League"])
        writer.writeheader()
        for entry in submissions:
            team_a_id = entry["team_a"]
            team_b_id = entry["team_b"]
            team_a_name = team_names.get(team_a_id, f"Team {team_a_id}")
            team_b_name = team_names.get(team_b_id, f"Team {team_b_id}")
            league = "men" if team_a_id < 3000 else "women"
            writer.writerow({
                "Team A": team_a_name,
                "Team B": team_b_name,
                "Outcome": entry["outcome"],
                "League": league
            })

def main(args):
    print("Loading team names from database...")
    team_names = load_team_names(args.db_path)
    print(f"Loaded {len(team_names)} team names.")

    print("Loading seeds for season", args.season, "...")
    seeds = load_team_seeds(args.db_path, season_filter=args.season)
    print(f"Loaded seeds for {len(seeds)} teams.")

    print(f"Parsing submission file: {args.submission_csv} for season {args.season}...")
    submissions = parse_submission_file(args.submission_csv, season_filter=args.season, seeds_dict=seeds)
    print(f"Found {len(submissions)} matchups for season {args.season} where both teams are seeded.")

    print(f"Writing bracket CSV to {args.output_csv}...")
    write_bracket_csv(submissions, team_names, args.output_csv)
    print("Bracket CSV created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bracket CSV from submission file and team names, filtering for seeded teams.")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the SQLite database file.")
    parser.add_argument("--submission_csv", type=str, required=True, help="Path to the submission CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path for the output CSV file.")
    parser.add_argument("--season", type=str, default="2025", help="Season to filter on (default: 2025)")
    args = parser.parse_args()
    main(args)