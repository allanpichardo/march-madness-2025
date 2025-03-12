import argparse
import os
import sqlite3

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from data.dataset import MarchMadnessDataset
from models.predictor import MatchOutcomePredictor

# ------------------ Helper Functions ------------------ #
def brier_score(probs, labels):
    """
    Computes the Brier score as the mean squared error between
    predicted probabilities and true labels.
    """
    return torch.mean((probs - labels) ** 2)

def enable_dropout(m):
    """
    Enables dropout layers during inference.
    """
    if isinstance(m, nn.Dropout):
        m.train()

def mc_predict(model, inputs_team_a, inputs_team_b, mc_runs=10):
    """
    Performs Monte Carlo dropout inference.
    Temporarily enables dropout and runs several forward passes,
    then returns the average predicted probability.
    """
    model.eval()
    model.apply(enable_dropout)  # Force dropout layers to remain active
    preds_list = []
    with torch.no_grad():
        for _ in range(mc_runs):
            logits = model(inputs_team_a=inputs_team_a, inputs_team_b=inputs_team_b)
            probs = torch.sigmoid(logits)
            preds_list.append(probs)
    preds_stack = torch.stack(preds_list, dim=0)
    avg_prob = preds_stack.mean(dim=0)
    return avg_prob
# ------------------------------------------------------ #

def main(args):
    # Detect device: Use GPU if available, otherwise fallback to CPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # Use Metal Performance Shaders (Apple Silicon)
    print(f"Using device: {device}")

    # Connect to DB
    conn = sqlite3.connect(args.db_path)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Dataset setup
    print("Loading dataset...")
    full_dataset = MarchMadnessDataset(conn, seasons=args.seasons, num_games=5, matchup=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model initialization (using the transformer version)
    model = MatchOutcomePredictor().to(device)

    # Load model checkpoint if resuming training
    checkpoint_path = os.path.join(args.weights_dir, "predictor.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found. Initializing FIN weights from pretrained models.")
        for fin_key, fin_model in model.team_fins.items():
            fin_weights_path = os.path.join(args.weights_dir, f"{fin_key}.pth")
            if os.path.exists(fin_weights_path):
                print(f"Loading FIN weights for {fin_key} from {fin_weights_path}")
                fin_weights = torch.load(fin_weights_path, map_location=device)
                model.team_fins[fin_key].load_state_dict(fin_weights)
                model.team_fins[fin_key] = model.team_fins[fin_key].to(device)
            else:
                print(f"WARNING: No pretrained weights found for {fin_key}, initializing randomly.")

    model.train()
    criterion = BCEWithLogitsLoss().to(device)
    optimizer = Adam([
        {'params': model.team_fins.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ])

    # Early stopping setup using Brier score
    best_val_brier = float('inf')
    patience_counter = 0
    batch_log_interval = 20

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        train_brier = 0.0
        batch_idx = 0

        for batch in train_loader:
            inputs_a = {key: tensor.to(device, non_blocking=True) for key, tensor in batch["inputs_team_a"].items()}
            inputs_b = {key: tensor.to(device, non_blocking=True) for key, tensor in batch["inputs_team_b"].items()}
            labels = batch["label"].unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            if device.type != "mps":
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)
                    loss = criterion(logits, labels)
            else:
                logits = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(logits)
            batch_brier = brier_score(probs, labels)
            train_brier += batch_brier.item()
            preds_binary = (probs > 0.5).float()
            correct_train += (preds_binary == labels).sum().item()
            total_train += labels.size(0)

            if batch_idx % batch_log_interval == 0:
                writer.add_scalar("Loss/Batch", loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Brier/Batch", batch_brier.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Accuracy/Batch", (correct_train / total_train) * 100,
                                  epoch * len(train_loader) + batch_idx)
            print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}: Loss: {loss.item():.4f}, Brier: {batch_brier.item():.4f}, Train Accuracy: {(correct_train/total_train)*100:.2f}%")
            batch_idx += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_train_brier = train_brier / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Brier: {avg_train_brier:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Brier/Train", avg_train_brier, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Validation phase with MC dropout for uncertainty estimates
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_brier = 0.0

        print("Evaluating...")
        with torch.no_grad():
            for batch in val_loader:
                inputs_a = {key: tensor.to(device, non_blocking=True) for key, tensor in batch["inputs_team_a"].items()}
                inputs_b = {key: tensor.to(device, non_blocking=True) for key, tensor in batch["inputs_team_b"].items()}
                labels = batch["label"].unsqueeze(1).to(device, non_blocking=True)

                if device.type != "mps":
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        logits = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)
                        loss = criterion(logits, labels)
                else:
                    logits = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)
                    loss = criterion(logits, labels)
                val_loss += loss.item()

                # Use MC dropout to get a probabilistic prediction for metrics
                avg_prob = mc_predict(model, inputs_a, inputs_b, mc_runs=10)
                batch_brier = brier_score(avg_prob, labels)
                val_brier += batch_brier.item()
                preds_binary = (avg_prob > 0.5).float()
                correct_val += (preds_binary == labels).sum().item()
                total_val += labels.size(0)
                print(f"Eval Batch: Loss: {loss.item():.4f}, Brier: {batch_brier.item():.4f}, Val Accuracy: {(correct_val/total_val)*100:.2f}%")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_brier = val_brier / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100

        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Brier: {avg_val_brier:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Brier/Validation", avg_val_brier, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        # Early stopping based on Brier score improvement
        if avg_val_brier < best_val_brier - args.brier_threshold:
            best_val_brier = avg_val_brier
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without sufficient improvement in Brier score.")
                break

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='sql/madness2025.db')
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--brier_threshold', type=float, default=0.01,
                        help="Minimum improvement in Brier score to reset patience")
    parser.add_argument('--seasons', nargs='+', type=int, default=[year for year in range(1984, 2025)])
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("Starting training...")
    print(f"Using database: {args.db_path}")
    print(f"Saving weights to: {args.weights_dir}")
    print(f"Logging to: {args.log_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Resuming from checkpoint: {args.resume}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Brier improvement threshold: {args.brier_threshold}")
    print(f"Seasons: {args.seasons}")

    main(args)