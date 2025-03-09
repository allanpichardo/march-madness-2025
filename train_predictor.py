import torch
from torch.utils.data import DataLoader, random_split
import argparse
import os
from models.predictor import MatchOutcomePredictor
from data.dataset import MarchMadnessDataset
import sqlite3
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

def main(args):
    # Connect to DB
    conn = sqlite3.connect(args.db_path)

    writer = SummaryWriter(log_dir=args.log_dir)

    # Dataset setup
    print("Loading dataset...")
    full_dataset = MarchMadnessDataset(conn, seasons=args.seasons, num_games=5, matchup=True)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model initialization
    model = MatchOutcomePredictor()

    # Load model checkpoint
    checkpoint_path = os.path.join(args.weights_dir, "predictor.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    model.train()
    criterion = BCELoss()
    optimizer = Adam([
        {'params': model.team_fins.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ])

    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0

    batch_log_interval = 20

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        batch_idx = 0

        for batch in train_loader:
            inputs_a, inputs_b, labels = batch["inputs_team_a"], batch["inputs_team_b"], batch["label"].unsqueeze(1)
            optimizer.zero_grad()
            preds = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct_train += (preds.round() == labels).sum().item()
            total_train += labels.size(0)

            if batch_idx % batch_log_interval == 0:
                writer.add_scalar("Loss/Batch", loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Accuracy/Batch", (correct_train / total_train) * 100, epoch * len(train_loader) + batch_idx)

            print(f"Epoch {epoch+1}/{args.epochs}, Batch Loss: {loss.item():.4f}, Train Accuracy: {(correct_train / total_train) * 100:.2f}%")
            batch_idx += 1

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        print("Evaluating...")
        with torch.no_grad():
            for batch in val_loader:
                inputs_a, inputs_b, labels = batch["inputs_team_a"], batch["inputs_team_b"], batch["label"].unsqueeze(1)
                preds = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                correct_val += (preds.round() == labels).sum().item()
                total_val += labels.size(0)
                print(f"Eval {epoch+1}/{args.epochs}, Batch Loss: {loss.item():.4f}, Val Accuracy: {(correct_val / total_val) * 100:.2f}%")

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100

        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
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
    print(f"Seasons: {args.seasons}")


    main(args)