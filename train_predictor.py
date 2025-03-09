import torch
from torch.utils.data import DataLoader, random_split
import argparse
import os
from models.predictor import MatchOutcomePredictor
from data.dataset import MarchMadnessDataset
import sqlite3
from torch.nn import BCELoss
from torch.optim import Adam


def main(args):
    # Connect to DB
    conn = sqlite3.connect(args.db_path)

    # Dataset setup with matchups enabled
    print("Loading dataset...")
    full_dataset = MarchMadnessDataset(conn, seasons=args.seasons, num_games=5, matchup=True)

    # Split dataset into train and validation sets (80% train, 20% validation)
    print("Splitting dataset into train and validation sets...")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model initialization
    model = MatchOutcomePredictor()

    # Check if a model checkpoint exists
    checkpoint_path = os.path.join(args.weights_dir, "predictor.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
    else:
        print("Loading pretrained FIN weights.")
        for fin_key, fin_model in model.team_fins.items():
            fin_weights_path = os.path.join(args.weights_dir, f"{fin_key}.pth")
            if os.path.exists(fin_weights_path):
                print(f"Loading pretrained weights for {fin_key} from {fin_weights_path}")
                fin_weights = torch.load(fin_weights_path)
                model.team_fins[fin_key].load_state_dict(fin_weights)
            else:
                print(f"Warning: No pretrained weights found for {fin_key}, initializing randomly.")

    model.train()

    criterion = BCELoss()
    optimizer = Adam([
        {'params': model.team_fins.parameters(), 'lr': 1e-5},  # Small LR for FIN fine-tuning
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ])

    # Early stopping setup
    best_val_loss = float('inf')
    patience = args.patience  # Number of epochs without improvement before stopping
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0  # Track correct predictions
        total_train = 0  # Track total predictions

        for batch in train_loader:
            inputs_a, inputs_b, labels = batch["inputs_team_a"], batch["inputs_team_b"], batch["label"].unsqueeze(1)

            optimizer.zero_grad()
            preds = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Compute accuracy
            predicted = (preds >= 0.5).float()  # Convert probabilities to binary (0/1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            print(f"Epoch {epoch+1}/{args.epochs}, Batch Loss: {loss.item():.4f}, Batch Accuracy: {correct_train/total_train:.2f}")

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100  # Convert to percentage

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        print("Starting validation...")
        model.eval()
        val_loss = 0.0
        correct_val = 0  # Track correct validation predictions
        total_val = 0  # Track total validation predictions

        with torch.no_grad():
            for batch in val_loader:
                inputs_a, inputs_b, labels = batch["inputs_team_a"], batch["inputs_team_b"], batch["label"].unsqueeze(1)
                preds = model(inputs_team_a=inputs_a, inputs_team_b=inputs_b)
                loss = criterion(preds, labels)
                val_loss += loss.item()

                # Compute accuracy
                predicted = (preds >= 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                print(f"Validation Batch Loss: {loss.item():.4f}, Validation Batch Accuracy: {correct_val/total_val:.2f}")

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100  # Convert to percentage

        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience
            torch.save(model.state_dict(), checkpoint_path)  # Save best model
            print(f"Validation loss improved. Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs.")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break  # Stop training if no improvement

    print(f"Training complete. Best model saved at {checkpoint_path}")


if __name__ == "__main__":
    default_seasons = [year for year in range(1984, 2025)]  # Default seasons from 1984 to 2025

    parser = argparse.ArgumentParser(description='Train March Madness Predictor')
    parser.add_argument('--db_path', type=str, default='sql/madness2025.db', help='Path to sqlite DB')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory with pretrained weights and to save models')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for predictor layers')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from existing weights')
    parser.add_argument('--seasons', nargs='+', type=int, default=default_seasons, help='Seasons to use for training')
    parser.add_argument('--patience', type=int, default=5, help='Epochs without improvement before early stopping')

    args = parser.parse_args()
    print(f"Using seasons: {args.seasons}")
    print(f"Using batch size: {args.batch_size}")
    print(f"Using learning rate: {args.lr}")
    print(f"Using epochs: {args.epochs}")
    print(f"Using patience: {args.patience}")
    print(f"Using resume: {args.resume}")
    print(f"Using weights directory: {args.weights_dir}")

    # Create weights directory if it does not exist
    os.makedirs(args.weights_dir, exist_ok=True)

    main(args)