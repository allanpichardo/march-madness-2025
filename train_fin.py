import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os

from data.dataset import SyntheticMarchMadnessDataset
from models.fin import FIN


def save_model(model, fin_key, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{fin_key}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved at {save_path}")


def train_fin(model, fin_key, save_dir, epochs=10, batch_size=64, lr=1e-3):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = SyntheticMarchMadnessDataset(num_samples=1000000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        print(f"Training Epoch {epoch + 1}/{epochs} for {fin_key} FIN")
        for batch in dataloader:
            inputs = batch['inputs'][fin_key]  # Select appropriate FIN input
            labels = batch['label'].unsqueeze(1)

            # Move inputs and labels to GPU if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, preds = model(inputs)
            # Compute loss
            loss = criterion(preds, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

    # Save trained weights
    save_model(model, fin_key, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a specific FIN model.")
    parser.add_argument('--fin_key', type=str, required=True,
                        help="Input feature key: shooting, turnover, rebounding, defense, ft_foul, game_control")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Directory where trained weights will be saved")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training")

    args = parser.parse_args()

    input_dims = {
        'shooting': 2,
        'turnover': 2,
        'rebounding': 2,
        'defense': 3,
        'ft_foul': 3,
        'game_control': 4
    }

    # Validate input
    if args.fin_key not in input_dims:
        raise ValueError(f"Invalid input key '{args.fin_key}'. Choose from {list(input_dims.keys())}")

    # Instantiate the FIN model
    model = FIN(num_features=input_dims[args.fin_key])

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        train_fin(model=model, fin_key=args.fin_key, save_dir=args.save_dir, epochs=args.epochs,
                  batch_size=args.batch_size)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state.")
        save_model(model, args.fin_key, args.save_dir)
        print("Model state saved. Exiting.")