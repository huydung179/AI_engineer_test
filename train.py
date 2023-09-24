import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb as w

from data_preparation import prepare_train_valid
from utils import load_model


def parse_arguments():
    """
    Parse Paths for training data and saving the model.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Training script for the model.")
    parser.add_argument('--data_dir', type=str, default='./data/train_images',
                        help='Directory for training images.')
    parser.add_argument('--save_dir', type=str, default='./output_models',
                        help='Directory to save output models.')

    return parser.parse_args()


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    train_loader: torch.utils.data.DataLoader,
                    prog_bar: tqdm,
                    device: torch.device) -> None:
    """
    Train the model for one epoch.

    :param model: PyTorch model to train
    :param criterion: Loss function used for optimization
    :param optimizer: Optimizer used for backpropagation
    :param train_loader: DataLoader for training data
    :param prog_bar: Progress bar object to update training progress (command-line and WandB logger)
    :param device: Computation device ('cuda' or 'cpu')
    """

    model.train()
    for images, labels in train_loader:
        # Move images and labels to device
        images, labels = images.to(device), labels.float().to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(images).squeeze()

        # Loss calculation and backpropagation
        loss = criterion(output, labels)
        prog_bar.set_postfix(loss=f'{loss.item():.3f}')
        loss.backward()
        optimizer.step()

        # Log the training loss
        w.log({"training_loss": loss.item()})


def validate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             valid_loader: torch.utils.data.DataLoader,
             device: torch.device) -> (float, float):
    """
    Validate the model and return average loss and f1 score.

    :param model: Trained PyTorch model
    :param criterion: Loss function used
    :param valid_loader: DataLoader for validation data
    :param device: Computation device ('cuda' or 'cpu')
    :return: average validation loss, f1 score
    """
    model.eval()
    loss = 0
    outputs, targets = [], []

    # Evaluation without gradient calculation
    with torch.no_grad():
        for images, labels in valid_loader:
            # Predict and calculate batch loss
            images, labels = images.to(device), labels.float().to(device)
            output = model(images).squeeze()
            loss += criterion(output, labels).item()

            # Store predictions and targets for calculating f1 score
            outputs.append(output.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    # Average loss and f1 score calculation
    loss /= len(valid_loader)
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    preds = (outputs > 0).astype(int)
    f1 = f1_score(targets, preds, average='macro')

    # Log the validation loss and f1 score to WandB
    w.log({"validation_loss": loss, "validation_f1": f1})

    return loss, f1


def initialize_device() -> torch.device:
    """Initialize CUDA device if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_model(model_name: str, device: torch.device) -> (torch.nn.Module, torch.nn.parameter.Parameter):
    """
    Setup the model for training.

    :param model_name: Name of the model to initialize
    :param device: Computation device ('cuda' or 'cpu')
    :return: Initialized model, parameters to optimize (to pass to the optimizer)
    """
    model, params2optimize = load_model(
        model_name, 1, pretrained=True, training=True)
    return model.to(device), params2optimize


def train_model(model: torch.nn.Module,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_loader: torch.utils.data.DataLoader,
                valid_loader: torch.utils.data.DataLoader,
                save_name: str,
                device: torch.device) -> None:
    """
    Train the model and save when validation loss improves.

    :param model: PyTorch model to train
    :param criterion: Loss function used for optimization
    :param optimizer: Optimizer used for backpropagation
    :param train_loader: DataLoader for training data
    :param valid_loader: DataLoader for validation data
    :param save_name: Name for saving the model
    :param device: Computation device ('cuda' or 'cpu')
    """

    # Initialization and setup
    prog_bar = tqdm(range(w.config.epochs))
    os.makedirs(args.save_dir, exist_ok=True)
    cur_loss = np.inf

    # Training loop
    for epoch in prog_bar:
        # Update progress bar
        prog_bar.set_description(f'Epoch: {epoch+1}')

        # Train and validate
        train_one_epoch(model, criterion, optimizer,
                        train_loader, prog_bar, device)
        valid_loss, valid_f1 = validate(model, criterion, valid_loader, device)

        # Save the model if validation loss improves
        if valid_loss < cur_loss:
            cur_loss = valid_loss
            w.log({"best_validation_loss": cur_loss,
                  "best_validation_f1": valid_f1})
            prog_bar.write(
                f'Epoch: {epoch+1} - Valid loss decreased ({cur_loss:.3f}) - Valid F1 score ({valid_f1:.3f}). Saving model ...')
            torch.save(model.state_dict(), os.path.join(
                args.save_dir, f'{save_name}.pt'))


def main():
    # Initialize WandB
    w.init(project='trimble-test')
    # Define the run name and the model name
    w.run.name = f"{w.config['model_name']}_{w.config['learning_rate']}"

    # Initialize the device, model, loss function, optimizer, and data loaders
    device = initialize_device()
    model, params2optimize = setup_model(w.config['model_name'], device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params2optimize, lr=w.config['learning_rate'])
    train_loader, valid_loader = prepare_train_valid(
        args.data_dir, w.config.batch_size)

    # Train the model
    train_model(model, criterion, optimizer, train_loader,
                valid_loader, w.run.name, device)


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    # Run the training script
    main()
