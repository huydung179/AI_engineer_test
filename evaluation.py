import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from data_preparation import prepare_test
from utils import load_model


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")

    parser.add_argument('--data_dir', type=str,
                        default='./data/additional_test_images', help="Path to the test images.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for DataLoader.")
    parser.add_argument('--model_path', type=str, default='./output_models/swin_v2_t_0.003.pt',
                        help="Full path to the model checkpoint.")
    parser.add_argument('--model_name', type=str, default='swin_v2_t',
                        help="Name of the model architecture.")

    return parser.parse_args()


def predict(model: torch.nn.Module, valid_loader: torch.utils.data.DataLoader, device: torch.device) -> (np.ndarray, np.ndarray):
    """
    Make predictions using the model for the given data.

    :param model: Trained PyTorch model.
    :param valid_loader: DataLoader for the validation/test data.
    :param device: Computation device ('cuda' or 'cpu').
    :return: Model's outputs and true targets.
    """
    model.eval()
    outputs, targets = [], []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            outputs.append(output.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    return np.concatenate(outputs), np.concatenate(targets)


def main(args: argparse.Namespace) -> None:
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, _ = load_model(args.model_name, 1, pretrained=False, training=False)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Load the test data
    test_loader = prepare_test(args.data_dir, args.batch_size)

    # Predict and analyze results
    outputs, targets = predict(model, test_loader, device)
    predictions = (outputs > 0).astype(int)
    f1 = f1_score(targets, predictions, average='macro')

    print(f'F1 score: {f1:.3f}')
    print(
        f'Classification report:\n {classification_report(targets, predictions)}')
    print(f'Confusion matrix:\n {confusion_matrix(targets, predictions)}')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
