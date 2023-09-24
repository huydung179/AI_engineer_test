import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define transformations for training data
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define transformations for validation/test data
VALID_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def prepare_train_valid(data_dir: str, batch_size: int, seed: int = 42) -> (DataLoader, DataLoader):
    """
    Prepare DataLoaders for training and validation datasets.
    
    :param data_dir: Directory path containing the data.
    :param batch_size: Size of each batch.
    :param seed: Random seed for reproducibility.
    :return: Training and validation DataLoader objects.
    """
    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=TRAIN_TF)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))  # Seed set for reproducibility

    # Apply different transforms to training and validation sets (post splitting)
    train_dataset.dataset.transform = TRAIN_TF
    valid_dataset.dataset.transform = VALID_TF

    # Commented out: Weighted sampler for imbalanced datasets
    # Uncomment if needed and adjust accordingly.
    # train_targets = [train_dataset.dataset.targets[i]
    #                  for i in train_dataset.indices]
    # class_sample_count = np.unique(train_targets, return_counts=True)[1]
    # weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    # samples_weights = weights[train_targets]
    # sampler = WeightedRandomSampler(
    #     weights=samples_weights, num_samples=len(samples_weights))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def prepare_test(data_dir: str, batch_size: int) -> DataLoader:
    """
    Prepare DataLoader for the test dataset.
    
    :param data_dir: Directory path containing the test data.
    :param batch_size: Size of each batch.
    :return: Test DataLoader object.
    """
    test_dataset = datasets.ImageFolder(root=data_dir, transform=VALID_TF)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
