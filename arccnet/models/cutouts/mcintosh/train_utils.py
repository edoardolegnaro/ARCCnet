import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_z: nn.Module,
    criterion_p: nn.Module,
    criterion_c: nn.Module,
    scaler: torch.cuda.amp.GradScaler = None,
) -> tuple:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the training on.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer for updating model weights.
        criterion_z (nn.Module): Loss function for Z component.
        criterion_p (nn.Module): Loss function for P component.
        criterion_c (nn.Module): Loss function for C component.
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision. Defaults to None.

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0  # If measuring sum across tasks

    for inputs, (labels_z, labels_p, labels_c) in tqdm(train_loader, desc="Training", unit="batch"):
        # Move data and labels to the device
        inputs = inputs.to(device)
        labels_z = labels_z.to(device)
        labels_p = labels_p.to(device)
        labels_c = labels_c.to(device)

        # Reset gradients
        optimizer.zero_grad()

        if scaler:
            # Mixed Precision
            with torch.amp.autocast("cuda"):
                output_z, output_p, output_c = model(inputs)
                loss_z = criterion_z(output_z, labels_z)
                loss_p = criterion_p(output_p, labels_p)
                loss_c = criterion_c(output_c, labels_c)
                loss = loss_z + loss_p + loss_c
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Full Precision
            output_z, output_p, output_c = model(inputs)
            loss_z = criterion_z(output_z, labels_z)
            loss_p = criterion_p(output_p, labels_p)
            loss_c = criterion_c(output_c, labels_c)
            loss = loss_z + loss_p + loss_c

            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Compute accuracies for each sub-task
        _, predicted_z = torch.max(output_z.data, 1)
        _, predicted_p = torch.max(output_p.data, 1)
        _, predicted_c = torch.max(output_c.data, 1)

        # For "micro" accuracy: count how many are correct across tasks
        total_correct += (predicted_z == labels_z).sum().item()
        total_correct += (predicted_p == labels_p).sum().item()
        total_correct += (predicted_c == labels_c).sum().item()

        # Each image has 3 label predictions (Z, P, C)
        total_predictions += inputs.size(0) * 3

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / total_predictions
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion_z: nn.Module,
    criterion_p: nn.Module,
    criterion_c: nn.Module,
) -> tuple:
    """
    Evaluates the model on a validation or test set.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the evaluation on.
        loader (DataLoader): DataLoader for validation/test data.
        criterion_z (nn.Module): Loss function for Z component.
        criterion_p (nn.Module): Loss function for P component.
        criterion_c (nn.Module): Loss function for C component.

    Returns:
        tuple: Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, (labels_z, labels_p, labels_c) in tqdm(loader, desc="Evaluating", unit="batch"):
            # Move to device
            inputs = inputs.to(device)
            labels_z = labels_z.to(device)
            labels_p = labels_p.to(device)
            labels_c = labels_c.to(device)

            output_z, output_p, output_c = model(inputs)

            loss_z = criterion_z(output_z, labels_z)
            loss_p = criterion_p(output_p, labels_p)
            loss_c = criterion_c(output_c, labels_c)
            loss = loss_z + loss_p + loss_c

            total_loss += loss.item() * inputs.size(0)

            # Accuracy
            _, predicted_z = torch.max(output_z.data, 1)
            _, predicted_p = torch.max(output_p.data, 1)
            _, predicted_c = torch.max(output_c.data, 1)

            total_correct += (predicted_z == labels_z).sum().item()
            total_correct += (predicted_p == labels_p).sum().item()
            total_correct += (predicted_c == labels_c).sum().item()

            total_predictions += inputs.size(0) * 3

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_predictions
    return avg_loss, accuracy


def test(model: nn.Module, device: torch.device, loader: DataLoader) -> tuple:
    """
    Tests the model and computes accuracy for each component.
    Also collects true and predicted labels for confusion matrices.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the testing on.
        loader (DataLoader): DataLoader for test data.

    Returns:
        tuple: Accuracy for Z, P, and C components respectively,
               and lists of true and predicted labels for each component.
    """
    model.eval()
    correct_z = 0
    correct_p = 0
    correct_c = 0

    total_z = 0
    total_p = 0
    total_c = 0

    true_labels_z = []
    pred_labels_z = []
    true_labels_p = []
    pred_labels_p = []
    true_labels_c = []
    pred_labels_c = []

    with torch.no_grad():
        for inputs, (labels_z, labels_p, labels_c) in tqdm(loader, desc="Testing", unit="batch"):
            inputs = inputs.to(device)
            labels_z = labels_z.to(device)
            labels_p = labels_p.to(device)
            labels_c = labels_c.to(device)

            output_z, output_p, output_c = model(inputs)

            _, predicted_z = torch.max(output_z, 1)
            _, predicted_p = torch.max(output_p, 1)
            _, predicted_c = torch.max(output_c, 1)

            correct_z += (predicted_z == labels_z).sum().item()
            correct_p += (predicted_p == labels_p).sum().item()
            correct_c += (predicted_c == labels_c).sum().item()

            total_z += labels_z.size(0)
            total_p += labels_p.size(0)
            total_c += labels_c.size(0)

            # Collect labels for confusion matrix
            true_labels_z.extend(labels_z.cpu().numpy())
            pred_labels_z.extend(predicted_z.cpu().numpy())

            true_labels_p.extend(labels_p.cpu().numpy())
            pred_labels_p.extend(predicted_p.cpu().numpy())

            true_labels_c.extend(labels_c.cpu().numpy())
            pred_labels_c.extend(predicted_c.cpu().numpy())

    accuracy_z = correct_z / total_z if total_z > 0 else 0
    accuracy_p = correct_p / total_p if total_p > 0 else 0
    accuracy_c = correct_c / total_c if total_c > 0 else 0

    return (
        accuracy_z,
        accuracy_p,
        accuracy_c,
        true_labels_z,
        pred_labels_z,
        true_labels_p,
        pred_labels_p,
        true_labels_c,
        pred_labels_c,
    )
