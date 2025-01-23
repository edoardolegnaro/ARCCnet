import os

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_z: nn.Module,
    criterion_p: nn.Module,
    criterion_c: nn.Module,
    teacher_forcing_ratio=None,
    scaler: torch.cuda.amp.GradScaler = None,
) -> tuple:
    """
    Trains the model for one epoch with optional Teacher Forcing.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the training on.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion_z (nn.Module): Loss function for Z component.
        criterion_p (nn.Module): Loss function for P component.
        criterion_c (nn.Module): Loss function for C component.
        teacher_forcing_ratio (float or None): Probability of using ground truth labels for Teacher Forcing.
                                               If None, Teacher Forcing is disabled.
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision. Defaults to None.

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for inputs, (labels_z, labels_p, labels_c) in tqdm(train_loader, desc="Training", unit="batch"):
        inputs, labels_z, labels_p, labels_c = (
            inputs.to(device),
            labels_z.to(device),
            labels_p.to(device),
            labels_c.to(device),
        )

        optimizer.zero_grad()

        use_teacher_forcing = teacher_forcing_ratio is not None

        if scaler:
            with torch.amp.autocast("cuda"):
                output_z, output_p, output_c = model(
                    inputs,
                    Z_true=labels_z if use_teacher_forcing else None,
                    P_true=labels_p if use_teacher_forcing else None,
                    teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
                )
                loss_z = criterion_z(output_z, labels_z)
                loss_p = criterion_p(output_p, labels_p)
                loss_c = criterion_c(output_c, labels_c)
                loss = loss_z + loss_p + loss_c
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output_z, output_p, output_c = model(
                inputs,
                Z_true=labels_z if use_teacher_forcing else None,
                P_true=labels_p if use_teacher_forcing else None,
                teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
            )
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

        total_correct += (predicted_z == labels_z).sum().item()
        total_correct += (predicted_p == labels_p).sum().item()
        total_correct += (predicted_c == labels_c).sum().item()

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
    teacher_forcing_ratio=None,
) -> tuple:
    """
    Evaluates the model on a validation or test set with optional Teacher Forcing.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the evaluation on.
        loader (DataLoader): DataLoader for validation/test data.
        criterion_z (nn.Module): Loss function for Z component.
        criterion_p (nn.Module): Loss function for P component.
        criterion_c (nn.Module): Loss function for C component.
        teacher_forcing_ratio (float or None): Probability of using ground truth labels for Teacher Forcing.
                                               If None, Teacher Forcing is disabled.

    Returns:
        tuple: Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, (labels_z, labels_p, labels_c) in tqdm(loader, desc="Evaluating", unit="batch"):
            # Move inputs and labels to device
            inputs, labels_z, labels_p, labels_c = (
                inputs.to(device),
                labels_z.to(device),
                labels_p.to(device),
                labels_c.to(device),
            )

            use_teacher_forcing = teacher_forcing_ratio is not None

            output_z, output_p, output_c = model(
                inputs,
                Z_true=labels_z if use_teacher_forcing else None,
                P_true=labels_p if use_teacher_forcing else None,
                teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
            )

            # Compute loss
            loss_z = criterion_z(output_z, labels_z)
            loss_p = criterion_p(output_p, labels_p)
            loss_c = criterion_c(output_c, labels_c)
            loss = loss_z + loss_p + loss_c
            total_loss += loss.item() * inputs.size(0)

            # Compute accuracy
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


def test(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    teacher_forcing_ratio=None,
) -> tuple:
    """
    Tests the model and computes accuracy and F1 scores for each component with optional Teacher Forcing.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the testing on.
        loader (DataLoader): DataLoader for test data.
        teacher_forcing_ratio (float or None): Probability of using ground truth labels for Teacher Forcing.
                                               If None, Teacher Forcing is disabled.

    Returns:
        tuple: Accuracy and F1 scores for Z, P, and C components respectively,
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
            inputs, labels_z, labels_p, labels_c = (
                inputs.to(device),
                labels_z.to(device),
                labels_p.to(device),
                labels_c.to(device),
            )

            use_teacher_forcing = teacher_forcing_ratio is not None

            output_z, output_p, output_c = model(
                inputs,
                Z_true=labels_z if use_teacher_forcing else None,
                P_true=labels_p if use_teacher_forcing else None,
                teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
            )

            # Compute predictions
            _, predicted_z = torch.max(output_z, 1)
            _, predicted_p = torch.max(output_p, 1)
            _, predicted_c = torch.max(output_c, 1)

            # Count correct predictions
            correct_z += (predicted_z == labels_z).sum().item()
            correct_p += (predicted_p == labels_p).sum().item()
            correct_c += (predicted_c == labels_c).sum().item()

            # Total samples for accuracy computation
            total_z += labels_z.size(0)
            total_p += labels_p.size(0)
            total_c += labels_c.size(0)

            # Collect true and predicted labels for F1 score computation
            true_labels_z.extend(labels_z.cpu().numpy())
            pred_labels_z.extend(predicted_z.cpu().numpy())

            true_labels_p.extend(labels_p.cpu().numpy())
            pred_labels_p.extend(predicted_p.cpu().numpy())

            true_labels_c.extend(labels_c.cpu().numpy())
            pred_labels_c.extend(predicted_c.cpu().numpy())

    # Compute accuracy
    accuracy_z = correct_z / total_z if total_z > 0 else 0
    accuracy_p = correct_p / total_p if total_p > 0 else 0
    accuracy_c = correct_c / total_c if total_c > 0 else 0

    # Compute F1 scores
    f1_score_z = f1_score(true_labels_z, pred_labels_z, average="weighted")
    f1_score_p = f1_score(true_labels_p, pred_labels_p, average="weighted")
    f1_score_c = f1_score(true_labels_c, pred_labels_c, average="weighted")

    return (
        accuracy_z,
        accuracy_p,
        accuracy_c,
        f1_score_z,
        f1_score_p,
        f1_score_c,
        true_labels_z,
        pred_labels_z,
        true_labels_p,
        pred_labels_p,
        true_labels_c,
        pred_labels_c,
    )


def check_early_stopping(val_metric, best_val_metric, patience_counter, model, weights_dir, patience):
    stop_training = False
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        patience_counter = 0
        model_save_path = os.path.join(weights_dir, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
    else:
        patience_counter += 1
        print(f"Early Stopping: {patience_counter}/{patience} without improvement.")
        if patience_counter >= patience:
            print("Stopping early due to no improvement in validation metric.")
            stop_training = True

    return best_val_metric, patience_counter, stop_training


def print_test_scores(accuracy_z, accuracy_p, accuracy_c, f1_z, f1_p, f1_c):
    """
    Prints test scores (accuracy and F1 scores) in a structured format using a DataFrame.

    Args:
        accuracy_z (float): Accuracy for Z component.
        accuracy_p (float): Accuracy for P component.
        accuracy_c (float): Accuracy for C component.
        f1_z (float): F1 score for Z component.
        f1_p (float): F1 score for P component.
        f1_c (float): F1 score for C component.
    """
    # Create a DataFrame
    data = {
        "Component": ["Z", "P", "C"],
        "Accuracy": [accuracy_z, accuracy_p, accuracy_c],
        "F1 Score": [f1_z, f1_p, f1_c],
    }
    df = pd.DataFrame(data)
    return df


def predict_region_class(model, region_input, encoders, device):
    model.eval()
    with torch.no_grad():
        outputs_z, outputs_p, outputs_c = model(region_input.to(device))

    # Get predicted indices
    _, pred_z = torch.max(outputs_z, 1)
    _, pred_p = torch.max(outputs_p, 1)
    _, pred_c = torch.max(outputs_c, 1)

    # Map indices to class labels
    final_class_z = encoders["Z_encoder"].inverse_transform(pred_z.cpu().numpy())
    final_class_p = encoders["p_encoder"].inverse_transform(pred_p.cpu().numpy())
    final_class_c = encoders["c_encoder"].inverse_transform(pred_c.cpu().numpy())

    # Combine predictions into a final class
    final_class = [(z, p, c) for z, p, c in zip(final_class_z, final_class_p, final_class_c)]

    return final_class


def calculate_f1_macro(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """
    Computes macro-averaged F1 scores for Z, P, and C components on the given DataLoader.

    Args:
        model (nn.Module): The trained neural network model.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: Macro-averaged F1 scores for Z, P, and C components (f1_z, f1_p, f1_c).
    """
    model.eval()
    true_labels_z = []
    pred_labels_z = []
    true_labels_p = []
    pred_labels_p = []
    true_labels_c = []
    pred_labels_c = []

    with torch.no_grad():
        for inputs, (labels_z, labels_p, labels_c) in loader:
            inputs = inputs.to(device)
            labels_z = labels_z.to(device)
            labels_p = labels_p.to(device)
            labels_c = labels_c.to(device)

            # Get predictions
            output_z, output_p, output_c = model(inputs)
            _, predicted_z = torch.max(output_z, 1)
            _, predicted_p = torch.max(output_p, 1)
            _, predicted_c = torch.max(output_c, 1)

            # Collect true and predicted labels
            true_labels_z.extend(labels_z.cpu().numpy())
            pred_labels_z.extend(predicted_z.cpu().numpy())
            true_labels_p.extend(labels_p.cpu().numpy())
            pred_labels_p.extend(predicted_p.cpu().numpy())
            true_labels_c.extend(labels_c.cpu().numpy())
            pred_labels_c.extend(predicted_c.cpu().numpy())

    # Compute macro-averaged F1 scores
    f1_z = f1_score(true_labels_z, pred_labels_z, average="macro")
    f1_p = f1_score(true_labels_p, pred_labels_p, average="macro")
    f1_c = f1_score(true_labels_c, pred_labels_c, average="macro")

    return f1_z, f1_p, f1_c
