# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: ARCAFF
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import time
import socket

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import arccnet.models.cutouts.mcintosh.dataset_utils as mci_ut_d
import arccnet.models.cutouts.mcintosh.models as models
import arccnet.models.cutouts.mcintosh.train_utils as mci_ut_t
from arccnet.models import train_utils as ut_t
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)

# %%
gpu_index = 0
device = f"cuda:{gpu_index}"
EPOCHS = 50
batch_size = 32
num_workers = os.cpu_count()
plot_histograms = False
patience = 2

# %%
t = time.localtime()
current_time = time.strftime("%Y%m%d-%H%M%S", t)
run_id = f"{current_time}_mcintosh_GPU{torch.cuda.get_device_name()}_{socket.gethostname()}"
weights_dir = os.path.join(os.path.dirname(os.path.abspath(ut_t.__file__)), "weights", f"{run_id}")
os.makedirs(weights_dir, exist_ok=True)

# %%
weights_dir

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")
dataset_folder = "arccnet-cutout-dataset-v20240715"

AR_df, encoders, mappings = mci_ut_d.process_ar_dataset(
    data_folder=data_folder,
    dataset_folder=dataset_folder,
    df_name="cutout-magnetic-catalog-v20240715.parq",
    plot_histograms=plot_histograms,
)

# %% Sample Image
mci_ut_d.display_sample_image(data_folder, dataset_folder, AR_df, index=15)


# %%
train_df, val_df, test_df = mci_ut_d.split_dataset(
    df=AR_df,
    group_column="number",
    plot_histograms=plot_histograms,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    verbose=True,
)

train_transforms = v2.Compose(
    [
        v2.RandomRotation(degrees=30),
    ]
)


train_dataset = mci_ut_d.SunspotDataset(data_folder, dataset_folder, train_df, transform=train_transforms)
val_dataset = mci_ut_d.SunspotDataset(data_folder, dataset_folder, val_df)
test_dataset = mci_ut_d.SunspotDataset(data_folder, dataset_folder, test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# %%
num_classes_Z = len(AR_df["Z_component_grouped"].unique())
num_classes_P = len(AR_df["p_component_grouped"].unique())
num_classes_C = len(AR_df["c_component_grouped"].unique())

model = models.TeacherForcingResNet18(
    num_classes_Z=num_classes_Z, num_classes_P=num_classes_P, num_classes_C=num_classes_C
)
ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)
model.to(device)

# Loss functions
z_weights = mci_ut_d.compute_weights(train_df["Z_grouped_encoded"], num_classes_Z)
p_weights = mci_ut_d.compute_weights(train_df["p_grouped_encoded"], num_classes_P)
c_weights = mci_ut_d.compute_weights(train_df["c_grouped_encoded"], num_classes_C)
criterion_Z = nn.CrossEntropyLoss(weight=z_weights.to(device))
criterion_P = nn.CrossEntropyLoss(weight=p_weights.to(device))
criterion_C = nn.CrossEntropyLoss(weight=c_weights.to(device))

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
cuda_version = torch.version.cuda
if cuda_version and float(cuda_version) < 11.8:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = torch.amp.GradScaler("cuda")

# %%
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
teacher_forcing_ratios = []

best_val_metric = 0.0
patience_counter = 0
epochs_used = 0
initial_teacher_forcing_ratio = 1.0
decay_rate = 0.95
min_teacher_forcing_ratio = 0.5
teacher_forcing_ratio = initial_teacher_forcing_ratio

for epoch in range(EPOCHS):
    epochs_used += 1
    train_loss, train_acc = mci_ut_t.train_teacher(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion_z=criterion_Z,
        criterion_p=criterion_P,
        criterion_c=criterion_C,
        teacher_forcing_ratio=teacher_forcing_ratio,  # Pass the current ratio
        scaler=scaler,
    )

    val_loss, val_acc = mci_ut_t.evaluate_teacher(
        model=model,
        device=device,
        loader=val_loader,
        criterion_z=criterion_Z,
        criterion_p=criterion_P,
        criterion_c=criterion_C,
    )

    # Append metrics to lists
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    teacher_forcing_ratios.append(teacher_forcing_ratio)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}")

    best_val_metric, patience_counter, stop_training = mci_ut_t.check_early_stopping(
        val_acc, best_val_metric, patience_counter, model, weights_dir, patience
    )
    if stop_training:
        break

    teacher_forcing_ratio = max(teacher_forcing_ratio * decay_rate, min_teacher_forcing_ratio)

# %%
# Plotting the metrics
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs_used + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs_used + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs_used + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs_used + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

# %%
model = ut_t.load_model_test(weights_dir, model, device)
(
    test_accuracy_z,
    test_accuracy_p,
    test_accuracy_c,
    f1_z,
    f1_p,
    f1_c,
    true_labels_z,
    pred_labels_z,
    true_labels_p,
    pred_labels_p,
    true_labels_c,
    pred_labels_c,
) = mci_ut_t.test_teacher(model, device, test_loader)

print("\n=== Test Results ===")
mci_ut_t.print_test_scores(test_accuracy_z, test_accuracy_p, test_accuracy_c, f1_z, f1_p, f1_c)

# %%
# Get label names from encoders
labels_z = encoders["Z_encoder"].classes_
labels_p = encoders["p_encoder"].classes_
labels_c = encoders["c_encoder"].classes_

# Compute confusion matrices using sklearn
cm_z = confusion_matrix(true_labels_z, pred_labels_z)
cm_p = confusion_matrix(true_labels_p, pred_labels_p)
cm_c = confusion_matrix(true_labels_c, pred_labels_c)

# %%
# Plot confusion matrices
ut_v.plot_confusion_matrix(cm_z, labels_z, "Confusion Matrix - Z Component", figsize=(6, 6))
ut_v.plot_confusion_matrix(cm_p, labels_p, "Confusion Matrix - p Component", figsize=(6, 6))
ut_v.plot_confusion_matrix(cm_c, labels_c, "Confusion Matrix - c Component", figsize=(6, 6))

# %%
