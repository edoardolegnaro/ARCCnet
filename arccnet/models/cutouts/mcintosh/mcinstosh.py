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

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import arccnet.models.cutouts.mcintosh.dataset_utils as mci_ut_d
import arccnet.models.cutouts.mcintosh.train_utils as mci_ut_t
from arccnet.models import train_utils as ut_t
from arccnet.models.cutouts.mcintosh.models import HierarchicalResNet18
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)

# %%
gpu_index = 0
device = f"cuda:{gpu_index}"
EPOCHS = 10
batch_size = 32
num_workers = os.cpu_count()
plot_histograms = False

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

train_dataset = mci_ut_d.SunspotDataset(data_folder, dataset_folder, train_df)
val_dataset = mci_ut_d.SunspotDataset(data_folder, dataset_folder, val_df)
test_dataset = mci_ut_d.SunspotDataset(data_folder, dataset_folder, test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# %%
num_classes_Z = len(AR_df["Z_component_grouped"].unique())
num_classes_P = len(AR_df["p_component_grouped"].unique())
num_classes_C = len(AR_df["c_component_grouped"].unique())

model = HierarchicalResNet18(num_classes_Z=num_classes_Z, num_classes_P=num_classes_P, num_classes_C=num_classes_C)
ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.1)
model.to(device)

# Loss functions
z_weights = mci_ut_d.compute_weights(train_df["Z_grouped_encoded"], num_classes_Z)
p_weights = mci_ut_d.compute_weights(train_df["p_grouped_encoded"], num_classes_P)
c_weights = mci_ut_d.compute_weights(train_df["c_grouped_encoded"], num_classes_C)
criterion_Z = nn.CrossEntropyLoss(weight=z_weights.to(device))
criterion_P = nn.CrossEntropyLoss(weight=p_weights.to(device))
criterion_C = nn.CrossEntropyLoss(weight=c_weights.to(device))

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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

for epoch in range(EPOCHS):
    train_loss, train_acc = mci_ut_t.train(
        model, device, train_loader, optimizer, criterion_Z, criterion_P, criterion_C, scaler
    )
    val_loss, val_acc = mci_ut_t.evaluate(model, device, val_loader, criterion_Z, criterion_P, criterion_C)

    # Append metrics to lists (optional)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

# %%
# Plotting the metrics
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

# %%
(
    test_accuracy_z,
    test_accuracy_p,
    test_accuracy_c,
    true_labels_z,
    pred_labels_z,
    true_labels_p,
    pred_labels_p,
    true_labels_c,
    pred_labels_c,
) = mci_ut_t.test(model, device, test_loader)

print("\n=== Test Results ===")
print(f"Test Accuracy - Z Component: {test_accuracy_z:.4f}")
print(f"Test Accuracy - P Component: {test_accuracy_p:.4f}")
print(f"Test Accuracy - C Component: {test_accuracy_c:.4f}")

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
ut_v.plot_confusion_matrix(cm_p, labels_p, "Confusion Matrix - Z Component", figsize=(6, 6))
ut_v.plot_confusion_matrix(cm_c, labels_c, "Confusion Matrix - Z Component", figsize=(6, 6))

# %%
