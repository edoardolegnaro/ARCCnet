# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from astropy.io import fits

from arccnet.models import dataset_utils as ut_d
from arccnet.models import train_utils as ut_t
import arccnet.models.cutouts.mcintosh.dataset_utils as mci_ut_d
import arccnet.models.cutouts.mcintosh.train_utils as mci_ut_t
from arccnet.models.cutouts.mcintosh.models import HierarchicalResNet18
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)

# %%
gpu_index = 1
device = f'cuda:{gpu_index}'
EPOCHS = 5
batch_size = 32
num_workers = 40

data_folder=os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")
dataset_folder="arccnet-cutout-dataset-v20240715"

AR_df, encoders, mappings = mci_ut_d.process_ar_dataset(
    data_folder=data_folder,
    dataset_folder=dataset_folder,
    df_name="cutout-magnetic-catalog-v20240715.parq",
    plot_histograms=True 
)

# %% Sample Image
mci_ut_d.display_sample_image(data_folder, dataset_folder, AR_df, index=15)


# %%
train_df, val_df, test_df = mci_ut_d.split_dataset(
        df=AR_df,
        group_column="number",
        plot_histograms=True, 
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        verbose=True
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
ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.5)
model.to(device)

# Loss functions
criterion_Z = nn.CrossEntropyLoss()
criterion_P = nn.CrossEntropyLoss()
criterion_C = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
cuda_version = torch.version.cuda
if cuda_version and float(cuda_version) < 11.8:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = torch.amp.GradScaler("cuda")

# Training loop
for epoch in range(EPOCHS):
    train_loss, train_acc = mci_ut_t.train(model, device, train_loader, optimizer, criterion_Z, criterion_P, criterion_C, scaler)
    val_loss, val_acc = mci_ut_t.evaluate(model, device, val_loader, criterion_Z, criterion_P, criterion_C)
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )