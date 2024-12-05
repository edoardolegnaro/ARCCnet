# %%
import os

import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import arccnet.models.cutouts.config as config
import arccnet.models.dataset_utils as ut_d
import arccnet.models.train_utils as ut_t
from arccnet.models import labels as lbs

# %%
# Generate run ID and weights directory
run_id, weights_dir = ut_t.generate_run_id(config)

# Create weights directory if it doesn't exist
os.makedirs(weights_dir, exist_ok=True)

# %%
# Data preparation
print("Making dataframe...")
df, AR_df = ut_d.make_dataframe(config.data_folder, config.dataset_folder, config.df_file_name)

# Undersample and filter the dataframe
df, df_du, zero_indexed_mapping = ut_d.undersample_group_filter(
    df, config.label_mapping, long_limit_deg=60, undersample=True, buffer_percentage=0.1
)

# Split data into folds for cross-validation
fold_df = ut_d.split_data(df_du, label_col="grouped_labels", group_col="number", random_state=42)
df = ut_d.assign_fold_sets(df, fold_df)
print("Dataframe preparation done.")

# %%
fold = 1
df_train = df[df[f"Fold {fold}"] == "train"]
df_val = df[df[f"Fold {fold}"] == "val"]
df_test = df[df[f"Fold {fold}"] == "test"]
num_classes = len(np.unique(df_train["encoded_labels"].values))

train_dataset = ut_t.FITSDataset(config.data_folder, config.dataset_folder, df_train, config.train_transforms)
val_dataset = ut_t.FITSDataset(config.data_folder, config.dataset_folder, df_val, config.val_transforms)
test_dataset = ut_t.FITSDataset(config.data_folder, config.dataset_folder, df_test, config.val_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
)

# %%
model = timm.create_model(config.model_name, pretrained=False, num_classes=num_classes, in_chans=1)
ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)
device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %%
num_params = ut_t.count_trainable_parameters(model, print_num=True)

# %%
class_weights = compute_class_weight(
    "balanced", classes=np.unique(df_train["encoded_labels"].values), y=df_train["encoded_labels"].values
)
class_weights

# %%
alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=alpha_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# Get the CUDA version
cuda_version = torch.version.cuda
if cuda_version and float(cuda_version) < 11.8:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = torch.amp.GradScaler("cuda")

# Training Loop
best_val_metric = 0.0
patience_counter = 0

for epoch in range(config.num_epochs):
    avg_train_loss, train_accuracy = ut_t.train_one_epoch(
        epoch, model, train_loader, criterion, optimizer, device, scaler
    )
    avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = ut_t.evaluate(model, val_loader, criterion, device)
    val_metric = val_accuracy

    # early stopping
    best_val_metric, patience_counter, stop_training = ut_t.check_early_stopping(
        val_metric, best_val_metric, patience_counter, model, weights_dir, config
    )
    if stop_training:
        break

    # Print epoch summary
    ut_t.print_epoch_summary(
        epoch, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, val_precision, val_recall, val_f1
    )

# %%
model

# %%
# Evaluate the best model on the test set
print("Testing...")
model = ut_t.load_model_test(weights_dir, model, device)
(
    avg_test_loss,
    test_accuracy,
    all_labels,
    all_preds,
    test_precision,
    test_recall,
    test_f1,
    cm_test,
    report_df,
) = ut_t.test_model(model, test_loader, device, criterion)

# %%
lbls = [value for value in config.label_mapping.values() if value is not None]
unique_lbls = []

for item in lbls:
    if item not in unique_lbls:
        unique_lbls.append(item)
unique_lbls

# Calculate the row percentages
row_sums = cm_test.sum(axis=1, keepdims=True)
cm_percentage = cm_test / row_sums * 100

# Create a custom annotation that includes both count and percentage
annotations = np.empty_like(cm_test).astype(str)

for i in range(cm_test.shape[0]):
    for j in range(cm_test.shape[1]):
        annotations[i, j] = f"{cm_test[i, j]}\n({cm_percentage[i, j]:.1f}%)"
greek_labels = lbs.convert_to_greek_label(unique_lbls)
# Plot the heatmap with the annotations, using cm_percentage for the color mapping
plt.figure(figsize=(5, 5))
sns.heatmap(
    cm_percentage,
    annot=annotations,
    fmt="",
    cmap="Blues",
    xticklabels=greek_labels,
    yticklabels=greek_labels,
    cbar=False,
)
plt.title(config.model_name)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# %%
misclassified_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]

# %%
idx = misclassified_indices[190]
img, true_label = test_dataset[idx]
pred_label = all_preds[idx]

plt.imshow(img.squeeze(0), cmap="gray", vmin=-1, vmax=1)
plt.colorbar()
plt.title(f"True: {unique_lbls[int(true_label)]} - Pred: {unique_lbls[int(pred_label)]}")
plt.show()
