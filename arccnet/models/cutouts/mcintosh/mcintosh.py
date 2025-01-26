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
#     display_name: ARCAFF
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from comet_ml import Experiment  # isort: skip
from comet_ml.integration.pytorch import log_model  # isort: skip

import os
import time
import socket

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import arccnet.models.cutouts.mcintosh.dataset_utils as mci_ut_d
import arccnet.models.cutouts.mcintosh.train_utils as mci_ut_t
from arccnet.models import train_utils as ut_t
from arccnet.models.cutouts.mcintosh import config
from arccnet.models.cutouts.mcintosh.models import HierarchicalResNet
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)


# %%
device = f"cuda:{config.gpu_index}" if torch.cuda.is_available() else "cpu"

experiment = None
if config.use_comet:
    experiment = Experiment(project_name=config.project_name, workspace=config.workspace)
    experiment.log_parameters(
        {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "model": config.resnet_version,
        }
    )
    experiment.log_code(config.__file__)
    experiment.log_code(mci_ut_t.__file__)
    experiment.log_code(mci_ut_d.__file__)
    experiment.log_code(mci_ut_d.__file__)
    augmentation_tags = [type(transform).__name__ for transform in config.train_transforms.transforms]
    experiment.add_tags(augmentation_tags)

# %%
t = time.localtime()
current_time = time.strftime("%Y%m%d-%H%M%S", t)
run_id = f"{current_time}_mcintosh_GPU{torch.cuda.get_device_name()}_{socket.gethostname()}"
ut_t_file_path = os.path.abspath(ut_t.__file__)
weights_dir = os.path.join(os.path.dirname(ut_t_file_path), "weights", f"{run_id}")
os.makedirs(weights_dir, exist_ok=True)

# %%
AR_df, encoders, mappings = mci_ut_d.process_ar_dataset(
    data_folder=config.data_folder,
    dataset_folder=config.dataset_folder,
    df_name=config.df_name,
    plot_histograms=config.plot_histograms,
)

train_df, val_df, test_df = mci_ut_d.split_dataset(
    df=AR_df,
    group_column="number",
    plot_histograms=config.plot_histograms,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    verbose=True,
)

if experiment:
    experiment.log_dataset_hash(train_df)
    experiment.log_dataset_hash(val_df)
    experiment.log_dataset_hash(test_df)

train_dataset = mci_ut_d.SunspotDataset(
    config.data_folder, config.dataset_folder, train_df, transform=config.train_transforms
)
val_dataset = mci_ut_d.SunspotDataset(config.data_folder, config.dataset_folder, val_df)
test_dataset = mci_ut_d.SunspotDataset(config.data_folder, config.dataset_folder, test_df)

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
ut_v.make_classes_histogram(
    AR_df["Z_component_grouped"] + AR_df["p_component_grouped"] + AR_df["c_component_grouped"],
    figsz=(21, 8),
    y_off=25,
    text_fontsize=8,
)

# %%
valid_combined_classes = sorted(
    list(
        set(
            [
                (
                    AR_df["Z_component_grouped"].iloc[i],
                    AR_df["p_component_grouped"].iloc[i],
                    AR_df["c_component_grouped"].iloc[i],
                )
                for i in range(len(AR_df))
            ]
        )
    )
)

# %%
num_classes_Z = len(AR_df["Z_component_grouped"].unique())
num_classes_P = len(AR_df["p_component_grouped"].unique())
num_classes_C = len(AR_df["c_component_grouped"].unique())

# Initialize the new model
model = HierarchicalResNet(
    num_classes_Z=num_classes_Z,
    num_classes_P=num_classes_P,
    num_classes_C=num_classes_C,
    resnet_version=config.resnet_version,
).to(device)

ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)
num_params = ut_t.count_trainable_parameters(model, print_num=True)

if experiment:
    experiment.set_model_graph(str(model))
    experiment.log_metric("trainable_parameters", num_params)


# Loss functions
z_weights = mci_ut_d.compute_weights(train_df["Z_grouped_encoded"], num_classes_Z)
p_weights = mci_ut_d.compute_weights(train_df["p_grouped_encoded"], num_classes_P)
c_weights = mci_ut_d.compute_weights(train_df["c_grouped_encoded"], num_classes_C)

criterion_Z = nn.CrossEntropyLoss(weight=z_weights.to(device))
criterion_P = nn.CrossEntropyLoss(weight=p_weights.to(device))
criterion_C = nn.CrossEntropyLoss(weight=c_weights.to(device))

# Optimizer and scaler
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
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

if config.teacher_forcing:
    teacher_forcing_ratio = config.initial_teacher_forcing_ratio
else:
    teacher_forcing_ratio = None

for epoch in range(config.epochs):
    epochs_used += 1

    # Training
    train_loss, train_acc = mci_ut_t.train(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion_z=criterion_Z,
        criterion_p=criterion_P,
        criterion_c=criterion_C,
        teacher_forcing_ratio=teacher_forcing_ratio,
        scaler=scaler,
    )

    # Validation
    val_loss, val_acc = mci_ut_t.evaluate(
        model=model,
        device=device,
        loader=val_loader,
        criterion_z=criterion_Z,
        criterion_p=criterion_P,
        criterion_c=criterion_C,
    )

    if experiment:
        experiment.log_metrics(
            {
                "avg_train_loss": train_loss,
                "train_accuracy": train_acc,
                "avg_val_loss": val_loss,
                "val_accuracy": val_acc,
                "teacher_forcing_ratio": teacher_forcing_ratio if teacher_forcing_ratio else 0,
            },
            epoch=epoch,
        )

    if teacher_forcing_ratio:
        print(f"Epoch {epoch + 1}/{config.epochs}: Teacher Forcing Ratio = {teacher_forcing_ratio:.3f}")
    else:
        print(f"Epoch {epoch + 1}/{config.epochs}")
    # Log the metrics
    print(
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # Append the current teacher forcing ratio to the list
    teacher_forcing_ratios.append(teacher_forcing_ratio)

    # Early stopping logic
    best_val_metric, patience_counter, stop_training = mci_ut_t.check_early_stopping(
        val_acc, best_val_metric, patience_counter, model, weights_dir, config.patience
    )
    if stop_training:
        break

    # Apply decay to teacher forcing ratio at the end of the epoch
    if config.teacher_forcing and teacher_forcing_ratio is not None:
        teacher_forcing_ratio = max(
            config.min_teacher_forcing_ratio,
            teacher_forcing_ratio * config.teacher_forcing_decay,
        )


# %%
# Test the model
model = ut_t.load_model_test(weights_dir, model, device)

(
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
) = mci_ut_t.test(
    model=model,
    device=device,
    loader=test_loader,
)

if experiment:
    log_model(experiment, model=model, model_name=config.resnet_version)
    experiment.log_metrics(
        {
            "accuracy_z": accuracy_z,
            "accuracy_p": accuracy_p,
            "accuracy_c": accuracy_c,
            "f1_score_z": f1_score_z,
            "f1_score_p": f1_score_p,
            "f1_score_c": f1_score_c,
        }
    )

# %%
# Print test results
mci_ut_t.print_test_scores(accuracy_z, accuracy_p, accuracy_c, f1_score_z, f1_score_p, f1_score_c)

# %%
# Compute confusion matrices
labels_z = [str(label) for label in encoders["Z_encoder"].classes_]
labels_p = [str(label) for label in encoders["p_encoder"].classes_]
labels_c = [str(label) for label in encoders["c_encoder"].classes_]

figsize = (6, 6)
z_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_z.png")
p_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_p.png")
c_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_c.png")

cm_z = confusion_matrix(true_labels_z, pred_labels_z)
cm_p = confusion_matrix(true_labels_p, pred_labels_p)
cm_c = confusion_matrix(true_labels_c, pred_labels_c)

ut_v.plot_confusion_matrix(cm_z, labels_z, "Z Component", figsize=figsize, save_path=z_cm_path)
ut_v.plot_confusion_matrix(cm_p, labels_p, "p Component", figsize=figsize, save_path=p_cm_path)
ut_v.plot_confusion_matrix(cm_c, labels_c, "c Component", figsize=figsize, save_path=c_cm_path)
if experiment:
    experiment.log_image(z_cm_path, name="Z Component")
    experiment.log_image(p_cm_path, name="p Component")
    experiment.log_image(c_cm_path, name="c Component")

    experiment.log_confusion_matrix(
        matrix=np.array(cm_z),
        title="Confusion Matrix at best val epoch - Z Component",
        file_name="test_confusion_matrix_Z.json",
        labels=labels_z,
    )
    experiment.log_confusion_matrix(
        matrix=np.array(cm_p),
        title="Confusion Matrix at best val epoch - p Component",
        file_name="test_confusion_matrix_p.json",
        labels=labels_p,
    )
    experiment.log_confusion_matrix(
        matrix=np.array(cm_c),
        title="Confusion Matrix at best val epoch - c Component",
        file_name="test_confusion_matrix_c.json",
        labels=labels_c,
    )

# %%
true_grouped = [(true_labels_z[i], true_labels_p[i], true_labels_c[i]) for i in range(len(true_labels_z))]
pred_grouped = [(pred_labels_z[i], pred_labels_p[i], pred_labels_c[i]) for i in range(len(pred_labels_z))]

true_grouped_labels = [
    labels_z[true_grouped[i][0]] + labels_p[true_grouped[i][1]] + labels_c[true_grouped[i][2]]
    for i in range(len(true_grouped))
]
pred_grouped_labels = [
    labels_z[pred_grouped[i][0]] + labels_p[pred_grouped[i][1]] + labels_c[pred_grouped[i][2]]
    for i in range(len(pred_grouped))
]

# Encode valid true_grouped classes
encoder = LabelEncoder()
encoder.fit(true_grouped_labels)  # Fit only on valid true classes
class_mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
unknown_class_index = len(encoder.classes_)  # Index for "unknown"

encoded_true = encoder.transform(true_grouped_labels)  # Encode true labels

# Encode predictions, assigning "unknown" index if not found in true classes
encoded_pred = []
unknown_count = 0  # Counter for unknown predictions
for pred in pred_grouped_labels:
    if pred in class_mapping:
        encoded_pred.append(class_mapping[pred])
    else:
        encoded_pred.append(unknown_class_index)
        unknown_count += 1

print(f"Number of 'unknown' predictions: {unknown_count}")

# Confusion matrix
grouped_cm_path = os.path.join(os.path.dirname(ut_t_file_path), "temp", "confusion_matrix_grouped.png")
all_classes = list(encoder.classes_) + ["unknown"]  # Include "unknown" class
cm_gr = confusion_matrix(encoded_true, encoded_pred, labels=range(len(all_classes)))
ut_v.plot_confusion_matrix(
    cmc=cm_gr,
    labels=all_classes,
    title="Confusion Matrix for Grouped Classes",
    figsize=(12, 12),
    save_path=grouped_cm_path,
)

# Compute grouped accuracy and F1 score macro
grouped_accuracy = np.mean(np.array(encoded_true) == np.array(encoded_pred))
grouped_f1_score = f1_score(encoded_true, encoded_pred, average="macro")

print(f"Grouped Accuracy: {grouped_accuracy:.4f}")
print(f"Grouped F1 Score (Macro): {grouped_f1_score:.4f}")

if experiment:
    experiment.log_metrics(
        {
            "grouped_accuracy": grouped_accuracy,
            "grouped_f1_score_macro": grouped_f1_score,
        }
    )

if experiment:
    experiment.log_image(grouped_cm_path, name="Grouped Confusion Matrix")
    experiment.log_confusion_matrix(
        matrix=np.array(cm_gr),
        title="Confusion Matrix at best val epoch - Grouped Classes",
        file_name="grouped_confusion_matrix.json",
        labels=all_classes,
    )

# %%
idx = 3567

row = test_df.iloc[idx]

test_dataset = test_loader.dataset
region_input, label = test_dataset[idx]
region_input = region_input.unsqueeze(0)  # Add batch dimension (1, C, H, W)

model.eval()
with torch.no_grad():
    outputs_z, outputs_p, outputs_c = model(region_input.to(device))

_, pred_z = torch.max(outputs_z, 1)
_, pred_p = torch.max(outputs_p, 1)
_, pred_c = torch.max(outputs_c, 1)

final_class_z = encoders["Z_encoder"].inverse_transform(pred_z.cpu().numpy())
final_class_p = encoders["p_encoder"].inverse_transform(pred_p.cpu().numpy())
final_class_c = encoders["c_encoder"].inverse_transform(pred_c.cpu().numpy())
final_class = final_class_z[0] + final_class_p[0] + final_class_c[0]

print(f"Original class: {row['mcintosh_class']}")
print(f"Original Grouped class: {row['Z_component_grouped']}{row['p_component_grouped']}{row['c_component_grouped']}")
print(f"Predicted Final Class: {final_class}")

mci_ut_d.display_sample_image(config.data_folder, config.dataset_folder, test_df, idx)

# %%
