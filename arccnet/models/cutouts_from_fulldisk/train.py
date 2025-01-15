# +
import os

import pandas as pd
import torch
import torch.nn as nn
from comet_ml import Experiment
import numpy as np

import arccnet.models.cutouts.config as config
import arccnet.models.dataset_utils as ut_d
import arccnet.models.train_utils as ut_t
import arccnet.visualisation.utils as ut_v

from matplotlib import pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import timm

from arccnet.models import labels as lbs
import seaborn as sns

# +
# Generate run ID and weights directory
run_id, weights_dir = ut_t.generate_run_id(config)

# Create weights directory if it doesn't exist
os.makedirs(weights_dir, exist_ok=True)
# -

experiment = None
if experiment:
    # Initialize Comet experiment
    run_comet = Experiment(project_name=config.project_name, workspace="arcaff")
    run_comet.add_tags([config.model_name])
    run_comet.log_parameters(
        {
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "GPU": f"GPU{config.gpu_index}_{torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU",
            "num_epochs": config.num_epochs,
            "patience": config.patience,
        }
    )
    run_comet.log_code(config.__file__)
    run_comet.log_code(ut_t.__file__)
    run_comet.add_tags(config.other_tags)

# +
# Load Dataset
base_dir = '/ARCAFF/data/cutouts_from_fulldisk/'
df = pd.read_parquet(os.path.join(base_dir, 'dataframe.parquet'))

label_mapping = {
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": None,
    "Gamma-Delta": None,
}

df["grouped_labels"] = df["magnetic_class"].map(label_mapping)
df['Index'] = range(len(df))
df
# -

split_df = df.dropna(subset=["grouped_labels", "NOAA"])
fold_df = ut_d.split_data(split_df, label_col="grouped_labels", group_col="NOAA", random_state=42)
df_tr = ut_d.assign_fold_sets(df, fold_df)

# +
data = np.load(os.path.join(base_dir, 'processed_data.npz'))
images = data['images']
labels = np.array(df_tr['grouped_labels'])

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

df_tr['encoded_labels'] = encoded_labels

fold_n = 1
train_df = df_tr[df_tr[f'Fold {fold_n}']=='train']
val_df = df_tr[df_tr[f'Fold {fold_n}']=='val']
test_df = df_tr[df_tr[f'Fold {fold_n}']=='test']

print("Mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
# -

len(images)


class Cutouts(Dataset):

    def __init__(
        self, images, df, transform=None
    ):
        self.images = images
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_idx = row['Index']
        image = self.images[image_idx]
        label = row['encoded_labels']
        
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# +
num_classes = 3

train_dataset = Cutouts(images, train_df, config.train_transforms)
val_dataset = Cutouts(images, val_df, config.val_transforms)
test_dataset = Cutouts(images, test_df, config.val_transforms)
# -

train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
)

model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=num_classes, in_chans=1)
ut_t.replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)
device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
num_params = ut_t.count_trainable_parameters(model, print_num=True)

class_weights = compute_class_weight(
    "balanced", classes=np.unique(train_df["encoded_labels"].values), y=train_df["encoded_labels"].values
)

alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=alpha_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# Get the CUDA version
cuda_version = torch.version.cuda
if cuda_version and float(cuda_version) < 11.8:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = torch.amp.GradScaler("cuda")

# +
# Training Loop
best_val_metric = 0.0
patience_counter = 0

for epoch in range(config.num_epochs):
    avg_train_loss, train_accuracy = ut_t.train_one_epoch(
        epoch, model, train_loader, criterion, optimizer, device, scaler
    )
    avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = ut_t.evaluate(model, val_loader, criterion, device)
    val_metric = val_accuracy

    if experiment:
        experiment.log_metrics(
            {
                "avg_train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "avg_val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            },
            epoch=epoch,
        )

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

if experiment:
    log_model(experiment, model=model, model_name=config.model_name)
    lbls = [value for value in config.label_mapping.values() if value is not None]
    unique_lbls = []
    for item in lbls:
        if item not in unique_lbls:
            unique_lbls.append(item)
    experiment.log_metrics(
        {
            "avg_test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        }
    )
    experiment.log_confusion_matrix(
        matrix=cm_test,
        title="Confusion Matrix at best val epoch",
        file_name="test_confusion_matrix_best_epoch.json",
        labels=unique_lbls,
    )

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

    # Save the plot to a file
    heatmap_filename = os.path.join(script_dir, "temp", "confusion_matrix_heatmap.png")
    plt.savefig(heatmap_filename)
    plt.close()
    experiment.log_image(heatmap_filename, name="Confusion Matrix Heatmap")

    experiment.log_text(report_df.to_string(), metadata={"type": "Classification Report"})
    csv_file_path = os.path.join(weights_dir, "classification_report.csv")
    report_df.to_csv(csv_file_path, index=False)
    experiment.log_table("classification_report.csv", tabular_data=report_df)

# Log some misclassified examples
if experiment:
    misclassified_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]
    random.shuffle(misclassified_indices)  # Shuffle to select random samples
    for idx in misclassified_indices[:20]:  # Log 20 misclassified examples
        img, true_label = test_dataset[idx]
        pred_label = all_preds[idx]
        experiment.log_image(
            img,
            name=f"Misclassified_{idx}_true{unique_lbls[int(true_label)]}_pred{unique_lbls[int(pred_label)]}",
            metadata={
                "predicted_label": unique_lbls[int(true_label)].title(),
                "true_label": unique_lbls[int(true_label)].title(),
            },
        )

# -

encoded_labels

labels

images


