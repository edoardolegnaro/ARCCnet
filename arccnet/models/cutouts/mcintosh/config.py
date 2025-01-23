import os

### General ###
resnet_version = "resnet18"
gpu_index = 0
epochs = 5
batch_size = 32
num_workers = os.cpu_count()
learning_rate = 1e-4
random_state = 42
train_transforms = None
# train_transforms = v2.Compose(
#     [
#         v2.RandomVerticalFlip(),
#         v2.RandomHorizontalFlip(),
#         v2.RandomPerspective(distortion_scale=0.1, p=0.25),
#         v2.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
#     ]
# )

### Teacher Forcing ###
initial_teacher_forcing_ratio = 1.0
min_teacher_forcing_ratio = 0.0
teacher_forcing_decay = 0.95
teacher_forcing = True

### Dataset ###
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")
dataset_folder = "arccnet-cutout-dataset-v20240715"
df_name = "cutout-magnetic-catalog-v20240715.parq"
train_size = 0.7
val_size = 0.15
test_size = 0.15

### Logging ###
plot_histograms = False
patience = 5

### Comet ###
use_comet = True
project_name = "arcaff-mcintosh"
workspace = "arcaff"
