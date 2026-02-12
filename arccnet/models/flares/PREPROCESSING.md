# Flares Data Preprocessing System

## Overview

The flares preprocessing system reuses filtering logic from the cutouts pipeline to ensure consistent data quality across all model types. This document explains how to configure and use preprocessing when training flare models.

## Preprocessing Filters

The preprocessing system applies the following filters in order:

### 1. **Quality Flag Filtering** (`apply_quality_filter`)
Removes low-quality magnetograms based on HMI and MDI quality flags.

**Parameters:**
- **APPLY_QUALITY_FILTER**: `True` (recommended)
- Good quality flags:
  - HMI: `""`, `"0x00000000"`, `"0x00000400"`
  - MDI: `""`, `"00000000"`, `"00000200"`

**Effect:** Typically removes 5-15% of data depending on quality issues.

### 2. **Path Filtering** (`apply_path_filter`)
Removes rows where both HMI and MDI image paths are missing.

**Parameters:**
- **APPLY_PATH_FILTER**: `True` (recommended)

**Effect:** Removes records without any available magnetogram cutout.

### 3. **Longitude Filtering** (`apply_longitude_filter`)
Keeps only front-hemisphere observations (|longitude| ≤ limit).

**Parameters:**
- **APPLY_LONGITUDE_FILTER**: `True` (for binary classification)
- **MAX_LONGITUDE**: `65.0` (degrees)

**Rationale:** Front-facing active regions have more reliable magnetic field measurements.

**Effect:** Typically filters out 20-40% of samples depending on observational geometry.

**Note:** For multiclass, this is disabled by default because `filter_solar_limb()` already performs similar filtering.

### 4. **NaN Filtering** (`apply_nan_filter`) - Optional
Removes magnetograms with excessive NaN values.

**Parameters:**
- **APPLY_NAN_FILTER**: `False` (⚠️ Expensive - requires loading all FITS files)
- **NAN_THRESHOLD**: `0.05` (5% - maximum allowed fraction)

**Warning:** This filter is computationally expensive because it loads and inspects every FITS file. Enable only if data quality issues are observed.

## Configuration

### Binary Classification (`arccnet/models/flares/binary_classification/config.py`)

```python
# Enable cutouts-style preprocessing
APPLY_QUALITY_FILTER = True
APPLY_PATH_FILTER = True
APPLY_LONGITUDE_FILTER = True
MAX_LONGITUDE = 65.0
APPLY_NAN_FILTER = False  # Disable for speed
NAN_THRESHOLD = 0.05
```

### Multiclass (`arccnet/models/flares/multiclass/config.py`)

```python
# multiclass already has filter_solar_limb(), so disable duplicate filtering
APPLY_QUALITY_FILTER = True
APPLY_PATH_FILTER = True
APPLY_LONGITUDE_FILTER = False  # Already done by filter_solar_limb()
APPLY_NAN_FILTER = False  # Disable for speed
```

## Usage

### Binary Classification Training

No code changes needed. Simply set configuration parameters, then run:

```bash
python -m arccnet.models.flares.binary_classification.train
```

The preprocessing is automatically applied during data loading:

```python
# In train.py, the load_data() function now includes:
combined_df_preprocessed = preprocessing.preprocess_flare_data(
    combined_df,
    apply_quality_filter=config.APPLY_QUALITY_FILTER,
    apply_path_filter=config.APPLY_PATH_FILTER,
    apply_longitude_filter=config.APPLY_LONGITUDE_FILTER,
    apply_nan_filter=config.APPLY_NAN_FILTER,
    max_longitude=config.MAX_LONGITUDE,
    data_folder=config.DATA_FOLDER,
    dataset_folder=config.CUTOUT_DATASET_FOLDER,
)
```

### Multiclass Training

No code changes needed. The multiclass training now integrates preprocessing after the existing `filter_solar_limb()` call:

```bash
python -m arccnet.models.flares.multiclass.train
```

### Standalone Preprocessing

You can also use the preprocessing module independently:

```python
from arccnet.models.flares import preprocessing
import pandas as pd

# Load your data
df = pd.read_parquet("flares.parquet")

# Apply preprocessing
df_clean = preprocessing.preprocess_flare_data(
    df,
    apply_quality_filter=True,
    apply_path_filter=True,
    apply_longitude_filter=True,
    apply_nan_filter=False,
    max_longitude=65.0,
    data_folder="/ARCAFF/data",
    dataset_folder="arcnet-v20251017/04_final",
)

print(f"Retained {len(df_clean) / len(df) * 100:.1f}% of data after preprocessing")
```

## Logging Output

When preprocessing runs, it logs statistics for each filter:

```
============================================================
FLARE DATA PREPROCESSING PIPELINE
============================================================
Starting dataset size: 12,345 records
Applying quality flag filtering...
Quality filtering: 12,345 → 11,890 (455 removed)
Applying path filtering...
Path filtering: 11,890 → 11,878 (12 removed)
Applying longitude filtering (max |lon| = 65.0°)...
Longitude filtering: 11,878 → 9,234 (2,644 removed)
============================================================
Final dataset size: 9,234 records
Total removed: 3,111 records (25.2%)
Retention rate: 74.8%
============================================================
```

## Performance Impact

### Training Time
Preprocessing does NOT increase training time significantly because:
- Quality filtering is fast (flag comparisons)
- Path filtering is fast (string checks)
- Longitude filtering is fast (numeric comparisons)
- NaN filtering is **slow** but disabled by default

### Data Retention
Typical data retention rates:
- **Quality filtering alone**: 95-98%
- **+ Path filtering**: 94-98%
- **+ Longitude filtering**: 60-80% (depends on observation geometry)
- **Total (quality + path + longitude)**: 60-80% of original data

### Recommendation
Use the default configuration (quality + path + longitude filtering) for best results. Enable NaN filtering only if you observe specific artifacts in training due to NaN patterns.

## Why Reuse Cutouts Filtering?

The cutouts models (Hale/McIntosh) successfully use this filtering pipeline to train high-quality classifiers. By reusing the same filters for flares:

1. **Consistency**: Same quality standards across all models
2. **Proven approach**: Filters validated on cutout models
3. **Cross-dataset compatibility**: Ensures flare and cutout datasets are similarly cleaned
4. **Reduced data leakage**: Front-hemisphere filtering focuses on reliable observations

## Preprocessing vs. Augmentation

Note the difference:
- **Preprocessing** (this module): Data cleaning and filtering applied to raw data
- **Augmentation** (in config): Random transformations applied during training
- Order: Preprocessing → Splitting → Augmentation during training

Both are important for model quality.

## Troubleshooting

### "Too much data removed"
If excessive data is removed, check:
1. Are quality flags configured correctly?
2. Is MAX_LONGITUDE too restrictive? (try 70-80°)
3. Enable debug logging: `logging.getLogger().setLevel(logging.DEBUG)`

### "NaN filtering is too slow"
Solution: Set `APPLY_NAN_FILTER = False` (default). Use only if you observe NaN-related artifacts in training.

### "Data distribution changed unexpectedly"
Likely cause: Longitude filtering affected class distribution differently. Check logging output to see distribution before/after each filter.
