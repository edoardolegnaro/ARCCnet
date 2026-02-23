import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_sample_dirname(dirname):
    """
    Parse: YYYY-MM-DD_NOAA_MagClass_McIntosh_Xb#_Mb#_Cb#_Xa#_Ma#_Ca#
    Example: 2011-01-03_11142_Beta_Dso_Xb0_Mb0_Cb0_Xa0_Ma0_Ca1
    """
    parts = dirname.split("_")
    if len(parts) < 10:
        return None

    try:
        return {
            "date": parts[0],
            "noaa_ar": int(parts[1]),
            "hale_class": parts[2],
            "mcintosh": parts[3],
            "xb": int(parts[4].replace("Xb", "")),
            "mb": int(parts[5].replace("Mb", "")),
            "cb": int(parts[6].replace("Cb", "")),
            "xa": int(parts[7].replace("Xa", "")),
            "ma": int(parts[8].replace("Ma", "")),
            "ca": int(parts[9].replace("Ca", "")),
        }
    except Exception as e:
        print(f"Failed to parse {dirname}: {e}")
        return None


def build_sample_record(sample_dir):
    """
    Build record for one sample directory.
    Returns dict with paths organized as [T=6][C=10] grid.
    """
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name

    metadata = parse_sample_dirname(sample_id)
    if metadata is None:
        return None

    csv_path = sample_dir / f"{sample_id}.csv"
    if not csv_path.exists():
        print(f"Warning: CSV not found for {sample_id}")
        return None

    df_csv = pd.read_csv(csv_path)

    aia_wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
    hmi_wavelength = 6173
    channel_order = aia_wavelengths + [hmi_wavelength]

    paths_grid = []  # Will be [T][C] list of file paths
    timestamps = []

    wavelength_groups = {}
    for _, row in df_csv.iterrows():
        wl = int(row["AIA wavelength"])
        aia_file = row["AIA files"]
        hmi_file = row["HMI files"]

        if wl == hmi_wavelength:
            file_path = hmi_file
        else:
            file_path = aia_file

        if wl not in wavelength_groups:
            wavelength_groups[wl] = []
        wavelength_groups[wl].append(sample_dir / file_path)

    timestep_count = min(len(wavelength_groups.get(wl, [])) for wl in channel_order)
    if timestep_count == 0:
        print(f"Warning: No valid timesteps for {sample_id}")
        return None

    for t in range(timestep_count):
        timestep_paths = []
        for wl in channel_order:
            if wl in wavelength_groups and t < len(wavelength_groups[wl]):
                timestep_paths.append(str(wavelength_groups[wl][t]))
            else:
                timestep_paths.append(None)
        paths_grid.append(timestep_paths)

        if channel_order[0] in wavelength_groups and t < len(wavelength_groups[channel_order[0]]):
            first_file = wavelength_groups[channel_order[0]][t]
            match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}\d{2}\d{2}Z)", first_file.name)
            if match:
                timestamps.append(match.group(1))
            else:
                timestamps.append(f"t{t}")

    cplus = int(metadata["ca"] > 0 or metadata["ma"] > 0 or metadata["xa"] > 0)
    mplus = int(metadata["ma"] > 0 or metadata["xa"] > 0)
    xplus = int(metadata["xa"] > 0)

    return {
        "sample_id": sample_id,
        "sample_path": str(sample_dir),
        "noaa_ar": metadata["noaa_ar"],
        "date": metadata["date"],
        "hale_class": metadata["hale_class"],
        "mcintosh": metadata["mcintosh"],
        "num_timesteps": timestep_count,
        "timestamps": timestamps,
        "paths": paths_grid,
        "xb": metadata["xb"],
        "mb": metadata["mb"],
        "cb": metadata["cb"],
        "xa": metadata["xa"],
        "ma": metadata["ma"],
        "ca": metadata["ca"],
        "c_plus": cplus,
        "m_plus": mplus,
        "x_plus": xplus,
    }


def build_manifest(root_dir, output_path=None, max_samples=None):
    """
    Scan root_dir for sample folders and build manifest.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing sample folders
    output_path : str or Path, optional
        Where to save manifest parquet file
    max_samples : int, optional
        Limit number of samples (for testing)

    Returns
    -------
    pd.DataFrame
        Manifest dataframe
    """
    root_dir = Path(root_dir)
    sample_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    if max_samples:
        sample_dirs = sample_dirs[:max_samples]

    print(f"Building manifest from {len(sample_dirs)} samples...")

    records = []
    for sample_dir in tqdm(sample_dirs):
        record = build_sample_record(sample_dir)
        if record:
            records.append(record)

    df = pd.DataFrame(records)

    print(f"\nManifest built: {len(df)} valid samples")
    print(f"  NOAA ARs: {df['noaa_ar'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print("  Label distribution:")
    print(f"    C+: {df['c_plus'].sum()} ({df['c_plus'].mean() * 100:.1f}%)")
    print(f"    M+: {df['m_plus'].sum()} ({df['m_plus'].mean() * 100:.1f}%)")
    print(f"    X+: {df['x_plus'].sum()} ({df['x_plus'].mean() * 100:.1f}%)")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\nManifest saved to: {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="Dataset root directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    build_manifest(args.root_dir, args.output, args.max_samples)
