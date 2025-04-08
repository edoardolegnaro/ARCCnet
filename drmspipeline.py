import os
import tarfile
from pathlib import PurePath
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from astropy.table import Table

from SDOprocessing import DrmsDownload, SDOproc, load_config, match_files, read_data

if __name__ == "__main__":
    config = load_config()
    keys = ["T_REC", "T_OBS", "QUALITY", "WAVELNTH", "*recnum*", "INSTRUME"]
    starts = read_data(
        "/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/hek_swpc_1996-01-01T00:00:00-2023-01-01T00:00:00_dev.parq", 10, 6
    )

    with ProcessPoolExecutor(6) as executor:
        for record in starts[0:3]:
            print(record)
            print("Downloading")
            noaa_ar, fl_class, start, end = record[0], record[1], record[2], record[3]
            aia_maps, hmi_maps = DrmsDownload.drms_pipeline(
                start, end, config["path"], keys, config["wavelengths"], config["sample"]
            )
            date = start.value.split("T")[0]
            print("Processing HMI")
            hmi_proc = tqdm(executor.map(SDOproc.hmi_l2, hmi_maps), total=len(hmi_maps), desc="HMI Processing")
            packed_files = match_files(aia_maps, hmi_maps)
            print("Processing AIA")
            aia_proc = tqdm(executor.map(SDOproc.aia_l2, packed_files), total=len(aia_maps), desc="AIA Processing")
            # For some reason, aia_proc becomes an empty list after the next line.
            home_table, aia_maps, aia_quality, hmi_maps, hmi_quality = SDOproc.l2_table_match(
                list(aia_proc), list(hmi_proc)
            )
            print("Batching and Tarring")

            # This can probably streamlined/functionalized to make the pipeline look better.
            batched_name = f'{config["path"]}/03_batched'
            os.makedirs(f"{batched_name}/records", exist_ok=True)
            os.makedirs(f"{batched_name}/tars", exist_ok=True)
            file_name = f"{fl_class}_{noaa_ar}_{date}"
            aia_away = ["AIA/" + PurePath(file).name for file in aia_maps]
            hmi_away = ["HMI/" + PurePath(file).name for file in hmi_maps]
            away_table = Table(
                {"AIA files": aia_away, "AIA quality": aia_quality, "HMI files": hmi_away, "HMI quality": hmi_quality}
            )
            away_table.write(f"{batched_name}/records/out_{file_name}.csv", overwrite=True)
            home_table.write(f"{batched_name}/records/{file_name}.csv", overwrite=True)

            with tarfile.open(f"{batched_name}/tars/{file_name}.tar", "w") as tar:
                for file in aia_maps:
                    name = PurePath(file).name
                    tar.add(file, arcname=f"AIA/{name}")
                for file in np.unique(hmi_maps):
                    name = PurePath(file).name
                    tar.add(file, arcname=f"HMI/{name}")
                tar.add(f"{batched_name}/records/out_{file_name}.csv", arcname=f"{file_name}.csv")


# 70 X class flares.
# Read the flare list.
# Not just HEK, look for save files for flares.
# Random sample (100 ish for M and below flares)
