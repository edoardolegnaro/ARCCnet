import logging
from time import perf_counter
from pathlib import Path
from itertools import repeat
from multiprocessing import Semaphore
from concurrent.futures import ProcessPoolExecutor

from aiapy import calibrate
from tqdm import tqdm

import astropy.units as u
from astropy import log as astropy_log
from astropy.table import Table
from astropy.time import Time

from arccnet import config
from arccnet.data_generation.timeseries.sdo_processing import (
    aia_l2,
    crop_map,
    drms_pipeline,
    hmi_l2,
    l4_file_pack,
    map_reproject,
    match_files,
    read_data,
    table_match,
    vid_match,
)

if __name__ == "__main__":
    __all__ = []

    ss = perf_counter()

    # Logging settings here.
    drms_log = logging.getLogger("drms")
    drms_log.setLevel("ERROR")
    reproj_log = logging.getLogger("reproject.common")
    reproj_log.setLevel("ERROR")
    astropy_log.setLevel("ERROR")
    data_path = config["paths"]["data_folder"]
    wavelengths = config["drms"]["wavelengths"]
    num_wavelengths = len([wvl for wvl in wavelengths.split(",") if wvl.strip()])
    cores = int(config["drms"]["cores"])
    drms_limit = Semaphore(int(config["drms"].get("max_drms_connections", 6)))
    duration = int(config["timeseries"].get("duration_hours", 6))
    timesteps = int(config["timeseries"].get("timesteps", 6))
    long_lim = int(config["timeseries"].get("long_lim_degrees", 65))
    year_start = int(config["timeseries"].get("year_start", 2010))
    year_end = int(config["timeseries"].get("year_end", 2022))
    keep_no_flare = config["timeseries"].getboolean("keep_no_flare", fallback=True)
    samples_per_year = int(config["timeseries"].get("samples_per_year", -1))
    sdo_start_date = config["timeseries"].get("sdo_start_date", "2010-05-13")
    catalog_end_date = config["timeseries"].get("catalog_end_date", "2023-01-01")
    hek_file = config["timeseries"].get("hek_file", "hek_swpc_1996-01-01T00:00:00-2023-01-01T00:00:00_dev.parq")
    srs_file = config["timeseries"].get("srs_file", "srs_processed_catalog.parq")
    # AIA wavelengths plus the HMI continuum frame per timestep.
    expected_frames = (num_wavelengths + 1) * timesteps

    starts, before_fl_tables, after_fl_tables = read_data(
        hek_path=Path(f"{data_path}/flare_files/{hek_file}"),
        srs_path=Path(f"{data_path}/flare_files/{srs_file}"),
        size=samples_per_year,
        duration=duration,
        long_lim=long_lim,
        # Use read_data_old with types=["F1", "F2", "N1", "N2"] to generate old flare target data.
        years=list(range(year_start, year_end + 1)),
        keep_no_flare=keep_no_flare,
        start_date=sdo_start_date,
        label_end_date=catalog_end_date,
    )
    logging.info(
        f"Generation run: {len(starts)} AR-day samples, {sdo_start_date} to {catalog_end_date}, "
        f"keep_no_flare={keep_no_flare}, degradation_correction="
        f"{config['timeseries'].getboolean('aia_degradation_correction', fallback=True)}"
    )

    # Fetch the AIA pointing table once for the full run span instead of once per record:
    # it is metadata-only, and per-record fetches add a JSOC round-trip and a failure mode
    # that previously skipped the whole record.
    run_start = Time(starts["run_start_time"]).min() - 6 * u.hour
    run_end = Time(starts["target_time"]).max()
    pointing_table = None
    for attempt in range(3):
        try:
            pointing_table = calibrate.util.get_pointing_table(source="jsoc", time_range=[run_start, run_end])
            break
        except Exception:
            logging.warning(f"Pointing table fetch failed (attempt {attempt + 1}/3)")
    if pointing_table is None:
        raise RuntimeError("Could not fetch AIA pointing table from JSOC; aborting run.")

    # Degradation correction table: fetched once per run and passed to every AIA frame.
    correction_table = None
    if config["timeseries"].getboolean("aia_degradation_correction", fallback=True):
        for attempt in range(3):
            try:
                correction_table = calibrate.util.get_correction_table(source="jsoc")
                break
            except Exception:
                logging.warning(f"Degradation correction table fetch failed (attempt {attempt + 1}/3)")
        if correction_table is None:
            raise RuntimeError(
                "Could not fetch AIA degradation correction table from JSOC; aborting run. "
                "Set aia_degradation_correction = False in [timeseries] to skip correction."
            )

    with ProcessPoolExecutor(cores) as executor:
        for rec_num in range(len(starts)):
            print(f" {rec_num}/{len(starts)} ".center(70, "!"))
            record = starts[rec_num]
            noaa_ar, mag_class, mcintosh, end, start, date, center = record
            before_fls = before_fl_tables[rec_num]
            after_fls = after_fl_tables[rec_num]
            b_x, b_m, b_c = before_fls[1]["X"], before_fls[1]["M"], before_fls[1]["C"]
            a_x, a_m, a_c = after_fls[1]["X"], after_fls[1]["M"], after_fls[1]["C"]
            start_split = end.value.split("T")[0]
            file_name = (
                f"{start_split}_{noaa_ar}_{mag_class}_{mcintosh}_Xb{b_x}_Mb{b_m}_Cb{b_c}_Xa{a_x}_Ma{a_m}_Ca{a_c}"
            )
            patch_height = int(config["drms"]["patch_height"]) * u.pix
            patch_width = int(config["drms"]["patch_width"]) * u.pix
            try:
                logging.info(file_name)
                aia_maps, hmi_maps = drms_pipeline(
                    start_t=start,
                    end_t=end,
                    path=config["paths"]["data_folder"],
                    hmi_keys=config["drms"]["hmi_keys"],
                    aia_keys=config["drms"]["aia_keys"],
                    wavelengths=config["drms"]["wavelengths"],
                    sample=config["drms"]["sample"],
                    drms_limit=drms_limit,
                )
                if len(aia_maps) != expected_frames:
                    logging.info(
                        f"Bad run - expected {expected_frames} frames, got {len(aia_maps)}, skipping."
                    )
                    continue

                hmi_proc = list(
                    tqdm(
                        executor.map(hmi_l2, hmi_maps),
                        total=len(hmi_maps),
                    )
                )

                packed_files = match_files(aia_maps, hmi_maps, pointing_table, correction_table)
                aia_proc = tqdm(executor.map(aia_l2, packed_files), total=len(aia_maps), desc="AIA prep")
                hmi_origin_patch = crop_map(hmi_proc[0], center, patch_height, patch_width, date)

                hmi_patch_paths = tqdm(
                    executor.map(map_reproject, repeat(hmi_origin_patch.wcs), hmi_proc, repeat(noaa_ar)),
                    total=len(hmi_proc),
                    desc="HMI reprojection",
                )
                aia_patch_paths = tqdm(
                    executor.map(map_reproject, repeat(hmi_origin_patch.wcs), aia_proc, repeat(noaa_ar)),
                    total=len(aia_proc),
                    desc="AIA reprojection",
                )

                # For some reason, aia_proc becomes an empty list after this function call.
                home_table, aia_patch_paths, aia_quality, aia_time, hmi_patch_paths, hmi_quality, hmi_time = (
                    table_match(list(aia_patch_paths), list(hmi_patch_paths))
                )

                batched_name = f"{config['paths']['data_folder']}/04_final"
                Path(f"{batched_name}/records").mkdir(parents=True, exist_ok=True)
                Path(f"{batched_name}/tars").mkdir(parents=True, exist_ok=True)
                hmi_away = ["HMI/" + Path(file).name for file in hmi_patch_paths]
                aia_away = ["AIA/" + Path(file).name for file in aia_patch_paths]
                aia_wvl = home_table["Wavelength"]
                away_table = Table(
                    {
                        "AIA wavelength": aia_wvl,
                        "AIA files": aia_away,
                        "AIA quality": aia_quality,
                        "HMI files": hmi_away,
                        "HMI quality": hmi_quality,
                    }
                )

                home_table.write(f"{batched_name}/records/{file_name}.csv", overwrite=True)

                vid_path = vid_match(home_table, file_name, batched_name)
                l4_file_pack(
                    aia_patch_paths,
                    hmi_patch_paths,
                    batched_name,
                    file_name,
                    away_table,
                    before_fls[0],
                    after_fls[0],
                    vid_path,
                )

            except Exception as error:
                logging.error(error, exc_info=True)

    ee = perf_counter()
    print(f"Total time took {(ee - ss) / 60} minutes.")
