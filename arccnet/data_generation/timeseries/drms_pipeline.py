import csv
import fcntl
import json
import hashlib
import logging
from time import perf_counter, sleep
from pathlib import Path
from itertools import islice, repeat
from collections import deque
from multiprocessing import Semaphore
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import sunpy.map
from aiapy import calibrate
from tqdm import tqdm

import astropy.units as u
from astropy import log as astropy_log
from astropy.table import Table, unique, vstack
from astropy.time import Time

from arccnet import config
from arccnet.data_generation.timeseries.sdo_processing import (
    add_fnames,
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


def get_pointing_table_chunked(run_start, run_end, chunk_days=180, attempts=3):
    """Fetch the AIA pointing table in chunks to avoid large JSOC query failures."""
    chunk_start = Time(run_start)
    run_end = Time(run_end)
    tables = []

    while chunk_start < run_end:
        chunk_end = chunk_start + chunk_days * u.day
        if chunk_end > run_end:
            chunk_end = run_end

        table = None
        for attempt in range(attempts):
            try:
                table = calibrate.util.get_pointing_table(source="jsoc", time_range=[chunk_start, chunk_end])
                break
            except Exception as error:
                logging.warning(
                    "Pointing table fetch failed for %s to %s (attempt %s/%s): %s",
                    chunk_start.isot,
                    chunk_end.isot,
                    attempt + 1,
                    attempts,
                    error,
                )

        if table is None:
            raise RuntimeError(f"Could not fetch AIA pointing table from JSOC for {chunk_start.isot} to {chunk_end.isot}")

        if len(table) > 0:
            tables.append(table)

        chunk_start = chunk_end

    if not tables:
        raise RuntimeError("Could not fetch AIA pointing table from JSOC; no rows returned.")

    pointing_table = vstack(tables, metadata_conflicts="silent")
    if {"T_START", "T_STOP"}.issubset(pointing_table.colnames):
        pointing_table = unique(pointing_table, keys=["T_START", "T_STOP"])
    return pointing_table


def is_jsoc_pending_export_error(error):
    """Return True when JSOC rejects an export because this user has pending requests."""
    message = str(error)
    return "pending export request" in message or "pending export requests" in message or "[status=7]" in message


def append_status_log(status_path, meta, status, expected_frames="", observed_frames="", error=""):
    """Append a structured per-sample generation status row for availability analysis."""
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "logged_at_utc",
        "sample_id",
        "noaa_ar",
        "run_start_time",
        "target_time",
        "srs_date",
        "status",
        "expected_frames",
        "observed_frames",
        "error",
    ]
    row = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": meta["file_name"],
        "noaa_ar": meta["noaa_ar"],
        "run_start_time": getattr(meta["start"], "isot", str(meta["start"])),
        "target_time": getattr(meta["end"], "isot", str(meta["end"])),
        "srs_date": meta["date"],
        "status": status,
        "expected_frames": expected_frames,
        "observed_frames": observed_frames,
        "error": str(error).replace("\n", " ")[:2000],
    }
    with status_path.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, 2)
            write_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sample_complete(final_root, meta):
    """Return True when a sample's final marker file already exists."""
    return (Path(final_root) / "data" / meta["file_name"] / "after_flares.parquet").exists()


def time_window_key(meta):
    """Key samples by the SDO acquisition window they share."""
    return Time(meta["start"]).isot, Time(meta["end"]).isot


def group_metas_by_time(metas):
    """Group AR samples that can reuse the same downloaded SDO files."""
    groups = []
    groups_by_key = {}
    for meta in metas:
        key = time_window_key(meta)
        if key not in groups_by_key:
            groups_by_key[key] = {
                "key": key,
                "start": meta["start"],
                "end": meta["end"],
                "metas": [],
            }
            groups.append(groups_by_key[key])
        groups_by_key[key]["metas"].append(meta)
    return groups


def group_metas_individually(metas):
    """Build one group per sample, matching the pre-grouping pipeline behavior."""
    return [
        {
            "key": (meta["file_name"],),
            "start": meta["start"],
            "end": meta["end"],
            "metas": [meta],
        }
        for meta in metas
    ]


def group_label(group):
    """Human-readable label for logging a grouped SDO acquisition window."""
    start = getattr(group["start"], "isot", str(group["start"]))
    end = getattr(group["end"], "isot", str(group["end"]))
    return f"{start} to {end}"


def manifest_path(data_path, group, sample, wavelengths, hmi_keys, aia_keys):
    """Path for the cached raw-file manifest for one SDO acquisition window."""
    start = Time(group["start"]).isot
    end = Time(group["end"]).isot
    cache_key = "|".join([start, end, str(sample), str(wavelengths), str(hmi_keys), str(aia_keys)])
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
    start_tag = start.replace("-", "").replace(":", "").replace(".", "")
    end_tag = end.replace("-", "").replace(":", "").replace(".", "")
    return Path(data_path) / "02_intermediate" / "metadata" / "timeseries_manifests" / f"{start_tag}_{end_tag}_{digest}.json"


def map_paths(maps):
    """Extract raw FITS paths from maps loaded by the download pipeline."""
    paths = []
    for sdo_map in maps:
        raw_path = getattr(sdo_map, "_arcaff_raw_path", None) or sdo_map.meta.get("raw_path")
        if not raw_path:
            return None
        paths.append(str(Path(raw_path)))
    return paths


def load_maps_from_paths(paths):
    """Load SunPy maps from cached raw FITS paths and restore local filename metadata."""
    paths = [str(Path(path)) for path in paths]
    return add_fnames(sunpy.map.Map(paths), paths)


def load_group_manifest(manifest):
    """Return cached image/HMI maps when all manifest paths are still present."""
    manifest = Path(manifest)
    if not manifest.exists():
        return None
    try:
        with manifest.open() as handle:
            payload = json.load(handle)
        image_paths = payload["image_paths"]
        hmi_paths = payload["hmi_paths"]
        missing = [path for path in [*image_paths, *hmi_paths] if not Path(path).exists()]
        if missing:
            logging.info("Ignoring stale manifest %s; %s files are missing.", manifest, len(missing))
            return None
        return load_maps_from_paths(image_paths), load_maps_from_paths(hmi_paths)
    except Exception as error:
        logging.warning("Ignoring unreadable manifest %s: %s", manifest, error)
        return None


def write_group_manifest(manifest, group, aia_maps, hmi_maps, sample, wavelengths):
    """Cache the raw FITS paths used for a successfully downloaded time window."""
    manifest = Path(manifest)
    image_paths = map_paths(aia_maps)
    hmi_paths = map_paths(hmi_maps)
    if image_paths is None or hmi_paths is None:
        logging.warning("Could not write manifest for %s; raw paths are missing from maps.", group_label(group))
        return
    if not image_paths or not hmi_paths:
        logging.warning("Could not write manifest for %s; no raw paths were returned.", group_label(group))
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": Time(group["start"]).isot,
        "end": Time(group["end"]).isot,
        "sample": str(sample),
        "wavelengths": str(wavelengths),
        "image_paths": image_paths,
        "hmi_paths": hmi_paths,
    }
    tmp = manifest.with_name(f".{manifest.name}.{datetime.now(timezone.utc).timestamp()}.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(manifest)


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
    resume = config["timeseries"].getboolean("resume", fallback=True)
    make_animations = config["timeseries"].getboolean("make_animations", fallback=False)
    group_by_time = config["timeseries"].getboolean("group_by_time", fallback=True)
    use_time_window_manifest = config["timeseries"].getboolean("use_time_window_manifest", fallback=True)
    final_root = f"{data_path}/04_final"
    status_log = Path(data_path) / "logs" / "timeseries_generation_status.csv"
    # Frames per timestep: EUV wavelengths from config, plus the 1600/1700 UV
    # channels (hardcoded in aia_query_export) and the HMI continuum frame.
    frames_per_step = num_wavelengths + 2 + 1
    expected_frames = frames_per_step * timesteps

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
    pointing_chunk_days = int(config["timeseries"].get("pointing_chunk_days", 180))
    pointing_table = get_pointing_table_chunked(run_start, run_end, chunk_days=pointing_chunk_days)

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

    download_workers = int(config["drms"].get("download_workers", 3))
    jsoc_pending_retries = int(config["drms"].get("jsoc_pending_retries", 20))
    jsoc_pending_sleep = int(config["drms"].get("jsoc_pending_sleep", 120))
    patch_height = int(config["drms"]["patch_height"]) * u.pix
    patch_width = int(config["drms"]["patch_width"]) * u.pix

    def build_record_meta(rec_num):
        """Collect the per-record fields needed for download and processing."""
        record = starts[rec_num]
        noaa_ar, mag_class, mcintosh, end, start, date, center = record
        before_fls = before_fl_tables[rec_num]
        after_fls = after_fl_tables[rec_num]
        b_x, b_m, b_c = before_fls[1]["X"], before_fls[1]["M"], before_fls[1]["C"]
        a_x, a_m, a_c = after_fls[1]["X"], after_fls[1]["M"], after_fls[1]["C"]
        start_split = end.value.split("T")[0]
        file_name = f"{start_split}_{noaa_ar}_{mag_class}_{mcintosh}_Xb{b_x}_Mb{b_m}_Cb{b_c}_Xa{a_x}_Ma{a_m}_Ca{a_c}"
        return {
            "noaa_ar": noaa_ar,
            "start": start,
            "end": end,
            "date": date,
            "center": center,
            "before_fls": before_fls,
            "after_fls": after_fls,
            "file_name": file_name,
        }

    def download_group(group):
        """Runs in a download thread: JSOC queries, exports and L1 downloads for one time window."""
        group_manifest = manifest_path(
            data_path,
            group,
            config["drms"]["sample"],
            wavelengths,
            config["drms"]["hmi_keys"],
            config["drms"]["aia_keys"],
        )
        if use_time_window_manifest:
            cached_maps = load_group_manifest(group_manifest)
            if cached_maps is not None:
                logging.info("Using cached SDO manifest for %s", group_label(group))
                aia_maps, hmi_maps = cached_maps
                return group, aia_maps, hmi_maps, None

        for attempt in range(jsoc_pending_retries + 1):
            try:
                aia_maps, hmi_maps = drms_pipeline(
                    start_t=group["start"],
                    end_t=group["end"],
                    path=data_path,
                    hmi_keys=config["drms"]["hmi_keys"],
                    aia_keys=config["drms"]["aia_keys"],
                    wavelengths=wavelengths,
                    sample=config["drms"]["sample"],
                    drms_limit=drms_limit,
                )
                if use_time_window_manifest:
                    write_group_manifest(group_manifest, group, aia_maps, hmi_maps, config["drms"]["sample"], wavelengths)
                return group, aia_maps, hmi_maps, None
            except Exception as error:
                if is_jsoc_pending_export_error(error) and attempt < jsoc_pending_retries:
                    logging.warning(
                        "JSOC export queue is full for %s (%s samples); sleeping %ss before retry %s/%s.",
                        group_label(group),
                        len(group["metas"]),
                        jsoc_pending_sleep,
                        attempt + 1,
                        jsoc_pending_retries,
                    )
                    sleep(jsoc_pending_sleep)
                    continue
                return group, None, None, error

    metas = []
    for rec_num in range(len(starts)):
        meta = build_record_meta(rec_num)
        # after_flares.parquet is the last artifact l4_file_pack writes, so its
        # presence marks a fully generated sample.
        if resume and sample_complete(final_root, meta):
            logging.info(f"Skipping already generated sample {meta['file_name']}")
            continue
        metas.append(meta)
    groups = group_metas_by_time(metas) if group_by_time else group_metas_individually(metas)
    already_complete = len(starts) - len(metas)
    logging.info(f"{len(metas)} samples to generate ({already_complete} already complete).")
    if metas:
        logging.info(
            "%s SDO time-window groups to generate (group_by_time=%s, mean %.2f samples/group, max %s).",
            len(groups),
            group_by_time,
            len(metas) / len(groups),
            max(len(group["metas"]) for group in groups),
        )

    # Producer-consumer overlap: download threads prefetch the next few time
    # windows while the process pool works on the current group.  In grouped
    # mode, all AR samples with the same acquisition window reuse one JSOC
    # export/download and one set of full-disk L2 files.
    with ProcessPoolExecutor(cores) as executor, ThreadPoolExecutor(download_workers) as dl_pool:
        def process_sample(meta, hmi_proc, aia_proc):
            """Crop, reproject, and package one AR sample from shared L2 files."""
            if resume and sample_complete(final_root, meta):
                logging.info(f"Skipping already generated sample {meta['file_name']}")
                return

            noaa_ar = meta["noaa_ar"]
            date = meta["date"]
            center = meta["center"]
            before_fls = meta["before_fls"]
            after_fls = meta["after_fls"]
            file_name = meta["file_name"]

            logging.info(file_name)
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

            home_table, aia_patch_paths, aia_quality, aia_time, hmi_patch_paths, hmi_quality, hmi_time = table_match(
                list(aia_patch_paths), list(hmi_patch_paths)
            )

            batched_name = final_root
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

            vid_path = vid_match(home_table, file_name, batched_name) if make_animations else None
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
            append_status_log(
                status_log,
                meta,
                "completed",
                expected_frames=expected_frames,
                observed_frames=expected_frames,
            )

        group_iter = iter(groups)
        in_flight = deque(dl_pool.submit(download_group, group) for group in islice(group_iter, download_workers + 1))

        done_groups = 0
        done_samples = 0
        while in_flight:
            future = in_flight.popleft()
            group, aia_maps, hmi_maps, dl_error = future.result()
            next_group = next(group_iter, None)
            if next_group is not None:
                in_flight.append(dl_pool.submit(download_group, next_group))
            done_groups += 1
            group_size = len(group["metas"])
            print(
                f" group {done_groups}/{len(groups)} samples {done_samples + group_size}/{len(metas)} ".center(
                    70, "!"
                )
            )
            if dl_error is not None:
                logging.error("Download failed for %s: %s", group_label(group), dl_error)
                for meta in group["metas"]:
                    append_status_log(
                        status_log,
                        meta,
                        "download_failed",
                        expected_frames=expected_frames,
                        error=dl_error,
                    )
                    done_samples += 1
                continue

            if len(aia_maps) != expected_frames:
                logging.info(
                    f"Bad run for {group_label(group)} - expected {expected_frames} frames, "
                    f"got {len(aia_maps)}, skipping {group_size} samples."
                )
                for meta in group["metas"]:
                    append_status_log(
                        status_log,
                        meta,
                        "incomplete_frames",
                        expected_frames=expected_frames,
                        observed_frames=len(aia_maps),
                    )
                    done_samples += 1
                continue

            try:
                logging.info(f"Processing SDO time window {group_label(group)} ({group_size} samples)")
                hmi_proc = list(
                    tqdm(
                        executor.map(hmi_l2, hmi_maps),
                        total=len(hmi_maps),
                        desc="HMI prep",
                    )
                )

                packed_files = match_files(aia_maps, hmi_maps, pointing_table, correction_table)
                aia_proc = list(
                    tqdm(
                        executor.map(aia_l2, packed_files),
                        total=len(aia_maps),
                        desc="AIA prep",
                    )
                )
            except Exception as error:
                logging.error(f"Shared processing failed for {group_label(group)}", exc_info=True)
                for meta in group["metas"]:
                    append_status_log(status_log, meta, "processing_failed", expected_frames=expected_frames, error=error)
                    done_samples += 1
                continue
            finally:
                del aia_maps, hmi_maps

            for meta in group["metas"]:
                try:
                    process_sample(meta, hmi_proc, aia_proc)
                except Exception as error:
                    logging.error(error, exc_info=True)
                    append_status_log(status_log, meta, "processing_failed", expected_frames=expected_frames, error=error)
                finally:
                    done_samples += 1

    ee = perf_counter()
    print(f"Total time took {(ee - ss) / 60} minutes.")
