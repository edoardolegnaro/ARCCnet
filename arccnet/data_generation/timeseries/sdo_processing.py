import os
import re
import sys
import glob
import logging
import warnings
import itertools
import threading
import urllib.request
from random import sample
from pathlib import Path

import drms
import numpy as np
import sunpy.map
from aiapy.calibrate import correct_degradation, register, update_pointing
from aiapy.psf import deconvolve
from sunpy.coordinates import frames, propagate_with_solar_surface
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.physics.differential_rotation import solar_rotate_coordinate

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import CompImageHDU
from astropy.table import Table, join, vstack
from astropy.time import Time

from arccnet import config
from arccnet.data_generation.mag_processing import pixel_to_bboxcoords
from arccnet.data_generation.utils.utils import save_compressed_map
from arccnet.visualisation.data import mosaic_animate, mosaic_plot

rng = np.random.default_rng(42)


data_path = config["paths"]["data_folder"]
qry_log = "bad_queries"

warnings.simplefilter("ignore", RuntimeWarning)
reproj_log = logging.getLogger("reproject.common")
reproj_log.setLevel("ERROR")
data_path = config["paths"]["data_folder"]


def get_jsoc_email():
    r"""
    Return the JSOC-registered notification email from config or environment.

    Raises
    ------
    ValueError
        If no email is configured via the ``[drms] jsoc_email`` config entry
        or the ``JSOC_EMAIL`` environment variable.
    """
    email = config["drms"].get("jsoc_email", "") or os.environ.get("JSOC_EMAIL", "")
    email = email.strip()
    # Unexpanded interpolation placeholder means the env var was not set.
    if not email or email.startswith("$"):
        raise ValueError(
            "No JSOC export email configured. Set the JSOC_EMAIL environment variable "
            "or 'jsoc_email' in the [drms] section of your arccnetrc "
            "(register at http://jsoc.stanford.edu/ajax/register_email.html)."
        )
    return email


__all__ = [
    "read_data",
    "hmi_l2",
    "aia_l2",
    "change_time",
    "comp_list",
    "match_files",
    "drms_pipeline",
    "add_fnames",
    "table_match",
    "crop_map",
    "map_reproject",
    "l4_file_pack",
    "vid_match",
]


def bad_query(qstr, data_path, name):
    r"""
    Logs bad drms queries to document for future reference.

    Parameters
    ----------
        qstr: `str`
            Drms query string which returned an empty dataframe.
        data_path : `str`
            Path to arccnet level 04 data.
        name: `str`
            Log filename.
    """
    logging.warning(f"Bad Query Detected - {qstr}")
    log_dir = Path(data_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    f_name = log_dir / f"{name}.txt"
    entry = f"{qstr}\n"
    with open(f_name, "a+") as file:
        file.seek(0)
        if entry not in file.readlines():
            file.write(entry)


def rand_select(table, years: list, size):
    r"""
    Randomly selects targets from provided table and list of data subsets.

    Parameters
    ----------
        table : `Astropy.Table`
            Provided full table of records.
        size : `int`
            The size of the returned subsets.
        types: `list`
            List of datatypes to include.

    Returns
    -------
        comb_sample : `Astropy.Table`
            Returned random subset containing size records from each data subset.
    """

    selection = []
    for year in years:
        table["year"] = table["target_time"].ymdhms.year
        subtable = table[table["year"] == year]
        if size == -1:
            selection.append(subtable)
        else:
            selection.append(subtable[rng.choice(len(subtable), size=int(size), replace=False)])
    comb_sample = vstack(selection)

    return comb_sample


# Old rand select, use for generating v2 data.
def rand_select_old(table, size, types: list):
    r"""
    Randomly selects targets from provided table and list of data subsets.

    Parameters
    ----------
        table : `Astropy.Table`
            Provided full table of records.
        size : `int`
            The size of the returned subsets.
        types: `list`
            List of datatypes to include.

    Returns
    -------
        comb_sample : `Astropy.Table`
            Returned random subset containing size records from each data subset.
    """
    selection = []
    for type in types:
        subtable = table[table["category"] == type]
        selection.append(subtable[sample(range(len(subtable)), k=int(size))])
    comb_sample = vstack(selection)
    return comb_sample


def read_data(
    hek_path: str,
    srs_path: str,
    size: int,
    duration: int,
    long_lim: int,
    years: list,
    keep_no_flare: bool = True,
    start_date: str = None,
    label_end_date: str = None,
):
    r"""
    Read and process data from a parquet file containing HEK catalogue information regarding flaring events.

    Parameters
    ----------
        hek_path : `str`
            The path to the parquet file containing hek flare information.
        srs_path : `str`
            The path to the parquet file containing parsed noaa srs active region information.
        size : `int`
            The size of each subsample to be generated.
        duration : `int`
            The duration of the data sample in hours.
        flares : `str`
            Determines if runs provided 'positive' (flares), 'negative' (no flares), or 'both' (50/50 split of both)
        long_lim : `int`
            The longitudinal limit of Active Regions which are accepted for target runs.
        types : `list`
            Types of data to include in final subsection, corresponds to flares vs non flares (F v N) and incidental and clear runs (1 v 2)
        p_window : `int`
            Prediction window for training set for prediction. 6 = six hour window containing flare, etc.
        years : `list[int]`
            List of years
        keep_no_flare : `bool`
            Keep AR-day samples with no C+ flare in the following 24 hours (negatives). Negatives
            must be sampled alongside positives (same years, same procedure) to avoid the class
            label being confounded with acquisition epoch.
        start_date : `str`, optional
            Earliest SRS target date to include (e.g. first date with SDO science data).
            Defaults to January 1st of the earliest requested year.
        label_end_date : `str`, optional
            End of flare-catalogue coverage. Samples whose 24 h label window extends past
            this date are dropped, since their labels would be silently truncated.

    Returns
    -------
        `Astropy.Table`
            A table of tuples containing the following columns for each flare:
            - NOAA Active Region Number
            - GOES Flare class (C,M,X classes or N for none)
            - Start time (Duration + 1 hours before event in FITS format)
            - End time (1 hour before flaring event start time)
            - The date of the run, used for reprojection
            - The coordinate of the noaa active region
            - The classes of X, M, and C flares within the observed period
    """
    table = Table.read(hek_path)
    srs = Table.read(srs_path)

    catalog_start = f"{min(years)}-01-01"
    if start_date is None:
        start_date = catalog_start
    noaa_num_df = table[table["noaa_number"] > 0]
    flares = noaa_num_df[noaa_num_df["event_type"] == "FL"]
    # Keep flares from before the first sample date so the 6 h "before" windows stay complete.
    flares = flares[flares["frm_daterun"] > catalog_start]
    flares = flares[
        [flare.startswith("C") or flare.startswith("M") or flare.startswith("X") for flare in flares["goes_class"]]
    ]

    srs = srs[srs["number"] > 0]
    srs = srs[srs["target_time"] > start_date]
    srs = srs[abs(srs["longitude"].value) <= long_lim]
    srs = srs[~srs["filtered"]]

    srs["srs_date"] = srs["target_time"].value
    srs["srs_date"] = [date.split("T")[0] for date in srs["srs_date"]]
    srs["target_time"] = srs["target_time"] + 30 * u.min
    if label_end_date is not None:
        # Drop samples whose 24 h label window extends past catalogue coverage.
        srs = srs[(Time(srs["target_time"]) + 24 * u.hour) <= Time(label_end_date)]
    srs["run_start_time"] = [(Time(time) - duration * u.hour) for time in srs["target_time"]]
    srs["c_coord"] = [
        SkyCoord(lon * u.deg, lat * u.deg, obstime=t_time, observer="earth", frame=frames.HeliographicStonyhurst)
        for lat, lon, t_time in zip(srs["latitude"], srs["longitude"], srs["target_time"])
    ]
    logging.info("Parsing Flares")

    logging.info("Parsing Active Regions")
    srs_exp = srs["number", "magnetic_class", "mcintosh_class", "target_time", "run_start_time", "srs_date", "c_coord"]

    final = rand_select(srs_exp, years, size)
    flares_before = [
        flare_log(Time(row["run_start_time"]), Time(row["target_time"]), row["number"], flares) for row in final
    ]
    flares_after = [
        flare_log(Time(row["target_time"]), Time(row["target_time"] + 24 * u.hour), row["number"], flares)
        for row in final
    ]
    subset = final["number", "magnetic_class", "mcintosh_class", "target_time", "run_start_time", "srs_date", "c_coord"]

    if keep_no_flare:
        return subset, flares_before, flares_after

    # Positives only: keep samples with at least one C+ flare in the next 24 hours.
    # WARNING: generating positives and negatives in separate runs confounds the class
    # label with acquisition epoch (e.g. AIA degradation state); prefer keep_no_flare=True.
    only_flares = [i for i, fa in enumerate(flares_after) if len(fa[0]) > 0]

    return subset[only_flares], [flares_before[i] for i in only_flares], [flares_after[i] for i in only_flares]


# Old data parser - use if you want to generate v2 data.
def read_data_old(hek_path: str, srs_path: str, size: int, duration: int, long_lim: int, types: list):
    r"""
    Read and process data from a parquet file containing HEK catalogue information regarding flaring events.

    Parameters
    ----------
        hek_path : `str`
            The path to the parquet file containing hek flare information.
        srs_path : `str`
            The path to the parquet file containing parsed noaa srs active region information.
        size : `int`
            The size of each subsample to be generated.
        duration : `int`
            The duration of the data sample in hours.
        flares : `str`
            Determines if runs provided 'positive' (flares), 'negative' (no flares), or 'both' (50/50 split of both)
        long_lim : `str`
            The longitudinal limit of Active Regions which are accepted for target runs.
        types : `list`
            Types of data to include in final subsection, corresponds to flares vs non flares (F v N) and incidental and clear runs (1 v 2)
        p_window : `int`
            Prediction window for training set for prediction. 6 = six hour window containing flare, etc.


    Returns
    -------
        `Astropy.Table`
            A table of tuples containing the following columns for each flare:
            - NOAA Active Region Number
            - GOES Flare class (C,M,X classes or N for none)
            - Start time (Duration + 1 hours before event in FITS format)
            - End time (1 hour before flaring event start time)
            - The date of the run, used for reprojection
            - The coordinate of the noaa active region
            - The classes of X, M, and C flares within the observed period
    """
    table = Table.read(hek_path)
    srs = Table.read(srs_path)

    noaa_num_df = table[table["noaa_number"] > 0]
    flares = noaa_num_df[noaa_num_df["event_type"] == "FL"]
    flares = flares[flares["frm_daterun"] > "2011-01-01"]
    flares = flares[
        [flare.startswith("C") or flare.startswith("M") or flare.startswith("X") for flare in flares["goes_class"]]
    ]

    srs = srs[srs["number"] > 0]
    srs = srs[srs["target_time"] > "2011-01-01"]
    srs = srs[abs(srs["longitude"].value) <= long_lim]
    srs = srs[~srs["filtered"]]
    srs["srs_date"] = srs["target_time"].value
    srs["srs_date"] = [date.split("T")[0] for date in srs["srs_date"]]
    srs["srs_end_time"] = [(Time(time) + duration * u.hour) for time in srs["target_time"]]
    logging.info("Parsing Flares")
    flares["start_time"].format = "fits"
    flares["run_start_time"] = [time - (duration + 1) * u.hour for time in flares["start_time"]]
    flares["tb_date"] = flares["run_start_time"].value
    flares["tb_date"] = [date.split("T")[0] for date in flares["tb_date"]]
    flares["run_end_time"] = flares["run_start_time"] + duration * u.hour

    flare_splits = [
        flare_check(Time(row["run_start_time"]), Time(row["run_end_time"]), row["noaa_number"], flares)
        for row in flares
    ]
    flares["category"] = [f"F{flare[0]}" for flare in flare_splits]
    flares["fl_count"] = [flare[1] for flare in flare_splits]
    flares = join(flares, srs, keys_left="noaa_number", keys_right="number")
    flares = flares[flares["tb_date"] == flares["srs_date"]]
    flares["c_coord"] = [
        SkyCoord(lon * u.deg, lat * u.deg, obstime=t_time, observer="earth", frame=frames.HeliographicStonyhurst)
        for lat, lon, t_time in zip(flares["latitude"], flares["longitude"], flares["target_time"])
    ]
    srs["c_coord"] = [
        SkyCoord(lon * u.deg, lat * u.deg, obstime=t_time, observer="earth", frame=frames.HeliographicStonyhurst)
        for lat, lon, t_time in zip(srs["latitude"], srs["longitude"], srs["target_time"])
    ]
    logging.info("Parsing Active Regions")
    ar_cat, fl_cat = [], []
    for ar in srs:
        erl_time = Time(ar["target_time"]) - (duration + 1) * u.hour
        n_splits = flare_check(erl_time, Time(ar["target_time"]) - 1 * u.hour, ar["number"], flares)
        ar_cat.append(f"N{n_splits[0]}")
        fl_cat.append(n_splits[1])
    srs["category"] = ar_cat
    srs["n_fl_count"] = fl_cat
    srs["ar"] = "N"
    srs_exp = srs["number", "ar", "target_time", "srs_end_time", "srs_date", "c_coord", "category", "n_fl_count"]
    flares_exp = flares[
        "noaa_number", "goes_class", "run_start_time", "run_end_time", "tb_date", "c_coord", "category", "fl_count"
    ]

    srs_exp.rename_columns(
        names=("number", "ar", "target_time", "srs_end_time", "srs_date", "c_coord", "category", "n_fl_count"),
        new_names=(
            "noaa_number",
            "goes_class",
            "run_start_time",
            "run_end_time",
            "tb_date",
            "c_coord",
            "category",
            "fl_count",
        ),
    )
    combined = vstack([flares_exp, srs_exp])

    combined["X_fl"] = [flare["X"] for flare in combined["fl_count"]]
    combined["M_fl"] = [flare["M"] for flare in combined["fl_count"]]
    combined["C_fl"] = [flare["C"] for flare in combined["fl_count"]]

    final = rand_select(combined, size, types)

    flares_before = [
        flare_log(Time(row["run_start_time"]), Time(row["run_end_time"]), row["noaa_number"], flares) for row in final
    ]
    flares_after = [
        flare_log(
            Time(row["run_end_time"] + 1 * u.hour), Time(row["run_end_time"] + 25 * u.hour), row["noaa_number"], flares
        )
        for row in final
    ]
    subset = final[
        "noaa_number",
        "goes_class",
        "run_start_time",
        "run_end_time",
        "tb_date",
        "c_coord",
        "category",
        "X_fl",
        "M_fl",
        "C_fl",
    ]

    return subset, flares_before, flares_after


def change_time(time: str, shift: int):
    r"""
    Change the timestamp by a given time shift.

    Parameters
    ----------
        time : `str`
            A timestamp in FITS format.
        shift : `int`
            The time shift in seconds.

    Returns
    -------
        `str`
            The updated timestamp in FITS format.
    """
    time_d = Time(time, format="fits") + shift * (u.second)
    return time_d.to_value("fits")


def comp_list(file: str, file_list: list):
    r"""
    Check if a file is present in a list of files.

    Parameters
    ----------
        file : `str`
            The file to check.
        file_list :
            `list` The list of files.

    Returns
    -------
        `list`
            A list of booleans for each element in list. True if the file is present, False otherwise.
    """
    return any(file in name for name in file_list)


def match_files(aia_maps, hmi_maps, table, correction_table=None):
    r"""
    Matches AIA maps with corresponding HMI maps based on the closest time difference.

    Parameters
    ----------
        aia_maps : `list`
            List of AIA maps.
        hmi_maps : `list`
            List of HMI maps.
        table : `JSOC Response`
            AIA pointing table as provided by JSOC.
        correction_table : `astropy.table.Table`, optional
            AIA degradation correction table (aiapy get_correction_table).

    Returns
    -------
        packed_files : `list`
            A list of [aia_map, matched_hmi_map, pointing_table, correction_table] entries.
    """
    packed_files = []
    for aia_map in aia_maps:
        t_d = [abs(aia_map.date - hmi_map.date).to_value(u.s) for hmi_map in hmi_maps]
        hmi_match = hmi_maps[t_d.index(min(t_d))]
        packed_files.append([aia_map, hmi_match, table, correction_table])
    return packed_files


def add_fnames(maps, paths):
    r"""
    Adds file names to fits map metadata.

    Parameters
    ----------
        maps : `list`
            List of fits maps.
        paths : `list`
            List of file paths.

    Returns
    -------
        named_map : `list`
            List of fits maps with file names added to metadata.
    """
    named_maps = []
    for map, fname in zip(maps, paths):
        map.meta["fname"] = Path(fname).name
        map._arcaff_raw_path = str(Path(fname))
        named_maps.append(map)
    return named_maps


def drms_pipeline(
    start_t,
    end_t,
    path: str,
    hmi_keys: list,
    aia_keys: list,
    wavelengths: str = "171, 193, 304, 211, 335, 94, 131, 1600, 4500, 1700",
    sample: int = 60,
    drms_limit=None,
):
    r"""
    Performs pipeline to download and process AIA and HMI data.

    Parameters
    ----------
        starts : `list`
            List of start and end times for the data retrieval.
        path : `str`
            Path to save the downloaded data.
        keys : `list`
            List of keys for the data query.
        wavelengths : `str`
            String of wavelengths in list formatting for the AIA data to be provided to drms quotestrings (default all AIA wvl).
        sample : `int`
            Sample rate for the data cadence (default 1/hr).
    Returns
    -------
        aia_maps, hmi_maps : `tuple`
            A tuple containing the AIA maps and HMI maps.
    """
    with drms_limit:
        hmi_query, hmi_export, ic_query, ic_export = hmi_query_export(start_t, end_t, hmi_keys, sample)
        aia_query, aia_export = aia_query_export(hmi_query, aia_keys, wavelengths)

    hmi_dls, hmi_exs = l1_file_save(hmi_export, hmi_query, path)
    cnt_dls, cnt_exs = l1_file_save(ic_export, ic_query, path)
    aia_dls, aia_exs = l1_file_save(aia_export, aia_query, path)
    img_exs = list(itertools.chain(aia_exs, cnt_exs))

    hmi_maps = sunpy.map.Map(hmi_exs)
    hmi_maps = add_fnames(hmi_maps, hmi_exs)
    img_maps = sunpy.map.Map(img_exs)
    img_maps = add_fnames(img_maps, img_exs)
    return img_maps, hmi_maps


def hmi_query_export(time_1, time_2, keys: list, sample: int):
    r"""
    Query and export HMI magnetogram data from the JSOC database.

    Parameters
    ----------
        time_1 : `str`
            The start timestamp in FITS format.
        time_2 : `str`
            The end timestamp in FITS format.
        keys : `list`
            A list of keys to query.
        sample : `int`
            The sample rate in minutes.

    Returns
    -------
        hmi_query_full, hmi_result, ic_query_full, ic_export : `tuple`
            A tuple containing the query results of the hmi mag and ic_no_limbdark (pandas df) and the export data response (drms export object).
    """
    client = drms.Client()
    retries = int(config["drms"].get("quality_retries", 3))
    duration = round((time_2 - time_1).to_value(u.hour))
    qstr_m_hmi = f"hmi.M_720s[{time_1.value}/{duration}h@{sample}m]{{magnetogram}}"
    hmi_query = client.query(ds=qstr_m_hmi, key=keys)

    good_result = hmi_query[hmi_query.QUALITY == 0]
    good_num = good_result["*recnum*"].values
    bad_result = hmi_query[hmi_query.QUALITY != 0]

    qstrs_m_hmi = [f"hmi.M_720s[{time}]{{magnetogram}}" for time in bad_result["T_REC"]]
    hmi_values = [hmi_rec_find(qstr, keys, retries, 720) for qstr in qstrs_m_hmi]
    patched_num = [num for num in hmi_values if num is not None]

    joined_num = [*good_num, *patched_num]
    joined_num = [str(num) for num in joined_num]
    hmi_num_str = str(joined_num).strip("[]")

    hmi_qstr = f"hmi.M_720s[! recnum in ({hmi_num_str}) !]{{magnetogram}}"
    hmi_query_full = client.query(ds=hmi_qstr, key=keys)
    hmi_result = client.export(hmi_qstr, method="url", protocol="fits", email=get_jsoc_email())
    hmi_result.wait()
    ic_query_full, ic_result = hmi_continuum_export(hmi_query_full, keys)
    return hmi_query_full, hmi_result, ic_query_full, ic_result


def hmi_continuum_export(hmi_query, keys):
    r"""
    Query and export HMI continuum data from the JSOC database.

    Parameters
    ----------
        hmi_query : `drms query`
            HMI query result containing target times.
        keys : `list`
            A list of keys to query.

    Returns
    -------
        hmi_query_full, hmi_result, ic_query_full, ic_export : `tuple`
            A tuple containing the query results of the hmi mag and ic_no_limbdark (pandas df) and the export data response (drms export object).
    """
    client = drms.Client()
    retries = int(config["drms"].get("quality_retries", 3))
    qstrs_ic = [f"hmi.Ic_noLimbDark_720s[{time}]{{continuum}}" for time in hmi_query["T_REC"]]
    # Retry step must match the 720 s series cadence; smaller shifts re-query the same record.
    ic_value = [hmi_rec_find(qstr, keys, retries, 720, cont=True) for qstr in qstrs_ic]
    joined_num = [str(num) for num in ic_value if num is not None]
    ic_num_str = str(joined_num).strip("[]")
    ic_comb_qstr = f"hmi.Ic_noLimbDark_720s[! recnum in ({ic_num_str}) !]{{continuum}}"

    ic_query_full = client.query(ds=ic_comb_qstr, key=keys)

    ic_result = client.export(ic_comb_qstr, method="url", protocol="fits", email=get_jsoc_email())
    ic_result.wait()
    return ic_query_full, ic_result


def aia_query_export(hmi_query, keys, wavelength):
    r"""
    Query and export AIA data from the JSOC database.

    Parameters
    ----------
        hmi_query : `drms query`
            The HMI query result containing target times.
        keys : `list`
            List of keys to query.
        wavelength : `list`
            List of AIA wavelengths.

    Returns
    -------
        aia_query_full, aia_result (tuple): A tuple containing the query result and the export data response.
    """
    client = drms.Client()
    retries = int(config["drms"].get("quality_retries", 3))
    qstrs_euv = [f"aia.lev1_euv_12s[{time}][{wavelength}]{{image}}" for time in hmi_query["T_REC"]]
    qstrs_uv = [f"aia.lev1_uv_24s[{time}]{[1600, 1700]}{{image}}" for time in hmi_query["T_REC"]]
    euv_value = [aia_rec_find(qstr, keys, retries, 12) for qstr in qstrs_euv]
    uv_value = [aia_rec_find(qstr, keys, retries, 24) for qstr in qstrs_uv]
    unpacked_aia = list(itertools.chain(euv_value, uv_value))
    unpacked_aia = [fsn_set for fsn_set in unpacked_aia if fsn_set is not None]
    unpacked_aia = list(itertools.chain.from_iterable(unpacked_aia))
    joined_num = [str(num) for num in unpacked_aia]
    aia_num_str = str(joined_num).strip("[]")
    aia_comb_qstr = f"aia.lev1[! FSN in ({aia_num_str}) !]{{image_lev1}}"

    aia_query_full = client.query(ds=aia_comb_qstr, key=keys)

    aia_result = client.export(aia_comb_qstr, method="url", protocol="fits", email=get_jsoc_email())
    aia_result.wait()
    return aia_query_full, aia_result


def hmi_rec_find(qstr, keys, retries, sample, cont=False):
    r"""
    Find the HMI record number for a given query string.

    Parameters
    ----------
        qstr : `str`
            A query string.
        keys : `list`
            List of keys to query.
        retries : `int`
            Number of later records to try when the target record has bad quality.
        sample: `int`
            Time shift between retries in seconds (series cadence).
        cont : `bool`
            indicates whether continuum is needed, searches for magnetogram if false.
    Returns
    -------
        `int` or `None`
            The HMI record number, or `None` if no good-quality record was found.
    """
    seg = "{magnetogram}"
    series = "hmi.M_720s"
    if cont:
        seg = "{continuum}"
        series = "hmi.Ic_noLimbDark_720s"
    client = drms.Client()
    qry = client.query(ds=qstr, key=keys)
    if qry.empty:
        logging.warning("Bad Query - HMI")
        bad_query(qstr, data_path, qry_log)
        time = sunpy.time.parse_time(re.search(r"\[(.*?)\]", qstr).group(1)).fits
    else:
        if qry["QUALITY"].values[0] == 0:
            return qry["*recnum*"].values[0]
        time = sunpy.time.parse_time(qry["T_REC"].values[0]).fits
    for _ in range(retries):
        time = change_time(time, sample)
        retry_qstr = f"{series}[{time}]" + seg
        qry = client.query(ds=retry_qstr, key=keys)
        if qry.empty:
            logging.warning("Bad Query - HMI")
            bad_query(retry_qstr, data_path, qry_log)
            continue
        if qry["QUALITY"].values[0] == 0:
            return qry["*recnum*"].values[0]
    logging.warning(f"No good-quality HMI record within {retries} retries of {qstr}; dropping frame.")
    return None


def aia_rec_find(qstr, keys, retries, time_add):
    r"""
    Find good-quality AIA FSNs for a given query string.

    Quality is checked per returned record (the EUV queries return one record
    per wavelength), and bad-quality records are retried individually at later
    times.

    Parameters
    ----------
        qstr : `str`
            A query string.
        keys : `list`
            List of keys to query.
        retries : `int`
            Number of later records to try when a record has bad quality.
        time_add : `int`
            Time shift between retries in seconds (series cadence).

    Returns
    -------
        `list` or `None`
            Good-quality AIA FSNs, or `None` if the query returned nothing.
    """
    client = drms.Client()
    qry = client.query(ds=qstr, key=keys)
    qstr_head = qstr.split("[")[0]
    if qry.empty:
        logging.warning("Bad Query - AIA")
        bad_query(qstr, data_path, qry_log)
        return None

    fsns = []
    for _, row in qry.iterrows():
        wvl = row["WAVELNTH"]
        # 4500 has no reliable quality flagging; accept as-is.
        if row["QUALITY"] == 0 or str(wvl) == "4500":
            fsns.append(row["FSN"])
            continue
        time = row["T_REC"][0:-1]
        for _ in range(retries):
            time = change_time(time, time_add)
            retry_qstr = f"{qstr_head}[{time}][{wvl}]{{image}}"
            retry_qry = client.query(ds=retry_qstr, key=keys)
            if retry_qry.empty:
                logging.warning("Bad Query - AIA")
                bad_query(retry_qstr, data_path, qry_log)
                continue
            if retry_qry["QUALITY"].values[0] == 0:
                fsns.append(retry_qry["FSN"].values[0])
                break
        else:
            logging.warning(
                f"No good-quality AIA {wvl} record within {retries} retries of {row['T_REC']}; dropping frame."
            )
    return fsns if fsns else None


def _atomic_download(url, dest, retries=3):
    r"""
    Download a URL to a destination path atomically (temp file + rename).

    Safe under concurrent download threads: records from the same day share
    full-disk L1 files, and two threads may try to fetch the same target. The
    rename is atomic, so a partially written file can never appear at the final
    path (which also makes interrupted runs safe to resume).

    Parameters
    ----------
        url : `str`
            Source URL.
        dest : `str`
            Destination file path.
        retries : `int`
            Number of download attempts before giving up.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    last_error = None
    for _ in range(retries):
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            return dest
        except Exception as error:
            last_error = error
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def l1_file_save(export, query, path):
    r"""
    Save the exported data as level 1 FITS files.

    Parameters
    ----------
        export : `drms export`
            A drms data export.
        query : `drms query`
            A drms query result.
        path : `str`
            A base path to save the files.

    Returns
    -------
        export (drms export), total_files (list) : `tuple`
            A tuple containing the updated export data and the list of saved file paths.
    """
    instr = query["INSTRUME"][0][0:3]
    path_prefix = []
    export.urls.drop_duplicates(ignore_index=True, inplace=True)
    query.drop_duplicates(ignore_index=True, inplace=True)
    if len(export.urls) != len(query):
        raise ValueError(
            f"DRMS export URL count ({len(export.urls)}) does not match query record count "
            f"({len(query)}); cannot safely assign download paths."
        )
    for time, wvl in zip(query["T_REC"], query["WAVELNTH"]):
        time = sunpy.time.parse_time(time).to_value("ymdhms")
        year, month, day = time["year"], time["month"], time["day"]
        newdir = f"{path}/01_raw/{year}/{month}/{day}/SDO/{instr}/"
        Path(newdir).mkdir(parents=True, exist_ok=True)
        path_prefix.append(f"{newdir}{int(wvl)}.")

    existing_files = [glob.glob(f"{dirs}*.fits") for dirs in np.unique(path_prefix)]
    existing_files = list(itertools.chain.from_iterable(existing_files))
    matching_files = [comp_list(file, existing_files) for file in export.urls["filename"]]
    missing_files = [not value for value in matching_files]
    export.urls["filename"] = path_prefix + export.urls["filename"]
    missing_rows = export.urls[missing_files]
    if len(missing_rows.index) > 0:
        total_files = list(export.urls[matching_files].index) + list(missing_rows.index)
        total_files = export.urls["filename"][total_files]
        for url, fname in zip(missing_rows["url"], missing_rows["filename"]):
            _atomic_download(url, fname)
    else:
        total_files = export.urls["filename"][matching_files]
    return export, total_files.to_list()


def aia_process(
    aia_map, table, deconv: bool = False, degcorr: bool = False, correction_table=None, exnorm: bool = True
):
    r"""
    Process an AIA map to level 1.5.

    Parameters
    ----------
        aia_map : `sunpy.map.Map`
            The AIA map to process.
        table : `JSOCResponse`
            A pointing table as provided by get_pointing_table
        deconv : `bool`
            Whether to deconvolve the PSF.
        degcorr : `bool`
            Whether to correct for instrument degradation. Requires ``correction_table``.
        correction_table : `astropy.table.Table`, optional
            Degradation correction table as provided by
            `aiapy.calibrate.util.get_correction_table`; fetch once per run and reuse.
        exnorm : `bool`
            Whether to normalize exposure.

    Returns
    -------
        aia_map : `sunpy.map.Map`
            Processed AIA map. The ``DEGCORR`` FITS keyword records whether
            degradation correction was applied.
    """
    if deconv:
        aia_map = deconvolve(aia_map)
    aia_map = update_pointing(aia_map, pointing_table=table)
    aia_map = register(aia_map)
    degcorr_applied = False
    if degcorr:
        if correction_table is None:
            raise ValueError("degcorr=True requires a correction_table (aiapy get_correction_table).")
        try:
            aia_map = correct_degradation(aia_map, correction_table=correction_table)
            degcorr_applied = True
        except Exception as error:
            logging.warning(
                f"Degradation correction failed for {aia_map.meta.get('wavelnth')} A at "
                f"{aia_map.meta.get('date-obs')}: {error}; frame left uncorrected."
            )
    aia_map.meta["degcorr"] = degcorr_applied
    if exnorm:
        # Keep float32: int truncation destroys sub-1 DN/s values (94/131 A quiet regions).
        aiad = (aia_map.data / aia_map.exposure_time).astype(np.float32)
        aia_map = sunpy.map.Map(aiad, aia_map.meta)
    return aia_map


def aia_reproject(aia_map, hmi_map):
    r"""
    Reproject an AIA map to the same coordinate system as an HMI map.

    Parameters
    ----------
        aia_map : `sunpy.map.Map`
            The AIA map to reproject.
        hmi_map : `sunpy.map.Map`
            The HMI map to use as the target coordinate system.

    Returns
    -------
        rpr_aia_map : `sunpy.map.Map`
            Reprojected AIA map.
    """
    rpr_aia_map = aia_map.reproject_to(hmi_map.wcs)
    rpr_aia_map.meta["wavelnth"] = aia_map.meta["wavelnth"]
    rpr_aia_map.meta["waveunit"] = aia_map.meta["waveunit"]
    rpr_aia_map.meta["quality"] = aia_map.meta["quality"]
    rpr_aia_map.meta["t_rec"] = aia_map.meta["t_rec"]
    rpr_aia_map.meta["instrume"] = aia_map.meta["instrume"]
    rpr_aia_map.meta["fname"] = aia_map.meta["fname"]
    rpr_aia_map.meta["degcorr"] = aia_map.meta.get("degcorr", False)
    rpr_aia_map.nickname = aia_map.nickname

    return rpr_aia_map


def hmi_mask(hmi_map):
    r"""
    Mask pixels outside of Rsun_obs in an HMI map.

    Parameters
    ----------
        hmimap : `sunpy.map.Map`
            The HMI map.

    Returns
    -------
        hmimap : `sunpy.map.Map`
            The masked HMI map.
    """
    hpc_coords = all_coordinates_from_map(hmi_map)
    mask = ~coordinate_is_on_solar_disk(hpc_coords)
    hmi_data = hmi_map.data
    hmi_data[mask] = np.nan
    hmi_map = sunpy.map.Map(hmi_data, hmi_map.meta)
    return hmi_map


def hmi_l2(hmi_map):
    r"""
    Processes the HMI map to "level 2" by applying a mask and saving it to the output directory.

    Parameters
    ----------
        hmi_map : `sunpy.map.Map`
            HMI map to be processed.
        overwrite : `bool`
            Flag which determines if l2 files are reproduced and overwritten.

    Returns
    -------
    proc_path : `str`
        Path to the processed HMI map.
    """
    path = config["paths"]["data_folder"]
    time = hmi_map.date.to_value("ymdhms")
    year, month, day = time[0], time[1], time[2]
    map_path = f"{path}/02_intermediate/{year}/{month}/{day}/SDO/{hmi_map.nickname}"
    proc_path = f"{map_path}/02_{hmi_map.meta['fname']}"

    if not os.path.exists(proc_path):
        hmi_map = hmi_mask(hmi_map)
        proc_path = l2_file_save(hmi_map, path)
    # This updates process status for tqdm more effectively.
    sys.stdout.flush()

    return proc_path


def aia_l2(packed_maps):
    r"""
    Processes the AIA map to "level 2" by leveling, rescaling, trimming, and reprojecting it to match the nearest HMI map.

    Parameters
    ----------
        packed_maps : `list`
            [aia_map, matched_hmi_map, pointing_table, correction_table] as packed
            by `match_files`.

    Returns
    -------
        proc_path : `str`
            Path to the processed AIA map.
    """
    path = config["paths"]["data_folder"]
    sdo_map, hmi_match, table, correction_table = packed_maps
    if sdo_map.nickname == "HMI":
        proc_path = hmi_l2(sdo_map)
    else:
        time = sdo_map.date.to_value("ymdhms")
        year, month, day = time[0], time[1], time[2]
        map_path = f"{path}/02_intermediate/{year}/{month}/{day}/SDO/{sdo_map.nickname}"
        proc_path = f"{map_path}/02_{sdo_map.meta['fname']}"
        degcorr = config["timeseries"].getboolean("aia_degradation_correction", fallback=True)
        stale = False
        if os.path.exists(proc_path):
            # Files produced with different calibration settings (e.g. before degradation
            # correction was enabled) must be reprocessed, not silently reused.
            try:
                existing_degcorr = bool(fits.getheader(proc_path, ext=1).get("DEGCORR", False))
                stale = existing_degcorr != degcorr
            except Exception:
                stale = True
        if stale or not os.path.exists(proc_path):
            if stale:
                logging.info(f"Reprocessing stale L2 file (calibration settings changed): {proc_path}")
            sdo_map = aia_process(sdo_map, table, degcorr=degcorr, correction_table=correction_table)
            sdo_map = aia_reproject(sdo_map, hmi_match)
            proc_path = l2_file_save(sdo_map, path, overwrite=stale)
        # This updates process status for tqdm more effectively.
        sys.stdout.flush()

    return proc_path


def l2_file_save(fits_map, path: str, overwrite: bool = False):
    r"""
    Save a "level 2" FITS map.

    Parameters
    ----------
        fits_map : `sunpy.map.Map`
            The FITS map to save.
        path : `str`
            The path to save the file.
        overwrite : `bool`
            Whether to overwrite existing files.

    Returns
    -------
        fits_path : `str`
            The path of the saved file.
    """
    time = fits_map.date.to_value("ymdhms")
    year, month, day = time[0], time[1], time[2]
    map_path = f"{path}/02_intermediate/{year}/{month}/{day}/SDO/{fits_map.nickname}"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    fits_path = f"{map_path}/02_{fits_map.meta['fname']}"
    if (not Path(fits_path).exists()) or overwrite:
        save_compressed_map(fits_map, fits_path, hdu_type=CompImageHDU, overwrite=True)
    return fits_path


def table_match(aia_maps, hmi_maps):
    r"""
    Matches l3 AIA submaps with corresponding HMI submaps based on the closest time difference, and returns Astropy Table

    Parameters
    ----------
        aia_maps : `list`
            List of AIA map paths.
        hmi_maps : `list`
            List of HMI map paths.

    Returns
    -------
        paired_table : `Astropy.table`
            A list containing tuples of paired AIA and HMI maps.
    """
    aia_wavelnth = []
    aia_paths = []
    aia_quality = []
    hmi_paths = []
    hmi_headers = [fits.getheader(hmi_map, ext=1) for hmi_map in hmi_maps]
    hmi_times = [Time(header["date-obs"]) for header in hmi_headers]
    paired_times = []
    aia_times = []
    hmi_quality = []

    for aia_map in aia_maps:
        aia_header = fits.getheader(aia_map, ext=1)
        date = aia_header["date-obs"]
        t_d = [abs((Time(date) - hmi_time).value) for hmi_time in hmi_times]
        match_idx = t_d.index(min(t_d))
        aia_paths.append(aia_map)
        hmi_paths.append(hmi_maps[match_idx])
        paired_times.append(hmi_headers[match_idx]["date-obs"])
        hmi_quality.append(hmi_headers[match_idx]["quality"])
        aia_quality.append(aia_header["quality"])
        aia_wavelnth.append(aia_header["wavelnth"])
        aia_times.append(date)
    paired_table = Table(
        {
            "Wavelength": aia_wavelnth,
            "AIA files": aia_paths,
            "AIA quality": aia_quality,
            # "AIA time": aia_times,
            "HMI files": hmi_paths,
            "HMI quality": hmi_quality,
            # "HMI time": paired_times,
        }
    )
    return paired_table, aia_paths, aia_quality, aia_times, hmi_paths, hmi_quality, paired_times


def crop_map(sdo_map, center, height, width, noaa_time):
    r"""
    Crops a provided SDO map and returns a submap centered on a flare according to provided parameters of lat, lon, height and width.

    Parameters
    ----------
        sdo_map : `sunpy.map.Map`
            Provided SDO map path (AIA/HMI).
        center : `SkyCoord`
            Coordinate of center of active region (Heliographic Stonyhurst coordinate system).
        height : `Quantity`
            The height of the submap in u.pix.
        width : `Quantity`
            The width of the submap in u.pix.
        noaa_time : `int`
            The start time registered with the noaa active region.

    Returns
    -------
        s_map : `sunpy.map.Map.submap`
            A submap centered around the provided coordinates.
    """
    sdo_map = sunpy.map.Map(sdo_map)
    t_diff = Time(sdo_map.date) - Time(noaa_time)
    t_diff = t_diff.to_value(u.s) * u.s
    new_center = solar_rotate_coordinate(center, time=t_diff)
    new_center = new_center.transform_to(sdo_map.coordinate_frame)
    pix_center = new_center.to_pixel(sdo_map.wcs)
    top_right, bottom_left = pixel_to_bboxcoords(width, height, pix_center * u.pix)
    s_map = sdo_map.submap(bottom_left=bottom_left, top_right=top_right)
    return s_map


def map_reproject(hmi_origin_wcs, sdo_path, ar_num):
    r"""
    Reprojects a provided SDO map onto the wcs of a provided origin map. As intended, this is to reproject a "level 2" map onto the wcs of a cropped and centered level 3 HMI map.

    Parameters
    ----------
        sdo_packed : `namedtuple`
            Contains the origin map and the sdo map which is to be projected.

    Returns
    ----------
        fits_path : `str`
            The path location of the saved submap.
    """
    sdo_map = sunpy.map.Map(sdo_path)
    with propagate_with_solar_surface():
        sdo_rpr = sdo_map.reproject_to(hmi_origin_wcs)
    time = sdo_map.date.to_value("ymdhms")
    year, month, day = time[0], time[1], time[2]
    path = config["paths"]["data_folder"]
    map_path = f"{path}/03_processed/{year}/{month}/{day}/SDO/{sdo_map.nickname}"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    fits_path = f"{map_path}/{ar_num}_03_{sdo_map.meta['fname']}"
    sdo_rpr.meta["quality"] = sdo_map.meta["quality"]
    sdo_rpr.meta["wavelnth"] = sdo_map.meta["wavelnth"]
    sdo_rpr.meta["date-obs"] = sdo_map.meta["date-obs"]
    sdo_rpr.meta["degcorr"] = sdo_map.meta.get("degcorr", False)
    if sdo_rpr.dimensions[0].value < int(config["drms"]["patch_width"]):
        sdo_rpr = pad_map(sdo_rpr, config["drms"]["patch_width"])
    save_compressed_map(sdo_rpr, fits_path, hdu_type=CompImageHDU, overwrite=True)

    return fits_path


def vid_match(table, name, path):
    r"""
    Creates an animated mosaic of images from the data run - including all AIA wavelengths and HMI 720s magnetogram.

    Parameters
    ----------
        table : `AstropyTable`
            An astropy table containing the filenames of all AIA wavelengths and their paired HMI files.
        name : `str`
            A string containing the name of the file to be appended, matching all versions of a file above level 1.
        path: `str`
            The base path of the processed data.

    Returns
    ----------
        output_file : `str`
            A string containing the path of the completed mosaic animation.
    """
    hmi_files = np.unique(table["HMI files"])
    table["Wavelength"] = table["Wavelength"].astype(int)
    wvls = np.unique(table["Wavelength"])

    aia_lookup = {(row["HMI files"], row["Wavelength"]): row["AIA files"] for row in table}

    nrows, ncols = 4, 3

    for idx, hmi in enumerate(hmi_files):
        mosaic_plot(
            hmi=hmi,
            name=name,
            frame_idx=idx,
            nrows=nrows,
            ncols=ncols,
            wvls=wvls,
            aia_lookup=aia_lookup,
            path=path,
        )

    return mosaic_animate(path, name)


def l4_file_pack(aia_paths, hmi_paths, dir_path, rec, out_table, before_fls, after_fls, anim_path):
    r"""
    Packs files into folders along with folder specific records identifying .fits files

    Parameters
    ----------
        aia_paths : `list`
            The paths to the aia files to be packed.
        hmi_paths : `list`
            The paths to the hmi files to be packed.
        dir_path : `str`
            The path to the directory of the l4 data.
        rec : `str`
            The record name unique to the current run.
        out_table : `AstropyTable`
            The table containing the records specific to the folder containing information for the current run.
        before_fls : `AstropyTable`
            Table containing log of flares occurring within the current run.
        after_fls : `AstropyTable`
            Table containing log of flares occurring within 24 hours of target time (end of run).
        anim_path: `str` or `None`
            The path to the mosaic animation of the current run, or `None` if
            animations are disabled.
    """
    base_path = Path(dir_path) / "data" / rec
    folder_hmi = base_path / "HMI"
    folder_aia = base_path / "AIA"

    folder_hmi.mkdir(parents=True, exist_ok=True)
    folder_aia.mkdir(parents=True, exist_ok=True)

    # Symlink files instead of copying
    _link_files(aia_paths, folder_aia)
    _link_files(set(hmi_paths), folder_hmi)

    if anim_path is not None:
        anim_dst = base_path / Path(anim_path).name
        anim_dst.unlink(missing_ok=True)
        anim_dst.symlink_to(os.path.relpath(anim_path, start=base_path))

    # Need to be real writes
    out_table.write(base_path / f"{rec}.csv", overwrite=True)
    before_fls.write(base_path / "before_flares.parquet", overwrite=True)
    after_fls.write(base_path / "after_flares.parquet", overwrite=True)


def _link_files(paths, destination):
    for file in paths:
        src = Path(file)
        dst = destination / src.name

        # Remove existing file/symlink if present
        dst.unlink(missing_ok=True)

        # Create relative symlink (portable)
        dst.symlink_to(os.path.relpath(src, start=destination))


def pad_map(map, targ_width):
    r"""
    Pads the map data of a submap which is narrower than the specified submap width due to reaching the edge of the image array.

    Parameters
    ----------
        map : `sunpy.map`
            The sunpy submap to be resized.
        targ_width : `int`
            The target width of the submaps within the data run.

    Returns
    ----------
        new_map : `sunpy.map`
            The new, padded submap.
    """
    x_dim = map.dimensions[0].value
    targ_width = int(targ_width)
    diff = int(targ_width - x_dim)
    if map.center.Tx > 0:
        data = np.pad(map.data, ((0, 0), (diff, 0)), constant_values=np.nan)
    else:
        data = np.pad(map.data, ((0, 0), (0, diff)), constant_values=np.nan)
    new_map = sunpy.map.Map(data, map.meta)
    return new_map


def flare_check(start, end, ar_num, table):
    r"""
    Checks a provided start time and end time, along with an active region number, and determines if a flare occurs within that period.

    Parameters
    ----------
        start : `Astropy.Time`
            The start time (duration - 1 hour) of a target run.
        end : `Astropy.Time`
            The end time (start_time - 1 hour) of a target run.
        ar_num : `str`
            The active region number of a target run.
        table : `int`
            The table of flare times used to check for flare events within run duration.

    Returns
    ----------
        category : `int`
            The category of the run. If no flare detected in run, category = 1, otherwise, category = 2.
        flares : `dict`
            A dictionary containing the number of flares in a given category, used for instructional filenames.
    """
    ar_table = table[table["noaa_number"] == ar_num]
    ar_table = ar_table[Time(ar_table["start_time"]) < end]
    ar_table = ar_table[Time(ar_table["start_time"]) > start]
    category = 1
    flares = {"X": 0, "M": 0, "C": 0}
    if len(ar_table) > 0:
        category = 2
        for cat in flares:
            flares[cat] = int(len(ar_table[[flare.startswith(cat) for flare in ar_table["goes_class"]]]))
    return category, flares


def flare_log(start, end, ar_num, table):
    r"""
    Checks a provided start time and end time, along with an active region number, and logs flares within that period.

    Parameters
    ----------
        start : `Astropy.Time`
            The start time of a target window.
        end : `Astropy.Time`
            The end time of a target window.
        ar_num : `str`
            The active region number of a target run.
        table : `int`
            The table of flare times used to check for flare events within run duration.

    Returns
    ----------
        ar_table : `Atropy.Table
            A table containing information on flares occurring within the time period for a given active region.
    """
    ar_table = table[table["noaa_number"] == ar_num]
    ar_table = ar_table[Time(ar_table["start_time"]) < end]
    ar_table = ar_table[Time(ar_table["start_time"]) > start]
    ar_table = ar_table["noaa_number", "goes_class", "start_time", "peak_time", "end_time"]
    ar_table = ar_table[ar_table.argsort("start_time")]
    flares = {"X": 0, "M": 0, "C": 0}
    if len(ar_table) > 0:
        for cat in flares:
            flares[cat] = int(len(ar_table[[flare.startswith(cat) for flare in ar_table["goes_class"]]]))
    return ar_table, flares
