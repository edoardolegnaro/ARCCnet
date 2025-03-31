# Define required libraries - check to see if Arccnet already has these as requirements.
import os
import glob
import math
import itertools
import copy

import drms
import numpy as np
import sunpy.map
from aiapy.calibrate import correct_degradation, register, update_pointing
from aiapy.psf import deconvolve
from numpy import char
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.net import Fido
from sunpy.net import attrs as a

import astropy.units as u
from astropy.io.fits import CompImageHDU
from astropy.time import Time


def load_config():
    r"""
    Load the configuration for the SDO processing pipeline.

    Returns:
    --------
        config (dict): The configuration dictionary.
    """
    # Replace this with whatever email(s) we want to use for this purpose.
    os.environ["JSOC_EMAIL"] = "danielgass192@gmail.com"
    config = {"path": "/Users/danielgass/Desktop", "wavelengths": [171, 193, 304, 211], "rep_tol": 60, "sample": 60}
    return config


def time_delta(start: str, end: str):
    r"""
    Calculate the time difference in hours between two FITS timestamps.

    Parameters:
    -----------
        start (str): A start timestamp in FITS format.
        end (str): An end timestamp in FITS format.

    Returns:
    --------
        t_del_h (int): The time difference in hours.
    """
    start_t = Time(start, format="fits")
    end_t = Time(end, format="fits")
    t_del_h = math.ceil((end_t - start_t).to_value(u.hour))
    return t_del_h


def change_time(time: str, shift: int):
    r"""
    Change the timestamp by a given time shift.

    Parameters:
    -----------
        time (str): A timestamp in FITS format.
        shift (int): The time shift in seconds.

    Returns:
    --------
        str: The updated timestamp in FITS format.
    """
    time_d = Time(time, format="fits") + shift * (u.second)
    return time_d.to_value("fits")


def comp_list(file: str, file_list: list):
    r"""
    Check if a file is present in a list of files.

    Parameters:
    -----------
        file (str): The file to check.
        file_list (list): The list of files.

    Returns:
    --------
        list: A list of booleans for each element in list. True if the file is present, False otherwise.
    """
    return any(file in name for name in file_list)


class DrmsDownload:
    def hmi_query_export(time_1: str, time_2: str, keys: list, sample: int):
        r"""
        Query and export HMI data from the JSOC database.

        Paramaters:
        -----------
            time_1 (str): The start timestamp in FITS format.
            time_2 (str): The end timestamp in FITS format.
            keys (list): A list of keys to query.
            sample (int): The sample rate in minutes.

        Returns:
        --------
            hmi_query_full (pandas df), hmi_result (drms export) (tuple): A tuple containing the query result and the export data response.
        """
        client = drms.Client()
        duration = time_delta(time_1, time_2)
        qstr_hmi = f"hmi.M_720s[{time_1}/{duration}h@{sample}m]" + "{magnetogram}"
        hmi_query = client.query(qstr_hmi, keys)
        good_result = hmi_query[hmi_query.QUALITY == 0]
        good_num = good_result["*recnum*"].values
        bad_result = hmi_query[hmi_query.QUALITY != 0]

        hmi_values = []
        qstrs_hmi = [f"hmi.M_720s[{time}]" + "{magnetogram}" for time in bad_result["T_OBS"]]
        for qstr in qstrs_hmi:
            hmi_values.append(DrmsDownload.hmi_rec_find(qstr, keys))
        patched_num = [*hmi_values]

        joined_num = [*good_num, *patched_num]
        hmi_num_str = str(joined_num).strip("[]")
        hmi_qstr = f"hmi.M_720s[! recnum in ({hmi_num_str}) !]" + "{magnetogram}"
        hmi_query_full = client.query(hmi_qstr, keys)
        hmi_result = client.export(hmi_qstr, method="url", protocol="fits", email=os.environ["JSOC_EMAIL"])
        hmi_result.wait()
        return hmi_query_full, hmi_result

    def aia_query_export(hmi_query, keys, wavelength):
        r"""
        Query and export AIA data from the JSOC database.

        Parameters:
        -----------
            hmi_query: The HMI query result.
            keys: List of keys to query.
            wavelength: An AIA wavelength.

        Returns:
        --------
            aia_query_full, aia_result (tuple): A tuple containing the query result and the export data response.
        """
        client = drms.Client()
        value = []
        qstrs_aia = [f"aia.lev1_euv_12s[{time}]{wavelength}" + "{image}" for time in hmi_query["T_REC"]]
        for qstr in qstrs_aia:
            value.append(DrmsDownload.aia_rec_find(qstr, keys))
        unpacked_aia = list(itertools.chain.from_iterable(value))
        aia_num_str = str(unpacked_aia).strip("[]")
        aia_qstr = f"aia.lev1_euv_12s[! recnum in ({aia_num_str}) !]" + "{image}"
        aia_query_full = client.query(aia_qstr, keys)
        aia_result = client.export(aia_qstr, method="url", protocol="fits", email=os.environ["JSOC_EMAIL"])
        aia_result.wait()
        return aia_query_full, aia_result

    def hmi_rec_find(qstr, keys):
        r"""
        Find the HMI record number for a given query string.

        Parameters:
        -----------
            qstr (str): A query string.
            keys (list): List of keys to query.

        Returns:
        --------
            int: The HMI record number.
        """
        client = drms.Client()
        retries = 0
        qry = client.query(qstr, keys)
        time = sunpy.time.parse_time(qry["T_REC"].values[0])
        while qry["QUALITY"].values[0] != 0 and retries < 3:
            qry = client.query(f"hmi.M_720s[{time}]" + "{magnetogram}", keys)
            time = change_time(time, 720)
            retries += 1
        return qry["*recnum*"].values[0]

    def aia_rec_find(qstr, keys):
        r"""
        Find the AIA record number for a given query string.

        Parameters:
        -----------
            qstr (str): A query string.
            keys (list): List of keys to query.

        Returns:
        --------
            int: The AIA record number.
        """
        client = drms.Client()
        retries = 0
        qry = client.query(qstr, keys)
        time, wvl = qry["T_REC"].values[0][0:-1], qry["WAVELNTH"].values[0]
        while qry["QUALITY"].values[0] != 0 and retries < 10:
            qry = client.query(f"aia.lev1_euv_12s[{time}][{wvl}]" + "{image}", keys)
            time = change_time(time, 12)
            retries += 1
        if qry["QUALITY"].values[0] == 0:
            return qry["*recnum*"].values

    def l1_file_save(export, query, path):
        r"""
        Save the exported data as level 1 FITS files.

        Parameters:
        -----------
            export: A drms data export.
            query: A drms query result.
            path (str): A base path to save the files.

        Returns:
        --------
            export (drms export), total_files (list) (tuple): A tuple containing the updated export data and the list of saved file paths.
        """
        instr = query["INSTRUME"][0][0:3]
        path_prefix = []

        for time in query["T_OBS"]:
            time = sunpy.time.parse_time(time).to_value("ymdhms")
            year, month, day = time["year"], time["month"], time["day"]
            path_prefix.append(f"{path}/01_raw/{year}/{month}/{day}/SDO/{instr}/")

        existing_files = []
        for dirs in np.unique(path_prefix):
            os.makedirs(dirs, exist_ok=True)
            existing_files.append(glob.glob(f"{dirs}/*.fits"))

        existing_files = [*existing_files][0]
        matching_files = [comp_list(file, existing_files) for file in export.urls["filename"]]
        missing_files = [not value for value in matching_files]
        export.urls["filename"] = path_prefix + export.urls["filename"]
        if len(export.urls[missing_files].index) > 0:
            total_files = list(export.urls[matching_files].index) + list(export.urls[missing_files].index)
            total_files = export.urls["filename"][total_files]
            export.download(directory="", index=export.urls[missing_files].index)
        else:
            total_files = list(export.urls["filename"][matching_files])
        return export, total_files


class SDOproc:
    @staticmethod
    def aia_process(aia_map, deconv: bool = False, degcorr: bool = False, exnorm: bool = True):
        r"""
        Process an AIA map to level 1.5.

        Parameters:
        -----------
            aia_map: The AIA map to process.
            deconv (bool): Whether to deconvolve the PSF.
            degcorr (bool): Whether to correct for degradation.
            exnorm (bool): Whether to normalize exposure.

        Returns:
        --------
            aia_map (sunpy.map.Map):  Processed AIA map.
        """
        if deconv:
            aia_map = deconvolve(aia_map)
        aia_map = update_pointing(aia_map)
        aia_map = register(aia_map)
        if degcorr:
            aia_map = correct_degradation(aia_map)
        if exnorm:
            aiad = aia_map.data / aia_map.exposure_time
            aia_map = sunpy.map.Map(aiad.astype(int), aia_map.fits_header)
        return aia_map

    @staticmethod
    def aia_reproject(aia_map, hmi_map):
        r"""
        Reproject an AIA map to the same coordinate system as an HMI map.

        Parameters:
        -----------
            aia_map: The AIA map to reproject.
            hmi_map: The HMI map to use as the target coordinate system.

        Returns:
        --------
            rpr_aia_map (sunpy.map.Map): Reprojected AIA map.
        """
        rpr_aia_map = aia_map.reproject_to(hmi_map.wcs)
        rpr_aia_map.meta["wavelnth"] = aia_map.meta["wavelnth"]
        rpr_aia_map.meta["t_obs"] = aia_map.meta["t_obs"]
        rpr_aia_map.meta["instrume"] = aia_map.meta["instrume"]
        rpr_aia_map.meta["fname"] = aia_map.meta["fname"]
        rpr_aia_map.nickname = aia_map.nickname

        return rpr_aia_map

    @staticmethod
    def hmi_mask(hmimap):
        r"""
        Mask pixels outside of Rsun_obs in an HMI map.

        Parameters:
        -----------
            hmimap: The HMI map.

        Returns:
        --------
            hmimap (sunpy.map.Map): The masked HMI map.
        """
        hpc_coords = all_coordinates_from_map(hmimap)
        mask = coordinate_is_on_solar_disk(hpc_coords)
        hmidata = hmimap.data
        hmidata[mask is False] = np.nan
        hmimap = sunpy.map.Map(hmidata, hmimap.meta)
        return hmimap

    @staticmethod
    def l2_file_save(fits_map, path: str, overwrite: bool = False):
        r"""
        Save a "level 2" FITS map.

        Args:
            fits_map (sunpy.map.Map): The FITS map to save.
            path (str): The path to save the file.
            overwrite (bool): Whether to overwrite existing files.

        Returns:
            fits_path (str): The path of the saved file.
        """
        time = fits_map.date.to_value("ymdhms")
        year, month, day = time[0], time[1], time[2]
        map_path = f"{path}/02_processed/{year}/{month}/{day}/SDO/{fits_map.nickname}"
        os.makedirs(map_path, exist_ok=True)
        fits_path = f"{map_path}/02_{fits_map.meta['fname']}"
        if (not os.path.exists(fits_path)) or overwrite:
            fits_map.save(fits_path, hdu_type=CompImageHDU, overwrite=True)
        return fits_path