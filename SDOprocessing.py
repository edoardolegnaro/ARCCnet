# Define required libraries - check to see if Arccnet already has these as requirements.
import os
import glob
import math
import itertools

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


# This will be sourced from the utils module - simulating usage at the moment.
def load_config():
    # Replace this with whatever email(s) we want to use for this purpose.
    os.environ["JSOC_EMAIL"] = "danielgass192@gmail.com"
    config = {"path": "/Users/danielgass/Desktop", "wavelengths": [171, 193], "rep_tol": 60, "sample": 60}
    return config


class PipeUtils:
    # Contains utility functions to help with time/string handling and list comparisons with strings.

    def time_delta(start: str, end: str):
        start_t = Time(start, format="fits")
        end_t = Time(end, format="fits")
        t_del_h = math.ceil((end_t - start_t).to_value(u.hour))
        return t_del_h

    def time_diff_s(start: str, end: str):
        start_t = sunpy.time.parse_time(start)
        end_t = sunpy.time.parse_time(end)
        t_del_s = abs(end_t - start_t).to_value(u.second)
        return t_del_s

    def change_time(time: str, shift: int):
        time_d = Time(time, format="fits") + shift * (u.second)
        return time_d.to_value("fits")

    def comp_list(file: str, file_list: list):
        return any(file in name for name in file_list)


class DrmsDownload:
    def hmi_query(time_1: str, time_2: str, keys: list, sample: int):
        client = drms.Client()
        duration = PipeUtils.time_delta(time_1, time_2)
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

    def aia_query(hmi_query, keys, wavelength):
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
        client = drms.Client()
        retries = 0
        qry = client.query(qstr, keys)
        time = sunpy.time.parse_time(qry["T_REC"].values[0])
        while qry["QUALITY"].values[0] != 0 and retries < 3:
            qry = client.query(f"hmi.M_720s[{time}]" + "{magnetogram}", keys)
            time = PipeUtils.change_time(time, 720)
            retries += 1
        return qry["*recnum*"].values[0]

    def aia_rec_find(qstr, keys):
        client = drms.Client()
        retries = 0
        qry = client.query(qstr, keys)
        time, wvl = qry["T_REC"].values[0][0:-1], qry["WAVELNTH"].values[0]
        while qry["QUALITY"].values[0] != 0 and retries < 10:
            qry = client.query(f"aia.lev1_euv_12s[{time}][{wvl}]" + "{image}", keys)
            time = PipeUtils.change_time(time, 12)
            retries += 1
        if qry["QUALITY"].values[0] == 0:
            return qry["*recnum*"].values

    def l1_file_save(export, query, path):
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
        matching_files = [PipeUtils.comp_list(file, existing_files) for file in export.urls["filename"]]
        missing_files = [not value for value in matching_files]

        export.urls["filename"] = path_prefix + export.urls["filename"]
        if len(export.urls[missing_files].index) > 0:
            export.download(directory="", index=export.urls[missing_files].index)
            total_files = [export.urls["filename"][matching_files] + export.urls["filename"][existing_files]]
        else:
            total_files = export.urls["filename"][matching_files]
        return export, total_files


# This class is currently not used - Fido having some issues with reliable downloads, but retained code for possible future use.
class FidoDownload:
    # Caching will probably be a good idea - file keeping track of which fits are available.
    # global fetch_request
    global aia_drms_qual_check, hmi_drms_qual_check

    def fido_aia_fetch_request(
        req,
        euv_path: str,
        tmpstorage: bool = False,
    ):
        npe = ""
        filepaths = []
        if tmpstorage:
            npe = "/temp/"
        for result in req:
            wvl = int(min(result[0][0]["Wavelength"].value))
            path = f"{euv_path}/{wvl}"
            files = Fido.fetch(result, max_conn=10, path=f"{npe}{path}")
            # 8. Create loop to ensure all files downloaded/were redownloaded as necessary
            filepaths.append(files)
            while files.errors != []:
                files = Fido.fetch(result, max_conn=10, path=f"{npe}{path}")
                filepaths.append(files)
                # filepaths = np.reshape(filepaths, (-1,1))
        return filepaths

    def fido_hmi_fetch_request(
        req,
        hmi_path: str,
        tmpstorage: bool = False,
    ):
        npe = ""
        filepaths = []
        if tmpstorage:
            npe = "/temp/"
        for result in req:
            files = Fido.fetch(result, max_conn=10, path=f"{npe}{hmi_path}")
            # 8. Create loop to ensure all files downloaded/were redownloaded as necessary
            filepaths.append(files)
            while files.errors != []:
                files = Fido.fetch(result, max_conn=10, path=f"{npe}{hmi_path}")
                filepaths = filepaths.append(files)
        return np.unique(filepaths)

    def fido_aia_ts_request(start: str, end: str, wavelengths: list, time: int):
        # Convert to Time Delta
        time_d = a.Time(start, end)

        # Pass argument to Fido (May switch to DRMS when refactoring/updating python versions)
        res = []
        for wvl in wavelengths:
            res.append(Fido.search(time_d, a.Wavelength(wvl * u.AA), a.Instrument("AIA"), a.Sample(time * u.min)))
        return res

    def fido_hmi_ts_request(
        start: str,
        end: str,
        verif: str,
        time: int,
        tmpstorage: bool = True,
    ):
        time_d = a.Time(start, end)
        res = []
        res.append(Fido.search(time_d, a.jsoc.Series("hmi.m_720s"), a.Sample(time * u.min), a.jsoc.Notify(verif)))
        return res

    def aia_drms_qual_check(qstr: str, keys: list):
        client = drms.Client()
        bad_results = []
        request = client.query(qstr, key=keys)
        bad_result = request[request.QUALITY != 0]["T_OBS"]
        if len(bad_result > 0):
            bad_results.append(bad_result)
        result = request[request.QUALITY == 0]
        good_results = result.index

        return good_results, bad_results.to_list()

    def hmi_drms_qual_check(qstr: str, keys: list):
        client = drms.Client()
        bad_results = []
        request = client.query(qstr, key=keys)
        bad_result = request[request.QUALITY != 0]["T_OBS"]
        if len(bad_result > 0):
            bad_results.append(bad_result)
        result = request[request.QUALITY == 0]
        good_results = result.index

        return good_results, bad_results.to_list()

    def hmi_quality_check(
        # Checks provided map for list of approved QUALITY flag values.
        # Returns the header.name of bad files.
        map,
        accepted: list = [0],
    ):
        bad_list = []
        # if type(map) != list:
        # 	map = [map]
        for m in map:
            m_map = sunpy.map.Map(m)
            if m_map.meta["quality"] in accepted is False:
                bad_list.append(m_map.name)
        # status_list = f'Quality check complete - {len(bad_list)}/{len(map)} map(s) failed.'
        # print(status_list)
        return bad_list

    def aia_quality_check(
        # 9. Filter .fits for QUALITY flag, remove certain images, and check for nearest file to replace.
        # Checks provided map for list of approved QUALITY flag values.
        # Returns the header.name of bad files.
        map,
        accepted: list = [0],
    ):
        bad_list = []
        # if type(map) != list:
        # 	map = [map]
        for m in map:
            m_map = sunpy.map.Map(m)
            if m_map.meta["quality"] in accepted is False:
                bad_list.append(m_map.name)
        # status_list = f'Quality check complete - {len(bad_list)}/{len(map)} map(s) failed.'
        # print(status_list)
        return bad_list

    def dl_paths(fin_result, filelist):
        # Function to match filenames to a UnifiedResponse function, used after download for quality checks.
        timestamp = np.array(fin_result["T_REC"])
        timestamp = char.translate(timestamp, str.maketrans("", "", ".:"))
        matching_files = [file for file in filelist if any(time in file for time in timestamp)]
        return matching_files

    def return_missing_files(result, filelist):
        # Reads in a UnifiedResponse and a list of files and returns modified response containing only missing entries if any.
        # fileset = set(np.array(filelist))
        result = np.reshape(result, -1)
        timestamp = np.array(result["T_REC"])
        timestamp2 = char.translate(timestamp, str.maketrans("", "", ".:"))
        missing_timestamps = [name for name in timestamp2 if not any(name in file for file in filelist)]
        missing_indices = np.where(np.isin(timestamp2, missing_timestamps))[0]
        res_new = result[missing_indices]
        return res_new


class SDOproc:
    # Levels an AIA map to level 1.5, can also correct for degradation and deconvolve psf if needed.
    def aia_process(aia_map, deconv: bool = False, degcorr: bool = False, exnorm: bool = True):
        # This step needs to be done before prep if selected.
        if deconv:
            aia_map = SDOproc.aia_deconv(aia_map)
        aia_map = SDOproc.aia_prep(aia_map)
        if degcorr:
            aia_map = SDOproc.aia_degcorr(aia_map)
        if exnorm:
            aia_map = SDOproc.aia_expnorm(aia_map)
        return aia_map

    # Prepares aia map to level 1.5, standard aiapy process.
    def aia_prep(aia_map):
        aia_map = update_pointing(aia_map)
        aia_map = register(aia_map)
        return aia_map

    # Assuming default parameters for deconvolve function for now
    def aia_deconv(aia_map):
        aia_map = deconvolve(aia_map)
        return aia_map

    # Corrects for exposure time, has to cast to int and repackage fits file.
    def aia_expnorm(aia_map):
        aiad = aia_map.data / aia_map.exposure_time
        aia_map = sunpy.map.Map(aiad.astype(int), aia_map.fits_header)
        return aia_map

    # Corrects degradation.
    # May want to get our own correction tables, but downloads each time this is called at present.
    def aia_degcorr(aia_map):
        aia_map = correct_degradation(aia_map)
        return aia_map

    # Reprojects AIA map to HMI map using wcs.
    # The maps need to be in closest possible time frames.
    def aia_reproject(aia_map, hmi_map):
        rpr_aia_map = aia_map.reproject_to(hmi_map.wcs)
        rpr_aia_map.meta["wavelnth"] = aia_map.meta["wavelnth"]
        rpr_aia_map.meta["t_obs"] = aia_map.meta["t_obs"]
        rpr_aia_map.meta["instrume"] = aia_map.meta["instrume"]
        print(rpr_aia_map.meta["instrume"])
        del hmi_map
        return rpr_aia_map

    # Finds the coordinates of pixels outside of Rsun_obs for an HMI image and assigns NaN.

    def hmi_mask(hmimap):
        hpc_coords = all_coordinates_from_map(hmimap)
        mask = coordinate_is_on_solar_disk(hpc_coords)
        hmidata = hmimap.data
        hmidata[mask is False] = np.nan
        hmimap = sunpy.map.Map(hmidata, hmimap.meta)
        del hmidata
        return hmimap

    # Function for saving l2 files - sorts by instrument and date and checks for redundancies. Returns path name.
    def l2_file_save(fits_map, filename: str, path: str, overwrite: bool = False):
        name = str.split(filename, "/")[-1]
        instr = fits_map.meta["INSTRUME"].split("_")[0]
        print(instr)
        if instr == "HMI":
            fits_map.meta["t_obs"] = fits_map.meta["t_obs"][0:-8]
        time = sunpy.time.parse_time(fits_map.meta["t_obs"]).to_value("ymdhms")
        year, month, day = time[0], time[1], time[2]
        map_path = f"{path}/02_processed/{year}/{month}/{day}/SDO/{instr}"
        os.makedirs(map_path, exist_ok=True)
        hmi_fits = f"{map_path}/02_{name}"
        if glob.glob(hmi_fits) == [] or overwrite is True:
            fits_map.save(hmi_fits, hdu_type=CompImageHDU, overwrite=True)

        return hmi_fits
