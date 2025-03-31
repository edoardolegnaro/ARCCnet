from concurrent.futures import ProcessPoolExecutor

import sunpy.map

from SDOprocessing import DrmsDownload, SDOproc, load_config

from pathlib import PurePath

def match_files(aia_maps, hmi_maps):
    r"""
    Matches AIA maps with corresponding HMI maps based on the closest time difference.
    
    Parameters:
    -----------
        aia_maps (list): List of AIA maps.
        hmi_maps (list): List of HMI maps.
    
    Returns:
    --------
        packed_files(list): A list containing tuples of paired AIA and HMI maps.
    """
    packed_files = []
    for aia_map in aia_maps:
        t_d = [abs((aia_map.date - hmi_map.date).value*24*3600) for hmi_map in hmi_maps]
        hmi_match = hmi_maps[t_d.index(min(t_d))]
        packed_files.append([aia_map, hmi_match])
    return packed_files

def add_fnames(maps, paths):
    r"""
    Adds file names to fits map metadata.
    
    Parameters:
    -----------
        maps (list): List of fits maps.
        paths (list): List of file paths.
    
    Returns:
    --------
        named_map (list) : List of fits maps with file names added to metadata.
    """
    named_maps = []
    for map,fname in zip(maps, paths):
        map.meta['fname'] = PurePath(fname).name
        named_maps.append(map)
    return named_maps

def drms_pipeline(starts: list, path: str, keys: list, wavelengths: list = [171, 193, 304, 211, 335, 94, 131, 1600, 4500, 1700] , sample: int = 60):
    r"""
    Performs pipeline to download and process AIA and HMI data.
    
    Parameters:
    -----------
        starts (list): List of start and end times for the data retrieval.
        path (str): Path to save the downloaded data.
        keys (list): List of keys for the data query.
        wavelengths (list): List of wavelengths for the AIA data (default all AIA wvl).
        sample (int): Sample rate for the data cadence (default 1/hr).
    
    Returns:
    --------
        aia_maps, hmi_maps (tuple): A tuple containing the AIA maps and HMI maps.
    """
    start_t, end_t = starts[1], starts[2]

    hmi_query, hmi_export = DrmsDownload.hmi_query_export(start_t, end_t, keys, sample)
    aia_query, aia_export = DrmsDownload.aia_query_export(hmi_query, keys, wavelengths)

    hmi_dls, hmi_exs = DrmsDownload.l1_file_save(hmi_export, hmi_query, path)
    aia_dls, aia_exs = DrmsDownload.l1_file_save(aia_export, aia_query, path)
    
    hmi_maps = sunpy.map.Map(hmi_exs)
    hmi_maps = add_fnames(hmi_maps, hmi_exs)
    aia_maps = sunpy.map.Map(aia_exs)
    aia_maps = add_fnames(aia_maps, aia_exs)
    return aia_maps, hmi_maps

def hmi_l2(hmi_map):
    r"""
    Processes the HMI map to "level 2" by applying a mask and saving it to the output directory.
    
    Parameters:
    -----------
        hmi_map (sunpy.map.Map): HMI map to be processed.
    
    Returns:
    --------
       proc_path (str): Path to the processed HMI map.
    """
    path = load_config()["path"]
    hmi_map = SDOproc.hmi_mask(hmi_map)
    proc_path = SDOproc.l2_file_save(hmi_map, path)
    return proc_path

def aia_l2(packed_maps):
    r"""
    Processes the AIA map to "level 2" by leveling, rescaling, trimming, and reprojecting it to match the nearest HMI map.
    
    Parameters:
    -----------
        packed_maps (list): List containing the AIA map and its corresponding HMI map.
    
    Returns:
    --------
        proc_path (str): Path to the processed AIA map.
    """
    path = load_config()["path"]
    aia_map, hmi_match = packed_maps[0], packed_maps[1]
    aia_map = SDOproc.aia_process(aia_map)
    aia_map = SDOproc.aia_reproject(aia_map, hmi_match)
    proc_path = SDOproc.l2_file_save(aia_map, path)
    return proc_path

if __name__ == "__main__":
    config = load_config()
    keys = ["T_REC", "T_OBS", "QUALITY", "WAVELNTH", "*recnum*", "INSTRUME"]
    starts = [[0, "2017-02-01T01:00:00", "2017-02-01T03:00:00"]]

    for range in starts:
        aia_maps, hmi_maps = drms_pipeline(
            range, config["path"], keys, config["wavelengths"], config["sample"]
        )

    with ProcessPoolExecutor(6) as executor:
        hmi_proc = executor.map(hmi_l2, hmi_maps)
        packed_files = match_files(aia_maps, hmi_maps)
        aia_proc = executor.map(aia_l2, packed_files)

# 70 X class flares.
# Read the flare list.
# Not just HEK, look for save files for flares.
# Random sample (100 ish for M and below flares) 