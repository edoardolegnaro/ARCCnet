from concurrent.futures import ProcessPoolExecutor

import sunpy.map

from SDOprocessing import DrmsDownload, PipeUtils, SDOproc, load_config


def match_files(aia_paths, hmi_paths):
    hmi_times = [sunpy.map.Map(hmi).meta["T_OBS"][0:-8] for hmi in hmi_paths]
    packed_files = []
    for map_name in aia_paths:
        aia_map = sunpy.map.Map(map_name)
        t_d = [PipeUtils.time_diff_s(aia_map.meta["T_OBS"], time) for time in hmi_times]
        hmi_match = hmi_paths[t_d.index(min(t_d))]
        packed_files.append([map_name, hmi_match])
    return packed_files


def drms_pipeline(starts: list, path: str, keys: list, wavelengths: list, sample: int = 60):
    # Extract and define the record number, start and end time.
    start_t, end_t = starts[1], starts[2]

    # Define the query, find high quality file recnums and export for hmi and aia series.
    hmi_query, hmi_export = DrmsDownload.hmi_query(start_t, end_t, keys, sample)
    aia_query, aia_export = DrmsDownload.aia_query(hmi_query, keys, wavelengths)

    # Use the queries and exports to download data, checking for existing data and producing paths if needed.
    hmi_dls, hmi_exs = DrmsDownload.l1_file_save(hmi_export, hmi_query, path)
    aia_dls, aia_exs = DrmsDownload.l1_file_save(aia_export, aia_query, path)

    return hmi_dls, aia_dls, hmi_exs, aia_exs


# Loads saved l1 HMI, masks, and saves to output l2 dir (creates paths if needed). Will not resave existing files.
def hmi_l2(hmi_path):
    path = load_config()["path"]
    hmi_map = sunpy.map.Map(hmi_path)
    hmi_map = SDOproc.hmi_mask(hmi_map)
    proc_path = SDOproc.l2_file_save(hmi_map, hmi_path, path)
    del hmi_map, path
    return proc_path


# Process AIA files - levelling to 1.5, rescaling and trimming limb. Matches to nearest HMI map in provided hmi series to reproject.
def aia_l2(packed_paths):
    ## This is VERY slow at present, will need to implement multithreading and maybe look at memory usage.
    path = load_config()["path"]
    aia_path, hmi_path = packed_paths[0], packed_paths[1]
    hmi_match = sunpy.map.Map(hmi_path)
    aia_map = sunpy.map.Map(aia_path)
    aia_map = SDOproc.aia_process(aia_map)
    # This step takes a long time. Reprojecting 4x4k maps is expensive.
    aia_map = SDOproc.aia_reproject(aia_map, hmi_match)
    proc_path = SDOproc.l2_file_save(aia_map, aia_path, path)
    del aia_map, hmi_match, path
    return proc_path

    # TO-DO
    # - Create file-list record ie; matching HMI and AIA images. Include some metadata (ie; quality flags etc) +
    # - Update file structure for saving and uploading. Atm this only works on my machine. +
    # - Implement tests (pytest).
    # - Better comments to make methods easier to read.
    # - Explore using query and observation (file) id to construct targeted queries and save JSOC resources/time - 24 queries per run -> 2 queries (1 HMI, 1 AIA). +
    # - query by RECNUM +
    # - How do we see the API working


# Imagining this will work by inputting a list of dates in tuple form w/ AR number ie; [[ARnumber_1, start_1, end_1,...],[ARnumber_2, start_2, end_2,...][]..]
# - Simulating intended usage - list of times and samples for different periods/requirements

if __name__ == "__main__":
    config = load_config()
    keys = ["T_REC", "T_OBS", "QUALITY", "WAVELNTH", "*recnum*", "INSTRUME"]
    starts = [[0, "2017-02-01T01:00:00", "2017-02-01T03:00:00"]]

    for range in starts:
        hmi_dls, aia_dls, hmi_exs, aia_exs = drms_pipeline(
            range, config["path"], keys, config["wavelengths"], config["sample"]
        )
    # This
    with ProcessPoolExecutor(6) as executor:
        hmi_proc = executor.map(hmi_l2, hmi_exs)

        packed_files = match_files(aia_exs, hmi_exs)

        aia_proc = executor.map(aia_l2, packed_files)
