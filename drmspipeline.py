import glob

import sunpy.map
from numpy import char

from astropy.io.fits import CompImageHDU

from SDOprocessing import SDOproc, drmsDownload, pipeutils

# import multiprocessing.pool

# Define parameters - This will later be handled by accepting times from ARs of interest.
time_1 = "2013-01-01T00:00:00"
time_2 = "2013-01-03T00:00:00"
wavelengths = [171, 193]
rep_tol = 5
sample = 60

# Replace this with path we want to use.
path_aia = "/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/data_generation/test_files/euv"
path_hmi = "/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/data_generation/test_files/hmi"

# Begin HMI download - this is needed for all image pairings.
hmi_dls, hmi_ind, hmi_bad = drmsDownload.drms_hmi_download(time_1, time_2, path_hmi, sample)
print(hmi_bad)

# Begin AIA download - hmi_ind used to align AIA downloads with valid HMI files.
aia_dls, aia_bad = drmsDownload.drms_aia_download(wavelengths, time_1, time_2, path_aia, sample, hmi_ind)

# Patch missing aia files from first run.
p_aia_dls, p_aia_bad = drmsDownload.drms_aia_patcher(time_1, time_2, aia_bad, path_aia)

# This repeats the patch attempt until the returned bad downloads list contains only empty values, or until rep_tol is exceeded.
retries = 0
if len(p_aia_bad) > 0:
    new_start, new_end = pipeutils.change_time(time_1, 12), pipeutils.change_time(time_2, 12)
    while max([len(rs) != 0 for rs in p_aia_bad]) and retries < rep_tol:
        print(f"Attempt - {retries + 1}")
        p_aia_dls, p_aia_bad = drmsDownload.drms_aia_patcher(time_1, time_2, p_aia_bad, path_aia)
        new_start, new_end = pipeutils.change_time(new_start, 12), pipeutils.change_time(new_end, 12)
        retries += 1

# Redownload bad/missing HMI and fetch accompanying AIA files.
p_hmi_dls, p_hmi_ind, p_hmi_bad = drmsDownload.drms_sdo_patcher(
    time_1, time_2, hmi_bad, path_hmi, wavelengths, path_aia
)

# Attempts to redownload missing HMI files from the first patch attempt - repeats until no bad files or rep_tol is exceeded
retries = 0
print(p_hmi_bad)
if len(p_hmi_bad) > 0:
    new_start, new_end = pipeutils.change_time(time_1, 720), pipeutils.change_time(time_2, 720)
    while len(p_hmi_bad) > 0 and retries < rep_tol:
        print(f"Attempt - {retries + 1}")
        p_hmi_dls, p_hmi_ind, p_hmi_bad = drmsDownload.drms_sdo_patcher(
            new_start, new_end, p_hmi_bad, path_hmi, wavelengths, path_aia
        )
        print(p_hmi_bad)
        new_start, new_end = pipeutils.change_time(new_start, 720), pipeutils.change_time(new_end, 720)
        retries += 1

print("Downloads Complete. Processing files.")

## Could possibly speed this up with multiprocessing. Should be 3/4 x faster with 4 core utilization.
# Process HMI files / apply mask beyond limb.
hmi_filelist = sorted(glob.glob(f"{path_hmi}/*.fits"))
print("Loading HMI")
hmi_maps = sunpy.map.Map(hmi_filelist)
print("Masking HMI")
hmi_maps = SDOproc.hmi_mask(hmi_maps)
print("Saving HMI")
hmi_save = "/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/data_generation/test_files/proc/hmi"
for map in hmi_maps:
    wvl = "hmi_720s"
    time = map.meta["t_obs"]
    # print(time)
    time = char.translate(time, str.maketrans("", "", ".:"))
    map.save(f"{hmi_save}/{wvl}_{time}.fits", hdu_type=CompImageHDU, overwrite=True)

## This is VERY slow at present, will need to implement multithreading and maybe look at memory usage.
# Process AIA files - levelling to 1.5, rescaling and trimming limb.
print("Processing AIA")
for wvl in wavelengths:
    aia_filelist = sorted(glob.glob(f"{path_aia}/{wvl}/*.fits"))
    print("Loading maps")
    aia_maps = sunpy.map.Map(list(aia_filelist))
    print("Processing maps")
    aia_maps = SDOproc.aia_process(aia_maps)
    # This step takes a long time. Reprojecting 40-60 4x4k maps is expensive.
    print("Reprojecting maps")
    aia_maps = SDOproc.aia_reproject(aia_maps, hmi_maps)
    print("Saving AIA fits.")
    aia_save = "/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/data_generation/test_files/proc/aia/"
    for map in aia_maps:
        wvl = map.meta["wavelnth"]
        time = map.meta["t_obs"]
        # print(str(time).strip(':'))
        time = char.translate(time, str.maketrans("", "", ".:"))
        map.save(f"{aia_save}/{wvl}/{time}.fits", hdu_type=CompImageHDU, overwrite=True)

# TO-DO
# - Create file-list record ie; matching HMI and AIA images. Include data on
# - Update file structure for saving and uploading. Atm this only works on my machine.
# - Implement tests (pytest).
# - Better comments to make methods easier to read.
# - Explore using query and observation (file) id to construct targeted queries and save JSOC resources/time - 24 queries per run -> 2 queries (1 HMI, 1 AIA).
