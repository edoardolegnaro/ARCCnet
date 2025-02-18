from SDOprocessing import SDOproc, SDODownload
import sunpy.map
import glob
import numpy as np

# Define parameters - This will later be handled by accepting variables from flare timelines or cli
time_1 = '2016-01-01T00:00:00'
time_2 = '2016-01-01T01:00:00'
wavelengths = [171,193,211]
rep_tol = 12
path_euv = '/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/data_generation/test_files/euv'
path_hmi = '/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/data_generation/test_files/hmi'

# Get series requests
print('Creating AIA and HMI series requests.')
aia_request = SDODownload.aia_ts_request(time_1,time_2,wavelengths)
hmi_request = SDODownload.hmi_ts_request(time_1,time_2, 'danielgass192@gmail.com')

# Collect file paths
paths_aia = []
# AIA files
for wvl in wavelengths:
	filelist = glob.glob(f'{path_euv}{wvl}/*')
	paths_aia.append(filelist)
# HMI files
paths_hmi = glob.glob(f'{path_hmi}*')

# Check exiting files for duplicates, modifying request if needed
# print('Checking for redundancy in data.')
# hmi_request = SDODownload.return_missing_files(hmi_request, paths_hmi)
# aia_request = SDODownload.return_missing_files(aia_request, paths_aia)
# break1
# Start series download
print('Downloading data and retrieving paths.')
aia_dls = SDODownload.aia_fetch_request(aia_request, path_euv)
hmi_dls = SDODownload.hmi_fetch_request(hmi_request, path_hmi)
# # Collect filepaths of downloaded files
# new_aia_files = SDODownload.dl_paths(aia_request, paths_aia)
# new_hmi_files = SDODownload.dl_paths(hmi_request, paths_hmi)

# Check quality
print('Checking quality of downloaded data.')
bad_files_aia = SDODownload.aia_quality_check(aia_dls)
# TO-DO: Need to do quality check on Hex values for HMI.
bad_files_hmi = SDODownload.hmi_quality_check(hmi_dls)

# Redownload nearest if bad_files has filenames - Currently placeholder
if bad_files_aia != []:
	print('Bad data detected in AIA download! Reacquiring close match.')
	SDODownload.aia_replace(bad_files_aia, rep_tol)
if bad_files_hmi != []:
	print('Bad data detected in HMI download! Reacquiring close match.')
	SDODownload.hmi_replace(bad_files_hmi, rep_tol)

# Process AIA + HMI
print('Processing data.')
# print(list(aia_dls))
aia_maps = sunpy.map.Map(list(aia_dls))
hmi_maps = sunpy.map.Map(list(hmi_dls))
aia_maps = SDOproc.aia_process(aia_maps)
hmi_maps = SDOproc.hmi_mask(hmi_maps)
aia_maps = SDOproc.aia_reproject(aia_maps, hmi_maps)

# TO-DO: 
# Output files (Need Output Directories)
# print('Writing processed data.')
# aia_maps.save('{wvl}map_{index:03}.fits')
# hmi_maps.save('hmi_map_{index:03}.fits')




