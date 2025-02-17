from SDOprocessing import AIAproc, HMIproc, SDODownload
import sunpy.map
import glob

# Define parameters
time_1 = '2015-01-01T00:00:00'
time_2 = '2015-01-01T01:00:00'
wavelengths = [171,193,211]
path_euv = '/Users/danielgass/Desktop/ARCCnet/arccnet/data_generation/test_files/euv'
path_hmi = '/Users/danielgass/Desktop/ARCCnet/arccnet/data_generation/test_files/hmi'

# Start series download
SDODownload.aia_ts_ingestion(time_1,time_2,wavelengths)
SDODownload.hmi_ts_ingestion(time_1,time_2)

# Collect new file paths
paths_aia, paths_hmi = [],[]
for wvl in wavelengths:
	filelist = glob.glob(f'/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/arccnet/data_generation/test_files/euv/{wvl}/*')
	paths_aia.append = filelist
paths_hmi = glob.glob(f'/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/arccnet/data_generation/test_files/hmi/*')

# Check quality
bad_files_aia = SDODownload.aia_quality_check(paths_aia)

# TO-DO: Need to do quality check on Hex values for HMI.
bad_files_hmi = SDODownload.hmi_quality_check(paths_hmi)

# Redownload nearest if bad_files has filenames
if bad_files_aia != []:
	SDODownload.update_paths_aia()
if bad_files_hmi != []:
	SDODownload.update_paths_hmi()

# Process AIA + HMI
aia_maps = sunpy.map.Map(paths_aia)
hmi_maps = sunpy.map.Map(paths_hmi)
aia_maps = AIAproc.aia_process(aia_maps)
hmi_maps = HMIproc.hmi_mask(hmi_maps)

# Output files (Need Output Directories)
# TO-DO: 



