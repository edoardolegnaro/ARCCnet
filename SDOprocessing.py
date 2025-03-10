# Define required libraries - check to see if Arccnet already has these as requirements.
import sunpy.map
from sunpy.net import Fido, attrs as a
import astropy.units as u
from aiapy.psf import deconvolve
from aiapy.calibrate import register, update_pointing
from aiapy.calibrate import correct_degradation
import glob
import numpy as np
from numpy import char
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
import os
import math
import drms
from astropy.time import Time

# import tqdm

client = drms.Client()
os.environ["JSOC_EMAIL"] = 'danielgass192@gmail.com'

class SDODownload():
	# Caching will probably be a good idea - file keeping track of which fits are available.
	# global fetch_request
	global time_parse, time_delta, aia_drms_qual_check, hmi_drms_qual_check
	def fido_aia_fetch_request(req, euv_path:str, tmpstorage:bool = False,):
		npe = ''
		filepaths = []
		if tmpstorage:
			npe = '/temp/'
		for result in req:
			wvl = int(min(result[0][0]['Wavelength'].value))
			path = f'{euv_path}/{wvl}'
			files = Fido.fetch(result, max_conn = 10, path =f'{npe}{path}')
		# 8. Create loop to ensure all files downloaded/were redownloaded as necessary
			filepaths.append(files)
			while files.errors != []:
				files = Fido.fetch(result, max_conn= 10, path = f'{npe}{path}')
				filepaths.append(files)
				# filepaths = np.reshape(filepaths, (-1,1))
		return filepaths
	
	def fido_hmi_fetch_request(
		req,
		hmi_path:str,
		tmpstorage:bool = False,
		):
		npe = ''
		filepaths = []
		if tmpstorage:
			npe = '/temp/'
		for result in req:
			files = Fido.fetch(result, max_conn = 10, path =f'{npe}{hmi_path}')
		# 8. Create loop to ensure all files downloaded/were redownloaded as necessary
			filepaths.append(files)
			while files.errors != []:
				files = Fido.fetch(result, max_conn= 10, path =f'{npe}{hmi_path}')
				filepaths = filepaths.append(files)
		return np.unique(filepaths)
		
	def fido_aia_ts_request(
		start:str,
		end:str,
		wavelengths:list,
		time:int):

		# Convert to Time Delta
		time_d = a.Time(start,end)

		# Pass argument to Fido (May switch to DRMS when refactoring/updating python versions)
		res = []
		for wvl in wavelengths:
			res.append(Fido.search(time_d, a.Wavelength(wvl*u.AA), a.Instrument('AIA'), a.Sample(time*u.min)))
		return res

		# Fetch Fido response.
		# npe = ''
		# if tmpstorage:
		# 	npe = '/temp/'
		# for result, wvl in zip(res, wavelengths):
		# 	fetch_request(result, path = f'{npe}path_euv' + f'/{wvl}')
	
	def fido_hmi_ts_request(
	start:str,
	end:str,
	verif:str,
	time:int,
	tmpstorage:bool = True,):

	# 1. Convert date range to Time Delta
	# 2. Define required data product(s) 
	# 3. Pass argument to Fido (Will switch to DRMS when refactoring/updating python versions)
	# 4. Fetch argument.
	# 5. Check for failed downloads and reattempt if needed.

		time_d = a.Time(start,end)
		res = []
		nph = ''
		# if tmpstorage: 
		# 	nph = '/temp/'
		# new_path_hmi = f'{nph}path_hmi'
			# for prod in products:
		res.append(Fido.search(time_d, a.jsoc.Series('hmi.m_720s'), a.Sample(time*u.min), a.jsoc.Notify(verif)))
		return res

	def time_parse(
		start:str,
		end:str):
		start_datetime = sunpy.time.parse_time(start)
		end_datetime = sunpy.time.parse_time(end)
		t_diff = end_datetime.tai_seconds - start_datetime.tai_seconds
		time_diff_days = math.ceil(t_diff / (24 * 3600))
		return(time_diff_days)
	
	def time_delta(start:str, end:str):
		start_t = sunpy.time.parse_time(start)
		end_t = sunpy.time.parse_time(end)
		t_diff = end_t.tai_seconds - start_t.tai_seconds
		# time_diff_days = math.ceil(t_diff / (24 * 3600))
		time_diff_hours = math.ceil(t_diff / (3600))
		return time_diff_hours
	
	def time_shift(time:str, shift:int):
		time_d = sunpy.time.parse_time(time).tai_seconds + shift
		time_d = Time(time_d, format='tai_seconds').isot
		return time_d
	
	def aia_drms_qual_check(qstr:str, keys:list):
		bad_results = []
		request = client.query(qstr, key=keys)
		bad_result = request[request.QUALITY != 0]['T_OBS']
		if len(bad_result > 0):
			bad_results.append(bad_result)
		result = request[request.QUALITY == 0]
		good_results = result.index

		return good_results, bad_results.to_list()
	
	def hmi_drms_qual_check(qstr:str, keys:list):
		bad_results = []
		request = client.query(qstr, key=keys)
		bad_result = request[request.QUALITY != 0]['T_OBS']
		if len(bad_result > 0):
			bad_results.append(bad_result)
		result = request[request.QUALITY == 0]
		good_results = result.index

		return good_results, bad_results.to_list()

	def aia_drms_download(
		start:str,
		end:str,
		wavelengths:list,
		sample:int,
		retry_dl:bool = True):

		results = []
		t_range = time_parse(start, end)
		keys = ["T_OBS", "QUALITY", "WAVELNTH",]
		for wv in wavelengths:
			qstr = f"aia.lev1_euv_12s[{start}/{t_range}d@{sample}m]["+str(wv)+"]{image}"
			result_ind, bad_results = aia_drms_qual_check(qstr, keys)
			result = client.export(qstr, method="url", protocol='fits', email=os.environ["JSOC_EMAIL"])
			print(f'Downloading')
			downloads = result.download(directory=f'./data_generation/test_files/euv/{wv}/', index = result_ind)
			if bad_results != [] and retry_dl:
				for brs in bad_results:
					next_time = sunpy.time.parse_time(brs.to_list()).tai_seconds + 12
					next_time = Time(next_time, format='tai_seconds').isot
				results.append(downloads)

		return results

	def drms_hmi_download(start:str, end:str, path:str, sample:str):
		dur = time_delta(start, end)
		hmi_filelist = glob.glob(f'{path}/*.fits')
		qstr = f"hmi.M_720s[{start}/{dur}h@{sample}m]"+"{magnetogram}"
		request = client.query(qstr, key=keys)
		bad_result = request[request.QUALITY != 0]
		ind_result = request[request.QUALITY == 0].index
		result = client.export(qstr, method="url", protocol='fits', email=os.environ["JSOC_EMAIL"])
		result_urls = result.urls['filename']
		comparison_list = [comp_list(time, hmi_filelist) for time in result_urls]
		dl_indices = np.where(np.array(comparison_list) == False)[0]

		if dl_indices.size != 0:
			print(f'Downloading - HMI')
			downloads = result.download(directory=f'./data_generation/test_files/hmi/', index = dl_indices)
		else:
			print('All files already present - skipping download.')
		return downloads, dl_indices, bad_result

	def hmi_quality_check(
		# TO-DO - HMI QUALITY CHECK - HEX HANDLING NEEDED
		# 9. Filter .fits for QUALITY flag, remove certain images, and check for nearest file to replace.
		# Checks provided map for list of approved QUALITY flag values.
		# Returns the header.name of bad files.
		map,
		accepted:list = [0]):
		bad_list = []
		# if type(map) != list:
		# 	map = [map]
		for m in map:
			m_map = sunpy.map.Map(m)
			if m_map.meta['quality'] in accepted == False:
				bad_list.append(m_map.name)
		# status_list = f'Quality check complete - {len(bad_list)}/{len(map)} map(s) failed.'
		# print(status_list)
		return (bad_list)
				
	def aia_quality_check(
		# 9. Filter .fits for QUALITY flag, remove certain images, and check for nearest file to replace.
		# Checks provided map for list of approved QUALITY flag values.
		# Returns the header.name of bad files.
		map,
		accepted:list = [0]):
		bad_list = []
		# if type(map) != list:
		# 	map = [map]
		for m in map:
			m_map = sunpy.map.Map(m)
			if m_map.meta['quality'] in accepted == False:
				bad_list.append(m_map.name)
		# status_list = f'Quality check complete - {len(bad_list)}/{len(map)} map(s) failed.'
		# print(status_list)
		return (bad_list)
	
	def dl_paths(fin_result, filelist):
		# Function to match filenames to a UnifiedResponse function, used after download for quality checks.
		timestamp = np.array(fin_result['T_REC'])
		timestamp = char.translate(timestamp,str.maketrans('', '', '.:'))
		matching_files = [file for file in filelist if any(time in file for time in timestamp)]
		return matching_files

	def return_missing_files(result, filelist):
		# Reads in a UnifiedResponse and a list of files and returns modified response containing only missing entries if any.
		# fileset = set(np.array(filelist))
		result = np.reshape(result, -1)
		timestamp = np.array(result['T_REC'])
		timestamp2 = char.translate(timestamp,str.maketrans('', '', '.:'))
		missing_timestamps = [name for name in timestamp2 if not any(name in file for file in filelist)]
		missing_indices = np.where(np.isin(timestamp2, missing_timestamps))[0]
		res_new = result[missing_indices]
		return res_new

class SDOproc():

	global aia_prep, aia_deconv, aia_expnorm, aia_degcorr

	def aia_process(aia_map, 
			 deconv:bool = False, 
			 degcorr:bool = False,
			 exnorm:bool = True):
		map_list = []
		for map in aia_map:
			if deconv:
				aia_map = aia_deconv(map)
			aia_map = aia_prep(map)
			if degcorr:
				aia_map = aia_degcorr(map)
			if exnorm:
				aia_map = aia_expnorm(map) 
			map_list.append(aia_map)
		return map_list
	
	def aia_prep(aia_map):
		# Prepares aia map to level 1.5, standard aiapy process.
		# pointing_table = get_pointing_table("JSOC", start = (aia_map.date - 12 * u.h), end = (aia_map.date + 12 * u.h))
		aia_map = update_pointing(aia_map)
		aia_map = register(aia_map)
		return aia_map
	
	def aia_deconv(aia_map):
		# assuming default parameters for deconvolve function for now
		aia_map = deconvolve(aia_map)
		return aia_map
	
	def aia_expnorm(aia_map):
		# Corrects for exposure time, has to cast to int and repackage fits file.
		aiad = aia_map.data/aia_map.exposure_time
		aia_map = sunpy.map.Map(aiad.astype(int), aia_map.fits_header)
		return aia_map
	
	def aia_degcorr(aia_map):
		# Corrects degradation.
		# May want to get our own correction tables, but downloads each time this is called at present.
		aia_map = correct_degradation(aia_map)
		return aia_map

	def aia_reproject(aia_maps, hmi_maps):
		# Reprojects AIA map to HMI map using wcs.
		# The maps need to be in closest possible time frames.

		hmi_times = []
		for hmi in hmi_maps:
			hmi_times.append(hmi.meta['t_obs'])

		new_aia_maps = []
		for aia_map in aia_maps:
			rpr_aia_map = aia_map.reproject_to(hmi_maps[0].wcs)
			rpr_aia_map.meta['wavelnth'] = aia_map.meta['wavelnth']
			rpr_aia_map.meta['t_obs'] = aia_map.meta['t_obs']
			new_aia_maps.append(rpr_aia_map)
		return new_aia_maps
		
	def hmi_mask(
	# Finds the coordinates of pixels outside of Rsun_obs for an HMI image and assigns NaN.
		hmimap
			):
		hmimaps = []
		for map in hmimap:
			hpc_coords = all_coordinates_from_map(map)
			mask = coordinate_is_on_solar_disk(hpc_coords)
			hmidata = map.data
			hmidata[mask == False] = np.nan
			hmimaps.append(sunpy.map.Map(hmidata, map.meta))
		return hmimaps