# Define required libraries - check to see if Arccnet already has these as requirements.
import sunpy.map
from sunpy.net import Fido, attrs as a
import astropy as ap
import astropy.units as u
from aiapy.psf import deconvolve
from aiapy.calibrate import register, update_pointing
# from aiapy.calibrate import get_pointing_table
from aiapy.calibrate import correct_degradation
import glob
import numpy as np
from numpy import char
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
# import tqdm

class SDODownload():
	# Caching will probably be a good idea - file keeping track of which fits are available.
	# global fetch_request
	def aia_fetch_request(
		req,
		euv_path:str,
		tmpstorage:bool = False,
		):
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
				files = Fido.fetch(req, max_conn= 10, path = f'{npe}{path}')
				filepaths.append(files)
		return np.unique(filepaths)
	
	def hmi_fetch_request(
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
				files = Fido.fetch(req, max_conn= 10, path =f'{npe}{hmi_path}')
				filepaths.append(files)
		return np.unique(filepaths)
		
	def aia_ts_request(
		start:str,
		end:str,
		wavelengths:list,):

		# Convert to Time Delta
		time_d = a.Time(start,end)

		# Pass argument to Fido (May switch to DRMS when refactoring/updating python versions)
		res = []
		for wvl in wavelengths:
			res.append(Fido.search(time_d, a.Wavelength(wvl*u.AA), a.Instrument('AIA'), a.Sample(1*u.hr)))
		return res

		# Fetch Fido response.
		# npe = ''
		# if tmpstorage:
		# 	npe = '/temp/'
		# for result, wvl in zip(res, wavelengths):
		# 	fetch_request(result, path = f'{npe}path_euv' + f'/{wvl}')
		

	def hmi_ts_request(
		start:str,
		end:str,
		verif:str,
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
		res.append(Fido.search(time_d, a.jsoc.Series('hmi.m_720s'), a.Sample(1*u.hr), a.jsoc.Notify(verif)))
		return res
		
		# files = Fido.fetch(res[0], max_conn= 10, path = path_hmi)
		# 	while files.errors != []:
		# 		files = Fido.fetch(res[0], max_conn= 10, path = path_hmi)
		# else:
		# 	res.append(Fido.search(time_d, a.jsoc.Series('hmi.m_720s'), a.Sample(1*u.hr), a.jsoc.Notify(verif)))
		# 	files = Fido.fetch(res[0], max_conn= 10, path = path_hmi + f'/temp')
		# 	while files.errors != []:
		# 		files = Fido.fetch(res[0], max_conn= 10, path = path_hmi + f'/temp')
				



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

	global aia_prep, aia_deconv, aia_expnorm, aia_degcorr, aia_reproject

	def aia_process(aia_map, 
			 deconv:bool = False, 
			 degcorr:bool = False,
			 exnorm:bool = True):
		aia_maps = []
		for aiamap in aia_maps:
			if deconv:
				aia_map = aia_deconv(aiamap)
			aia_map = aia_prep(aiamap)
			if degcorr:
				aia_map = aia_degcorr(aiamap)
			if exnorm:
				aia_map = aia_expnorm(aiamap) 
			aia_maps.append(aia_map)
			return aia_maps
	
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

	def aia_reproject(aia_map, hmi_map):
		# Reprojects AIA map to HMI map using wcs.
		# The maps need to be in similar time frames.
		aia_map= aia_map.reproject_to(hmi_map.wcs)
		return aia_map
		
	def hmi_mask(
	# Finds the coordinates of pixels outside of Rsun_obs for an HMI image and assigns NaN.
		hmimap
			):
		hpc_coords = all_coordinates_from_map(hmimap)
		mask = coordinate_is_on_solar_disk(hpc_coords)
		hmidata = hmimap.data
		hmidata[mask == False] = np.nan
		hmimap = sunpy.map.Map(hmidata, hmimap.meta)
		return hmimap