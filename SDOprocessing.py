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

client = drms.Client()
# Replace this with whatever email(s) we want to use for this purpose.
os.environ["JSOC_EMAIL"] = 'danielgass192@gmail.com'

# Not all functions used, will need to refactor if we decide to keep/remove FIDO pipeline.
class pipeutils():

	# Contains utility functions to help with time/string handling and list comparisons with strings.
	
	def time_parse(start:str, end:str):
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
	
	def change_time(time:str, shift:int):
		time_d = sunpy.time.parse_time(time).tai_seconds + shift
		time_d = Time(time_d, format='tai_seconds').isot
		return time_d
	
	def comp_list(file:str, file_list:list):
		return any(file in name for name in file_list)
	
class drmsDownload():

	# This downloads hmi files starting with queries to produce indices of good, missing files.
	# Method collects indexes of bad files for use later in patching.
	def drms_hmi_download(start:str, end:str, path:str, sample:str, index:list = []):
		dur = pipeutils.time_delta(start, end)
		print(dur)
		keys = ["T_REC", "T_OBS", "QUALITY"]
		hmi_filelist = glob.glob(f'{path}/*.fits')
		qstr = f"hmi.M_720s[{start}/{dur}h@{sample}m]"+"{magnetogram}"

		request = client.query(qstr, key=keys)
		bad_result = request[request.QUALITY != 0]
		ind_result = request[request.QUALITY == 0].index
		result = client.export(qstr, method="url", protocol='fits', email=os.environ["JSOC_EMAIL"])
		result_urls = result.urls['filename']
		# query and export sometimes produce different lengths - not a huge fan of dropping the last one but it should be okay.
		ind_result = ind_result[ind_result <= max(result_urls.index)]
		result_urls_filtered = result_urls[ind_result]
		comparison_list = [not pipeutils.comp_list(time, hmi_filelist) for time in result_urls_filtered]
		final_ind = result_urls_filtered[comparison_list].index
		if len(index) > 0:
				final_ind = np.intersect1d(final_ind, index)
		downloads = []
		if final_ind.size > 0:
			print(f'Downloading - HMI')
			downloads = result.download(directory=f'./data_generation/test_files/hmi/', index = final_ind)
		else:
			print('HMI files already present - skipping download.')
		return downloads, final_ind, bad_result
	
	# This function works per wavelength to find good files via query and download via export as above.
	def drms_aia_download(wvl:list, start:str, end:str,  path:str, sample:int, indices:list = []):
		dur = pipeutils.time_delta(start, end)
		hmi_ind = indices
		results = []
		bad_results = []
		keys = ["T_REC", "T_OBS", "QUALITY", "WAVELNTH"]

		for wv in wvl:

			aia_filelist = glob.glob(f'{path}/{wv}/*.fits')
			qstr = f"aia.lev1_euv_12s[{start}/{dur}h@{sample}m]["+str(wv)+"]{image}"

			request = client.query(qstr, key=keys)
			bad_result = request[request.QUALITY != 0]
			ind_result = request[request.QUALITY == 0].index
			result = client.export(qstr, method="url", protocol='fits', email=os.environ["JSOC_EMAIL"])
			result_urls = result.urls['filename']
			ind_result = ind_result[ind_result <= max(result_urls.index)]
			result_urls_filtered = result_urls[ind_result]
			comparison_list = [not pipeutils.comp_list(time, aia_filelist) for time in result_urls_filtered]
			final_ind = result_urls_filtered[comparison_list].index
			if len(hmi_ind) > 0:
				final_ind = np.intersect1d(final_ind, hmi_ind)
			if len(bad_result) > 0:
				bad_results.append(bad_result)
			downloads = []
			if len(final_ind) > 0:
				print(f'Downloading AIA - {wv} Å')
				downloads = result.download(directory=f'./data_generation/test_files/euv/{wv}/', index = final_ind)
			else:
				print(f'AIA {wv} Å files already present - skipping download.')
				results.append(downloads)
			
		return downloads, bad_results
	
	## This part is a bit baroque/slow - may want to refactor at some point.
	# 
	def drms_aia_patcher(start, end, result, path):
		bad_results = []
		downloads = []
		for channel in result:
			wv = list(channel['WAVELNTH'])[0]
			print(f'Attempting to replace {len(channel)} bad/missing {wv} Å record(s) in range.')
			new_start, new_end = pipeutils.change_time(start, 12), pipeutils.change_time(end, 12)
			download, bad_aia = drmsDownload.drms_aia_download([wv], new_start, new_end, path, 60, channel.index)
			bad_results.append(bad_aia)
			downloads.append(download)
		return download, sum(bad_results, [])

	# This patches hmi and fetches corresponding AIA files for all specified wavelengths.
	def drms_sdo_patcher(start, end, result, path_hmi, wvl, path_euv):
		print(f'Attempting to replace {len(result)} bad/missing HMI record(s) in range.')
		new_start, new_end = pipeutils.change_time(start, 720), pipeutils.change_time(end, 720)
		p_hmi_dls, p_hmi_ind, p_hmi_bad = drmsDownload.drms_hmi_download(new_start, new_end, path_hmi, 60, result.index)
		
		# Find matching AIA images for recollected HMI magnetograms
		if len(p_hmi_ind) > 0:
			p_aia_dls, p_aia_bad = drmsDownload.drms_aia_download(wvl, new_start, new_end, path_euv, 60, p_hmi_ind)
			if len(p_aia_bad) > 0:
				drmsDownload.drms_aia_patcher(p_aia_dls, path_euv)
		return p_hmi_dls, p_hmi_ind, p_hmi_bad
	
# This class is currently not used - Fido having some issues with reliable downloads, but retained code for possible future use.
class FidoDownload():
	# Caching will probably be a good idea - file keeping track of which fits are available.
	# global fetch_request
	global aia_drms_qual_check, hmi_drms_qual_check
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

	def fido_hmi_ts_request(start:str, end:str, verif:str, time:int, tmpstorage:bool = True,):

		time_d = a.Time(start,end)
		res = []
		nph = ''
		res.append(Fido.search(time_d, a.jsoc.Series('hmi.m_720s'), a.Sample(time*u.min), a.jsoc.Notify(verif)))
		return res
	
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

	def hmi_quality_check(
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

	# Levels an AIA map to level 1.5, can also correct for degradation and deconvolve psf if needed.
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
		aiad = aia_map.data/aia_map.exposure_time
		aia_map = sunpy.map.Map(aiad.astype(int), aia_map.fits_header)
		return aia_map
	
	# Corrects degradation.
	# May want to get our own correction tables, but downloads each time this is called at present.
	def aia_degcorr(aia_map):
		aia_map = correct_degradation(aia_map)
		return aia_map

	# Reprojects AIA map to HMI map using wcs.
	# The maps need to be in closest possible time frames.
	def aia_reproject(aia_maps, hmi_maps):
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
		
	# Finds the coordinates of pixels outside of Rsun_obs for an HMI image and assigns NaN.
	def hmi_mask(
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