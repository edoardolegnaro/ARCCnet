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
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
# import tqdm

class SDODownload():
	# Replace this with where you want your files to go.
	path_euv = '/Users/danielgass/Desktop/ARCCnet/arccnet/data_generation/test_files/euv'
	path_hmi = '/Users/danielgass/Desktop/ARCCnet/arccnet/data_generation/test_files/hmi'

	# Caching will probably be a good idea - file keeping track of which fits are available.
	global fetch_request

	def fetch_request(
		req,
		path: str,
		):
		files = Fido.fetch(req, max_conn = 10, path = path)
		# 8. Create loop to ensure all files downloaded/were redownloaded as necessary
		while files.errors != []:
			files = Fido.fetch(req, max_conn= 10, path = path)
		return files
		
	def aia_ts_ingestion(
		start:str,
		end:str,
		wavelengths:list,
		tmpstorage:bool = False):

		# Convert to Time Delta
		time_d = a.Time(start,end)

		# Pass argument to Fido (Will switch to DRMS when refactoring/updating python versions)
		res = []
		for wvl in wavelengths:
			res.append(Fido.search(time_d, a.Wavelength(wvl*u.AA), a.Instrument('AIA'), a.Sample(1*u.min)))
		# 3. Receive Fido response
		# 4. TO-DO Compare response to requested/cached files (if any)
		# 5. TO-DO Determine nearest match to requested files within indicated tolerance (default 15 mins)
		# 6. TO-DO Remove duplicate requests.

		# Fetch Fido response.
		npe = ''
		if tmpstorage:
			npe = '/temp/'
		for result, wvl in zip(res, wavelengths):
			fetch_request(result, path = f'{npe}path_euv' + f'/{wvl}')
		

	def hmi_ts_ingestion(
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
		if tmpstorage:
			nph = '/temp/'
		new_path_hmi = f'{nph}path_hmi'
			# for prod in products:
		res.append(Fido.search(time_d, a.jsoc.Series('hmi.m_720s'), a.Sample(1*u.hr), a.jsoc.Notify(verif)))
		fetch_request(res, path = new_path_hmi)
		
		# files = Fido.fetch(res[0], max_conn= 10, path = path_hmi)
		# 	while files.errors != []:
		# 		files = Fido.fetch(res[0], max_conn= 10, path = path_hmi)
		# else:
		# 	res.append(Fido.search(time_d, a.jsoc.Series('hmi.m_720s'), a.Sample(1*u.hr), a.jsoc.Notify(verif)))
		# 	files = Fido.fetch(res[0], max_conn= 10, path = path_hmi + f'/temp')
		# 	while files.errors != []:
		# 		files = Fido.fetch(res[0], max_conn= 10, path = path_hmi + f'/temp')
				

	## ADD HMI QUALITY CHECK SEPARATELY - HEX HANDLING NEEDED

	def hmi_quality_check(
		# 9. Filter .fits for QUALITY flag, remove certain images, and check for nearest file to replace.
		# Checks provided map for list of approved QUALITY flag values.
		# Works for both AIA and HMI maps, and map cubes.
		# Returns the header.name of bad files.
		map,
		accepted:list = [0]):
		bad_list = []
		if type(map) != list:
			map = [map]
		for m in map:
			if m.meta['quality'] in accepted == False:
				bad_list.append(m.name)
		status_list = f'Quality check complete - {len(bad_list)}/{len(map)} map(s) failed.'
		print(status_list)
		return (bad_list)
				
	def aia_quality_check(
		# 9. Filter .fits for QUALITY flag, remove certain images, and check for nearest file to replace.
		# Checks provided map for list of approved QUALITY flag values.
		# Works for both AIA and HMI maps, and map cubes.
		# Returns the header.name of bad files.
		map,
		accepted:list = [0]):
		bad_list = []
		if type(map) != list:
			map = [map]
		for m in map:
			if m.meta['quality'] in accepted == False:
				bad_list.append(m.name)
		status_list = f'Quality check complete - {len(bad_list)}/{len(map)} map(s) failed.'
		print(status_list)
		return (bad_list)
	
	def update_paths_aia():
		return

	def update_paths_hmi():
		return

class AIAproc():

	global aia_prep, aia_deconv, aia_expnorm, aia_degcorr, aia_reproject

	def aia_process(aia_map, 
			 deconv:bool = False, 
			 degcorr:bool = False,
			 exnorm:bool = True):
		if deconv:
			aia_map = aia_deconv(aia_map)
		aia_map = aia_prep(aia_map)
		if degcorr:
			aia_map = aia_degcorr(aia_map)
		if exnorm:
			aia_map = aia_expnorm(aia_map) 
		return aia_map
	
	def aia_prep(aia_map):
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
		# Corrects degradation according to native aiapy routines.
		# May want to get our own correction tables, but downloads each time this is called at present.
		aia_map = correct_degradation(aia_map)
		return aia_map

	def aia_reproject(aia_map, hmi_map):
		# The maps need to be in similar time frames
		aia_map= aia_map.reproject_to(hmi_map.wcs)
		return aia_map
		
class HMIproc():
	# Define Time Range and Wavelengths/HMI components
	time_1 = '2014-01-01T00:00:00'
	time_2 = '2014-01-01T01:00:00'
	path_hmi = '/Users/danielgass/Desktop/ARCCnet/arccnet/data_generation/test_files/hmi'
	products = [''] 
	def hmi_ts_ingestion(
			start:str,
			end:str,
			# products:list,
			permstorage:bool = True,):
		
		# 1. Convert date range to Time Delta
		# 2. Define required data product(s) 
		# 3. Pass argument to Fido (Will switch to DRMS when refactoring/updating python versions)
		# 4. Fetch argument.
		# 5. Check for failed downloads and reattempt if needed.

		time_d = a.Time(start,end)
		res = []
		if permstorage:
			# for prod in products:
			res.append(Fido.search(time_d, a.Instrument('HMI'), a.Sample(1*u.min)))
			files = Fido.fetch(res[0], max_conn= 10, path = path_hmi)
			while files.errors != []:
				files = Fido.fetch(res[0], max_conn= 10, path = path_hmi)
		else:
			res.append(Fido.search(time_d, a.Instrument('HMI'), a.Sample(1*u.min)))
			files = Fido.fetch(res[0], max_conn= 10, path = path_hmi + f'/temp')
			while files.errors != []:
				files = Fido.fetch(res[0], max_conn= 10, path = path_hmi + f'/temp')
	
	def hmi_mask(
	# Calculates the coordinate pixels outside of Rsun_obs for HMI image and clips them.
		hmimap
			):
		hpc_coords = all_coordinates_from_map(hmimap)
		mask = coordinate_is_on_solar_disk(hpc_coords)
		hmidata = hmimap.data
		hmidata[mask == False] = np.nan
		hmimap = sunpy.map.Map(hmidata, hmimap.meta)
		return hmimap