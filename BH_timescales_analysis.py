import h5py
import numpy as np
import scipy
from scipy import stats
import math
import random
import uuid
import hashlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, NullFormatter, ScalarFormatter, FuncFormatter)
import seaborn as sns
import pandas as pd
from plotbin.sauron_colormap import register_sauron_colormap
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import astropy.units as u
from astropy.cosmology import z_at_value, FlatLambdaCDM
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID, ConvertID_noMK, MergerTree
import eagleSqlTools as sql
from graphformat import set_rc_params, lighten_color
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
register_sauron_colormap()
#====================================


#--------------------------------
# Reads in tree and extracts galaxies that meet criteria
ID_list = [108988077, 479647060, 21721896, 390595970, 401467650, 182125463, 192213531, 24276812, 116404995, 239808134, 215988755, 86715463, 6972011, 475772617, 374037507, 429352532, 441434976]
ID_list = [1361598, 1403994, 10421872, 17879310, 21200847, 21659372, 24053428, 182125501, 274449295]
ID_list = [21200847, 182125516, 462956141]
def _extract_tree(csv_tree = 'L100_galaxy_tree_',
                    GalaxyID_list = None,             # [ None / ID_list ]
                    print_summary             = True,
                  #====================================================================================================
                  load_csv_file  = '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph',   # [ 'file_name' / False ] load existing misalignment tree  
                  plot_annotate  = False,                                                  # [ False / 'ETG → ETG' 
                  #====================================================================================================
                  # Mergers
                  # Processed after sample, applied to first index_misaligned rather than entire misalignment
                  use_alt_merger_criteria = False,
                    half_window         = 0.3,      # [ 0.2 / +/-Gyr ] window centred on first misaligned snap to look for mergers
                    min_ratio           = 0.1,   
                    merger_lookback_time = 2,       # Gyr, number of years to check for peak stellar mass
                    
                  use_alt_relaxation_morph   = False,     # False / ['ETG-ETG'],
                  
                  use_alt_min_trelax    = False,    # [ False / Gyr ] sets min trelax post-extraction
                  use_alt_min_tdyn      = False,
                  use_alt_min_ttorque   = False,
                  #====================================================================================================
                  print_progress = False,
                  debug = False):
    
    
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    
    #===================================================================================================
    # Load previous csv if asked for
    """ GUIDE TO misalignment_tree['%s' %GalaxyID] VALUES:

    GalaxyID is first ID in window of misalignment.
    Arrays contain no missing entries

    Mergers searched for from first misaligned [index_m+1], to first relax [index_r] -> for range from relax to relax do [index_m:index_r+1]
	Values specified for general (eg. stelmass) use_hmr_general (typically 2.0)

    'GalaxyID'				-
    'SnapNum'				-
    'Redshift'				-
    'Lookbacktime'			- [ Gyr ]

    'SubGroupNum'			- 
    'halomass'				- [ Msun ]
    'stelmass'				- [ Msun ]
    'stelmass_1hmr'			- [ Msun ]
    'gasmass'				- [ Msun ]
    'gasmass_1hmr'			- [ Msun ]
    'sfmass'				- [ Msun ]
    'sfmass_1hmr'			- [ Msun ]
    'nsfmass'				- [ Msun ]				
    'dmmass'				- [ Msun ] in 30pkpc

    'sfr'					- [ Msun/yr]
    'ssfr'					- [ /yr ]

    'stars_l'               - [ pkpc/kms-1 ]
    
    'stars_Z'				-
    'gas_Z'					-
    'sf_Z'					-	
    'nsf_Z'					-

    'kappa_stars'			-
    'kappa_gas'				-
    'kappa_sf'				-
    'kappa_nsf'				-
    'ellip'					-
    'triax'					-
    'disp_ani'				-
    'disc_to_total'			-
    'rot_to_disp_ratio'		-

    'rad'					- [ pkpc ]
    'radproj'				- [ pkpc ]
    'rad_sf'				- [ pkpc ]

    'vcirc'					- [ km/s ]
    'tdyn'					- [ Gyr ]
    'ttorque'				- [ Gyr ] 


    'inflow_mass_1hmr'			- [ Msun ]
    'inflow_mass_2hmr'			- [ Msun ]
    'outflow_mass_1hmr'			- [ Msun ]
    'outflow_mass_2hmr'			- [ Msun ]
    'insitu_mass_1hmr'			- [ Msun ]
    'insitu_mass_2hmr'			- [ Msun ]
    'inflow_rate_1hmr'			- [ Msun/yr ]
    'inflow_rate_2hmr'			- [ Msun/yr ]
    'outflow_rate_1hmr'			- [ Msun/yr ]
    'outflow_rate_2hmr'			- [ Msun/yr ]
    'stelmassloss_rate_1hmr'	- [ Msun/yr ]
    'stelmassloss_rate_2hmr'	- [ Msun/yr ]
    's_inflow_rate_1hmr'		- [ /yr ]
    's_inflow_rate_2hmr'		- [ /yr ]
    's_outflow_rate_1hmr'		- [ /yr ]
    's_outflow_rate_2hmr'		- [ /yr ]
    'inflow_Z_1hmr'				-
    'inflow_Z_2hmr'				-
    'outflow_Z_1hmr'			-
    'outflow_Z_2hmr'			-
    'insitu_Z_1hmr'				-
    'insitu_Z_2hmr'				-
	
    'inflow_sf_mass_1hmr'			- [ Msun ]
    'inflow_sf_mass_2hmr'			- [ Msun ]
    'outflow_sf_mass_1hmr'			- [ Msun ]
    'outflow_sf_mass_2hmr'			- [ Msun ]
    'insitu_sf_mass_1hmr'			- [ Msun ]
    'insitu_sf_mass_2hmr'			- [ Msun ]
    'inflow_sf_rate_1hmr'			- [ Msun/yr ]
    'inflow_sf_rate_2hmr'			- [ Msun/yr ]
    'outflow_sf_rate_1hmr'			- [ Msun/yr ]
    'outflow_sf_rate_2hmr'			- [ Msun/yr ]
    'stelmassloss_sf_rate_1hmr'	- [ Msun/yr ]
    'stelmassloss_sf_rate_2hmr'	- [ Msun/yr ]
    's_inflow_sf_rate_1hmr'		- [ /yr ]
    's_inflow_sf_rate_2hmr'		- [ /yr ]
    's_outflow_sf_rate_1hmr'		- [ /yr ]
    's_outflow_sf_rate_2hmr'		- [ /yr ]
    'inflow_Z_sf_1hmr'				-
    'inflow_Z_sf_2hmr'				-
    'outflow_Z_sf_1hmr'			-
    'outflow_Z_sf_2hmr'			-
    'insitu_Z_sf_1hmr'				-
    'insitu_Z_sf_2hmr'				-

    'bh_mass'				- [ Msun ]
	'bh_cumlmass'			- [ Msun ]
	'bh_cumlseeds'			- 
    'bh_mdot_av'			- [ Msun/yr ]
    'bh_mdot_inst'			- [ Msun/yr ]
    'bh_edd'				-
    'bh_lbol'				- [ erg/s ]

    'stars_gas_sf'			- all in degrees
    'stars_gas_sf_err'		-
	'stars_gas_sf_halo'		- inner stars vs outer gas_sf
    'stars_dm'				- not checked for inclination
    'stars_dm_err'			-
    'gas_dm'				- not checked for inclination
    'gas_dm_err'			-
    'gas_sf_dm'				- not checked for inclination
    'gas_sf_dm_err'			-

    'merger_ratio_stars'	- Includes ALL mergers in window
    'merger_ratio_gas'		- ^

    # single values:
    'index_s'				- Last before leaving stable regime, adjusted for current window
	'index_m'				- First index to be misaligned by more than 30 degrees
    'index_r'				- First to be relaxed back in stable regime
    'index_merger' 			- Index of mergers that met locations. Use this to sample 'merger_ratio_stars'
    'relaxation_time' 		- [ Gyr ] Time between index_m and index_r
	'relaxation_tdyn'		- [fraction] trelax/tdyn between index_m and index_r
	'relaxation_ttorque'	- [fraction] trelax/ttorque between index_m and index_r
    'relaxation_type'		- co-co, counter-counter, co-counter, counter-co
    'relaxation_morph'		- ETG-ETG, LTG-LTG, ETG-LTG, LTG-ETG, other					BASED OFF MORPH_LIMITS
    'misalignment_morph'	- ETG, LTG, other
    'angle_peak'			- peak misalignment angle from where it relaxes to (-co (0 deg), -counter (180 deg))
    'index_peak'			- index of the above w.r.t array
	
    
    """
    if load_csv_file:
        dict_tree = json.load(open('%s/L100_misalignment_tree_%s.csv' %(output_dir, load_csv_file), 'r'))
        misalignment_input = dict_tree['misalignment_input']
        sample_input       = dict_tree['sample_input']
        output_input       = dict_tree['output_input']
        misalignment_tree  = dict_tree['misalignment_tree']
        
        relaxation_type    = dict_tree['misalignment_input']['relaxation_type']
        relaxation_morph   = dict_tree['misalignment_input']['relaxation_morph']
        misalignment_morph = dict_tree['misalignment_input']['misalignment_morph']
        morph_limits       = dict_tree['misalignment_input']['morph_limits']
        peak_misangle      = dict_tree['misalignment_input']['peak_misangle']
        min_trelax         = dict_tree['misalignment_input']['min_trelax']        
        max_trelax         = dict_tree['misalignment_input']['max_trelax']
        
        savefig_txt = load_csv_file
      
    #------------------------------
    # Extract list of IDs matching criteria
    if GalaxyID_list:
        collect_IDs = []
        for ID_i in misalignment_tree.keys():
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 5:
                collect_IDs.append(ID_i)
        print('-------------------------------------------')
        print('Number of >5 ttorque relaxations:  ', len(collect_IDs))
        print(collect_IDs)
    
    # Apply a morph criteria
    if use_alt_relaxation_morph:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] in use_alt_relaxation_morph:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    
    # Apply a min trelax criteria
    if use_alt_min_trelax:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_time'] > use_alt_min_trelax:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    if use_alt_min_tdyn:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > use_alt_min_tdyn:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    if use_alt_min_ttorque:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > use_alt_min_ttorque:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    
    # Apply a merger criteria
    if use_alt_merger_criteria:
        # Loading mergertree file to establish windows
        f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
        GalaxyID_tree             = np.array(f['Histories']['GalaxyID'])
        DescendantID_tree         = np.array(f['Histories']['DescendantID'])
        Lookbacktime_tree         = np.array(f['Snapnum_Index']['LookbackTime'])
        StellarMass_tree          = np.array(f['Histories']['StellarMass'])
        GasMass_tree              = np.array(f['Histories']['GasMass'])
        f.close()
        
        misalignment_tree_new = {}
        tally_minor = 0
        tally_major = 0
        tally_sample = len(misalignment_tree.keys())
        for ID_i in misalignment_tree.keys():
            """  
            index_merger_window = np.where(np.absolute(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1]) < half_window)[0]
    
            meets_criteria = False
            for merger_i in np.array(misalignment_tree['%s' %ID_i]['merger_ratio_stars'])[index_merger_window]:
                if len(merger_i) > 0:
                    if max(merger_i) > min_ratio:
                        meets_criteria = True
            if meets_criteria == False:
                continue
            else:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
            """
                
            #---------------------------------------------------------
            # Find location of begin of misalignment in merger tree
            (row_i, snap_i) = np.where(GalaxyID_tree == int(misalignment_tree['%s' %ID_i]['GalaxyID'][misalignment_tree['%s' %ID_i]['index_s']+1]))
            row_mask  = row_i[0]
            snap_mask = snap_i[0]
            
            # Find window limits of mergers [SnapNum_merger_min:SnapNum_merger_max]
            SnapNum_merger_min = 1 + np.where(Lookbacktime_tree >= (Lookbacktime_tree[snap_i] + half_window))[0][-1]
            if len(np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0]) > 0:
                SnapNum_merger_max = np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0][0]
            else:
                SnapNum_merger_max = snap_mask
            
            # List of all elligible descendants
            GalaxyID_list       = np.array(GalaxyID_tree)[row_mask, SnapNum_merger_min:SnapNum_merger_max]
            

            merger_ID_array_array    = []
            merger_ratio_array_array = []
            merger_gas_array_array   = []
            for SnapNum_i, GalaxyID_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), GalaxyID_list):
                if int(GalaxyID_i) == -1:
                    continue
                
                merger_mask = [i for i in np.where(np.array(DescendantID_tree)[:,int(SnapNum_i-1)] == GalaxyID_i)[0] if i != row_mask]
                
                # If misalignment found, its position is given by i in merger_mask, SnapNum_i
                merger_ID_array    = []
                merger_ratio_array = []
                merger_gas_array   = []
                if len(merger_mask) > 0:
                    # find peak stelmass of those galaxies
                    for mask_i in merger_mask:
                        # Find last snap up to 2 Gyr ago
                        SnapNum_merger = np.where(Lookbacktime_tree >= (Lookbacktime_tree[SnapNum_i] + merger_lookback_time))[0][-1]
                
                        # Find largest stellar mass of this satellite, per method of Rodriguez-Gomez et al. 2015, Qu et al. 2017 (see crain2017)
                        mass_mask = np.argmax(StellarMass_tree[mask_i][int(SnapNum_merger-100):int(SnapNum_i)]) + (SnapNum_merger-100)
                
                        # Extract secondary properties
                        primary_stelmass   = StellarMass_tree[row_mask][mass_mask]
                        primary_gasmass    = GasMass_tree[row_mask][mass_mask]
                        component_stelmass = StellarMass_tree[mask_i][mass_mask]
                        component_gasmass  = GasMass_tree[mask_i][mass_mask]
                
                        if primary_stelmass <= 0.0:
                            # Adjust stelmass
                            primary_stelmass   = math.nan
                            primary_gasmass    = math.nan
                    
    
                        # Find ratios
                        merger_ratio = component_stelmass / primary_stelmass 
                        if merger_ratio > 1:
                            merger_ratio = 1/merger_ratio
                        gas_ratio    = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)

                        # Append
                        merger_ID_array.append(GalaxyID_tree[mask_i][int(SnapNum_i-1)])
                        merger_ratio_array.append(merger_ratio)
                        merger_gas_array.append(gas_ratio)
                        
                merger_ID_array_array.append(merger_ID_array)
                merger_ratio_array_array.append(merger_ratio_array)
                merger_gas_array_array.append(merger_gas_array)      
            if debug:
                print(misalignment_tree['%s' %ID_i]['SnapNum'])
                print(misalignment_tree['%s' %ID_i]['merger_ratio_stars'])
                for snap_i, star_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), merger_ratio_array_array):
                    print(snap_i, star_i)
                    
                    
            meets_criteria = False
            for merger_i in merger_ratio_array_array:
                if len(merger_i) > 0:
                    if max(merger_i) > min_ratio:
                        meets_criteria = True
            if meets_criteria == False:
                continue
            else:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
                    
                
            merger_count = 0
            for merger_i in merger_ratio_array_array:
                if len(merger_i) > 0:
                    if (max(merger_i) > min_ratio):
                        if merger_count == 0:
                            if 0.3 > max(merger_i) > 0.1:
                                tally_minor += 1
                            if max(merger_i) > 0.3:
                                tally_major += 1
                        merger_count += 1
                        
                        
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
        
        print('======================================')
        print('Using merger criteria, half_window = %.1f Gyr, min_ratio = %.1f' %(half_window, min_ratio))
        print('Original anyMerger sample:   %i\t' %(tally_sample))
        print('        Number of mergers:   %i\t%.2f %%' %(tally_minor+tally_major, (tally_minor+tally_major)*100/tally_sample))
        print('                 ...major:   %i\t%.2f %%' %(tally_major, (tally_major*100/tally_sample)))
        print('                 ...minor:   %i\t%.2f %%' %(tally_minor, (tally_minor*100/tally_sample)))
    
    #------------------------------
    """ # Extract specific misalignment
    for ID_i in misalignment_tree.keys():
        if int(ID_i) in range(251899973, 251899973+80):            
            print('Found misalignment: %s\tLookbacktime: %.2f' %(ID_i, misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_m']]))
            print('\t%s' %misalignment_tree['%s' %ID_i]['relaxation_type'])
            print('\t%.2f Gyr' %misalignment_tree['%s' %ID_i]['relaxation_time'])
            print('\t%.2f trelax/tdyn' %misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            print('\t%.2f trelax/ttorque' %misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            print('\t%s' %misalignment_tree['%s' %ID_i]['relaxation_morph'])
            print('\t%.2f peak offset angle' %misalignment_tree['%s' %ID_i]['angle_peak'])
            print('\t%.2e Stellar mass, %.2e gas mass, %.1f pkpc hmr' %(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1], misalignment_tree['%s' %ID_i]['gasmass'][misalignment_tree['%s' %ID_i]['index_s']+1], misalignment_tree['%s' %ID_i]['rad'][misalignment_tree['%s' %ID_i]['index_s']+1]))
            for i, time_i, snap_i, angle_i in zip(np.arange(0, len(misalignment_tree['%s' %ID_i]['Lookbacktime'])), misalignment_tree['%s' %ID_i]['Lookbacktime'], misalignment_tree['%s' %ID_i]['SnapNum'], misalignment_tree['%s' %ID_i]['stars_gas_sf']):
                print('\t\t%.2f\t%i\t%.1f' %(time_i, snap_i, angle_i), misalignment_tree['%s' %ID_i]['merger_ratio_stars'][i], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][i], misalignment_tree['%s' %ID_i]['sfmass'][i])
            print(' ')
    """
    
    #==================================================================================================
    plt.close()
    # Summary and K-S KS test
    if print_summary:
        summary_dict = {'trelax':       {'array': [],               # relaxation times in [Gyr]
                                         'co-co': [],
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []},     
                        'tdyn':         {'array': [],               # trelax/tdyn multiples
                                         'co-co': [],
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []},     
                        'ttorque':      {'array': [],               # trelax/ttorque multiples
                                         'co-co': [],
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []},    
                        'ID':           {'array': [],               # lists all IDs of galaxy types, for quicker calculations
                                         'co-co': [],               
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []}}     
        

        for ID_i in misalignment_tree.keys():
            summary_dict['trelax']['array'].append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            summary_dict['tdyn']['array'].append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            summary_dict['ttorque']['array'].append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            summary_dict['ID']['array'].append(ID_i)
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                summary_dict['trelax']['co-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['co-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['co-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['co-co'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                summary_dict['trelax']['co-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['co-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['co-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['co-counter'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                summary_dict['trelax']['counter-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['counter-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['counter-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['counter-co'].append(ID_i) 
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                summary_dict['trelax']['counter-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['counter-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['counter-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['counter-counter'].append(ID_i) 
            
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                summary_dict['trelax']['ETG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['ETG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['ETG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['ETG-ETG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                summary_dict['trelax']['LTG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['LTG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['LTG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['LTG-LTG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                summary_dict['trelax']['ETG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['ETG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['ETG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['ETG-LTG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                summary_dict['trelax']['LTG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['LTG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['LTG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['LTG-ETG'].append(ID_i)
                
            if misalignment_tree['%s' %ID_i]['misalignment_morph'] == 'ETG':
                summary_dict['trelax']['ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['ETG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['misalignment_morph'] == 'LTG':
                summary_dict['trelax']['LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['LTG'].append(ID_i)
        

        # Average timescales
        mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
        median_timescale = np.median(np.array(summary_dict['trelax']['array']))
        std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
        # Average tdyn
        mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
        median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
        std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
        # Average ttorque
        mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
        median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
        std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
        
        # KS test on ETG and LTG   
        if (len(summary_dict['trelax']['ETG']) > 0) and (len(summary_dict['trelax']['LTG']) > 0):
            res1_trelax = stats.ks_2samp(summary_dict['trelax']['ETG'], summary_dict['trelax']['LTG'])
        if (len(summary_dict['trelax']['ETG-ETG']) > 0) and (len(summary_dict['trelax']['LTG-LTG']) > 0):
            res2_trelax = stats.ks_2samp(summary_dict['trelax']['ETG-ETG'], summary_dict['trelax']['LTG-LTG'])
        if (len(summary_dict['tdyn']['ETG']) > 0) and (len(summary_dict['tdyn']['LTG']) > 0):
            res1_tdyn = stats.ks_2samp(summary_dict['tdyn']['ETG'], summary_dict['tdyn']['LTG'])
        if (len(summary_dict['tdyn']['ETG-ETG']) > 0) and (len(summary_dict['tdyn']['LTG-LTG']) > 0):
            res2_tdyn = stats.ks_2samp(summary_dict['tdyn']['ETG-ETG'], summary_dict['tdyn']['LTG-LTG'])
        if (len(summary_dict['ttorque']['ETG']) > 0) and (len(summary_dict['ttorque']['LTG']) > 0):
            res1_ttorque = stats.ks_2samp(summary_dict['ttorque']['ETG'], summary_dict['ttorque']['LTG'])
        if (len(summary_dict['ttorque']['ETG-ETG']) > 0) and (len(summary_dict['ttorque']['LTG-LTG']) > 0):
            res2_ttorque = stats.ks_2samp(summary_dict['ttorque']['ETG-ETG'], summary_dict['ttorque']['LTG-LTG'])
            
        
        
        """
        print('\nChecking if ETG_ETG or just ETG is good enough:')
        print('length of ETG array:', len(summary_dict['trelax']['ETG']), len(summary_dict['ID']['ETG']))
        print('length of ETG_ETG array:', len(summary_dict['trelax']['ETG-ETG']), len(summary_dict['ID']['ETG-ETG']))
        print('\tnumber of IDs in common:            ', len([ID_i for ID_i in summary_dict['ID']['ETG'] if ID_i in summary_dict['ID']['ETG-ETG']]))
        print('\tnumber of IDs in ETG but not ETG_ETG', len([ID_i for ID_i in summary_dict['ID']['ETG'] if ID_i not in summary_dict['ID']['ETG-ETG']]))
        print('\tnumber of IDs in ETG_ETG but not ETG', len([ID_i for ID_i in summary_dict['ID']['ETG-ETG'] if ID_i not in summary_dict['ID']['ETG']]))
        
        print('length of LTG array:', len(summary_dict['trelax']['LTG']), len(summary_dict['ID']['LTG']))
        print('length of LTG_LTG array:', len(summary_dict['trelax']['LTG-LTG']), len(summary_dict['ID']['LTG-LTG']))
        print('\tnumber of IDs in common:            ', len([ID_i for ID_i in summary_dict['ID']['LTG'] if ID_i in summary_dict['ID']['LTG-LTG']]))
        print('\tnumber of IDs in LTG but not LTG_LTG', len([ID_i for ID_i in summary_dict['ID']['LTG'] if ID_i not in summary_dict['ID']['LTG-LTG']]))
        print('\tnumber of IDs in LTG_LTG but not LTG', len([ID_i for ID_i in summary_dict['ID']['LTG-LTG'] if ID_i not in summary_dict['ID']['LTG']]))
        """
        
        
        print('\n======================================')
        print('NUMBER OF MISALIGNMENTS RECORDED: ', len(misalignment_tree.keys()))    
        print('   co-co: %s \tcounter-counter: %s \tco-counter: %s \tcounter-co: %s' %(('n/a' if 'co-co' not in relaxation_type else len(summary_dict['ID']['co-co'])), ('n/a' if 'counter-counter' not in relaxation_type else len(summary_dict['ID']['counter-counter'])), ('n/a' if 'co-counter' not in relaxation_type else len(summary_dict['ID']['co-counter'])), ('n/a' if 'counter-co' not in relaxation_type else len(summary_dict['ID']['counter-co']))))
        print('   ETG-ETG: %s \tLTG-LTG: %s \tETG-LTG: %s \t\tLTG-ETG: %s' %(('n/a' if 'ETG-ETG' not in relaxation_morph else len(summary_dict['ID']['ETG-ETG'])), ('n/a' if 'LTG-LTG' not in relaxation_morph else len(summary_dict['ID']['LTG-LTG'])), ('n/a' if 'ETG-LTG' not in relaxation_morph else len(summary_dict['ID']['ETG-LTG'])), ('n/a' if 'LTG-ETG' not in relaxation_morph else len(summary_dict['ID']['LTG-ETG']))))
        print('   ETG: %s \tLTG: %s' %(('n/a' if 'ETG' not in misalignment_morph else len(summary_dict['ID']['ETG'])), ('n/a' if 'LTG' not in misalignment_morph else len(summary_dict['ID']['LTG']))))
        print('======================================')
        """
        print('Long relaxations:')
        print('   trelax > 1 Gyr:\t%i\t%.2f%%   |    trelax > 2 Gyr:\t%i\t%.2f%%' %((np.array(summary_dict['trelax']['array']) > 1).sum(), 100*(np.array(summary_dict['trelax']['array']) > 1).sum()/len(misalignment_tree.keys()), (np.array(summary_dict['trelax']['array']) > 2).sum(), 100*(np.array(summary_dict['trelax']['array']) > 2).sum()/len(misalignment_tree.keys())))
        print('      ETG: %i   LTG: %i\t        | ETG: %i   LTG: %i\t' %((np.array(summary_dict['trelax']['ETG']) > 1).sum(), (np.array(summary_dict['trelax']['LTG']) > 1).sum(), (np.array(summary_dict['trelax']['ETG']) > 2).sum(), (np.array(summary_dict['trelax']['LTG']) > 2).sum()))
        print('  ETG-ETG: %i   LTG-LTG: %i\t\t    | ETG-ETG: %i   LTG-LTG: %i\t' %((np.array(summary_dict['trelax']['ETG-ETG']) > 1).sum(), (np.array(summary_dict['trelax']['LTG-LTG']) > 1).sum(), (np.array(summary_dict['trelax']['ETG-ETG']) > 2).sum(), (np.array(summary_dict['trelax']['LTG-LTG']) > 2).sum()))
        print('  ETG-LTG: %i   LTG-ETG: %i\t\t    | ETG-LTG: %i   LTG-ETG: %i\t' %((np.array(summary_dict['trelax']['ETG-LTG']) > 1).sum(), (np.array(summary_dict['trelax']['LTG-ETG']) > 1).sum(), (np.array(summary_dict['trelax']['ETG-LTG']) > 2).sum(), (np.array(summary_dict['trelax']['LTG-ETG']) > 2).sum()))
        print('   tdyn   > 10   :\t%i\t%.2f%%   |    tdyn   > 20   :\t%i\t%.2f%%' %((np.array(summary_dict['tdyn']['array']) > 10).sum(), 100*(np.array(summary_dict['tdyn']['array']) > 10).sum()/len(misalignment_tree.keys()), (np.array(summary_dict['tdyn']['array']) > 20).sum(), 100*(np.array(summary_dict['tdyn']['array']) > 20).sum()/len(misalignment_tree.keys())))
        print('      ETG: %i   LTG: %i\t\t\t        | ETG: %i   LTG: %i\t' %((np.array(summary_dict['tdyn']['ETG']) > 10).sum(), (np.array(summary_dict['tdyn']['LTG']) > 10).sum(), (np.array(summary_dict['tdyn']['ETG']) > 20).sum(), (np.array(summary_dict['tdyn']['LTG']) > 20).sum()))
        print('  ETG-ETG: %i   LTG-LTG: %i\t\t    | ETG-ETG: %i   LTG-LTG: %i\t' %((np.array(summary_dict['tdyn']['ETG-ETG']) > 10).sum(), (np.array(summary_dict['tdyn']['LTG-LTG']) > 10).sum(), (np.array(summary_dict['tdyn']['ETG-ETG']) > 20).sum(), (np.array(summary_dict['tdyn']['LTG-LTG']) > 20).sum()))
        print('  ETG-LTG: %i   LTG-ETG: %i\t\t    | ETG-LTG: %i   LTG-ETG: %i\t' %((np.array(summary_dict['tdyn']['ETG-LTG']) > 10).sum(), (np.array(summary_dict['tdyn']['LTG-ETG']) > 10).sum(), (np.array(summary_dict['tdyn']['ETG-LTG']) > 20).sum(), (np.array(summary_dict['tdyn']['LTG-ETG']) > 20).sum()))
        print('   torque > 3   :\t%i\t%.2f%%    |    torque > 5   :\t%i\t%.2f%%' %((np.array(summary_dict['ttorque']['array']) > 3).sum(), 100*(np.array(summary_dict['ttorque']['array']) > 3).sum()/len(misalignment_tree.keys()), (np.array(summary_dict['ttorque']['array']) > 5).sum(), 100*(np.array(summary_dict['ttorque']['array']) > 5).sum()/len(misalignment_tree.keys())))
        print('      ETG: %i   LTG: %i\t\t\t        | ETG: %i   LTG: %i\t' %((np.array(summary_dict['ttorque']['ETG']) > 3).sum(), (np.array(summary_dict['ttorque']['LTG']) > 3).sum(), (np.array(summary_dict['ttorque']['ETG']) > 5).sum(), (np.array(summary_dict['ttorque']['LTG']) > 5).sum()))
        print('  ETG-ETG: %i   LTG-LTG: %i\t\t    | ETG-ETG: %i   LTG-LTG: %i\t' %((np.array(summary_dict['ttorque']['ETG-ETG']) > 3).sum(), (np.array(summary_dict['ttorque']['LTG-LTG']) > 3).sum(), (np.array(summary_dict['ttorque']['ETG-ETG']) > 5).sum(), (np.array(summary_dict['ttorque']['LTG-LTG']) > 5).sum()))
        print('  ETG-LTG: %i   LTG-ETG: %i\t\t    | ETG-LTG: %i   LTG-ETG: %i\t' %((np.array(summary_dict['ttorque']['ETG-LTG']) > 3).sum(), (np.array(summary_dict['ttorque']['LTG-ETG']) > 3).sum(), (np.array(summary_dict['ttorque']['ETG-LTG']) > 5).sum(), (np.array(summary_dict['ttorque']['LTG-ETG']) > 5).sum()))
        print(' ')
        print('Relaxation timescales:')
        print('   (Gyr)     all    ETG    LTG  ETG-ETG  LTG-LTG  ETG-LTG  LTG-ETG')
        print('   Mean:    %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['ETG']) == 0 else np.mean(np.array(summary_dict['trelax']['ETG']))), (math.nan if len(summary_dict['trelax']['LTG']) == 0 else np.mean(np.array(summary_dict['trelax']['LTG']))), (math.nan if len(summary_dict['trelax']['ETG-ETG']) == 0 else np.mean(np.array(summary_dict['trelax']['ETG-ETG']))), (math.nan if len(summary_dict['trelax']['LTG-LTG']) == 0 else np.mean(np.array(summary_dict['trelax']['LTG-LTG']))), (math.nan if len(summary_dict['trelax']['ETG-LTG']) == 0 else np.mean(np.array(summary_dict['trelax']['ETG-LTG']))), (math.nan if len(summary_dict['trelax']['LTG-ETG']) == 0 else np.mean(np.array(summary_dict['trelax']['LTG-ETG'])))))   
        print('   Median:  %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['ETG']) == 0 else np.median(np.array(summary_dict['trelax']['ETG']))), (math.nan if len(summary_dict['trelax']['LTG']) == 0 else np.median(np.array(summary_dict['trelax']['LTG']))), (math.nan if len(summary_dict['trelax']['ETG-ETG']) == 0 else np.median(np.array(summary_dict['trelax']['ETG-ETG']))), (math.nan if len(summary_dict['trelax']['LTG-LTG']) == 0 else np.median(np.array(summary_dict['trelax']['LTG-LTG']))), (math.nan if len(summary_dict['trelax']['ETG-LTG']) == 0 else np.median(np.array(summary_dict['trelax']['ETG-LTG']))), (math.nan if len(summary_dict['trelax']['LTG-ETG']) == 0 else np.median(np.array(summary_dict['trelax']['LTG-ETG'])))))   
        print('   std:     %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['ETG']) == 0 else np.std(np.array(summary_dict['trelax']['ETG']))), (math.nan if len(summary_dict['trelax']['LTG']) == 0 else np.std(np.array(summary_dict['trelax']['LTG']))), (math.nan if len(summary_dict['trelax']['ETG-ETG']) == 0 else np.std(np.array(summary_dict['trelax']['ETG-ETG']))), (math.nan if len(summary_dict['trelax']['LTG-LTG']) == 0 else np.std(np.array(summary_dict['trelax']['LTG-LTG']))), (math.nan if len(summary_dict['trelax']['ETG-LTG']) == 0 else np.std(np.array(summary_dict['trelax']['ETG-LTG']))), (math.nan if len(summary_dict['trelax']['LTG-ETG']) == 0 else np.std(np.array(summary_dict['trelax']['LTG-ETG'])))))
        if (len(summary_dict['ID']['ETG']) > 0) and (len(summary_dict['ID']['LTG']) > 0):
            print('K-S TEST FOR ETG and LTG general:    %s %s' %(len(summary_dict['ID']['ETG']), len(summary_dict['ID']['LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res1_trelax.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG']) + len(summary_dict['ID']['LTG']))/(len(summary_dict['ID']['ETG'])*len(summary_dict['ID']['LTG']))))))
            print('   p-value: %s' %res1_trelax.pvalue)
        if (len(summary_dict['ID']['ETG-ETG']) > 0) and (len(summary_dict['ID']['LTG-LTG']) > 0):
            print('K-S TEST FOR ETG-ETG and LTG-LTG general:    %s %s' %(len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res2_trelax.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG-ETG']) + len(summary_dict['ID']['LTG-LTG']))/(len(summary_dict['ID']['ETG-ETG'])*len(summary_dict['ID']['LTG-LTG']))))))
            print('   p-value: %s' %res2_trelax.pvalue)
        print(' ')
        print('trelax/tdyn multiples:')
        print('             all    ETG    LTG  ETG-ETG  LTG-LTG  ETG-LTG  LTG-ETG')
        print('   Mean:    %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['ETG']) == 0 else np.mean(np.array(summary_dict['tdyn']['ETG']))), (math.nan if len(summary_dict['tdyn']['LTG']) == 0 else np.mean(np.array(summary_dict['tdyn']['LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-ETG']) == 0 else np.mean(np.array(summary_dict['tdyn']['ETG-ETG']))), (math.nan if len(summary_dict['tdyn']['LTG-LTG']) == 0 else np.mean(np.array(summary_dict['tdyn']['LTG-LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-LTG']) == 0 else np.mean(np.array(summary_dict['tdyn']['ETG-LTG']))), (math.nan if len(summary_dict['tdyn']['LTG-ETG']) == 0 else np.mean(np.array(summary_dict['tdyn']['LTG-ETG'])))))   
        print('   Median:  %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['ETG']) == 0 else np.median(np.array(summary_dict['tdyn']['ETG']))), (math.nan if len(summary_dict['tdyn']['LTG']) == 0 else np.median(np.array(summary_dict['tdyn']['LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-ETG']) == 0 else np.median(np.array(summary_dict['tdyn']['ETG-ETG']))), (math.nan if len(summary_dict['tdyn']['LTG-LTG']) == 0 else np.median(np.array(summary_dict['tdyn']['LTG-LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-LTG']) == 0 else np.median(np.array(summary_dict['tdyn']['ETG-LTG']))), (math.nan if len(summary_dict['tdyn']['LTG-ETG']) == 0 else np.median(np.array(summary_dict['tdyn']['LTG-ETG'])))))   
        print('   std:     %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(std_tdyn, (math.nan if len(summary_dict['tdyn']['ETG']) == 0 else np.std(np.array(summary_dict['tdyn']['ETG']))), (math.nan if len(summary_dict['tdyn']['LTG']) == 0 else np.std(np.array(summary_dict['tdyn']['LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-ETG']) == 0 else np.std(np.array(summary_dict['tdyn']['ETG-ETG']))), (math.nan if len(summary_dict['tdyn']['LTG-LTG']) == 0 else np.std(np.array(summary_dict['tdyn']['LTG-LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-LTG']) == 0 else np.std(np.array(summary_dict['tdyn']['ETG-LTG']))), (math.nan if len(summary_dict['tdyn']['LTG-ETG']) == 0 else np.std(np.array(summary_dict['tdyn']['LTG-ETG'])))))
        if (len(summary_dict['ID']['ETG']) > 0) and (len(summary_dict['ID']['LTG']) > 0):
            print('K-S TEST FOR ETG and LTG general:    %s %s' %(len(summary_dict['ID']['ETG']), len(summary_dict['ID']['LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res1_tdyn.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG']) + len(summary_dict['ID']['LTG']))/(len(summary_dict['ID']['ETG'])*len(summary_dict['ID']['LTG']))))))
            print('   p-value: %s' %res1_tdyn.pvalue)
        if (len(summary_dict['ID']['ETG-ETG']) > 0) and (len(summary_dict['ID']['LTG-LTG']) > 0):
            print('K-S TEST FOR ETG-ETG and LTG-LTG general:    %s %s' %(len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res2_tdyn.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG-ETG']) + len(summary_dict['ID']['LTG-LTG']))/(len(summary_dict['ID']['ETG-ETG'])*len(summary_dict['ID']['LTG-LTG']))))))
            print('   p-value: %s' %res2_tdyn.pvalue)
        print(' ')    
        print('trelax/ttorque multiples:')
        print('             all    ETG    LTG  ETG-ETG  LTG-LTG  ETG-LTG  LTG-ETG')
        print('   Mean:    %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['ETG']) == 0 else np.mean(np.array(summary_dict['ttorque']['ETG']))), (math.nan if len(summary_dict['ttorque']['LTG']) == 0 else np.mean(np.array(summary_dict['ttorque']['LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-ETG']) == 0 else np.mean(np.array(summary_dict['ttorque']['ETG-ETG']))), (math.nan if len(summary_dict['ttorque']['LTG-LTG']) == 0 else np.mean(np.array(summary_dict['ttorque']['LTG-LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-LTG']) == 0 else np.mean(np.array(summary_dict['ttorque']['ETG-LTG']))), (math.nan if len(summary_dict['ttorque']['LTG-ETG']) == 0 else np.mean(np.array(summary_dict['ttorque']['LTG-ETG'])))))   
        print('   Median:  %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['ETG']) == 0 else np.median(np.array(summary_dict['ttorque']['ETG']))), (math.nan if len(summary_dict['ttorque']['LTG']) == 0 else np.median(np.array(summary_dict['ttorque']['LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-ETG']) == 0 else np.median(np.array(summary_dict['ttorque']['ETG-ETG']))), (math.nan if len(summary_dict['ttorque']['LTG-LTG']) == 0 else np.median(np.array(summary_dict['ttorque']['LTG-LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-LTG']) == 0 else np.median(np.array(summary_dict['ttorque']['ETG-LTG']))), (math.nan if len(summary_dict['ttorque']['LTG-ETG']) == 0 else np.median(np.array(summary_dict['ttorque']['LTG-ETG'])))))   
        print('   std:     %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['ETG']) == 0 else np.std(np.array(summary_dict['ttorque']['ETG']))), (math.nan if len(summary_dict['ttorque']['LTG']) == 0 else np.std(np.array(summary_dict['ttorque']['LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-ETG']) == 0 else np.std(np.array(summary_dict['ttorque']['ETG-ETG']))), (math.nan if len(summary_dict['ttorque']['LTG-LTG']) == 0 else np.std(np.array(summary_dict['ttorque']['LTG-LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-LTG']) == 0 else np.std(np.array(summary_dict['ttorque']['ETG-LTG']))), (math.nan if len(summary_dict['ttorque']['LTG-ETG']) == 0 else np.std(np.array(summary_dict['ttorque']['LTG-ETG'])))))
        if (len(summary_dict['ID']['ETG']) > 0) and (len(summary_dict['ID']['LTG']) > 0):
            print('K-S TEST FOR ETG and LTG general:    %s %s' %(len(summary_dict['ID']['ETG']), len(summary_dict['ID']['LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res1_ttorque.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG']) + len(summary_dict['ID']['LTG']))/(len(summary_dict['ID']['ETG'])*len(summary_dict['ID']['LTG']))))))
            print('   p-value: %s' %res1_ttorque.pvalue)
        if (len(summary_dict['ID']['ETG-ETG']) > 0) and (len(summary_dict['ID']['LTG-LTG']) > 0):
            print('K-S TEST FOR ETG-ETG and LTG-LTG general:    %s %s' %(len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res2_ttorque.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG-ETG']) + len(summary_dict['ID']['LTG-LTG']))/(len(summary_dict['ID']['ETG-ETG'])*len(summary_dict['ID']['LTG-LTG']))))))
            print('   p-value: %s' %res2_ttorque.pvalue)
    print('======================================')
    """
    if len(misalignment_tree.keys()) == 0:
        print('Insufficient galaxies meeting criteria')
        quit()
    
    
    return misalignment_tree, misalignment_input, summary_dict




#BH code:

#We want a way to overlay the BH growths of misaligned galaxies, over that of non-misaligned and counter-rotators

# CREATE BH TREE:
#- import galaxy tree
#- import misalignment tree, 0 latency time
#	- extract selection criteria
#	- Option to include/exclude/any mergers
#	- Needs minimum trelax to consider (0.5 Gyr?)
#	- Set window to consider (1-2 Gyr?)
#	- Additional inputs of:
#		- Average sSFR, SFR, stellar mass, BH mass, kappa, kappa_pre/post
#	- Go through misalignment tree, find all misalignments meeting min trelax and other inputs
#	- Ensure BH mass is always growing
#	- Extract sample from misalignment tree, extract all BH values + cumlmass to a misaligned_BH_dict + average values
#- start search in galaxy tree for galaxies that:
#	- meet basic selection criteria over min trelax
#	- are NOT misaligned / are COUNTER ROTATING for entire period
#	- meet average stellar mass, SFR, sSFR, BH mass, Kappa, kappa_pre/post, as single value
#	- Check mergers
#	- Extract all BH values to aligned_BH_dict and counter_BH_dict + average values
#- Create a new tree to save, as BH_tree{'misaligned': {}, 'aligned': {}, 'counter': {}} + inputs

## ANALYSE TREE:
#- import BH_tree
#- Should now have a sample of misaligned, aligned, and counter-rotating galaxies with all BH growth values
#	- Overlay all BH growths
#	- Trim to minimum trelax, so that even a 1 Gyr misalignment we only have the first 0.5 Gyr of it
#	- plot medians
#	- Randomly select X for control to plot
#	- Do we see a difference in value?
	

#do we need a max trelax too?
#issue being that misalignment_tree will give us different lengths of time - some only 0.5 Gyr, some 1 Gyr
#taking from misalignment_tree.... extract based on min trelax, but then keep extracting beyond that to a minimum of 1 gyr
#alternatively... see final line of analyse tree
#... depends a lot on how big our misalignment sample is that we get to work with here






#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Set starting parameters
load_csv_file_in = '_20Thresh_30Peak_noLatency'
plot_annotate_in                                           = False
savefig_txt_in   = load_csv_file_in               # [ 'manual' / load_csv_file_in ] 'manual' will prompt txt before saving

#  False 
# 'ETG → ETG' , 'LTG → LTG'
#  r'ETG ($\bar{\kappa}_{\mathrm{co}}^{\mathrm{*}} < 0.35$)'
# '$t_{\mathrm{relax}}>3\bar{t}_{\mathrm{torque}}$'    



#==================================================================================================================================
misalignment_tree, misalignment_input, summary_dict = _extract_tree(load_csv_file=load_csv_file_in, plot_annotate=plot_annotate_in)
#==================================================================================================================================
if load_csv_file_in == '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010':
    print('\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\t\tUSING 1010 SAMPLE\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')


tally = 0
for ID_i in misalignment_tree.keys():
    if misalignment_tree['%s' %ID_i]['relaxation_time'] > 0.6:
        tally += 1
print('number of >0.6 Gyr relaxations: ', tally)

    
