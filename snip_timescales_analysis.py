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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, NullFormatter)
import seaborn as sns
import pandas as pd
from plotbin.sauron_colormap import register_sauron_colormap
from matplotlib.ticker import PercentFormatter
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
    'gasmass'				- [ Msun ]
    'sfmass'				- [ Msun ]
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

    'bh_mass'				- [ Msun ]
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
        print(' ')
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
    
    if len(misalignment_tree.keys()) == 0:
        print('Insufficient galaxies meeting criteria')
        quit()
    
    
    return misalignment_tree, misalignment_input, summary_dict
                  
                               
#-------------------------
# Plot sample histogram of misalignments extracted              
def _plot_sample_hist(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None,
                      #==============================================
                      # General formatting
                        set_bin_width_mass                  = 0.01,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
    
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax']
    #-------------------------
    
    
    #==========================================================================
    # Gather data, average stelmass over misalignment
    stelmass_plot = []
    for ID_i in misalignment_tree.keys():
        stelmass_plot.append(np.log10(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    axs.hist(stelmass_plot, bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), histtype='bar', edgecolor='none', facecolor='b', alpha=0.1)
    axs.hist(stelmass_plot, bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), histtype='bar', edgecolor='b', facecolor='none', alpha=1.0)
    
    print('Median stellar mass: \t%.2e.' %(10**np.median(stelmass_plot)))
    
    #-------------
    ### Formatting
    axs.set_xlabel(r'log$_{10}$ M$_{*}$ ($2r_{50}$) [M$_{\odot}$]')
    axs.set_ylabel('Galaxies in sample')
    axs.set_xticks(np.arange(9, 12.1, 0.5))
    axs.set_xlim(9, 12.1)
    
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': 'MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%ssample_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%ssample_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plot sample histogram of misalignments extracted       
def _plot_timescale_histogram(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_trelax                = 5,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.25,     # [ 0.25 / Gyr ]
                      set_thist_ymax_trelax               = 0.45,             # 0.45 / 500  yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_percentage               = True,
                      set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_inset                       = True,     # whether to have smaller second plot
                        add_inset_bestfit               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    #-------------------------
                      
                      
    #========================================================         
    # Gather data
    relaxationtime_plot = []
    co_co_array           = []
    co_counter_array      = []
    counter_co_array      = []
    counter_counter_array = []
    collect_array         = []
    for ID_i in misalignment_tree.keys():
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            if misalignment_tree['%s' %ID_i]['angle_peak'] > 135:
                collect_array.append(ID_i)
    
    #print('-------------------------------------------------------------')
    #print('Number of >135 co-co misalignments: ', len(collect_array))
    #print(collect_array)
    print('  Using sample: ', len(relaxationtime_plot))
    print('\nMax trelax:  %.2f Gyr' %max(relaxationtime_plot))  
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if set_plot_relaxation_type:
        if set_plot_percentage:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        else:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
    else:
        if set_plot_percentage:
            axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        else:
            axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
    
    #-------------
    ### Inset second axes
    if add_inset:
        axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
        
        if set_plot_percentage:
            axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        else: 
            axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
            
            
        #----------
        # Formatting
        axins.set_yscale('log')
        axins.set_xlim(0, set_bin_limit_trelax)
        axins.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axins.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)', fontsize = 5)
        if set_plot_percentage:
            axins.set_ylim(0.0002, set_thist_ymax_trelax)
            axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
            axins.set_ylabel('Percentage of misalignments', fontsize=5)
        else:
            axins.set_ylim(0.5, set_thist_ymax_trelax)
            axins.set_ylabel('Number of misalignments', fontsize=5)
            
            
        #----------
        # Number of values in bin
        hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        print('bin_count of hist:', bin_count)
        
        #-----------
        # Add uncertainty
        #print('bin_count of hist:', bin_count)
        #print('hist_n of hist:', hist_n)
        #print('poisson uncertainty:', np.sqrt(hist_n))
        
        
        # Add poisson errors to each bin (sqrt N)
        axins.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor='k', ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.5)
        
        # Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        xdata = np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax)[mask]
        ydata = (hist_n/np.sum(hist_n))[mask]
        yerr  = (np.sqrt(hist_n)/np.sum(hist_n))[mask]
        
        ydata_log = np.log10(ydata)
        yerr_log  = np.log10(ydata + yerr) - np.log10(ydata)
        
        
        # Define linear bestfit
        def cal_func(x,c,m):
            return m*x + c
            
        popt, pcov = scipy.optimize.curve_fit(cal_func, xdata[1:], ydata_log[1:], sigma=yerr_log[1:], absolute_sigma=True)
        intercept = popt[0]
        slope = popt[1]
            
        """# Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
        """
        
        print('Best fit line:     frac = %.2f x 10^(%.2f t)' %((10**intercept), slope))
        print('             log10 frac = %.2f t + %.2f ' %(slope, intercept))
            
        if add_inset_bestfit:
            axins.plot([xdata[1], 8], [(10**intercept) * (10**(slope*xdata[1])), (10**intercept) * (10**(slope*8))], lw=0.7, ls='--', alpha=1, c='purple', label='best-fit')
            #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
    
    
    #-----------
    ### General formatting
    # Axis labels
    if set_plot_histogram_log:
        axs.set_yscale('log')
    if set_plot_percentage:
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
    if set_plot_percentage:
        axs.set_ylabel('Percentage of misalignments')
    else:
        axs.set_ylabel('Number of misalignments')
    axs.set_ylim(0, set_thist_ymax_trelax)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    # add z
    #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('k')
    
    if set_plot_relaxation_type:
        if 'co-co' in relaxation_type:
            legend_labels.append('     co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
    if add_inset:
        ncol=2
    else:
        ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
    metadata_plot = {'Author': 'MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr\nmax: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale, max(relaxationtime_plot)),
                     'Producer': str(hist_n)}
                     
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_inset' if add_inset else '') + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%stime_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%stime_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn histogram    
def _plot_tdyn_histogram(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_tdyn                  = 32,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 1,        # [ multiples ]
                      set_thist_ymax_tdyn                 = 0.35,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_percentage               = True,
                      set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_inset                       = True,     # whether to have smaller second plot
                        add_inset_bestfit               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax']    
    use_angle          = misalignment_input['use_angle']
    #-------------------------       
            
    # Gather data
    relaxationtime_plot = []
    co_co_array           = []
    co_counter_array      = []
    counter_co_array      = []
    counter_counter_array = []
    for ID_i in misalignment_tree.keys():
        # append average tdyn over misalignment
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 20:
            print('\tFound >20 tdyn:   ID: %s\tMass: %.2e Msun' %(ID_i, np.mean(misalignment_tree['%s' %ID_i]['stelmass'])))
            print('\t  type: %s, morph: %s, time: %.2f, tdyn: %.2f, ttorque: %.2f' %(misalignment_tree['%s' %ID_i]['relaxation_type'], misalignment_tree['%s' %ID_i]['relaxation_morph'], misalignment_tree['%s' %ID_i]['relaxation_time'], misalignment_tree['%s' %ID_i]['relaxation_tdyn'], misalignment_tree['%s' %ID_i]['relaxation_ttorque']))
    
    print('  Using sample: ', len(relaxationtime_plot))    
    print('\nMax tdyn/trelax:  %.2f' %max(relaxationtime_plot)) 
            
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if set_plot_relaxation_type:
        if set_plot_percentage:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        else:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
    else:
        if set_plot_percentage:
            axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn+0.5*set_bin_width_tdyn, set_bin_width_tdyn), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        else:
            axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn+0.5*set_bin_width_tdyn, set_bin_width_tdyn), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
    
    #-------------
    ### Inset second axes
    if add_inset:
        axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
        
        if set_plot_percentage:
            axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        else:
            axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        
                
        #-----------
        # Formatting
        axins.set_yscale('log')
        axins.set_xlim(0, set_bin_limit_tdyn)
        axins.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
        axins.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$', fontsize=5)
        
        if set_plot_percentage:
            axins.set_ylim(0.0002, set_thist_ymax_tdyn)
            axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
            axins.set_ylabel('Percentage of misalignments', fontsize=5)
        else:
            axins.set_ylim(0.5, set_thist_ymax_tdyn)
            axins.set_ylabel('Number of misalignments', fontsize=5)
        
        #----------
        # Number of values in bin
        hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        print('bin_count of hist:', bin_count)
        
        
        #-----------
        # Add uncertainty
        #print('bin_count of hist:', bin_count)
        #print('hist_n of hist:', hist_n)
        #print('poisson uncertainty:', np.sqrt(hist_n))
        
        # Add poisson errors to each bin (sqrt N)
        axins.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor='k', ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.5)
        
        
        #-----------
        # Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        xdata = np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn)[mask]
        ydata = (hist_n/np.sum(hist_n))[mask]
        yerr  = (np.sqrt(hist_n)/np.sum(hist_n))[mask]
        
        ydata_log = np.log10(ydata)
        yerr_log  = np.log10(ydata + yerr) - np.log10(ydata)
        
        
        # Define linear bestfit
        def cal_func(x,c,m):
            return m*x + c
            
        popt, pcov = scipy.optimize.curve_fit(cal_func, xdata[1:], ydata_log[1:], sigma=yerr_log[1:], absolute_sigma=True)
        intercept = popt[0]
        slope = popt[1]
            
        """# Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
        """
        
        print('Best fit line:     frac = %.2f x 10^(%.2f t)' %((10**intercept), slope))
        print('             log10 frac = %.2f t + %.2f ' %(slope, intercept))
            
        if add_inset_bestfit:
            axins.plot([xdata[1], 100], [(10**intercept) * (10**(slope*xdata[1])), (10**intercept) * (10**(slope*100))], lw=0.7, ls='--', alpha=1, c='purple', label='best-fit')
            #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
        
        
        
    #-----------
    ### General formatting
    # Axis labels
    if set_plot_histogram_log:
        axs.set_yscale('log')
    if set_plot_percentage:
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    if set_plot_percentage:
        axs.set_ylabel('Percentage of misalignments')
    else:
        axs.set_ylabel('Number of misalignments')
    axs.set_ylim(0, set_thist_ymax_tdyn)

    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    # add z
    #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('k')
            
    if set_plot_relaxation_type:
        if 'co-co' in relaxation_type:
            legend_labels.append('     co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
    if add_inset:
        ncol=2
    else:
        ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
        
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f\nmax: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn, max(relaxationtime_plot)),
                     'Producer': str(hist_n)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_inset' if add_inset else '') + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%stdyn_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%stdyn_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque histogram    
def _plot_ttorque_histogram(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      set_bin_limit_ttorque               = 12,       # [ None / multiples ]
                      set_bin_width_ttorque               = 0.5,      # [ multiples ]
                      set_thist_ymax_ttorque              = 0.35,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_percentage               = True,
                      set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_inset                       = True,     # whether to have smaller second plot
                        add_inset_bestfit               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    #-------------------------   


    #==================================================================
    # Gather data
    relaxationtime_plot = []
    co_co_array           = []
    co_counter_array      = []
    counter_co_array      = []
    counter_counter_array = []
    for ID_i in misalignment_tree.keys():
        # append average ttorque over misalignment
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        
    print('  Using sample: ', len(relaxationtime_plot))
    print('\nMax tdyn/ttorque:  %.2f' %max(relaxationtime_plot)) 
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if set_plot_relaxation_type:
        if set_plot_percentage:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        else:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
    else:
        if set_plot_percentage:
            axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque+0.5*set_bin_width_ttorque, set_bin_width_ttorque), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        else:
            axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque+0.5*set_bin_width_ttorque, set_bin_width_ttorque), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
    
    
    #-------------
    ### Inset second axes
    if add_inset:
        axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
        
        if set_plot_percentage:
            axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        else:
            axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        
        
        #-----------
        # Formatting
        axins.set_yscale('log')
        axins.set_xlim(0, set_bin_limit_ttorque)
        axins.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
        axins.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$', fontsize=5)
        if set_plot_percentage:
            axins.set_ylim(0.0002, set_thist_ymax_ttorque)
            axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
            axins.set_ylabel('Percentage of misalignments', fontsize=5)
        else:
            axins.set_ylim(0.5, set_thist_ymax_ttorque)
            axins.set_ylabel('Percentage of misalignments', fontsize=5)
        
        
        #----------
        # Number of values in bin
        hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        print('bin_count of hist:', bin_count)
            
        #-----------
        # Add uncertainty
        #print('bin_count of hist:', bin_count)
        #print('hist_n of hist:', hist_n)
        #print('poisson uncertainty:', np.sqrt(hist_n))
    
        # Add poisson errors to each bin (sqrt N)
        axins.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor='k', ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.5)
    
    
        #-----------
        # Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        xdata = np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque)[mask]
        ydata = (hist_n/np.sum(hist_n))[mask]
        yerr  = (np.sqrt(hist_n)/np.sum(hist_n))[mask]
    
        ydata_log = np.log10(ydata)
        yerr_log  = np.log10(ydata + yerr) - np.log10(ydata)
    
    
        # Define linear bestfit
        def cal_func(x,c,m):
            return m*x + c
        
        popt, pcov = scipy.optimize.curve_fit(cal_func, xdata[1:], ydata_log[1:], sigma=yerr_log[1:], absolute_sigma=True)
        intercept = popt[0]
        slope = popt[1]
        
        """# Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
        """
    
        print('Best fit line:     frac = %.2f x 10^(%.2f t)' %((10**intercept), slope))
        print('             log10 frac = %.2f t + %.2f ' %(slope, intercept))
        
        if add_inset_bestfit:
            axins.plot([xdata[1], 100], [(10**intercept) * (10**(slope*xdata[1])), (10**intercept) * (10**(slope*100))], lw=0.7, ls='--', alpha=1, c='purple', label='best-fit')
            #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor') 
            
        
    #-----------
    ### General formatting
    # Axis labels
    if set_plot_histogram_log:
        axs.set_yscale('log')
    if set_plot_percentage:
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    if set_plot_percentage:
        axs.set_ylabel('Percentage of misalignments')
    else:
        axs.set_ylabel('Number of misalignments')
    axs.set_ylim(0, set_thist_ymax_ttorque)

    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    # add z
    #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('k')
            
    if set_plot_relaxation_type:
        if 'co-co' in relaxation_type:
            legend_labels.append('     co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
    if add_inset:
        ncol=2
    else:
        ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
        
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f\nmax: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque, max(relaxationtime_plot)),
                     'Producer': str(hist_n)}
                     
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_inset' if add_inset else '') + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%sttorque_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%sttorque_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Stacked single 1x1 graphs
def _plot_stacked_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_trelax                = 5,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.25,     # [ 0.25 / Gyr ]
                      set_thist_ymax_trelax               = 0.45,             # 0.45 / 500  yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    time_collect = []
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
            
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        if set_plot_type == 'time':
            timeaxis_plot = -1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']])
        elif set_plot_type == 'raw_time':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - (0)
        elif set_plot_type == 'snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - misalignment_tree['%s' %ID_i]['SnapNum'][misalignment_tree['%s' %ID_i]['index_s']]
        elif set_plot_type == 'raw_snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - (0)
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_trelax:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1]+0.120)
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_time'] > 2:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.02)
            c     = lighten_color(line_color, (misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            
            
            
        #axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
        axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c='k', alpha=1)
        
        ### Annotate
        if set_add_GalaxyIDs:
            axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    #print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    #print('  List of >2 Gyr trelax ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 5.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 5.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 5.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 5.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                 
    #-----------
    # Add threshold
    axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #ßfor ID_i in ID_plot:
    #    print(' ')
    #    print(' ID: ', ID_i)
    #    for time_i, angle_i, merger_i in zip(misalignment_tree['%s' %ID_i]['Lookbacktime'], misalignment_tree['%s' %ID_i][use_angle], misalignment_tree['%s' %ID_i]['merger_ratio_stars']):
    #        print('%.2f\t%.1f\t' %(time_i, angle_i), merger_i)
    print(' remaining sample aaaaaa: ', len(ID_plot))
    
    
    
    #-----------
    ### Formatting
    axs.set_ylim(0, 180)
    axs.set_yticks(np.arange(0, 181, 30))
    axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    if set_plot_type == 'time':
        axs.set_xlim(0-3*time_extra, set_bin_limit_trelax)
        axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, 1))
        axs.set_xlabel('Time since misalignment (Gyr)')
    elif set_plot_type == 'raw_time':
        axs.set_xlim(8, 0)
        axs.set_xticks(np.arange(8, -0.1, -1))
        axs.set_xlabel('Lookbacktime (Gyr)')
    elif set_plot_type == 'snap':
        axs.set_xlim(-10, 70)
        axs.set_xticks(np.arange(-10, 71, 10))
        axs.set_xlabel('Snapshots since misalignment')
    elif set_plot_type == 'raw_snap':
        axs.set_xlim(140, 200)
        axs.set_xticks(np.arange(140, 201, 5))
        axs.set_xlabel('Snapshots')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    if (set_plot_type == 'time') or (set_plot_type == 'snap'):
        axs.axvline(0, ls='-', lw=1, c='grey')
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        legend_elements = []
        legend_labels = []
        legend_colors = []
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels.append('co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
            
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
        legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        axs.add_artist(legend2)
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['trelax']['co-co']))
    median_co_co = np.median(np.array(summary_dict['trelax']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['trelax']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['trelax']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['trelax']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['trelax']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['trelax']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['trelax']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['trelax']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['trelax']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['trelax']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['trelax']['counter-co']))
    print('Relaxation timescales:')
    print('   (Gyr)     all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/trelax_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/trelax_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_stacked_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_tdyn                  = 32,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 1,        # [ multiples ]
                      set_thist_ymax_tdyn                 = 0.35,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================        
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    time_collect = []
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
                    
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_tdyn:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 20:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.02)
            c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5))
        
            
            
        axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
        
        ### Annotate
        if set_add_GalaxyIDs:
            axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    print('  Using sample: ', len(ID_plot))
    print('  List of >20 trelax/tdyn ', len(ID_collect))
    print(ID_collect)
    print('asugaigfaiuf')
    print('median: ', np.median(time_collect))
    
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 32.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                        
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 32.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 32.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 32.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                 
    #-----------
    # Add threshold
    axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    axs.set_ylim(0, 180)
    axs.set_yticks(np.arange(0, 181, 30))
    axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    axs.set_xlim(-1, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, 2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    
    #-----------
    ### Annotations
    if (set_plot_type == 'time') or (set_plot_type == 'snap'):
        axs.axvline(0, ls='-', lw=1, c='k')
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        legend_elements = []
        legend_labels = []
        legend_colors = []
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels.append('co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
            
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
        legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        axs.add_artist(legend2)
    
    
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['tdyn']['co-co']))
    median_co_co = np.median(np.array(summary_dict['tdyn']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['tdyn']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['tdyn']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['tdyn']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['tdyn']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['tdyn']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['tdyn']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['tdyn']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['tdyn']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['tdyn']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['tdyn']['counter-co']))
    print('trelax/tdyn multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/tdyn_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/tdyn_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_stacked_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_ttorque               = 12,       # [ None / multiples ]
                      set_bin_width_ttorque               = 0.5,      # [ multiples ]
                      set_thist_ymax_ttorque              = 0.35,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #-------------------------        
        
    
    #===============================================================
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    time_collect = []
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_ttorque:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 10:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.02)
            c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5))
        
            
            
        axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
        
        ### Annotate
        if set_add_GalaxyIDs:
            axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    #print('-------------------------------------------------------------')
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    print('  Using sample: ', len(ID_plot))
    print('  List of >10 trelax/ttorque ', len(ID_collect))
    print(ID_collect)
    print('asugaigfaiuf')
    print('median: ', np.median(time_collect))
    
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 12.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                        
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 12.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 12.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 12.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                 
    #-----------
    # Add threshold
    axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    axs.set_ylim(0, 180)
    axs.set_yticks(np.arange(0, 181, 30))
    axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    axs.set_xlim(-0.5, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, 1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        
    #-----------
    ### Annotations
    if (set_plot_type == 'time') or (set_plot_type == 'snap'):
        axs.axvline(0, ls='-', lw=1, c='k')
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        legend_elements = []
        legend_labels = []
        legend_colors = []
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels.append('co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
            
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
        legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        axs.add_artist(legend2)
    
    
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['ttorque']['co-co']))
    median_co_co = np.median(np.array(summary_dict['ttorque']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['ttorque']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['ttorque']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['ttorque']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['ttorque']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['ttorque']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['ttorque']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['ttorque']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['ttorque']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['ttorque']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['ttorque']['counter-co']))
    print('trelax/ttorque multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/ttorque_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/ttorque_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()

    
#-------------------------
# Stacked 2x2 graphs
def _plot_stacked_trelax_2x2(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_trelax                = 5,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.25,     # [ 0.25 / Gyr ]
                      set_thist_ymax_trelax               = 0.45,             # 0.45 / 500  yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================
    # Graph initialising and base formatting
    fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        if set_plot_type == 'time':
            timeaxis_plot = -1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']])
        elif set_plot_type == 'raw_time':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - (0)
        elif set_plot_type == 'snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - misalignment_tree['%s' %ID_i]['SnapNum'][misalignment_tree['%s' %ID_i]['index_s']]
        elif set_plot_type == 'raw_snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - (0)
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                ax = ax_co_co
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                ax = ax_co_counter
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                ax = ax_counter_co
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                ax = ax_counter_counter
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_trelax:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1]+0.120)
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_time'] > 2:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.02)
            c     = lighten_color(line_color, (misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            
            
        ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
        ### Annotate
        if set_add_GalaxyIDs:
            ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of >2 Gyr trelax ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 5.09, np.median(diff_co_co))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 5.09, np.median(diff_co_counter))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 5.09, np.median(diff_counter_counter))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 5.09, np.median(diff_counter_co))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
    
             
    #-----------
    # Add threshold
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    ax_co_co.set_ylim(0, 180)
    ax_co_co.set_yticks(np.arange(0, 181, 30))
    ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
    if set_plot_type == 'time':
        ax_counter_counter.set_xlim(0-3*time_extra, set_bin_limit_trelax)
        ax_counter_counter.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, 1))
        ax_counter_co.set_xlabel('Time since misalignment (Gyr)')
        ax_counter_counter.set_xlabel('Time since misalignment (Gyr)')
        #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    elif set_plot_type == 'raw_time':
        ax_counter_counter.set_xlim(8, 0)
        ax_counter_counter.set_xticks(np.arange(8, -0.1, -1))
        ax_counter_counter.set_xlabel('Lookbacktime (Gyr)')
        ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    elif set_plot_type == 'snap':
        ax_counter_counter.set_xlim(-10, 70)
        ax_counter_counter.set_xticks(np.arange(-10, 71, 10))
        ax_counter_counter.set_xlabel('Snapshots since misalignment')
        ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    elif set_plot_type == 'raw_snap':
        ax_counter_counter.set_xlim(140, 200)
        ax_counter_counter.set_xticks(np.arange(140, 201, 5))
        ax_counter_counter.set_xlabel('Snapshots')
        ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
            
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels = ['co → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C0']
            legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_co.add_artist(legend2)
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels = ['counter → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C1']
            legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_counter.add_artist(legend2)
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels = ['     co → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C2']
            legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_counter.add_artist(legend2)
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels = ['counter → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C3']
            legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_co.add_artist(legend2)
            
    #-----------
    ### title
    if plot_annotate:
        ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['trelax']['co-co']))
    median_co_co = np.median(np.array(summary_dict['trelax']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['trelax']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['trelax']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['trelax']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['trelax']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['trelax']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['trelax']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['trelax']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['trelax']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['trelax']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['trelax']['counter-co']))
    print('Relaxation timescales:')
    print('   (Gyr)     all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else std_counter_co)))
    
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/trelax_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/trelax_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_stacked_tdyn_2x2(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_tdyn                  = 32,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 1,        # [ multiples ]
                      set_thist_ymax_tdyn                 = 0.35,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================
    # Graph initialising and base formatting
    fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                ax = ax_co_co
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)     
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                ax = ax_co_counter
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                ax = ax_counter_co
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                ax = ax_counter_counter
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_tdyn:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 10:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.02)
            c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5))
            
            
            
            
        ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
        ### Annotate
        if set_add_GalaxyIDs:
            ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    #print('  List of >10 trelax/tdyn ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 32.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 32.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 32.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 32.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            
    #-----------
    # Add threshold
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    ax_co_co.set_ylim(0, 180)
    ax_co_co.set_yticks(np.arange(0, 181, 30))
    ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
    ax_counter_counter.set_xlim(-1, set_bin_limit_tdyn)
    ax_counter_counter.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, 2))
    ax_counter_counter.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    ax_counter_co.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
    
    #-----------
    ### title
    if plot_annotate:
        ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
            
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels = ['co → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C0']
            legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_co.add_artist(legend2)
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels = ['counter → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C1']
            legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_counter.add_artist(legend2)
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels = ['     co → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C2']
            legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_counter.add_artist(legend2)
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels = ['counter → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C3']
            legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_co.add_artist(legend2)
            
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['tdyn']['co-co']))
    median_co_co = np.median(np.array(summary_dict['tdyn']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['tdyn']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['tdyn']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['tdyn']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['tdyn']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['tdyn']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['tdyn']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['tdyn']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['tdyn']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['tdyn']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['tdyn']['counter-co']))
    print('trelax/tdyn multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else std_counter_co)))
    
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/tdyn_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/tdyn_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_stacked_ttorque_2x2(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_ttorque               = 12,       # [ None / multiples ]
                      set_bin_width_ttorque               = 0.5,      # [ multiples ]
                      set_thist_ymax_ttorque              = 0.35,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #-------------------------         


    #=================================================================================================
    # Graph initialising and base formatting
    fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                ax = ax_co_co
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                ax = ax_co_counter
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                ax = ax_counter_co
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                ax = ax_counter_counter
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_ttorque:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 5:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.02)
            c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5))
            
            
            
            
        ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
        ### Annotate
        if set_add_GalaxyIDs:
            ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    #print('  List of >5 trelax/ttorque ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 12.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 12.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 12.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 12.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            
    #-----------
    # Add threshold
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    ax_co_co.set_ylim(0, 180)
    ax_co_co.set_yticks(np.arange(0, 181, 30))
    ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
    ax_counter_counter.set_xlim(-0.5, set_bin_limit_ttorque)
    ax_counter_counter.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, 2))
    ax_counter_counter.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    ax_counter_co.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
    
    #-----------
    ### title
    if plot_annotate:
        ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
            
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels = ['co → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C0']
            legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_co.add_artist(legend2)
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels = ['counter → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C1']
            legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_counter.add_artist(legend2)
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels = ['     co → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C2']
            legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_counter.add_artist(legend2)
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels = ['counter → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C3']
            legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_co.add_artist(legend2)
            
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['ttorque']['co-co']))
    median_co_co = np.median(np.array(summary_dict['ttorque']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['ttorque']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['ttorque']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['ttorque']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['ttorque']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['ttorque']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['ttorque']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['ttorque']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['ttorque']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['ttorque']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['ttorque']['counter-co']))
    print('trelax/ttorque multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/ttorque_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/ttorque_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plot box and whisker of relaxation distributions
def _plot_box_and_whisker_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_whisker_morphs                  = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    for ID_i in misalignment_tree.keys():
        
        # only plot morphs we care about (default ETG-ETG, LTG-LTG)
        if 'ETG' in set_whisker_morphs:
            if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                continue
        elif 'ETG-ETG' in set_whisker_morphs: 
            if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                continue
        
        # Gather data
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            relaxationtype_plot.append('co\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            relaxationtype_plot.append('co\n ↓ \ncounter')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            relaxationtype_plot.append('counter\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            relaxationtype_plot.append('counter\n ↓ \ncounter')
        
        if 'ETG' in set_whisker_morphs:
            relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
        elif 'ETG-ETG' in set_whisker_morphs: 
            #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
    
    #----------------------
    # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
    print('\n--------------------------------------')
    print('trelax')
    print('  Using sample: ', len(relaxationtype_plot))
    for relaxation_type_i in relaxation_type:
        
        if relaxation_type_i == 'co-co':
            relaxation_type_i = 'co\n ↓ \nco'
        if relaxation_type_i == 'co-counter':
            relaxation_type_i = 'co\n ↓ \ncounter'
        if relaxation_type_i == 'counter-co':
            relaxation_type_i = 'counter\n ↓ \nco'
        if relaxation_type_i == 'counter-counter':
            relaxation_type_i = 'counter\n ↓ \ncounter'
        
        # Select only relaxation morphs
        if 'ETG' in set_whisker_morphs:
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
        elif 'ETG-ETG' in set_whisker_morphs:    
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
        
        # KS test relaxation types between ETG and LTG
        if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

            res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
            
            if relaxation_type_i == 'co\n ↓ \nco':
                relaxation_type_i = 'co-co'
            if relaxation_type_i == 'co\n ↓ \ncounter':
                relaxation_type_i = 'co-counter'
            if relaxation_type_i == 'counter\n ↓ \nco':
                relaxation_type_i = 'counter-co'
            if relaxation_type_i == 'counter\n ↓ \ncounter':
                relaxation_type_i = 'counter-counter'
        
            if 'ETG' in set_whisker_morphs:
                print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            elif 'ETG-ETG' in set_whisker_morphs:  
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
            print('   p-value: %s' %res.pvalue)
        else:
            print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
    print('--------------------------------------')
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    

    if 'ETG' in set_whisker_morphs:
        order = ['ETG', 'LTG']
    elif 'ETG-ETG' in set_whisker_morphs:   
        order = ['ETG → ETG', 'LTG → LTG']
    #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
    #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
    my_pal = {order[0]: "r", order[1]: "b"}
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', fill=False, linewidth=0.7, alpha=1)
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', linewidth=0, alpha=0.1, legend=False)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-------------
    ### Formatting
    axs.set_xlim(left=0)
    #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
    
    #print(max(relaxationtime_plot))
    
    
    #------------
    # Legend
    axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
    
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/violinplot_relaxation_morph/trelax_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/violinplot_relaxation_morph/trelax_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_box_and_whisker_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_whisker_morphs                  = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================   
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    for ID_i in misalignment_tree.keys():
        
        # only plot morphs we care about (default ETG-ETG, LTG-LTG)
        if 'ETG' in set_whisker_morphs:
            if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                continue
        elif 'ETG-ETG' in set_whisker_morphs: 
            if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                continue
        
        # Gather data
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            relaxationtype_plot.append('co\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            relaxationtype_plot.append('co\n ↓ \ncounter')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            relaxationtype_plot.append('counter\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            relaxationtype_plot.append('counter\n ↓ \ncounter')
        
        if 'ETG' in set_whisker_morphs:
            relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
        elif 'ETG-ETG' in set_whisker_morphs: 
            #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
    
    #----------------------
    # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
    print('\n--------------------------------------')
    print('tdyn')
    print('  Using sample: ', len(relaxationtype_plot))
    for relaxation_type_i in relaxation_type:
        
        if relaxation_type_i == 'co-co':
            relaxation_type_i = 'co\n ↓ \nco'
        if relaxation_type_i == 'co-counter':
            relaxation_type_i = 'co\n ↓ \ncounter'
        if relaxation_type_i == 'counter-co':
            relaxation_type_i = 'counter\n ↓ \nco'
        if relaxation_type_i == 'counter-counter':
            relaxation_type_i = 'counter\n ↓ \ncounter'
        
        # Select only relaxation morphs
        if 'ETG' in set_whisker_morphs:
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
        elif 'ETG-ETG' in set_whisker_morphs:    
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
        
        # KS test relaxation types between ETG and LTG
        if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

            res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
            
            if relaxation_type_i == 'co\n ↓ \nco':
                relaxation_type_i = 'co-co'
            if relaxation_type_i == 'co\n ↓ \ncounter':
                relaxation_type_i = 'co-counter'
            if relaxation_type_i == 'counter\n ↓ \nco':
                relaxation_type_i = 'counter-co'
            if relaxation_type_i == 'counter\n ↓ \ncounter':
                relaxation_type_i = 'counter-counter'
        
            if 'ETG' in set_whisker_morphs:
                print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            elif 'ETG-ETG' in set_whisker_morphs:  
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
            print('   p-value: %s' %res.pvalue)
        else:
            print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
    print('--------------------------------------')
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    

    if 'ETG' in set_whisker_morphs:
        order = ['ETG', 'LTG']
    elif 'ETG-ETG' in set_whisker_morphs:   
        order = ['ETG → ETG', 'LTG → LTG']
    #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
    #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
    my_pal = {order[0]: "r", order[1]: "b"}
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', fill=False, linewidth=0.7, alpha=1)
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', linewidth=0, alpha=0.1)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-------------
    ### Formatting
    axs.set_xlim(left=0)
    #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    
    #------------
    # Legend
    axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
    
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/violinplot_relaxation_morph/tdyn_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/violinplot_relaxation_morph/tdyn_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_box_and_whisker_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_whisker_morphs                  = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================  
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    for ID_i in misalignment_tree.keys():
        
        # only plot morphs we care about (default ETG-ETG, LTG-LTG)
        if 'ETG' in set_whisker_morphs:
            if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                continue
        elif 'ETG-ETG' in set_whisker_morphs: 
            if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                continue
        
        # Gather data
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            relaxationtype_plot.append('co\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            relaxationtype_plot.append('co\n ↓ \ncounter')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            relaxationtype_plot.append('counter\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            relaxationtype_plot.append('counter\n ↓ \ncounter')
        
        if 'ETG' in set_whisker_morphs:
            relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
        elif 'ETG-ETG' in set_whisker_morphs: 
            #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
    
    #----------------------
    # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
    print('\n--------------------------------------')
    print('ttorque')
    print('  Using sample: ', len(relaxationtype_plot))
    for relaxation_type_i in relaxation_type:
        
        if relaxation_type_i == 'co-co':
            relaxation_type_i = 'co\n ↓ \nco'
        if relaxation_type_i == 'co-counter':
            relaxation_type_i = 'co\n ↓ \ncounter'
        if relaxation_type_i == 'counter-co':
            relaxation_type_i = 'counter\n ↓ \nco'
        if relaxation_type_i == 'counter-counter':
            relaxation_type_i = 'counter\n ↓ \ncounter'
        
        # Select only relaxation morphs
        if 'ETG' in set_whisker_morphs:
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
        elif 'ETG-ETG' in set_whisker_morphs:    
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
        
        # KS test relaxation types between ETG and LTG
        if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

            res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
            
            if relaxation_type_i == 'co\n ↓ \nco':
                relaxation_type_i = 'co-co'
            if relaxation_type_i == 'co\n ↓ \ncounter':
                relaxation_type_i = 'co-counter'
            if relaxation_type_i == 'counter\n ↓ \nco':
                relaxation_type_i = 'counter-co'
            if relaxation_type_i == 'counter\n ↓ \ncounter':
                relaxation_type_i = 'counter-counter'
        
            if 'ETG' in set_whisker_morphs:
                print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            elif 'ETG-ETG' in set_whisker_morphs:  
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
            print('   p-value: %s' %res.pvalue)
        else:
            print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
    print('--------------------------------------')
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    

    if 'ETG' in set_whisker_morphs:
        order = ['ETG', 'LTG']
    elif 'ETG-ETG' in set_whisker_morphs:   
        order = ['ETG → ETG', 'LTG → LTG']
    #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
    #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
    my_pal = {order[0]: "r", order[1]: "b"}
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', fill=False, linewidth=0.7, alpha=1)
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', linewidth=0, alpha=0.1)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-------------
    ### Formatting
    axs.set_xlim(left=0)
    #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    
    #------------
    # Legend
    axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
    
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/violinplot_relaxation_morph/ttorque_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/violinplot_relaxation_morph/ttorque_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plot delta angle, trelax. Looks at peak angle from 180
def _plot_offset_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_plot_offset_type                = ['co-co', 'counter-counter'],                     # [ 'co-co', 'co-counter' ]  or False
                      use_offset_morphs                   = True,
                        set_offset_morphs                 = ['LTG-LTG', 'ETG-ETG'],           # [ None / ['LTG-LTG', 'ETG-ETG'] ] Can be either relaxation_morph or misalignment_morph
                      #-----------------------
                      # General formatting
                      set_plot_offset_range               = [30, 150],                         # [min angle , max angle] of peak misangle
                        set_plot_offset_log               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    angles_plot = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # check offset angle within range            
        if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
            continue
        if set_plot_offset_type:
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                continue
                            
        # Add angles
        angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
            
        ID_plot.append(ID_i)
    
    # Collect data into dataframe
    print('  Using sample: ', len(relaxationmorph_plot))
    df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
            
            
    #-------------
    # Plotting
    fig = plt.figure(figsize=(10/3, 10/3))
    gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    bin_width = (set_plot_offset_range[1] - set_plot_offset_range[0])/7
    bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
    c = 'k'
    
    # Bin hist data, find sigma percentiles
    binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    #print(bin_medians)
    
    #-------------------
    ### Plot scatter
    ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.2, c='k', edgecolor='grey', marker='.', alpha=1, zorder=20)
    
    ### Plot upper, median, and lower sigma
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
    ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.2, zorder=6)
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
    ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=99, label='sample')
    #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
    
    
    ### Plot histograms
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    #-------------
    ### Plot by morphology
    if use_offset_morphs:
        for offset_morph_i in set_offset_morphs:
            if offset_morph_i == 'ETG-ETG':
                offset_morph_i = 'ETG → ETG'
                c = 'r'
                ls = '--'
            elif offset_morph_i == 'LTG-LTG':
                offset_morph_i = 'LTG → LTG'
                c = 'b'
                ls = 'dashdot'
        
            # Dataframe of morphs matching this
            df_morph = df.loc[df['Morphology'] == offset_morph_i]
        
            # Bin hist data, find sigma percentiles
            binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
            bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            #-------------------
            ### Plot scatter
            #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
        
            ### Plot upper, median, and lower sigma
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
            ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.5)
            
            yerr_top    = bin_medians[:,3] - bin_medians[:,2]
            yerr_bottom = bin_medians[:,2] - bin_medians[:,1]
            ax.errorbar(bins[:-1]+(bin_width/2), bin_medians[:,2], yerr=(yerr_bottom, yerr_top), lw=0.7, c=c, ls=None, zorder=99, alpha=0.5, ecolor=c, elinewidth=0.5, capsize=1.5)
            
            #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
            ### Plot histograms
            ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
            ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    
    #-----------
    ### General formatting
    # Axis labels
    ax.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
    ax.set_xlabel('Peak angle from stability')
    ax_histy.set_xlabel('Count')
    ax_histx.set_ylabel('Count')
    if set_plot_offset_log:
        ax.set_yscale('log')
        ax.set_ylim(0.1, 10)
        ax.set_yticks([0.1, 1, 10])
        ax.set_yticklabels(['0.1', '1', '10'])
    else:
        ax.set_ylim(0, 3)
    ax.set_xticks(np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, 15))
    ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
    
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([1, 10, 100, 1000])
    ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
    
    #------------
    ### Add title
    if set_plot_offset_type:
        
        set_title = ''
        
        if 'co-co' in set_plot_offset_type:
            set_title = set_title + r'co → co'
        if 'counter-counter' in set_plot_offset_type:
            set_title = set_title + r', counter → counter'
        if 'co-counter' in set_plot_offset_type:
            set_title = set_title + r', co → counter'
        if 'counter-co' in set_plot_offset_type:
            set_title = set_title + r', counter → co'
            
        ax_histx.set_title(r'%s' %set_title, size=7, loc='left', pad=3)
            
    
    #------------
    # Legend
    ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #------------
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/offset_angle/trelax_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/offset_angle/trelax_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_offset_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_plot_offset_type                = ['co-co', 'counter-counter'],                     # [ 'co-co', 'co-counter' ]  or False
                      use_offset_morphs                   = True,
                        set_offset_morphs                 = ['LTG-LTG', 'ETG-ETG'],           # [ None / ['LTG-LTG', 'ETG-ETG'] ] Can be either relaxation_morph or misalignment_morph
                      #-----------------------
                      # General formatting
                      set_plot_offset_range               = [30, 150],                         # [min angle , max angle] of peak misangle
                        set_plot_offset_log               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    angles_plot = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # check offset angle within range            
        if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
            continue
        if set_plot_offset_type:
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                continue
                            
        # Add angles
        angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
            
        ID_plot.append(ID_i)
        
    # Collect data into dataframe
    print('  Using sample: ', len(relaxationmorph_plot))
    df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
            
    #-------------
    # Plotting
    fig = plt.figure(figsize=(10/3, 10/3))
    gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    bin_width = (set_plot_offset_range[1] - set_plot_offset_range[0])/7
    bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
    c = 'k'
    
    #-------------
    # Bin hist data, find sigma percentiles
    binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    #print(bin_medians)
    
    #-------------------
    ### Plot scatter
    ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.2, c='k', edgecolor='grey', marker='.', alpha=1, zorder=20)
    
    ### Plot upper, median, and lower sigma
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
    ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.2, zorder=6)
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
    ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=99, label='sample')
    #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
    
    
    ### Plot histograms
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    
    
    #-------------
    ### Plot by morphology
    for offset_morph_i in set_offset_morphs:

        if offset_morph_i == 'ETG-ETG':
            offset_morph_i = 'ETG → ETG'
            c = 'r'
            ls = '--'
        elif offset_morph_i == 'LTG-LTG':
            offset_morph_i = 'LTG → LTG'
            c = 'b'
            ls = 'dashdot'
        
        # Dataframe of morphs matching this
        df_morph = df.loc[df['Morphology'] == offset_morph_i]
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.5)
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        yerr_top    = bin_medians[:,3] - bin_medians[:,2]
        yerr_bottom = bin_medians[:,2] - bin_medians[:,1]
        ax.errorbar(bins[:-1]+(bin_width/2), bin_medians[:,2], yerr=(yerr_bottom, yerr_top), lw=0.7, c=c, ls=None, zorder=99, alpha=0.5, ecolor=c, elinewidth=0.5, capsize=1.5)
        
        
        ### Plot histograms
        ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    
    #-----------
    ### General formatting
    # Axis labels
    ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    ax.set_xlabel('Peak misalignment angle')
    ax_histy.set_xlabel('Count')
    ax_histx.set_ylabel('Count')
    if set_plot_offset_log:
        ax.set_yscale('log')
        ax.set_ylim(0.5, 35)
        ax.set_yticks([1, 10])
        ax.set_yticklabels(['1', '10'])
    else:
        ax.set_ylim(0, 15)
    ax.set_xticks(np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, 15))
    ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
    
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([1, 10, 100, 1000])
    ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
    
    #------------
    ### Add title
    if set_plot_offset_type:
        
        set_title = ''
        
        if 'co-co' in set_plot_offset_type:
            set_title = set_title + r'co → co'
        if 'counter-counter' in set_plot_offset_type:
            set_title = set_title + r', counter → counter'
        if 'co-counter' in set_plot_offset_type:
            set_title = set_title + r', co → counter'
        if 'counter-co' in set_plot_offset_type:
            set_title = set_title + r', counter → co'
            
        ax_histx.set_title(r'%s' %set_title, size=7, loc='left', pad=3)
            
    
    #------------
    # Legend
    ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #------------
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/offset_angle/tdyn_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/offset_angle/tdyn_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_offset_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_plot_offset_type                = ['co-co', 'counter-counter'],                     # [ 'co-co', 'co-counter' ]  or False
                      use_offset_morphs                   = True,
                        set_offset_morphs                 = ['LTG-LTG', 'ETG-ETG'],           # [ None / ['LTG-LTG', 'ETG-ETG'] ] Can be either relaxation_morph or misalignment_morph
                      #-----------------------
                      # General formatting
                      set_plot_offset_range               = [30, 150],                         # [min angle , max angle] of peak misangle
                        set_plot_offset_log               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #========================================================================== 
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    angles_plot = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # check offset angle within range            
        if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
            continue
        if set_plot_offset_type:
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                continue
                            
        # Add angles
        angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
            
        ID_plot.append(ID_i)
    
    # Collect data into dataframe
    print('  Using sample: ', len(relaxationmorph_plot))
    df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
            
    #-------------
    # Plotting
    fig = plt.figure(figsize=(10/3, 10/3))
    gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    bin_width = (set_plot_offset_range[1] - set_plot_offset_range[0])/7
    bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
    c = 'k'
    
    #-------------
    # Bin hist data, find sigma percentiles
    binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    #print(bin_medians)
        
    #-------------------
    ### Plot scatter
    ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.2, c='k', edgecolor='grey', marker='.', alpha=1, zorder=20)
        
    ### Plot upper, median, and lower sigma
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
    ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.2, zorder=6)
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
    ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=99, label='sample')
    #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
    ### Plot histograms
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    #-------------
    ### Plot by morphology
    for offset_morph_i in set_offset_morphs:

        if offset_morph_i == 'ETG-ETG':
            offset_morph_i = 'ETG → ETG'
            c = 'r'
            ls = '--'
        elif offset_morph_i == 'LTG-LTG':
            offset_morph_i = 'LTG → LTG'
            c = 'b'
            ls = 'dashdot'
        
        # Dataframe of morphs matching this
        df_morph = df.loc[df['Morphology'] == offset_morph_i]
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.5)
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        yerr_top    = bin_medians[:,3] - bin_medians[:,2]
        yerr_bottom = bin_medians[:,2] - bin_medians[:,1]
        ax.errorbar(bins[:-1]+(bin_width/2), bin_medians[:,2], yerr=(yerr_bottom, yerr_top), lw=0.7, c=c, ls=None, zorder=99, alpha=0.5, ecolor=c, elinewidth=0.5, capsize=1.5)
        
        
        ### Plot histograms
        ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=0.8, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    
    #-----------
    ### General formatting
    # Axis labels
    ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    ax.set_xlabel('Peak misalignment angle')
    ax_histy.set_xlabel('Count')
    ax_histx.set_ylabel('Count')
    if set_plot_offset_log:
        ax.set_yscale('log')
        ax.set_ylim(0.2, 25)
        ax.set_yticks([0.2, 1, 10])
        ax.set_yticklabels(['0.2', '1', '10'])
    else:
        ax.set_ylim(0, 10)
    ax.set_xticks(np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, 15))
    ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
    
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([1, 10, 100, 1000])
    ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
    
    #------------
    ### Add title
    if set_plot_offset_type:
        
        set_title = ''
        
        if 'co-co' in set_plot_offset_type:
            set_title = set_title + r'co → co'
        if 'counter-counter' in set_plot_offset_type:
            set_title = set_title + r', counter → counter'
        if 'co-counter' in set_plot_offset_type:
            set_title = set_title + r', co → counter'
        if 'counter-co' in set_plot_offset_type:
            set_title = set_title + r', counter → co'
            
        ax_histx.set_title(r'%s' %set_title, size=7, loc='left', pad=3)       
    
    #------------
    # Legend
    ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #------------
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/offset_angle/ttorque_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/offset_angle/ttorque_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()   


#-------------------------
# Number of mergers within window with relaxation time (suited for mass of 1010 and above)
def _plot_merger_count_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_min_merger_trelax               = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                      set_plot_merger_count_lim           = 0.1,          # stellar ratio (will also pick reciprocal)
                      add_plot_merger_count_gas           = True,         # will colour by gas ratio
                        set_plot_merger_count_log         = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    mergercount_plot     = []
    mergerstarratio_plot = []
    mergergasratio_plot  = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_merger_trelax:
            continue
        
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        
        # Find mergers within window
        merger_count = 0
        ratio_star_list = []
        ratio_gas_list  = [] 
        for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
            if len(merger_i) > 0:
                for merger_ii, gas_ii in zip(merger_i, gas_i):
                    if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                        merger_count += 1
                        ratio_star_list.append(merger_ii)
                        ratio_gas_list.append(gas_ii)
        
        # Append number of mergers, and average stellar and gas ratio of these mergers
        mergercount_plot.append(merger_count)
        mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
        mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
        

    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
    print('\n--------------------------------------')
    print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
    print('NUMBER OF MERGERS > 0.1 - RELAXATION TIME SPEARMAN:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    
    #-------------
    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
    
    df_0       = df.loc[(df['Number of mergers'] == 0)]
    df_mergers = df.loc[(df['Number of mergers'] > 0)]
    
    im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
    axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
    
    plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')
    
    
    #-------------
    ### Formatting
    axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
    axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
    if set_plot_merger_count_log:
        axs.set_xscale('log')
        axs.set_xlim(0.1, 6)
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(['0.1', '1', '10'])
    else:
        axs.set_xlim(0.5, 6)
        axs.set_xticks(np.arange(1, 6.1))
    axs.set_ylim(-0.3, 3.3)
    axs.set_yticks(np.arange(0, 3.1, 1))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    axs.set_title(r'$\rho$ = %.2f p-value = %.1e' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #------------
    # Legend
    axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/number_mergers_relaxtime/trelax_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/number_mergers_relaxtime/trelax_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_merger_count_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_min_merger_tdyn                 = 0,          # [ trelax/dyn ] min relaxation time, as we dont care about short relaxers
                      set_plot_merger_count_lim           = 0.1,          # stellar ratio (will also pick reciprocal)
                      add_plot_merger_count_gas           = True,         # will colour by gas ratio
                        set_plot_merger_count_log         = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    #===================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    mergercount_plot     = []
    mergerstarratio_plot = []
    mergergasratio_plot  = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] <= set_min_merger_tdyn:
            continue
        
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        
        # Find mergers within window
        merger_count = 0
        ratio_star_list = []
        ratio_gas_list  = [] 
        for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
            if len(merger_i) > 0:
                for merger_ii, gas_ii in zip(merger_i, gas_i):
                    if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                        merger_count += 1
                        ratio_star_list.append(merger_ii)
                        ratio_gas_list.append(gas_ii)
        
        # Append number of mergers, and average stellar and gas ratio of these mergers
        mergercount_plot.append(merger_count)
        mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
        mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
        

    print('  Using sample: ', len(ID_plot))    
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
    print('\n--------------------------------------')
    print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
    print('NUMBER OF MERGERS > 0.1 - RELAXATION TDYN SPEARMAN:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    
    #-------------
    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
    
    df_0       = df.loc[(df['Number of mergers'] == 0)]
    df_mergers = df.loc[(df['Number of mergers'] > 0)]
    
    im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
    axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
    
    plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')
    
    
    #-------------
    ### Formatting
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
    if set_plot_merger_count_log:
        axs.set_xscale('log')
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(['0.1', '1', '10'])
        axs.set_xlim(0.6, 25)
    else:
        axs.set_xlim(0, 20)
        axs.set_xticks(np.arange(4, 20.1, 4))
    axs.set_ylim(-0.3, 3.3)
    axs.set_yticks(np.arange(0, 3.1, 1))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    axs.set_title(r'$\rho$ = %.2f p-value = %.2f' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #------------
    # Legend
    axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/number_mergers_relaxtime/tdyn_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/number_mergers_relaxtime/tdyn_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_merger_count_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_min_merger_ttorque              = 0,          # [ trelax/torque ] min relaxation time, as we dont care about short relaxers
                      set_plot_merger_count_lim           = 0.1,          # stellar ratio (will also pick reciprocal)
                      add_plot_merger_count_gas           = True,         # will colour by gas ratio
                        set_plot_merger_count_log         = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    #===================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    mergercount_plot     = []
    mergerstarratio_plot = []
    mergergasratio_plot  = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] <= set_min_merger_ttorque:
            continue
        
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        
        # Find mergers within window
        merger_count = 0
        ratio_star_list = []
        ratio_gas_list  = [] 
        for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
            if len(merger_i) > 0:
                for merger_ii, gas_ii in zip(merger_i, gas_i):
                    if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                        merger_count += 1
                        ratio_star_list.append(merger_ii)
                        ratio_gas_list.append(gas_ii)
        
        # Append number of mergers, and average stellar and gas ratio of these mergers
        mergercount_plot.append(merger_count)
        mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
        mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
        

    print('  Using sample: ', len(ID_plot))
              
    # Collect data into dataframe
    df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
    print('\n--------------------------------------')
    print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
    print('NUMBER OF MERGERS > 0.1 - RELAXATION TTORQUE SPEARMAN:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    
    #-------------
    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
    
    df_0       = df.loc[(df['Number of mergers'] == 0)]
    df_mergers = df.loc[(df['Number of mergers'] > 0)]
    
    im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
    axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
    
    plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')

    
    #-------------
    ### Formatting
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
    if set_plot_merger_count_log:
        axs.set_xscale('log')
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(['0.1', '1', '10'])
        axs.set_xlim(0.6, 15)
    else:
        axs.set_xlim(0.5, 12)
        axs.set_xticks(np.arange(2, 12.1, 2))
    axs.set_ylim(-0.3, 3.3)
    axs.set_yticks(np.arange(0, 3.1, 1))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    axs.set_title(r'$\rho$ = %.2f p-value = %.2f' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #------------
    # Legend
    axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/number_mergers_relaxtime/ttorque_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/number_mergers_relaxtime/ttorque_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots hist of average DM-stars misangle co-co preceeding misalignment with fraction
def _plot_halo_misangle_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution          = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_trelax                 = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_halo_trelax:
            continue
            
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
                        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle trelax plot: ', len(ID_plot))
    print('Stars-DM misangle vs trelax:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
    plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line for total sample
    ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
    
    if set_plot_halo_misangle_log:
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylim(0.1, 10)
        ax_scatter.set_yticks([0.1, 1, 10])
        ax_scatter.set_yticklabels(['0.1', '1', '10'])
    else:
        ax_scatter.set_ylim(0, 6)
        ax_scatter.set_yticks(np.arange(0, 6.1, 1))
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter.set_xlim(0, 180)
    ax_scatter.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if plot_annotate:
        ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/trelax_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/trelax_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_halo_misangle_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution        = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_tdyn                 = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] <= set_min_halo_tdyn:
            continue
            
            
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
                        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle tdyn plot: ', len(ID_plot))
    print('Stars-DM misangle vs tdyn:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
    plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #--------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line
    ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
    if set_plot_halo_misangle_log:
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylim(0.3, 25)
        ax_scatter.set_yticks([1, 10])
        ax_scatter.set_yticklabels(['1', '10'])
    else:
        ax_scatter.set_ylim(0, 15)
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter.set_xlim(0, 180)
    ax_scatter.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if plot_annotate:
        ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/tdyn_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/tdyn_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_halo_misangle_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution          = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_ttorque                = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] <= set_min_halo_ttorque:
            continue
            
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
                        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle ttorque plot: ', len(ID_plot))
    print('Stars-DM misangle vs ttorque:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
    plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    #--------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line
    ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101)
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
        
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
    if set_plot_halo_misangle_log:
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylim(0.1, 25)
        ax_scatter.set_yticks([0.1, 1, 10])
        ax_scatter.set_yticklabels(['0.1', '1', '10'])
    else:
        ax_scatter.set_ylim(0, 10)
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter.set_xlim(0, 180)
    ax_scatter.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if plot_annotate:
        ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/ttorque_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/ttorque_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# manual    
def _plot_halo_misangle_manual(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution          = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_trelax                 = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationtorque_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_halo_trelax:
            continue
            
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        relaxationtorque_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        

    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation torque': relaxationtorque_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle trelax plot: ', len(ID_plot))
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter1, ax_scatter2, ax_line) = plt.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [2.5, 2.5, 1]}, figsize=[10/3, 4.0], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = ax_scatter1.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=0.9)
    plt.colorbar(im1, ax=ax_scatter1, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    im2 = ax_scatter2.scatter(df['DM-stars angle'], df['Relaxation torque'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=0.9)
    plt.colorbar(im2, ax=ax_scatter2, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line for total sample
    #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
    #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='--', zorder=101)
    ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=101, label='sample')
    #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=1, c='k', ls='--', zorder=101)

    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation torque'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line for total sample
    #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
    #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='--', zorder=101)
    ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=101, label='sample')
    #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=1, c='k', ls='--', zorder=101)
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='b', ls='--', zorder=101)
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='b', ls='--', zorder=101)
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$')
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='r', ls='--', zorder=101)
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='r', ls='--', zorder=101)
    
        #-------------
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation torque'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='b', ls='--', zorder=101)
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='b', ls='--', zorder=101)
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation torque'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$')
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='r', ls='--', zorder=101)
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='r', ls='--', zorder=101)
        
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    #ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter1.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
    ax_scatter2.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel('Average stellar-DM misalignment angle')
    if set_plot_halo_misangle_log:
        ax_scatter1.set_yscale('log')
        ax_scatter1.set_ylim(0.1, 10)
        ax_scatter1.set_yticks([0.1, 1, 10])
        ax_scatter1.set_yticklabels(['0.1', '1', '10'])
        ax_scatter2.set_yscale('log')
        ax_scatter2.set_ylim(0.1, 25)
        ax_scatter2.set_yticks([0.1, 1, 10])
        ax_scatter2.set_yticklabels(['0.1', '1', '10'])
    else:
        ax_scatter1.set_ylim(0, 6)
        ax_scatter1.set_yticks(np.arange(0, 6.1, 1))
        ax_scatter2.set_ylim(0, 10)
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter1.set_xlim(0, 180)
    ax_scatter1.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter1, ax_scatter2, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if plot_annotate:
        ax_scatter1.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter1.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    ax_scatter2.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/both_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/both_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots hist of average DM-stars misangle co-co preceeding misalignment with fraction
def _plot_halo_misangle_pre_frac(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_misanglepre_morph               = True,         # will use a stacked histogram for ETG-ETG, LTG-LTG, ETG-LTG, etc.
                      set_misanglepre_type                = ['co-co', 'co-counter'],           # [ 'co-co', 'co-counter' ]  or False
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    halomisangle_plot    = []
    relaxationmorph_plot = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # If not a co-co or co-counter, skip
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_misanglepre_type:
            continue
        
        # Collect average stellar-DM misangle before misalignment.
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][0:misalignment_tree['%s' %ID_i]['index_s']+1]))
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        else:
            relaxationmorph_plot.append('other')
        

    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'DM-stars angle': halomisangle_plot, 'Relaxation morph': relaxationmorph_plot, 'GalaxyIDs': ID_plot})
    ETG_ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    LTG_LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    ETG_LTG_df = df.loc[(df['Relaxation morph'] == 'ETG → LTG')]
    LTG_ETG_df = df.loc[(df['Relaxation morph'] == 'LTG → ETG')]
    
    #-------------
    # Plotting ongoing fraction histogram
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    colors = ['orangered', 'orange', 'cornflowerblue', 'mediumblue']
    
    if set_misanglepre_morph:
        axs.hist([ETG_ETG_df['DM-stars angle'], ETG_LTG_df['DM-stars angle'], LTG_ETG_df['DM-stars angle'], LTG_LTG_df['DM-stars angle']], weights=(np.ones(len(ETG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(ETG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs'])), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', color = colors, alpha=0.5, stacked=True)
        hist_n, _, _ = axs.hist([ETG_ETG_df['DM-stars angle'], ETG_LTG_df['DM-stars angle'], LTG_ETG_df['DM-stars angle'], LTG_LTG_df['DM-stars angle']], weights=(np.ones(len(ETG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(ETG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs'])), bins=np.arange(0, 181, 10), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
    else:
        axs.hist(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', alpha=0.5)
        hist_n, _, _ = axs.hist(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7)
        
    print('Hist bins morphology: ', hist_n)
    hist_n, _ = np.histogram(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), range=(0, 181))
    print('Hist bins total: ', hist_n)
        
    #-------------
    ### Formatting
    axs.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ pre-instability')
    axs.set_xlim(0, 180)
    axs.set_xticks(np.arange(0, 181, step=30))
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(0, 0.2)
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    
    if set_misanglepre_morph:
        legend_labels.append('ETG → ETG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orangered')
            
        legend_labels.append('ETG → LTG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orange')
            
        legend_labels.append('LTG → ETG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('cornflowerblue')
            
        legend_labels.append('LTG → LTG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('mediumblue')
        
        axs.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        
    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_premisangle_fraction/halopremisangle_fraction_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_premisangle_fraction/halopremisangle_fraction_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots stacked histograms for different morphologies, and fraction caused by major, minor, or other origins
def _plot_origins(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      use_only_start_morph     = True,          # use only ETG - and LTG - 
                      set_origins_morph        = ['ETG-ETG', 'ETG-LTG', 'LTG-ETG', 'LTG-LTG'],
                      #-----------------------
                      # Mergers
                      use_alt_merger_criteria = True,
                        half_window         = 0.3,      # [ 0.2 / +/-Gyr ] window centred on first misaligned snap to look for mergers
                        min_ratio           = 0.1,   
                        merger_lookback_time = 2,       # Gyr, number of years to check for peak stellar mass
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Loading mergertree file to establish windows
    f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
    GalaxyID_tree             = np.array(f['Histories']['GalaxyID'])
    DescendantID_tree         = np.array(f['Histories']['DescendantID'])
    Lookbacktime_tree         = np.array(f['Snapnum_Index']['LookbackTime'])
    StellarMass_tree          = np.array(f['Histories']['StellarMass'])
    GasMass_tree              = np.array(f['Histories']['GasMass'])
    f.close()
    
    tally_minor = []
    tally_major = []
    tally_other = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_origins_morph:
            continue
            
        
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
                
            
        merger_count = 0
        for merger_i in merger_ratio_array_array:
            if len(merger_i) > 0:
                if (max(merger_i) > min_ratio):
                    if merger_count == 0:
                        if 0.3 > max(merger_i) > 0.1:
                            tally_minor.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                        if max(merger_i) > 0.3:
                            tally_major.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                    merger_count += 1
        if merger_count == 0:
            tally_other.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
    

    print('  Using sample: ', len(merger_ID_array_array))     
    
    print('\n======================================')
    print('Using merger criteria, half_window = %.1f Gyr, min_ratio = %.1f' %(half_window, min_ratio))
    print('\tmajor ', len(tally_major))
    print('\tminor ', len(tally_minor))
    print('\tother ', len(tally_other))
    print('\ttotal ', (len(tally_major)+len(tally_minor)+len(tally_other)))
    
    
    # left with tallo_minor = ['ETG-ETG', 'LTG-ETG', 'ETG-ETG', etc..]
    plot_dict = {'ETG-ETG': {}, 'LTG-ETG': {}, 'ETG-LTG': {}, 'LTG-LTG': {}}
    for dict_i in plot_dict.keys():
        plot_dict[dict_i] = {'array': [],
                             'major': [],
                             'minor': [],
                             'other': []}
    
    for morph_i in set_origins_morph:
        plot_dict['%s' %morph_i]['major'] = len([i for i in tally_major if i == morph_i])
        plot_dict['%s' %morph_i]['minor'] = len([i for i in tally_minor if i == morph_i])
        plot_dict['%s' %morph_i]['other'] = len([i for i in tally_other if i == morph_i])
        
        temp_array = []
        temp_array.extend([i for i in tally_major if i == morph_i])
        temp_array.extend([i for i in tally_minor if i == morph_i])
        temp_array.extend([i for i in tally_other if i == morph_i])
        plot_dict['%s' %morph_i]['array'] = temp_array
        
        
        
    #---------------------
    ### Plotting
    if not use_only_start_morph:
        #-----------
        # Plotting ongoing fraction histogram
        fig, ((ax_ETG_ETG, ax_ETG_LTG), (ax_LTG_ETG, ax_LTG_LTG), (ax_spare_1, ax_spare_2)) = plt.subplots(3, 2, figsize=[10/3, 10/3], gridspec_kw={'height_ratios': [1, 1, 0.1]}, sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0, hspace=0)
        
        #-----------
        # Add pie plots
        colors = ['royalblue', 'orange', 'orangered']
        for morph_i in set_origins_morph:
            y = np.array([plot_dict['%s' %morph_i]['other'], plot_dict['%s' %morph_i]['minor'], plot_dict['%s' %morph_i]['major']])
            mylabels = ['%s' %(' ' if plot_dict['%s' %morph_i]['other'] == 0 else plot_dict['%s' %morph_i]['other']), '%s' %(' ' if plot_dict['%s' %morph_i]['minor'] == 0 else plot_dict['%s' %morph_i]['minor']), '%s' %(' ' if plot_dict['%s' %morph_i]['major'] == 0 else plot_dict['%s' %morph_i]['major'])]
        
            if morph_i == 'ETG-ETG':
                ax_ETG_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_ETG_ETG.set_title('ETG → ETG', loc='center', pad=0, fontsize=8)
            if morph_i == 'ETG-LTG':
                ax_ETG_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_ETG_LTG.set_title('ETG → LTG', loc='center', pad=0, fontsize=8)
            if morph_i == 'LTG-ETG':
                ax_LTG_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_spare_1.set_title('\nLTG → ETG', loc='center', y=-1, fontsize=8)
            if morph_i == 'LTG-LTG':
                ax_LTG_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_spare_2.set_title('\nLTG → LTG', loc='center', y=-1, fontsize=8)
        ax_spare_1.axis('off')
        ax_spare_2.axis('off')
    
            
        #------------
        # Add legend
        legend_labels   = []
        legend_elements = []
        legend_colors   = []

        legend_labels.append('Major\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C2')
        legend_labels.append('Minor\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C1')
        legend_labels.append('Other')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C0')
    
        ax_LTG_LTG.legend(handles=legend_elements, labels=legend_labels, loc='lower left', bbox_to_anchor=(0.9, 0.7), labelspacing=1, frameon=False, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        #-----------
        # savefig
        if savefig:
            savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/misangle_origins/origins_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misangle_origins/origins_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()  
    else:
        #-----------
        # Plotting ongoing fraction histogram
        fig, (ax_ETG, ax_LTG) = plt.subplots(1, 2, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0, hspace=0)
        
        #-----------
        # Add pie plots
        #colors = ['orange', 'cornflowerblue', 'blue']
        colors = ['royalblue', 'orange', 'orangered']
        for morph_i in ['ETG', 'LTG']:
            
            y = np.array([plot_dict['%s-ETG' %morph_i]['other']+plot_dict['%s-LTG' %morph_i]['other'], plot_dict['%s-ETG' %morph_i]['minor']+plot_dict['%s-LTG' %morph_i]['minor'], plot_dict['%s-ETG' %morph_i]['major']+plot_dict['%s-LTG' %morph_i]['major']])
            mylabels = ['%s' %(' ' if y[0] == 0 else y[0]), '%s' %(' ' if y[1] == 0 else y[1]), '%s' %(' ' if y[2] == 0 else y[2])]
            
            if morph_i == 'ETG':
                ax_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_ETG.set_title('ETG', loc='center', pad=0, fontsize=8)
            if morph_i == 'LTG':
                ax_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_LTG.set_title('LTG', loc='center', pad=0, fontsize=8)
        
            
        #------------
        # Add legend
        legend_labels   = []
        legend_elements = []
        legend_colors   = []

        legend_labels.append('Major\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orangered')
        legend_labels.append('Minor\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orange')
        legend_labels.append('other')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('royalblue')
    
        ax_LTG.legend(handles=legend_elements, labels=legend_labels, loc='center left', bbox_to_anchor=(0.9, 0.5), labelspacing=1, frameon=False, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        #-----------
        # savefig
        if savefig:
            savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/misangle_origins/origins_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misangle_origins/origins_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    """  
    # Plotting ongoing fraction histogram
    fig, axs = plt.subplots(1, 1, figsize=[2.5, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #------------
    # plotting bar
    bottom = np.zeros(4)
    weight_counts = {
        'Major': np.array([plot_dict['ETG-ETG']['major']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['major']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['major']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['major']/len(plot_dict['LTG-LTG']['array'])]),
        'Minor': np.array([plot_dict['ETG-ETG']['minor']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['minor']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['minor']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['minor']/len(plot_dict['LTG-LTG']['array'])]),
        'Other': np.array([plot_dict['ETG-ETG']['other']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['other']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['other']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['other']/len(plot_dict['LTG-LTG']['array'])])}
    bar_x = []
    for morph_i in set_origins_morph:
        if morph_i == 'ETG-ETG':
            bar_x.append('ETG →\nETG')
        if morph_i == 'LTG-ETG':
            bar_x.append('LTG →\nETG')
        if morph_i == 'ETG-LTG':
            bar_x.append('ETG →\nLTG')
        if morph_i == 'LTG-LTG':
            bar_x.append('LTG →\nLTG')
    
    for boolean, weight_count in weight_counts.items():
        axs.bar(bar_x, weight_count, width=0.5, bottom=bottom, edgecolor='none', alpha=0.5, label=boolean)
        axs.bar(bar_x, weight_count, width=0.5, bottom=bottom, facecolor='none', edgecolor='k', alpha=0.9, lw=0.7)
        bottom += weight_count
    
    
    #-------------
    ### Formatting
    axs.set_xlabel('Relaxation type')
    axs.set_ylabel('Percentage of misalignments')
    #axs.set_ylim(0, 1)
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    
    #------------
    # Legend
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, labelspacing=0.1, handlelength=0.5)
    """
    

#-------------------------
# Plots scatter of gas fraction vs relax time
def _plot_timescale_gas_scatter_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['co-co'],          # which paths to use
                      set_gashist_min_trelax              = 0.5,                # removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
    print('  Using sample: ', len(ID_plot))
        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #-------------
    ### Plotting scatter
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-------------
    # Colourbar for kappa
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = axs.scatter(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    plt.colorbar(im1, ax=axs, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    if add_plot_gas_morph_median:

        # Bin hist data, find sigma percentiles
        bin_width = 0.1
        bins = np.arange(0, 0.7, bin_width)
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    #-------------
    ### Formatting
    axs.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
    axs.set_xlabel('$f_{\mathrm{%s}}(<r_{50})$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    axs.set_ylim(0, 4.5)
    axs.set_yticks(np.arange(0, 4.1, 1))
    axs.set_xlim(0, 0.6)
    axs.set_xticks(np.arange(0, 0.61, 0.1))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    ### Savefig
    savefig_txt_2 = savefig_txt    
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned_gas/scatter_trelax_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/scatter_trelax_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt_2, file_format)) 
    if showfig:
        plt.show()
    plt.close()        
# tdyn
def _plot_timescale_gas_scatter_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['co-co'],          # which paths to use
                      set_gashist_min_trelax              = 0.5,                # removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
    print('  Using sample: ', len(ID_plot))
        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #-------------
    ### Plotting scatter
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-------------
    # Colourbar for kappa
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = axs.scatter(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    plt.colorbar(im1, ax=axs, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    if add_plot_gas_morph_median:

        # Bin hist data, find sigma percentiles
        bin_width = 0.1
        bins = np.arange(0, 0.7, bin_width)
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    #-------------
    ### Formatting
    axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_xlabel('$f_{\mathrm{%s}}(<r_{50})$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    axs.set_yscale('log')
    axs.set_ylim(0.5, 35)
    axs.set_yticks([1, 10])
    axs.set_yticklabels(['1', '10'])
    axs.set_xlim(0, 0.6)
    axs.set_xticks(np.arange(0, 0.61, 0.1))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    ### Savefig
    savefig_txt_2 = savefig_txt    
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned_gas/scatter_tdyn_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/scatter_tdyn_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt_2, file_format)) 
    if showfig:
        plt.show()
    plt.close()        
# ttorque
def _plot_timescale_gas_scatter_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['co-co'],          # which paths to use
                      set_gashist_min_trelax              = 0.5,                # removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
    print('  Using sample: ', len(ID_plot))
        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #-------------
    ### Plotting scatter
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-------------
    # Colourbar for kappa
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = axs.scatter(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    plt.colorbar(im1, ax=axs, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    if add_plot_gas_morph_median:

        # Bin hist data, find sigma percentiles
        bin_width = 0.1
        bins = np.arange(0, 0.7, bin_width)
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    #-------------
    ### Formatting
    axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_xlabel('$f_{\mathrm{%s}}(<r_{50})$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    axs.set_yscale('log')
    axs.set_ylim(0.2, 25)
    axs.set_yticks([0.2, 1, 10])
    axs.set_yticklabels(['0.2', '1', '10'])
    axs.set_xlim(0, 0.6)
    axs.set_xticks(np.arange(0, 0.61, 0.1))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    ### Savefig
    savefig_txt_2 = savefig_txt    
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned_gas/scatter_ttorque_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/scatter_ttorque_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt_2, file_format)) 
    if showfig:
        plt.show()
    plt.close()        
         
# Plots histogram of different gas fraction vs relax time
def _plot_timescale_gas_histogram_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['enter below'],          # which paths to use
                      set_gashist_min_trelax              = 0.25,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_gas_morph_median         = False,
                        add_plot_errorbars                = True,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                        gas_fraction_limits               = [0.1, 0.3],         # [ < lower - upper < ] e.g. [0.2, 0.4] means <0.2, 0.2-0.4, >0.4
                      #--------------------
                      # General formatting
                      set_bin_limit_trelax                = 5,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.25,     # [ 0.25 / Gyr ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    if gas_fraction_type == 'gas':
        gas_1_df = df.loc[(df['Gas fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[0]) & (df['Gas fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[1])]
    elif gas_fraction_type == 'gas_sf':
        gas_1_df = df.loc[(df['SF fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['SF fraction'] > gas_fraction_limits[0]) & (df['SF fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['SF fraction'] > gas_fraction_limits[1])]
    
    
    
    #print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tGas %s fractions:' %('' if gas_fraction_type == 'gas' else 'SF'))
    print('\t.  0 - %.2f:    ' %(gas_fraction_limits[0]), len(gas_1_df))
    print('\t%.2f - %.2f:    ' %(gas_fraction_limits[0], gas_fraction_limits[1]), len(gas_2_df))
    print('\t%.2f +    :     ' %(gas_fraction_limits[1]), len(gas_3_df))
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for gas_df, plot_color in zip([gas_1_df, gas_2_df, gas_3_df], ['turquoise', 'teal', 'mediumblue']):
        # Add hist
        axs.hist(gas_df['Relaxation time'], weights=np.ones(len(gas_df['Relaxation time']))/len(gas_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.7, lw=0.7, edgecolor=plot_color)
        hist_n, _ = np.histogram(gas_df['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
    axs.set_ylabel('Percentage of misalignments')
    axs.set_ylim(bottom=0.001)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$0.0<f_{\mathrm{%s}}<%.1f$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('turquoise')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<%.1f$' %(gas_fraction_limits[0], 'gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('teal')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<1.0$' %(gas_fraction_limits[1], 'gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('mediumblue')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
    savefig_txt_2 = savefig_txt    
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned_gas/%strelax_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/%strelax_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt_2, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_timescale_gas_histogram_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['enter below'],          # which paths to use
                      set_gashist_min_trelax              = 0.25,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                        gas_fraction_limits               = [0.1, 0.3],         # [ < lower - upper < ] e.g. [0.2, 0.4] means <0.2, 0.2-0.4, >0.4
                        add_plot_errorbars                = True,
                      #--------------------
                      # General formatting
                      set_bin_limit_tdyn                  = 35,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 1,        # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):


    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    if gas_fraction_type == 'gas':
        gas_1_df = df.loc[(df['Gas fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[0]) & (df['Gas fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[1])]
    elif gas_fraction_type == 'gas_sf':
        gas_1_df = df.loc[(df['SF fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['SF fraction'] > gas_fraction_limits[0]) & (df['SF fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['SF fraction'] > gas_fraction_limits[1])]
    
    
    
    #print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tGas %s fractions:' %('' if gas_fraction_type == 'gas' else 'SF'))
    print('\t.  0 - %.2f:    ' %(gas_fraction_limits[0]), len(gas_1_df))
    print('\t%.2f - %.2f:    ' %(gas_fraction_limits[0], gas_fraction_limits[1]), len(gas_2_df))
    print('\t%.2f +    :     ' %(gas_fraction_limits[1]), len(gas_3_df))
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for gas_df, plot_color in zip([gas_1_df, gas_2_df, gas_3_df], ['turquoise', 'teal', 'mediumblue']):
        # Add hist
        axs.hist(gas_df['Relaxation time'], weights=np.ones(len(gas_df['Relaxation time']))/len(gas_df['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.7, lw=0.7, edgecolor=plot_color)
        hist_n, _ = np.histogram(gas_df['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Percentage of misalignments')
    axs.set_ylim(bottom=0.001)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$0.0<f_{\mathrm{%s}}<%.1f$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('turquoise')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<%.1f$' %(gas_fraction_limits[0], 'gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('teal')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<1.0$' %(gas_fraction_limits[1], 'gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('mediumblue')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
    savefig_txt_2 = savefig_txt    
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned_gas/%stdyn_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/%stdyn_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt_2, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_timescale_gas_histogram_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['enter below'],          # which paths to use
                      set_gashist_min_trelax              = 0.25,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                        gas_fraction_limits               = [0.1, 0.3],         # [ < lower - upper < ] e.g. [0.2, 0.4] means <0.2, 0.2-0.4, >0.4
                        add_plot_errorbars                = True,
                      #--------------------
                      # General formatting
                      set_bin_limit_ttorque               = 12,       # [ None / multiples ]
                      set_bin_width_ttorque               = 0.5,      # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):


    #-------------------------
    # Average timescales from input (for use in metadata)
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
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    if gas_fraction_type == 'gas':
        gas_1_df = df.loc[(df['Gas fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[0]) & (df['Gas fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[1])]
    elif gas_fraction_type == 'gas_sf':
        gas_1_df = df.loc[(df['SF fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['SF fraction'] > gas_fraction_limits[0]) & (df['SF fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['SF fraction'] > gas_fraction_limits[1])]
    
    
    
    #print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tGas %s fractions:' %('' if gas_fraction_type == 'gas' else 'SF'))
    print('\t.  0 - %.2f:    ' %(gas_fraction_limits[0]), len(gas_1_df))
    print('\t%.2f - %.2f:    ' %(gas_fraction_limits[0], gas_fraction_limits[1]), len(gas_2_df))
    print('\t%.2f +    :     ' %(gas_fraction_limits[1]), len(gas_3_df))
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for gas_df, plot_color in zip([gas_1_df, gas_2_df, gas_3_df], ['turquoise', 'teal', 'mediumblue']):
        # Add hist
        axs.hist(gas_df['Relaxation time'], weights=np.ones(len(gas_df['Relaxation time']))/len(gas_df['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(gas_df['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Percentage of misalignments')
    axs.set_ylim(bottom=0.001)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$0.0<f_{\mathrm{%s}}<%.1f$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('turquoise')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<%.1f$' %(gas_fraction_limits[0], 'gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('teal')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<1.0$' %(gas_fraction_limits[1], 'gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('mediumblue')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
    savefig_txt_2 = savefig_txt    
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned_gas/%sttorque_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/%sttorque_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt_2, file_format)) 
    if showfig:
        plt.show()
    plt.close()      



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Set starting parameters
load_csv_file_in = '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph'
plot_annotate_in                                           = False
savefig_txt_in   = load_csv_file_in                # [ 'manual' / load_csv_file_in ] 'manual' will prompt txt before saving

#'_20Thresh_30Peak_normalLatency_anyMergers_anyMorph',         
#'_20Thresh_30Peak_normalLatency_anyMergers_anyMorph'                                                                                                                                               
#'_20Thresh_30Peak_normalLatency_anyMergers_hardMorph'                                                                                                                                               
#'_20Thresh_30Peak_normalLatency_anyMergers_ETG-ETG'                                                                                                                                                 
#'_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010'

#  False 
# 'ETG → ETG' , 'LTG → LTG'
#  r'ETG ($\bar{\kappa}_{\mathrm{co}}^{\mathrm{*}} < 0.35$)'
# '$t_{\mathrm{relax}}>3\bar{t}_{\mathrm{torque}}$'    



#==================================================================================================================================
misalignment_tree, misalignment_input, summary_dict = _extract_tree(load_csv_file=load_csv_file_in, plot_annotate=plot_annotate_in)
#==================================================================================================================================



# SAMPLE MASS
"""_plot_sample_hist(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt=savefig_txt_in,
                            showfig = True,
                            savefig = False)"""

# TIMESCALE HISTOGRAMS
"""_plot_timescale_histogram(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                            set_plot_histogram_log            = False,    # set yaxis as log
                              add_inset                       = True,
                            showfig = True,
                            savefig = False)"""
"""_plot_tdyn_histogram(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                            set_plot_histogram_log            = False,    # set yaxis as log
                              add_inset                       = True, 
                            showfig = True,
                            savefig = False) """   
"""_plot_ttorque_histogram(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                            set_plot_histogram_log            = False,    # set yaxis as log
                              add_inset                       = True, 
                            showfig = True,
                            savefig = False)"""

# STACKED SINGLE
"""_plot_stacked_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""
"""_plot_stacked_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""
"""_plot_stacked_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""


# STACKED 2x2
"""_plot_stacked_trelax_2x2(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""
"""_plot_stacked_tdyn_2x2(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""
"""_plot_stacked_ttorque_2x2(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""
                            

# PLOT BOX AND WHISKER OF RELAXATION DISTRIBUTIONS
"""_plot_box_and_whisker_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_whisker_morphs = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = False)"""
"""_plot_box_and_whisker_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_whisker_morphs = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = False)"""
"""_plot_box_and_whisker_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_whisker_morphs = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = False)"""


# PLOT DELTA ANGLE, LOOKS AT PEAK ANGLE FROM 0
"""_plot_offset_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            use_offset_morphs    = True,
                              set_offset_morphs  = ['LTG-LTG', 'ETG-ETG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = False)"""
"""_plot_offset_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            use_offset_morphs    = True,
                              set_offset_morphs  = ['LTG-LTG', 'ETG-ETG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = False)"""
"""_plot_offset_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            use_offset_morphs    = True,
                              set_offset_morphs  = ['LTG-LTG', 'ETG-ETG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = False)"""


# AVERAGE HALO MISALIGNMENT WITH RELAXTIME
"""_plot_halo_misangle_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_trelax = 0,
                            showfig = True,
                            savefig = False)"""
"""_plot_halo_misangle_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_tdyn = 0,
                            showfig = True,
                            savefig = False)"""
"""_plot_halo_misangle_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_ttorque = 0,
                            showfig = True,
                            savefig = False)"""
"""_plot_halo_misangle_manual(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_trelax = 0,
                            showfig = True,
                            savefig = False)"""


# PLOTS RAW SCATTER OF GAS FRACTION WITH RELAXATION TIME
"""_plot_timescale_gas_scatter_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter'],
                            set_gashist_min_trelax              = 0.25,
                              add_plot_gas_morph_median         = True,
                            showfig = True,
                            savefig = False)"""
"""_plot_timescale_gas_scatter_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter'],
                            set_gashist_min_trelax              = 0.25,
                              add_plot_gas_morph_median         = True,
                            showfig = True,
                            savefig = False)"""
"""_plot_timescale_gas_scatter_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter'],
                            set_gashist_min_trelax              = 0.25,
                              add_plot_gas_morph_median         = True,
                            showfig = True,
                            savefig = False)"""


# PLOTS HISTOGRAM OF GAS FRACTION WITH RELAXATION TIME
"""_plot_timescale_gas_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter'],
                            set_gashist_min_trelax              = 0.25,
                              add_plot_gas_morph_median         = False,
                            showfig = True,
                            savefig = True)"""
"""_plot_timescale_gas_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter'],
                            set_gashist_min_trelax              = 0.25,
                              add_plot_gas_morph_median         = False,
                            showfig = True,
                            savefig = True)"""
_plot_timescale_gas_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter'],
                            set_gashist_min_trelax              = 0.25,
                              add_plot_gas_morph_median         = False,
                            showfig = True,
                            savefig = True)
                            


# PLOTS HISTOGRAM OF GAS FRACTION WITH CENTRALS VS SATELLITES

                            
#--------------------------------
# SUITED FOR 1010 SAMPLE AND ABOVE   

# PLOT NUMBER OF MERGERS WITH RELAXATION TIME, SUITED FOR 1010 SAMPLE AND ABOVE
"""_plot_merger_count_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = False)"""
"""_plot_merger_count_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = False)"""
"""_plot_merger_count_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = False)"""


# AVERAGE HALO MISALIGNMENT BEFORE MISALIGNMENT WITH FRACTIONAL OCCURENCE
"""_plot_halo_misangle_pre_frac(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_misanglepre_type = ['co-co', 'co-counter'],           # [ 'co-co', 'co-counter' ]  or False
                            showfig = True,
                            savefig = False)"""

  
# PLOTS STACKED BAR CHART. WILL USE use_alt_merger_criteria FROM EARLIER TO FIND MERGERS,          
"""_plot_origins(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            # Mergers
                            use_alt_merger_criteria = True,
                              half_window         = 0.3,      # [ 0.2 / +/-Gyr ] window centred on first misaligned snap to look for mergers
                              min_ratio           = 0.1,   
                              merger_lookback_time = 2,       # Gyr, number of years to check for peak stellar mass
                            showfig = True,
                            savefig = False)"""
                            
#====================================














#==================================================================================================
### OLD PLOTS
# Plot delta angle. Looks at peak angle from 180
"""if plot_delta_timescale:
        relaxationtime_plot = []
        angles_plot = []
        err_plot_l  = []
        err_plot_u  = []
        other_plot  = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # Find index of peak misalignment from where it relaxes to (-co or -counter)
            all_angles = np.array(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
            index_peak = misalignment_tree['%s' %ID_i]['index_peak']
            angle_peak = misalignment_tree['%s' %ID_i]['angle_peak']
            
            angles_plot.append(angle_peak)
            #angles_plot.append(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_peak])
            err_plot_l.append(misalignment_tree['%s' %ID_i]['%s_err' %use_angle][index_peak][0] - misalignment_tree['%s' %ID_i][use_angle][index_peak])
            err_plot_u.append(misalignment_tree['%s' %ID_i]['%s_err' %use_angle][index_peak][1] - misalignment_tree['%s' %ID_i][use_angle][index_peak])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            other_plot.append(np.average(np.array(misalignment_tree['%s' %ID_i][plot_delta_color][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']]), weights=np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']-1:misalignment_tree['%s' %ID_i]['index_r']-1]) - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']])))
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            
        
        #-------------
        # Stats
        res = stats.spearmanr(angles_plot, relaxationtime_plot)
        print('\n--------------------------------------')
        print('DELTA TIMESCALE SPEARMAN:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=20, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
        
        axs.errorbar(relaxationtime_plot, angles_plot, yerr=[np.abs(err_plot_l), err_plot_u], ls='none', fmt='o', capsize=3, elinewidth=0.7, markeredgewidth=1, color='k', ecolor='grey', ms=1, alpha=0.5)
        im1 = axs.scatter(relaxationtime_plot, angles_plot, c=other_plot, s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3)
        
        #-------------
        # Colorbar
        cax = plt.axes([0.98, 0.20, 0.03, 0.71])
        plt.colorbar(mapper, cax=cax, label='time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$', extend='max')
        
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_yticks(np.arange(0, 181, 30))
        axs.set_ylim(0, 180)
        axs.set_xlim(0, set_bin_limit_trelax)
        axs.set_xlabel('Relaxation time (Gyr)')
        axs.set_ylabel('Peak misalignment angle')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        axs.set_title(r'$\rho$ = %s, p-value = %s' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
        
        #-----------
        ### Annotations
        if set_add_GalaxyIDs:
            for ID_plot_i, x_i, y_i in zip(ID_plot, angles_plot, relaxationtime_plot):
                axs.text(x_i+5, y_i+0.1, '%s' %ID_plot_i, fontsize=7)
                
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/delta_misangle_t_relax/trelax_delta_misangle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/delta_misangle_t_relax/trelax_delta_misangle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
"""
    
""" 
    #-----------------------------
    # Plot scatter and extract spearman from it
    plot_spearman         = False,                     # Available variables: halomass, stelmass, gasmass, sfmass, nsfmass, dmmass,   sfr, ssfr,   stars_Z, gas_Z, inflow_Z, outflow_Z
                                                      #                      kappa_stars, kappa_gas, kappa_sf,   ellip, triax,   rad, radproj, rad_sf
                                                      #                      inflow_rate, outflow_rate, stelmassloss_rate,   s_inflow_rate, s_outflow_rate,   inflow_cum, outflow_cum, 
                                                      #                      angle_peak,   number_of_mergers,  relaxation_time
      plot_spearman_x     = 'stelmass',        # [ array name ]
      plot_spearman_y     = 'sfmass',            # [ array name ]
      plot_spearman_c     = 'kappa_stars',               # [ None / array name]
        inflow_skip       = 0.5,                      # [ Gyr ] time to ignore inflow, after which to sample. Relaxations below this time are automatically filetred out
    #-----------------------------
    # Plots trelax - tdyn, which should be proportional
    plot_trelax_tdyn      = False,
    # Plots log(ellip) - log(trelax/tdyn), which should have a (rough) gradient of -1 
    plot_ellip_trelaxtdyn = False,
"""
# Plot spearman
"""if plot_spearman:
        
        # Collect datapoints
        x_array = []
        y_array = []
        c_array = []
        for ID_i in misalignment_tree.keys():
            
            #--------------
            # IF we're using inflows, skip galaxies that dont have enough misalignment snipshots
            if not set(['inflow_rate', 'inflow_cum', 's_inflow_rate', 's_inflow_cum', 'outflow_rate']).isdisjoint([plot_spearman_x, plot_spearman_y, plot_spearman_c]):
                if misalignment_tree['%s' %ID_i]['relaxation_time'] < inflow_skip:
                    continue
            
            #--------------
            # Isolate inflow_rate, inflow_cum, s_inflow_rate, s_inflow_cum, outflow_rate, peak_angle, number_of_mergers
            if (plot_spearman_x not in misalignment_tree['%s' %ID_i].keys()) or (plot_spearman_x == 'inflow_rate') or (plot_spearman_x == 'outflow_rate') or (plot_spearman_x == 's_inflow_rate') or (plot_spearman_x == 's_outflow_rate'):
            
                if plot_spearman_x == 'inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_x == 'outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_x == 'inflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.sum(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_x == 'outflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.sum(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_x == 's_inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_x == 's_outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_x == 'number_of_mergers':
                    # count mergers during relaxation
                    check = 0
                    for merger_i in misalignment_tree['%s' %ID_i]['merger_ratio_stars'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]:
                        if len(merger_i) > 0:
                            if max(merger_i) > 0.1:
                                check += 1
                
                    x_array.append(check)
            else:
                if plot_spearman_x in ['angle_peak', 'relaxation_time', 'relaxation_type', 'relaxation_morph', 'misalignment_morph']:
                    x_array.append(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_x]) 
                else:
                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_x][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                
            if (plot_spearman_y not in misalignment_tree['%s' %ID_i].keys()) or (plot_spearman_y == 'inflow_rate') or (plot_spearman_y == 'outflow_rate') or (plot_spearman_y == 's_inflow_rate') or (plot_spearman_x == 's_outflow_rate'):
            
                if plot_spearman_y == 'inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_y == 'outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_y == 'inflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.sum(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_y == 'outflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.sum(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_y == 's_inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_y == 's_outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_y == 'number_of_mergers':
                    # count mergers during relaxation
                    check = 0
                    for merger_i in misalignment_tree['%s' %ID_i]['merger_ratio_stars'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]:
                        if len(merger_i) > 0:
                            if max(merger_i) > 0.1:
                                check += 1
                
                    y_array.append(check)
            else:
                if plot_spearman_y in ['angle_peak', 'relaxation_time', 'relaxation_type', 'relaxation_morph', 'misalignment_morph']:
                    y_array.append(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_y]) 
                else:
                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_y][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                    
            if plot_spearman_c != None:
                if (plot_spearman_c not in misalignment_tree['%s' %ID_i].keys()) or (plot_spearman_c == 'inflow_rate') or (plot_spearman_c == 'outflow_rate') or (plot_spearman_c == 's_inflow_rate') or (plot_spearman_x == 's_outflow_rate'):
                
                    if plot_spearman_c == 'inflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    elif plot_spearman_c == 'outflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    
                    elif plot_spearman_c == 'inflow_cum':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.sum(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    elif plot_spearman_c == 'outflow_cum':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.sum(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    
                    elif plot_spearman_c == 's_inflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    elif plot_spearman_c == 's_outflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    
                    elif plot_spearman_c == 'number_of_mergers':
                        # count mergers during relaxation
                        check = 0
                        for merger_i in misalignment_tree['%s' %ID_i]['merger_ratio_stars'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]:
                            if len(merger_i) > 0:
                                if max(merger_i) > 0.1:
                                    check += 1
                    
                        c_array.append(check)  
                else:
                    if plot_spearman_c in ['angle_peak', 'relaxation_time', 'relaxation_type', 'relaxation_morph', 'misalignment_morph']:
                        c_array.append(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_c]) 
                    else:
                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_c][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                
        
        #-------------
        # Making logs of values that need them
        if plot_spearman_x in ['halomass', 'stelmass', 'gasmass', 'sfmass', 'nsfmass', 'dmmass', 'sfr', 'ssfr', 'inflow_rate', 'outflow_rate', 's_inflow_rate', 's_outflow_cum', 's_inflow_rate', 's_outflow_rate', 'stelmassloss_rate']:
            x_array = np.log10(np.array(x_array))
        if plot_spearman_y in ['halomass', 'stelmass', 'gasmass', 'sfmass', 'nsfmass', 'dmmass', 'sfr', 'ssfr', 'inflow_rate', 'outflow_rate', 's_inflow_rate', 's_outflow_cum', 's_inflow_rate', 's_outflow_rate', 'stelmassloss_rate']:
            y_array = np.log10(np.array(y_array))
        if plot_spearman_c != None:
            if plot_spearman_c in ['halomass', 'stelmass', 'gasmass', 'sfmass', 'nsfmass', 'dmmass', 'sfr', 'ssfr', 'inflow_rate', 'outflow_rate', 's_inflow_rate', 's_outflow_cum', 's_inflow_rate', 's_outflow_rate', 'stelmassloss_rate']:
                c_array = np.log10(np.array(c_array))
        
        
        #------------
        # Stats
        res = stats.spearmanr(x_array, y_array)
        print('\n--------------------------------------')
        print('SPEARMAN:  %s  -  %s' %(plot_spearman_x, plot_spearman_y))
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
    
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
         #-----------
        # Colormap
        if plot_spearman_c != None:
            
            if plot_spearman_c == 'halomass':
                vmin = 11
                vmax = 15
                label = 'log$_{10}$ M$_{\mathrm{halo}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'stelmass':
                vmin = 8
                vmax = 12.5
                label = 'log$_{10}$ M$_{\mathrm{*}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'gasmass':
                vmin = 7.5
                vmax = 11
                label = 'log$_{10}$ M$_{\mathrm{gas}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'sfmass':
                vmin = 7.5
                vmax = 11
                label = 'log$_{10}$ M$_{\mathrm{SF}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'nsfmass':
                vmin = 7
                vmax = 11
                label = 'log$_{10}$ M$_{\mathrm{NSF}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'dmmass':
                vmin = 8
                vmax = 13.5
                label = 'log$_{10}$ M$_{\mathrm{DM}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'sfr':
                vmin = -5
                vmax = 2
                label = 'log$_{10}$ SFR $(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'ssfr':
                vmin = -12
                vmax = -8
                label = 'log$_{10}$ sSFR $(\mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'stars_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{*}}$'
            elif plot_spearman_c == 'gas_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{gas}}$'
            elif plot_spearman_c == 'inflow_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{inflow}}$'
            elif plot_spearman_c == 'outflow_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{outflow}}$'
            elif plot_spearman_c == 'kappa_stars':
                vmin = 0.2
                vmax = 0.8
                label = '$\kappa_{\mathrm{co}}^{\mathrm{*}}$'
            elif plot_spearman_c == 'kappa_gas':
                vmin = 0
                vmax = 1
                label = '$\kappa_{\mathrm{co}}^{\mathrm{gas}}$'
            elif plot_spearman_c == 'kappa_sf':
                vmin = 0
                vmax = 1
                label = '$\kappa_{\mathrm{co}}^{\mathrm{SF}}$'
            elif plot_spearman_c == 'kappa_nsf':
                vmin = 0
                vmax = 1
                label = '$\kappa_{\mathrm{co}}^{\mathrm{NSF}}$'
            elif plot_spearman_c == 'ellip':
                vmin = 0
                vmax = 1
                label = '$\epsilon$'
            elif plot_spearman_c == 'triax':
                vmin = 0
                vmax = 1
                label = '$T$'
            elif plot_spearman_c == 'rad':
                vmin = 0
                vmax = 15
                label = '$r_{1/2}$ (pkpc)'
            elif plot_spearman_c == 'radproj':
                vmin = 0
                vmax = 15
                label = '$r_{1/2,\mathrm{z}}$ (pkpc)'
            elif plot_spearman_c == 'rad_sf':
                vmin = 0
                vmax = 15
                label = '$r_{1/2,\mathrm{SF}}$ (pkpc)'
            elif plot_spearman_c == 'inflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'outflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged gas outflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'stelmassloss_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged stellar mass-loss\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 's_inflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged specific gas\ninflow $\mathrm{yr}^{-1})$'
            elif plot_spearman_c == 's_outflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged specific gas\noutflow $\mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'inflow_cum':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ cumulative gas inflow\n$(\mathrm{M}_{\odot})$'
            elif plot_spearman_c == 'outflow_cum':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ cumulative gas outflow\n$(\mathrm{M}_{\odot})$'
            elif plot_spearman_c == 'angle_peak':
                vmin = 0
                vmax = 180
                label = 'Peak misalignment angle'
            elif plot_spearman_c == 'number_of_mergers':
                vmin = 0
                vmax = max(c_array)
                label = 'Number of mergers'
            elif plot_spearman_c == 'relaxation_time':
                vmin = 0
                vmax = 4
                label = 'Relaxation time (Gyr)'
            
            #-------------
            # Normalise colormap
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
            
            #-------------
            # Colorbar
            #cax = plt.axes([0.98, 0.20, 0.03, 0.71])
            plt.colorbar(mapper, ax=axs, label=label, extend='max')
    
         #-----------
        # Plot scatter
        if plot_spearman_c != None:
            axs.scatter(x_array, y_array, c=c_array, s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.5)
        else:
            axs.scatter(x_array, y_array, c=c_array, s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.5)
    
        
        #-----------
        ### General formatting
        if plot_spearman_x != None and plot_spearman_y != None:
            
            # x-axis labels
            if plot_spearman_x == 'halomass':
                axs.set_xticks(np.arange(11, 15.1, 1))
                axs.set_xlim(11, 15)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{halo}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'stelmass':
                axs.set_xticks(np.arange(9, 12.5, 1))
                axs.set_xlim(9, 12.5)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{*}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'gasmass':
                axs.set_xticks(np.arange(7, 11.1, 1))
                axs.set_xlim(7.5, 11)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{gas}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'sfmass':
                axs.set_xticks(np.arange(7.5, 11.1, 1))
                axs.set_xlim(7.5, 12)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{SF}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'nsfmass':
                axs.set_xticks(np.arange(7, 11.1, 1))
                axs.set_xlim(7, 11)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{NSF}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'dmmass':
                axs.set_xticks(np.arange(8, 13.5, 1))
                axs.set_xlim(8, 13.5)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{DM}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'sfr':
                axs.set_xticks(np.arange(-2, 3.1, 1))
                axs.set_xlim(-2, 3)
                axs.set_xlabel('log$_{10}$ SFR $(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'ssfr':
                axs.set_xticks(np.arange(-12, -7.9, 1))
                axs.set_xlim(-12, -8)
                axs.set_xlabel('log$_{10}$ sSFR $(\mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'stars_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{*}}$')
            elif plot_spearman_x == 'gas_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{gas}}$')
            elif plot_spearman_x == 'inflow_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{inflow}}$')
            elif plot_spearman_x == 'outflow_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{outflow}}$')
            elif plot_spearman_x == 'kappa_stars':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{*}}$')
            elif plot_spearman_x == 'kappa_gas':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{gas}}$')
            elif plot_spearman_x == 'kappa_sf':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
            elif plot_spearman_x == 'kappa_stars':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{NSF}}$')
            elif plot_spearman_x == 'ellip':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\epsilon$')
            elif plot_spearman_x == 'triax':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$T$')
            elif plot_spearman_x == 'rad':
                axs.set_xticks(np.arange(0, 15.1, 3))
                axs.set_xlim(0, 15)
                axs.set_xlabel('$r_{1/2}$ (pkpc)')
            elif plot_spearman_x == 'radproj':
                axs.set_xticks(np.arange(0, 15.1, 3))
                axs.set_xlim(0, 15)
                axs.set_xlabel('$r_{1/2,\mathrm{z}}$ (pkpc)')
            elif plot_spearman_x == 'rad_sf':
                axs.set_xticks(np.arange(0, 15.1, 3))
                axs.set_xlim(0, 15)
                axs.set_xlabel('$r_{1/2,\mathrm{SF}}$ (pkpc)')
            elif plot_spearman_x == 'inflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'outflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged gas outflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'stelmassloss_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged stellar mass-loss\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 's_inflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged specific\ngas inflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_x == 's_outflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged specific\ngas outflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'inflow_cum':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ cumulative gas inflow\n$(\mathrm{M}_{\odot})$')
            elif plot_spearman_x == 'outflow_cum':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ cumulative gas outflow\n$(\mathrm{M}_{\odot})$')
            elif plot_spearman_x == 'angle_peak':
                axs.set_xticks(np.arange(0, 181, 30))
                axs.set_xlim(0, 180)
                axs.set_xlabel('Peak misalignment angle')
            elif plot_spearman_x == 'number_of_mergers':
                #axs.set_xticks(np.arange(0, max(c_array)+0.1, 1))
                #axs.set_xlim(0, max(c_array))
                axs.set_xlabel('Number of mergers')
            elif plot_spearman_x == 'relaxation_time':
                axs.set_xticks(np.arange(0, 4.1, 0.5))
                axs.set_xlim(0, 4)
                axs.set_xlabel('Relaxation time (Gyr)')
                
            # y-axis labels
            if plot_spearman_y == 'halomass':
                axs.set_yticks(np.arange(11, 15.1, 1))
                axs.set_ylim(11, 15)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{halo}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'stelmass':
                axs.set_yticks(np.arange(9, 12.5, 1))
                axs.set_ylim(9, 12.5)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{*}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'gasmass':
                axs.set_yticks(np.arange(7, 11.1, 1))
                axs.set_ylim(7.5, 11)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{gas}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'sfmass':
                axs.set_yticks(np.arange(7, 11.1, 1))
                axs.set_ylim(7.5, 11)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{SF}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'nsfmass':
                axs.set_yticks(np.arange(7.5, 11.1, 1))
                axs.set_ylim(7.5, 11)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{NSF}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'dmmass':
                axs.set_yticks(np.arange(8, 13.5, 1))
                axs.set_ylim(8, 13.5)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{DM}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'sfr':
                axs.set_yticks(np.arange(-2, 3.1, 1))
                axs.set_ylim(-2, 3)
                axs.set_ylabel('log$_{10}$ SFR $(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'ssfr':
                axs.set_yticks(np.arange(-12, -7.9, 1))
                axs.set_ylim(-12, -8)
                axs.set_ylabel('log$_{10}$ sSFR $(\mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'stars_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{*}}$')
            elif plot_spearman_y == 'gas_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{gas}}$')
            elif plot_spearman_y == 'inflow_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{inflow}}$')
            elif plot_spearman_y == 'outflow_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{outflow}}$')
            elif plot_spearman_y == 'kappa_stars':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{*}}$')
            elif plot_spearman_y == 'kappa_gas':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{gas}}$')
            elif plot_spearman_y == 'kappa_sf':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
            elif plot_spearman_y == 'kappa_stars':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{NSF}}$')
            elif plot_spearman_y == 'ellip':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\epsilon$')
            elif plot_spearman_y == 'triax':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$T$')
            elif plot_spearman_y == 'rad':
                axs.set_yticks(np.arange(0, 15.1, 3))
                axs.set_ylim(0, 15)
                axs.set_ylabel('$r_{1/2}$ (pkpc)')
            elif plot_spearman_y == 'radproj':
                axs.set_yticks(np.arange(0, 15.1, 3))
                axs.set_ylim(0, 15)
                axs.set_ylabel('$r_{1/2,\mathrm{z}}$ (pkpc)')
            elif plot_spearman_y == 'rad_sf':
                axs.set_yticks(np.arange(0, 15.1, 3))
                axs.set_ylim(0, 15)
                axs.set_ylabel('$r_{1/2,\mathrm{SF}}$ (pkpc)')
            elif plot_spearman_y == 'inflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'outflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged gas outflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'stelmassloss_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged stellar mass-loss\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 's_inflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged specific gas\ninflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_y == 's_outflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged specific gas\noutflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'inflow_cum':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ cumulative gas inflow\n$(\mathrm{M}_{\odot}$)')
            elif plot_spearman_y == 'outflow_cum':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ cumulative gas outflow\n$(\mathrm{M}_{\odot}$()')
            elif plot_spearman_y == 'angle_peak':
                axs.set_yticks(np.arange(0, 181, 30))
                axs.set_ylim(0, 180)
                axs.set_ylabel('Peak misalignment angle')
            elif plot_spearman_y == 'number_of_mergers':
                #axs.set_yticks(np.arange(0, max(c_array)+0.1, 1))
                #axs.set_ylim(0, max(c_array))
                axs.set_ylabel('Number of mergers')
            elif plot_spearman_y == 'relaxation_time':
                axs.set_yticks(np.arange(0, 4.1, 0.5))
                axs.set_ylim(0, 4)
                axs.set_ylabel('Relaxation time (Gyr)')
            
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #-----------
        ### other
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        plt.tight_layout()
        
        #-----------
        #### Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/spearman_plots/%s%s-%s_%s_%s.%s" %(fig_dir, 'L100_', plot_spearman_x, plot_spearman_y, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/spearman_plots/%s%s-%sdelta_misangle_%s_%s.%s" %(fig_dir, 'L100_', plot_spearman_x, plot_spearman_y, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
"""
# Plots trelax - tdyn, which should be proportional
"""if plot_trelax_tdyn:  
        relaxationtime_plot = []
        dyntime_plot        = []
        ID_plot             = []
        for ID_i in misalignment_tree.keys():
            # Actual relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            # Average the dynamical time and ellip
            dyntime_plot.append(np.mean(np.array(misalignment_tree['%s' %ID_i]['relaxation_tdyn'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            

        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, dyntime_plot)
        print('\n--------------------------------------')
        print('SPEARMAN:  trelax/tdyn - ellip')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------') 
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        axs.scatter(relaxationtime_plot, dyntime_plot, s=10, zorder=99, c='k', edgecolors='k', linewidths=0.3, alpha=0.7)
        
        #------------
        # Annotation (gradient line of -1)
        print('\n   add bestfit?')
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(0, 5)
        axs.set_xlabel('Relaxation time (Gyr)')
        axs.set_yticks(np.arange(0, 2, 0.2))
        axs.set_ylim(0, 2)
        axs.set_ylabel('Dynamical time (Gyr)')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        #-----------
        ### Annotations
        #if set_add_GalaxyIDs:
        #    for ID_plot_i, x_i, y_i in zip(ID_plot, angles_plot, relaxationtime_plot):
        #        axs.text(x_i+5, y_i+0.1, '%s' %ID_plot_i, fontsize=7)
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/tdyn_plots/%stdyn_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/tdyn_plots/%stdyn_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
"""
    # Plots log(ellip) - log(trelax/tdyn), which should have a (rough) gradient of -1 
"""if plot_ellip_trelaxtdyn:
        relaxationtime_plot = []
        dyntime_plot        = []
        ellip_plot          = []
        ID_plot             = []
        for ID_i in misalignment_tree.keys():
            # Actual relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            # Average the dynamical time and ellip
            dyntime_plot.append(np.mean(np.array(misalignment_tree['%s' %ID_i]['relaxation_tdyn'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
            ellip_plot.append(np.mean(np.array(misalignment_tree['%s' %ID_i]['ellip'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            
        #-------------
        # Log values
        y_array = np.log10(np.divide(np.array(relaxationtime_plot), np.array(dyntime_plot)))
        ellip_plot = np.log10(np.array(ellip_plot))

        #-------------
        # Stats
        res = stats.spearmanr(ellip_plot, y_array)
        print('\n--------------------------------------')
        print('SPEARMAN:  trelax/tdyn - ellip')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------') 
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        axs.scatter(ellip_plot, y_array, s=10, zorder=99, c='k', edgecolors='k', linewidths=0.3, alpha=0.7)
        
        #------------
        # Annotation (gradient line of -1)
        print('\n   CHECK ANNOTATION')
        axs.plot([-1, 0], [0, -1], c='grey', alpha=0.5, ls='-')
        axs.text(0.5, -0.5, 'gradient = -1', rotation=45, c='grey')
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(-1, 0)
        axs.set_xlabel('log$_{10}$ $\epsilon$')
        #axs.set_yticks(np.arange(0, 181, 30))
        #axs.set_ylim(0, 180)
        axs.set_ylabel('log$_{10}$ $t_{\mathrm{relax}}/t_{\mathrm{dyn}}$')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        #-----------
        ### Annotations
        #if set_add_GalaxyIDs:
        #    for ID_plot_i, x_i, y_i in zip(ID_plot, angles_plot, relaxationtime_plot):
        #        axs.text(x_i+5, y_i+0.1, '%s' %ID_plot_i, fontsize=7)
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        # Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/tdyn_plots/%sellip_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/tdyn_plots/%sellip_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
"""
    
    

    