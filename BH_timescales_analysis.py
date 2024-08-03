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
from itertools import groupby
from operator import itemgetter
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
# Reads in tree and extracts galaxies from tree
ID_list = [108988077, 479647060, 21721896, 390595970, 401467650, 182125463, 192213531, 24276812, 116404995, 239808134, 215988755, 86715463, 6972011, 475772617, 374037507, 429352532, 441434976]
ID_list = [1361598, 1403994, 10421872, 17879310, 21200847, 21659372, 24053428, 182125501, 274449295]
ID_list = [21200847, 182125516, 462956141]
def _extract_sample(csv_tree = 'L100_BH_tree',
                    GalaxyID_list = None,             # [ None / ID_list ]
                    print_summary             = True,
                    #------------------------------
                    load_csv_file  = 'test_run_',   # [ 'file_name' / False ] load existing misalignment tree  
                    plot_annotate  = False,                                                  # [ False / 'ETG → ETG' 
                    #------------------------------
                    print_progress = False,
                    debug = False):
    
    
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    
    #------------------------------
    # Load previous csv if asked for
    dict_tree = json.load(open('%s/L100_BH_tree%s.csv' %(output_dir, load_csv_file), 'r'))
    BH_input           = dict_tree['BH_input']
    sample_input       = dict_tree['sample_input']
    output_input       = dict_tree['output_input']
    BH_tree            = dict_tree['BH_tree']
    
    if print_summary:
        print('\n===================================================')
        print('Loaded BH tree sample extracted with min %s Gyr window:' %BH_input['min_time'])
        print('  aligned:     ', len(BH_tree['aligned'].keys()))
        print('  misaligned:  ', len(BH_tree['misaligned'].keys()))
        print('  counter:     ', len(BH_tree['counter'].keys()))
    
    
    summary_dict = {}
    
    return BH_tree, BH_input, summary_dict

#--------------------------------
# Reads in tree and extracts galaxies from tree that meet criteria
def _refine_sample(BH_tree = None, BH_input = None, summary_dict = None,
                   print_summary = True,
                   print_checks  = False,
                   #====================================================================================================
                   # Matching criteria
                     check_for_subhalo_switch  = True,
                     check_for_BH_grow         = True,
                     use_CoP_BH                = True,       # CoP BH or largest within 1 hmr
                    
                   # Galaxy properties
                     min_window_size   = 1,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                   #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                     min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                     min_bhmass   = None,      max_bhmass   = None,
                     min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                     min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                  
                   # Mergers, looked for within range considered +/- halfwindow
                   use_merger_criteria = False,
                     half_window         = 0.3,      # [ 0.2 / +/-Gyr ] window centred on first misaligned snap to look for mergers
                     min_ratio           = 0.1,   
                     merger_lookback_time = 2,       # Gyr, number of years to check for peak stellar mass
                   #====================================================================================================
                   print_progress = False,
                   debug = False):
    
    #===================================================================================================
    # Go through sample
    BH_subsample = {'aligned': {}, 'misaligned': {}, 'counter': {}}
    
    
    ID_plot     = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_tree['%s' %galaxy_state].keys():
            
            stelmass_array = np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['stelmass'])
            if use_CoP_BH:
                bhmass_array   = np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['bh_mass'])
            #else:
            #    bhmass_array   = np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['bh_mass_alt'])
            sfr_array      = np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['sfr'])
            ssfr_array     = np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['ssfr'])
            

            # Check subhalo switch
            if check_for_subhalo_switch:
                # Check stelmass doesnt drop randomly
                subhalo_switch_array = []
                for ii, check_i in enumerate(stelmass_array):
                    if ii == 0:
                        check_i_previous = check_i
                        subhalo_switch_array.append(True)
                        continue
                    else:
                        # Ensure ratio between stelmasses doesnt drop by half or worse
                        if check_i/check_i_previous >= 0.5:
                            subhalo_switch_array.append(True)
                        else:
                            subhalo_switch_array.append(False)
                        check_i_previous = check_i
                subhalo_switch_array = np.array(subhalo_switch_array)
            else:
                subhalo_switch_array = np.full(len(stelmass_array), True)
                
            # Check BH switch
            if check_for_BH_grow:
                # Check bhmass doesnt drop randomly
                BH_grow_array = []
                for ii, check_i in enumerate(bhmass_array):
                    if ii == 0:
                        check_i_previous = check_i
                        BH_grow_array.append(True)
                        continue
                    else:                        
                        # Ensure ratio between BH doesnt drop by half or worse
                        if np.isnan(check_i) == True:
                            BH_grow_array.append(False)
                        elif check_i/check_i_previous >= 0.5:
                            BH_grow_array.append(True)
                            check_i_previous = check_i
                        else:
                            BH_grow_array.append(False)
                            check_i_previous = check_i
                BH_grow_array = np.array(BH_grow_array)
            else:
                BH_grow_array = np.full(len(stelmass_array), True)
                   
            # Mask out regions with matching stelmass, bhmass, sfr, ssfr, and have non-nan BH mass
            mask_window = np.where((stelmass_array > (0 if min_stelmass == None else min_stelmass)) & (stelmass_array < (1e99 if max_stelmass == None else max_stelmass)) & (bhmass_array > (0 if min_bhmass == None else min_bhmass)) & (bhmass_array < (1e99 if max_bhmass == None else max_bhmass)) & (np.isnan(bhmass_array) == False) & (subhalo_switch_array == True) & (BH_grow_array == True) & (sfr_array > (0 if min_sfr == None else min_sfr)) & (sfr_array < (1e99 if max_sfr == None else max_sfr)) & (ssfr_array > (0 if min_ssfr == None else min_ssfr)) & (ssfr_array < (1e99 if max_ssfr == None else max_ssfr)))
            
            # If regions exist, check if they are consecutive and long enough to be included
            if len(mask_window) > 0:
                test_snaps     = np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['SnapNum'])[mask_window]
                for k, g in groupby(enumerate(test_snaps), lambda ix : ix[0] - ix[1]):
                
                    adjacent_regions_snaps = list(map(itemgetter(1), g))
                    if len(adjacent_regions_snaps) < 2:
                        if print_checks:
                            print(' ')
                            print(adjacent_regions_snaps)
                            print('REJECTED: only 1 snap for region')
                        continue
                

                    # Find args of adjacent snaps
                    adjacent_regions_arg = np.nonzero(np.in1d(np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['SnapNum']), np.array(adjacent_regions_snaps)))[0]
                    if print_checks:
                        print('\nCurrent iteration counter:')
                        print(adjacent_regions_snaps)
                        print(adjacent_regions_arg)
            
                    #----------------------------------
                    # We have co_regions of args, now test if they meet the criteria
                    start_index = adjacent_regions_arg[0]
                    end_index   = adjacent_regions_arg[-1]
                    entry_duration = abs(np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['Lookbacktime'])[start_index] - np.array(BH_tree['%s' %galaxy_state]['%s'%ID_i]['Lookbacktime'])[end_index])
                
                    # Filter for duration min_window_size
                    if entry_duration < (0 if min_window_size == None else min_window_size):
                        if print_checks:
                            print('\nREJECTED: min_window_size not met')
                        continue
                        
                    #-------------------------------------
                    # If it passes, add to sample
                    index_start = adjacent_regions_arg[0]
                    index_stop  = adjacent_regions_arg[-1] + 1
                        
                    #-------------------------------------
                    # Apply merger criteria
            
                    
            
                    #-------------------------------------
                    # If it passes, add to sample
                    ID_plot.append(ID_i)
                    BH_subsample['%s' %galaxy_state].update({'%s' %ID_i: {'entry_duration': entry_duration}})
                    
                    # ADD INDEXES, NOT ENTIRE:
                    for array_name_i in BH_tree['%s' %galaxy_state]['%s' %ID_i].keys():
                        
                        # Skip single-entry
                        if array_name_i in ['entry_duration', 'stelmass_2hmr_max', 'stelmass_2hmr_min', 'sfr_2hmr_max', 'sfr_2hmr_min', 'ssfr_2hmr_max', 'ssfr_2hmr_min']:
                            continue
                            
                        array_i = np.array(BH_tree['%s' %galaxy_state]['%s' %ID_i]['%s' %array_name_i], dtype=object)[index_start:index_stop]
                        
                        # Update array
                        BH_subsample['%s' %galaxy_state]['%s' %ID_i].update({'%s' %array_name_i: array_i})
                        
        
    # Apply a merger criteria
    if use_merger_criteria:
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
    #==================================================================================================



    # Summary and K-S KS test
    if print_summary:
        print('\n===================================================')
        print('BH_tree sample vs BH_subsample extracted trimmed to min %s Gyr window:' %min_window_size)
        print('  aligned:      %s \t->\t%s' %(len(BH_tree['aligned'].keys()), len(BH_subsample['aligned'].keys())))
        print('  misaligned:   %s \t->\t%s' %(len(BH_tree['misaligned'].keys()), len(BH_subsample['misaligned'].keys())))
        print('  counter:      %s \t->\t%s' %(len(BH_tree['counter'].keys()), len(BH_subsample['counter'].keys())))
        print('  IDs collected: ', len(ID_plot))
    
    
    summary_dict_subsample = {}
    
    return BH_subsample, summary_dict_subsample


#--------------------------------
# Create plots of SFR and sSFR within our sample
def _BH_sample_SFR(BH_tree, BH_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Sample options
                        min_window_size   = 1,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                        #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                        min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                        min_bhmass   = None,      max_bhmass   = None,
                        min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                        min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                        # Mergers, looked for within range considered +/- halfwindow
                        use_merger_criteria = False,
                      # Plot options

                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-----------------------------------
    # Establish sub-sample we wish to focus on
    BH_subsample, summary_dict_subsample = _refine_sample(BH_tree = BH_tree, BH_input = BH_input, summary_dict = summary_dict,
                                                              min_window_size   = min_window_size,     
                                                              min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                              min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                              min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                              min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                              use_merger_criteria = use_merger_criteria)
    
    
    #===================================================================================================
    # Go through subsample
    stelmass_plot  = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    state_plot      = []
    ID_plot     = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # Append means
            stelmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])))
            sfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])))
            ssfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])))
            kappa_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])))
            state_plot.append(galaxy_state)
            ID_plot.append(ID_i)
            
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Morphology': kappa_plot, 'State': state_plot, 'GalaxyIDs': ID_plot})   
    print('\nSub-sample after processing:')
    print('  aligned:     ', len(df.loc[(df['State'] == 'aligned')]['GalaxyIDs']))
    print('  misaligned:  ', len(df.loc[(df['State'] == 'misaligned')]['GalaxyIDs']))
    print('  counter:     ', len(df.loc[(df['State'] == 'counter')]['GalaxyIDs']))
    
    #df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'ETG → ETG')]
    
    
    
    #------------------------
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict = {'aligned':'grey', 'misaligned':'r', 'counter':'b'}
    plt.scatter(np.log10(df['stelmass']), df['SFR'], s=0.1, c=[color_dict[i] for i in df['State']], edgecolor=[color_dict[i] for i in df['State']], marker='.', alpha=0.8)
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(9, 11.5)
    axs.set_ylim(0.01, 100)
    axs.set_yscale('log')
    axs.set_xlabel('Stellar mass 2r50')
    axs.set_ylabel('SFR [Msun/yr]')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('k')
    legend_labels.append('misaligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('r')
    legend_labels.append('counter')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('b')
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    
    #-----------
    # other
    plt.tight_layout()
    
    #-----------
    ### Savefig                   
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
        plt.savefig("%s/BH_sample_analysis/%sstelmass_SFR_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(BH_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_SFR_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(BH_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()   
# sSFR
def _BH_sample_sSFR(BH_tree, BH_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Sample options
                        min_window_size   = 1,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                        #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                        min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                        min_bhmass   = None,      max_bhmass   = None,
                        min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                        min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                        # Mergers, looked for within range considered +/- halfwindow
                        use_merger_criteria = False,
                      # Plot options

                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      

    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    state_plot      = []
    ID_plot     = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_tree['%s' %galaxy_state].keys():
            
            # Append means
            stelmass_plot.append(np.mean(np.array(BH_tree['%s' %galaxy_state]['%s' %ID_i]['stelmass'])))
            sfr_plot.append(np.mean(np.array(BH_tree['%s' %galaxy_state]['%s' %ID_i]['sfr'])))
            ssfr_plot.append(np.mean(np.array(BH_tree['%s' %galaxy_state]['%s' %ID_i]['ssfr'])))
            kappa_plot.append(np.mean(np.array(BH_tree['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])))
            state_plot.append(galaxy_state)
            ID_plot.append(ID_i)
            
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Morphology': kappa_plot, 'State': state_plot, 'GalaxyIDs': ID_plot})        
    print('\nSample after processing:\n   aligned: %s\n   misaligned: %s\n   counter: %s' %(len(df.loc[(df['State'] == 'aligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'misaligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'counter')]['GalaxyIDs'])))
    
    
    #df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'ETG → ETG')]
    
    
    #------------------------
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict = {'aligned':'grey', 'misaligned':'r', 'counter':'b'}
    plt.scatter(np.log10(df['stelmass']), df['sSFR'], s=0.1, c=[color_dict[i] for i in df['State']], edgecolor=[color_dict[i] for i in df['State']], marker='.', alpha=0.8)
    #plt.show()
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(9, 11.5)
    axs.set_ylim(3e-13, 3e-9)
    axs.set_yscale('log')
    axs.set_xlabel('Stellar mass 2r50')
    axs.set_ylabel('sSFR [/yr]')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('k')
    legend_labels.append('misaligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('r')
    legend_labels.append('counter')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('b')
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    
    #-----------
    # other
    plt.tight_layout()
    
    #-----------
    ### Savefig                   
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
        plt.savefig("%s/BH_sample_analysis/%sstelmass_sSFR_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(BH_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_sSFR_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(BH_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()




#We want a way to overlay the BH growths of misaligned galaxies, over that of non-misaligned and counter-rotators
## ANALYSE TREE:
#- import BH_tree
#- create sample that does not have any BH/subhalo issues
#- Should now have a sample of misaligned, aligned, and counter-rotating galaxies with all BH growth values
#	- Overlay all BH growths
#	- Trim to minimum trelax, so that even a 1 Gyr misalignment we only have the first 0.5 Gyr of it
#	- plot medians
#	- Randomly select X for control to plot
#	- Do we see a difference in value?
	
# may also want to define a sf_gas_kappa, and look at *when* BH growth is boosted





#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Set starting parameters
load_csv_file_in = '_test_run_'
plot_annotate_in                                           = False
savefig_txt_in   = load_csv_file_in               # [ 'manual' / load_csv_file_in ] 'manual' will prompt txt before saving

#  False 
# 'ETG → ETG' , 'LTG → LTG'
#  r'ETG ($\bar{\kappa}_{\mathrm{co}}^{\mathrm{*}} < 0.35$)'
# '$t_{\mathrm{relax}}>3\bar{t}_{\mathrm{torque}}$'    

#==================================================================================================================================
BH_tree, BH_input, summary_dict = _extract_sample(load_csv_file=load_csv_file_in, plot_annotate=plot_annotate_in)
#==================================================================================================================================


# SAMPLE SFR
"""_BH_sample_SFR(BH_tree=BH_tree, BH_input=BH_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                                min_bhmass   = None,      max_bhmass   = None,
                                min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                                min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                                use_merger_criteria = False,
                              showfig = True,
                              savefig = False)"""
# SAMPLE sSFR
"""_BH_sample_sSFR(BH_tree=BH_tree, BH_input=BH_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                                min_bhmass   = None,      max_bhmass   = None,
                                min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                                min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                                use_merger_criteria = False,
                              showfig = True,
                              savefig = False)"""
                            
                            
_BH_sample_SFR(BH_tree=BH_tree, BH_input=BH_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                min_stelmass = 10**10,      max_stelmass = 10**(10.5),        # [ 10**9 / Msun ]
                                min_bhmass   = None,      max_bhmass   = None,
                                min_sfr      = 0.1,      max_sfr      = 1,        # [ Msun/yr ] SF limit of ~ 0.1
                                min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                                use_merger_criteria = False,
                              showfig = True,
                              savefig = False)           
                            