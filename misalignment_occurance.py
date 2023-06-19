import h5py
import numpy as np
import math
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
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
from graphformat import set_rc_params


# Directories
EAGLE_dir       = '/Users/c22048063/Documents/EAGLE'
dataDir_main    = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
dataDir_alt     = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
# Directories serpens
#EAGLE_dir       = '/home/user/c22048063/Documents/EAGLE'
#dataDir_main   = '/home/universe/spxtd1-shared/RefL0100N1504/'
#dataDir_alt    = '/home/cosmos/c22048063/EAGLE_snapshots/RefL0100N1504/'


# Other directories
sample_dir      = EAGLE_dir + '/samples'
output_dir      = EAGLE_dir + '/outputs'
fig_dir         = EAGLE_dir + '/plots'

# Directories of data hdf5 file(s)
dataDir_dict = {}
dataDir_dict['10'] = dataDir_alt + 'snapshot_010_z003p984/snap_010_z003p984.0.hdf5'
dataDir_dict['11'] = dataDir_alt + 'snapshot_011_z003p528/snap_011_z003p528.0.hdf5'
dataDir_dict['12'] = dataDir_alt + 'snapshot_012_z003p017/snap_012_z003p017.0.hdf5'
dataDir_dict['13'] = dataDir_alt + 'snapshot_013_z002p478/snap_013_z002p478.0.hdf5'
dataDir_dict['14'] = dataDir_alt + 'snapshot_014_z002p237/snap_014_z002p237.0.hdf5'
dataDir_dict['15'] = dataDir_alt + 'snapshot_015_z002p012/snap_015_z002p012.0.hdf5'
dataDir_dict['16'] = dataDir_alt + 'snapshot_016_z001p737/snap_016_z001p737.0.hdf5'
dataDir_dict['17'] = dataDir_alt + 'snapshot_017_z001p487/snap_017_z001p487.0.hdf5'
dataDir_dict['18'] = dataDir_alt + 'snapshot_018_z001p259/snap_018_z001p259.0.hdf5'
dataDir_dict['19'] = dataDir_alt + 'snapshot_019_z001p004/snap_019_z001p004.0.hdf5'
dataDir_dict['20'] = dataDir_alt + 'snapshot_020_z000p865/snap_020_z000p865.0.hdf5'
dataDir_dict['21'] = dataDir_alt + 'snapshot_021_z000p736/snap_021_z000p736.0.hdf5'
dataDir_dict['22'] = dataDir_alt + 'snapshot_022_z000p615/snap_022_z000p615.0.hdf5'
dataDir_dict['23'] = dataDir_alt + 'snapshot_023_z000p503/snap_023_z000p503.0.hdf5'
dataDir_dict['24'] = dataDir_alt + 'snapshot_024_z000p366/snap_024_z000p366.0.hdf5'
dataDir_dict['25'] = dataDir_main + 'snapshot_025_z000p271/snap_025_z000p271.0.hdf5'
dataDir_dict['26'] = dataDir_main + 'snapshot_026_z000p183/snap_026_z000p183.0.hdf5'
dataDir_dict['27'] = dataDir_main + 'snapshot_027_z000p101/snap_027_z000p101.0.hdf5'
dataDir_dict['28'] = dataDir_main + 'snapshot_028_z000p000/snap_028_z000p000.0.hdf5'
#dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
#dataDir = '/home/universe/spxtd1-shared/RefL0100N1504/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)


# COPY OUTPUT
#scp -r c22048063@physxlogin.astro.cf.ac.uk:/home/user/c22048063/Documents/EAGLE/outputs /Users/c22048063/Documents/EAGLE/
# COPY SAMPLE
#scp -r c22048063@physxlogin.astro.cf.ac.uk:/home/user/c22048063/Documents/EAGLE/samples /Users/c22048063/Documents/EAGLE/
# COPY CODE
#scp -r /Users/c22048063/Documents/EAGLE/code  c22048063@physxlogin.astro.cf.ac.uk:/home/user/c22048063/Documents/EAGLE/

     
# Goes through existing CSV files to locate the snapshots at which misalignments are most frequent
def _find_misalignment_occurance(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                 csv_sample2 = '_all_sample_misalignment_9.0',
                                 csv_sample_range = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],   # snapnums
                                 csv_output_in = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                                 #--------------------------
                                 # Galaxy plotting
                                 print_summary = True,
                                   use_angle          = 'stars_gas_sf',         # Which angles to plot
                                   use_hmr            = 2.0,                    # Which HMR to use
                                   use_proj_angle     = True,                   # Whether to use projected or absolute angle 10**9
                                   lower_mass_limit   = 10**9,             # Whether to plot only certain masses 10**15
                                   upper_mass_limit   = 10**15,         
                                   ETG_or_LTG         = 'ETG',             # Whether to plot only ETG/LTG
                                   group_or_field     = 'both',            # Whether to plot only field/group
                                 #--------------------------
                                 showfig       = True,
                                 savefig       = True,
                                   file_format = 'pdf',
                                   savefig_txt = '',
                                 #--------------------------
                                 print_progress = False,
                                 debug = False):
                                 
    #================================================  
    # Load sample csv
    if print_progress:
        print('Cycling through CSV files')
        time_start = time.time()
    
    print('===================')
    print('PLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %.2f M*\n  Upper mass limit: %.2f M*\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
    print('===================\n')
    
    
    #----------------------
    # Creating dictionary to collect aligned and misaligned IDs
    alignment_dict = {}
    
    
    #================================================ 
    # Cycling over all the csv samples we want
    for csv_sample_range_i in tqdm(csv_sample_range):
        
        # Ensuring the sample and output originated together
        csv_sample = csv_sample1 + str(csv_sample_range_i) + csv_sample2
        csv_output = csv_sample + csv_output_in
        
        
        #================================================  
        # Load sample csv
        if print_progress:
            print('Loading initial sample')
            time_start = time.time()
    
        #--------------------------------
        # Loading sample
        dict_sample = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
        GroupNum_List       = np.array(dict_sample['GroupNum'])
        SubGroupNum_List    = np.array(dict_sample['SubGroupNum'])
        GalaxyID_List       = np.array(dict_sample['GalaxyID'])
        DescendantID_List   = np.array(dict_sample['DescendantID'])
        SnapNum_List        = np.array(dict_sample['SnapNum'])
        
        # Loading output
        dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
        all_general         = dict_output['all_general']
        all_spins           = dict_output['all_spins']
        all_coms            = dict_output['all_coms']
        all_counts          = dict_output['all_counts']
        all_masses          = dict_output['all_masses']
        all_misangles       = dict_output['all_misangles']
        all_misanglesproj   = dict_output['all_misanglesproj']
        all_flags           = dict_output['all_flags']
    
        # Loading sample criteria
        sample_input        = dict_sample['sample_input']
        output_input        = dict_output['output_input']
    
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print(sample_input)
            print(GroupNum_List)
            print(SubGroupNum_List)
            print(GalaxyID_List)
            print(DescendantID_List)
            print(SnapNum_List)
   
        if debug:
            print('\n===================')
            print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift'], output_input['galaxy_mass_limit'], use_satellites))
            print('  SAMPLE LENGTH: ', len(GroupNum_List))
            print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
            print('===================')
        print('===================')
        print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %.2f\n' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift']))
        
        #------------------------------
        # Check if requested plot is possible with loaded data
        assert use_angle in output_input['angle_selection'], 'Requested angle %s not in output_input' %use_angle
        assert use_hmr in output_input['spin_hmr'], 'Requested HMR %s not in output_input' %use_hmr
    
        # Create particle list of interested values (used later for flag), and plot labels:
        use_particles = []
        if use_angle == 'stars_gas':
            if 'stars' not in use_particles:
                use_particles.append('stars')
            if 'gas' not in use_particles:
                use_particles.append('gas')
            plot_label = 'Stars-gas'
        if use_angle == 'stars_gas_sf':
            if 'stars' not in use_particles:
                use_particles.append('stars')
            if 'gas_sf' not in use_particles:
                use_particles.append('gas_sf')
            plot_label = 'Stars-gas$_{\mathrm{sf}}$'
        if use_angle == 'stars_gas_nsf':
            if 'stars' not in use_particles:
                use_particles.append('stars')
            if 'gas_nsf' not in use_particles:
                use_particles.append('gas_nsf')
            plot_label = 'Stars-gas$_{\mathrm{nsf}}$'
        if use_angle == 'gas_sf_gas_nsf':
            if 'gas_sf' not in use_particles:
                use_particles.append('gas_sf')
            if 'gas_nsf' not in use_particles:
                use_particles.append('gas_nsf')
            plot_label = 'gas$_{\mathrm{sf}}$-gas$_{\mathrm{nsf}}$'
        if use_angle == 'stars_dm':
            if 'stars' not in use_particles:
                use_particles.append('stars')
            if 'dm' not in use_particles:
                use_particles.append('dm')
            plot_label = 'Stars-DM'
        if use_angle == 'gas_dm':
            if 'gas' not in use_particles:
                use_particles.append('gas')
            if 'dm' not in use_particles:
                use_particles.append('dm')
            plot_label = 'Gas-DM'
        if use_angle == 'gas_sf_dm':
            if 'gas_sf' not in use_particles:
                use_particles.append('gas_sf')
            if 'dm' not in use_particles:
                use_particles.append('dm')
            plot_label = 'Gas$_{\mathrm{sf}}$-DM'
        if use_angle == 'gas_nsf_dm':
            if 'gas_nsf' not in use_particles:
                use_particles.append('gas_nsf')
            if 'dm' not in use_particles:
                use_particles.append('dm')
            plot_label = 'Gas$_{\mathrm{nsf}}$-DM'
    
    
        #-----------------------------
        # Set definitions
        group_threshold     = 10**14
        LTG_threshold       = 0.4
    
        # Setting morphology lower and upper boundaries based on inputs
        if ETG_or_LTG == 'both':
            lower_morph = 0
            upper_morph = 1
        elif ETG_or_LTG == 'ETG':
            lower_morph = 0
            upper_morph = LTG_threshold
        elif ETG_or_LTG == 'LTG':
            lower_morph = LTG_threshold
            upper_morph = 1
        
        # Setting group lower and upper boundaries based on inputs
        if group_or_field == 'both':
            lower_halo = 0
            upper_halo = 10**16
        elif group_or_field == 'group':
            lower_halo = group_threshold
            upper_halo = 10**16
        elif group_or_field == 'field':
            lower_halo = 0
            upper_halo = group_threshold
    
        # Setting satellite criteria
        satellite_criteria = 99999999
        
        #----------------------------
        # Creation dictionary
        alignment_dict['%s' %output_input['snapNum']] = {'aligned': {'GalaxyID': [], 'DescendantID': []},
                                                         'misaligned': {'GalaxyID': [], 'DescendantID': []}}
                          
                          
        #=================================
        def _collect_misalignment_distributions_z(debug=False):
            
            # Looping over all GalaxyIDs
            for GalaxyID, DescendantID in zip(GalaxyID_List, DescendantID_List):
                
                #-----------------------------
                # Determine if galaxy has flags:
                if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                    
                    # Determine if galaxy meets criteria:
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                        
                        # Mask correct integer (formatting weird but works)
                        mask_rad = int(np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == use_hmr)[0])
                        
                        # Determine if it is aligned or misaligned at the radius of interest
                        if use_proj_angle:
                            if all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_rad] <= 30:
                                alignment_dict['%s' %output_input['snapNum']]['aligned']['GalaxyID'].append(GalaxyID)
                                alignment_dict['%s' %output_input['snapNum']]['aligned']['DescendantID'].append(DescendantID)
                            else:
                                alignment_dict['%s' %output_input['snapNum']]['misaligned']['GalaxyID'].append(GalaxyID)
                                alignment_dict['%s' %output_input['snapNum']]['misaligned']['DescendantID'].append(DescendantID)
                        else:
                            if all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_rad] <= 30:
                                alignment_dict['%s' %output_input['snapNum']]['aligned']['GalaxyID'].append(GalaxyID)
                                alignment_dict['%s' %output_input['snapNum']]['aligned']['DescendantID'].append(DescendantID)
                            else:
                                alignment_dict['%s' %output_input['snapNum']]['misaligned']['GalaxyID'].append(GalaxyID)
                                alignment_dict['%s' %output_input['snapNum']]['misaligned']['DescendantID'].append(DescendantID)
                            
        #--------------------------------------
        _collect_misalignment_distributions_z()
        #--------------------------------------
        
    
    #======================================================================
    # Analysis of collected misaligned and aligned galaxyIDs
    def _plot_misalignment_snap(debug=False):
        
        #--------------------------------
        # Create arrays to collect data
        plot_dict = {}
        for SnapNum in alignment_dict.keys():
        
            # Append known values
            plot_dict['%s' %SnapNum] = {'became_aligned': [],
                                        'became_misaligned': [],
                                        'stayed_aligned': [],
                                        'stayed_misaligned': []}
        
        
        #--------------------------------
        for SnapNum in alignment_dict.keys():
        
            # Ignore last snap (28)
            if int(SnapNum) == csv_sample_range[-1]:
                continue
        
            # We assume only main-line galaxies (DescendantID = GalaxyID - 1)
            # Check for galaxies that stayed aligned or became misaligned:
            for GalaxyID, DescendantID in zip(alignment_dict['%s' %SnapNum]['aligned']['GalaxyID'], alignment_dict['%s' %SnapNum]['aligned']['DescendantID']):
                if (DescendantID in alignment_dict['%s' %(int(SnapNum)+1)]['aligned']['GalaxyID']) and (DescendantID == GalaxyID-1):
                    plot_dict['%s' %(int(SnapNum)+1)]['stayed_aligned'].append(GalaxyID)
            
                elif (DescendantID in alignment_dict['%s' %(int(SnapNum)+1)]['misaligned']['GalaxyID']) and (DescendantID == GalaxyID-1):
                    plot_dict['%s' %(int(SnapNum)+1)]['became_misaligned'].append(GalaxyID)
        
            # Check for galaxies that stayed misaligned or became aligned:      
            for GalaxyID, DescendantID in zip(alignment_dict['%s' %SnapNum]['misaligned']['GalaxyID'], alignment_dict['%s' %SnapNum]['misaligned']['DescendantID']):
                if (DescendantID in alignment_dict['%s' %(int(SnapNum)+1)]['aligned']['GalaxyID']) and (DescendantID == GalaxyID-1):
                    plot_dict['%s' %(int(SnapNum)+1)]['became_aligned'].append(GalaxyID)
            
                elif (DescendantID in alignment_dict['%s' %(int(SnapNum)+1)]['misaligned']['GalaxyID']) and (DescendantID == GalaxyID-1):
                    plot_dict['%s' %(int(SnapNum)+1)]['stayed_misaligned'].append(GalaxyID)
        
        
        #--------------------------------
        # analyse dictionary and extract plot values:
        plot_snap = []
        plot_stayed_aligned     = []
        plot_stayed_misaligned  = []
        plot_became_aligned     = []
        plot_became_misaligned  = []
        
        for SnapNum in plot_dict.keys():
            plot_snap.append(int(SnapNum))
            plot_stayed_aligned.append(len(plot_dict['%s' %SnapNum]['stayed_aligned']))
            plot_stayed_misaligned.append(len(plot_dict['%s' %SnapNum]['stayed_misaligned']))
            plot_became_aligned.append(len(plot_dict['%s' %SnapNum]['became_aligned']))
            plot_became_misaligned.append(len(plot_dict['%s' %SnapNum]['became_misaligned']))
                
            
        
        #--------------------------------
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[6, 5.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        
        #----------------------
        # Plotting graphs
        axs.plot(plot_snap[1:], plot_stayed_aligned[1:], label='Stayed aligned', c='b', ls='-')
        axs.plot(plot_snap[1:], plot_stayed_misaligned[1:], label='Stayed misaligned', c='r', ls='-')
        axs.plot(plot_snap[1:], plot_became_aligned[1:], label='Became aligned', c='cyan', ls='-')
        axs.plot(plot_snap[1:], plot_became_misaligned[1:], label='Became misaligned', c='orange', ls='-')
        
        
        #----------------------
        ### General formatting
        # Setting regular axis
        axs.set_xlim(18.5, 28.5)
        axs.set_xlabel('SnapNum')
        axs.set_ylim(0, 8000)
        axs.set_ylabel('Number of galaxies')
        
        #-----------
        ### Legend
        axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
        
        #-----------
        # other
        plt.tight_layout()
        
        if print_summary:
            print('===================')
            print('SUMMARY DATA:')
            print('  | Snap | Stayed aligned | Stayed misaligned | New aligned | New misaligned |')
            for snap_p, s_ali, s_mis, b_ali, b_mis in zip(plot_snap, plot_stayed_aligned, plot_stayed_misaligned, plot_became_aligned, plot_became_misaligned):
                print('  | %s |   %s   |   %s   |   %s   |   %s   |' %(snap_p, s_ali, s_mis, b_ali, b_mis))
        
        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        metadata_rows = '| Snap | Stayed ali | Stayed mis | New ali | New mis |\n'
        for snap_p, s_ali, s_mis, b_ali, b_mis in zip(plot_snap, plot_stayed_aligned, plot_stayed_misaligned, plot_became_aligned, plot_became_misaligned):
            metadata_rows += ('| %s | %s | %s | %s | %s |\n' %(snap_p, s_ali, s_mis, b_ali, b_mis))
        metadata_plot = {'Title': metadata_rows}
        
        if savefig:
            plt.savefig("%s/L%s_ALL_becameMisligned_%s_%s_HMR%s_proj%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], np.log10(float(output_input['galaxy_mass_limit'])), use_angle, str(use_hmr), use_proj_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, group_or_field, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/L%s_ALL_becameMisaligned_%s_%s_HMR%s_proj%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], np.log10(float(output_input['galaxy_mass_limit'])), use_angle, str(use_hmr), use_proj_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, group_or_field, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
        
    #------------------------
    _plot_misalignment_snap()
    #------------------------
    
#=============================
_find_misalignment_occurance()
#=============================
























