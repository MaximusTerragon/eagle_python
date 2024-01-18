import h5py
import numpy as np
import math
from scipy import stats
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import astropy.units as u
from astropy.cosmology import z_at_value, FlatLambdaCDM
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
import eagleSqlTools as sql
from graphformat import set_rc_params
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================


#--------------------------------
# Plots singular graphs by reading in existing csv file
# SAVED: /plots/misalignment_distributions/
def _plot_misalignment(csv_sample = 'L100_27_all_sample_misalignment_9.5',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                       csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                       #--------------------------
                       # Galaxy plotting
                       print_summary = True,
                         use_angle          = 'stars_gas_sf',         # Which angles to plot
                         use_hmr            = 1.0,                    # Which HMR to use
                         use_proj_angle     = True,                   # Whether to use projected or absolute angle 10**9
                           min_inc_angle    = 0,                     # min. degrees of either spin vector to z-axis, if use_proj_angle
                           min_particles    = 20,               # [ 20 ] number of particles
                           min_com          = 2.0,              # [ 2.0 ] pkpc
                           max_uncertainty  = 30,            # [ None / 30 / 45 ]                  Degrees
                         lower_mass_limit   = 10**9.5,            # Whether to plot only certain masses 10**15
                         upper_mass_limit   = 10**15,         
                         ETG_or_LTG         = 'both',           # Whether to plot only ETG/LTG/both
                         cluster_or_field   = 'both',           # Whether to plot only field/cluster/both
                         use_satellites     = True,             # Whether to include SubGroupNum =/ 0
                       #--------------------------
                       add_observational  = 'Bryant',       # [ False / Bryant / Raimundo ]
                       misangle_threshold = 30,             # what we classify as misaligned
                       #--------------------------
                       # Specify an angle range from which to compare to uniform random distribution
                       # NOT WORKING ATM
                       KS_test_from_angle = None,            # [ deg / 60] angle above which to start KS test for flat distribution
                       KS_test_to_angle   = None,            # [ deg / 180] angle to which to start KS test for flat distribution
                       #--------------------------
                       use_alternative_format = True,          # COMPACT/Poster formatting
                       #--------------------------
                       showfig       = True,
                       savefig       = True,
                         file_format = 'pdf',
                         savefig_txt = '',
                       #--------------------------
                       print_progress = False,
                       debug = False):
                        
                        
                        
    # Ensuring the sample and output originated together
    csv_output = csv_sample + csv_output 
    csv_output
    
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
    SnapNum_List        = np.array(dict_sample['SnapNum'])
        
    # Loading output
    dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    all_general         = dict_output['all_general']
    all_spins           = dict_output['all_spins']
    all_coms            = dict_output['all_coms']
    all_counts          = dict_output['all_counts']
    all_masses          = dict_output['all_masses']
    all_sfr             = dict_output['all_sfr']
    all_Z               = dict_output['all_Z']
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
        print(SnapNum_List)
   
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift'], output_input['galaxy_mass_min'], output_input['galaxy_mass_max'], use_satellites))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Min. inclination: %s\n  Min particles: %s\n  Min COM: %.1f pkpc\n  Min Mass: %.2E M*\n  Max limit: %.2E M*\n  ETG or LTG: %s\n  Cluster or field: %s\n  Use satellites:  %s' %(use_angle, use_hmr, use_proj_angle, min_inc_angle, min_particles, min_com, lower_mass_limit, upper_mass_limit, ETG_or_LTG, cluster_or_field, use_satellites))
    print('===================')
    
    #------------------------------
    # Check if requested plot is possible with loaded data
    assert use_angle in output_input['angle_selection'], 'Requested angle %s not in output_input' %use_angle
    assert use_hmr in output_input['spin_hmr'], 'Requested HMR %s not in output_input' %use_hmr
    if use_satellites:
        assert use_satellites == sample_input['use_satellites'], 'Sample does not contain satellites'

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
    
    # Set projection angle criteria
    if not use_proj_angle:
        min_inc_angle = 0
    max_inc_angle = 180 - min_inc_angle
    if output_input['viewing_axis'] == 'x':
        viewing_vector = [1., 0, 0]
    elif output_input['viewing_axis'] == 'y':
        viewing_vector = [0, 1., 0]
    elif output_input['viewing_axis'] == 'z':
        viewing_vector = [0, 0, 1.]
    else:
        raise Exception('Cant read viewing_axis')
        
    #-----------------------------
    # Set definitions
    cluster_threshold     = 1e14
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
        
    # Setting cluster lower and upper boundaries based on inputs
    if cluster_or_field == 'both':
        lower_halo = 0
        upper_halo = 10**16
    elif cluster_or_field == 'cluster':
        lower_halo = cluster_threshold
        upper_halo = 10**16
    elif cluster_or_field == 'field':
        lower_halo = 0
        upper_halo = cluster_threshold
    
    # Setting satellite criteria
    if use_satellites:
        satellite_criteria = 99999999
    if not use_satellites:
        satellite_criteria = 0
    
    collect_ID = []
    #------------------------------
    def _plot_misalignment_distributions(debug=False):
        # We have use_angle = 'stars_gas_sf', and use_particles = ['stars', 'gas_sf'] 
        
        #=================================
        # Collect values to plot
        plot_angles     = []
        plot_angles_err = []
        
        # Collect other useful values
        catalogue = {'total': {},          # Total galaxies in mass range
                     'sample': {},         # Sample of galaxies that meet particle count, COM, inclination angle, regardless of morphology, environment, or satellite status
                     'plot': {}}           # Sub-sample that is plotted (environment/morphology/satellite)
        for key in catalogue.keys():
            catalogue[key] = {'all': 0,        # Size of cluster
                              'cluster': 0,      # number of cluster galaxies
                              'field': 0,      # number of field galaxies
                              'ETG': 0,        # number of ETGs
                              'LTG': 0}        # number of LTGs
        if print_progress:
            print('Analysing extracted sample and collecting angles')
            time_start = time.time()
        
        # Find angle galaxy makes with viewing axis
        def _find_angle(vector1, vector2):
            return np.rad2deg(np.arccos(np.clip(np.dot(vector1/np.linalg.norm(vector1), vector2/np.linalg.norm(vector2)), -1.0, 1.0)))     # [deg]
        
        # Find distance between coms
        def _evaluate_com(com1, com2, abs_proj, debug=False):
            if abs_proj == 'abs':
                d = np.linalg.norm(np.array(com1) - np.array(com2))
            elif abs_proj == 'x':
                d = np.linalg.norm(np.array([com1[1], com1[2]]) - np.array([com2[1], com2[2]]))
            elif abs_proj == 'y':
                d = np.linalg.norm(np.array([com1[0], com1[2]]) - np.array([com2[0], com2[2]]))
            elif abs_proj == 'z':
                d = np.linalg.norm(np.array([com1[0], com1[1]]) - np.array([com2[0], com2[1]]))
            else:
                raise Exception('unknown entery')
            return d
        
        # Add all galaxies loaded to catalogue
        catalogue['total']['all'] = len(GroupNum_List)        # Total galaxies initially
        
        
        #--------------------------
        # Loop over all galaxies we have available, and analyse output of flags
        for GalaxyID in GalaxyID_List:
            
            #-----------------------------
            # Determine if cluster or field, and morphology
            if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                catalogue['total']['cluster'] += 1
            elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                catalogue['total']['field'] += 1
            if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                catalogue['total']['LTG'] += 1
            elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                catalogue['total']['ETG'] += 1
            
            
            #-----------------------------
            # Check if galaxy meets criteria
            
            # check if hmr exists 
            if (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                # creating masks
                mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                
                # find particle counts
                if use_particles[0] == 'dm':
                    count_1 = all_counts['%s' %GalaxyID][use_particles[0]]
                else:
                    count_1 = all_counts['%s' %GalaxyID][use_particles[0]][mask_counts]
                if use_particles[1] == 'dm':
                    count_2 = all_counts['%s' %GalaxyID][use_particles[1]]
                else:
                    count_2 = all_counts['%s' %GalaxyID][use_particles[1]][mask_counts]
                
                # find inclination angle(s)
                inc_angle_1 = _find_angle(all_spins['%s' %GalaxyID][use_particles[0]][mask_spins], viewing_vector)
                inc_angle_2 = _find_angle(all_spins['%s' %GalaxyID][use_particles[1]][mask_spins], viewing_vector)
                
                # find CoMs = com_abs
                if use_angle != 'stars_dm':
                    com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]][mask_angles], 'abs')
                else:
                    com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]], 'abs')
                
                #--------------
                # Determine if this is a galaxy we want to plot and meets the remaining criteria (stellar mass, halo mass, kappa, uncertainty, satellite)
                if use_proj_angle:
                    max_error = max(np.abs((np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle][mask_angles]) - all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])))
                else:
                    max_error = max(np.abs((np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle][mask_angles]) - all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles])))
                
                
                # applying selection criteria for min_inc_angle, min_com, min_particles
                if (count_1 >= min_particles) and (count_2 >= min_particles) and (com_abs <= min_com) and (inc_angle_1 >= min_inc_angle) and (inc_angle_1 <= max_inc_angle) and (inc_angle_2 >= min_inc_angle) and (inc_angle_2 <= max_inc_angle) and (max_error <= (999 if max_uncertainty == None else max_uncertainty)):
                    
                    #--------------
                    catalogue['sample']['all'] += 1
                    
                    # Determine if cluster or field, and morphology
                    if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                        catalogue['sample']['cluster'] += 1
                    elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                        catalogue['sample']['field'] += 1
                    if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                        catalogue['sample']['LTG'] += 1
                    elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                        catalogue['sample']['ETG'] += 1
                    
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria):
                        
                        #--------------
                        catalogue['plot']['all'] += 1
                        
                        # Determine if cluster or field, and morphology
                        if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                            catalogue['plot']['cluster'] += 1
                        elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                            catalogue['plot']['field'] += 1
                        if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                            catalogue['plot']['LTG'] += 1
                        elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                            catalogue['plot']['ETG'] += 1
                        #--------------
                        
                        if all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles] > 170:
                            collect_ID.append(GalaxyID)
                        
                        
                        # Collect misangle or misangleproj
                        if use_proj_angle:
                            plot_angles.append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])
                        else:
                            plot_angles.append(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles])
            
                        
        assert catalogue['plot']['all'] == len(plot_angles), 'Number of angles collected does not equal number in catalogue... for some reason'
        if debug:
            print(catalogue['total'])
            print(catalogue['sample'])
            print(catalogue['plot'])
        
        
        #print('\tcollect_ID of counter-rotatos:')
        #print(collect_ID)
        #print(len(collect_ID))
        
        
        #================================
        # Plotting
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Plotting')
            time_start = time.time()
        
        # Graph initialising and base formatting
        if use_alternative_format:
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        else:
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        
        # Setting colors
        if 'gas' in use_particles:
            plot_color = 'green'
        if 'gas_sf' in use_particles:
            plot_color = 'b'
        if 'gas_nsf' in use_particles:
            plot_color = 'indigo'
        if 'dm' in use_particles:
            plot_color = 'r'
        
        if ETG_or_LTG == 'ETG':
            plot_color = 'darkorange'
        elif ETG_or_LTG == 'LTG':
            plot_color = 'dodgerblue'
        elif ETG_or_LTG == 'both':
            plot_color = 'indigo'
        
        
        """ Some useful quantities:
        catalogue['sample']['all']  = number of galaxies that meet criteria
        catalogue['plot']['all']    = number of galaxies in this particular plot
        """
        
        #-----------
        ### Creating graphs
        
        # Add raimundo observational data
        if add_observational:
                
            # Define percentages
            obs_percent = {'Raimundo': {},
                           'Bryant': {}}
            
            # Add raimundo results:
            obs_percent['Raimundo'] = {'ETG_both':   [0.480, 0.180, 0.048, 0.016, 0.020, 0.016, 0.028, 0.016, 0.016, 0.028, 0.008, 0.005, 0.010, 0.004, 0.012, 0.018, 0.020, 0.065],
                                       'LTG_both':   [0.660, 0.192, 0.072, 0.016, 0.010, 0.006, 0.010, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.001, 0.003, 0.001, 0.003, 0.007]}
            
            # Add Bryant results:
            obs_percent['Bryant'] = {'both_both':    [0.593, 0.214, 0.082, 0.032, 0.017, 0.005, 0.003, 0.003, 0.003, 0.002, 0.008, 0.005, 0.000, 0.000, 0.002, 0.003, 0.010, 0.021],
                                     'both_field':   [0.593, 0.214, 0.082, 0.027, 0.019, 0.004, 0.004, 0.004, 0.004, 0.002, 0.008, 0.004, 0.000, 0.000, 0.002, 0.002, 0.010, 0.023],
                                     'both_cluster': [0.596, 0.213, 0.081, 0.044, 0.015, 0.007, 0.000, 0.000, 0.000, 0.000, 0.007, 0.007, 0.000, 0.000, 0.000, 0.007, 0.007, 0.015],
                                     'ETG_both':     [0.383, 0.159, 0.131, 0.065, 0.037, 0.019, 0.010, 0.019, 0.010, 0.010, 0.047, 0.028, 0.000, 0.000, 0.000, 0.010, 0.019, 0.056],
                                     'ETG_field':    [0.339, 0.145, 0.065, 0.065, 0.065, 0.000, 0.016, 0.032, 0.016, 0.016, 0.081, 0.048, 0.000, 0.000, 0.000, 0.016, 0.032, 0.081],
                                     'ETG_cluster':  [0.444, 0.178, 0.222, 0.067, 0.000, 0.022, 0.000, 0.000, 0.000, 0.000, 0.022, 0.022, 0.000, 0.000, 0.000, 0.000, 0.000, 0.022],
                                     'LTG_both':     [0.647, 0.223, 0.075, 0.021, 0.014, 0.000, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.002, 0.007, 0.002],
                                     'LTG_field':    [0.643, 0.216, 0.089, 0.019, 0.014, 0.000, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.000, 0.008, 0.003],
                                     'LTG_cluster':  [0.677, 0.262, 0.000, 0.031, 0.015, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.015, 0.000, 0.000]}
                                     
            obs_number = {'Bryant': {'both_both': 36+26+45+69+133+72+221,
                                     'both_field': 17+16+29+48+102+57+202,
                                     'both_cluster': 19+10+16+21+31+15+19,
                                     'ETG_both': 36+26+45,
                                     'ETG_field': 17+16+29,
                                     'ETG_cluster': 19+10+16,
                                     'LTG_both': 133+72+221,
                                     'LTG_field': 102+57+202,
                                     'LTG_cluster': 31+15+19}}
                                     
                                     
            obs_data = {'Raimundo': {'ETG_both': [],
                                     'LTG_both': []},
                        'Bryant':   {'both_both': [],
                                     'both_field': [],
                                     'both_cluster': [],
                                     'ETG_both': [],
                                     'ETG_field': [],
                                     'ETG_cluster': [],
                                     'LTG_both': [],
                                     'LTG_field': [],
                                     'LTG_cluster': []}}
            
            for author_i in obs_percent.keys():
                for obs_type_i in obs_percent['%s' %author_i].keys():
                
                    obs_count = 1000*np.array(obs_percent['%s' %author_i][obs_type_i])
                
                    # Append number of galaxies for each angle histogram
                    for obs_count_i, obs_angle_i in zip(obs_count, [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175]):
                        i = 0
                        while i < obs_count_i:
                            obs_data['%s' %author_i][obs_type_i].append(obs_angle_i)
                            i += 1
                
                    
            # Plot
            axs.hist(obs_data['%s' %add_observational]['%s_%s' %(ETG_or_LTG, cluster_or_field)], weights=np.ones(len(obs_data['%s' %add_observational]['%s_%s' %(ETG_or_LTG, cluster_or_field)]))/len(obs_data['%s' %add_observational]['%s_%s' %(ETG_or_LTG, cluster_or_field)]), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='grey', facecolor=None, fill=False, linestyle='-', linewidth=1, alpha=1)
            axs.errorbar(np.arange(5, 181, 10), obs_percent['%s' %author_i]['%s_%s' %(ETG_or_LTG, cluster_or_field)], xerr=None, yerr=np.sqrt(np.array(obs_percent['%s' %author_i]['%s_%s' %(ETG_or_LTG, cluster_or_field)]) * obs_number['%s' %author_i]['%s_%s' %(ETG_or_LTG, cluster_or_field)]) / obs_number['%s' %author_i]['%s_%s' %(ETG_or_LTG, cluster_or_field)], ecolor='grey', ls='none', capsize=3, elinewidth=1, markeredgewidth=1, alpha=0.9)
        
        
        # Plot histogram
        axs.hist(plot_angles, weights=np.ones(catalogue['plot']['all'])/catalogue['plot']['all'], bins=np.arange(0, 191, 10), histtype='bar', edgecolor='none', facecolor=plot_color, linewidth=1.0, alpha=0.1)
        bin_count, _, _ = axs.hist(plot_angles, weights=np.ones(catalogue['plot']['all'])/catalogue['plot']['all'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color, facecolor='none', linewidth=1.0, alpha=0.8)
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(plot_angles, bins=np.arange(0, 181, 10), range=(0, 180))
        axs.errorbar(np.arange(5, 181, 10), hist_n/catalogue['plot']['all'], xerr=None, yerr=np.sqrt(hist_n)/catalogue['plot']['all'], ecolor=plot_color, ls='none', capsize=3, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        
        #-----------
        ### General formatting
        # Axis labels
        axs.yaxis.set_major_formatter(PercentFormatter(1, decimals=1, symbol=''))
        axs.set_xlim(0, 180)
        axs.set_ylim(bottom=0)
        axs.set_xticks(np.arange(0, 181, step=30))
        if use_proj_angle:
            axs.set_xlabel('Misalignment angle, $\psi_{z}$')
        else:
            axs.set_xlabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        axs.set_ylabel('Percentage of galaxies')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
        #-----------
        # Annotations
        axs.axvline(misangle_threshold, ls='--', lw=0.5, c='k')
        
        
        #-----------
        ### Legend 1
        # NEW LEGEND
        if use_alternative_format:
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            if (cluster_or_field != 'both') & (ETG_or_LTG != 'both'):
                legend_labels = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field) + ' ' + ETG_or_LTG]
            elif (cluster_or_field == 'both') & (ETG_or_LTG != 'both'):
                legend_labels = [ETG_or_LTG]
            elif (cluster_or_field != 'both') & (ETG_or_LTG == 'both'):
                legend_labels = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field)]
            else:
                legend_labels = ['EAGLE sample']
            legend_colors = [plot_color]
        
            if add_observational:
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_labels.append('%s+%s' %(add_observational, '19' if add_observational == 'Bryant' else '23'))
                legend_colors.append('grey')
        
            # Add mass range
            if (lower_mass_limit != 10**9.5) and (upper_mass_limit != 10**15):
                legend_labels.append('$10 ^{%.1f} - 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit), np.log10(upper_mass_limit)))    
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
            elif (lower_mass_limit != 10**9.5):
                legend_labels.append('$> 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit)))    
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
            elif (upper_mass_limit != 10**15):
                legend_labels.append('$< 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(upper_mass_limit)))    
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
                        
            # Add redshift
            legend_labels.append('${z\sim%.1f}$' %sample_input['Redshift'])
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('k')
        
            legend1 = axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            axs.add_artist(legend1)
            #legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            #axs.add_artist(legend2)
        
        # OLD LEGEND
        if not use_alternative_format:
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_labels = [plot_label]
            legend_colors = [plot_color]
        
            # Add mass range
            if (lower_mass_limit != 10**9) and (upper_mass_limit != 10**15):
                legend_labels.append('$10 ^{%.1f} - 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit), np.log10(upper_mass_limit)))    
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
            elif (lower_mass_limit != 10**9):
                legend_labels.append('$> 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit)))    
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
            elif (upper_mass_limit != 10**15):
                legend_labels.append('$< 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(upper_mass_limit)))    
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
        
            # Add LTG/ETG if specified
            if ETG_or_LTG != 'both':
                legend_labels.append('%s' %ETG_or_LTG)
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
        
            if cluster_or_field != 'both':
                legend_labels.append('%s-galaxies' %cluster_or_field)
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('grey')
        
            # Add redshift
            legend_labels.append('${z=%.2f}$' %sample_input['Redshift'])
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('k')
        
            axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            
            
        #-----------
        # other
        plt.tight_layout()
        
        
        #=====================================
        ### Print summary
        
        aligned_tally           = 0
        aligned_err_tally       = 0
        misaligned_tally        = 0 
        misaligned_err_tally    = 0 
        counter_tally           = 0
        counter_err_tally       = 0
        
        # Bin values
        if (int(misangle_threshold) == 30) or (int(misangle_threshold) == 40):
            bins = np.arange(0, 181, 10)
        else:
            bins = np.arange(0, 181, 5)
        hist_n, hist_bins = np.histogram(plot_angles, bins=bins, range=(0, 180))
        
        if print_summary:
            print('CATALOGUE:')
            print('Total: ',  catalogue['total'])
            print('Sample:', catalogue['sample'])
            print('Plot:  ',  catalogue['plot'])
            print('\nRAW BIN VALUES:')      
        for angle_i, bin_count_i in zip(bins, hist_n):
            if print_summary:
                print('  %i' %bin_count_i, end='')
            if angle_i < misangle_threshold:
                aligned_tally += bin_count_i
                aligned_err_tally += bin_count_i**0.5
            if angle_i >= misangle_threshold:
                misaligned_tally += bin_count_i
                misaligned_err_tally += bin_count_i**0.5
            if angle_i >= (180-misangle_threshold):
                counter_tally += bin_count_i
                counter_err_tally += bin_count_i**0.5        
        
        # K-S test
        # test plot - where we need linspace(60, 181, number of misalignments above 60 degrees), and weights are weights=np.ones(number again)/total plotted
        #mask_KS = np.where(np.logical_and(np.array(hist_bins) >= KS_test_from_angle, np.array(hist_bins) < KS_test_to_angle))[0]
        #sum_above_angle = np.sum(np.array(hist_n)[mask_KS])
        #flat_distribution_angles = np.linspace(KS_test_from_angle, KS_test_to_angle+1, sum_above_angle)
        #flat_distribution_angles = np.random.uniform(low=KS_test_from_angle, high=KS_test_to_angle, size=sum_above_angle)
        #axs.hist(flat_distribution_angles, weights=np.ones(sum_above_angle)/np.sum(np.array(hist_n)), bins=np.arange(0, 181, 10), histtype='step', edgecolor='r', facecolor='none', linewidth=1.2, alpha=1, zorder=99)
        if print_summary:
            KS_test_from_angle  = 60
            KS_test_to_angle    = 180
            print('\n\nK-S TEST FOR OUR SAMPLE IN [~uniform] RANGE ([%s - %s] deg compared to uniform)' %(KS_test_from_angle, KS_test_to_angle))
            res1 = stats.kstest([i for i in plot_angles if i >= KS_test_from_angle and i <= KS_test_to_angle], 'uniform', args=(KS_test_from_angle, KS_test_to_angle - KS_test_from_angle))
            print('   D:       %.2f' %res1.statistic)
            print('   p-value: %s' %res1.pvalue)
            
            KS_test_from_angle  = 60
            KS_test_to_angle    = 150
            print('K-S TEST FOR OUR SAMPLE IN [~uniform sans counter] RANGE [%s - %s]' %(KS_test_from_angle, KS_test_to_angle))
            res2 = stats.kstest([i for i in plot_angles if i >= KS_test_from_angle and i <= KS_test_to_angle], 'uniform', args=(KS_test_from_angle, KS_test_to_angle - KS_test_from_angle))
            print('   D:       %.2f' %res2.statistic)
            print('   p-value: %s' %res2.pvalue)
            
            KS_test_from_angle  = 60
            KS_test_to_angle    = 120
            print('K-S TEST FOR OUR SAMPLE IN [~polar] RANGE [%s - %s]' %(KS_test_from_angle, KS_test_to_angle))
            res3 = stats.kstest([i for i in plot_angles if i >= KS_test_from_angle and i <= KS_test_to_angle], 'uniform', args=(KS_test_from_angle, KS_test_to_angle - KS_test_from_angle))
            print('   D:       %.2f' %res3.statistic)
            print('   p-value: %s' %res3.pvalue)
            
            KS_test_from_angle  = 135
            KS_test_to_angle    = 180
            print('K-S TEST FOR OUR SAMPLE IN [~counter] RANGE [%s - %s]' %(KS_test_from_angle, KS_test_to_angle))
            res4 = stats.kstest([i for i in plot_angles if i >= KS_test_from_angle and i <= KS_test_to_angle], 'uniform', args=(KS_test_from_angle, KS_test_to_angle - KS_test_from_angle))
            print('   D:       %.2f' %res4.statistic)
            print('   p-value: %s' %res4.pvalue)
            
            
            
        if print_summary:    
            print('\n')     # total population includes galaxies that failed sample, so can add to less than 100% (ei. remaining % is galaxies that make up non-sample)
            print('OF TOTAL POPULATION: \t(all galaxies in mass range)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['total']['all'], aligned_err_tally*100/catalogue['total']['all'], misaligned_tally*100/catalogue['total']['all'], misaligned_err_tally*100/catalogue['total']['all'], counter_tally*100/catalogue['total']['all'], counter_err_tally*100/catalogue['total']['all']))
            print('OF TOTAL SAMPLE: \t(meeting CoM, inc angle, particle counts, error):\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['sample']['all'], aligned_err_tally*100/catalogue['sample']['all'], misaligned_tally*100/catalogue['sample']['all'], misaligned_err_tally*100/catalogue['sample']['all'], counter_tally*100/catalogue['sample']['all'], counter_err_tally*100/catalogue['sample']['all']))
            print('OF PLOT SAMPLE: \t(specific plot criteria - morph, environ, satellite)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['plot']['all'], aligned_err_tally*100/catalogue['plot']['all'], misaligned_tally*100/catalogue['plot']['all'], misaligned_err_tally*100/catalogue['plot']['all'], counter_tally*100/catalogue['plot']['all'], counter_err_tally*100/catalogue['plot']['all']))       
        
        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        metadata_plot = {'Title': 'TOTAL SAMPLE:\nAligned: %.1f ± %.1f %%\nMisaligned: %.1f ± %.1f %%\nCounter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['sample']['all'], aligned_err_tally*100/catalogue['sample']['all'], misaligned_tally*100/catalogue['sample']['all'], misaligned_err_tally*100/catalogue['sample']['all'], counter_tally*100/catalogue['sample']['all'], counter_err_tally*100/catalogue['sample']['all']),
                         'Author': 'PLOT SAMPLE:\nAligned: %.1f ± %.1f %%\nMisaligned: %.1f ± %.1f %%\nCounter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['plot']['all'], aligned_err_tally*100/catalogue['plot']['all'], misaligned_tally*100/catalogue['plot']['all'], misaligned_err_tally*100/catalogue['plot']['all'], counter_tally*100/catalogue['plot']['all'], counter_err_tally*100/catalogue['plot']['all']),
                         'Subject': str(hist_n),
                         'Producer': str(catalogue)}
       
        if use_satellites:
            sat_str = 'all'
        if not use_satellites:
            sat_str = 'cent'
       
        if add_observational:
            obs_txt = 'OBSERVATIONAL'
        else:
            obs_txt = ''
       
        if savefig:
            plt.savefig("%s/misalignment_distributions/L%s_%s_%s_misalignment_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misalignment_distributions/L%s_%s_%s_misalignment_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
            
        # Check what is plotted
        """fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        masses = []
        hmr_rad = []    
        colors = []
        count = 0
        ID_failure = []
        for GalaxyID in GalaxyID_List:
            masses.append(all_general['%s' %GalaxyID]['stelmass'])
            hmr_rad.append(all_general['%s' %GalaxyID]['halfmass_rad_proj'])
            
            if use_hmr in all_misangles['%s' %GalaxyID]['hmr']:
                count += 1
                colors.append('b')
            else:
                colors.append('r')
                ID_failure.append(GalaxyID)
        
        print(len(masses))
        print(count)
        print(all_general['8230966']['stelmass'])
        print(all_general['8230966']['halfmass_rad_proj'])
        axs.scatter(np.log10(np.array(masses)), hmr_rad, s=0.5, alpha=0.5, c=colors)
        axs.set_xlim(9.4, 9.6)
        axs.set_ylim(0, 20)
        axs.set_xlabel('Mass', fontsize=12)
        axs.set_ylabel('HMR', fontsize=12)
        set_rc_params(0.9) 
        plt.show()
        """
        
    #---------------------------------
    _plot_misalignment_distributions()
    #---------------------------------
    

#--------------------------------
# Plots double graphs of ETG/LTG by reading in existing csv file
# SAVED: /plots/misalignment_distributions/ ... double...
def _plot_misalignment_double(csv_sample = 'L100_27_all_sample_misalignment_9.5',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                              csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                              #--------------------------
                              # Galaxy plotting
                              print_summary = True,
                                use_angle          = 'stars_gas_sf',         # Which angles to plot
                                use_hmr            = 1.0,                    # Which HMR to use
                                use_proj_angle     = True,                   # Whether to use projected or absolute angle 10**9
                                  min_inc_angle    = 0,                     # min. degrees of either spin vector to z-axis, if use_proj_angle
                                  min_particles    = 20,               # [ 20 ] number of particles
                                  min_com          = 2.0,              # [ 2.0 ] pkpc
                                  max_uncertainty  = 30,            # [ None / 30 / 45 ]                  Degrees
                                lower_mass_limit   = 10**9.5,            # Whether to plot only certain masses 10**15
                                upper_mass_limit   = 10**15,         
                                ETG_or_LTG         = 'both',           # Whether to plot only ETG/LTG/both
                                cluster_or_field   = 'both',           # Whether to plot only field/cluster/both
                                use_satellites     = True,             # Whether to include SubGroupNum =/ 0
                              #--------------------------
                              add_observational  = 'Bryant',       # [ False / Bryant / Raimundo ]
                              misangle_threshold = 30,             # what we classify as misaligned
                              #--------------------------
                              use_alternative_format = True,          # COMPACT/Poster formatting
                              #--------------------------
                              showfig       = True,
                              savefig       = True,
                                file_format = 'pdf',
                                savefig_txt = '',
                              #--------------------------
                              print_progress = False,
                              debug = False):
                        
                        
                        
    # Ensuring the sample and output originated together
    csv_output = csv_sample + csv_output 
    csv_output
    
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
    SnapNum_List        = np.array(dict_sample['SnapNum'])
        
    # Loading output
    dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    all_general         = dict_output['all_general']
    all_spins           = dict_output['all_spins']
    all_coms            = dict_output['all_coms']
    all_counts          = dict_output['all_counts']
    all_masses          = dict_output['all_masses']
    all_sfr             = dict_output['all_sfr']
    all_Z               = dict_output['all_Z']
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
        print(SnapNum_List)
   
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift'], output_input['galaxy_mass_min'], output_input['galaxy_mass_max'], use_satellites))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Min. inclination: %s\n  Min particles: %s\n  Min COM: %.1f pkpc\n  Min Mass: %.2E M*\n  Max limit: %.2E M*\n  ETG or LTG: %s\n  Cluster or field: %s\n  Use satellites:  %s' %(use_angle, use_hmr, use_proj_angle, min_inc_angle, min_particles, min_com, lower_mass_limit, upper_mass_limit, ETG_or_LTG, cluster_or_field, use_satellites))
    print('===================')
    
    #------------------------------
    # Check if requested plot is possible with loaded data
    assert use_angle in output_input['angle_selection'], 'Requested angle %s not in output_input' %use_angle
    assert use_hmr in output_input['spin_hmr'], 'Requested HMR %s not in output_input' %use_hmr
    if use_satellites:
        assert use_satellites == sample_input['use_satellites'], 'Sample does not contain satellites'

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
    
    # Set projection angle criteria
    if not use_proj_angle:
        min_inc_angle = 0
    max_inc_angle = 180 - min_inc_angle
    if output_input['viewing_axis'] == 'x':
        viewing_vector = [1., 0, 0]
    elif output_input['viewing_axis'] == 'y':
        viewing_vector = [0, 1., 0]
    elif output_input['viewing_axis'] == 'z':
        viewing_vector = [0, 0, 1.]
    else:
        raise Exception('Cant read viewing_axis')
        
    #-----------------------------
    # Set definitions
    cluster_threshold     = 1e14
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
        
    # Setting cluster lower and upper boundaries based on inputs
    if cluster_or_field == 'both':
        lower_halo = 0
        upper_halo = 10**16
    elif cluster_or_field == 'cluster':
        lower_halo = cluster_threshold
        upper_halo = 10**16
    elif cluster_or_field == 'field':
        lower_halo = 0
        upper_halo = cluster_threshold
    
    # Setting satellite criteria
    if use_satellites:
        satellite_criteria = 99999999
    if not use_satellites:
        satellite_criteria = 0
    
    collect_ID = []
    #------------------------------
    def _plot_misalignment_distributions(debug=False):
        # We have use_angle = 'stars_gas_sf', and use_particles = ['stars', 'gas_sf'] 
        
        #=================================
        # Collect values to plot
        plot_angles     = []
        plot_angles_err = []
        ID_list  = []
        ETG_list = []
        LTG_list = []
        
        # Collect other useful values
        catalogue = {'total': {},          # Total galaxies in mass range
                     'sample': {},         # Sample of galaxies that meet particle count, COM, inclination angle, regardless of morphology, environment, or satellite status
                     'plot': {}}           # Sub-sample that is plotted (environment/morphology/satellite)
        for key in catalogue.keys():
            catalogue[key] = {'all': 0,        # Size of cluster
                              'cluster': 0,      # number of cluster galaxies
                              'field': 0,      # number of field galaxies
                              'ETG': 0,        # number of ETGs
                              'LTG': 0}        # number of LTGs
        if print_progress:
            print('Analysing extracted sample and collecting angles')
            time_start = time.time()
        
        # Find angle galaxy makes with viewing axis
        def _find_angle(vector1, vector2):
            return np.rad2deg(np.arccos(np.clip(np.dot(vector1/np.linalg.norm(vector1), vector2/np.linalg.norm(vector2)), -1.0, 1.0)))     # [deg]
        
        # Find distance between coms
        def _evaluate_com(com1, com2, abs_proj, debug=False):
            if abs_proj == 'abs':
                d = np.linalg.norm(np.array(com1) - np.array(com2))
            elif abs_proj == 'x':
                d = np.linalg.norm(np.array([com1[1], com1[2]]) - np.array([com2[1], com2[2]]))
            elif abs_proj == 'y':
                d = np.linalg.norm(np.array([com1[0], com1[2]]) - np.array([com2[0], com2[2]]))
            elif abs_proj == 'z':
                d = np.linalg.norm(np.array([com1[0], com1[1]]) - np.array([com2[0], com2[1]]))
            else:
                raise Exception('unknown entery')
            return d
        
        # Add all galaxies loaded to catalogue
        catalogue['total']['all'] = len(GroupNum_List)        # Total galaxies initially
        
        
        #--------------------------
        # Loop over all galaxies we have available, and analyse output of flags
        for GalaxyID in GalaxyID_List:
            
            #-----------------------------
            # Determine if cluster or field, and morphology
            if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                catalogue['total']['cluster'] += 1
            elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                catalogue['total']['field'] += 1
            if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                catalogue['total']['LTG'] += 1
            elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                catalogue['total']['ETG'] += 1
            
            
            #-----------------------------
            # Check if galaxy meets criteria
            
            # check if hmr exists 
            if (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                # creating masks
                mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                
                # find particle counts
                if use_particles[0] == 'dm':
                    count_1 = all_counts['%s' %GalaxyID][use_particles[0]]
                else:
                    count_1 = all_counts['%s' %GalaxyID][use_particles[0]][mask_counts]
                if use_particles[1] == 'dm':
                    count_2 = all_counts['%s' %GalaxyID][use_particles[1]]
                else:
                    count_2 = all_counts['%s' %GalaxyID][use_particles[1]][mask_counts]
                
                # find inclination angle(s)
                inc_angle_1 = _find_angle(all_spins['%s' %GalaxyID][use_particles[0]][mask_spins], viewing_vector)
                inc_angle_2 = _find_angle(all_spins['%s' %GalaxyID][use_particles[1]][mask_spins], viewing_vector)
                
                # find CoMs = com_abs
                if use_angle != 'stars_dm':
                    com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]][mask_angles], 'abs')
                else:
                    com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]], 'abs')
                
                #--------------
                # Determine if this is a galaxy we want to plot and meets the remaining criteria (stellar mass, halo mass, kappa, uncertainty, satellite)
                if use_proj_angle:
                    max_error = max(np.abs((np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle][mask_angles]) - all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])))
                else:
                    max_error = max(np.abs((np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle][mask_angles]) - all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles])))
                
                
                # applying selection criteria for min_inc_angle, min_com, min_particles
                if (count_1 >= min_particles) and (count_2 >= min_particles) and (com_abs <= min_com) and (inc_angle_1 >= min_inc_angle) and (inc_angle_1 <= max_inc_angle) and (inc_angle_2 >= min_inc_angle) and (inc_angle_2 <= max_inc_angle) and (max_error <= (999 if max_uncertainty == None else max_uncertainty)):
                    
                    #--------------
                    catalogue['sample']['all'] += 1
                    
                    # Determine if cluster or field, and morphology
                    if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                        catalogue['sample']['cluster'] += 1
                    elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                        catalogue['sample']['field'] += 1
                    if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                        catalogue['sample']['LTG'] += 1
                    elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                        catalogue['sample']['ETG'] += 1
                    
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria):
                        
                        #--------------
                        catalogue['plot']['all'] += 1
                        
                        # Determine if cluster or field, and morphology
                        if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                            catalogue['plot']['cluster'] += 1
                        elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                            catalogue['plot']['field'] += 1
                        if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                            catalogue['plot']['LTG'] += 1
                            LTG_list.append(GalaxyID)
                        elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                            catalogue['plot']['ETG'] += 1
                            ETG_list.append(GalaxyID)
                            
                        ID_list.append(GalaxyID)
                        #--------------
                        
                        if all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles] > 170:
                            collect_ID.append(GalaxyID)
                        
                        
                        # Collect misangle or misangleproj
                        if use_proj_angle:
                            plot_angles.append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])
                        else:
                            plot_angles.append(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles])
            
                        
        assert catalogue['plot']['all'] == len(plot_angles), 'Number of angles collected does not equal number in catalogue... for some reason'
        if debug:
            print(catalogue['total'])
            print(catalogue['sample'])
            print(catalogue['plot'])
        
        
        #print('\tcollect_ID of counter-rotatos:')
        #print(collect_ID)
        #print(len(collect_ID))
        
        
        #==========================================================================================
        # Plotting
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Plotting')
            time_start = time.time()
        
        # Graph initialising and base formatting
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[10/3, 3], gridspec_kw = {'wspace':0, 'hspace':0}, sharex=True, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=None)
        
        # Setting colors
        if 'gas' in use_particles:
            plot_color = 'green'
        if 'gas_sf' in use_particles:
            plot_color = 'b'
        if 'gas_nsf' in use_particles:
            plot_color = 'indigo'
        if 'dm' in use_particles:
            plot_color = 'r'
        
        plot_color_ETG = 'darkorange'
        plot_color_LTG = 'dodgerblue'
        
        
        """ Some useful quantities:
        catalogue['sample']['all']  = number of galaxies that meet criteria
        catalogue['plot']['all']    = number of galaxies in this particular plot
        """
        
        #-----------
        ### Creating graphs
        
        # Add raimundo observational data
        if add_observational:
                
            # Define percentages
            obs_percent = {'Raimundo': {},
                           'Bryant': {}}
            
            # Add raimundo results:
            obs_percent['Raimundo'] = {'ETG_both':   [0.480, 0.180, 0.048, 0.016, 0.020, 0.016, 0.028, 0.016, 0.016, 0.028, 0.008, 0.005, 0.010, 0.004, 0.012, 0.018, 0.020, 0.065],
                                       'LTG_both':   [0.660, 0.192, 0.072, 0.016, 0.010, 0.006, 0.010, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.001, 0.003, 0.001, 0.003, 0.007]}
            
            # Add Bryant results:
            obs_percent['Bryant'] = {'both_both':    [0.593, 0.214, 0.082, 0.032, 0.017, 0.005, 0.003, 0.003, 0.003, 0.002, 0.008, 0.005, 0.000, 0.000, 0.002, 0.003, 0.010, 0.021],
                                     'both_field':   [0.593, 0.214, 0.082, 0.027, 0.019, 0.004, 0.004, 0.004, 0.004, 0.002, 0.008, 0.004, 0.000, 0.000, 0.002, 0.002, 0.010, 0.023],
                                     'both_cluster': [0.596, 0.213, 0.081, 0.044, 0.015, 0.007, 0.000, 0.000, 0.000, 0.000, 0.007, 0.007, 0.000, 0.000, 0.000, 0.007, 0.007, 0.015],
                                     'ETG_both':     [0.383, 0.159, 0.131, 0.065, 0.037, 0.019, 0.010, 0.019, 0.010, 0.010, 0.047, 0.028, 0.000, 0.000, 0.000, 0.010, 0.019, 0.056],
                                     'ETG_field':    [0.339, 0.145, 0.065, 0.065, 0.065, 0.000, 0.016, 0.032, 0.016, 0.016, 0.081, 0.048, 0.000, 0.000, 0.000, 0.016, 0.032, 0.081],
                                     'ETG_cluster':  [0.444, 0.178, 0.222, 0.067, 0.000, 0.022, 0.000, 0.000, 0.000, 0.000, 0.022, 0.022, 0.000, 0.000, 0.000, 0.000, 0.000, 0.022],
                                     'LTG_both':     [0.647, 0.223, 0.075, 0.021, 0.014, 0.000, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.002, 0.007, 0.002],
                                     'LTG_field':    [0.643, 0.216, 0.089, 0.019, 0.014, 0.000, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.000, 0.008, 0.003],
                                     'LTG_cluster':  [0.677, 0.262, 0.000, 0.031, 0.015, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.015, 0.000, 0.000]}
                                     
            obs_number = {'Bryant': {'both_both': 36+26+45+69+133+72+221,
                                     'both_field': 17+16+29+48+102+57+202,
                                     'both_cluster': 19+10+16+21+31+15+19,
                                     'ETG_both': 36+26+45,
                                     'ETG_field': 17+16+29,
                                     'ETG_cluster': 19+10+16,
                                     'LTG_both': 133+72+221,
                                     'LTG_field': 102+57+202,
                                     'LTG_cluster': 31+15+19}}
                                     
                                     
            obs_data = {'Raimundo': {'ETG_both': [],
                                     'LTG_both': []},
                        'Bryant':   {'both_both': [],
                                     'both_field': [],
                                     'both_cluster': [],
                                     'ETG_both': [],
                                     'ETG_field': [],
                                     'ETG_cluster': [],
                                     'LTG_both': [],
                                     'LTG_field': [],
                                     'LTG_cluster': []}}
            
            for author_i in obs_percent.keys():
                for obs_type_i in obs_percent['%s' %author_i].keys():
                
                    obs_count = 1000*np.array(obs_percent['%s' %author_i][obs_type_i])
                
                    # Append number of galaxies for each angle histogram
                    for obs_count_i, obs_angle_i in zip(obs_count, [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175]):
                        i = 0
                        while i < obs_count_i:
                            obs_data['%s' %author_i][obs_type_i].append(obs_angle_i)
                            i += 1
                
            #---------------       
            # Plot
            axs[0].hist(obs_data['%s' %add_observational]['LTG_%s' %(cluster_or_field)], weights=np.ones(len(obs_data['%s' %add_observational]['LTG_%s' %(cluster_or_field)]))/len(obs_data['%s' %add_observational]['LTG_%s' %(cluster_or_field)]), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='grey', facecolor=None, fill=False, linestyle='-', linewidth=1, alpha=1)
            axs[0].errorbar(np.arange(5, 181, 10), obs_percent['%s' %author_i]['LTG_%s' %(cluster_or_field)], xerr=None, yerr=np.sqrt(np.array(obs_percent['%s' %author_i]['LTG_%s' %(cluster_or_field)]) * obs_number['%s' %author_i]['LTG_%s' %(cluster_or_field)]) / obs_number['%s' %author_i]['LTG_%s' %(cluster_or_field)], ecolor='grey', ls='none', capsize=2, elinewidth=1, markeredgewidth=1, alpha=0.9)
            
            axs[1].hist(obs_data['%s' %add_observational]['ETG_%s' %(cluster_or_field)], weights=np.ones(len(obs_data['%s' %add_observational]['ETG_%s' %(cluster_or_field)]))/len(obs_data['%s' %add_observational]['ETG_%s' %(cluster_or_field)]), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='grey', facecolor=None, fill=False, linestyle='-', linewidth=1, alpha=1)
            axs[1].errorbar(np.arange(5, 181, 10), obs_percent['%s' %author_i]['ETG_%s' %(cluster_or_field)], xerr=None, yerr=np.sqrt(np.array(obs_percent['%s' %author_i]['ETG_%s' %(cluster_or_field)]) * obs_number['%s' %author_i]['ETG_%s' %(cluster_or_field)]) / obs_number['%s' %author_i]['ETG_%s' %(cluster_or_field)], ecolor='grey', ls='none', capsize=2, elinewidth=1, markeredgewidth=1, alpha=0.9)
            
        #-----------    
        # Mask out all LTGs
        mask = np.in1d(np.array(ID_list), np.array(LTG_list))
        plot_angles_LTG = np.array(plot_angles)[mask]

        mask = np.in1d(np.array(ID_list), np.array(LTG_list), invert=True)
        plot_angles_ETG = np.array(plot_angles)[mask]
        
        
        # Plot histogram with ETG, then LTG
        # LTG
        axs[0].hist(plot_angles_LTG, weights=np.ones(catalogue['plot']['LTG'])/catalogue['plot']['LTG'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color_LTG, linewidth=1, alpha=0.1)
        bin_count, _, _ = axs[0].hist(plot_angles_LTG, weights=np.ones(catalogue['plot']['LTG'])/catalogue['plot']['LTG'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color_LTG, facecolor='none', linewidth=1, alpha=0.8)
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(plot_angles_LTG, bins=np.arange(0, 181, 10), range=(0, 180))
        axs[0].errorbar(np.arange(5, 181, 10), hist_n/catalogue['plot']['LTG'], xerr=None, yerr=np.sqrt(hist_n)/catalogue['plot']['LTG'], ecolor=plot_color_LTG, ls='none', capsize=2, elinewidth=1, markeredgewidth=1, alpha=0.9)
        
        # ETG
        axs[1].hist(plot_angles_ETG, weights=np.ones(catalogue['plot']['ETG'])/catalogue['plot']['ETG'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color_ETG, linewidth=1, alpha=0.1)
        bin_count, _, _ = axs[1].hist(plot_angles_ETG, weights=np.ones(catalogue['plot']['ETG'])/catalogue['plot']['ETG'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color_ETG, facecolor='none', linewidth=1, alpha=0.8)
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(plot_angles_ETG, bins=np.arange(0, 181, 10), range=(0, 180))
        axs[1].errorbar(np.arange(5, 181, 10), hist_n/catalogue['plot']['ETG'], xerr=None, yerr=np.sqrt(hist_n)/catalogue['plot']['ETG'], ecolor=plot_color_ETG, ls='none', capsize=2, elinewidth=1, markeredgewidth=0.7, alpha=0.9)
        
        hist_n, _ = np.histogram(plot_angles, bins=np.arange(0, 181, 10), range=(0, 180))
        
        
        #-----------
        ### General formatting
        # Axis labels
        for ax in axs:
            ax.set_xlim(0, 180)
            #ax.set_ylim(bottom=0)
            axs[0].set_ylim(0, 0.84)
            axs[1].set_ylim(0, 0.48)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1, symbol=''))
            ax.set_xticks(np.arange(0, 181, step=30))
            axs[1].set_ylabel('Percentage of galaxies')
            axs[1].get_yaxis().set_label_coords(-0.12,1.0)
        
        if use_proj_angle:
            axs[1].set_xlabel('Misalignment angle, $\psi_{z}$')
        else:
            axs[1].set_xlabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        
        #-----------
        # Annotations
        for ax in axs:
            ax.axvline(misangle_threshold, ls='--', lw=0.5, c='k')
        
        
        #-----------
        ### Legend 1
        # NEW LEGEND
        if use_alternative_format:
            
            #---------------
            # top graph
            legend_elements1 = [Line2D([0], [0], marker=' ', color='w')]
            if (cluster_or_field != 'both'):
                legend_labels1 = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field) + ' LTG']
            elif (cluster_or_field == 'both'):
                legend_labels1 = ['LTG']
            elif (cluster_or_field != 'both'):
                legend_labels1 = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field)]
            else:
                legend_labels = ['EAGLE sample']
            legend_colors1 = [plot_color_LTG]
        
            if add_observational:
                legend_elements1.append(Line2D([0], [0], marker=' ', color='w'))
                legend_labels1.append('%s+%s' %(add_observational, '19' if add_observational == 'Bryant' else '23'))
                legend_colors1.append('grey')
        
            # Add mass range
            if (lower_mass_limit != 10**9.5) and (upper_mass_limit != 10**15):
                legend_labels1.append('$10 ^{%.1f} - 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit), np.log10(upper_mass_limit)))    
                legend_elements1.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors1.append('grey')
            elif (lower_mass_limit != 10**9.5):
                legend_labels1.append('$> 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit)))    
                legend_elements1.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors1.append('grey')
            elif (upper_mass_limit != 10**15):
                legend_labels1.append('$< 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(upper_mass_limit)))    
                legend_elements1.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors1.append('grey')
                
        
            # Add redshift
            legend_labels1.append('${z\sim%.1f}$' %sample_input['Redshift'])
            legend_elements1.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors1.append('k')
        
            legend1 = axs[0].legend(handles=legend_elements1, labels=legend_labels1, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors1, handlelength=0)
            axs[0].add_artist(legend1)
            
            
            #---------------
            # top graph
            legend_elements2 = [Line2D([0], [0], marker=' ', color='w')]
            if (cluster_or_field != 'both'):
                legend_labels2 = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field) + ' ETG']
            elif (cluster_or_field == 'both'):
                legend_labels2 = ['ETG']
            elif (cluster_or_field != 'both'):
                legend_labels2 = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field)]
            else:
                legend_labels = ['EAGLE sample']
            legend_colors2 = [plot_color_ETG]
        
            if add_observational:
                legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
                legend_labels2.append('%s+%s' %(add_observational, '19' if add_observational == 'Bryant' else '23'))
                legend_colors2.append('grey')
        
            # Add mass range
            if (lower_mass_limit != 10**9.5) and (upper_mass_limit != 10**15):
                legend_labels2.append('$10 ^{%.1f} - 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit), np.log10(upper_mass_limit)))    
                legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors2.append('grey')
            elif (lower_mass_limit != 10**9.5):
                legend_labels2.append('$> 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit)))    
                legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors2.append('grey')
            elif (upper_mass_limit != 10**15):
                legend_labels2.append('$< 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(upper_mass_limit)))    
                legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors2.append('grey')
                
            if not add_observational:
                legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
                legend_labels2.append(' ')
                legend_colors2.append('w')
        
            # Add redshift
            legend_labels2.append('${z\sim%.1f}$' %sample_input['Redshift'])
            legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors2.append('k')
        
            legend1 = axs[1].legend(handles=legend_elements2, labels=legend_labels2, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors2, handlelength=0)
            axs[1].add_artist(legend1)
        
           
            
        #-----------
        # other
        plt.tight_layout()
        
        
        #=====================================
        ### Print summary
        
        if print_summary:
            print('CATALOGUE:')
            print('Total: ',  catalogue['total'])
            print('Sample:', catalogue['sample'])
            print('Plot:  ',  catalogue['plot'])
            print('\nRAW BIN VALUES:')      
        aligned_tally           = 0
        aligned_err_tally       = 0
        misaligned_tally        = 0 
        misaligned_err_tally    = 0 
        counter_tally           = 0
        counter_err_tally       = 0
        
        if (int(misangle_threshold) == 30) or (int(misangle_threshold) == 40):
            bins = np.arange(0, 181, 10)
        else:
            bins = np.arange(0, 181, 5)
        hist_n, _ = np.histogram(plot_angles, bins=bins, range=(0, 180))
        for angle_i, bin_count_i in zip(bins, hist_n):
            if print_summary:
                print('  %i' %bin_count_i, end='')
            if angle_i < misangle_threshold:
                aligned_tally += bin_count_i
                aligned_err_tally += bin_count_i**0.5
            if angle_i >= misangle_threshold:
                misaligned_tally += bin_count_i
                misaligned_err_tally += bin_count_i**0.5
            if angle_i >= (180-misangle_threshold):
                counter_tally += bin_count_i
                counter_err_tally += bin_count_i**0.5        
                
                
            
        if print_summary:    
            print('\n')     # total population includes galaxies that failed sample, so can add to less than 100% (ei. remaining % is galaxies that make up non-sample)
            print('OF TOTAL POPULATION: \t(all galaxies in mass range)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['total']['all'], aligned_err_tally*100/catalogue['total']['all'], misaligned_tally*100/catalogue['total']['all'], misaligned_err_tally*100/catalogue['total']['all'], counter_tally*100/catalogue['total']['all'], counter_err_tally*100/catalogue['total']['all']))
            print('OF TOTAL SAMPLE: \t(meeting CoM, inc angle, particle counts, error):\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['sample']['all'], aligned_err_tally*100/catalogue['sample']['all'], misaligned_tally*100/catalogue['sample']['all'], misaligned_err_tally*100/catalogue['sample']['all'], counter_tally*100/catalogue['sample']['all'], counter_err_tally*100/catalogue['sample']['all']))
            print('OF PLOT SAMPLE: \t(specific plot criteria - morph, environ, satellite)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['plot']['all'], aligned_err_tally*100/catalogue['plot']['all'], misaligned_tally*100/catalogue['plot']['all'], misaligned_err_tally*100/catalogue['plot']['all'], counter_tally*100/catalogue['plot']['all'], counter_err_tally*100/catalogue['plot']['all']))       
        
        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        metadata_plot = {'Title': 'TOTAL SAMPLE:\nAligned: %.1f ± %.1f %%\nMisaligned: %.1f ± %.1f %%\nCounter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['sample']['all'], aligned_err_tally*100/catalogue['sample']['all'], misaligned_tally*100/catalogue['sample']['all'], misaligned_err_tally*100/catalogue['sample']['all'], counter_tally*100/catalogue['sample']['all'], counter_err_tally*100/catalogue['sample']['all']),
                         'Author': 'PLOT SAMPLE:\nAligned: %.1f ± %.1f %%\nMisaligned: %.1f ± %.1f %%\nCounter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['plot']['all'], aligned_err_tally*100/catalogue['plot']['all'], misaligned_tally*100/catalogue['plot']['all'], misaligned_err_tally*100/catalogue['plot']['all'], counter_tally*100/catalogue['plot']['all'], counter_err_tally*100/catalogue['plot']['all']),
                         'Subject': str(hist_n),
                         'Producer': str(catalogue)}
       
        if use_satellites:
            sat_str = 'all'
        if not use_satellites:
            sat_str = 'cent'
       
        if add_observational:
            obs_txt = 'OBSERVATIONAL'
        else:
            obs_txt = ''
       
        if savefig:
            plt.savefig("%s/misalignment_distributions/L%s_%s_%s_double_misalignment_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misalignment_distributions/L%s_%s_%s_double_misalignment_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
            
        # Check what is plotted
        """fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        masses = []
        hmr_rad = []    
        colors = []
        count = 0
        ID_failure = []
        for GalaxyID in GalaxyID_List:
            masses.append(all_general['%s' %GalaxyID]['stelmass'])
            hmr_rad.append(all_general['%s' %GalaxyID]['halfmass_rad_proj'])
            
            if use_hmr in all_misangles['%s' %GalaxyID]['hmr']:
                count += 1
                colors.append('b')
            else:
                colors.append('r')
                ID_failure.append(GalaxyID)
        
        print(len(masses))
        print(count)
        print(all_general['8230966']['stelmass'])
        print(all_general['8230966']['halfmass_rad_proj'])
        axs.scatter(np.log10(np.array(masses)), hmr_rad, s=0.5, alpha=0.5, c=colors)
        axs.set_xlim(9.4, 9.6)
        axs.set_ylim(0, 20)
        axs.set_xlabel('Mass', fontsize=12)
        axs.set_ylabel('HMR', fontsize=12)
        set_rc_params(0.9) 
        plt.show()
        """
        
    #---------------------------------
    _plot_misalignment_distributions()
    #---------------------------------
 

#--------------------------------
# Requires manual entry, double plots a graph tracking share of aligned, misaligned, and counter-rotating systems for ETG/LTG, and using values from snips using abs/abs
# SAVED: /plots/misalignment_distributions/combined_...
def _manual_plot_misalignment_double(ETG_proj_array = [649, 341, 206, 121, 76, 56, 53, 37, 30, 25, 19, 27, 33, 39, 32, 46, 79, 89], # from snaps, at z=0.1
                                     LTG_proj_array = [2377, 443, 120, 47, 27, 15, 10, 6, 5, 8, 3, 12, 3, 2, 7, 13, 29, 64],
                                     # from snips, at z=0.1... where rad abs is used
                                     ETG_abs_array  = [558, 420, 232, 167, 116, 77, 67, 49, 50, 48, 39, 45, 35, 39, 50, 58, 82, 70],
                                     LTG_abs_array  = [2500, 423, 72, 38, 15, 18, 9, 8, 5, 3, 6, 8, 5, 9, 5, 15, 28, 59],
                                     #--------------------------
                                     misangle_threshold = 30,             # what we classify as misaligned
                                     #--------------------------
                                     showfig       = True,
                                     savefig       = True,
                                       file_format = 'pdf',
                                       savefig_txt = 'snips',
                                     #--------------------------
                                     print_progress = False,
                                     debug = False):
        
    assert answer == '4', 'will save in snips'
        
    #------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=[10/3, 3], gridspec_kw = {'wspace':0, 'hspace':0}, sharex=True, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=None)
    
    #mpl.bar()
    plot_color_ETG_proj = 'darkorange'
    plot_color_LTG_proj = 'dodgerblue'
    plot_color_ETG_abs  = 'r'
    plot_color_LTG_abs  = 'b'
    
    #-----------
    ### Creating graphs
    
    # Add Bryant observational data
    # Define percentages
    obs_percent = {'Raimundo': {},
                   'Bryant': {}}
    # Add Bryant results:
    obs_percent['Bryant'] = {'both_both':    [0.593, 0.214, 0.082, 0.032, 0.017, 0.005, 0.003, 0.003, 0.003, 0.002, 0.008, 0.005, 0.000, 0.000, 0.002, 0.003, 0.010, 0.021],
                             'both_field':   [0.593, 0.214, 0.082, 0.027, 0.019, 0.004, 0.004, 0.004, 0.004, 0.002, 0.008, 0.004, 0.000, 0.000, 0.002, 0.002, 0.010, 0.023],
                             'both_cluster': [0.596, 0.213, 0.081, 0.044, 0.015, 0.007, 0.000, 0.000, 0.000, 0.000, 0.007, 0.007, 0.000, 0.000, 0.000, 0.007, 0.007, 0.015],
                             'ETG_both':     [0.383, 0.159, 0.131, 0.065, 0.037, 0.019, 0.010, 0.019, 0.010, 0.010, 0.047, 0.028, 0.000, 0.000, 0.000, 0.010, 0.019, 0.056],
                             'ETG_field':    [0.339, 0.145, 0.065, 0.065, 0.065, 0.000, 0.016, 0.032, 0.016, 0.016, 0.081, 0.048, 0.000, 0.000, 0.000, 0.016, 0.032, 0.081],
                             'ETG_cluster':  [0.444, 0.178, 0.222, 0.067, 0.000, 0.022, 0.000, 0.000, 0.000, 0.000, 0.022, 0.022, 0.000, 0.000, 0.000, 0.000, 0.000, 0.022],
                             'LTG_both':     [0.647, 0.223, 0.075, 0.021, 0.014, 0.000, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.002, 0.007, 0.002],
                             'LTG_field':    [0.643, 0.216, 0.089, 0.019, 0.014, 0.000, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.000, 0.008, 0.003],
                             'LTG_cluster':  [0.677, 0.262, 0.000, 0.031, 0.015, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.015, 0.000, 0.000]}
    obs_number = {'Bryant': {'both_both': 36+26+45+69+133+72+221,
                             'both_field': 17+16+29+48+102+57+202,
                             'both_cluster': 19+10+16+21+31+15+19,
                             'ETG_both': 36+26+45,
                             'ETG_field': 17+16+29,
                             'ETG_cluster': 19+10+16,
                             'LTG_both': 133+72+221,
                             'LTG_field': 102+57+202,
                             'LTG_cluster': 31+15+19}}
    
    #---------------       
    # Plot observational
    axs[0].bar(np.arange(5, 181, 10), obs_percent['Bryant']['LTG_both'], edgecolor='grey', facecolor=None, fill=False, linestyle='-', linewidth=1, alpha=1, width=10)  
    axs[0].errorbar(np.arange(5, 181, 10), obs_percent['Bryant']['LTG_both'], xerr=None, yerr=np.sqrt(np.array(obs_percent['Bryant']['LTG_both']) * obs_number['Bryant']['LTG_both']) / obs_number['Bryant']['LTG_both'], ecolor='grey', ls='none', capsize=2, elinewidth=0.7, markeredgewidth=0.7, alpha=0.9)
    
    axs[1].bar(np.arange(5, 181, 10), obs_percent['Bryant']['ETG_both'], edgecolor='grey', facecolor=None, fill=False, linestyle='-', linewidth=1, alpha=1, width=10)  
    axs[1].errorbar(np.arange(5, 181, 10), obs_percent['Bryant']['ETG_both'], xerr=None, yerr=np.sqrt(np.array(obs_percent['Bryant']['ETG_both']) * obs_number['Bryant']['ETG_both']) / obs_number['Bryant']['ETG_both'], ecolor='grey', ls='none', capsize=2, elinewidth=0.7, markeredgewidth=0.7, alpha=0.9)
    
    #---------------       
    # Plot main results
    
    # LTG, proj
    #axs[0].bar(np.arange(5, 181, 10), np.array(LTG_proj_array)/np.sum(LTG_proj_array), edgecolor='none', facecolor=plot_color_LTG_proj, linewidth=1, alpha=0.1, width=10)
    axs[0].bar(np.arange(5, 181, 10), np.array(LTG_proj_array)/np.sum(LTG_proj_array), edgecolor=plot_color_LTG_proj, facecolor='none', linewidth=1, alpha=0.8, width=10)
    # Add poisson errors (sqrt N)
    axs[0].errorbar(np.arange(5, 181, 10), LTG_proj_array/np.sum(LTG_proj_array), xerr=None, yerr=np.sqrt(np.array(LTG_proj_array))/np.sum(LTG_proj_array), ecolor=plot_color_LTG_proj, ls='none', capsize=2, elinewidth=0.7, markeredgewidth=0.7, alpha=0.9)
    
    # LTG, abs
    #axs[0].bar(np.arange(5, 181, 10), np.array(LTG_abs_array)/np.sum(LTG_abs_array), edgecolor='none', facecolor=plot_color_LTG_abs, linewidth=1, alpha=0.1, width=10)
    axs[0].bar(np.arange(5, 181, 10), np.array(LTG_abs_array)/np.sum(LTG_abs_array), edgecolor=plot_color_LTG_abs, facecolor='none', linewidth=1, alpha=0.8, width=10)
    # Add poisson errors (sqrt N)
    axs[0].errorbar(np.arange(5, 181, 10), LTG_abs_array/np.sum(LTG_abs_array), xerr=None, yerr=np.sqrt(np.array(LTG_abs_array))/np.sum(LTG_abs_array), ecolor=plot_color_LTG_abs, ls='none', capsize=2, elinewidth=0.7, markeredgewidth=0.7, alpha=0.9)
    
    # ETG, proj
    #axs[1].bar(np.arange(5, 181, 10), np.array(ETG_proj_array)/np.sum(ETG_proj_array), edgecolor='none', facecolor=plot_color_ETG_proj, linewidth=1, alpha=0.1, width=10)
    axs[1].bar(np.arange(5, 181, 10), np.array(ETG_proj_array)/np.sum(ETG_proj_array), edgecolor=plot_color_ETG_proj, facecolor='none', linewidth=1, alpha=0.8, width=10)
    # Add poisson errors (sqrt N)
    axs[1].errorbar(np.arange(5, 181, 10), ETG_proj_array/np.sum(ETG_proj_array), xerr=None, yerr=np.sqrt(np.array(ETG_proj_array))/np.sum(ETG_proj_array), ecolor=plot_color_ETG_proj, ls='none', capsize=2, elinewidth=0.7, markeredgewidth=0.7, alpha=0.9)
    
    # ETG, abs
    #axs[1].bar(np.arange(5, 181, 10), np.array(ETG_abs_array)/np.sum(ETG_abs_array), edgecolor='none', facecolor=plot_color_ETG_abs, linewidth=1, alpha=0.1, width=10)
    axs[1].bar(np.arange(5, 181, 10), np.array(ETG_abs_array)/np.sum(ETG_abs_array), edgecolor=plot_color_ETG_abs, facecolor='none', linewidth=1, alpha=0.8, width=10)
    # Add poisson errors (sqrt N)
    axs[1].errorbar(np.arange(5, 181, 10), ETG_abs_array/np.sum(ETG_abs_array), xerr=None, yerr=np.sqrt(np.array(ETG_abs_array))/np.sum(ETG_abs_array), ecolor=plot_color_ETG_abs, ls='none', capsize=2, elinewidth=0.7, markeredgewidth=0.7, alpha=0.9)
    
    
      
    #-----------
    ### General formatting
    # Axis labels
    for ax in axs:
        ax.set_xlim(0, 180)
        #ax.set_ylim(bottom=0)
        axs[0].set_ylim(0, 0.84)
        axs[1].set_ylim(0, 0.48)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1, symbol=''))
        ax.set_xticks(np.arange(0, 181, step=30))
        axs[1].set_ylabel('Percentage of galaxies')
        axs[1].get_yaxis().set_label_coords(-0.12,1.0)
    
    axs[1].set_xlabel('Misalignment angle, $\psi$')
    
    #-----------
    # Annotations
    for ax in axs:
        ax.axvline(misangle_threshold, ls='--', lw=0.5, c='k')
    
    
    #-----------
    # LEGEND
    # top graph
    legend_elements1 = [Line2D([0], [0], marker=' ', color='w'), Line2D([0], [0], marker=' ', color='w')]
    legend_labels1 = ['LTG, $\psi_{z}$', 'LTG, $\psi_{\mathrm{3D}}$']
    legend_colors1 = [plot_color_LTG_proj, plot_color_LTG_abs]
            
    legend_elements1.append(Line2D([0], [0], marker=' ', color='w'))
    legend_labels1.append('Bryant+19')
    legend_colors1.append('grey')
    
    # Add redshift
    legend_labels1.append('${z\sim0.1}$')
    legend_elements1.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors1.append('k')

    legend1 = axs[0].legend(handles=legend_elements1, labels=legend_labels1, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors1, handlelength=0)
    axs[0].add_artist(legend1)
    
    # bottom graph
    legend_elements2 = [Line2D([0], [0], marker=' ', color='w'), Line2D([0], [0], marker=' ', color='w')]
    legend_labels2 = ['ETG, $\psi_{z}$', 'ETG, $\psi_{\mathrm{3D}}$']
    legend_colors2 = [plot_color_ETG_proj, plot_color_ETG_abs]
            
    legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
    legend_labels2.append('Bryant+19')
    legend_colors2.append('grey')
    
    # Add redshift
    legend_labels2.append('${z\sim0.1}$')
    legend_elements2.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors2.append('k')

    legend2 = axs[1].legend(handles=legend_elements2, labels=legend_labels2, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors2, handlelength=0)
    axs[1].add_artist(legend2)
        
        
    #-----------
    # other
    plt.tight_layout()
    
    if savefig:
        plt.savefig("%s/misalignment_distributions/combined_double_misalignment_%s.%s" %(fig_dir, savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/misalignment_distributions/combined_double_misalignment_%s.%s" %(fig_dir, savefig_txt, file_format))
    if showfig:
        plt.show()
    plt.close()
    

    
    
    

#======================================================
#--------------------------------
# Plots a graph tracking share of aligned, misaligned, and counter-rotating systems with z
# SAVED: /plots/misalignment_distributions_z/
def _plot_misalignment_z(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                         csv_sample_range = np.arange(19, 29, 1),   # snapnums
                         csv_sample2 = '_all_sample_misalignment_9.5',
                         csv_output_in = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                         #--------------------------
                         # Galaxy plotting
                         print_summary = True,
                           use_angle          = 'stars_gas_sf',         # Which angles to plot
                           use_hmr            = 1.0,                    # Which HMR to use
                           use_proj_angle     = True,                   # Whether to use projected or absolute angle 
                             min_inc_angle    = 10,                     # min. degrees of either spin vector to z-axis, if use_proj_angle
                             min_particles    = 20,               # [ 20 ] number of particles
                             min_com          = 2.0,              # [ 2.0 ] pkpc
                             max_uncertainty  = 30,               # [ None / 30 / 45 ] max uncertainty 
                           lower_mass_limit   = 10**9.5,            # Whether to plot only certain masses 10**15
                           upper_mass_limit   = 10**15,         
                           ETG_or_LTG         = 'both',           # Whether to plot only ETG/LTG
                           cluster_or_field   = 'both',           # Whether to plot only field/cluster
                           use_satellites     = True,             # Whether to include SubGroupNum =/ 0
                         #--------------------------
                         # Misalignment criteria
                         misangle_threshold   = 30,             # what we classify as misaligned
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
    print('PLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Min. inclination: %s\n  Min particles: %s\n  Min CoM: %s pkpc\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  ETG or LTG: %s\n  Cluster or field: %s\n  Use satellites:  %s' %(use_angle, use_hmr, use_proj_angle, min_inc_angle, min_particles, min_com, lower_mass_limit, upper_mass_limit, ETG_or_LTG, cluster_or_field, use_satellites))
    print('===================\n')
    
    #--------------------------------
    # Create arrays to collect data
    plot_dict = {'SnapNum': [],
                 'Redshift': [],
                 'LookbackTime': [],
                 'aligned': [],
                 'aligned_tally': [],
                 'aligned_err': [],
                 'misaligned': [],
                 'misaligned_tally': [],
                 'misaligned_err': [],
                 'counter': [],
                 'counter_tally': [],
                 'counter_err': []}
    
    
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
        SnapNum_List        = np.array(dict_sample['SnapNum'])
        
        # Loading output
        dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
        all_general         = dict_output['all_general']
        all_spins           = dict_output['all_spins']
        all_coms            = dict_output['all_coms']
        all_counts          = dict_output['all_counts']
        all_masses          = dict_output['all_masses']
        all_sfr             = dict_output['all_sfr']
        all_Z               = dict_output['all_Z']
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
        if use_satellites:
            assert use_satellites == sample_input['use_satellites'], 'Sample does not contain satellites'

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
    
        # Set projection angle criteria
        if not use_proj_angle:
            min_inc_angle = 0
        max_inc_angle = 180 - min_inc_angle
        if output_input['viewing_axis'] == 'x':
            viewing_vector = [1., 0, 0]
        elif output_input['viewing_axis'] == 'y':
            viewing_vector = [0, 1., 0]
        elif output_input['viewing_axis'] == 'z':
            viewing_vector = [0, 0, 1.]
        else:
            raise Exception('Cant read viewing_axis')
        
        #-----------------------------
        # Set definitions
        cluster_threshold     = 10**14
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
        
        # Setting cluster lower and upper boundaries based on inputs
        if cluster_or_field == 'both':
            lower_halo = 0
            upper_halo = 10**16
        elif cluster_or_field == 'cluster':
            lower_halo = cluster_threshold
            upper_halo = 10**16
        elif cluster_or_field == 'field':
            lower_halo = 0
            upper_halo = cluster_threshold
    
        # Setting satellite criteria
        if use_satellites:
            satellite_criteria = 99999999
        if not use_satellites:
            satellite_criteria = 0
        
        
        #------------------------------
        def _collect_misalignment_distributions_z(debug=False):
            #=================================
            # Collect values to plot
            plot_angles     = []
            plot_angles_err = []
        
            # Collect other useful values
            catalogue = {'total': {},          # Total galaxies in mass range
                         'sample': {},         # Sample of galaxies
                         'plot': {}}           # Sub-sample that is plotted
            for key in catalogue.keys():
                catalogue[key] = {'all': 0,        # Size of cluster
                                  'cluster': 0,      # number of cluster galaxies
                                  'field': 0,      # number of field galaxies
                                  'ETG': 0,        # number of ETGs
                                  'LTG': 0}        # number of LTGs
        
            # Add all galaxies loaded to catalogue
            catalogue['total']['all'] = len(GroupNum_List)        # Total galaxies initially
        
            if debug:
                print(catalogue['total'])
                print(catalogue['sample'])
                print(catalogue['plot'])
            if print_progress:
                print('Analysing extracted sample and collecting angles')
                time_start = time.time()
        
            # Find angle galaxy makes with viewing axis
            def _find_angle(vector1, vector2):
                return np.rad2deg(np.arccos(np.clip(np.dot(vector1/np.linalg.norm(vector1), vector2/np.linalg.norm(vector2)), -1.0, 1.0)))     # [deg]
            
            # Find distance between coms
            def _evaluate_com(com1, com2, abs_proj, debug=False):
                if abs_proj == 'abs':
                    d = np.linalg.norm(np.array(com1) - np.array(com2))
                elif abs_proj == 'x':
                    d = np.linalg.norm(np.array([com1[1], com1[2]]) - np.array([com2[1], com2[2]]))
                elif abs_proj == 'y':
                    d = np.linalg.norm(np.array([com1[0], com1[2]]) - np.array([com2[0], com2[2]]))
                elif abs_proj == 'z':
                    d = np.linalg.norm(np.array([com1[0], com1[1]]) - np.array([com2[0], com2[1]]))
                else:
                    raise Exception('unknown entery')
                return d
            
            # Add all galaxies loaded to catalogue
            catalogue['total']['all'] = len(GroupNum_List)        # Total galaxies initially
            
            #--------------------------
            # Loop over all galaxies we have available, and analyse output of flags
            for GalaxyID in GalaxyID_List:
            
                #-----------------------------
                # Determine if cluster or field, and morphology
                if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                    catalogue['total']['cluster'] += 1
                elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                    catalogue['total']['field'] += 1
                if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                    catalogue['total']['LTG'] += 1
                elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                    catalogue['total']['ETG'] += 1
            
            
                #-----------------------------
                # Check if galaxy meets criteria
            
                # check if hmr exists 
                if (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                    # creating masks
                    mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                    mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                    mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                    mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                
                    # find particle counts
                    if use_particles[0] == 'dm':
                        count_1 = all_counts['%s' %GalaxyID][use_particles[0]]
                    else:
                        count_1 = all_counts['%s' %GalaxyID][use_particles[0]][mask_counts]
                    if use_particles[1] == 'dm':
                        count_2 = all_counts['%s' %GalaxyID][use_particles[1]]
                    else:
                        count_2 = all_counts['%s' %GalaxyID][use_particles[1]][mask_counts]
                
                    # find inclination angle(s)
                    inc_angle_1 = _find_angle(all_spins['%s' %GalaxyID][use_particles[0]][mask_spins], viewing_vector)
                    inc_angle_2 = _find_angle(all_spins['%s' %GalaxyID][use_particles[1]][mask_spins], viewing_vector)
                
                    # find CoMs = com_abs
                    if use_angle != 'stars_dm':
                        com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]][mask_angles], 'abs')
                    else:
                        com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]], 'abs')
                
                
                    # applying selection criteria for min_inc_angle, min_com, min_particles
                    if (count_1 >= min_particles) and (count_2 >= min_particles) and (com_abs <= min_com) and (inc_angle_1 >= min_inc_angle) and (inc_angle_1 <= max_inc_angle) and (inc_angle_2 >= min_inc_angle) and (inc_angle_2 <= max_inc_angle):
                    
                        #--------------
                        catalogue['sample']['all'] += 1
                    
                        # Determine if cluster or field, and morphology
                        if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                            catalogue['sample']['cluster'] += 1
                        elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                            catalogue['sample']['field'] += 1
                        if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                            catalogue['sample']['LTG'] += 1
                        elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                            catalogue['sample']['ETG'] += 1

                        #--------------
                        # Determine if this is a galaxy we want to plot and meets the remaining criteria (stellar mass, halo mass, kappa, satellite, uncertainty)
                        if use_proj_angle:
                            max_error = max(np.abs((np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle][mask_angles]) - all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])))
                        else:
                            max_error = max(np.abs((np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle][mask_angles]) - all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles])))
                        
                        if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (max_error <= (999 if max_uncertainty == None else max_uncertainty)):
                        
                            #--------------
                            catalogue['plot']['all'] += 1
                        
                            # Determine if cluster or field, and morphology
                            if all_general['%s' %GalaxyID]['halo_mass'] > cluster_threshold:
                                catalogue['plot']['cluster'] += 1
                            elif all_general['%s' %GalaxyID]['halo_mass'] <= cluster_threshold:
                                catalogue['plot']['field'] += 1
                            if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                                catalogue['plot']['LTG'] += 1
                            elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                                catalogue['plot']['ETG'] += 1
                            #--------------
                        
                            # Collect misangle or misangleproj
                            if use_proj_angle:
                                plot_angles.append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])
                            else:
                                plot_angles.append(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles])
                                
        
            assert catalogue['plot']['all'] == len(plot_angles), 'Number of angles collected does not equal number in catalogue... for some reason'
            if debug:
                print(catalogue['total'])
                print(catalogue['sample'])
                print(catalogue['plot'])
        
        
            #=====================================
            # Extracting histogram values
            hist_n, _ = np.histogram(plot_angles, bins=np.arange(0, 181, 5), range=(0, 180))
            
            #=====================================
            ### Print summary    
            aligned_tally           = 0
            aligned_err_tally       = 0
            misaligned_tally        = 0 
            misaligned_err_tally    = 0 
            counter_tally           = 0
            counter_err_tally       = 0
        
            if (int(misangle_threshold) == 30) or (int(misangle_threshold) == 40):
                bins = np.arange(0, 181, 10)
            else:
                bins = np.arange(0, 181, 5)
            hist_n, _ = np.histogram(plot_angles, bins=bins, range=(0, 180))
            for angle_i, bin_count_i in zip(bins, hist_n):
                if angle_i < misangle_threshold:
                    aligned_tally += bin_count_i
                    aligned_err_tally += bin_count_i**0.5
                if angle_i >= misangle_threshold:
                    misaligned_tally += bin_count_i
                    misaligned_err_tally += bin_count_i**0.5
                if angle_i >= (180-misangle_threshold):
                    counter_tally += bin_count_i
                    counter_err_tally += bin_count_i**0.5        
            
            
            if debug:    
                print('\n')     # total population includes galaxies that failed sample, so can add to less than 100% (ei. remaining % is galaxies that make up non-sample)
                print('OF TOTAL POPULATION: \t(all galaxies in mass range)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['total']['all'], aligned_err_tally*100/catalogue['total']['all'], misaligned_tally*100/catalogue['total']['all'], misaligned_err_tally*100/catalogue['total']['all'], counter_tally*100/catalogue['total']['all'], counter_err_tally*100/catalogue['total']['all']))
                print('OF TOTAL SAMPLE: \t(no flags, hmr exists, +/- subhalo):\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['sample']['all'], aligned_err_tally*100/catalogue['sample']['all'], misaligned_tally*100/catalogue['sample']['all'], misaligned_err_tally*100/catalogue['sample']['all'], counter_tally*100/catalogue['sample']['all'], counter_err_tally*100/catalogue['sample']['all']))
                print('OF PLOT SAMPLE: \t(specific plot criteria)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['plot']['all'], aligned_err_tally*100/catalogue['plot']['all'], misaligned_tally*100/catalogue['plot']['all'], misaligned_err_tally*100/catalogue['plot']['all'], counter_tally*100/catalogue['plot']['all'], counter_err_tally*100/catalogue['plot']['all']))       
            

            #-------------
            # Append these values to the plot dict
            plot_dict['SnapNum'].append(output_input['snapNum'])
            plot_dict['Redshift'].append(output_input['Redshift'])
            plot_dict['aligned'].append(aligned_tally*100/catalogue['plot']['all'])
            plot_dict['aligned_tally'].append(aligned_tally)
            plot_dict['aligned_err'].append(aligned_err_tally*100/catalogue['plot']['all'])
            plot_dict['misaligned'].append(misaligned_tally*100/catalogue['plot']['all'])
            plot_dict['misaligned_tally'].append(misaligned_tally)
            plot_dict['misaligned_err'].append(misaligned_err_tally*100/catalogue['plot']['all'])
            plot_dict['counter'].append(counter_tally*100/catalogue['plot']['all'])
            plot_dict['counter_tally'].append(counter_tally)
            plot_dict['counter_err'].append(counter_err_tally*100/catalogue['plot']['all'])
            
         
        #--------------------------------------
        _collect_misalignment_distributions_z()
        #--------------------------------------
            
            
    #================================================ 
    # End of snap loop
    
    # Finding lookbacktimes
    plot_dict['LookbackTime'] = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(plot_dict['Redshift'])).value
        
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        print('Plotting')
        time_start = time.time()
    if debug:
        print(plot_dict['Redshift'])
        print(plot_dict['LookbackTime'])
        print(plot_dict['SnapNum'])
        print(plot_dict['aligned'])
        print(plot_dict['aligned_err'])
        print(plot_dict['misaligned'])
        print(plot_dict['misaligned_err'])
        print(plot_dict['counter_err'])
        print(plot_dict['counter_err'])
        
    
    #-----------------------------------------------
    def _plot_misalignment_z(debug=False):
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        #-----------
        ### Creating graphs
        if len(csv_sample_range) > 30:
            axs.plot(plot_dict['LookbackTime'], plot_dict['aligned'], lw=0.7, color='b', zorder=7)
            axs.fill_between(plot_dict['LookbackTime'], np.array(plot_dict['aligned'])-np.array(plot_dict['aligned_err']), np.array(plot_dict['aligned'])+np.array(plot_dict['aligned_err']), color='b', lw=0, alpha=0.15, zorder=5)
            axs.plot(plot_dict['LookbackTime'], plot_dict['misaligned'], lw=0.7, color='r', zorder=7)
            axs.fill_between(plot_dict['LookbackTime'], np.array(plot_dict['misaligned'])-np.array(plot_dict['misaligned_err']), np.array(plot_dict['misaligned'])+np.array(plot_dict['misaligned_err']), color='r', lw=0, alpha=0.15, zorder=5)
            axs.plot(plot_dict['LookbackTime'], plot_dict['counter'], lw=0.7, color='indigo', zorder=7)
            axs.fill_between(plot_dict['LookbackTime'], np.array(plot_dict['counter'])-np.array(plot_dict['counter_err']), np.array(plot_dict['counter'])+np.array(plot_dict['counter_err']), color='indigo', lw=0, alpha=0.15, zorder=5)
        else:
            axs.errorbar(plot_dict['LookbackTime'], plot_dict['aligned'], yerr=plot_dict['aligned_err'], capsize=3, elinewidth=0.7, markeredgewidth=1, color='b')
            axs.errorbar(plot_dict['LookbackTime'], plot_dict['misaligned'], yerr=plot_dict['misaligned_err'], capsize=3, elinewidth=0.7, markeredgewidth=1, color='r')
            axs.errorbar(plot_dict['LookbackTime'], plot_dict['counter'], yerr=plot_dict['counter_err'], capsize=3, elinewidth=0.7, markeredgewidth=1, color='indigo')
            
        
        #------------------------
        ### General formatting
        # Setting regular axis
        axs.set_xlim(0, 8)
        axs.set_xlabel('Lookback time (Gyr)')
        axs.invert_xaxis()
        axs.set_ylim(0, 100)
        axs.set_yticks(np.arange(0, 101, 20))
        axs.set_ylabel('Percentage of galaxies')
        axs.minorticks_on()
        
        
        #-----------
        # Create redshift axis:
        redshiftticks = [0, 0.1, 0.2, 0.5, 1, 1.5, 2]
        ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
        
        ax_top = axs.twiny()
        ax_top.set_xticks(ageticks)
        ax_top.set_xlim(0, 8)
        ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
        ax_top.set_xlabel('Redshift')
        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major')
        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
        ax_top.invert_xaxis()
        
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels   = []
        legend_colors   = []
        
        
        if (cluster_or_field != 'both') & (ETG_or_LTG != 'both'):
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_labels = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field) + ' ' + ETG_or_LTG]
            legend_colors = ['k']
        elif (cluster_or_field == 'both') & (ETG_or_LTG != 'both'):
            legend_labels = [ETG_or_LTG]
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors = ['k']
        elif (cluster_or_field != 'both') & (ETG_or_LTG == 'both'):
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_labels = ['%s'%('field/group' if cluster_or_field == 'field' else cluster_or_field)]
            legend_colors = ['k']
        
        for line_name, line_color in zip(['Aligned', 'Misaligned', 'Counter-rotating'], ['b', 'r', 'indigo']):
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_labels.append(line_name)
            legend_colors.append(line_color)
        
        # Add mass range
        if (lower_mass_limit != 10**9.5) and (upper_mass_limit != 10**15):
            legend_labels.append('$10 ^{%.1f} - 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit), np.log10(upper_mass_limit)))    
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        elif (lower_mass_limit != 10**9.5):
            legend_labels.append('$> 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit)))    
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        elif (upper_mass_limit != 10**15):
            legend_labels.append('$< 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(upper_mass_limit)))    
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        
        axs.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        
        
        #-----------
        # other
        plt.tight_layout()
        
        
        if print_summary:    
            print('===================')
            print('SUMMARY DATA:')
            #print('  | Redshift | time | Aligned     | Misaligned | Counter-rot |')
            #for snap_p, red_p, time_p, ali_t, mis_t, coun_t in zip(plot_dict['SnapNum'], plot_dict['Redshift'], plot_dict['LookbackTime'], plot_dict['aligned_tally'], plot_dict['misaligned_tally'], plot_dict['counter_tally']):
            #    print('  |   %.2f   | %.2f | %i | %i | %i  |' %(red_p, time_p, ali_t, mis_t, coun_t))
            #print(' ')
            print('  | Redshift | time | Aligned     | Misaligned | Counter-rot |')
            for snap_p, red_p, time_p, ali_p, alierr_p, mis_p, miserr_p, coun_p, counerr_p in zip(plot_dict['SnapNum'], plot_dict['Redshift'], plot_dict['LookbackTime'], plot_dict['aligned'], plot_dict['aligned_err'], plot_dict['misaligned'], plot_dict['misaligned_err'], plot_dict['counter'], plot_dict['counter_err']):
                print('  |   %.2f   | %.2f | %.1f ± %.1f%% | %.1f ± %.1f%% | %.1f ± %.1f%%  |' %(red_p, time_p, ali_p, alierr_p, mis_p, miserr_p, coun_p, counerr_p))
            print('===================')
        
        
        
        
        #print(plot_dict['Redshift'])
        #print(plot_dict['LookbackTime'])
        #print(plot_dict['aligned'])
        #print(plot_dict['aligned_err'])
        #print(plot_dict['misaligned'])
        #print(plot_dict['misaligned_err'])
        #print(plot_dict['counter'])
        #print(plot_dict['counter_err'])
        
            
        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        metadata_rows = '| Redshift | time | Aligned     | Misaligned | Counter-rot |\n'
        for snap_p, red_p, time_p, ali_p, alierr_p, mis_p, miserr_p, coun_p, counerr_p in zip(plot_dict['SnapNum'], plot_dict['Redshift'], plot_dict['LookbackTime'], plot_dict['aligned'], plot_dict['aligned_err'], plot_dict['misaligned'], plot_dict['misaligned_err'], plot_dict['counter'], plot_dict['counter_err']):
            metadata_rows += ('|   %.2f   | %.2f | %.1f ± %.1f%% | %.1f ± %.1f%% | %.1f ± %.1f%%  |\n' %(red_p, time_p, ali_p, alierr_p, mis_p, miserr_p, coun_p, counerr_p))
        metadata_plot = {'Title': metadata_rows}
       
        if use_satellites:
            sat_str = 'all'
        if not use_satellites:
            sat_str = 'cent'
       
        if savefig:
            plt.savefig("%s/misalignment_distributions_z/L%s_ALL_%s_misalignment_summary_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], sat_str, misangle_threshold, use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misalignment_distributions_z/L%s_ALL_%s_misalignment_summary_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], sat_str, misangle_threshold, use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
            
    
    #---------------------
    _plot_misalignment_z()
    #---------------------
    

#--------------------------------
# Requires manual entry, plots a graph tracking share of aligned, misaligned, and counter-rotating systems with z
# SAVED: /plots/misalignment_distributions_z/
def _manual_plot_misalignment_z(Lookbacktime_array         = [7.94, 7.32, 6.65, 5.94, 5.26, 4.12, 3.23, 2.34, 1.33, 0.00],
                                #--- tally :
                                  LTG_aligned_array        = [95.81, 95.18, 95.15, 94.79, 94.60, 94.49, 94.28, 93.52, 93.11, 92.52],    # %
                                  LTG_aligned_err_array    = [2.54, 2.44, 2.36, 2.33, 2.33, 2.34, 2.37, 2.41, 2.46, 2.59],    # % error
                                  LTG_misaligned_array     = [4.18, 4.81, 4.84, 5.20, 5.39, 5.50, 5.71, 6.47, 6.88, 7.47],
                                  LTG_misaligned_err_array = [1.29, 1.36, 1.34, 1.38, 1.36, 1.38, 1.42, 1.52, 1.60, 1.63],
                                  LTG_counter_array        = [1.09, 1.41, 1.65, 1.57, 1.95, 2.31, 2.73, 2.70, 3.21, 4.13],
                                  LTG_counter_err_array    = [0.31, 0.35, 0.36, 0.33, 0.38, 0.42, 0.47, 0.45, 0.52, 0.60],
                                #----
                                  ETG_aligned_array        = [66.94, 66.29, 66.76, 65.67, 64.33, 64.28, 63.45, 63.08, 61.29, 62.57],
                                  ETG_aligned_err_array    = [2.66, 2.67, 2.75, 2.75, 2.78, 2.78, 2.84, 2.89, 2.88, 3.00],
                                  ETG_misaligned_array     = [33.05, 33.70, 33.23, 34.32, 35.66, 35.71, 36.54, 36.91, 38.70, 37.42],
                                  ETG_misaligned_err_array = [4.14, 4.25, 4.31, 4.44, 4.65, 4.67, 4.79, 4.94, 5.11, 5.21],
                                  ETG_counter_array        = [5.74, 6.55, 6.88, 7.60, 8.51, 8.53, 11.09, 10.75, 12.25, 12.05],
                                  ETG_counter_err_array    = [0.80, 0.85, 0.89, 0.96, 1.03, 1.03, 1.21, 1.22, 1.32, 1.35],
                                #--------------------------
                                showfig       = True,
                                savefig       = True,
                                  file_format = 'pdf',
                                  savefig_txt = '',
                                #--------------------------
                                print_progress = False,
                                debug = False):
    
    assert answer == '4', 'will save in snips'
    
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    #-----------
    ### Creating graphs
    # LTG
    axs.errorbar(Lookbacktime_array, LTG_aligned_array, yerr=LTG_aligned_err_array, capsize=3, elinewidth=0.7, linewidth=0.7, markeredgewidth=0.7, color='b', linestyle='-')
    axs.errorbar(Lookbacktime_array, LTG_misaligned_array, yerr=LTG_misaligned_err_array, capsize=3, elinewidth=0.7, linewidth=0.7, markeredgewidth=0.7, color='r', linestyle='-')
    axs.errorbar(Lookbacktime_array, LTG_counter_array, yerr=LTG_counter_err_array, capsize=3, elinewidth=0.7, linewidth=0.7, markeredgewidth=0.7, color='indigo', linestyle='-')
    axs.errorbar(Lookbacktime_array, ETG_aligned_array, yerr=ETG_aligned_err_array, capsize=3, elinewidth=0.7, linewidth=0.7, markeredgewidth=0.7, color='b', linestyle='dashdot')
    axs.errorbar(Lookbacktime_array, ETG_misaligned_array, yerr=ETG_misaligned_err_array, capsize=3, elinewidth=0.7, linewidth=0.7, markeredgewidth=0.7, color='r', linestyle='dashdot')
    axs.errorbar(Lookbacktime_array, ETG_counter_array, yerr=ETG_counter_err_array, capsize=3, elinewidth=0.57, linewidth=0.7, markeredgewidth=0.7, color='indigo', linestyle='dashdot')
    
    #------------------------
    ### General formatting
    # Setting regular axis
    axs.set_xlim(0, 8)
    axs.set_xlabel('Lookback time (Gyr)')
    axs.invert_xaxis()
    axs.set_ylim(0, 100)
    axs.set_yticks(np.arange(0, 101, 20))
    axs.set_ylabel('Percentage of galaxies')
    axs.minorticks_on()
    
    
    #-----------
    # Create redshift axis:
    redshiftticks = [0, 0.1, 0.2, 0.5, 1, 1.5, 2]
    ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
    
    ax_top = axs.twiny()
    ax_top.set_xticks(ageticks)
    ax_top.set_xlim(0, 8)
    ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
    ax_top.set_xlabel('Redshift')
    ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major')
    ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
    ax_top.invert_xaxis()
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels   = []
    legend_colors   = []
    
    legend_elements.append(Line2D([-10], [-8], marker=' ', linestyle='-', linewidth=1, color='k'))
    legend_labels.append('LTG')
    legend_colors.append('k')  
    legend_elements.append(Line2D([-10], [-8], marker=' ', linestyle=':', linewidth=1, color='k'))
    legend_labels.append('ETG')
    legend_colors.append('k')  
    for line_name, line_color in zip(['Aligned', 'Misaligned', 'Counter-rotating'], ['b', 'r', 'indigo']):
        legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
        legend_labels.append(line_name)
        legend_colors.append(line_color)
    
    axs.legend(handles=legend_elements, labels=legend_labels, loc=[0.6, 0.65], frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=1)
    
    
    #-----------
    # other
    plt.tight_layout()
    
    #-----------
    # Savefig   
    if savefig:
        plt.savefig("%s/misalignment_distributions_z/L100_ALL_combined_misalignment_summary_%s.%s" %(fig_dir, savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/misalignment_distributions_z/L100_combined_misalignment_summary_%s.%s" %(fig_dir, savefig_txt, file_format))
    if showfig:
        plt.show()
    plt.close()

    
    



#===========================    
#_plot_misalignment()
#_plot_misalignment_double()
#_manual_plot_misalignment_double()

#_plot_misalignment_z()
_manual_plot_misalignment_z()
#===========================




#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'field')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'cluster')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'both')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'both')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'field')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'field')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'cluster')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'cluster')

#_plot_misalignment(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'both', use_proj_angle     = False, add_observational  = None)
#_plot_misalignment(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'both', use_proj_angle     = False, add_observational  = None)


#_plot_misalignment_double(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both', use_proj_angle     = True, use_hmr = 1.0)
#_plot_misalignment_double(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both', use_proj_angle     = False, use_hmr = 1.0, add_observational  = None)
#_plot_misalignment_double(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both', use_proj_angle     = True, use_hmr = 2.0, add_observational  = None)
#_plot_misalignment_double(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both', use_proj_angle     = False, use_hmr = 2.0, add_observational  = None)
#_plot_misalignment_double(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both', use_proj_angle     = True, add_observational  = None)
#_plot_misalignment_double(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both', use_proj_angle     = False, add_observational  = None)


#_plot_misalignment(csv_sample = 'L100_188_all_sample_misalignment_10.0', csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_')



#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'field')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'cluster')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'both')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'both')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'both', use_proj_angle     = False)
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'both', use_proj_angle     = False)
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'field')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'field')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'cluster')
#_plot_misalignment_z(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'cluster')

#_plot_misalignment_z(csv_sample1 = 'L100_', 
#                     csv_sample_range = np.array([134, 140, 146, 152, 157, 166, 173, 180, 188, 200]), 
#                     csv_sample2 = '_all_sample_misalignment_9.5', 
#                     csv_output_in = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', 
#                     use_angle = 'stars_gas_sf', 
#                     ETG_or_LTG = 'ETG', 
#                     cluster_or_field   = 'both')
#_plot_misalignment_z(csv_sample1 = 'L100_', 
#                     csv_sample_range = np.array([134, 140, 146, 152, 157, 166, 173, 180, 188, 200]), 
#                     csv_sample2 = '_all_sample_misalignment_9.5', 
#                     csv_output_in = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', 
#                     use_angle = 'stars_gas_sf', 
#                     ETG_or_LTG = 'LTG', 
#                     cluster_or_field   = 'both')
#_plot_misalignment_z(csv_sample1 = 'L100_', 
#                     csv_sample_range = np.array([134, 140, 146, 152, 157, 166, 173, 180, 188, 200]), 
#                     csv_sample2 = '_all_sample_misalignment_9.5', 
#                     csv_output_in = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', 
#                     use_angle = 'stars_gas_sf', 
#                     ETG_or_LTG = 'ETG', 
#                     cluster_or_field   = 'field')
#_plot_misalignment_z(csv_sample1 = 'L100_', 
#                     csv_sample_range = np.array([134, 140, 146, 152, 157, 166, 173, 180, 188, 200]), 
#                     csv_sample2 = '_all_sample_misalignment_9.5', 
#                     csv_output_in = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', 
#                     use_angle = 'stars_gas_sf', 
#                     ETG_or_LTG = 'LTG', 
#                     cluster_or_field   = 'field')
#_plot_misalignment_z(csv_sample1 = 'L100_', 
#                     csv_sample_range = np.array([134, 140, 146, 152, 157, 166, 173, 180, 188, 200]), 
#                     csv_sample2 = '_all_sample_misalignment_9.5', 
#                     csv_output_in = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', 
#                     use_angle = 'stars_gas_sf', 
#                     ETG_or_LTG = 'ETG', 
#                     cluster_or_field   = 'cluster')
#_plot_misalignment_z(csv_sample1 = 'L100_', 
#                     csv_sample_range = np.array([134, 140, 146, 152, 157, 166, 173, 180, 188, 200]), 
#                     csv_sample2 = '_all_sample_misalignment_9.5', 
#                     csv_output_in = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', 
#                     use_angle = 'stars_gas_sf', 
#                     ETG_or_LTG = 'LTG', 
#                     cluster_or_field   = 'cluster')


