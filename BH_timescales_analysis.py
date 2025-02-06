import h5py
import numpy as np
import scipy
from scipy import stats
from scipy.stats import gaussian_kde
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, NullFormatter, ScalarFormatter, FuncFormatter)
import seaborn as sns
import pandas as pd
from plotbin.sauron_colormap import register_sauron_colormap
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import astropy.units as u
import csv
import json
import time
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID, ConvertID_noMK, MergerTree
import eagleSqlTools as sql
from graphformat import set_rc_params, lighten_color
from read_dataset_directories import _assign_directories
from extract_misalignment_trees import _extract_BHmis_tree, _refine_BHmis_sample


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#register_sauron_colormap()
#====================================

np.random.seed(0)
random.seed(0)

#--------------------------------
# SCATTER x-y of stellar mass in 2r50 and BH mass at z=0
# HISTOGRAM of M_bh / M_* and KS tests
def _plot_stelmass_BH_scatter(csv_sample = 'L100_27_all_sample_misalignment_9.5',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
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
                         ETG_or_LTG         = 'ETG',           # Whether to plot only ETG/LTG/both
                         cluster_or_field   = 'both',           # Whether to plot only field/cluster/both
                         use_satellites     = True,             # Whether to include SubGroupNum =/ 0
                       #--------------------------
                       misangle_threshold = 30,             # what we classify as misaligned
                       #=====================================
                       plot_histogram = True,               # Plot histogram of Mbh/M* ratio 
                       #=====================================
                       showfig       = True,
                       savefig       = False,
                         file_format = 'pdf',
                         savefig_txt = '',
                         paper_formatting = True,           # turn on for better formatting
                       #--------------------------
                       print_progress = False,
                       debug = False):
                        
                        
                        
    # Ensuring the sample and output originated together
    csv_output = csv_sample + csv_output 
    
        
    #print(output_dir)
    
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
    if answer == '4':
        csv_output = 'july_run/' + csv_output
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
    collect_value = []
    bhmass_plot = []
    stelmass_plot = []
    state_abs_plot = []
    state_proj_plot = []
    ID_plot = []
    #------------------------------
    def _plot_misalignment_distributions(debug=False):
        # We have use_angle = 'stars_gas_sf', and use_particles = ['stars', 'gas_sf'] 
        
        #=================================
        # Collect values to plot
        plot_angles     = []
        plot_angles_err = []
        collect_stelmass = []
        collect_gassf    = []
        
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
        plot_sfgas_counts = []
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
                        
                        if output_input['snapNum'] < 30:
                            bh_i = all_general['%s' %GalaxyID]['bh_mass']
                        elif output_input['snapNum'] > 100:
                            bh_i = all_general['%s' %GalaxyID]['bh_mass_old']
                            
                            
                        if math.isnan(bh_i) == True:
                            continue
                            
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
                            collect_value.append(all_general['%s' %GalaxyID]['kappa_stars'])
                            
                            
                        bhmass_plot.append(bh_i)
                        ID_plot.append(GalaxyID)
                        
                        
                        # CHECKING FOR SPECIFIC GALAXY
                        #if int(GalaxyID) == 24492215:
                        #    print('projected angle of galaxyID: ', GalaxyID)
                        #    print(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])
                        
                        
                        # Collect misangle or misangleproj
                        if use_proj_angle:
                            mis_angle_i = all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles]
                            plot_angles.append(mis_angle_i)
                        else:
                            mis_angle_i = all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles]
                            plot_angles.append(mis_angle_i)
                             
                                
                        # Collecting current state info
                        if all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles] < 30:
                            state_proj_plot.append('aligned')
                        elif all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles] > 150:
                            state_proj_plot.append('counter')
                        else:
                            state_proj_plot.append('misaligned')
                        if all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles] < 30:
                            state_abs_plot.append('aligned')
                        elif all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles] > 150:
                            state_abs_plot.append('counter')
                        else:
                            state_abs_plot.append('misaligned')
                        
                        
                        if (all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles] > 30) & (all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles] < 150):
                            plot_sfgas_counts.append(all_counts['%s' %GalaxyID]['gas_sf'][mask_counts])
                        
                        # Collect stelmass, sfgas
                        if 2.0 in all_masses['%s' %GalaxyID]['hmr']:
                            mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(2.0))[0][0]
                            collect_stelmass.append(all_masses['%s' %GalaxyID]['stars'][mask_masses])
                            stelmass_plot.append(all_masses['%s' %GalaxyID]['stars'][mask_masses])
                            collect_gassf.append(all_masses['%s' %GalaxyID]['gas_sf'][mask_masses])
                        else:
                            mask_masses = -1
                            collect_stelmass.append(all_masses['%s' %GalaxyID]['stars'][mask_masses])
                            stelmass_plot.append(all_masses['%s' %GalaxyID]['stars'][mask_masses])
                            collect_gassf.append(all_masses['%s' %GalaxyID]['gas_sf'][mask_masses])
            
                        
        assert catalogue['plot']['all'] == len(plot_angles), 'Number of angles collected does not equal number in catalogue... for some reason'
        if debug:
            print(catalogue['total'])
            print(catalogue['sample'])
            print(catalogue['plot'])
        
        
        #================================================================================================================================================================================
        # Make dataframe
        df = pd.DataFrame(data={'stelmass': stelmass_plot, 'BH mass': bhmass_plot, 'State proj': state_proj_plot, 'State abs': state_abs_plot, 'GalaxyIDs': ID_plot, 'Angle': plot_angles})
        df['stel BH ratio'] = df['BH mass']/df['stelmass'] 
        
        if use_proj_angle:
            df_co  = df.loc[(df['State proj'] == 'aligned')]
            df_mis = df.loc[(df['State proj'] == 'misaligned')]
            df_cnt = df.loc[(df['State proj'] == 'counter')]
        else:
            df_co  = df.loc[(df['State abs'] == 'aligned')]
            df_mis = df.loc[(df['State abs'] == 'misaligned')]
            df_cnt = df.loc[(df['State abs'] == 'counter')]
        
        
        print('Sub-samples at z=%.2f, number:' %output_input['Redshift'])
        print('  total:       %s' %len(df['GalaxyIDs']))
        print('  aligned:     %s' %len(df_co['GalaxyIDs']))
        print('  misaligned:  %s' %len(df_mis['GalaxyIDs']))
        print('  counter:     %s' %len(df_cnt['GalaxyIDs']))
        print(' ')
        
        #---------------
        # KS test
        print('-------------')
        print('KS-test of (M_bh / M*) distributions:')
        res = stats.ks_2samp(df_co['stel BH ratio'], df_mis['stel BH ratio'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['stel BH ratio'], df_cnt['stel BH ratio'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['stel BH ratio'], df_cnt['stel BH ratio'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        
        
        #---------------------------
        # Check distributions of overmassive/undermassive BH ratios
        if plot_histogram:
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.0], sharex=False, sharey=False) 
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
            #color_dict = {'aligned':'grey', 'misaligned':'orangered', 'counter':'dodgerblue'}
            color_dict         = {'total': 'darkviolet', 'aligned': 'k', 'misaligned':'r', 'counter':'b'}
        
            for df_i, state_i in zip([df_co, df_mis, df_cnt],['aligned', 'misaligned', 'counter']):
                bins = np.arange(-6, -1.5, 0.25)
                axs.hist(np.log10(df_i['stel BH ratio']), log=True, bins=bins, weights=np.ones(len(df_i['stel BH ratio']))/len(df_i['stel BH ratio']), facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=1)
                if state_i == 'aligned':
                    axs.hist(np.log10(df_i['stel BH ratio']), log=True, bins=bins, weights=np.ones(len(df_i['stel BH ratio']))/len(df_i['stel BH ratio']), facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
            
            #--------------
            ### General formatting
            # Axis labels
            axs.set_xlim(-6, -1.5)
            axs.set_ylim(0.001, 1)
            #axs.set_yticks([1, 10, 100, 1000])
            axs.set_xlabel(r'log$_{10}$ $M_{\mathrm{BH}}/M_{*}$ $(2r_{50})$')
            axs.set_ylabel('fraction of sub-sample')
            #axs.minorticks_on()
            #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

            #-----------
            ### Legend
            legend_elements = []
            legend_labels = []
            legend_colors = []
            legend_labels.append('aligned')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append(lighten_color(color_dict['aligned'], 1))
            legend_labels.append('relaxing to stable')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append(lighten_color(color_dict['misaligned'], 1))
            legend_labels.append('counter-rotating')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append(lighten_color(color_dict['counter'], 1))
            axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
            
            #------------
            # Title
            if use_proj_angle:
                axs.set_title(r'%s2D angles, $z\approx0.1$' %('' if ETG_or_LTG == 'both' else '%ss, '%ETG_or_LTG), size=7, loc='left', pad=3)
            else:
                axs.set_title(r'%s$z\approx0.1$' %('' if ETG_or_LTG == 'both' else '%ss, '%ETG_or_LTG), size=7, loc='left', pad=3)
            
            #-----------
            # other
            plt.tight_layout()
            
            if use_satellites:
                sat_str = 'all'
            if not use_satellites:
                sat_str = 'cent'
            obs_txt = ''
            
            if savefig:
                plt.savefig("/Users/c22048063/Documents/EAGLE/plots_snips/BH_massstelmass_z01/L%s_%s_samplehist_%s_misalignment_BH_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
                print("\n  SAVED: /Users/c22048063/Documents/EAGLE/plots_snips/BH_massstelmass_z01/L%s_%s_samplehist_%s_misalignment_BH_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format))
            if showfig:
                plt.show()
            plt.close()
            
        
        #-------------
        ### Plotting
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 1, height_ratios=(4, 1.5),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1])
        
    
        #-----------------
        ### Plot scatter
        color_dict         = {'total': 'darkviolet', 'aligned': 'k', 'misaligned':'r', 'counter':'b'}
        color_dict_scatter = {'total': 'darkviolet', 'aligned': 'k', 'misaligned':'r', 'counter':'b'}
        
        
        if use_proj_angle:
            df_color = df['State proj']
        else:
            df_color = df['State abs']
        axs.scatter(np.log10(df['stelmass']), np.log10(df['BH mass']), s=3, c=[color_dict_scatter[i] for i in df_color], edgecolor='w', marker='.', linewidths=0.1, alpha=0.4, zorder=-2)
        
        
        #----------------
        ### Plot median bins + residuals
        bins = np.arange(9.25+0.125, 11.5+0.1, 0.25)
        delta = bins[1]-bins[0]
        use_percentiles = 16        # 1 sigma
        min_bin_size = 10
        
        median_dict = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        upper_dict  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        lower_dict  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        upper_dict_fill  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        lower_dict_fill  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        mask_dict   = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        for df_i, state_i, ls_i in zip([df, df_co, df_mis, df_cnt], ['total', 'aligned', 'misaligned', 'counter'], ['--', '-', '-.', '-.']):
            linecol_i = color_dict[state_i]
            
            idx  = np.digitize(np.log10(df_i['stelmass']), bins)
                
            running_median = [np.median(df_i['BH mass'][idx==k]) for k in np.arange(1, len(bins)+1)]
            mask_count = np.where(np.array([len(df_i['BH mass'][idx==k]) for k in np.arange(1, len(bins)+1)]) >= min_bin_size)[0]
            running_upper  = [np.percentile(df_i['BH mass'][idx==k], 100-use_percentiles) for k in np.arange(1, len(bins)+1)[mask_count]]
            running_lower  = [np.percentile(df_i['BH mass'][idx==k], use_percentiles) for k in np.arange(1, len(bins)+1)[mask_count]]
            
            bins_plot = np.array(bins+(delta/2))[mask_count]
            median_plot = np.array(running_median)[mask_count]
            
            if state_i in ['aligned', 'misaligned', 'counter']:
                axs.plot(bins_plot, np.log10(median_plot), c=linecol_i, ls=ls_i, lw=1, alpha=1)
            if state_i in ['total']:
                axs.plot(bins_plot, np.log10(median_plot), c=linecol_i, ls=ls_i, lw=1, alpha=0.8)
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_median), c=linecol_i, ls=ls_i, lw=1.1, alpha=1)
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_upper), c=linecol_i, lw=0.5, alpha=0.8, ls='--')
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_lower), c=linecol_i, lw=0.5, alpha=0.8, ls='--')
            
            
            median_dict['%s'%state_i] = running_median
            mask_dict['%s'%state_i]   = mask_count
            upper_dict['%s'%state_i]  = np.log10(np.array(running_upper)) - np.log10(median_plot)      # for use in plt.errorbars
            lower_dict['%s'%state_i]  = np.log10(median_plot) - np.log10(np.array(running_lower))
            upper_dict_fill['%s'%state_i]  = np.log10(np.array(running_upper))                          # for use in fill
            lower_dict_fill['%s'%state_i]  = np.log10(np.array(running_lower))
            
            
        
        #----------------
        ### Plot medians
        #axs.errorbar(np.array(bins+(delta/2))[mask_dict['total']], np.log10(median_dict['total'])[mask_dict['total']], yerr=(lower_dict['total'],upper_dict['total']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['total'], linestyle='-')
        #axs.errorbar(np.array(bins+(delta/2))[mask_dict['aligned']], np.log10(median_dict['aligned'])[mask_dict['aligned']], yerr=(lower_dict['aligned'],upper_dict['aligned']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['aligned'], linestyle='--')
        #axs.errorbar(np.array(bins+(delta/2))[mask_dict['misaligned']], np.log10(median_dict['misaligned'])[mask_dict['misaligned']], yerr=(lower_dict['misaligned'],upper_dict['misaligned']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['misaligned'], linestyle='-.')
        #axs.errorbar(np.array(bins+(delta/2))[mask_dict['counter']], np.log10(median_dict['counter'])[mask_dict['counter']], yerr=(lower_dict['counter'],upper_dict['counter']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['counter'], linestyle='-.')
        
        #axs.fill_between(np.array(bins+(delta/2))[mask_dict['total']], upper_dict_fill['total'], lower_dict_fill['total'], alpha=0.25, lw=0, zorder=-20, color=color_dict['total'])
        axs.fill_between(np.array(bins+(delta/2))[mask_dict['aligned']], upper_dict_fill['aligned'], lower_dict_fill['aligned'], alpha=0.1, lw=0, zorder=-20, color=color_dict['aligned'])
        axs.fill_between(np.array(bins+(delta/2))[mask_dict['misaligned']], upper_dict_fill['misaligned'], lower_dict_fill['misaligned'], alpha=0.2, lw=0, zorder=-20, color=color_dict['misaligned'])
        axs.fill_between(np.array(bins+(delta/2))[mask_dict['counter']], upper_dict_fill['counter'], lower_dict_fill['counter'], alpha=0.2, lw=0, zorder=-20, color=color_dict['counter'])
        
        
        #----------------
        ### Plot residuals
        #ax_res.errorbar(np.array(bins+(delta/2))[mask_dict['total']], (np.log10(median_dict['total']) - np.log10(median_dict['total']))[mask_dict['total']], yerr=(lower_dict['total'],upper_dict['total']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['total'], linestyle='-')
        #ax_res.errorbar(np.array(bins+(delta/2))[mask_dict['aligned']], (np.log10(median_dict['aligned']) - np.log10(median_dict['total']))[mask_dict['aligned']], yerr=(lower_dict['aligned'],upper_dict['aligned']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['aligned'], linestyle='--')
        #ax_res.errorbar(np.array(bins+(delta/2))[mask_dict['misaligned']], (np.log10(median_dict['misaligned']) - np.log10(median_dict['total']))[mask_dict['misaligned']], yerr=(lower_dict['misaligned'],upper_dict['misaligned']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['misaligned'], linestyle='-.')
        #ax_res.errorbar(np.array(bins+(delta/2))[mask_dict['counter']], (np.log10(median_dict['counter']) - np.log10(median_dict['total']))[mask_dict['counter']], yerr=(lower_dict['counter'],upper_dict['counter']), capsize=3, elinewidth=0.7, linewidth=1.1, markeredgewidth=0.7, color=color_dict['counter'], linestyle='-.')
        
        
        ax_res.plot(np.array(bins+(delta/2))[mask_dict['total']], (np.log10(median_dict['total']) - np.log10(median_dict['total']))[mask_dict['total']], c=color_dict['total'], lw=1, ls='--')
        ax_res.plot(np.array(bins+(delta/2))[mask_dict['aligned']], (np.log10(median_dict['aligned']) - np.log10(median_dict['total']))[mask_dict['aligned']], c=color_dict['aligned'], lw=1, ls='-')
        ax_res.plot(np.array(bins+(delta/2))[mask_dict['misaligned']], (np.log10(median_dict['misaligned']) - np.log10(median_dict['total']))[mask_dict['misaligned']], c=color_dict['misaligned'], lw=1, ls='-.')
        ax_res.plot(np.array(bins+(delta/2))[mask_dict['counter']], (np.log10(median_dict['counter']) - np.log10(median_dict['total']))[mask_dict['counter']], c=color_dict['counter'], lw=1, ls='-.')
        
        
        #----------------
        ### KS test residuals
        
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9.3, 11.2)
        axs.set_xticklabels([])
        ax_res.set_xlim(9.3, 11.2)
        ax_res.set_xlabel(r'log$_{10}$ $M_{*}(2r_{50})$ [M$_{\odot}$]')
        axs.set_ylim(5, 9)
        axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{BH}}$ [M$_{\odot}]$')
        ax_res.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
        ax_res.set_ylim(-0.3, 0.7)
        ax_res.set_ylabel('residuals [dex]')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        ax_res.minorticks_on()
        ax_res.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax_res.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('total sample')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['total'])
        
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['aligned'])
        
        legend_labels.append('misaligned $[30^{\circ}-150^{\circ}]$')
        #legend_labels.append('unstable misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
        
        legend_labels.append('counter-rotating')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
        
        #legend_labels.append(r'$z\approx0.1$')
        #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        #legend_colors.append('grey')
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        #------------
        # Title
        if use_proj_angle:
            axs.set_title(r'%s2D angles, $z\approx0.1$' %('' if ETG_or_LTG == 'both' else '%ss, '%ETG_or_LTG), size=7, loc='left', pad=3)
        else:
            axs.set_title(r'%s$z\approx0.1$' %('' if ETG_or_LTG == 'both' else '%ss, '%ETG_or_LTG), size=7, loc='left', pad=3)
        
        
        if use_satellites:
            sat_str = 'all'
        if not use_satellites:
            sat_str = 'cent'
        obs_txt = ''
       
        if savefig:
            plt.savefig("/Users/c22048063/Documents/EAGLE/plots_snips/BH_massstelmass_z01/L%s_%s_%s_misalignment_BH_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: /Users/c22048063/Documents/EAGLE/plots_snips/BH_massstelmass_z01/L%s_%s_%s_misalignment_BH_%s_%s_HMR%s_proj%s_inc%s_m%sm%s_morph%s_env%s_%s_%s.%s" %(output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, cluster_or_field, obs_txt, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
            
    _plot_misalignment_distributions()
    #---------------------------------

#--------------------------------
# HEXBIN x-y of stellar mass in 2r50 and BH mass at z=0, using galaxy tree
def _plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt4Gyr___',
                      #==============================================
                      # Graph settings
                      type_of_fraction      = 'f_mis',      # [ f_mis / f_cnt / f_co ] for 30-150, >150 and <30
                      plot_only_current_aligned = False,
                      #==============================================
                      # Export stelmass
                      export_stelmass   = False,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-----------------------------------
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    
    # Load local_z01_tree
    dict_tree = json.load(open('%s/%s.csv' %(output_dir, local_z01_tree_load), 'r'))
    local_z01_tree = dict_tree['local_z01_tree']
    local_z01_input = dict_tree['local_z01_input']
    

    #------------
    # Extract values we want
    stelmass_plot = []
    stelmass_ap_plot = []
    bhmass_plot   = []
    ID_plot       = []
    f_co_plot     = []
    f_mis_plot    = []
    f_cnt_plot    = []
    state_plot    = []
    for ID_i in local_z01_tree.keys():
        
        stelmass_plot.append(local_z01_tree['%s' %ID_i]['stelmass'][-1])
        stelmass_ap_plot.append(local_z01_tree['%s' %ID_i]['stelmass_ap'][-1])
        bhmass_plot.append(local_z01_tree['%s' %ID_i]['bhmass'][-1])
        ID_plot.append(ID_i)
        
        f_co_plot.append(local_z01_tree['%s' %ID_i]['f_time_co'])
        f_mis_plot.append(local_z01_tree['%s' %ID_i]['f_time_mis'])
        f_cnt_plot.append(local_z01_tree['%s' %ID_i]['f_time_cnt'])
        
        
        if local_z01_tree['%s' %ID_i]['angle'][-1] < 30:
            state_plot.append('aligned')
        elif local_z01_tree['%s' %ID_i]['angle'][-1] > 150:
            state_plot.append('counter')
        else:
            state_plot.append('misaligned')
            
                
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'stelmass_ap': stelmass_ap_plot, 'bhmass': bhmass_plot, 'GalaxyID': ID_plot, 'f_co': f_co_plot, 'f_mis': f_mis_plot, 'f_cnt': f_cnt_plot, 'State': state_plot})
    df['stel BH ratio'] = df['bhmass']/df['stelmass'] 
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    
    
    if not export_stelmass:
        #---------------------------  
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.8], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        color_dict         = {'total': 'darkviolet', 'aligned': 'k', 'misaligned':'r', 'counter':'b'}
        #color_dict_scatter = {'total': 'darkviolet', 'aligned': 'k', 'misaligned':'r', 'counter':'b'}
        cmap = 'YlOrRd'
        norm = colors.Normalize(vmin=-0.)
    
        #------------
        # Hexbin and scatter
        if plot_only_current_aligned:
            print('\n-----------------------\nSample size of local_z01_tree:  ', len(df_co['stelmass']))
            print('    aligned:    ', len(df_co['stelmass']))
            print('    misaligned: ', len(df_mis['stelmass']))
            print('    counter:    ', len(df_cnt['stelmass']))
            hb = axs.hexbin(np.log10(df_co['stelmass']), np.log10(df_co['bhmass']), C=df_co['%s'%type_of_fraction], mincnt=5, gridsize=25, extent=(9.25, 11.5, 5, 10), cmap=cmap, norm=norm, zorder=1)
    

            #color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
            axs.scatter(np.log10(df_co['stelmass']), np.log10(df_co['bhmass']), s=8, c=[color_dict[i] for i in df_co['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)
        else:
            print('\n-----------------------\nSample size of local_z01_tree:  ', len(ID_plot))
            print('    aligned:    ', len(df_co['stelmass']))
            print('    misaligned: ', len(df_mis['stelmass']))
            print('    counter:    ', len(df_cnt['stelmass']))
            hb = axs.hexbin(np.log10(df['stelmass']), np.log10(df['bhmass']), C=df['%s'%type_of_fraction], mincnt=5, gridsize=25, extent=(9.25, 11.5, 5, 10), cmap=cmap, norm=norm, zorder=1)
    

            #color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
            axs.scatter(np.log10(df['stelmass']), np.log10(df['bhmass']), s=8, c=[color_dict[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)
   
        
        #---------------
        # KS test
        print('-------------')
        print('KS-test of (M_bh / M*) distributions:')
        res = stats.ks_2samp(df_co['stel BH ratio'], df_mis['stel BH ratio'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['stel BH ratio'], df_cnt['stel BH ratio'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['stel BH ratio'], df_cnt['stel BH ratio'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        
        
        #----------------
        ### Plot median bins + residuals
        bins = np.arange(9.25+0.125, 11.5+0.1, 0.25)
        delta = bins[1]-bins[0]
        use_percentiles = 16        # 1 sigma
        min_bin_size = 10
    
        median_dict = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        upper_dict  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        lower_dict  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        upper_dict_fill  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        lower_dict_fill  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        mask_dict   = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        for df_i, state_i, ls_i in zip([df, df_co, df_mis, df_cnt], ['total', 'aligned', 'misaligned', 'counter'], ['--', '-', '-.', '-.']):
            linecol_i = color_dict[state_i]
        
            idx  = np.digitize(np.log10(df_i['stelmass']), bins)
            
            running_median = [np.median(df_i['bhmass'][idx==k]) for k in np.arange(1, len(bins)+1)]
            mask_count = np.where(np.array([len(df_i['bhmass'][idx==k]) for k in np.arange(1, len(bins)+1)]) >= min_bin_size)[0]
            running_upper  = [np.percentile(df_i['bhmass'][idx==k], 100-use_percentiles) for k in np.arange(1, len(bins)+1)[mask_count]]
            running_lower  = [np.percentile(df_i['bhmass'][idx==k], use_percentiles) for k in np.arange(1, len(bins)+1)[mask_count]]
        
            bins_plot = np.array(bins+(delta/2))[mask_count]
            median_plot = np.array(running_median)[mask_count]
        
            if state_i in ['aligned', 'misaligned', 'counter']:
                axs.plot(bins_plot, np.log10(median_plot), c=linecol_i, ls=ls_i, lw=1, alpha=1)
            if state_i in ['total']:
                axs.plot(bins_plot, np.log10(median_plot), c=linecol_i, ls=ls_i, lw=1, alpha=0.8)
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_median), c=linecol_i, ls=ls_i, lw=1.1, alpha=1)
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_upper), c=linecol_i, lw=0.5, alpha=0.8, ls='--')
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_lower), c=linecol_i, lw=0.5, alpha=0.8, ls='--')
        
        
            median_dict['%s'%state_i] = running_median
            mask_dict['%s'%state_i]   = mask_count
            upper_dict['%s'%state_i]  = np.log10(np.array(running_upper)) - np.log10(median_plot)      # for use in plt.errorbars
            lower_dict['%s'%state_i]  = np.log10(median_plot) - np.log10(np.array(running_lower))
            upper_dict_fill['%s'%state_i]  = np.log10(np.array(running_upper))                          # for use in fill
            lower_dict_fill['%s'%state_i]  = np.log10(np.array(running_lower))
        
        
        
        #------------
        # Colorbar
        if type_of_fraction == 'f_mis':
            colorbar_angle = r'=[30^{\circ}-150^{\circ}]'
        if type_of_fraction == 'f_cnt':
            colorbar_angle = r'>150^{\circ}'
        if type_of_fraction == 'f_co':
            colorbar_angle = r'<30^{\circ}'
        cb = fig.colorbar(hb, ax=axs, label=r'$\langle$time spent in $\psi_{\mathrm{3D}}%s$$\rangle$' %colorbar_angle)
    
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9.3, 11.2)
        axs.set_ylim(5, 10)
        axs.set_xlabel(r'log$_{10}$ $M_{*}(2r_{50})$ [M$_{\odot}$]')
        axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{BH}}$ [M$_{\odot}]$')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
        #-----------
        # Title
        #axs.set_title(r'Reliable kinematics for %.1f Gyr %s' %(local_z01_input['require_min_lookbacktime'], (', currently aligned' if plot_only_current_aligned else '')), size=7, loc='left', pad=3)
        axs.set_title(r'%s$z\approx0.1$'%('Aligned sub-sample at ' if plot_only_current_aligned else ''), size=7, loc='left', pad=3)
    
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('total sample')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['total'])
    
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['aligned'])
    
        legend_labels.append('misaligned $[30^{\circ}-150^{\circ}]$')
        #legend_labels.append('unstable misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
    
        legend_labels.append('counter-rotating')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
    
        #legend_labels.append(r'$z\approx0.1$')
        #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        #legend_colors.append('grey')
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
        #-----------
        # other
        plt.tight_layout()
    
        if plot_only_current_aligned:
            savefig_txt = savefig_txt + '_currently_aligned'

        if savefig:
            plt.savefig("/Users/c22048063/Documents/EAGLE/plots_snips/BH_hexbin_overmassive/L100_overmassiveBH_%s_hexbin_window%sGyr_%s.%s" %(type_of_fraction, local_z01_input['require_min_lookbacktime'], savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: /Users/c22048063/Documents/EAGLE/plots_snips/BH_hexbin_overmassive/L100_overmassiveBH_%s_hexbin_window%sGyr_%s.%s" %(type_of_fraction, local_z01_input['require_min_lookbacktime'], savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
    else:
        return df['stelmass_ap'], "%s_hexbin_window%sGyr_%s.%s" %(type_of_fraction, local_z01_input['require_min_lookbacktime'], savefig_txt, file_format)
# HEXBIN x-y of stellar mass in 2r50 and BH mass at z=0, using galaxy tree, plotting gassf
def _plot_stelmass_BH_hexbin_gassf_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt4Gyr___',
                      #==============================================
                      # Graph settings
                      plot_only_current_aligned = False,
                      #==============================================
                      # Export stelmass
                      export_stelmass   = False,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-----------------------------------
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    
    # Load local_z01_tree
    dict_tree = json.load(open('%s/%s.csv' %(output_dir, local_z01_tree_load), 'r'))
    local_z01_tree = dict_tree['local_z01_tree']
    local_z01_input = dict_tree['local_z01_input']
    

    #------------
    # Extract values we want
    stelmass_plot = []
    stelmass_ap_plot = []
    bhmass_plot   = []
    ID_plot       = []
    f_fgassf_plot = []
    state_plot    = []
    for ID_i in local_z01_tree.keys():
        
        stelmass_plot.append(local_z01_tree['%s' %ID_i]['stelmass'][-1])
        stelmass_ap_plot.append(local_z01_tree['%s' %ID_i]['stelmass_ap'][-1])
        bhmass_plot.append(local_z01_tree['%s' %ID_i]['bhmass'][-1])
        ID_plot.append(ID_i)
        
        f_fgassf_plot.append(local_z01_tree['%s' %ID_i]['f_time_gasrich'])
        
        
        if local_z01_tree['%s' %ID_i]['angle'][-1] < 30:
            state_plot.append('aligned')
        elif local_z01_tree['%s' %ID_i]['angle'][-1] > 150:
            state_plot.append('counter')
        else:
            state_plot.append('misaligned')
            
                
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'stelmass_ap': stelmass_ap_plot, 'bhmass': bhmass_plot, 'GalaxyID': ID_plot, 'f_fgassf': f_fgassf_plot, 'State': state_plot})
    df['stel BH ratio'] = df['bhmass']/df['stelmass'] 
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    
    
    if not export_stelmass:
        #---------------------------  
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.8], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        color_dict         = {'total': 'darkviolet', 'aligned': 'k', 'misaligned':'r', 'counter':'b'}
        #color_dict_scatter = {'total': 'darkviolet', 'aligned': 'k', 'misaligned':'r', 'counter':'b'}
        cmap = 'YlGn'
        norm = colors.Normalize(vmin=-0.)
    
        #------------
        # Hexbin and scatter
        if plot_only_current_aligned:
            print('\n-----------------------\nSample size of local_z01_tree:  ', len(df_co['stelmass']))
            print('    aligned:    ', len(df_co['stelmass']))
            print('    misaligned: ', len(df_mis['stelmass']))
            print('    counter:    ', len(df_cnt['stelmass']))
            hb = axs.hexbin(np.log10(df_co['stelmass']), np.log10(df_co['bhmass']), C=df_co['f_fgassf'], mincnt=5, gridsize=25, extent=(9.25, 11.5, 5, 10), cmap=cmap, norm=norm, zorder=1)
    

            #color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
            axs.scatter(np.log10(df_co['stelmass']), np.log10(df_co['bhmass']), s=8, c=[color_dict[i] for i in df_co['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)
        else:
            print('\n-----------------------\nSample size of local_z01_tree:  ', len(ID_plot))
            print('    aligned:    ', len(df_co['stelmass']))
            print('    misaligned: ', len(df_mis['stelmass']))
            print('    counter:    ', len(df_cnt['stelmass']))
            hb = axs.hexbin(np.log10(df['stelmass']), np.log10(df['bhmass']), C=df['f_fgassf'], mincnt=5, gridsize=25, extent=(9.25, 11.5, 5, 10), cmap=cmap, norm=norm, zorder=1)
    

            #color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
            axs.scatter(np.log10(df['stelmass']), np.log10(df['bhmass']), s=8, c=[color_dict[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)
   
        
        #---------------
        # KS test
        print('-------------')
        print('KS-test of (M_bh / M*) distributions:')
        res = stats.ks_2samp(df_co['stel BH ratio'], df_mis['stel BH ratio'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['stel BH ratio'], df_cnt['stel BH ratio'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['stel BH ratio'], df_cnt['stel BH ratio'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        
        
        #----------------
        ### Plot median bins + residuals
        bins = np.arange(9.25+0.125, 11.5+0.1, 0.25)
        delta = bins[1]-bins[0]
        use_percentiles = 16        # 1 sigma
        min_bin_size = 10
    
        median_dict = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        upper_dict  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        lower_dict  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        upper_dict_fill  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        lower_dict_fill  = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        mask_dict   = {'total': 1, 'aligned': 1, 'misaligned': 1, 'counter': 1}
        for df_i, state_i, ls_i in zip([df, df_co, df_mis, df_cnt], ['total', 'aligned', 'misaligned', 'counter'], ['--', '-', '-.', '-.']):
            linecol_i = color_dict[state_i]
        
            idx  = np.digitize(np.log10(df_i['stelmass']), bins)
            
            running_median = [np.median(df_i['bhmass'][idx==k]) for k in np.arange(1, len(bins)+1)]
            mask_count = np.where(np.array([len(df_i['bhmass'][idx==k]) for k in np.arange(1, len(bins)+1)]) >= min_bin_size)[0]
            running_upper  = [np.percentile(df_i['bhmass'][idx==k], 100-use_percentiles) for k in np.arange(1, len(bins)+1)[mask_count]]
            running_lower  = [np.percentile(df_i['bhmass'][idx==k], use_percentiles) for k in np.arange(1, len(bins)+1)[mask_count]]
        
            bins_plot = np.array(bins+(delta/2))[mask_count]
            median_plot = np.array(running_median)[mask_count]
        
            if state_i in ['aligned', 'misaligned', 'counter']:
                axs.plot(bins_plot, np.log10(median_plot), c=linecol_i, ls=ls_i, lw=1, alpha=1)
            if state_i in ['total']:
                axs.plot(bins_plot, np.log10(median_plot), c=linecol_i, ls=ls_i, lw=1, alpha=0.8)
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_median), c=linecol_i, ls=ls_i, lw=1.1, alpha=1)
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_upper), c=linecol_i, lw=0.5, alpha=0.8, ls='--')
                #axs.plot(np.array(bins+(delta/2))[mask_count], np.log10(running_lower), c=linecol_i, lw=0.5, alpha=0.8, ls='--')
        
        
            median_dict['%s'%state_i] = running_median
            mask_dict['%s'%state_i]   = mask_count
            upper_dict['%s'%state_i]  = np.log10(np.array(running_upper)) - np.log10(median_plot)      # for use in plt.errorbars
            lower_dict['%s'%state_i]  = np.log10(median_plot) - np.log10(np.array(running_lower))
            upper_dict_fill['%s'%state_i]  = np.log10(np.array(running_upper))                          # for use in fill
            lower_dict_fill['%s'%state_i]  = np.log10(np.array(running_lower))
        
        
        
        #------------
        # Colorbar
        cb = fig.colorbar(hb, ax=axs, label=r'$\langle$time spent $f_{\mathrm{gas,SF}}>0.1\rangle$')
    
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9.3, 11.2)
        axs.set_ylim(5, 10)
        axs.set_xlabel(r'log$_{10}$ $M_{*}(2r_{50})$ [M$_{\odot}$]')
        axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{BH}}$ [M$_{\odot}]$')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
        #-----------
        # Title
        #axs.set_title(r'Reliable kinematics for %.1f Gyr %s' %(local_z01_input['require_min_lookbacktime'], (', currently aligned' if plot_only_current_aligned else '')), size=7, loc='left', pad=3)
        axs.set_title(r'%s$z\approx0.1$'%('Aligned sub-sample at ' if plot_only_current_aligned else ''), size=7, loc='left', pad=3)
    
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('total sample')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['total'])
    
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['aligned'])
    
        legend_labels.append('misaligned $[30^{\circ}-150^{\circ}]$')
        #legend_labels.append('unstable misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
    
        legend_labels.append('counter-rotating')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
    
        #legend_labels.append(r'$z\approx0.1$')
        #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        #legend_colors.append('grey')
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
        #-----------
        # other
        plt.tight_layout()
    
        if plot_only_current_aligned:
            savefig_txt = savefig_txt + '_currently_aligned'

        if savefig:
            plt.savefig("/Users/c22048063/Documents/EAGLE/plots_snips/BH_hexbin_overmassive/L100_overmassiveBH_fgassf_hexbin_window%sGyr_%s.%s" %(local_z01_input['require_min_lookbacktime'], savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: /Users/c22048063/Documents/EAGLE/plots_snips/BH_hexbin_overmassive/L100_overmassiveBH_fgassf_hexbin_window%sGyr_%s.%s" %(local_z01_input['require_min_lookbacktime'], savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
    else:
        return df['stelmass_ap'], "fgassf_hexbin_gassf_window%sGyr_%s.%s" %(local_z01_input['require_min_lookbacktime'], savefig_txt, file_format)

        
#--------------------------------
# LINEPLOT of stellar mass function for comparisson    
def _plot_stellar_mass_func(csv_sample = 'L100_27_all_sample_misalignment_9.5',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                            csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                            import_stelmass     = 1,
                            savefig_txt_input   = 1,
                            #==============================================
                            # What determines our final sample
                            print_summary = True,
                              use_angle           = 'stars_gas_sf',   # Which angles to ensure we have
                              use_hmr             = 1.0,              # Which HMR ^
                              use_proj_angle      = True,                   # Whether to use projected or absolute angle 10**9
                                min_inc_angle     = 0,                     # min. degrees of either spin vector to z-axis, if use_proj_angle
                                min_particles     = 20,               # [ 20 ] number of particles
                                min_com           = 2.0,              # [ 2.0 ] pkpc
                                max_uncertainty   = 30,            # [ None / 30 / 45 ]                  Degrees
                                lower_mass_limit  = 10**9.5,            # Lower limit of chosen sample
                                upper_mass_limit  = 10**15,            # Lower limit of chosen sample
                              pop_mass_limit      = 10**8.5,            # Lower limit of population plot sampled
                              use_satellites      = True,
                            #--------------------------
                            hist_bin_width = 0.1,
                            #==============================================
                            showfig       = True,
                            savefig       = False,    
                              file_format   = 'pdf',
                              savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                            #-----------------------------
                            debug = False):
        
        #-----------------------------------
        assert (answer == '4'), 'Must use snips' 
        # Directories
        EAGLE_dir_snap       = '/Users/c22048063/Documents/EAGLE'
        dataDir_main_snap    = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
        treeDir_main_snap    = '/Users/c22048063/Desktop/'
        
        # Other directories
        sample_dir_snap      = EAGLE_dir + '/samples'
        output_dir_snap      = EAGLE_dir + '/outputs'
        fig_dir_snap         = EAGLE_dir + '/plots'
    
        # Ensuring the sample and output originated together
        csv_output_snap = csv_sample + csv_output
    
        #-------------------------
        # Loading sample
        dict_sample = json.load(open('%s/%s.csv' %(sample_dir_snap, csv_sample), 'r'))
        GroupNum_List       = np.array(dict_sample['GroupNum'])
        SubGroupNum_List    = np.array(dict_sample['SubGroupNum'])
        GalaxyID_List       = np.array(dict_sample['GalaxyID'])
        SnapNum_List        = np.array(dict_sample['SnapNum'])
        
        # Loading output
        dict_output = json.load(open('%s/%s.csv' %(output_dir_snap, csv_output_snap), 'r'))
        all_general         = dict_output['all_general']
        all_spins           = dict_output['all_spins']
        all_coms            = dict_output['all_coms']
        all_counts          = dict_output['all_counts']
        all_masses          = dict_output['all_masses']
        #all_sfr             = dict_output['all_sfr']
        #all_Z               = dict_output['all_Z']
        all_misangles       = dict_output['all_misangles']
        all_misanglesproj   = dict_output['all_misanglesproj']
        all_flags           = dict_output['all_flags']
    
        # Loading sample criteria
        sample_input        = dict_sample['sample_input']
        output_input        = dict_output['output_input']
        
    
        print('\n===================')
        print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\nSatellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_min'], sample_input['galaxy_mass_max'], sample_input['use_satellites']))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
        print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Min. inclination: %s\n  Population mass limit: %.2E M*\n  Sample mass min/max:     %.2E / %.2E M*\n  Use satellites:  %s' %(use_angle, use_hmr, use_proj_angle, min_inc_angle, pop_mass_limit, lower_mass_limit, upper_mass_limit, use_satellites))
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
            plot_label = 'Stars-gas$_{sf}$'
        if use_angle == 'stars_gas_nsf':
            if 'stars' not in use_particles:
                use_particles.append('stars')
            if 'gas_nsf' not in use_particles:
                use_particles.append('gas_nsf')
            plot_label = 'Stars-gas$_{nsf}$'
        if use_angle == 'gas_sf_gas_nsf':
            if 'gas_sf' not in use_particles:
                use_particles.append('gas_sf')
            if 'gas_nsf' not in use_particles:
                use_particles.append('gas_nsf')
            plot_label = 'gas$_{sf}$-gas$_{nsf}$'
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
            plot_label = 'Gas$_{sf}$-DM'
        if use_angle == 'gas_nsf_dm':
            if 'gas_nsf' not in use_particles:
                use_particles.append('gas_nsf')
            if 'dm' not in use_particles:
                use_particles.append('dm')
            plot_label = 'Gas$_{nsf}$-DM'
    
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
    
        # Setting satellite criteria
        if use_satellites:
            satellite_criteria = 99999999
        if not use_satellites:
            satellite_criteria = 0
        
        
        #===========================================================
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
        #=========================================
        # Construct and execute query for each simulation. This query returns the number of galaxies 
        for sim_name, sim_size in sample_input['mySims']:
            con = sql.connect('lms192', password='dhuKAP62')
        
        	# for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width). 
            myQuery = 'SELECT \
                         %f+floor(log10(AP.Mass_Star)/%f)*%f as mass, \
                         count(*) as num \
                       FROM \
                         %s_SubHalo as SH, \
                         %s_Aperture as AP \
                       WHERE \
                         SH.SnapNum = %i \
    			         and AP.Mass_Star >= %f \
                         and AP.ApertureSize = 30 \
                         and SH.GalaxyID = AP.GalaxyID \
                       GROUP BY \
    			         %f+floor(log10(AP.Mass_Star)/%f)*%f \
                       ORDER BY \
    			         mass'%(hist_bin_width/2, hist_bin_width, hist_bin_width, sim_name, sim_name, sample_input['snapNum'], pop_mass_limit, hist_bin_width/2, hist_bin_width, hist_bin_width)
                    
            # Execute query.
            myData 	= sql.execute_query(con, myQuery)
        
            if debug:
                print('SQL number:  ', myData['num'][:])
            
        
            # Normalize by volume and bin width.
            hist = myData['num'][:] / (float(sim_size))**3.
            hist = hist / hist_bin_width
        
            axs.plot(myData['mass'], np.log10(hist), label=sim_name, linewidth=1, c='indigo')
    
        
        #=========================================
        # Collect values to plot from output_ data
        plot_stelmass = []
        
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
        
        
        #--------------------------
        # Loop over all galaxies we have available, and analyse output of flags
        for GalaxyID in GalaxyID_List:
            
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
                    # Determine if this is a galaxy we want to plot and meets the remaining criteria (stellar mass, halo mass, kappa, satellite, uncertainty)
                    max_error = max(np.abs((np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle][mask_angles]) - all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])))
                    
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (max_error <= (999 if max_uncertainty == None else max_uncertainty)):
                        
                        plot_stelmass.append(all_general['%s' %GalaxyID]['stelmass'])
                    
                    else:
                        continue
                else:
                    continue
            else:
                continue       
        if print_summary:
            print('\nPLOT CRITERIA SIZE: %s' %(len(plot_stelmass)))
            
        
        #--------------------------
        if np.log10(lower_mass_limit) == 9.5:
            mod_lower_mass_limit = 10**9.0
        else:
            mod_lower_mass_limit = np.log10(lower_mass_limit) 
        
        # Create histogram of our sample    
        hist_sample, _ = np.histogram((hist_bin_width/2)+np.floor(np.log10(plot_stelmass)/hist_bin_width)*hist_bin_width , bins=np.arange(np.log10(mod_lower_mass_limit)+(hist_bin_width/2), np.log10(upper_mass_limit), hist_bin_width))
        hist_sample = hist_sample[:] / (float(sample_input['mySims'][0][1]))**3
        hist_sample = hist_sample / hist_bin_width                      # why?
        hist_bins   = np.arange(np.log10(mod_lower_mass_limit)+(hist_bin_width/2), np.log10(upper_mass_limit)-hist_bin_width, hist_bin_width)
        
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask = np.isfinite(np.log10(hist_sample))
        hist_sample = hist_sample[hist_mask]
        hist_bins   = hist_bins[hist_mask]
            
        axs.plot(hist_bins, np.log10(hist_sample), label='Sample selection', ls='--', linewidth=1, c='r')
        
        
        #=============================================
        # Import the collected stellarmass, import_stelmass
        
        
        # Create histogram of our sample    
        hist_sample, _ = np.histogram((hist_bin_width/2)+np.floor(np.log10(import_stelmass)/hist_bin_width)*hist_bin_width , bins=np.arange(np.log10(mod_lower_mass_limit)+(hist_bin_width/2), np.log10(upper_mass_limit), hist_bin_width))
        hist_sample = hist_sample[:] / (float(sample_input['mySims'][0][1]))**3
        hist_sample = hist_sample / hist_bin_width                      # why?
        hist_bins   = np.arange(np.log10(mod_lower_mass_limit)+(hist_bin_width/2), np.log10(upper_mass_limit)-hist_bin_width, hist_bin_width)
        
        # Masking out nans
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_mask = np.isfinite(np.log10(hist_sample))
        hist_sample = hist_sample[hist_mask]
        hist_bins   = hist_bins[hist_mask]
            
        axs.plot(hist_bins, np.log10(hist_sample), label='Lookback sample', ls='-.', linewidth=1, c='b')
        
        print('PLOT Z01 TREE SIZE: ', len(import_stelmass))
        
        
        
        #-----------
        # Axis formatting
        plt.xlim(9.3, 12)
        plt.ylim(-5, -1.5)
        plt.yticks(np.arange(-5, -1.4, 0.5))
        plt.xlabel(r'log$_{10}$ $M_{*}$ [M$_{\odot}$]')
        plt.ylabel(r'log$_{10}$ dn/dlog$_{10}$($M_{*}$)'+'\n'+'[cMpc$^{-3}$]')
        plt.xticks(np.arange(9.5, 12.5, 0.5))
          
        #-----------  
        # Annotations
        
        #-----------
        # Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append(sim_name)
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('indigo')
        legend_labels.append('Baker+24 selection')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('r')
        legend_labels.append('Sample selection')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('b')
        legend_labels.append('${z=%.2f}$' %sample_input['Redshift'])
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')

        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        #axs.plot(0, 0, label='${z=%.2f}$' %sample_input['Redshift'], c='k')    # Creating fake plot to add redshift
        #axs.legend(loc='upper right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
        #-----------
        # other
        plt.tight_layout()

        if savefig:
            plt.savefig("/Users/c22048063/Documents/EAGLE/plots_snips/BH_hexbin_overmassive/%s" %(savefig_txt_input), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: /Users/c22048063/Documents/EAGLE/plots_snips/BH_hexbin_overmassive/%s" %(savefig_txt_input))
        if showfig:
            plt.show()
        plt.close()



#--------------------------------
# Create plots of SFR, sSFR, BH mass within our sample using BH_tree for misaligned
def _BH_sample(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                        existing_df = None,
                      #==============================================
                      # Plot options:
                        plot_yaxis = ['bhmass', 'window'],     # ['sfr', 'ssfr', 'bhmass', 'window']
                          add_seed_limit    = 1*10**6,       # [ False / value ] seed mass = 1.48*10**5, quality cut made at 5x
                          add_observational = False,                # not working
                          add_bulge_ratios  = True,
                          add_histograms    = True, 
                          add_title         = False,
                      # Sample options
                        apply_at_start = True,      # True = first snip, False = takes mean over window
                      # Sample refinement
                        run_refinement = False,
                          #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                          # basic properties
                          min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                          min_bhmass   = None,      max_bhmass   = None,
                          min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                          min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                          # Mergers, looked for within range considered +/- halfwindow
                          use_merger_criteria = False,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = 'better_format',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
        
    
    
    #===================================================================================================
    if existing_df is not None:
        df = existing_df
        df['BH mass'] = df['BH mass start']
    else:
        # Go through sample
        stelmass_plot  = []
        bhmass_plot    = []
        sfr_plot       = []
        ssfr_plot      = []
        kappa_plot     = []
        trelax_plot    = []
        duration_plot  = []
        state_plot     = []
        ID_plot        = []
        for galaxy_state in ['aligned', 'misaligned', 'counter']:
            for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
                if apply_at_start:
                    # establish starting index (0 for aligned/counter, )
                    if galaxy_state == 'misaligned':
                        starting_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                    else:
                        starting_index = 0
                
                    # Append first BH mass
                    if use_CoP_BH:
                        bhmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[starting_index])
                    else:
                        bhmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[starting_index])
                    
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[starting_index])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[starting_index])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[starting_index])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[starting_index])
                
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    if galaxy_state == 'misaligned':
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
            
                else:
                    # Append means
                    if use_CoP_BH:
                        bhmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])))
                    else:
                        bhmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])))
                    
                    stelmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])))
                    sfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])))
                    ssfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])))
                    kappa_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])))
                
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    if galaxy_state == 'misaligned':
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                    
            
        # Collect data into dataframe
        df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'BH mass': bhmass_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
        #print('\nSample after processing:\n   aligned: %s\n   misaligned: %s\n   counter: %s' %(len(df.loc[(df['State'] == 'aligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'misaligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'counter')]['GalaxyIDs'])))
    
    
    
    if 'bhmass' in plot_yaxis:
        #---------------------------  
        # Figure initialising
        
        #color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
        color_dict = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
        
        if add_histograms:
            
            fig = plt.figure(figsize=(10/3, 10/3))
            gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)
            # Create the Axes.
            axs = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=axs)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=axs)
    
            #---------------------
            ### Plot histograms
            for state_i in color_dict.keys():

                df_state = df.loc[df['State'] == state_i]
                
                ax_histx.hist(np.log10(df_state['stelmass']), bins=np.arange(9.25, 11.5+0.001, 0.25), log=True, facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=1)
                ax_histx.hist(np.log10(df_state['stelmass']), bins=np.arange(9.25, 11.5+0.001, 0.25), log=True, facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
                ax_histy.hist(np.log10(df_state['BH mass']), bins=np.arange(5, 10.1, 0.25), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=0.8)
                ax_histy.hist(np.log10(df_state['BH mass']), bins=np.arange(5, 10.1, 0.25), log=True, orientation='horizontal', facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
                
                
                #-------------
                # Formatting
                ax_histy.set_xlabel('Count')
                ax_histx.set_ylabel('Count')
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)
                ax_histx.set_yticks([1, 10, 100, 1000])
                ax_histx.set_yticklabels(['', '$10^1$', '', '$10^3$'])
                ax_histy.set_xticks([1, 10, 100, 1000])
                ax_histy.set_xticklabels(['', '$10^1$', '', '$10^3$'])
        else:
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        

        #--------------
        # scatter
        axs.scatter(np.log10(df['stelmass']), np.log10(df['BH mass']), s=6, c=[color_dict[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.5, zorder=-2)
        
    
        #-----------
        # Annotations
        if add_seed_limit:
            #axs.axhline(add_seed_limit, c='k', lw=0.5, ls='-', zorder=999, alpha=0.7)
            axs.axhspan(ymin=2,ymax=np.log10(add_seed_limit), facecolor='grey', zorder=999, alpha=0.3)
            axs.text(10.8, 6.05, 'sample limit', color='grey', alpha=1, fontsize=6)
        if add_bulge_ratios:
            axs.text(11.5, 9.58, '$M_{\mathrm{BH}}/M_{*}=10^{-2}$', color='grey', fontsize=6, rotation=22, rotation_mode='anchor', horizontalalignment='right')
            axs.plot([8, 12], [6, 10], ls='--', lw=0.5, color='grey')

            axs.text(11.5, 8.58, '$10^{-3}$', color='grey', fontsize=6, rotation=22, rotation_mode='anchor', horizontalalignment='right')
            axs.plot([8, 12], [5, 9], ls='--', lw=0.5, color='grey')

            axs.text(11.5, 7.58, '$10^{-4}$', color='grey', fontsize=6, rotation=22, rotation_mode='anchor', horizontalalignment='right')
            axs.plot([8, 12], [4, 8], ls='--', lw=0.5, color='grey')
        
    
    
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9.25, 11.5)
        axs.set_ylim(5, 10)
        #axs.set_yscale('log')
        axs.set_xlabel(r'log$_{10}$ $M_{*}(2r_{50})$ [M$_{\odot}$]')
        #axs.set_ylabel('BH mass [Msun]')
        axs.set_ylabel(r'log$_{10}$ $M_{\mathrm{BH}}$ [M$_{\odot}]$')
        #axs.minorticks_on()
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['aligned'])
        
        legend_labels.append('relaxing into stable')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
        
        legend_labels.append('counter-rotating')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
        #-----------
        ### Annotate
        if add_title:
            if add_histograms:
                ax_histx.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == None else plot_annotate), size=7, loc='left', pad=3)
            else:
                axs.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == None else plot_annotate), size=7, loc='left', pad=3)
            
            
        
    
        #-----------
        # other
        plt.tight_layout()

        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt) + ('hist' if add_histograms else '')
        
            plt.savefig("%s/BH_sample_analysis/%sstelmass_bhmass_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_bhmass_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()   
    if 'sfr' in plot_yaxis:
        #------------------------
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
        color_dict = {'aligned':'darkgrey', 'misaligned':'red', 'counter':'blue'}
    
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
        legend_colors.append(color_dict['aligned'])
        legend_labels.append('misaligned $[30^{\circ}-150^{\circ}]$')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
        legend_labels.append('counter')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        
        if add_title:
            axs.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == None else plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        # other
        plt.tight_layout()
    
        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/BH_sample_analysis/%sstelmass_sfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_sfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if 'ssfr' in plot_yaxis:
        #------------------------
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        color_dict = {'aligned':'darkgrey', 'misaligned':'red', 'counter':'blue'}
    
        plt.scatter(np.log10(df['stelmass']), df['sSFR'], s=0.1, c=[color_dict[i] for i in df['State']], edgecolor=[color_dict[i] for i in df['State']], marker='.', alpha=0.8)
    
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9, 11.5)
        axs.set_ylim(3*10**-13, 3*10**-9)
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
        legend_colors.append(color_dict['aligned'])
        legend_labels.append('misaligned $[30^{\circ}-150^{\circ}]$')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
        legend_labels.append('counter')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        

        if add_title:
            axs.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == None else plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        # other
        plt.tight_layout()
    
        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/BH_sample_analysis/%sstelmass_ssfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_ssfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if 'window' in plot_yaxis:
        
        #-------------
        ### Plotting
        fig, (ax_trelax, ax_mis, ax_cnt) = plt.subplots(nrows=3, ncols=1, figsize=[10/3, 10/3], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        xaxis_max = 8       #[Gyr]
        bin_width = 0.05
        color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
        
        #------------
        # Histograms
        # misaligned
        df_mis = df.loc[df['State'] == 'misaligned']
        ax_mis.hist(df_mis['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='none', facecolor=color_dict['misaligned'], alpha=0.1)
        bin_count, _, _ = ax_mis.hist(df_mis['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor=color_dict['misaligned'], lw=0.4, facecolor='none', alpha=1.0)

        # Add poisson errors to each bin (sqrt N)
        #hist_n, _ = np.histogram(df_mis['window'], bins=np.arange(0, xaxis_max+0.0001, 0.bin_width), range=(0, xaxis_max+0.01))
        #ax_mis.errorbar(np.arange(0.1/2, xaxis_max+0.5*bin_width, bin_width), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor=color_dict['misaligned'], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        
        # counter
        df_cnt = df.loc[df['State'] == 'counter']
        ax_cnt.hist(df_cnt['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='none', facecolor=color_dict['counter'], alpha=0.1)
        bin_count, _, _ = ax_cnt.hist(df_cnt['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor=color_dict['counter'], lw=0.4, facecolor='none', alpha=1.0)

        # Add poisson errors to each bin (sqrt N)
        #hist_n, _ = np.histogram(df_cnt['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), range=(0, xaxis_max+bin_width))
        #ax_cnt.errorbar(np.arange(0.1/2, xaxis_max+0.5*bin_width, bin_width), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor=color_dict['misaligned'], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        
        # trelax (misaligned)
        ax_trelax.hist(trelax_plot, bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='none', facecolor='r', alpha=0.1)
        bin_count, _, _ = ax_trelax.hist(trelax_plot, bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='r', lw=0.4, facecolor='none', alpha=1.0)
        
        print('For guide for later analysis:')
        print('Total misaligned: %s' %len(trelax_plot))
        print('Number of trelax >0.3  Gyr: %s' %len([i for i in trelax_plot if i > 0.3]))
        print('Number of trelax >0.4  Gyr: %s' %len([i for i in trelax_plot if i > 0.4]))
        print('Number of trelax >0.5  Gyr: %s' %len([i for i in trelax_plot if i > 0.5]))
        print('Number of trelax >0.6  Gyr: %s' %len([i for i in trelax_plot if i > 0.6]))
        print('Number of trelax >0.7  Gyr: %s' %len([i for i in trelax_plot if i > 0.7]))
        print('Number of trelax >0.8  Gyr: %s' %len([i for i in trelax_plot if i > 0.8]))
        print('Number of trelax >0.9  Gyr: %s' %len([i for i in trelax_plot if i > 0.9]))
        print('Number of trelax >1.0  Gyr: %s' %len([i for i in trelax_plot if i > 1.0]))
        
        #-----------
        ### General formatting
        # Axis labels
        ax_mis.set_xlim(0, xaxis_max)
        ax_mis.set_xticks(np.arange(0, xaxis_max+0.01, step=1))
        ax_cnt.set_xlabel('Duration [Gyr]')
        ax_mis.set_ylim(0.7, 500)
        ax_mis.set_yscale('log')
        ax_mis.set_ylabel('Number in sample')
    
    
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('misaligned (trelax)')
        legend_colors.append('r')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        ax_trelax.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)

        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('misaligned (window)')
        legend_colors.append('orangered')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        ax_mis.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('counter-rotating (window)')
        legend_colors.append('dodgerblue')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        ax_cnt.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    
        #-----------
        ### title
        if plot_annotate:
            axs_mis.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        #plt.tight_layout()
        
        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/BH_sample_analysis/%swindow_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%swindow_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
        
        
#--------------------------------
# Create stacked plots of BH growth over time, misaligned systems are not necessarily relaxed yet
def _BH_stacked_evolution(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            min_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                            add_stacked_median = True,
                            #use_random_sample  = 50,        # Number of aligned galaxies to randomly select (within limits set above)
                              target_stelmass       = False,          # [ 10** Msun / False ]
                                target_stelmass_err = 0.1,      # [ Dex ]
                              target_bhmass         = 6.5,              # [ 10**[] Msun / False ]
                                target_bhmass_err   = 0.1,        # [ Dex ] e.g. log(10**7) = 7 ± 0.1
                              
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    bhmass_plot    = []
    bhmass_start_plot = []
    lookbacktime_plot = []
    diff_co        = []     # arrays to collect an estimate of temporal separation
    diff_mis       = []
    diff_cnt       = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    
    stats_lines = {'aligned': {'time': [],
                               'mass': []},
                   'misaligned': {'time': [],
                                  'mass': []},
                   'counter': {'time': [],
                               'mass': []}}
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < min_window_size:
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                if use_CoP_BH:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[check_index]
                else:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[check_index]
                stel_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[check_index]
                time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (BH_check >= (10**(target_bhmass - target_bhmass_err) if target_bhmass else 0)) & (BH_check <= (10**(target_bhmass + target_bhmass_err) if target_bhmass else 10*15)) & (stel_check >= (10**(target_stelmass - target_stelmass_err) if target_stelmass else 0)) & (stel_check <= (10**(target_stelmass + target_stelmass_err) if target_stelmass else 10**15)) & (time_check >= min_window_size):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    #index_stop  = np.where(duration_array > min_window_size)[0][0] + 1
                    
                    
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    lookbacktime_plot.append(time_axis)
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:]
                        bhmass_plot.append(mass_axis)
                        bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start])
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:]
                        bhmass_plot.append(mass_axis)
                        bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start])
                    
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
            
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    
                    trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
            
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    
                    # Stats
                    diff_mis.append(time_axis[1])
                    stats_lines['misaligned']['time'].extend(time_axis)
                    stats_lines['misaligned']['mass'].extend(mass_axis)
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                
                if use_CoP_BH:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])
                else:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])
                stel_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])
                
                mask_check = np.where((BH_check >= (10**(target_bhmass - target_bhmass_err) if target_bhmass else 0)) & (BH_check <= (10**(target_bhmass + target_bhmass_err) if target_bhmass else 10*15)) & (stel_check >= (10**(target_stelmass - target_stelmass_err) if target_stelmass else 0)) & (stel_check <= (10**(target_stelmass + target_stelmass_err) if target_stelmass else 10**15)))[0]
                if len(mask_check) > 0:
                    # array of indexes that will meet the bhmass, stelmass, and min_window_size criteria:
                    check_index_array = []
                    for index_i in mask_check:
                        time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                        if time_check > min_window_size:
                            check_index_array.append(index_i)
                    
                    # If there exists at least one valid entry, pick random to append min_window_max entries to
                    if len(check_index_array) > 0:
                        # Pick random starting point
                        index_start = random.choice(check_index_array)
                        duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        #index_stop  = np.where(duration_array > min_window_size)[0][0] + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        lookbacktime_plot.append(time_axis)
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:]
                            bhmass_plot.append(mass_axis)
                            bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start])
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:]
                            bhmass_plot.append(mass_axis)
                            bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start])
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
            
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
            
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                        
                        # Stats
                        if galaxy_state == 'aligned':
                            diff_co.append(time_axis[1])
                            stats_lines['aligned']['time'].extend(time_axis)
                            stats_lines['aligned']['mass'].extend(mass_axis)
                        if galaxy_state == 'counter':
                            diff_cnt.append(time_axis[1])
                            stats_lines['counter']['time'].extend(time_axis)
                            stats_lines['counter']['mass'].extend(mass_axis)
                    

            
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'BH mass': bhmass_plot, 'Time axis': lookbacktime_plot, 'BH mass start': bhmass_start_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target BH_mass median of 10**%s:' %target_bhmass)
    print('  aligned:     %s\t\t%.4f (first snipshot, use below instead)' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start']))))
    print('  misaligned:  %s\t\t%.4f' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start']))))
    print('  counter:     %s\t\t%.4f' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start']))))
    print(' ')
    
    #------------------------
    # Figure initialising
    fig, (ax_combi, ax_co, ax_mis, ax_cnt) = plt.subplots(nrows=4, ncols=1, figsize=[10/3, 5.5], sharex=True, sharey=True) 
    plt.subplots_adjust(wspace=0.2, hspace=0, bottom=0.13)
    
    color_dict = {'aligned':'k', 'misaligned':'orangered', 'counter':'dodgerblue'}
    
    #------------
    # Plot lines
    for index, row in df_co.iterrows():        
        ax_co.plot(row['Time axis'], np.log10(row['BH mass']), c='lightgrey', lw=0.3, alpha=0.5, zorder=-1)
        #ax_co.plot(row['Time axis'], np.log10(row['BH mass']), 'k^', alpha=0.5, zorder=-1)
        
    for index, row in df_mis.iterrows():        
        ax_mis.plot(row['Time axis'], np.log10(row['BH mass']), c='lightsalmon', lw=0.3, alpha=0.3, zorder=-1)
        #ax_mis.plot(row['Time axis'], np.log10(row['BH mass']), 'k^', alpha=0.3, zorder=-1)
        
    for index, row in df_cnt.iterrows():        
        ax_cnt.plot(row['Time axis'], np.log10(row['BH mass']), c='lightsteelblue', lw=0.3, alpha=0.5, zorder=-1)
        #ax_cnt.plot(row['Time axis'], np.log10(row['BH mass']), 'k^', alpha=0.5, zorder=-1)
    
    #------------
    # Plot median and percentiles
    if add_stacked_median:
        
        stats_growth = {'aligned': {'median': [],
                                    'upper': [],
                                    'lower': []},
                        'misaligned': {'median': [],
                                       'upper': [],
                                       'lower': []},
                        'counter': {'median': [],
                                    'upper': [],
                                    'lower': []}}
        
        use_percentiles = 16        # 1 sigma
        bins = np.arange(-(0.125/2), min_window_size+(0.125/2)+0.01, 0.125)
        plot_bins = np.arange(0, min_window_size+0.01, 0.125)
        
        if len(diff_co) != 0:
            line_color = color_dict['aligned']
            #bins = np.arange((-1*np.median(diff_co))+0.00001, min_window_size+0.05, np.median(diff_co))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['aligned']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['aligned']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['aligned']['time'])[mask]
                current_mass   = np.array(stats_lines['aligned']['mass'])[mask]
                            
                median_array.append(np.percentile(current_mass, 50))
                median_upper.append(np.percentile(current_mass, 100-use_percentiles))
                median_lower.append(np.percentile(current_mass, use_percentiles))
            
            stats_growth['aligned']['median'] = np.log10(median_array)
            stats_growth['aligned']['upper'] = np.log10(median_upper)
            stats_growth['aligned']['lower'] = np.log10(median_lower)
                   
            #----------
            # plot
            ax_co.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100)
            ax_co.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100)
            ax_co.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100)
            
            ax_combi.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
        if len(diff_mis) != 0:
            line_color = color_dict['misaligned']
            #bins = np.arange((-1*np.median(diff_co))+0.00001, min_window_size+0.05, np.median(diff_mis))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['misaligned']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['misaligned']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['misaligned']['time'])[mask]
                current_mass   = np.array(stats_lines['misaligned']['mass'])[mask]
                            
                median_array.append(np.percentile(current_mass, 50))
                median_upper.append(np.percentile(current_mass, 100-use_percentiles))
                median_lower.append(np.percentile(current_mass, use_percentiles))
            
            stats_growth['misaligned']['median'] = np.log10(median_array)
            stats_growth['misaligned']['upper'] = np.log10(median_upper)
            stats_growth['misaligned']['lower'] = np.log10(median_lower)
                 
            #----------
            # plot
            ax_mis.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100)
            ax_mis.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100)
            ax_mis.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100)
            
            ax_combi.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
        if len(diff_cnt) != 0:
            line_color = color_dict['counter']
            #bins = np.arange((-1*np.median(diff_co))+0.00001, min_window_size+0.05, np.median(diff_cnt))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter']['time']), bins=bins)
                
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter']['time'])[mask]
                current_mass   = np.array(stats_lines['counter']['mass'])[mask]
                            
                median_array.append(np.percentile(current_mass, 50))
                median_upper.append(np.percentile(current_mass, 100-use_percentiles))
                median_lower.append(np.percentile(current_mass, use_percentiles))
            
            stats_growth['counter']['median'] = np.log10(median_array)
            stats_growth['counter']['upper'] = np.log10(median_upper)
            stats_growth['counter']['lower'] = np.log10(median_lower)
                       
            #----------
            # plot
            ax_cnt.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100)
            ax_cnt.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100)
            ax_cnt.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100)
            
            ax_combi.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
    
        print('Medians with 0.125 bins before/after/delta with min %s Gyr:' %min_window_size)
        print('  aligned:     %.4f\t\t%.4f\t\tdelta %.4f dex' %(stats_growth['aligned']['median'][0], stats_growth['aligned']['median'][-1], (stats_growth['aligned']['median'][-1]-stats_growth['aligned']['median'][0])))
        print('  misaligned:  %.4f\t\t%.4f\t\tdelta %.4f dex' %(stats_growth['misaligned']['median'][0], stats_growth['misaligned']['median'][-1], (stats_growth['misaligned']['median'][-1]-stats_growth['misaligned']['median'][0])))
        print('  counter:     %.4f\t\t%.4f\t\tdelta %.4f dex' %(stats_growth['counter']['median'][0], stats_growth['counter']['median'][-1], (stats_growth['counter']['median'][-1]-stats_growth['counter']['median'][0])))
    
    #-----------
    ### other
    #plt.tight_layout()
        
    #-----------
    ### General formatting
    # Axis labels
    ax_co.set_xlim(-0.05, min_window_size+0.05)
    ax_co.set_xticks(np.arange(0, min_window_size+0.05, 0.1))
    ax_cnt.set_xlabel('Time [Gyr]')
    ax_co.set_ylim(target_bhmass-(1.5*target_bhmass_err), np.log10(median_upper)[-1]+0.2)
    fig.supylabel(r'log$_{10}$ most massive $M_{\mathrm{BH}}(<r_{50}$) [M$_{\odot}]$', fontsize=9)
    ax_co.minorticks_on()
    ax_co.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax_co.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### Annotation
    ax_mis.axvline(x=0, ymin=0, ymax=1, color='grey', zorder=999, lw=0.7, ls='--', alpha=0.7)
    ax_mis.text(-0.012, target_bhmass+0.1, 'last stable', color='grey', alpha=0.7, fontsize=6, rotation=90, rotation_mode='anchor')
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_colors.append('k')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_labels.append('misaligned $[30^{\circ}-150^{\circ}]$')
    legend_colors.append('orangered')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_labels.append('counter-rotating')
    legend_colors.append('dodgerblue')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_combi.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_colors.append('k')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_co.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)

    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('misaligned $[30^{\circ}-150^{\circ}]$')
    legend_colors.append('orangered')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_mis.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('counter-rotating')
    legend_colors.append('dodgerblue')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_cnt.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    ### title
    plot_annotate = '$>%.2f$ Gyr window, target BHmass = %.2f±%.2f'%(min_window_size, target_bhmass, target_bhmass_err) + (plot_annotate if plot_annotate else '')
    ax_combi.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    
    #-----------
    ### Savefig       
    
    
    metadata_plot = {'Title': 'medians start/end/delta\nali: %.4f %.4f %.4f\nmis: %.4f %.4f %.4f\ncnt: %.4f %.4f %.4f' %(stats_growth['aligned']['median'][0], stats_growth['aligned']['median'][-1], (stats_growth['aligned']['median'][-1]-stats_growth['aligned']['median'][0]), stats_growth['misaligned']['median'][0], stats_growth['misaligned']['median'][-1], (stats_growth['misaligned']['median'][-1]-stats_growth['misaligned']['median'][0]), stats_growth['counter']['median'][0], stats_growth['counter']['median'][-1], (stats_growth['counter']['median'][-1]-stats_growth['counter']['median'][0]))}
    
                
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/BH_stacked_evolution/%sstacked_BH%s_M%s_t%s_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', target_bhmass, target_stelmass, min_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_stacked_evolution/%swstacked_BH%s_M%s_t%s_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', target_bhmass, target_stelmass, min_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
    

#--------------------------------
# x-y of BH mass at start vs delta mass, coloured for counter/misaligned/aligned -> a few of these for 500 mya (trim longer aligend/counter ones), 0.75 gya, 1 gyr. Will pick random min_window_segment per aligned/counter entry
def _BH_deltamass_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                                only_output_df = False,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            add_histograms    = False,   
                            add_growth_ratios = True,
                          # Morph and fgas
                            use_ETG_LTG            = False,           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                              morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                            select_high_fgas       = False,         # [ False / value ] trim based on average
                            select_low_err         = False,         # [ True / False ] trim to max error of 10 (3 sigma = 30)
                          #------------------------------------
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    bh_mdot_inst_plot = []      # particle level data
    bh_edd_plot       = []
    bh_lbol_plot      = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []     # first snip
    kappa_mean_plot = []   # average 
    kappa_snip_plot = []    # all snips
    fgassf_plot    = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    err_test       = []     # 'low' and 'high' for <10 and 10-30 error
    r50_plot       = []
    r50_sf_plot    = []
    ID_plot        = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                #check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']+1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]   
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                            bh_mdot_inst_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst'])[index_start:index_stop]))
                            bh_edd_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_edd'])[index_start:index_stop]))
                            bh_lbol_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol'])[index_start:index_stop]))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                            # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                            bh_mdot_inst_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst_alt'])[index_start:index_stop]))
                            bh_edd_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_edd_alt'])[index_start:index_stop]))
                            bh_lbol_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol_alt'])[index_start:index_stop]))
                            #bh_lbol_plot.append(np.max(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol_alt'])[index_start:index_stop]))
                            
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        
                        kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                        kappa_mean_plot.append(np.mean(kappa_array))

                        if (kappa_array > 0.4).all():
                            kappa_snip_plot.append('LTG')
                        elif (kappa_array < 0.4).all():
                            kappa_snip_plot.append('ETG')
                        else:
                            kappa_snip_plot.append('other')
                        
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                        
                        r50_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['rad'])[index_start:index_stop]))
                        r50_sf_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['rad_sf'])[index_start:index_stop]))
                        
                        angle_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start:index_stop]
                        err_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf_err'])[index_start:index_stop]
                        if np.max(np.abs(err_array - angle_array[:,None])) < 10:
                            err_test.append('low')
                        else:
                            err_test.append('high')
                            
                        
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                        bh_mdot_inst_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst'])[index_start:index_stop]))
                        bh_edd_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_edd'])[index_start:index_stop]))
                        bh_lbol_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol'])[index_start:index_stop]))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                        bh_mdot_inst_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst_alt'])[index_start:index_stop]))
                        bh_edd_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_edd_alt'])[index_start:index_stop]))
                        bh_lbol_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol_alt'])[index_start:index_stop]))
                        #bh_lbol_plot.append(np.max(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol_alt'])[index_start:index_stop]))
                        
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    # Array values
                    kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                    kappa_mean_plot.append(np.mean(kappa_array))
                    
                    if (kappa_array > 0.4).all():
                        kappa_snip_plot.append('LTG')
                    elif (kappa_array < 0.4).all():
                        kappa_snip_plot.append('ETG')
                    else:
                        kappa_snip_plot.append('other')
                        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    trelax_plot.append(math.nan)
                        
                    r50_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['rad'])[index_start:index_stop]))
                    r50_sf_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['rad_sf'])[index_start:index_stop]))
                    
                    angle_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start:index_stop]
                    err_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf_err'])[index_start:index_stop]
                    if np.max(np.abs(err_array - angle_array[:,None])) < 10:
                        err_test.append('low')
                    else:    
                        err_test.append('high')
                
                                    
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'trelax': trelax_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'BH mdot inst': bh_mdot_inst_plot, 'BH edd inst': bh_edd_plot, 'BH lbol inst': bh_lbol_plot, 'Morphology': kappa_plot, 'Kappa mean': kappa_mean_plot, 'Kappa snip': kappa_snip_plot, 'Gassf fraction': fgassf_plot, 'State': state_plot, 'window': duration_plot, 'err_test': err_test, 'r50': r50_plot, 'r50_sf': r50_sf_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    
    #-------------------
    # Select only ETG/LTG
    if use_ETG_LTG:
        if use_ETG_LTG == 'ETG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'ETG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] < 0.4)]
        if use_ETG_LTG == 'LTG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'LTG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] > 0.4)]
                
        print('Using sample of %s %ss:      %s' %(morph_average_or_snips, use_ETG_LTG, len(df_morph['GalaxyIDs'])))
        df = df_morph
        df_morph = 0
    # Select only high fgas
    if select_high_fgas:
        df_fgas = df.loc[(df['Gassf fraction'] > select_high_fgas)]
        
        print('Using sample of %s f_gas,SF:      %s' %(select_high_fgas, len(df_fgas['GalaxyIDs'])))
        df = df_fgas
        df_fgas = 0
    # Select only low uncertainty systems
    if select_low_err:
        # testing effect if using stricter 3 sigma uncertainty
        df = df.loc[(df['err_test'] == 'low')]
        
        print('Using sample of max. 10 deg error (3 sigma = 30):      %s' %(len(df['GalaxyIDs'])))
    
    
    #-------------------
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    print('Total misaligned: %s' %len(df_mis['trelax']))
    print('Number of trelax >0.3  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.3]))
    print('Number of trelax >0.4  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.4]))
    print('Number of trelax >0.5  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.5]))
    print('Number of trelax >0.6  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.6]))
    print('Number of trelax >0.7  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.7]))
    print('Number of trelax >0.8  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.8]))
    print('Number of trelax >0.9  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.9]))
    print('Number of trelax >1.0  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 1.0]))
    
    #print(' ')
    #print('Median r50_sf / r_50: ')
    #print('  aligned:       %.3f' %np.median(np.array(df_co['r50_sf']/df_co['r50'])))
    #print('  misaligned:    %.3f' %np.median(np.array(df_mis['r50_sf']/df_mis['r50'])))
    #print('  counter:       %.3f' %np.median(np.array(df_cnt['r50_sf']/df_cnt['r50'])))
    
    print(' ')
    print('Fraction of aligned / misaligned / counter with Lbol > 1043: \t%.2f\t%.2f\t%.2f' %((len([i for i in np.array(df_co['BH lbol inst']) if i > 10**43])/len(np.array(df_co['BH lbol inst']))), (len([i for i in np.array(df_mis['BH lbol inst']) if i > 10**43])/len(np.array(df_mis['BH lbol inst']))), (len([i for i in np.array(df_cnt['BH lbol inst']) if i > 10**43])/len(np.array(df_cnt['BH lbol inst'])))))
    print('Fraction of aligned / misaligned / counter with EddR > 0.01: \t%.2f\t%.2f\t%.2f' %((len([i for i in np.array(df_co['BH edd inst']) if i > 0.01])/len(np.array(df_co['BH edd inst']))), (len([i for i in np.array(df_mis['BH edd inst']) if i > 0.01])/len(np.array(df_mis['BH edd inst']))), (len([i for i in np.array(df_cnt['BH edd inst']) if i > 0.01])/len(np.array(df_cnt['BH edd inst'])))))
    
    
    
    # Plot
    if not only_output_df:
        color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
        color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
        #---------------------------  
        # Figure initialising
        if add_histograms:
        
            fig = plt.figure(figsize=(10/3, 10/3))
            gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)
            # Create the Axes.
            axs = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=axs)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=axs)

            #---------------------
            ### Plot histograms
            for state_i in color_dict.keys():

                df_state = df.loc[df['State'] == state_i]
            
                ax_histx.hist(np.log10(df_state['BH mass start']), bins=np.arange(6, 11, 0.25), log=True, facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=1)
                ax_histx.hist(np.log10(df_state['BH mass start']), bins=np.arange(6, 11, 0.25), log=True, facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
                ax_histy.hist(np.log10(df_state['BH mass delta']), bins=np.arange(0, 10.1, 0.25), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=0.8)
                ax_histy.hist(np.log10(df_state['BH mass delta']), bins=np.arange(0, 10.1, 0.25), log=True, orientation='horizontal', facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
            
            
                #-------------
                # Formatting
                ax_histy.set_xlabel('Count')
                ax_histx.set_ylabel('Count')
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)
                ax_histx.set_yticks([1, 10, 100, 1000])
                ax_histx.set_yticklabels(['', '$10^1$', '', '$10^3$'])
                ax_histy.set_xticks([1, 10, 100, 1000])
                ax_histy.set_xticklabels(['', '$10^1$', '', '$10^3$'])    
        else:
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
    
        #--------------
        # scatter
        ###axs.scatter(np.log10(df['BH mass start']), np.log10(df['BH mass delta']), s=6, c=[color_dict_scatter[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.7, zorder=-2)
        
        axs.scatter(np.log10(df['BH mass start']), np.log10(df['BH mass delta']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        
        #axs.scatter(np.log10(df['BH mass start']), np.log10(df['BH mdot inst']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        #axs.scatter(np.log10(df['BH mass start']), np.log10(df['BH lbol inst']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        #axs.scatter(np.log10(df['BH mass start']), np.log10(df['BH edd inst']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        
        
        
        #--------------
        # Contours
        levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
        for state_i in ['aligned', 'misaligned', 'counter']:
            df_i = df.loc[(df['State'] == state_i)]
        
            x = np.array(np.log10(df_i['BH mass start']))
            y = np.array(np.log10(df_i['BH mass delta']))
            
            #y = np.array(np.log10(df_i['BH mdot inst']))
            #y = np.array(np.log10(df_i['BH lbol inst']))
            #y = np.array(np.log10(df_i['BH edd inst']))

            k = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            #set zi to 0-1 scale
            zi = (zi-zi.min())/(zi.max() - zi.min())
            zi =zi.reshape(xi.shape)

            #set up plot
            origin = 'lower'
            #levels = [0.25, 0.5, 0.75]

            CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.6)
            axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
            axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
        
        
            #axs.clabel(CS, fmt='%.3f', colors='b', fontsize=8)

        
        # looking at alternative ways of expressing our findings
        """axs.set_xlim(5.95, 9)
        axs.set_ylim(38, 46)
        axs.set_xlabel(r'log$_{10}$ $M_{\mathrm{BH,initial}}$ [M$_{\odot}]$')
        #axs.set_ylabel(r'log$_{10}$ mean subgrid $\lambda_{\mathrm{Edd}}$')
        axs.set_ylabel(r'log$_{10}$ subgrid $L_{\mathrm{bol}}$ [erg/s]')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        plt.show()"""




        #--------------
        # annotation
        #axs.clabel(CS, fmt='%.3f', colors='b', fontsize=8)
        if add_growth_ratios:
            axs.text(8.9, 8.8, '$\Delta M_{\mathrm{BH}}/M_{\mathrm{BH,initial}}=1$', color='grey', alpha=1, fontsize=6, rotation=22, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [5, 10], ls='--', lw=0.5, color='grey', alpha=1, zorder=-5)

            axs.text(8.9, 7.8, '$0.1$', color='grey', alpha=1, fontsize=6, rotation=22, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [4, 9], ls='--', lw=0.5, color='grey', alpha=1, zorder=-5)
        
            axs.text(8.9, 6.8, '$0.01$', color='grey', alpha=1, fontsize=6, rotation=22, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [3, 8], ls='--', lw=0.5, color='grey', alpha=1, zorder=-5)
        
            axs.text(8.9, 5.8, '$0.001$', color='grey', alpha=1, fontsize=6, rotation=22, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [2, 7], ls='--', lw=0.5, color='grey', alpha=1, zorder=-5)


        #-----------
        ### title
        #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
        #if must_still_be_misaligned:
        #    plot_annotate = plot_annotate + '/trelax'
        #if add_histograms:
        #    ax_histx.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        #else:
        #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        plot_annotate = ''
        if use_ETG_LTG:
            plot_annotate = plot_annotate + '%s'%use_ETG_LTG
        if select_high_fgas:
            plot_annotate = plot_annotate + '%s $f_{\mathrm{gas,SF}}$'%select_high_fgas
        if select_low_err:
            plot_annotate = plot_annotate + '$<10^{\circ}$ error'
        if plot_annotate != '':
            if add_histograms:
                ax_histx.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            else:
                axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        

        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(5.95, 9)
        axs.set_ylim(2, 9)
        axs.set_xlabel(r'log$_{10}$ $M_{\mathrm{BH,initial}}$ [M$_{\odot}]$')
        axs.set_ylabel(r'log$_{10}$ $\Delta M_{\mathrm{BH}}$ [M$_{\odot}]$')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(lighten_color(color_dict['aligned'], 1))
        legend_labels.append('relaxing to stable')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(lighten_color(color_dict['misaligned'], 1))
        legend_labels.append('counter-rotating')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(lighten_color(color_dict['counter'], 1))
        axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


        #-----------
        # other
        plt.tight_layout()

        #-----------
        ### Savefig        
    
        metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt) + ('hist' if add_histograms else '')
    
            plt.savefig("%s/BH_deltamass/%sbhmass_delta_%s%s_%s%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), ('10degerr_' if select_low_err else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_deltamass/%sbhmass_delta_%s%s_%s%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), ('10degerr_' if select_low_err else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    else:
        return df
# x-y of BH mass at start vs average accretion rate (time-averaged using subgrid mass)
def _BH_mdot_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                                only_output_df = False,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            add_histograms    = False,   
                            add_growth_ratios = True,
                            window_or_snip_mdot = 'snip',       # average growth over window or time-weighted mean over each snip
                          # Morph and fgas
                            use_ETG_LTG            = False,           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                              morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                              mean_or_median         = 'mean',      # [ 'mean' / 'median' ] whether to take median value or mean
                            select_high_fgas       = False,         # [ False / value ] trim based on average
                          #------------------------------------
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []     # first snip
    kappa_mean_plot = []   # average 
    kappa_snip_plot = []    # all snips
    fgassf_plot    = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                #check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']+1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]   
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            if mean_or_median == 'mean':
                                bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                            if mean_or_median == 'median':
                                bhmdot_snip_plot.append(np.median(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            if mean_or_median == 'mean':
                                bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                            if mean_or_median == 'median':
                                bhmdot_snip_plot.append(np.median(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        
                        kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                        kappa_mean_plot.append(np.mean(kappa_array))

                        if (kappa_array > 0.4).all():
                            kappa_snip_plot.append('LTG')
                        elif (kappa_array < 0.4).all():
                            kappa_snip_plot.append('ETG')
                        else:
                            kappa_snip_plot.append('other')
                        
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        if mean_or_median == 'mean':
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        if mean_or_median == 'median':
                            bhmdot_snip_plot.append(np.median(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        if mean_or_median == 'mean':
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        if mean_or_median == 'median':
                            bhmdot_snip_plot.append(np.median(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    # Array values
                    kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                    kappa_mean_plot.append(np.mean(kappa_array))
                    
                    if (kappa_array > 0.4).all():
                        kappa_snip_plot.append('LTG')
                    elif (kappa_array < 0.4).all():
                        kappa_snip_plot.append('ETG')
                    else:
                        kappa_snip_plot.append('other')
                        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    trelax_plot.append(math.nan)
                
                                    
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'trelax': trelax_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Kappa mean': kappa_mean_plot, 'Kappa snip': kappa_snip_plot, 'Gassf fraction': fgassf_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    #-------------------
    # Select only ETG/LTG
    if use_ETG_LTG:
        if use_ETG_LTG == 'ETG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'ETG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] < 0.4)]
        if use_ETG_LTG == 'LTG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'LTG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] > 0.4)]
                
        print('Using sample of %s %ss:      %s' %(morph_average_or_snips, use_ETG_LTG, len(df_morph['GalaxyIDs'])))
        df = df_morph
        df_morph = 0
    
    # Select only high fgas
    if select_high_fgas:
        df_fgas = df.loc[(df['Gassf fraction'] > select_high_fgas)]
        
        print('Using sample of %s f_gas,SF:      %s' %(select_high_fgas, len(df_fgas['GalaxyIDs'])))
        df = df_fgas
        df_fgas = 0
    
    #-------------------
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    print('Total misaligned: %s' %len(df_mis['trelax']))
    print('Number of trelax >0.3  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.3]))
    print('Number of trelax >0.4  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.4]))
    print('Number of trelax >0.5  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.5]))
    print('Number of trelax >0.6  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.6]))
    print('Number of trelax >0.7  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.7]))
    print('Number of trelax >0.8  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.8]))
    print('Number of trelax >0.9  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 0.9]))
    print('Number of trelax >1.0  Gyr: %s' %len([i for i in df_mis['trelax'] if i > 1.0]))
    
    # Plot
    if not only_output_df:

        color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
        color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
        #---------------------------  
        # Figure initialising
        if add_histograms:
        
            fig = plt.figure(figsize=(10/3, 10/3))
            gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)
            # Create the Axes.
            axs = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=axs)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=axs)

            #---------------------
            ### Plot histograms
            for state_i in color_dict.keys():

                df_state = df.loc[df['State'] == state_i]
            
                ax_histx.hist(np.log10(df_state['BH mass start']), bins=np.arange(6, 11, 0.25), log=True, facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=1)
                ax_histx.hist(np.log10(df_state['BH mass start']), bins=np.arange(6, 11, 0.25), log=True, facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
                ax_histy.hist(np.log10(df_state['BH mdot %s'%window_or_snip_mdot]), bins=np.arange(-7, 0, 0.25), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=0.8)
                ax_histy.hist(np.log10(df_state['BH mdot %s'%window_or_snip_mdot]), bins=np.arange(-7, 0, 0.25), log=True, orientation='horizontal', facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
            
            
                #-------------
                # Formatting
                ax_histy.set_xlabel('Count')
                ax_histx.set_ylabel('Count')
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)
                ax_histx.set_yticks([1, 10, 100, 1000])
                ax_histx.set_yticklabels(['', '$10^1$', '', '$10^3$'])
                ax_histy.set_xticks([1, 10, 100, 1000])
                ax_histy.set_xticklabels(['', '$10^1$', '', '$10^3$'])    
        else:
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
    
        #--------------
        # scatter
        axs.scatter(np.log10(df['BH mass start']), np.log10(df['BH mdot %s'%window_or_snip_mdot]), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)

    
        #--------------
        # Contours
        levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
        for state_i in ['aligned', 'misaligned', 'counter']:
            df_i = df.loc[(df['State'] == state_i)]
        
            x = np.array(np.log10(df_i['BH mass start']))
            y = np.array(np.log10(df_i['BH mdot %s'%window_or_snip_mdot]))

            k = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            #set zi to 0-1 scale
            zi = (zi-zi.min())/(zi.max() - zi.min())
            zi =zi.reshape(xi.shape)

            #set up plot
            origin = 'lower'
            #levels = [0.25, 0.5, 0.75]

            CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.9), lighten_color(color_dict['%s' %state_i], 0.6),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.6)
            axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.9),), origin=origin, zorder=-8, alpha=0.1)
            axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.6),), origin=origin, zorder=-8, alpha=0.1)
        
        
            #axs.clabel(CS, fmt='%.3f', colors='b', fontsize=8)

        #--------------
        # annotation
        #axs.clabel(CS, fmt='%.3f', colors='b', fontsize=8)
        if add_growth_ratios:
            axs.text(8.7, np.log10(10**8.6/(target_window_size*10**9)), '$\Delta M_{\mathrm{BH}}/M_{\mathrm{BH,initial}}=1$', color='grey', alpha=1.0, fontsize=6, rotation=24, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [np.log10(10**5/(target_window_size*10**9)), np.log10(10**10/(target_window_size*10**9))], ls='--', lw=0.5, color='grey', alpha=1.0, zorder=-5)

            axs.text(8.9, np.log10(10**7.8/(target_window_size*10**9)), '$0.1$', color='grey', alpha=1.0, fontsize=6, rotation=24, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [np.log10(10**4/(target_window_size*10**9)), np.log10(10**9/(target_window_size*10**9))], ls='--', lw=0.5, color='grey', alpha=1.0, zorder=-5)
        
            axs.text(8.9, np.log10(10**6.8/(target_window_size*10**9)), '$0.01$', color='grey', alpha=1.0, fontsize=6, rotation=24, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [np.log10(10**3/(target_window_size*10**9)), np.log10(10**8/(target_window_size*10**9))], ls='--', lw=0.5, color='grey', alpha=1.0, zorder=-5)
        
            axs.text(8.9, np.log10(10**5.8/(target_window_size*10**9)), '$0.001$', color='grey', alpha=1.0, fontsize=6, rotation=24, rotation_mode='anchor', horizontalalignment='right', verticalalignment='top')
            axs.plot([5, 10], [np.log10(10**2/(target_window_size*10**9)), np.log10(10**7/(target_window_size*10**9))], ls='--', lw=0.5, color='grey', alpha=1.0, zorder=-5)


        #-----------
        ### title
        #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
        #if must_still_be_misaligned:
        #    plot_annotate = plot_annotate + '/trelax'
        #if add_histograms:
        #    ax_histx.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        #else:
        #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        plot_annotate = ''
        if use_ETG_LTG:
            plot_annotate = plot_annotate + '%s'%use_ETG_LTG
        if select_high_fgas:
            plot_annotate = plot_annotate + '%s $f_{\mathrm{gas,SF}}$'%select_high_fgas
        if mean_or_median == 'median':
            plot_annotate = plot_annotate + ' median value'
        if plot_annotate != '':
            if add_histograms:
                ax_histx.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            else:
                axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(5.95, 9)
        axs.set_ylim(-6.5, 0)
        axs.set_xlabel(r'log$_{10}$ $M_{\mathrm{BH,initial}}$ [M$_{\odot}]$')
        #axs.set_ylabel('BH mass [Msun]')
        #axs.set_ylabel(r'log$_{10}$ $\Delta M_{\mathrm{BH}}$ [M$_{\odot}]$')
        axs.set_ylabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{BH}} \rangle _{\mathrm{%s\:Gyr}}$ [M$_{\odot}$ yr$^{-1}$]' %target_window_size)
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(lighten_color(color_dict['aligned'], 1))
        legend_labels.append('relaxing into stable')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(lighten_color(color_dict['misaligned'], 1))
        legend_labels.append('counter-rotating')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(lighten_color(color_dict['counter'], 1))
        axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


        #-----------
        # other
        plt.tight_layout()

        #-----------
        ### Savefig        
    
        metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt) + ('hist' if add_histograms else '')
    
            plt.savefig("%s/BH_deltamass/%sbhmass_mdot_%s%s_%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_deltamass/%sbhmass_mdot_%s%s_%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    else:
        return df
         
    
#--------------------------------
# histogram of fractional BH growth delta M_BH / M_BH, as a measure of generally enhanced/reduced growth
def _BH_deltamassmass_hist_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            use_ETG_LTG            = False,           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                              morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                            select_high_fgas       = False,         # [ False / value ] trim based on average
                          #------------------------------------
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    kappa_mean_plot = []   # average 
    kappa_snip_plot = []    # all snips
    fgassf_plot    = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    angle_plot     = []
    ID_plot        = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        angle_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start])
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        
                        kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                        kappa_mean_plot.append(np.mean(kappa_array))
                        if (kappa_array > 0.4).all():
                            kappa_snip_plot.append('LTG')
                        elif (kappa_array < 0.4).all():
                            kappa_snip_plot.append('ETG')
                        else:
                            kappa_snip_plot.append('other')
            
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    angle_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start])
                    fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                    kappa_mean_plot.append(np.mean(kappa_array))

                    if (kappa_array > 0.4).all():
                        kappa_snip_plot.append('LTG')
                    elif (kappa_array < 0.4).all():
                        kappa_snip_plot.append('ETG')
                    else:
                        kappa_snip_plot.append('other')
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    trelax_plot.append(math.nan)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'trelax': trelax_plot, 'Kappa snip': kappa_snip_plot, 'Gassf fraction': fgassf_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'Angle': angle_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    #-------------------
    # Select only ETG/LTG
    if use_ETG_LTG:
        if use_ETG_LTG == 'ETG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'ETG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] < 0.4)]
        if use_ETG_LTG == 'LTG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'LTG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] > 0.4)]
                
        print('Using sample of %s %ss:      %s' %(morph_average_or_snips, use_ETG_LTG, len(df_morph['GalaxyIDs'])))
        df = df_morph
        df_morph = 0
    
    # Select only high fgas
    if select_high_fgas:
        df_fgas = df.loc[(df['Gassf fraction'] > select_high_fgas)]
        
        print('Using sample of %s f_gas,SF:      %s' %(select_high_fgas, len(df_fgas['GalaxyIDs'])))
        df = df_fgas
        df_fgas = 0
    
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    print('-------')
    print('Median growth:       [ delta / M_initial ]')
    print('   aligned:       %.4f' %(np.median(df_co['BH fraction growth'])))
    print('   misaligned:    %.4f' %(np.median(df_mis['BH fraction growth'])))
    print('   counter:       %.4f' %(np.median(df_cnt['BH fraction growth'])))
    
    
    
    #---------------
    # KS test
    print('-------------')
    res = stats.ks_2samp(df_co['BH fraction growth'], df_mis['BH fraction growth'])
    print('KS-test:     aligned - misaligned')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_co['BH fraction growth'], df_cnt['BH fraction growth'])
    print('KS-test:     aligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_mis['BH fraction growth'], df_cnt['BH fraction growth'])
    print('KS-test:     misaligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    
    #---------------
    # KS test ETG vs LTG
    if not use_ETG_LTG:
        if morph_average_or_snips == 'snips':
            df_co_ETG  = df.loc[(df['Kappa snip'] == 'ETG') & (df['State'] == 'aligned')]
            df_mis_ETG = df.loc[(df['Kappa snip'] == 'ETG') & (df['State'] == 'misaligned')]
            df_cnt_ETG = df.loc[(df['Kappa snip'] == 'ETG') & (df['State'] == 'counter')]

            df_co_LTG  = df.loc[(df['Kappa snip'] == 'LTG') & (df['State'] == 'aligned')]
            df_mis_LTG = df.loc[(df['Kappa snip'] == 'LTG') & (df['State'] == 'misaligned')]
            df_cnt_LTG = df.loc[(df['Kappa snip'] == 'LTG') & (df['State'] == 'counter')]
        elif morph_average_or_snips == 'average':
            df_co_ETG  = df.loc[(df['Kappa mean'] < 0.4) & (df['State'] == 'aligned')]
            df_mis_ETG = df.loc[(df['Kappa mean'] < 0.4) & (df['State'] == 'misaligned')]
            df_cnt_ETG = df.loc[(df['Kappa mean'] < 0.4) & (df['State'] == 'counter')]

            df_co_LTG  = df.loc[(df['Kappa mean'] > 0.4) & (df['State'] == 'aligned')]
            df_mis_LTG = df.loc[(df['Kappa mean'] > 0.4) & (df['State'] == 'misaligned')]
            df_cnt_LTG = df.loc[(df['Kappa mean'] > 0.4) & (df['State'] == 'counter')]
        
        print('\n-------------')
        print('KS morphology test for target window: %s' %target_window_size)
        res = stats.ks_2samp(df_co_LTG['BH fraction growth'], df_co_ETG['BH fraction growth'])
        print('KS-test:     aligned: ETG vs LTG')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co_ETG.index) + len(df_co_LTG.index))/(len(df_co_ETG.index)*len(df_co_LTG.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis_LTG['BH fraction growth'], df_mis_ETG['BH fraction growth'])
        print('KS-test:     misaligned: ETG vs LTG')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis_ETG.index) + len(df_mis_LTG.index))/(len(df_mis_ETG.index)*len(df_mis_LTG.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_cnt_ETG['BH fraction growth'], df_cnt_LTG['BH fraction growth'])
        print('KS-test:     counter: ETG vs LTG')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_cnt_ETG.index) + len(df_cnt_LTG.index))/(len(df_cnt_ETG.index)*len(df_cnt_LTG.index))))))
        print('   p-value: %s' %res.pvalue)
        
        df_co_ETG  = 0
        df_mis_ETG = 0
        df_cnt_ETG = 0
        df_co_LTG  = 0
        df_mis_LTG = 0
        df_cnt_LTG = 0
    
    
    
    
    """fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    bins = np.arange(0, 181, 10)
    axs.hist(df_co['Angle'], bins=bins, weights=np.ones(len(df_co['BH fraction growth']))/len(df_co['BH fraction growth']), linewidth=1, facecolor='none', edgecolor='k', alpha=0.8)
    axs.hist(df_mis['Angle'], bins=bins, weights=np.ones(len(df_mis['BH fraction growth']))/len(df_mis['BH fraction growth']), linewidth=1, facecolor='none', edgecolor='r', alpha=0.8)
    axs.hist(df_cnt['Angle'], bins=bins, weights=np.ones(len(df_cnt['BH fraction growth']))/len(df_cnt['BH fraction growth']), linewidth=1, facecolor='none', edgecolor='b', alpha=0.8)
    axs.set_xlabel('First angle in array')
    axs.set_xlim(0, 180)
    plt.show()
    plt.close()"""
    
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot histograms
    add_plot_errorbars = True
    for state_i in color_dict.keys():
        
        df_state = df.loc[df['State'] == state_i]
        
        bins = np.arange(-5, 1.51, 0.25)
        
        axs.hist(np.log10(df_state['BH fraction growth']), log=True, bins=bins, weights=np.ones(len(df_state['BH fraction growth']))/len(df_state['BH fraction growth']), facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=0.8)
        if state_i == 'aligned':
            axs.hist(np.log10(df_state['BH fraction growth']), log=True, bins=bins, weights=np.ones(len(df_state['BH fraction growth']))/len(df_state['BH fraction growth']), facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)

        # Add error bars
        if add_plot_errorbars:
            hist_n, _ = np.histogram(np.log10(df_state['BH fraction growth']), bins=bins)
            axs.errorbar(bins[:-1]+0.25/2, hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=color_dict[state_i], ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        


    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-3.8, 0.45), (-5, 0.45), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((0.8, 0.45), (2, 0.45), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    plot_annotate = ''
    if use_ETG_LTG:
        plot_annotate = plot_annotate + '%s'%use_ETG_LTG
    if select_high_fgas:
        plot_annotate = plot_annotate + '%s $f_{\mathrm{gas,SF}}$'%select_high_fgas
    if plot_annotate != '':
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-5, 1.5)
    axs.set_ylim(0.001, 1)
    #axs.set_yticks([1, 10, 100, 1000])
    #axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH,%s \:Gyr}}/M_{\mathrm{BH,initial}}$' %target_window_size)
    axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH}}/M_{\mathrm{BH,initial}}$')
    axs.set_ylabel('fraction of sub-sample')
    #axs.minorticks_on()
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_delta_hist/%sbhmass_delta_hist_%s%s_%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_delta_hist/%sbhmass_delta_hist_%s%s_%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# histogram of average Mdot / M_BH, as a measure of generally enhanced/reduced growth
def _BH_mdotmass_hist_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                              window_or_snip_mdot = 'snip',       # average growth over window or time-weighted mean over each snip
                            use_ETG_LTG            = False,           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                              morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                            select_high_fgas       = False,         # [ False / value ] trim based on average
                          #--------------------------------------------
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    lbol_snip_plot = [] # mean over window
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    kappa_mean_plot = []   # average 
    kappa_snip_plot = []    # all snips
    fgassf_plot    = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    angle_plot     = []
    ID_plot        = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                            
                            #bhmdot_snip_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst'])[index_start:index_stop]))
                            
                            # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                            lbol_snip_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol'])[index_start:index_stop]))
                            
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                            
                            #bhmdot_snip_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst_alt'])[index_start:index_stop]))
                        
                            # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                            lbol_snip_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol_alt'])[index_start:index_stop]))
                            
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        angle_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start])
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        
                        kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                        kappa_mean_plot.append(np.mean(kappa_array))
                        if (kappa_array > 0.4).all():
                            kappa_snip_plot.append('LTG')
                        elif (kappa_array < 0.4).all():
                            kappa_snip_plot.append('ETG')
                        else:
                            kappa_snip_plot.append('other')
            
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        
                        #bhmdot_snip_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst'])[index_start:index_stop]))
                        
                        # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                        lbol_snip_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol'])[index_start:index_stop]))       
                        
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        
                        #bhmdot_snip_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mdot_inst_alt'])[index_start:index_stop]))
                        
                        # converting simple Lbol to common e_r / 1-e_r version, multiply by 1.111
                        lbol_snip_plot.append(np.mean(1.111 * np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_lbol_alt'])[index_start:index_stop]))
                        
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    angle_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start])
                    fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                    kappa_mean_plot.append(np.mean(kappa_array))
                    if (kappa_array > 0.4).all():
                        kappa_snip_plot.append('LTG')
                    elif (kappa_array < 0.4).all():
                        kappa_snip_plot.append('ETG')
                    else:
                        kappa_snip_plot.append('other')
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    trelax_plot.append(math.nan)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'trelax': trelax_plot, 'Kappa snip': kappa_snip_plot, 'Gassf fraction': fgassf_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'Angle': angle_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step (default)
    df['BH lbol window']     = lbol_snip_plot           # time-weighted instantaneous lbol
    df['BH mdot growth']     = df['BH mdot snip']/(df['BH mass start'])
    
    #-------------------
    # Select only ETG/LTG
    if use_ETG_LTG:
        if use_ETG_LTG == 'ETG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'ETG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] < 0.4)]
        if use_ETG_LTG == 'LTG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'LTG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] > 0.4)]
                
        print('Using sample of %s %ss:      %s' %(morph_average_or_snips, use_ETG_LTG, len(df_morph['GalaxyIDs'])))
        df = df_morph
        df_morph = 0
    
    # Select only high fgas
    if select_high_fgas:
        df_fgas = df.loc[(df['Gassf fraction'] > select_high_fgas)]
        
        print('Using sample of %s f_gas,SF:      %s' %(select_high_fgas, len(df_fgas['GalaxyIDs'])))
        df = df_fgas
        df_fgas = 0
    
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    print('-------')
    print('Median fractional growth per year:       [ Mdot / M_initial ]')
    print('   aligned:       %.2e /yr' %(np.median(df_co['BH mdot growth'])))
    print('   misaligned:    %.2e /yr' %(np.median(df_mis['BH mdot growth'])))
    print('   counter:       %.2e /yr' %(np.median(df_cnt['BH mdot growth'])))
    #print('Median lbol:       [ erg / s ]')
    #print('   aligned:       %.2e erg/s' %(np.median(df_co['BH lbol window'])))
    #print('   misaligned:    %.2e erg/s' %(np.median(df_mis['BH lbol window'])))
    #print('   counter:       %.2e erg/s' %(np.median(df_cnt['BH lbol window'])))
    
    
    
    #---------------
    # KS test
    print('-------------')
    res = stats.ks_2samp(df_co['BH mdot growth'], df_mis['BH mdot growth'])
    print('KS-test:     aligned - misaligned')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_co['BH mdot growth'], df_cnt['BH mdot growth'])
    print('KS-test:     aligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_mis['BH mdot growth'], df_cnt['BH mdot growth'])
    print('KS-test:     misaligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    
    #---------------
    # KS test ETG vs LTG
    if not use_ETG_LTG:
        if morph_average_or_snips == 'snips':
            df_co_ETG  = df.loc[(df['Kappa snip'] == 'ETG') & (df['State'] == 'aligned')]
            df_mis_ETG = df.loc[(df['Kappa snip'] == 'ETG') & (df['State'] == 'misaligned')]
            df_cnt_ETG = df.loc[(df['Kappa snip'] == 'ETG') & (df['State'] == 'counter')]

            df_co_LTG  = df.loc[(df['Kappa snip'] == 'LTG') & (df['State'] == 'aligned')]
            df_mis_LTG = df.loc[(df['Kappa snip'] == 'LTG') & (df['State'] == 'misaligned')]
            df_cnt_LTG = df.loc[(df['Kappa snip'] == 'LTG') & (df['State'] == 'counter')]
        elif morph_average_or_snips == 'average':
            df_co_ETG  = df.loc[(df['Kappa mean'] < 0.4) & (df['State'] == 'aligned')]
            df_mis_ETG = df.loc[(df['Kappa mean'] < 0.4) & (df['State'] == 'misaligned')]
            df_cnt_ETG = df.loc[(df['Kappa mean'] < 0.4) & (df['State'] == 'counter')]

            df_co_LTG  = df.loc[(df['Kappa mean'] > 0.4) & (df['State'] == 'aligned')]
            df_mis_LTG = df.loc[(df['Kappa mean'] > 0.4) & (df['State'] == 'misaligned')]
            df_cnt_LTG = df.loc[(df['Kappa mean'] > 0.4) & (df['State'] == 'counter')]
        
        print('\n-------------')
        print('KS morphology test for target window: %s' %target_window_size)
        res = stats.ks_2samp(df_co_LTG['BH mdot growth'], df_co_ETG['BH mdot growth'])
        print('KS-test:     aligned: ETG vs LTG')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co_ETG.index) + len(df_co_LTG.index))/(len(df_co_ETG.index)*len(df_co_LTG.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis_LTG['BH mdot growth'], df_mis_ETG['BH mdot growth'])
        print('KS-test:     misaligned: ETG vs LTG')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis_ETG.index) + len(df_mis_LTG.index))/(len(df_mis_ETG.index)*len(df_mis_LTG.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_cnt_ETG['BH mdot growth'], df_cnt_LTG['BH mdot growth'])
        print('KS-test:     counter: ETG vs LTG')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_cnt_ETG.index) + len(df_cnt_LTG.index))/(len(df_cnt_ETG.index)*len(df_cnt_LTG.index))))))
        print('   p-value: %s' %res.pvalue)
        
        df_co_ETG  = 0
        df_mis_ETG = 0
        df_cnt_ETG = 0
        df_co_LTG  = 0
        df_mis_LTG = 0
        df_cnt_LTG = 0
    
    
    """
    # Plot initial angle
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    bins = np.arange(0, 181, 10)
    axs.hist(df_co['Angle'], bins=bins, weights=np.ones(len(df_co['BH mdot growth']))/len(df_co['BH mdot growth']), linewidth=1, facecolor='none', edgecolor='k', alpha=0.8)
    axs.hist(df_mis['Angle'], bins=bins, weights=np.ones(len(df_mis['BH mdot growth']))/len(df_mis['BH mdot growth']), linewidth=1, facecolor='none', edgecolor='r', alpha=0.8)
    axs.hist(df_cnt['Angle'], bins=bins, weights=np.ones(len(df_cnt['BH mdot growth']))/len(df_cnt['BH mdot growth']), linewidth=1, facecolor='none', edgecolor='b', alpha=0.8)
    axs.set_xlabel('First angle in array')
    axs.set_xlim(0, 180)
    plt.show()
    plt.close()
    """
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot histograms
    add_plot_errorbars = True
    for state_i in color_dict.keys():
        
        df_state = df.loc[df['State'] == state_i]
        
        bins = np.arange(-13, -7.5+0.01, 0.25)
        
        axs.hist(np.log10(df_state['BH mdot growth']), log=True, bins=bins, weights=np.ones(len(df_state['BH mdot growth']))/len(df_state['BH mdot growth']), facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=0.8)
        if state_i == 'aligned':
            axs.hist(np.log10(df_state['BH mdot growth']), log=True, bins=bins, weights=np.ones(len(df_state['BH mdot growth']))/len(df_state['BH mdot growth']), facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)

        # Add error bars
        if add_plot_errorbars:
            hist_n, _ = np.histogram(np.log10(df_state['BH mdot growth']), bins=bins)
            axs.errorbar(bins[:-1]+0.25/2, hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=color_dict[state_i], ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        


    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-3.8, 0.45), (-5, 0.45), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((0.8, 0.45), (2, 0.45), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    plot_annotate = ''
    if use_ETG_LTG:
        plot_annotate = plot_annotate + '%s'%use_ETG_LTG
    if select_high_fgas:
        plot_annotate = plot_annotate + '%s $f_{\mathrm{gas,SF}}$'%select_high_fgas
    if plot_annotate != '':
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-13, -7.5)
    axs.set_ylim(0.001, 1)
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{BH}} \rangle _{\mathrm{%s\:Gyr}}/M_{\mathrm{BH,initial}}$ [yr$^{-1}$]' %target_window_size)
    axs.set_ylabel('fraction of sub-sample')
    #axs.minorticks_on()
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_delta_hist/%sbhmass_mdot_hist_%s%s_%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_delta_hist/%sbhmass_mdot_hist_%s%s_%s%sclean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', ('trelax' if must_still_be_misaligned else 'window'), target_window_size, ((use_ETG_LTG + '_') if use_ETG_LTG else ''), ((str(select_high_fgas) + 'fgassf_') if select_high_fgas else ''), BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#--------------------------------
# x-y of fractional BH growth delta M_BH / M_BH vs gas fraction
def _BH_deltamassmass_fgas_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                              ring_or_enclosed   = 'enclosed',     # comparing rings of surface density or total enclosed
                              initial_or_average = 'average',       # [ 'initial', 'average', 'peak' ] averaged over window
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    fgas_plot      = []
    fgassf_plot    = []
    gas_density_plot    = []
    gassf_density_plot  = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        
                        #---------------
                        # Gas and gassf fraction
                        if initial_or_average == 'initial':
                            fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                            fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        elif initial_or_average == 'average':
                            fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        elif initial_or_average == 'peak':
                            fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    
                        # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                        if initial_or_average == 'initial':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                                gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                                gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                        elif initial_or_average == 'average':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        elif initial_or_average == 'peak':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        gas_density_plot.append(gas_surfdens_ratio)
                        gassf_density_plot.append(gassf_surfdens_ratio)
                                
                                    
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    #---------------
                    # Gas and gassf fraction
                    if initial_or_average == 'initial':
                        fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                    elif initial_or_average == 'average':
                        fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    elif initial_or_average == 'peak':
                        fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    
                    # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                    if initial_or_average == 'initial':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                            gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                            gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                    elif initial_or_average == 'average':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    elif initial_or_average == 'peak':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    gas_density_plot.append(gas_surfdens_ratio)
                    gassf_density_plot.append(gassf_surfdens_ratio)
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Gas fraction': fgas_plot, 'Gassf fraction': fgassf_plot, 'Gas surfdens ratio': gas_density_plot, 'Gassf surfdens ratio': gassf_density_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    # Medians and KS test
    if gas_fraction_type == 'gas':
        print('-------')
        print('Median fgas:       [ fgas fraction]')
        print('   aligned:       %.4f' %(np.median(df_co['Gas fraction'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gas fraction'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gas fraction'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gas fraction'], df_mis['Gas fraction'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gas fraction'], df_cnt['Gas fraction'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gas fraction'], df_cnt['Gas fraction'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    if gas_fraction_type == 'gas_sf':
        print('-------')
        print('Median fgas_sf:       [ fgas_sf fraction]')
        print('   aligned:       %.4f' %(np.median(df_co['Gassf fraction'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gassf fraction'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gassf fraction'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gassf fraction'], df_mis['Gassf fraction'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gassf fraction'], df_cnt['Gassf fraction'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gassf fraction'], df_cnt['Gassf fraction'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    
    
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    if gas_fraction_type == 'gas':
        axs.scatter(np.log10(df['BH fraction growth']), np.log10(df['Gas fraction']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    if gas_fraction_type == 'gas_sf':
        axs.scatter(np.log10(df['BH fraction growth']), np.log10(df['Gassf fraction']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH fraction growth']))
        if gas_fraction_type == 'gas':
            y = np.array(np.log10(df_i['Gas fraction']))
        if gas_fraction_type == 'gas_sf':
            y = np.array(np.log10(df_i['Gassf fraction']))

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-5, 1.5)
    #axs.set_yscale('log')
    #axs.set_ylim(0.0025, 1)
    axs.set_ylim(-2, -0.25)
    #axs.set_yticks([1, 10, 100, 1000])
    #axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH,%s \:Gyr}}/M_{\mathrm{BH,initial}}$' %target_window_size)
    axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH}}/M_{\mathrm{BH,initial}}$')
    if initial_or_average == 'initial':
        axs.set_ylabel('log$_{10}$ $f_{\mathrm{gas,SF}}(<r_{50})$')
    elif initial_or_average == 'average':
        #axs.set_ylabel(r'log$_{10}$ $\langle f_{\mathrm{gas,SF}}(<r_{50}) \rangle _{\mathrm{%s\:Gyr}}$' %target_window_size)
        axs.set_ylabel(r'log$_{10}$ $\bar{f}_{\mathrm{gas,SF}}(<r_{50})$')
        
    elif initial_or_average == 'peak':
        axs.set_ylabel(r'log$_{10}$ $f_{\mathrm{peak gas,SF}}(<r_{50})$')
        
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_fgas/%sbhmass_delta_%s_f%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_fgas/%sbhmass_delta_%s_f%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# x-y of average mdot / M_BH vs gas fraction
def _BH_mdotmass_fgas_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                              window_or_snip_mdot = 'snip',       # average growth over window or time-weighted mean over each snip
                            gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                              ring_or_enclosed   = 'enclosed',     # comparing rings of surface density or total enclosed
                              initial_or_average = 'average',       # [ 'initial', 'average', 'peak' ] averaged over window
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    fgas_plot      = []
    fgassf_plot    = []
    gas_density_plot    = []
    gassf_density_plot  = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        
                        #---------------
                        # Gas and gassf fraction
                        if initial_or_average == 'initial':
                            fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                            fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        elif initial_or_average == 'average':
                            fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        elif initial_or_average == 'peak':
                            fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    
                        # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                        if initial_or_average == 'initial':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                                gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                                gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                        elif initial_or_average == 'average':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        elif initial_or_average == 'peak':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        gas_density_plot.append(gas_surfdens_ratio)
                        gassf_density_plot.append(gassf_surfdens_ratio)
                                
                                    
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    #---------------
                    # Gas and gassf fraction
                    if initial_or_average == 'initial':
                        fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                    elif initial_or_average == 'average':
                        fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    elif initial_or_average == 'peak':
                        fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    
                    # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                    if initial_or_average == 'initial':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                            gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                            gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                    elif initial_or_average == 'average':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    elif initial_or_average == 'peak':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    gas_density_plot.append(gas_surfdens_ratio)
                    gassf_density_plot.append(gassf_surfdens_ratio)
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Gas fraction': fgas_plot, 'Gassf fraction': fgassf_plot, 'Gas surfdens ratio': gas_density_plot, 'Gassf surfdens ratio': gassf_density_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    df['BH mdot growth']     = df['BH mdot snip']/(df['BH mass start'])
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    # Medians and KS test
    if gas_fraction_type == 'gas':
        print('-------')
        print('Median fgas:       [ fgas fraction]')
        print('   aligned:       %.4f' %(np.median(df_co['Gas fraction'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gas fraction'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gas fraction'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gas fraction'], df_mis['Gas fraction'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gas fraction'], df_cnt['Gas fraction'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gas fraction'], df_cnt['Gas fraction'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    if gas_fraction_type == 'gas_sf':
        print('-------')
        print('Median fgas_sf:       [ fgas_sf fraction]')
        print('   aligned:       %.4f' %(np.median(df_co['Gassf fraction'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gassf fraction'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gassf fraction'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gassf fraction'], df_mis['Gassf fraction'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gassf fraction'], df_cnt['Gassf fraction'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gassf fraction'], df_cnt['Gassf fraction'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    
    
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    if gas_fraction_type == 'gas':
        axs.scatter(np.log10(df['BH mdot growth']), np.log10(df['Gas fraction']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    if gas_fraction_type == 'gas_sf':
        axs.scatter(np.log10(df['BH mdot growth']), np.log10(df['Gassf fraction']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH mdot growth']))
        if gas_fraction_type == 'gas':
            y = np.array(np.log10(df_i['Gas fraction']))
        if gas_fraction_type == 'gas_sf':
            y = np.array(np.log10(df_i['Gassf fraction']))

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.7,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-13, -7.5)
    #axs.set_yscale('log')
    #axs.set_ylim(0.0025, 1)
    axs.set_ylim(-2, -0.25)
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{BH}} \rangle _{\mathrm{%s\:Gyr}}/M_{\mathrm{BH,initial}}$ [yr$^{-1}$]' %target_window_size)
    if initial_or_average == 'initial':
        axs.set_ylabel('log$_{10}$ $f_{\mathrm{gas,SF}}(<r_{50})$')
    elif initial_or_average == 'average':
        axs.set_ylabel(r'log$_{10}$ $\langle f_{\mathrm{gas,SF}}(<r_{50}) \rangle _{\mathrm{%s\:Gyr}}$' %target_window_size)
    elif initial_or_average == 'peak':
        axs.set_ylabel(r'log$_{10}$ $f_{\mathrm{peak gas,SF}}(<r_{50})$')
        
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_fgas/%sbhmass_mdot_%s_f%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_fgas/%sbhmass_mdot_%s_f%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()

#--------------------------------
# x-y of fractional BH growth delta M_BH / M_BH vs ratio of gas surface density at r50 and between r50 and 2r50
def _BH_deltamassmass_gassurfratio_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                              ring_or_enclosed          = 'enclosed',               # gas surface density within 2r50, or between r50 and 2r50
                              initial_or_average        = 'average',       # [ 'initial', 'average', 'peak', 'delta'] in window
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    fgas_plot      = []
    fgassf_plot    = []
    gas_density_plot    = []
    gassf_density_plot  = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        
                        #---------------
                        # Gas and gassf fraction
                        if initial_or_average == 'initial':
                            fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                            fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        elif initial_or_average == 'average':
                            fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        elif initial_or_average == 'peak':
                            fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        elif initial_or_average == 'delta':
                            fgas_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                            fgassf_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                            fgas_plot.append(fgas_temp[-1] - fgas_temp[0])
                            fgassf_plot.append(fgassf_temp[-1] - fgassf_temp[0])
                            
                        
                        # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                        if initial_or_average == 'initial':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                                gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                                gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                        elif initial_or_average == 'average':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        elif initial_or_average == 'peak':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        elif initial_or_average == 'delta':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[-1] - (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[0]
                                gassf_surfdens_ratio = (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[-1] - (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[0]
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = (4*np.divide(gasmass_r50, gasmass_2r50))[-1] - (4*np.divide(gasmass_r50, gasmass_2r50))[0]
                                gassf_surfdens_ratio = (4*np.divide(gassfmass_r50, gassfmass_2r50))[-1] - (4*np.divide(gassfmass_r50, gassfmass_2r50))[0]                               
                                
                        gas_density_plot.append(gas_surfdens_ratio)
                        gassf_density_plot.append(gassf_surfdens_ratio)
                                
                                    
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    #---------------
                    # Gas and gassf fraction
                    if initial_or_average == 'initial':
                        fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                    elif initial_or_average == 'average':
                        fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    elif initial_or_average == 'peak':
                        fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    elif initial_or_average == 'delta':
                        fgas_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                        fgassf_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                        fgas_plot.append(fgas_temp[-1] - fgas_temp[0])
                        fgassf_plot.append(fgassf_temp[-1] - fgassf_temp[0])
                    
                    # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                    if initial_or_average == 'initial':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                            gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                            gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                    elif initial_or_average == 'average':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    elif initial_or_average == 'peak':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    elif initial_or_average == 'delta':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[-1] - (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[0]
                            gassf_surfdens_ratio = (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[-1] - (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[0]
                    
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = (4*np.divide(gasmass_r50, gasmass_2r50))[-1] - (4*np.divide(gasmass_r50, gasmass_2r50))[0]
                            gassf_surfdens_ratio = (4*np.divide(gassfmass_r50, gassfmass_2r50))[-1] - (4*np.divide(gassfmass_r50, gassfmass_2r50))[0]
                    gas_density_plot.append(gas_surfdens_ratio)
                    gassf_density_plot.append(gassf_surfdens_ratio)
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Gas fraction': fgas_plot, 'Gassf fraction': fgassf_plot, 'Gas surfdens ratio': gas_density_plot, 'Gassf surfdens ratio': gassf_density_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    # Medians and KS test
    if gas_fraction_type == 'gas':
        print('-------')
        if initial_or_average != 'delta':
            print('Median gas surfdens ratio:       [ r50/2r50 ]')
        else:
            print('Median gas surfdens ratio change over window:       [ r50/2r50 ]')
        print('   aligned:       %.4f' %(np.median(df_co['Gas surfdens ratio'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gas surfdens ratio'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gas surfdens ratio'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gas surfdens ratio'], df_mis['Gas surfdens ratio'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gas surfdens ratio'], df_cnt['Gas surfdens ratio'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gas surfdens ratio'], df_cnt['Gas surfdens ratio'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    if gas_fraction_type == 'gas_sf':
        print('-------')
        if initial_or_average != 'delta':
            print('Median gas_sf surfdens ratio:       [ r50/2r50 ]')
        else:
            print('Median gas_sf surfdens ratio change over window:       [ r50/2r50 ]')
        print('   aligned:       %.4f' %(np.median(df_co['Gassf surfdens ratio'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gassf surfdens ratio'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gassf surfdens ratio'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gassf surfdens ratio'], df_mis['Gassf surfdens ratio'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gassf surfdens ratio'], df_cnt['Gassf surfdens ratio'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gassf surfdens ratio'], df_cnt['Gassf surfdens ratio'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    if initial_or_average != 'delta':
        if gas_fraction_type == 'gas':
            axs.scatter(np.log10(df['BH fraction growth']), np.log10(df['Gas surfdens ratio']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        if gas_fraction_type == 'gas_sf':
            axs.scatter(np.log10(df['BH fraction growth']), np.log10(df['Gassf surfdens ratio']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    else:
        if gas_fraction_type == 'gas':
            axs.scatter(np.log10(df['BH fraction growth']), df['Gas surfdens ratio'], s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)
        if gas_fraction_type == 'gas_sf':
            axs.scatter(np.log10(df['BH fraction growth']), df['Gassf surfdens ratio'], s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)
        
        
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH fraction growth']))
        if gas_fraction_type == 'gas':
            if initial_or_average != 'delta':
                y = np.array(np.log10(df_i['Gas surfdens ratio']))
            else:
                y = np.array(df_i['Gas surfdens ratio'])
        if gas_fraction_type == 'gas_sf':
            if initial_or_average != 'delta':
                y = np.array(np.log10(df_i['Gassf surfdens ratio']))
            else:
                y = np.array(df_i['Gassf surfdens ratio'])
                

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
        
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 100), (-12.8, 100), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 100), (-4.2, 100), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-5, 1.5)
    #axs.set_yscale('log')
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH,%s \:Gyr}}/M_{\mathrm{BH,initial}}$' %target_window_size)
    if ring_or_enclosed == 'ring':
        axs.set_ylim(-2, 3)
        if initial_or_average == 'initial':
            axs.set_ylabel('log$_{10}$ $\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}}$')
        elif initial_or_average == 'average':
            axs.set_ylabel(r'log$_{10}$ $\langle \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}} \rangle _{\mathrm{%s \:Gyr}}$' %target_window_size)
        elif initial_or_average == 'peak':
            axs.set_ylabel('log$_{10}$ ${\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}}}_{\mathrm{peak}}$')
        elif initial_or_average == 'delta':
            axs.set_ylabel(r'$\Delta \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}}$')
    elif ring_or_enclosed == 'enclosed':
        axs.set_ylim(-0.2, 0.65)
        if initial_or_average == 'initial':
            axs.set_ylabel('log$_{10}$ $\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}}$')
        elif initial_or_average == 'average':
            axs.set_ylabel(r'log$_{10}$ $\langle \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}} \rangle _{\mathrm{%s \:Gyr}}$' %target_window_size)
        elif initial_or_average == 'peak':
            axs.set_ylabel('log$_{10}$ ${\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}}}_{\mathrm{peak}}$')
        elif initial_or_average == 'delta':
            axs.set_ylim(-2, 2)
            axs.set_ylabel(r'$\Delta \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_gas_surfdens/%sbhmass_delta_%s_surfdens%s_%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ring_or_enclosed, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_gas_surfdens/%sbhmass_delta_%s_surfdens%s_%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ring_or_enclosed, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
    if showfig:
        plt.show()
    plt.close()
# x-y of average mdot / M_BH vs ratio of gas surface density at r50 and between r50 and 2r50
def _BH_mdotmass_gassurfratio_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                              window_or_snip_mdot = 'snip',       # average growth over window or time-weighted mean over each snip
                            gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                              ring_or_enclosed          = 'enclosed',               # gas surface density within 2r50, or between r50 and 2r50
                              initial_or_average        = 'average',       # [ 'initial', 'average', 'peak', 'delta'] in window
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    fgas_plot      = []
    fgassf_plot    = []
    gas_density_plot    = []
    gassf_density_plot  = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        
                        #---------------
                        # Gas and gassf fraction
                        if initial_or_average == 'initial':
                            fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                            fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        elif initial_or_average == 'average':
                            fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        elif initial_or_average == 'peak':
                            fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                            fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        elif initial_or_average == 'delta':
                            fgas_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                            fgassf_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                            fgas_plot.append(fgas_temp[-1] - fgas_temp[0])
                            fgassf_plot.append(fgassf_temp[-1] - fgassf_temp[0])
                            
                        
                        # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                        if initial_or_average == 'initial':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                                gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                                gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                        elif initial_or_average == 'average':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        elif initial_or_average == 'peak':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                                gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                                gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                        elif initial_or_average == 'delta':
                            #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                            gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                            gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                            gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                            gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                            if ring_or_enclosed == 'ring':
                                gas_surfdens_ratio = (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[-1] - (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[0]
                                gassf_surfdens_ratio = (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[-1] - (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[0]
                        
                                # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                                if gasmass_2r50 - gasmass_r50 == 0:
                                    density_fail['gas']['%s' %galaxy_state].append(ID_i)
                                if gassfmass_2r50 - gassfmass_r50 == 0:
                                    density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                            elif ring_or_enclosed == 'enclosed': 
                                gas_surfdens_ratio = (4*np.divide(gasmass_r50, gasmass_2r50))[-1] - (4*np.divide(gasmass_r50, gasmass_2r50))[0]
                                gassf_surfdens_ratio = (4*np.divide(gassfmass_r50, gassfmass_2r50))[-1] - (4*np.divide(gassfmass_r50, gassfmass_2r50))[0]                               
                                
                        gas_density_plot.append(gas_surfdens_ratio)
                        gassf_density_plot.append(gassf_surfdens_ratio)
                                
                                    
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    #---------------
                    # Gas and gassf fraction
                    if initial_or_average == 'initial':
                        fgas_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                        fgassf_plot.append(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start]))
                    elif initial_or_average == 'average':
                        fgas_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    elif initial_or_average == 'peak':
                        fgas_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        fgassf_plot.append(np.max(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    elif initial_or_average == 'delta':
                        fgas_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                        fgassf_temp = (np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop]))
                        fgas_plot.append(fgas_temp[-1] - fgas_temp[0])
                        fgassf_plot.append(fgassf_temp[-1] - fgassf_temp[0])
                    
                    # Gas and gassf surface density ratios (within r50/<2r50-r50>)... the rad cancels out
                    if initial_or_average == 'initial':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = 3*(gasmass_r50/(gasmass_2r50 - gasmass_r50))
                            gassf_surfdens_ratio = 3*(gassfmass_r50/(gassfmass_2r50 - gassfmass_r50))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = 4*(gasmass_r50/(gasmass_2r50))
                            gassf_surfdens_ratio = 4*(gassfmass_r50/(gassfmass_2r50))
                    elif initial_or_average == 'average':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.mean(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.mean(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.mean(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.mean(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    elif initial_or_average == 'peak':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                    
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = np.max(3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))
                            gassf_surfdens_ratio = np.max(3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))
                        
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = np.max(4*np.divide(gasmass_r50, gasmass_2r50))
                            gassf_surfdens_ratio = np.max(4*np.divide(gassfmass_r50, gassfmass_2r50))
                    elif initial_or_average == 'delta':
                        #halfrad = np.array(BH_subsample['%s' %ID_i]['rad'])[index_start]
                        gasmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop]
                        gasmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass'])[index_start:index_stop]
                        gassfmass_r50  = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop]
                        gassfmass_2r50 = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass'])[index_start:index_stop]
                
                        if ring_or_enclosed == 'ring':
                            gas_surfdens_ratio = (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[-1] - (3*np.divide(gasmass_r50, (gasmass_2r50 - gasmass_r50)))[0]
                            gassf_surfdens_ratio = (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[-1] - (3*np.divide(gassfmass_r50, (gassfmass_2r50 - gassfmass_r50)))[0]
                    
                            # If basically all gas or gassf is within r50 (rare, but can happen, this will log number of divide by 0 from above equation)
                            if gasmass_2r50 - gasmass_r50 == 0:
                                density_fail['gas']['%s' %galaxy_state].append(ID_i)
                            if gassfmass_2r50 - gassfmass_r50 == 0:
                                density_fail['gas_sf']['%s' %galaxy_state].append(ID_i)
                        elif ring_or_enclosed == 'enclosed': 
                            gas_surfdens_ratio = (4*np.divide(gasmass_r50, gasmass_2r50))[-1] - (4*np.divide(gasmass_r50, gasmass_2r50))[0]
                            gassf_surfdens_ratio = (4*np.divide(gassfmass_r50, gassfmass_2r50))[-1] - (4*np.divide(gassfmass_r50, gassfmass_2r50))[0]
                    gas_density_plot.append(gas_surfdens_ratio)
                    gassf_density_plot.append(gassf_surfdens_ratio)
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Gas fraction': fgas_plot, 'Gassf fraction': fgassf_plot, 'Gas surfdens ratio': gas_density_plot, 'Gassf surfdens ratio': gassf_density_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    df['BH mdot growth']     = df['BH mdot snip']/(df['BH mass start'])
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    # Medians and KS test
    if gas_fraction_type == 'gas':
        print('-------')
        if initial_or_average != 'delta':
            print('Median gas surfdens ratio:       [ r50/2r50 ]')
        else:
            print('Median gas surfdens ratio change over window:       [ r50/2r50 ]')
        print('   aligned:       %.4f' %(np.median(df_co['Gas surfdens ratio'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gas surfdens ratio'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gas surfdens ratio'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gas surfdens ratio'], df_mis['Gas surfdens ratio'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gas surfdens ratio'], df_cnt['Gas surfdens ratio'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gas surfdens ratio'], df_cnt['Gas surfdens ratio'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    if gas_fraction_type == 'gas_sf':
        print('-------')
        if initial_or_average != 'delta':
            print('Median gas_sf surfdens ratio:       [ r50/2r50 ]')
        else:
            print('Median gas_sf surfdens ratio change over window:       [ r50/2r50 ]')
        print('   aligned:       %.4f' %(np.median(df_co['Gassf surfdens ratio'])))
        print('   misaligned:    %.4f' %(np.median(df_mis['Gassf surfdens ratio'])))
        print('   counter:       %.4f' %(np.median(df_cnt['Gassf surfdens ratio'])))
    
        #---------------
        # KS test
        print('-------------')
        res = stats.ks_2samp(df_co['Gassf surfdens ratio'], df_mis['Gassf surfdens ratio'])
        print('KS-test:     aligned - misaligned')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_co['Gassf surfdens ratio'], df_cnt['Gassf surfdens ratio'])
        print('KS-test:     aligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
        res = stats.ks_2samp(df_mis['Gassf surfdens ratio'], df_cnt['Gassf surfdens ratio'])
        print('KS-test:     misaligned - counter')
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
        print('   p-value: %s' %res.pvalue)
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    if initial_or_average != 'delta':
        if gas_fraction_type == 'gas':
            axs.scatter(np.log10(df['BH mdot growth']), np.log10(df['Gas surfdens ratio']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        if gas_fraction_type == 'gas_sf':
            axs.scatter(np.log10(df['BH mdot growth']), np.log10(df['Gassf surfdens ratio']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    else:
        if gas_fraction_type == 'gas':
            axs.scatter(np.log10(df['BH mdot growth']), df['Gas surfdens ratio'], s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        if gas_fraction_type == 'gas_sf':
            axs.scatter(np.log10(df['BH mdot growth']), df['Gassf surfdens ratio'], s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
        
        
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH mdot growth']))
        if gas_fraction_type == 'gas':
            if initial_or_average != 'delta':
                y = np.array(np.log10(df_i['Gas surfdens ratio']))
            else:
                y = np.array(df_i['Gas surfdens ratio'])
        if gas_fraction_type == 'gas_sf':
            if initial_or_average != 'delta':
                y = np.array(np.log10(df_i['Gassf surfdens ratio']))
            else:
                y = np.array(df_i['Gassf surfdens ratio'])
                

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
        
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 100), (-12.8, 100), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 100), (-4.2, 100), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-13, -7.5)
    #axs.set_yscale('log')
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{BH}} \rangle _{\mathrm{%s\:Gyr}}/M_{\mathrm{BH,initial}}$ [yr$^{-1}$]' %target_window_size)
    if ring_or_enclosed == 'ring':
        axs.set_ylim(-2, 3)
        if initial_or_average == 'initial':
            axs.set_ylabel('log$_{10}$ $\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}}$')
        elif initial_or_average == 'average':
            axs.set_ylabel(r'log$_{10}$ $\langle \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}} \rangle _{\mathrm{%s \:Gyr}}$' %target_window_size)
        elif initial_or_average == 'peak':
            axs.set_ylabel('log$_{10}$ ${\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}}}_{\mathrm{peak}}$')
        elif initial_or_average == 'delta':
            axs.set_ylabel(r'$\Delta \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,r_{50}-2r_{50}}}$')
    elif ring_or_enclosed == 'enclosed':
        axs.set_ylim(-0.2, 0.65)
        if initial_or_average == 'initial':
            axs.set_ylabel('log$_{10}$ $\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}}$')
        elif initial_or_average == 'average':
            axs.set_ylabel(r'log$_{10}$ $\langle \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}} \rangle _{\mathrm{%s \:Gyr}}$' %target_window_size)
        elif initial_or_average == 'peak':
            axs.set_ylabel('log$_{10}$ ${\Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}}}_{\mathrm{peak}}$')
        elif initial_or_average == 'delta':
            axs.set_ylim(-2, 2)
            axs.set_ylabel(r'$\Delta \Sigma_{\mathrm{SF,r_{50}}}/\Sigma_{\mathrm{SF,2r_{50}}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_gas_surfdens/%sbhmass_mdot_%s_surfdens%s_%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ring_or_enclosed, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_gas_surfdens/%sbhmass_mdot_%s_surfdens%s_%s_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ring_or_enclosed, gas_fraction_type, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
    if showfig:
        plt.show()
    plt.close()

#--------------------------------
# x-y of fractional BH growth delta M_BH / M_BH vs gas_sf kappa
def _BH_deltamassmass_gassfkappa_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                              initial_or_average       = 'average',
                          # Test if difference with low 
                            select_low_err         = False,         # [ True / False ] trim to max error of 10 (3 sigma = 30)
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    kappa_gas_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    err_test       = []     # 'low' and 'high' for <10 and 10-30 error
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        if initial_or_average == 'average':
                            kappa_gas_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start:index_stop]))
                        if initial_or_average == 'initial':
                            kappa_gas_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start])
                                
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                        
                        
                        angle_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start:index_stop]
                        err_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf_err'])[index_start:index_stop]
                        if np.max(np.abs(err_array - angle_array[:,None])) < 10:
                            err_test.append('low')
                        else:
                            err_test.append('high')
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    if initial_or_average == 'average':
                        kappa_gas_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start:index_stop]))
                    if initial_or_average == 'initial':
                        kappa_gas_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start])
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    
                    
                    angle_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf'])[index_start:index_stop]
                    err_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stars_gas_sf_err'])[index_start:index_stop]
                    if np.max(np.abs(err_array - angle_array[:,None])) < 10:
                        err_test.append('low')
                    else:
                        err_test.append('high')
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Kappa SF': kappa_gas_plot, 'State': state_plot, 'window': duration_plot, 'err_test': err_test, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    # Select only low uncertainty systems
    if select_low_err:
        # testing effect if using stricter 3 sigma uncertainty
        df = df.loc[(df['err_test'] == 'low')]
        
        print('Using sample of max. 10 deg error (3 sigma = 30):      %s' %(len(df['GalaxyIDs'])))
    
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    
    print('Median kappa_sf over window:       [ kappa_sf ]')
    print('   aligned:       %.4f' %(np.median(df_co['Kappa SF'])))
    print('   misaligned:    %.4f' %(np.median(df_mis['Kappa SF'])))
    print('   counter:       %.4f' %(np.median(df_cnt['Kappa SF'])))

    #---------------
    # KS test
    print('-------------')
    res = stats.ks_2samp(df_co['Kappa SF'], df_mis['Kappa SF'])
    print('KS-test:     aligned - misaligned')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_co['Kappa SF'], df_cnt['Kappa SF'])
    print('KS-test:     aligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_mis['Kappa SF'], df_cnt['Kappa SF'])
    print('KS-test:     misaligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    axs.scatter(np.log10(df['BH fraction growth']), df['Kappa SF'], s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH fraction growth']))
        y = np.array(df_i['Kappa SF'])

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    plot_annotate = ''
    if select_low_err:
        plot_annotate = plot_annotate + '$<10^{\circ}$ error'
    if plot_annotate != '':
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-5, 1.5)
    #axs.set_yscale('log')
    axs.set_ylim(0.2, 1)
    #axs.set_yticks([1, 10, 100, 1000])
    #axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH,%s \:Gyr}}/M_{\mathrm{BH,initial}}$' %target_window_size)
    axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH}}/M_{\mathrm{BH,initial}}$')
    if initial_or_average == 'average':
        #axs.set_ylabel(r'$\langle \kappa_{\mathrm{co}}^{\mathrm{SF}} \rangle _{\mathrm{%s \:Gyr}}$' %target_window_size)

        axs.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{\mathrm{SF}}$')
    if initial_or_average == 'initial':
        axs.set_ylabel(r'$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_gas_kappa/%sbhmass_delta_%s_SFkappa_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_gas_kappa/%sbhmass_delta_%s_SFkappa_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# x-y of average mdot / M_BH vs gas_sf kappa
def _BH_mdotmass_gassfkappa_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                              window_or_snip_mdot = 'snip',       # average growth over window or time-weighted mean over each snip
                              initial_or_average       = 'average',
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    kappa_gas_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        if initial_or_average == 'average':
                            kappa_gas_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start:index_stop]))
                        if initial_or_average == 'initial':
                            kappa_gas_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start])
                                
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    if initial_or_average == 'average':
                        kappa_gas_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start:index_stop]))
                    if initial_or_average == 'initial':
                        kappa_gas_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_sf'])[index_start])
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Kappa SF': kappa_gas_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    df['BH mdot growth']     = df['BH mdot snip']/(df['BH mass start'])
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    
    print('Median kappa_sf over window:       [ kappa_sf ]')
    print('   aligned:       %.4f' %(np.median(df_co['Kappa SF'])))
    print('   misaligned:    %.4f' %(np.median(df_mis['Kappa SF'])))
    print('   counter:       %.4f' %(np.median(df_cnt['Kappa SF'])))

    #---------------
    # KS test
    print('-------------')
    res = stats.ks_2samp(df_co['Kappa SF'], df_mis['Kappa SF'])
    print('KS-test:     aligned - misaligned')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_mis.index))/(len(df_co.index)*len(df_mis.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_co['Kappa SF'], df_cnt['Kappa SF'])
    print('KS-test:     aligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_co.index) + len(df_cnt.index))/(len(df_co.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(df_mis['Kappa SF'], df_cnt['Kappa SF'])
    print('KS-test:     misaligned - counter')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(df_mis.index) + len(df_cnt.index))/(len(df_mis.index)*len(df_cnt.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    axs.scatter(np.log10(df['BH mdot growth']), df['Kappa SF'], s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH mdot growth']))
        y = np.array(df_i['Kappa SF'])

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-13, -7.5)
    #axs.set_yscale('log')
    axs.set_ylim(0.2, 1)
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{BH}} \rangle _{\mathrm{%s\:Gyr}}/M_{\mathrm{BH,initial}}$ [yr$^{-1}$]' %target_window_size)
    if initial_or_average == 'average':
        axs.set_ylabel(r'$\langle \kappa_{\mathrm{co}}^{\mathrm{SF}} \rangle _{\mathrm{%s \:Gyr}}$' %target_window_size)
    if initial_or_average == 'initial':
        axs.set_ylabel(r'$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_gas_kappa/%sbhmass_mdot_%s_SFkappa_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_gas_kappa/%sbhmass_mdot_%s_SFkappa_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', initial_or_average, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()

#--------------------------------
# x-y of fractional BH growth delta M_BH / M_BH vs gas inflow
def _BH_deltamassmass_gas_inflow_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            gas_fraction_type                   = 'gas',              # [ 'gas' / 'gas_sf' ]
                            inflow_hmr                          = 1,            # 1 or 2
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    inflow_plot    = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        inflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['inflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', inflow_hmr)])[index_start:index_stop]))
                         
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    inflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['inflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', inflow_hmr)])[index_start:index_stop]))
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Inflow rate': inflow_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    
    # Medians and KS test
    print('-------')
    if gas_fraction_type == 'gas':
        print('Median gas inflow rate:       [ Msun / yr ]')
    if gas_fraction_type == 'gas_sf':
        print('Median gas_sf inflow rate:       [ Msun / yr ]')
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
        
        y = np.array(np.log10(df_i['Inflow rate']))
        y = y[~np.isnan(y)]             # handful of nan or inf values for inflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for inflow
        print('   %s:       %.4f (log)' %(state_i, np.median(y)))
    #---------------
    # KS test
    print('-------------')
    for state_i, state_ii  in zip(['aligned', 'aligned', 'misaligned'], ['misaligned', 'counter', 'counter']):
        df_i = df.loc[(df['State'] == state_i)]
        df_ii = df.loc[(df['State'] == state_ii)]
        
        y = np.array(df_i['Inflow rate'])
        y = y[~np.isnan(y)]             # handful of nan or inf values for inflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for inflow
        x = np.array(df_ii['Inflow rate'])
        x = x[~np.isnan(x)]             # handful of nan or inf values for inflow
        x = x[~np.isinf(x)]             # handful of nan or inf values for inflow
        
        res = stats.ks_2samp(y, x)
        print('KS-test:     %s - %s' %(state_i, state_ii))
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(x) + len(y))/(len(x)*len(y))))))
        print('   p-value: %s' %res.pvalue)
      
        
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    axs.scatter(np.log10(df['BH fraction growth']), np.log10(df['Inflow rate']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH fraction growth']))
        y = np.array(np.log10(df_i['Inflow rate']))
        x = x[~np.isnan(y)]                 # handful of nan or inf values for inflow
        y = y[~np.isnan(y)]
        x = x[~np.isinf(y)]                 # handful of nan or inf values for inflow
        y = y[~np.isinf(y)]

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-5, 1.5)
    #axs.set_yscale('log')
    axs.set_ylim(-1.5, 1.5)
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH,%s \:Gyr}}/M_{\mathrm{BH,initial}}$' %target_window_size)
    #axs.set_ylabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{%s}, %s r_{50}} \rangle _{\mathrm{%s \: Gyr}}$ [M$_{\odot}$ yr$^{-1}$]'%('gas' if gas_fraction_type == 'gas' else 'gas,SF', inflow_hmr, target_window_size))
    axs.set_ylabel(r'log$_{10}$ $\bar{\dot{M}}_{\mathrm{%s}} (%s r_{50})$ [M$_{\odot}$ yr$^{-1}$]'%('gas' if gas_fraction_type == 'gas' else 'gas,SF', ('' if inflow_hmr == 1 else '2')))
    
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_inflow/%sbhmass_delta_%s_inflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, inflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_inflow/%sbhmass_delta_%s_inflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, inflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# x-y of average mdot / M_BH vs gas_sf kappa
def _BH_mdotmass_gas_inflow_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                              window_or_snip_mdot = 'snip',       # average growth over window or time-weighted mean over each snip
                            gas_fraction_type                   = 'gas',              # [ 'gas' / 'gas_sf' ]
                            inflow_hmr                          = 1,            # 1 or 2
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    inflow_plot    = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        inflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['inflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', inflow_hmr)])[index_start:index_stop]))
                         
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    inflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['inflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', inflow_hmr)])[index_start:index_stop]))
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Inflow rate': inflow_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    df['BH mdot growth']     = df['BH mdot snip']/(df['BH mass start'])
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    
    # Medians and KS test
    print('-------')
    if gas_fraction_type == 'gas':
        print('Median gas inflow rate:       [ Msun / yr ]')
    if gas_fraction_type == 'gas_sf':
        print('Median gas_sf inflow rate:       [ Msun / yr ]')
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
        
        y = np.array(np.log10(df_i['Inflow rate']))
        y = y[~np.isnan(y)]             # handful of nan or inf values for inflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for inflow
        print('   %s:       %.4f (log)' %(state_i, np.median(y)))
    #---------------
    # KS test
    print('-------------')
    for state_i, state_ii  in zip(['aligned', 'aligned', 'misaligned'], ['misaligned', 'counter', 'counter']):
        df_i = df.loc[(df['State'] == state_i)]
        df_ii = df.loc[(df['State'] == state_ii)]
        
        y = np.array(df_i['Inflow rate'])
        y = y[~np.isnan(y)]             # handful of nan or inf values for inflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for inflow
        x = np.array(df_ii['Inflow rate'])
        x = x[~np.isnan(x)]             # handful of nan or inf values for inflow
        x = x[~np.isinf(x)]             # handful of nan or inf values for inflow
        
        res = stats.ks_2samp(y, x)
        print('KS-test:     %s - %s' %(state_i, state_ii))
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(x) + len(y))/(len(x)*len(y))))))
        print('   p-value: %s' %res.pvalue)
      
        
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    axs.scatter(np.log10(df['BH mdot growth']), np.log10(df['Inflow rate']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH mdot growth']))
        y = np.array(np.log10(df_i['Inflow rate']))
        x = x[~np.isnan(y)]                 # handful of nan or inf values for inflow
        y = y[~np.isnan(y)]
        x = x[~np.isinf(y)]                 # handful of nan or inf values for inflow
        y = y[~np.isinf(y)]

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-13, -7.5)
    #axs.set_yscale('log')
    axs.set_ylim(-1.5, 1.5)
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{BH}} \rangle _{\mathrm{%s\:Gyr}}/M_{\mathrm{BH,initial}}$ [yr$^{-1}$]' %target_window_size)
    axs.set_ylabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{%s}, %s r_{50}} \rangle _{\mathrm{%s \: Gyr}}$ [M$_{\odot}$ yr$^{-1}$]'%('gas' if gas_fraction_type == 'gas' else 'gas,SF', inflow_hmr, target_window_size))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_inflow/%sbhmass_mdot_%s_inflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, inflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_inflow/%sbhmass_mdot_%s_inflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, inflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()

#--------------------------------
# x-y of fractional BH growth delta M_BH / M_BH vs gas outflow
def _BH_deltamassmass_gas_outflow_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            gas_fraction_type                   = 'gas',              # [ 'gas' / 'gas_sf' ]
                            outflow_hmr                          = 1,            # 1 or 2
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    outflow_plot    = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        outflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['outflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', outflow_hmr)])[index_start:index_stop]))
                         
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    outflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['outflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', outflow_hmr)])[index_start:index_stop]))
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Inflow rate': outflow_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    
    # Medians and KS test
    print('-------')
    if gas_fraction_type == 'gas':
        print('Median gas outflow rate:       [ Msun / yr ]')
    if gas_fraction_type == 'gas_sf':
        print('Median gas_sf outflow rate:       [ Msun / yr ]')
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
        
        y = np.array(np.log10(df_i['Inflow rate']))
        y = y[~np.isnan(y)]             # handful of nan or inf values for outflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for outflow
        print('   %s:       %.4f (log)' %(state_i, np.median(y)))
    #---------------
    # KS test
    print('-------------')
    for state_i, state_ii  in zip(['aligned', 'aligned', 'misaligned'], ['misaligned', 'counter', 'counter']):
        df_i = df.loc[(df['State'] == state_i)]
        df_ii = df.loc[(df['State'] == state_ii)]
        
        y = np.array(df_i['Inflow rate'])
        y = y[~np.isnan(y)]             # handful of nan or inf values for outflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for outflow
        x = np.array(df_ii['Inflow rate'])
        x = x[~np.isnan(x)]             # handful of nan or inf values for outflow
        x = x[~np.isinf(x)]             # handful of nan or inf values for outflow
        
        res = stats.ks_2samp(y, x)
        print('KS-test:     %s - %s' %(state_i, state_ii))
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(x) + len(y))/(len(x)*len(y))))))
        print('   p-value: %s' %res.pvalue)
      
        
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    axs.scatter(np.log10(df['BH fraction growth']), np.log10(df['Inflow rate']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH fraction growth']))
        y = np.array(np.log10(df_i['Inflow rate']))
        x = x[~np.isnan(y)]                 # handful of nan or inf values for outflow
        y = y[~np.isnan(y)]
        x = x[~np.isinf(y)]                 # handful of nan or inf values for outflow
        y = y[~np.isinf(y)]

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-5, 1.5)
    #axs.set_yscale('log')
    axs.set_ylim(-1, 2)
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\Delta M_{\mathrm{BH,%s \:Gyr}}/M_{\mathrm{BH,initial}}$' %target_window_size)
    axs.set_ylabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{%s}, %s r_{50}} \rangle _{\mathrm{%s \: Gyr}}$ [M$_{\odot}$ yr$^{-1}$]'%('gas' if gas_fraction_type == 'gas' else 'gas,SF', outflow_hmr, target_window_size))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_outflow/%sbhmass_delta_%s_outflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, outflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_outflow/%sbhmass_delta_%s_outflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, outflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# x-y of average mdot / M_BH vs gas_sf kappa
def _BH_mdotmass_gas_outflow_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                              window_or_snip_mdot = 'snip',       # average growth over window or time-weighted mean over each snip
                            gas_fraction_type                   = 'gas',              # [ 'gas' / 'gas_sf' ]
                            outflow_hmr                          = 1,            # 1 or 2
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    outflow_plot    = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    density_fail = {'gas': {'aligned': [], 'misaligned': [], 'counter': []},
                    'gas_sf': {'aligned': [], 'misaligned': [], 'counter': []}}       # galaxies for which surface density ratio is infinite
    
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s'] + 1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        
                        outflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['outflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', outflow_hmr)])[index_start:index_stop]))
                         
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                  
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    
                    outflow_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['outflow%srate_%shmr'%('_' if gas_fraction_type == 'gas' else '_sf_', outflow_hmr)])[index_start:index_stop]))
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Inflow rate': outflow_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    df['BH mdot growth']     = df['BH mdot snip']/(df['BH mass start'])
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  total:       %s' %len(df['GalaxyIDs']))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    #print('Cannot estimate %s surface ratios for these galaxies:  (remove from above)' %gas_fraction_type)
    #print('  aligned:     %s' %len(density_fail['%s' %gas_fraction_type]['aligned']))
    #print('  misaligned:  %s' %len(density_fail['%s' %gas_fraction_type]['misaligned']))
    #print('  counter:     %s' %len(density_fail['%s' %gas_fraction_type]['counter']))
    
    
    # Medians and KS test
    print('-------')
    if gas_fraction_type == 'gas':
        print('Median gas outflow rate:       [ Msun / yr ]')
    if gas_fraction_type == 'gas_sf':
        print('Median gas_sf outflow rate:       [ Msun / yr ]')
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
        
        y = np.array(np.log10(df_i['Inflow rate']))
        y = y[~np.isnan(y)]             # handful of nan or inf values for outflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for outflow
        print('   %s:       %.4f (log)' %(state_i, np.median(y)))
    #---------------
    # KS test
    print('-------------')
    for state_i, state_ii  in zip(['aligned', 'aligned', 'misaligned'], ['misaligned', 'counter', 'counter']):
        df_i = df.loc[(df['State'] == state_i)]
        df_ii = df.loc[(df['State'] == state_ii)]
        
        y = np.array(df_i['Inflow rate'])
        y = y[~np.isnan(y)]             # handful of nan or inf values for outflow
        y = y[~np.isinf(y)]             # handful of nan or inf values for outflow
        x = np.array(df_ii['Inflow rate'])
        x = x[~np.isnan(x)]             # handful of nan or inf values for outflow
        x = x[~np.isinf(x)]             # handful of nan or inf values for outflow
        
        res = stats.ks_2samp(y, x)
        print('KS-test:     %s - %s' %(state_i, state_ii))
        print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(x) + len(y))/(len(x)*len(y))))))
        print('   p-value: %s' %res.pvalue)
      
        
    
    
    
    #---------------------------  
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.4], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict_scatter = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    color_dict         = {'aligned':'k', 'misaligned':'r', 'counter':'b'}
    
    #---------------------
    ### Plot scatter
    axs.scatter(np.log10(df['BH mdot growth']), np.log10(df['Inflow rate']), s=4, c=[color_dict_scatter[i] for i in df['State']], edgecolor='w', marker='.', linewidths=0.1, alpha=0.25, zorder=-2)
    
    #--------------
    # Contours
    levels = [1-0.68, 1-0.38, 1] # 1 - sigma, as contour will plot 'probability of lying outside this contour', not 'within contour'
    for state_i in ['aligned', 'misaligned', 'counter']:
        df_i = df.loc[(df['State'] == state_i)]
    
        x = np.array(np.log10(df_i['BH mdot growth']))
        y = np.array(np.log10(df_i['Inflow rate']))
        x = x[~np.isnan(y)]                 # handful of nan or inf values for outflow
        y = y[~np.isnan(y)]
        x = x[~np.isinf(y)]                 # handful of nan or inf values for outflow
        y = y[~np.isinf(y)]

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        #levels = [0.25, 0.5, 0.75]

        CS = axs.contour(xi, yi, zi,levels = levels, colors=(lighten_color(color_dict['%s' %state_i], 0.8), lighten_color(color_dict['%s' %state_i], 0.5),), linewidths=(0.8,), origin=origin, zorder=100, alpha=0.8)
        axs.contourf(xi, yi, zi,levels = [levels[0], levels[1]], colors=(lighten_color(color_dict['%s' %state_i], 0.8),), origin=origin, zorder=-3, alpha=0.1)
        axs.contourf(xi, yi, zi,levels = [levels[1], levels[2]], colors=(lighten_color(color_dict['%s' %state_i], 0.5),), origin=origin, zorder=-3, alpha=0.1)
    
    #--------------
    # annotation
    #arr = mpatches.FancyArrowPatch((-11, 0.65), (-12.8, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("reduced growth", (.8, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')
    #arr = mpatches.FancyArrowPatch((-6, 0.65), (-4.2, 0.65), arrowstyle='->,head_width=.15', mutation_scale=6, color='grey')
    #axs.add_patch(arr)
    #axs.annotate("enhanced growth", (.2, 1), xycoords=arr, ha='center', va='bottom', fontsize=6, c='grey')

    #-----------
    ### title
    #plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    #if must_still_be_misaligned:
    #    plot_annotate = plot_annotate + '/trelax'
    #else:
    #    axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(-13, -7.5)
    #axs.set_yscale('log')
    axs.set_ylim(-1, 2)
    #axs.set_yticks([1, 10, 100, 1000])
    axs.set_xlabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{BH}} \rangle _{\mathrm{%s\:Gyr}}/M_{\mathrm{BH,initial}}$ [yr$^{-1}$]' %target_window_size)
    axs.set_ylabel(r'log$_{10}$ $\langle \dot{M}_{\mathrm{%s}, %s r_{50}} \rangle _{\mathrm{%s \: Gyr}}$ [M$_{\odot}$ yr$^{-1}$]'%('gas' if gas_fraction_type == 'gas' else 'gas,SF', outflow_hmr, target_window_size))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['aligned'], 1))
    legend_labels.append('relaxing to stable')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['misaligned'], 1))
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(lighten_color(color_dict['counter'], 1))
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        

    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
               
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        plt.savefig("%s/BH_outflow/%sbhmass_mdot_%s_outflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, outflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_outflow/%sbhmass_mdot_%s_outflow_%sr50_%s%s_clean%s_subsample%s_plot%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, outflow_hmr, ('trelax' if must_still_be_misaligned else 'window'), target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], len(df['GalaxyIDs']), savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#--------------------------------
# check average number of mergers within window
def _count_average_mergers(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                                only_output_df = False,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr since misalignment to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                              must_still_be_misaligned = True,  # target window = target trelax
                            add_histograms    = False,   
                            add_growth_ratios = True,
                          # Morph and fgas
                            use_ETG_LTG            = False,           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                              morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                            select_high_fgas       = False,         # [ False / value ] trim based on average
                          #------------------------------------
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                          
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_snip_plot   = []
    bhmdot_window_plot = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []     # first snip
    kappa_mean_plot = []   # average 
    kappa_snip_plot = []    # all snips
    fgassf_plot    = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    total_merger_count = []
    major_merger_count = []
    minor_merger_count = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                #check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']+1
                
                if must_still_be_misaligned:
                    #time_check = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time']
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_r']]   
                else:
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                            
                            mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                            time_delta_axis = time_axis[1:] - time_axis[0:-1]
                            bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                        fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                        
                        kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                        kappa_mean_plot.append(np.mean(kappa_array))
                        if (kappa_array > 0.4).all():
                            kappa_snip_plot.append('LTG')
                        elif (kappa_array < 0.4).all():
                            kappa_snip_plot.append('ETG')
                        else:
                            kappa_snip_plot.append('other')
                        
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                        
                        # mergers
                        merger_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['merger_ratio_stars'], dtype=object)[index_start:index_stop]
                        merger_array = np.concatenate(merger_array)
                        if len(merger_array) != 0:
                            count_major = len(np.where(merger_array > 0.3)[0])
                            count_minor = len(np.where((merger_array > 0.1) & (merger_array < 0.3))[0])
                            count_total = count_major + count_minor
                        else:
                            count_major = 0
                            count_minor = 0
                            count_total = count_major + count_minor
                        total_merger_count.append(count_total)
                        major_merger_count.append(count_major)
                        minor_merger_count.append(count_minor)
                        
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_window_plot.append((mass_axis[-1] - mass_axis[0])/(time_axis[-1]*(10**9)))
                        
                        mass_delta_axis = mass_axis[1:] - mass_axis[0:-1]
                        time_delta_axis = time_axis[1:] - time_axis[0:-1]
                        bhmdot_snip_plot.append(np.mean(np.divide(mass_delta_axis, time_delta_axis*(10**9))))
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
                    fgassf_plot.append(np.mean(np.divide(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfmass_1hmr'])[index_start:index_stop], np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['gasmass_1hmr'])[index_start:index_stop] + np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass_1hmr'])[index_start:index_stop])))
                    
                    # Array values
                    kappa_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start:index_stop]
                    kappa_mean_plot.append(np.mean(kappa_array))
                    
                    if (kappa_array > 0.4).all():
                        kappa_snip_plot.append('LTG')
                    elif (kappa_array < 0.4).all():
                        kappa_snip_plot.append('ETG')
                    else:
                        kappa_snip_plot.append('other')
                        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    trelax_plot.append(math.nan)
                    
                    # mergers
                    merger_array = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['merger_ratio_stars'], dtype=object)[index_start:index_stop]
                    merger_array = np.concatenate(merger_array)
                    if len(merger_array) != 0:
                        count_major = len(np.where(merger_array > 0.3)[0])
                        count_minor = len(np.where((merger_array > 0.1) & (merger_array < 0.3))[0])
                        count_total = count_major + count_minor
                    else:
                        count_major = 0
                        count_minor = 0
                        count_total = count_major + count_minor
                    total_merger_count.append(count_total)
                    major_merger_count.append(count_major)
                    minor_merger_count.append(count_minor)
                
                                    
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Total mergers': total_merger_count, 'Major mergers': major_merger_count, 'Minor mergers': minor_merger_count, 'Time delta': time_delta_plot, 'trelax': trelax_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'Morphology': kappa_plot, 'Kappa mean': kappa_mean_plot, 'Kappa snip': kappa_snip_plot, 'Gassf fraction': fgassf_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df['BH deltamassmass2']  = df['BH mass delta']/(df['BH mass start']**2)
    df['BH fraction growth'] = df['BH mass delta']/(df['BH mass start'])
    df['BH mdot window']     = bhmdot_window_plot       # delta mass over window / window size
    df['BH mdot snip']       = bhmdot_snip_plot         # time-weighted delta at each step
    
    #-------------------
    # Select only ETG/LTG
    if use_ETG_LTG:
        if use_ETG_LTG == 'ETG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'ETG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] < 0.4)]
        if use_ETG_LTG == 'LTG':
            if morph_average_or_snips == 'snips':
                df_morph = df.loc[(df['Kappa snip'] == 'LTG')]
            elif morph_average_or_snips == 'average':
                df_morph = df.loc[(df['Kappa mean'] > 0.4)]
                
        print('Using sample of %s %ss:      %s' %(morph_average_or_snips, use_ETG_LTG, len(df_morph['GalaxyIDs'])))
        df = df_morph
        df_morph = 0
    # Select only high fgas
    if select_high_fgas:
        df_fgas = df.loc[(df['Gassf fraction'] > select_high_fgas)]
        
        print('Using sample of %s f_gas,SF:      %s' %(select_high_fgas, len(df_fgas['GalaxyIDs'])))
        df = df_fgas
        df_fgas = 0
    
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
       
    # Number of mergers              
    print('\t-----------------')
    print('Mean number of total / major / minor mergers:')
    print('\t\taligned   :    %.3f \t %.3f \t %.3f'%(np.mean(df_co['Total mergers']), np.mean(df_co['Major mergers']), np.mean(df_co['Minor mergers'])))  
    print('\t\tmisaligned:    %.3f \t %.3f \t %.3f'%(np.mean(df_mis['Total mergers']), np.mean(df_mis['Major mergers']), np.mean(df_mis['Minor mergers'])))  
    print('\t\tcounter   :    %.3f \t %.3f \t %.3f'%(np.mean(df_cnt['Total mergers']), np.mean(df_cnt['Major mergers']), np.mean(df_cnt['Minor mergers'])))                    
                          
                          

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Set starting parameters
load_BHmis_tree_in = '_CoPFalse_window0.5_trelax0_05Gyr__no_seed'
plot_annotate_in                  = None
savefig_txt_in     = load_BHmis_tree_in               # [ 'manual' / load_csv_file_in ] 'manual' will prompt txt before saving

# '_CoPFalse_window0.5_trelax0_05Gyr__no_seed'  # <-- default
# '_CoPFalse_window0.5_trelax0_05Gyr___'
# '_CoPTrue_window0.5_trelax0_05Gyr__no_seed'
# '_CoPTrue_window0.5_trelax0_05Gyr___'
# '_CoPFalse_window0.5_trelax0_05Gyr_20Thresh__no_seed'  # <-- default with a 'purer' co- and counter- rotation sample of <20 and >160
# '_CoPFalse_window0.5_trelax0_05Gyr_20Thresh___'
# '_CoPTrue_window0.5_trelax0_05Gyr_20Thresh__no_seed'
# '_CoPTrue_window0.5_trelax0_05Gyr_20Thresh___'
#  False 
# 'ETG → ETG' , 'LTG → LTG'
#  r'ETG ($\bar{\kappa}_{\mathrm{co}}^{\mathrm{*}} < 0.35$)'
# '$t_{\mathrm{relax}}>3\bar{t}_{\mathrm{torque}}$'    

#==================================================================================================================================
# Specify what BH to use in 'refine sample' function
BHmis_tree, BHmis_input, BH_input, BHmis_summary = _extract_BHmis_tree(csv_BHmis_tree=load_BHmis_tree_in, plot_annotate=plot_annotate_in, print_summary=True, EAGLE_dir=EAGLE_dir, sample_dir=sample_dir, tree_dir=tree_dir, output_dir=output_dir, fig_dir=fig_dir, dataDir_dict=dataDir_dict)
#==================================================================================================================================




# x-y of scatter of 2r50 stelmass and bh mass at z=0.1. In projection:
"""_plot_stelmass_BH_scatter(csv_sample = 'L100_27_all_sample_misalignment_9.5', csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                        use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both',
                        use_proj_angle = True,      # False if snip 188 used
                          plot_histogram = True,
                          paper_formatting = True,
                        showfig = True,
                        savefig = True)
_plot_stelmass_BH_scatter(csv_sample = 'L100_27_all_sample_misalignment_9.5', csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                        use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'both',
                        use_proj_angle = True,      # False if snip 188 used
                          paper_formatting = True,
                        showfig = True,
                        savefig = True)
_plot_stelmass_BH_scatter(csv_sample = 'L100_27_all_sample_misalignment_9.5', csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                        use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'both',
                        use_proj_angle = True,      # False if snip 188 used
                          plot_histogram = True,
                          paper_formatting = True,
                        showfig = True,
                        savefig = True)"""
# and as abs abs:
"""_plot_stelmass_BH_scatter(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                        use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both',
                        use_proj_angle = False,      # False if snip 188 used
                          plot_histogram = True,
                          paper_formatting = True,
                        showfig = True,
                        savefig = True)"""
"""_plot_stelmass_BH_scatter(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                        use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'both',
                        use_proj_angle = False,      # False if snip 188 used
                          plot_histogram = True,
                          paper_formatting = True,
                        showfig = True,
                        savefig = True)
_plot_stelmass_BH_scatter(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                        use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'both',
                        use_proj_angle = False,      # False if snip 188 used
                          plot_histogram = True,
                          paper_formatting = True,
                        showfig = True,
                        savefig = True)"""



# x-y of hexbin of 2r50 stelmass and bh mass at z=0.1, but looking at the galaxy's history
"""_plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt2Gyr___',
                                type_of_fraction = 'f_mis',     # f_mis f_cnt f_co
                                plot_only_current_aligned = False,      # galaxies that are aligned at z=0.1
                               showfig = True,
                               savefig = True)"""
"""_plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt4Gyr___',
                                type_of_fraction = 'f_mis',     # f_mis f_cnt f_co
                                plot_only_current_aligned = False,      # galaxies that are aligned at z=0.1
                               showfig = True,
                               savefig = True)
_plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt6Gyr___',
                                type_of_fraction = 'f_mis',     # f_mis f_cnt f_co
                                plot_only_current_aligned = False,
                               showfig = True,
                               savefig = True)
_plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt2Gyr___',
                                type_of_fraction = 'f_cnt',     # f_mis f_cnt f_co
                                plot_only_current_aligned = False,      # galaxies that are aligned at z=0.1
                               showfig = True,
                               savefig = True)
_plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt4Gyr___',
                                type_of_fraction = 'f_cnt',     # f_mis f_cnt f_co
                                plot_only_current_aligned = False,
                               showfig = True,
                               savefig = True)
_plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt6Gyr___',
                                type_of_fraction = 'f_cnt',     # f_mis f_cnt f_co
                                plot_only_current_aligned = False,
                               showfig = True,
                               savefig = True)"""
"""# x-y of hexbin of 2r50 stelmass and bh mass at z=0.1, but looking at the galaxy's cold gas fraction history (fgassf > 0.1)
_plot_stelmass_BH_hexbin_gassf_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt2Gyr___',
                                plot_only_current_aligned = False,      # galaxies that are aligned at z=0.1
                               showfig = True,
                               savefig = True)
_plot_stelmass_BH_hexbin_gassf_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt4Gyr___',
                                plot_only_current_aligned = False,      # galaxies that are aligned at z=0.1
                               showfig = True,
                               savefig = True)
_plot_stelmass_BH_hexbin_gassf_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt6Gyr___',
                                plot_only_current_aligned = False,
                               showfig = True,
                               savefig = True)"""


# STELLAR MASS FUNC export stelmass of the above to check distribution
"""stelmass_output, savefig_output = _plot_stelmass_BH_hexbin_z01_tree(local_z01_tree_load = 'L100_local_z01_input_windowt2Gyr___',
                                type_of_fraction = 'f_mis',     # f_mis f_cnt f_co
                                plot_only_current_aligned = False,      # galaxies that are aligned at z=0.1
                               export_stelmass = True)
_plot_stellar_mass_func(import_stelmass = stelmass_output, savefig_txt_input = savefig_output,
                               showfig = True,
                               savefig = True)"""



                        

# SAMPLE BH mass (RAW)
"""_BH_sample(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                plot_yaxis = ['bhmass', 'sfr', 'ssfr'],        # 'sfr', 'ssfr'
                                  apply_at_start = True,
                                  add_title = True,
                                run_refinement = False,
                                showfig = True,
                                savefig = True)"""
"""_BH_sample(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                plot_yaxis = ['window'],
                                  apply_at_start = True,
                                run_refinement = False,
                                showfig = True,
                                savefig = True)"""
"""_BH_sample(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                plot_yaxis = ['bhmass'],        # 'sfr', 'ssfr'
                                  apply_at_start = True,
                                  add_title = True,
                                run_refinement = False,
                                showfig = True,
                                savefig = False)"""
                                
                              
                                
# STACKED BH growth
"""for target_bh_mass_i in [6.1, 6.3, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8]:
    _BH_stacked_evolution(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    min_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                      target_bhmass         = target_bh_mass_i,              # [ 10**[] Msun / False ]
                                      target_stelmass       = False,                        # [ 10** Msun / False ]
                                    run_refinement = False,
                                      showfig = True,
                                      savefig = False)
for target_bh_mass_i in [6.1, 6.3, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8]:
    _BH_stacked_evolution(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    min_window_size   = 0.75,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                      target_bhmass         = target_bh_mass_i,              # [ 10**[] Msun / False ]
                                      target_stelmass       = False,                        # [ 10** Msun / False ]
                                    run_refinement = False,
                                      showfig = True,
                                      savefig = False)"""

# CHECK MERGER COUNT IN RANGE
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _count_average_mergers(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        run_refinement = False)"""


#------------------------
# Delta growth    
# SCATTER x-y mass at start and mass at end after X Gyr
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = False)
# SCATTER x-y mass at start and mass at end after X Gyr, morphology
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'ETG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'LTG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)                          
# SCATTER x-y mass at start and mass at end after X Gyr, fgas
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        select_high_fgas       = 0.1,         # [ False / value ] trim based on average
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
# SCATTER x-y mass at start and mass at end after X Gyr, max. 10 deg error
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          select_low_err = True,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""



# make sample plot of SCATTER (above) with _seed_
#       USE _seed_ DATA
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    df_out = _BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        only_output_df = True,          # outputs the df
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        run_refinement = False,
                                          showfig = False,
                                          savefig = False)
    _BH_sample(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    existing_df = df_out,
                                    plot_yaxis = ['bhmass'],
                                      apply_at_start = True,
                                    run_refinement = False,
                                    savefig_txt = '_using_analysis_sample_%s_'%str(target_window_size_i),
                                    showfig = True,
                                    savefig = True)"""
                                    

#------------------------
# Average mdot 
# SCATTER x-y mass at start vs average mdot after X Gyr
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdot_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          window_or_snip_mdot = 'snip',     # average growth over window or time-weighted mean over each snip
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
# SCATTER x-y mass at start vs average mdot after X Gyr, using median values to show gradual growth not spike
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdot_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          window_or_snip_mdot = 'snip',     # average growth over window or time-weighted mean over each snip
                                          mean_or_median = 'median',
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = False)
# SCATTER x-y mass at start and mass at end after X Gyr, morphology
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdot_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'ETG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdot_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'LTG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)                           
# SCATTER x-y mass at start and mass at end after X Gyr, fgas
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdot_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        select_high_fgas       = 0.1,         # [ False / value ] trim based on average
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""




#=================================================================
# DELTA M_BH / M_INITIAL

# HIST of fractional growth for a given time window/trelax
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""              
# HIST of fractional growth for a given time window/trelax - morphology
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'ETG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = False)
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'LTG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = False)"""
# HIST of fractional growth for a given time window/trelax - high fgas_sf
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        select_high_fgas       = 0.1,         # [ False / value ] trim based on average
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""

                                          
# SCATTER of fractional growth vs fgas in a given time window/trelax
# Can change initial/average/peak
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_fgas_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'initial',       # ['initial', 'average', 'peak'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_fgas_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'average',       # ['initial', 'average', 'peak'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
                                     
                                     
# SCATTER of fractional growth vs ratio of gas surface densities within r50, and 2r50 in a given time window/trelax
# Can change initial/average/peak
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gassurfratio_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'initial',       # ['initial', 'average', 'delta', 'delta'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gassurfratio_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'average',       # ['initial', 'average', 'delta', 'delta'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gassurfratio_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'peak',       # ['initial', 'average', 'delta', 'delta'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gassurfratio_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'delta',       # ['initial', 'average', 'delta', 'delta'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""

                                     
# SCATTER of fractional growth vs average kappa SF in a given time window/trelax
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gassfkappa_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          initial_or_average = 'initial',
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gassfkappa_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          initial_or_average = 'average',
                                        run_refinement = False,
                                        # select_low_err = True,
                                          showfig = True,
                                          savefig = True)"""




# SCATTER of fractional growth vs inflow rate
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_inflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          inflow_hmr                = 1,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_inflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas_sf',
                                          inflow_hmr                = 1,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_inflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          inflow_hmr                = 2,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_inflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas_sf',
                                          inflow_hmr                = 2,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
                                          

# outflows
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_outflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          outflow_hmr                = 1,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_outflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas_sf',
                                          outflow_hmr                = 1,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_outflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          outflow_hmr                = 2,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_deltamassmass_gas_outflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas_sf',
                                          outflow_hmr                = 2,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""


                      

#-----------------------
# AVERAGE MDOT / M_INITIAL
# HIST of fractional growth for a given time window/trelax
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = False)"""
# HIST of fractional growth for a given time window/trelax - morphology
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'ETG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        use_ETG_LTG            = 'LTG',           # [ False / 'ETG' / 'LTG' ] will go through sample and select based on ETG/LTG
                                          morph_average_or_snips = 'snips',     # [ 'snips' / 'average' ] whether to define the morphology as an average over period or over all snips
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
# HIST of fractional growth for a given time window/trelax - high fgas_sf
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_hist_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                        select_high_fgas       = 0.1,         # [ False / value ] trim based on average
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""

                                          
# SCATTER of fractional growth vs fgas in a given time window/trelax
# Can change initial/average/peak
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_fgas_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'average',       # ['initial', 'average', 'peak'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
                                     
                                     
# SCATTER of fractional growth vs ratio of gas surface densities within r50, and 2r50 in a given time window/trelax
# Can change initial/average/peak
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gassurfratio_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'average',       # ['initial', 'average', 'delta', 'delta'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gassurfratio_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'peak',       # ['initial', 'average', 'delta', 'delta'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gassurfratio_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                           gas_fraction_type  = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                                           initial_or_average = 'delta',       # ['initial', 'average', 'delta', 'delta'] first in window or averaged over window
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""

                                     
# SCATTER of fractional growth vs average kappa SF in a given time window/trelax
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gassfkappa_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          initial_or_average = 'average',
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""


# SCATTER of fractional growth vs inflow rate
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gas_inflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          inflow_hmr                = 1,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gas_inflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          inflow_hmr                = 2,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""
                                          

# outflows
"""for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gas_outflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          outflow_hmr                = 1,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)
for target_window_size_i, window_err_i in zip([0.5, 0.75, 1.0], [0.05, 0.075, 0.1]):
    _BH_mdotmass_gas_outflow_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                        target_window_size   = target_window_size_i,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                          window_err      = window_err_i,           # [ +/- Gyr ] trim
                                          must_still_be_misaligned = True,  # target window = target trelax
                                          gas_fraction_type         = 'gas',
                                          outflow_hmr                = 2,
                                        run_refinement = False,
                                          showfig = True,
                                          savefig = True)"""



                            