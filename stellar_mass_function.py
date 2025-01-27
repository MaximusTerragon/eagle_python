import h5py
import numpy as np
import math
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis
import eagleSqlTools as sql
from graphformat import set_rc_params
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local\n")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================


#================================
# Will plot sample selection given criteria
# SAVED: /outputs/sample_stellar_mass_function/
def _plot_stellar_mass_function(csv_sample = 'L100_27_all_sample_misalignment_9.5',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                                 #--------------------------
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
                           
    #=================================================         
    # Loading sample csv
    if print_progress:
        print('Loading initial sample')
        time_start = time.time()
    
    #-------------------------
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
    #all_sfr             = dict_output['all_sfr']
    #all_Z               = dict_output['all_Z']
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
    
    #=================================================  
    # Plotting
    
    def _plot_stellar_mass_func(debug=False):
        
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
    
        if print_progress:
            print('Analysing extracted sample and collecting masses')
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
        
        
        #=========================================
        # Plotting
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Plotting')
            time_start = time.time()
            
        axs.plot(hist_bins, np.log10(hist_sample), label='Sample selection', ls='--', linewidth=1, c='r')
        
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
        legend_labels.append('Sample selection')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('r')
        legend_labels.append('${z=%.2f}$' %sample_input['Redshift'])
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')

        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        #axs.plot(0, 0, label='${z=%.2f}$' %sample_input['Redshift'], c='k')    # Creating fake plot to add redshift
        #axs.legend(loc='upper right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
        #-----------
        # other
        plt.tight_layout()

        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        metadata_plot = {'Title': 'PLOT CRITERIA SIZE: %s' %len(plot_stelmass)}
        
        if use_satellites:
            sat_str = 'all'
        if not use_satellites:
            sat_str = 'cent'
         
        if savefig:
            plt.savefig("%s/sample_stellar_mass_function/L%s_%s_%s_misalignment_%s_m%sm%s_HMR%s_proj%s_inc%s_stellar_mass_func_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), np.log10(float(output_input['galaxy_mass_max'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/sample_stellar_mass_function/L%s_%s_%s_misalignment_%s_m%sm%s_HMR%s_proj%s_inc%s_stellar_mass_func_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), np.log10(float(output_input['galaxy_mass_max'])), use_angle, str(use_hmr), use_proj_angle, min_inc_angle, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()

    
    #------------------------
    _plot_stellar_mass_func()
    #------------------------


#----------------------------------------
# Will plot stellar mass of a given snip
def _plot_stellar_mass_function_snip(snipshot_in = 188,  
                                    #--------------------------
                                     hist_bin_width = 0.2,
                                     lower_mass_limit = 10**7,
                                     upper_mass_limit = 10**15,
                                     #--------------------------
                                     showfig       = True,
                                     savefig       = False,
                                       file_format = 'pdf',
                                       savefig_txt = '',
                                     #--------------------------
                                     print_progress = False,
                                     debug = False):
                                     
    #================================================   
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    
    #-------------------------
    # Loading sample
    dict_sample = json.load(open('%s/L100_%s_all_sample_misalignment_9.5.csv' %(sample_dir, snipshot_in), 'r'))
    GroupNum_List       = np.array(dict_sample['GroupNum'])
    SubGroupNum_List    = np.array(dict_sample['SubGroupNum'])
    GalaxyID_List       = np.array(dict_sample['GalaxyID'])
    SnapNum_List        = np.array(dict_sample['SnapNum'])
    sample_input        = dict_sample['sample_input']
    
    #---------------------- 
    # Loading mergertree file    
    f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
    GalaxyID_tree             = np.array(f['Histories']['GalaxyID'])
    DescendantID_tree         = np.array(f['Histories']['DescendantID'])
    Redshift_tree             = np.array(f['Snapnum_Index']['Redshift'])
    Lookbacktime_tree         = np.array(f['Snapnum_Index']['LookbackTime'])
    StellarMass_tree          = np.array(f['Histories']['StellarMass'])
    GasMass_tree              = np.array(f['Histories']['GasMass'])
    StarFormationRate_30_tree = np.array(f['Histories']['StarFormationRate_30'])
    f.close()
    
    
    #---------------------- 
    # Mask stellarmass, remove -1
    plot_stelmass = StellarMass_tree[:,int(snipshot_in)][StellarMass_tree[:,int(snipshot_in)] != -1]
    
    
    #--------------------------
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Create histogram of our sample    
    mod_lower_mass_limit = lower_mass_limit
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
    
    
    #-----------
    # Axis formatting
    plt.xlim(np.log10(lower_mass_limit), 12.5)
    plt.ylim(-5, -0.5)
    plt.yticks(np.arange(-5, 0, 0.5))
    plt.xlabel(r'log$_{10}$ M$_{*}$ [M$_{\odot}$]')
    plt.ylabel(r'log$_{10}$ dn/dlog$_{10}$(M$_{*}$) [cMpc$^{-3}$]')
    plt.xticks(np.arange(np.log10(lower_mass_limit), 12.5, 0.5))
    
    plt.show()



#===========================
#_plot_stellar_mass_function()
_plot_stellar_mass_function(use_angle = 'stars_gas_sf', 
                                showfig       = True,
                                savefig       = True)
#_plot_stellar_mass_function(use_angle = 'stars_dm')

#_plot_stellar_mass_function_snip()
#===========================


#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'both')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'both')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'field')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'field')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'ETG', cluster_or_field   = 'cluster')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'LTG', cluster_or_field   = 'cluster')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'cluster')
#_plot_misalignment(use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'field')


