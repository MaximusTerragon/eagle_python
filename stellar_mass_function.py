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


# Directories
EAGLE_dir       = '/Users/c22048063/Documents/EAGLE'
dataDir_main    = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
dataDir_alt     = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
output_dir      = EAGLE_dir + '/outputs'
# Directories serpens
#EAGLE_dir       = '/home/user/c22048063/Documents/EAGLE'
#dataDir_main   = '/home/universe/spxtd1-shared/RefL0100N1504/'
#dataDir_alt    = '/home/cosmos/c22048063/EAGLE_snapshots/RefL0100N1504/'
#output_dir      = '/home/cosmos/c22048063/outputs'


# Other directories
sample_dir      = EAGLE_dir + '/samples'
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


#================================
# Will plot sample selection given criteria
# SAVED: /outputs/sample_stellar_mass_function/
def _plot_stellar_mass_function(csv_sample = 'L100_28_all_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                                 #--------------------------
                                 # What determines our final sample
                                 print_summary = True,
                                   pop_mass_limit     = 10**7,            # Lower limit of population plot sampled
                                   sample_mass_limit  = 10**9,            # Lower limit of chosen sample
                                   use_angle          = 'stars_gas_sf',   # Which angles to ensure we have
                                   use_hmr            = 2.0,              # Which HMR ^
                                   use_satellites     = False,   
                                 #--------------------------
                                 hist_bin_width = 0.2,
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
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Lower mass limit: %.2E M*\n  Upper mass limit: %.2E M*\nSatellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_min'], sample_input['galaxy_mass_max'], sample_input['use_satellites']))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Population mass limit: %.2E M*\n  Sample mass limit:     %.2E M*\n  Use satellites:  %s' %(use_angle, use_hmr, pop_mass_limit, sample_mass_limit, use_satellites))
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
    
    # Setting satellite criteria
    if use_satellites:
        satellite_criteria = 99999999
    if not use_satellites:
        satellite_criteria = 0
    
    #=================================================  
    # Plotting
    
    def _plot_stellar_mass_func(debug=False):
        
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[6, 5.5], sharex=True, sharey=False)
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
        
            axs.plot(myData['mass'], np.log10(hist), label=sim_name, linewidth=2, c='indigo')
    
        
        #=========================================
        # Collect values to plot from output_ data
        plot_stelmass = []
    
        if print_progress:
            print('Analysing extracted sample and collecting masses')
            time_start = time.time()
        
        #--------------------------
        # Loop over all galaxies we have available, and analyse output of flags
        for GalaxyID in GalaxyID_List:
            
            # If galaxy is in mass range we want, check whether it is part of sample
            if all_general['%s' %GalaxyID]['stelmass'] >= sample_mass_limit:
                
                # Determine if criteria met. If it is, use stelmass
                if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria):
                    plot_stelmass.append(all_general['%s' %GalaxyID]['stelmass'])
                else:
                    continue
            else:
                continue
                
        if print_summary:
            print('\nPLOT CRITERIA SIZE: %s' %(len(plot_stelmass)))
            
        
        #--------------------------
        # Create histogram of our sample    
        hist_sample, _ = np.histogram((hist_bin_width/2)+np.floor(np.log10(plot_stelmass)/hist_bin_width)*hist_bin_width , bins=np.arange(np.log10(sample_mass_limit)+(hist_bin_width/2), np.log10(10**15), hist_bin_width))
        hist_sample = hist_sample[:] / (float(sample_input['mySims'][0][1]))**3
        hist_sample = hist_sample / hist_bin_width                      # why?
        hist_bins   = np.arange(np.log10(sample_mass_limit)+(hist_bin_width/2), np.log10(10**15)-hist_bin_width, hist_bin_width)
        
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
            
        axs.plot(hist_bins, np.log10(hist_sample), label='Sample selection', ls='--', linewidth=2, c='r')
        
        #-----------
        # Axis formatting
        plt.xlim(7, 12.5)
        plt.ylim(-5, -0.5)
        plt.yticks(np.arange(-5, 0, 0.5))
        plt.xlabel(r'log$_{10}$ M$_{*}$ [M$_{\odot}$]')
        plt.ylabel(r'log$_{10}$ dn/dlog$_{10}$(M$_{*}$) [cMpc$^{-3}$]')
        plt.xticks(np.arange(7, 12.5, 1))
          
        #-----------  
        # Annotations
        
        #-----------
        # Legend
        axs.plot(0, 0, label='${z=%.2f}$' %sample_input['Redshift'], c='k')    # Creating fake plot to add redshift
        axs.legend(loc='upper right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
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
            plt.savefig("%s/sample_stellar_mass_function/L%s_%s_%s_misalignment_%s_m%sm%s_HMR%s_stellar_mass_func_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), np.log10(float(output_input['galaxy_mass_max'])), use_angle, str(use_hmr), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/sample_stellar_mass_function/L%s_%s_%s_misalignment_%s_m%sm%s_HMR%s_stellar_mass_func_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_min'])), np.log10(float(output_input['galaxy_mass_max'])), use_angle, str(use_hmr), savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()

    
    #------------------------
    _plot_stellar_mass_func()
    #------------------------

#===========================
_plot_stellar_mass_function()
#===========================



