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


# Goes through existing CSV files to find how long misalignments persist for
def _create_misalignment_time_csv(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                 csv_sample2 = '_all_sample_misalignment_9.0',
                                 csv_sample_range = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],   # snapnums
                                 csv_output_in = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                                 #--------------------------
                                 # Galaxy analysis
                                 print_summary = True,
                                   use_angle          = 'stars_gas_sf',         # Which angles to plot
                                   use_hmr            = 2.0,                    # Which HMR to use
                                   use_proj_angle     = False,                   # Whether to use projected or absolute angle 10**9
                                   lower_mass_limit   = 10**9,             # Whether to plot only certain masses 10**15
                                   upper_mass_limit   = 10**15,         
                                   ETG_or_LTG         = 'both',             # Whether to plot only ETG/LTG
                                   group_or_field     = 'both',            # Whether to plot only field/group
                                 #--------------------------
                                 csv_file       = True,             # Will write sample to csv file in sample_dir
                                   csv_name     = '',               # extra stuff at end
                                 #--------------------------
                                 print_progress = False,
                                 debug = False):

    #================================================  
    # Load sample csv
    if print_progress:
        print('Cycling through CSV files and extracting galaxies')
        time_start = time.time()
    
    print('===================')
    print('CSV CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s M*\n  Upper mass limit: %s M*\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
    print('===================\n')
    
    
    #================================================ 
    
    # Creating dictionary to collect all galaxies that meet criteria
    galaxy_dict = {}
    #----------------------
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
        galaxy_dict['%s' %output_input['snapNum']] = {}      
                 
        # Looping over all GalaxyIDs
        for GalaxyID, DescendantID in zip(GalaxyID_List, DescendantID_List):
            
            #-----------------------------
            # Determine if galaxy has flags:
            if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                
                # Determine if galaxy meets criteria:
                if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                    
                    # Mask correct integer (formatting weird but works)
                    mask_rad = int(np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == use_hmr)[0])
                    
                    if use_proj_angle:
                        misangle = all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_rad]
                    else:
                        misangle = all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_rad]
                        
                    # find age
                    Lookbacktime = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(output_input['Redshift'])).value
                    
                    galaxy_dict['%s' %output_input['snapNum']]['%s' %GalaxyID] = {'GalaxyID': GalaxyID, 
                                                                                  'DescendantID': DescendantID, 
                                                                                  'GroupNum': all_general['%s' %GalaxyID]['GroupNum'],
                                                                                  'SubGroupNum': all_general['%s' %GalaxyID]['SubGroupNum'],
                                                                                  'SnapNum': output_input['snapNum'],
                                                                                  'Redshift': output_input['Redshift'],
                                                                                  'Lookbacktime': Lookbacktime,
                                                                                  'misangle': misangle, 
                                                                                  'stelmass': all_general['%s' %GalaxyID]['stelmass'],
                                                                                  'gasmass': all_general['%s' %GalaxyID]['gasmass'],
                                                                                  'gasmass_sf': all_general['%s' %GalaxyID]['gasmass_sf'],
                                                                                  'halfmass_rad': all_general['%s' %GalaxyID]['halfmass_rad'],
                                                                                  'halfmass_rad_proj': all_general['%s' %GalaxyID]['halfmass_rad_proj'],
                                                                                  'kappa_stars': all_general['%s' %GalaxyID]['kappa_stars'],
                                                                                  'kappa_gas_sf': all_general['%s' %GalaxyID]['kappa_gas_sf']}
                     
                     
    #================================================ 
    # Will include unique IDs of all galaxys' misalignment phase (same galaxy cannot be in twice, but can have multiple phases with ID from pre-misalignment)
    timescale_dict = {}
    
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        print('Identifying misalignments and relaxations')
        time_start = time.time()
        
    # Analysing all galaxies
    def _analyse_misangles(debug=False):
           
        # We want to follow each galaxy for teh duration it stays misaligned until it becomes co- or counter-rotating
        # galaxies  that met criteria already in galaxy_dict as ['%GalaxyID']
            
        for SnapNum in galaxy_dict.keys():
            SnapNum = int(SnapNum)
            
            # Ignore last snap (28)
            if SnapNum == csv_sample_range[-1]:
                continue
            
            if debug:
                print('NEW SNAP: ', SnapNum)
            
            for GalaxyID in tqdm(galaxy_dict['%s' %SnapNum].keys()):
        
                GalaxyID     = int(GalaxyID)
                DescendantID = int(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['DescendantID'])
                time_start   = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime'])
                Redshift_start  = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift'])
                SnapNum_start   = int(SnapNum)
                
                # if misaligned, skip as we have already included it or we can't constrain when it started (ei. joined sample misaligned)
                if (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) >= 30):
                    continue
     
                # if descendant met criteria and is root descendant:
                if (str(DescendantID) in galaxy_dict['%s' %(SnapNum+1)]) and (int(DescendantID) == int(GalaxyID)-1):
                    
                    if debug:
                        print('HAS DESCENDANT ', GalaxyID)
                        print('  Current misangle: ', galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle'])
                        print('  Descent misangle: ', galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle'])
     
                    # This will update for as long as galaxy remains misaligned
                    time_end     = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Lookbacktime']
     
                    #------------------------------------
                    # check if this galaxy is aligned and BECOMES misaligned, check for how long and update time_end
                    if (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) < 30) and (float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) >= 30) and (float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) <= 150):
                        if debug:
                            print('BECAME MISALIGNED ', GalaxyID)
                        
                        SnapNum_tmp = SnapNum
                        GalaxyID_tmp = GalaxyID
                        DescendantID_tmp = DescendantID
                        
                        # Creating temporary lists to gather data 
                        GalaxyID_list       = [GalaxyID]
                        DescendantID_list   = [DescendantID]
                        GroupNum_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['GroupNum']]
                        SubGroupNum_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SubGroupNum']]
                        misangle_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']]
                        stelmass_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']]
                        gasmass_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass']]
                        gasmass_sf_list     = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf']]
                        halfmass_rad_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad']]
                        halfmass_rad_proj_list = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad_proj']]
                        kappa_stars_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_stars']]
                        kappa_gas_sf_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_gas_sf']]
                        
                        if debug:
                            print(SnapNum_tmp)
                            print(GalaxyID_tmp)
                            print(DescendantID_tmp)
                            print(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle'])
                        
     
                        # If it becomes misaligned, check for how many subsequent snapshots and append FUTURE descendant stats
                        while (str(DescendantID_tmp) in galaxy_dict['%s' %(SnapNum_tmp+1)]) and (int(DescendantID_tmp) == int(GalaxyID_tmp)-1) and (float(galaxy_dict['%s' %(SnapNum_tmp+1)]['%s' %DescendantID_tmp]['misangle']) >= 30) and (float(galaxy_dict['%s' %(SnapNum_tmp+1)]['%s' %DescendantID_tmp]['misangle']) <= 150) and (int(SnapNum_tmp) <= (csv_sample_range[-1]-1)):
                            
                            # Make descendant new ID, snap, and extract descendant
                            SnapNum_tmp      = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum']
                            GalaxyID_tmp     = galaxy_dict['%s' %(SnapNum_tmp)]['%s' %DescendantID_tmp]['GalaxyID']
                            DescendantID_tmp = galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['DescendantID']
                            
                            if debug:
                                print(SnapNum_tmp)
                                print(GalaxyID_tmp)
                                print(DescendantID_tmp)
                            
                            # Append descendant stats
                            GalaxyID_list.append(GalaxyID_tmp)
                            DescendantID_list.append(DescendantID_tmp)
                            GroupNum_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['GroupNum'])
                            SubGroupNum_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['SubGroupNum'])
                            misangle_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle'])
                            stelmass_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['stelmass'])
                            gasmass_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['gasmass'])
                            gasmass_sf_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['gasmass_sf'])
                            halfmass_rad_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['halfmass_rad'])
                            halfmass_rad_proj_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['halfmass_rad_proj'])
                            kappa_stars_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['kappa_stars'])
                            kappa_gas_sf_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['kappa_gas_sf'])
                            
                            if debug:
                                print(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle'])
                            
     
                            # if galaxy stops being a mainline, or galaxy not in criteria, set time_end = NaN
                            # if it hasn't aligned by the end:
                            if (int(SnapNum_tmp) == 28) and (float(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle']) > 30) and (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle']) <= 150):
                                time_end = math.nan
                                SnapNum_end = math.nan
                                Redshift_end = math.nan
                            else:
                                # if descendant is misaligned but descendant of that does not meet criteria:
                                if ((float(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle']) > 30) and (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle']) <= 150) and (str(DescendantID_tmp) not in galaxy_dict['%s' %(int(SnapNum_tmp)+1)])):
                                    time_end = math.nan
                                    SnapNum_end = math.nan
                                    Redshift_end = math.nan
                                # if descendant is misaligned but stops being mainline:
                                elif ((float(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle']) > 30) and (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle']) <= 150) and (int(DescendantID_tmp) != int(GalaxyID_tmp)-1)):
                                    time_end = math.nan
                                    SnapNum_end = math.nan
                                    Redshift_end = math.nan
                                else:
                                    time_end         = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Lookbacktime']
                                    SnapNum_end      = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum']
                                    Redshift_end     = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Redshift']
                                    misangle_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['misangle'])
                                    stelmass_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['stelmass'])
                                    gasmass_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['gasmass'])
                                    gasmass_sf_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['gasmass_sf'])
                                    halfmass_rad_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['halfmass_rad'])
                                    halfmass_rad_proj_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['halfmass_rad_proj'])
                                    kappa_stars_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['kappa_stars'])
                                    kappa_gas_sf_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['kappa_gas_sf'])
                                
                
                            if (int(SnapNum_tmp) == 28):
                                break
                        #------------------------------------
                        
                        if debug:
                            print('TIME TAKEN TO RELAX: ', SnapNum, SnapNum_end)
                        
                        
                        
                        # time_end will have the last time at which galaxy was misaligned, assuming criteria was met throughout
                        if np.isnan(time_end) == False:
                            timescale_dict['%s' %GalaxyID] = {'GalaxyID': GalaxyID, 
                                                              'DescendantID_list': DescendantID_list,
                                                              'time_start': time_start,
                                                              'time_end': time_end,
                                                              'SnapNum_start': SnapNum_start,
                                                              'SnapNum_end': SnapNum_end,
                                                              'Redshift_start': Redshift_start,
                                                              'Redshift_end': Redshift_end,
                                                              'misangle_list': misangle_list,
                                                              'GroupNum_list': GroupNum_list,
                                                              'SubGroupNum_list': SubGroupNum_list,
                                                              'misangle_list': misangle_list,
                                                              'stelmass_list': stelmass_list,
                                                              'gasmass_list': gasmass_list,
                                                              'gasmass_sf_list': gasmass_sf_list,
                                                              'kappa_stars_list': kappa_stars_list,
                                                              'kappa_gas_sf_list': kappa_gas_sf_list}
                                        
     
                    #------------------------------------
                    # check if galaxy is aligned and immediately is counter-aligned, assume it relaxed between time_start and time_end
                    elif (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) < 30) and (float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) > 150):
                        
                        if debug:
                            print('BECAME COUNTERALIGNED ', GalaxyID)
                        
                        # Creating temporary lists to gather data 
                        GalaxyID_list       = [GalaxyID, DescendantID]
                        DescendantID_list   = [DescendantID, galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['DescendantID']]
                        GroupNum_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['GroupNum'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['GroupNum']]
                        SubGroupNum_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SubGroupNum'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['SubGroupNum']]
                        misangle_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']]
                        stelmass_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['stelmass']]
                        gasmass_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['gasmass']]
                        gasmass_sf_list     = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['gasmass_sf']]
                        halfmass_rad_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['halfmass_rad']]
                        halfmass_rad_proj_list = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad_proj'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['halfmass_rad_proj']]
                        kappa_stars_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_stars'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['kappa_stars']]
                        kappa_gas_sf_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_gas_sf'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['kappa_gas_sf']]
                        
                        # Make descendant new ID, snap, and extract descendant
                        time_end         = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Lookbacktime']
                        SnapNum_end      = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['SnapNum']
                        Redshift_end     = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Redshift']
                        
                        if debug:
                            print('TIME TAKEN TO RELAX: ', SnapNum, SnapNum_end)
 
                        #------------------------------------
                        # Appending to dictionary
                        timescale_dict['%s' %GalaxyID] = {'GalaxyID': GalaxyID, 
                                                          'DescendantID_list': DescendantID_list,
                                                          'time_start': time_start,
                                                          'time_end': time_end,
                                                          'SnapNum_start': SnapNum_start,
                                                          'SnapNum_end': SnapNum_end,
                                                          'Redshift_start': Redshift_start,
                                                          'Redshift_end': Redshift_end,
                                                          'misangle_list': misangle_list,
                                                          'GroupNum_list': GroupNum_list,
                                                          'SubGroupNum_list': SubGroupNum_list,
                                                          'misangle_list': misangle_list,
                                                          'stelmass_list': stelmass_list,
                                                          'gasmass_list': gasmass_list,
                                                          'gasmass_sf_list': gasmass_sf_list,
                                                          'kappa_stars_list': kappa_stars_list,
                                                          'kappa_gas_sf_list': kappa_gas_sf_list}
                        
                        
                    #------------------------------------
                    # if galaxy stays aligned, or stops being mainline, or stops meeting criteria, ignore
                    else:
                       continue                 
    #-------------------
    _analyse_misangles()
    #-------------------  
    
    if print_summary:
        print('===================')
        print('NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
        print('===================')
        
    
    #================================================ 
    if csv_file: 
        # Converting numpy arrays to lists. When reading, may need to simply convert list back to np.array() (easy)
        class NumpyEncoder(json.JSONEncoder):
            ''' Special json encoder for numpy types '''
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
                  
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Writing to csv...')
            time_start = time.time() 
        
        # Combining all dictionaries
        csv_dict = {'timescale_dict': timescale_dict}
        output_input = {'csv_sample1': csv_sample1,
                        'csv_sample_range': csv_sample_range,
                        'csv_sample2': csv_sample2,
                        'csv_output_in': csv_output_in,
                        'use_angle': use_angle,
                        'use_hmr': use_hmr,
                        'use_proj_angle': use_proj_angle,
                        'lower_mass_limit': lower_mass_limit,
                        'upper_mass_limit': upper_mass_limit,
                        'ETG_or_LTG': ETG_or_LTG,
                        'group_or_field': group_or_field,
                        'mySims': sample_input['mySims']}
                        
        csv_dict.update({'output_input': output_input})
        
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%stimescale_tree_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, csv_sample1, ETG_or_LTG, use_angle, use_hmr, use_proj_angle, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%stimescale_tree_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, csv_sample1, ETG_or_LTG, use_angle, use_hmr, use_proj_angle, csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
        # Reading JSON file
        """ 
        # Ensuring the sample and output originated together
        csv_timescales = ...
        
        # Loading sample
        dict_timetree = json.load(open('%s/%s.csv' %(sample_dir, csv_timescales), 'r'))
        timescale_dict  = dict_timetree['timescale_dict']
        
        # Loading sample criteria
        timescale_input = dict_timetree['output_input']
    
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print('NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
   
        print('\n===================')
        print('TIMESCALES LOADED:\n %s\n  Snapshots: %s\n  Angle type: %s\n  Angle HMR: %s\n  Projected angle: %s' %(output_input['mySims'][0][0]}, output_input['csv_sample_range'], output_input['use_angle'], output_input['use_hmr'], output_input['use_proj_angle']))
        print('  NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
        
        
        print('\nPLOT:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s\n  Upper mass limit: %s\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
        print('===================')
        """
        

# Goes through existing CSV files to find how long misalignments persist for
def _create_misalignment_merger_csv(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                    csv_sample2 = '_all_sample_misalignment_9.0',
                                    csv_sample_range = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],   # snapnums
                                    csv_output_in = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                                    csv_mergertree = 'L100_merger_tree_',
                                    #--------------------------
                                    # Galaxy analysis
                                    print_summary = True,
                                      use_angle          = 'stars_gas_sf',         # Which angles to plot
                                      use_hmr            = 2.0,                    # Which HMR to use
                                      use_proj_angle     = False,                   # Whether to use projected or absolute angle 10**9
                                      lower_mass_limit   = 10**9,             # Whether to plot only certain masses 10**15
                                      upper_mass_limit   = 10**15,         
                                      ETG_or_LTG         = 'both',             # Whether to plot only ETG/LTG
                                      group_or_field     = 'both',            # Whether to plot only field/group
                                    #--------------------------
                                    # Merger analysis
                                    merger_threshold_min = 0.1,             # >= to include
                                    merger_threshold_max = 1.0,             # <= to include
                                    #--------------------------
                                    csv_file       = True,             # Will write sample to csv file in sample_dir
                                      csv_name     = '',               # extra stuff at end
                                    #--------------------------
                                    print_progress = False,
                                    debug = False):

    #=====================================================================================  
    # Load sample csv
    if print_progress:
        print('Cycling through CSV files and extracting galaxies')
        time_start = time.time()
    
    print('===================')
    print('CSV CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s M*\n  Upper mass limit: %s M*\n  ETG or LTG: %s\n  Group or field: %s\n  Mergers min/max: %s | %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field, merger_threshold_min, merger_threshold_max))
    print('===================\n')
    
    
    #=====================================================================================
    # Creating dictionary to collect all galaxies that meet criteria
    galaxy_dict = {}
    
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
        galaxy_dict['%s' %output_input['snapNum']] = {}      
                 
        # Looping over all GalaxyIDs
        for GalaxyID, DescendantID in zip(GalaxyID_List, DescendantID_List):
            
            #-----------------------------
            # Determine if galaxy has flags:
            if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                
                # Determine if galaxy meets criteria:
                if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                    
                    # Mask correct integer (formatting weird but works)
                    mask_rad = int(np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == use_hmr)[0])
                    
                    if use_proj_angle:
                        misangle = all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_rad]
                    else:
                        misangle = all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_rad]
                        
                    # find age
                    Lookbacktime = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(output_input['Redshift'])).value
                    
                    galaxy_dict['%s' %output_input['snapNum']]['%s' %GalaxyID] = {'GalaxyID': GalaxyID, 
                                                                                  'DescendantID': DescendantID, 
                                                                                  'GroupNum': all_general['%s' %GalaxyID]['GroupNum'],
                                                                                  'SubGroupNum': all_general['%s' %GalaxyID]['SubGroupNum'],
                                                                                  'SnapNum': output_input['snapNum'],
                                                                                  'Redshift': output_input['Redshift'],
                                                                                  'Lookbacktime': Lookbacktime,
                                                                                  'misangle': misangle, 
                                                                                  'stelmass': all_general['%s' %GalaxyID]['stelmass'],
                                                                                  'gasmass': all_general['%s' %GalaxyID]['gasmass'],
                                                                                  'gasmass_sf': all_general['%s' %GalaxyID]['gasmass_sf'],
                                                                                  'halfmass_rad': all_general['%s' %GalaxyID]['halfmass_rad'],
                                                                                  'halfmass_rad_proj': all_general['%s' %GalaxyID]['halfmass_rad_proj'],
                                                                                  'kappa_stars': all_general['%s' %GalaxyID]['kappa_stars'],
                                                                                  'kappa_gas_sf': all_general['%s' %GalaxyID]['kappa_gas_sf']}
                     
                     
                     
    #=====================================================================================
    # Load merger tree csv
    if print_progress:
        print('Loading initial sample')
        time_start = time.time()


    # Loading merger tree
    merger_tree_csv = json.load(open('%s/%s.csv' %(output_dir, csv_mergertree), 'r'))
    merger_tree        = merger_tree_csv['tree_dict']     
    merger_tree_output = merger_tree_csv['output_input']
    
    
    # Will include unique IDs of all galaxys' misalignment phase (same galaxy cannot be in twice, but can have multiple phases with ID from pre-misalignment)
    timescale_dict = {}
    
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        print('Identifying misalignments and relaxations')
        time_start = time.time()
        
    
    # Analysing all galaxies
    def _analyse_merger_misangles(debug=False):
           
        # We want to follow each galaxy for teh duration it stays misaligned until it becomes co- or counter-rotating
        # galaxies  that met criteria already in galaxy_dict as ['%GalaxyID']
            
        for SnapNum in galaxy_dict.keys():
            SnapNum = int(SnapNum)
            
            # Ignore last snap (28)
            if SnapNum == csv_sample_range[-1]:                
                continue
            
            if debug:
                print('NEW SNAP: ', SnapNum)
            
            for GalaxyID in tqdm(galaxy_dict['%s' %SnapNum].keys()):
        
                GalaxyID     = int(GalaxyID)
                DescendantID = int(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['DescendantID'])
                time_start   = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime'])
                Redshift_start  = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift'])
                SnapNum_start   = int(SnapNum)
                
                # if misaligned, skip as we have already included it or we can't constrain when it started (ei. joined sample misaligned)
                if (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) >= 30):
                    continue
     
                # if descendant met criteria and is root descendant:
                if (str(DescendantID) in galaxy_dict['%s' %(SnapNum+1)]) and (int(DescendantID) == int(GalaxyID)-1):
                    
                    if debug:
                        print('HAS DESCENDANT ', GalaxyID)
                        print('  Current misangle: ', galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle'])
                        print('  Descent misangle: ', galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle'])
     
                    # This will update for as long as galaxy remains misaligned
                    time_end     = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Lookbacktime']
     
                    #------------------------------------
                    # check if this galaxy is aligned and BECOMES misaligned, AND has undergone a merger (more than one constituent galaxy)
                    if (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) < 30) and (float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) >= 30) and (float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) <= 150) and (len(merger_tree['%s' %DescendantID]) > 1):
                        
                        # Check for major or minor merger (there may be multiple)
                        merger_ratio_array       = []
                        merger_gas_ratio_array   = []
                        merger_gassf_ratio_array = []
                        merger_id_array          = []
                        
                        
                        #print('GALAXYID: %i     %.2f       %s' %(GalaxyID, np.log10(float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass'])), SnapNum))
                        #print('  DescendantID: ', DescendantID)
                        #print('  merger tree: ', merger_tree['%s' %DescendantID].keys())
                        
                        
                        for component_GalaxyID in merger_tree['%s' %DescendantID].keys():
                            
                            #print('     componentID: %s     %.2f' %(component_GalaxyID, np.log10(float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass']))))
                            
                            
                            # If main line progenitor, ignore
                            if int(DescendantID) == (int(component_GalaxyID) - 1):
                                continue
                            
                            # Find stellar mass merger ratio
                            merger_ratio = float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass'])/float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass'])
                            if merger_ratio > 1.0:
                                merger_ratio = 1 / merger_ratio
                            
                            # Find gas ratios
                            gas_ratio    = (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['gasmass'])) / (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass']))
                            gassf_ratio  = (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['gasmass_sf'])) / (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass']))
                            
                            # Append to lists
                            merger_ratio_array.append(float(merger_ratio))
                            merger_gas_ratio_array.append(float(gas_ratio))
                            merger_gassf_ratio_array.append(float(gassf_ratio))
                            merger_id_array.append(int(component_GalaxyID))
                            merger_snap = merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['SnapNum']
                            merger_age  = merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['Lookbacktime']
                            merger_z    = merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['Redshift']
                        
                        
                        #------------------------------------
                        # If mergers correspond to the limits imposed    
                        if (np.array(merger_ratio_array).max() >= merger_threshold_min) and (np.array(merger_ratio_array).max() <= merger_threshold_max):
                            
                            if debug:
                                print('BECAME MISALIGNED ', GalaxyID)
                        
                            SnapNum_tmp = SnapNum
                            GalaxyID_tmp = GalaxyID
                            DescendantID_tmp = DescendantID
                            
                            # Creating temporary lists to gather data 
                            GalaxyID_list       = [GalaxyID]
                            DescendantID_list   = [DescendantID]
                            GroupNum_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['GroupNum']]
                            SubGroupNum_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SubGroupNum']]
                            misangle_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']]
                            stelmass_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']]
                            gasmass_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass']]
                            gasmass_sf_list     = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf']]
                            halfmass_rad_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad']]
                            halfmass_rad_proj_list = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad_proj']]
                            kappa_stars_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_stars']]
                            kappa_gas_sf_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_gas_sf']]
                            SnapNum_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SnapNum']]
                            Lookbacktime_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime']]
                            Redshift_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift']]
                            
                            # Get ID of largest merger | Merger lists already exist under merger_ratio_list, merger_gas_ratio_list, merger_gassf_ratio_list, merger_id_list
                            MergerID = int(merger_id_array[np.array(merger_ratio_array).argmax()])
                            
                            
                            #print('     PASSED GalaxyID: %s' %GalaxyID)
                            #print('     PASSED MergerID: %s' %MergerID)
                            #print('       ', merger_id_array)
                            #print('       ', merger_ratio_array)
                            #print('       ', merger_snap)
                            
                            
                            merger_ratio_list       = [merger_ratio_array]
                            merger_gas_ratio_list   = [merger_gas_ratio_array]
                            merger_gassf_ratio_list = [merger_gassf_ratio_array]
                            merger_id_list          = [merger_id_array]
                            merger_snap_list        = [merger_snap]
                            merger_age_list         = [merger_age]
                            merger_z_list           = [merger_z]
                            
                            
                            if debug:
                                print(SnapNum_tmp)
                                print(GalaxyID_tmp)
                                print(DescendantID_tmp)
                                print(merger_ratio_list)
                                print(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle'])
                        
                            #------------------------------------
                            # If it becomes misaligned, check for how many subsequent snapshots and append FUTURE descendant stats
                            while (str(DescendantID_tmp) in galaxy_dict['%s' %(SnapNum_tmp+1)]) and (int(DescendantID_tmp) == int(GalaxyID_tmp)-1) and (float(galaxy_dict['%s' %(SnapNum_tmp+1)]['%s' %DescendantID_tmp]['misangle']) >= 30) and (float(galaxy_dict['%s' %(SnapNum_tmp+1)]['%s' %DescendantID_tmp]['misangle']) <= 150) and (int(SnapNum_tmp) <= (csv_sample_range[-1]-1)):
                            
                                # Make descendant new ID, snap, and extract descendant
                                SnapNum_tmp      = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum']
                                GalaxyID_tmp     = galaxy_dict['%s' %(SnapNum_tmp)]['%s' %DescendantID_tmp]['GalaxyID']
                                DescendantID_tmp = galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['DescendantID']
                            
                                if debug:
                                    print(SnapNum_tmp)
                                    print(GalaxyID_tmp)
                                    print(DescendantID_tmp)
                            
                                #----------------
                                # Append descendant stats
                                GalaxyID_list.append(GalaxyID_tmp)
                                DescendantID_list.append(DescendantID_tmp)
                                GroupNum_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['GroupNum'])
                                SubGroupNum_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['SubGroupNum'])
                                misangle_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle'])
                                stelmass_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['stelmass'])
                                gasmass_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['gasmass'])
                                gasmass_sf_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['gasmass_sf'])
                                halfmass_rad_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['halfmass_rad'])
                                halfmass_rad_proj_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['halfmass_rad_proj'])
                                kappa_stars_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['kappa_stars'])
                                kappa_gas_sf_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['kappa_gas_sf'])
                                SnapNum_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['SnapNum'])
                                Lookbacktime_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['Lookbacktime'])
                                Redshift_list.append(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['Redshift'])
                                
                                #-----------------
                                # Append any mergers (if there are any)
                                if len(merger_tree['%s' %DescendantID_tmp]) > 1:
                        
                                    # Check for major or minor merger (there may be multiple)
                                    merger_ratio_array       = []
                                    merger_gas_ratio_array   = []
                                    merger_gassf_ratio_array = []
                                    merger_id_array          = []
                        
                                    for component_GalaxyID in merger_tree['%s' %DescendantID_tmp].keys():
                                        # If main line progenitor, ignore
                                        if int(DescendantID) == (int(component_GalaxyID) - 1):
                                            continue
                            
                                        # Find stellar mass merger ratio
                                        merger_ratio = float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['stelmass'])/float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['stelmass'])
                                        if merger_ratio > 1.0:
                                            merger_ratio = 1 / merger_ratio
                            
                                        # Find gas ratios
                                        gas_ratio    = (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass']) + float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['gasmass'])) / (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['stelmass']) + float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['stelmass']))
                                        gassf_ratio  = (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass_sf']) + float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['gasmass_sf'])) / (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['stelmass']) + float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['stelmass']))
                            
                                        # Append to lists
                                        merger_ratio_array.append(float(merger_ratio))
                                        merger_gas_ratio_array.append(float(gas_ratio))
                                        merger_gassf_ratio_array.append(float(gassf_ratio))
                                        merger_id_array.append(int(component_GalaxyID))
                                        merger_snap = merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['SnapNum']
                                        merger_age  = merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['Lookbacktime']
                                        merger_z    = merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['Redshift']
                                
                                    # Append merger stats
                                    merger_ratio_list.append(merger_ratio_array)
                                    merger_gas_ratio_list.append(merger_gas_ratio_array)
                                    merger_gassf_ratio_list.append(merger_gassf_ratio_array)
                                    merger_id_list.append(merger_id_array)
                                    merger_snap_list.append(merger_snap)
                                    merger_age_list.append(merger_age)
                                    merger_z_list.append(merger_z)
                                
                                if debug:
                                    print(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle'])
                                    print(merger_ratio_array)
                            
                                
                                #----------------
                                # if galaxy stops being a mainline, or galaxy not in criteria, set time_end = NaN
                                # if it hasn't aligned by the end:
                                if (int(SnapNum_tmp) == 28) and (float(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle']) > 30) and (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle']) <= 150):
                                    time_end = math.nan
                                    SnapNum_end = math.nan
                                    Redshift_end = math.nan
                                    
                                    break
                                else:
                                    # if descendant is misaligned but descendant of that does not meet criteria:
                                    if ((float(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle']) > 30) and (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle']) <= 150) and (str(DescendantID_tmp) not in galaxy_dict['%s' %(int(SnapNum_tmp)+1)])):
                                        time_end = math.nan
                                        SnapNum_end = math.nan
                                        Redshift_end = math.nan
                                        
                                        break
                                    # if descendant is misaligned but stops being mainline:
                                    elif ((float(galaxy_dict['%s' %(SnapNum_tmp)]['%s' %GalaxyID_tmp]['misangle']) > 30) and (float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle']) <= 150) and (int(DescendantID_tmp) != int(GalaxyID_tmp)-1)):
                                        time_end = math.nan
                                        SnapNum_end = math.nan
                                        Redshift_end = math.nan
                                        
                                        break
                                    else:
                                        # Make NEXT Descendant the last
                                        time_end         = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Lookbacktime']
                                        SnapNum_end      = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum']
                                        Redshift_end     = galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Redshift']
                                        
                                        # Add stats of next descendant
                                        misangle_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['misangle'])
                                        stelmass_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['stelmass'])
                                        gasmass_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['gasmass'])
                                        gasmass_sf_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['gasmass_sf'])
                                        halfmass_rad_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['halfmass_rad'])
                                        halfmass_rad_proj_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['halfmass_rad_proj'])
                                        kappa_stars_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['kappa_stars'])
                                        kappa_gas_sf_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['kappa_gas_sf'])
                                        SnapNum_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum'])
                                        Lookbacktime_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Lookbacktime'])
                                        Redshift_list.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Redshift'])
                                        
                                        break
                                    
                                        
                                        
                                        
                                        
                                        
                                
                                        
                                        
                                if (int(SnapNum_tmp) == 28):
                                    break
                            #------------------------------------
                        
                            if debug:
                                print('TIME TAKEN TO RELAX: ', SnapNum, SnapNum_end)
                        
                        
                            # time_end will have the last time at which galaxy was misaligned, assuming criteria was met throughout
                            if np.isnan(time_end) == False:
                                timescale_dict['%s' %GalaxyID] = {'GalaxyID_list': GalaxyID_list, 
                                                                  'DescendantID_list': DescendantID_list,
                                                                  'time_start': time_start,
                                                                  'time_end': time_end,
                                                                  'SnapNum_start': SnapNum_start,
                                                                  'SnapNum_end': SnapNum_end,
                                                                  'Redshift_start': Redshift_start,
                                                                  'Redshift_end': Redshift_end,
                                                                  'misangle_list': misangle_list,
                                                                  'GroupNum_list': GroupNum_list,
                                                                  'SubGroupNum_list': SubGroupNum_list,
                                                                  'misangle_list': misangle_list,
                                                                  'stelmass_list': stelmass_list,
                                                                  'gasmass_list': gasmass_list,
                                                                  'gasmass_sf_list': gasmass_sf_list,
                                                                  'kappa_stars_list': kappa_stars_list,
                                                                  'kappa_gas_sf_list': kappa_gas_sf_list,
                                                                  'SnapNum_list': SnapNum_list,
                                                                  'Lookbacktime_list': Lookbacktime_list,
                                                                  'Redshift_list': Redshift_list,
                                                                  'merger_ratio_list': merger_ratio_list,
                                                                  'merger_gas_ratio_list': merger_gas_ratio_list,
                                                                  'merger_gassf_ratio_list': merger_gassf_ratio_list,
                                                                  'merger_id_list': merger_id_list,
                                                                  'merger_snap_list': merger_snap_list,
                                                                  'merger_age_list': merger_age_list,
                                                                  'merger_z_list': merger_z_list}
                        
                        
                    #------------------------------------
                    # check if galaxy is aligned and immediately is counter-aligned, assume it relaxed between time_start and time_end
                    elif (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) < 30) and (float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) > 150) and (len(merger_tree['%s' %DescendantID]) > 1):
                        
                        # Check for major or minor merger (there may be multiple)
                        merger_ratio_array       = []
                        merger_gas_ratio_array   = []
                        merger_gassf_ratio_array = []
                        merger_id_array          = []
                        
                        for component_GalaxyID in merger_tree['%s' %DescendantID].keys():
                            # If main line progenitor, ignore
                            if int(DescendantID) == (int(component_GalaxyID) - 1):
                                continue
                            
                            # Find stellar mass merger ratio
                            merger_ratio = float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass'])/float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass'])
                            if merger_ratio > 1.0:
                                merger_ratio = 1 / merger_ratio
                            
                            # Find gas ratios
                            gas_ratio    = (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['gasmass'])) / (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass']))
                            gassf_ratio  = (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['gasmass_sf'])) / (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']) + float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass']))
                            
                            # Append to lists
                            merger_ratio_array.append(float(merger_ratio))
                            merger_gas_ratio_array.append(float(gas_ratio))
                            merger_gassf_ratio_array.append(float(gassf_ratio))
                            merger_id_array.append(int(component_GalaxyID))
                            merger_snap = merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['SnapNum']
                            merger_age  = merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['Lookbacktime']
                            merger_z    = merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['Redshift']
                        
        
                        #------------------------------------
                        # If mergers correspond to the limits imposed    
                        if (np.array(merger_ratio_array).max() >= merger_threshold_min) and (np.array(merger_ratio_array).max() <= merger_threshold_max):
                            
                            if debug:
                                print('BECAME COUNTERALIGNED ', GalaxyID)
                        
                            #-----------------
                            # Creating temporary lists to gather data 
                            GalaxyID_list       = [GalaxyID, DescendantID]
                            DescendantID_list   = [DescendantID, galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['DescendantID']]
                            GroupNum_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['GroupNum'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['GroupNum']]
                            SubGroupNum_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SubGroupNum'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['SubGroupNum']]
                            misangle_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']]
                            stelmass_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['stelmass']]
                            gasmass_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['gasmass']]
                            gasmass_sf_list     = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['gasmass_sf']]
                            halfmass_rad_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['halfmass_rad']]
                            halfmass_rad_proj_list = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad_proj'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['halfmass_rad_proj']]
                            kappa_stars_list    = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_stars'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['kappa_stars']]
                            kappa_gas_sf_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_gas_sf'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['kappa_gas_sf']]
                            SnapNum_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SnapNum'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['SnapNum']]
                            Lookbacktime_list   = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Lookbacktime']]
                            Redshift_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift'], galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Redshift']]
                            merger_ratio_list       = [merger_ratio_array]
                            merger_gas_ratio_list   = [merger_gas_ratio_array]
                            merger_gassf_ratio_list = [merger_gassf_ratio_array]
                            merger_id_list          = [merger_id_array]
                            merger_snap_list        = [merger_snap]
                            merger_age_list         = [merger_age]
                            merger_z_list           = [merger_z]
                        
                            #-----------------
                            # Make descendant new ID, snap, and extract descendant
                            time_end         = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Lookbacktime']
                            SnapNum_end      = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['SnapNum']
                            Redshift_end     = galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['Redshift']
                        
                            if debug:
                                print('TIME TAKEN TO RELAX: ', SnapNum, SnapNum_end)
 
                            #------------------------------------
                            # Appending to dictionary
                            timescale_dict['%s' %GalaxyID] = {'GalaxyID_list': GalaxyID_list, 
                                                              'DescendantID_list': DescendantID_list,
                                                              'time_start': time_start,
                                                              'time_end': time_end,
                                                              'SnapNum_start': SnapNum_start,
                                                              'SnapNum_end': SnapNum_end,
                                                              'Redshift_start': Redshift_start,
                                                              'Redshift_end': Redshift_end,
                                                              'misangle_list': misangle_list,
                                                              'GroupNum_list': GroupNum_list,
                                                              'SubGroupNum_list': SubGroupNum_list,
                                                              'misangle_list': misangle_list,
                                                              'stelmass_list': stelmass_list,
                                                              'gasmass_list': gasmass_list,
                                                              'gasmass_sf_list': gasmass_sf_list,
                                                              'kappa_stars_list': kappa_stars_list,
                                                              'kappa_gas_sf_list': kappa_gas_sf_list,
                                                              'SnapNum_list': SnapNum_list,
                                                              'Lookbacktime_list': Lookbacktime_list,
                                                              'Redshift_list': Redshift_list,
                                                              'merger_ratio_list': merger_ratio_list,
                                                              'merger_gas_ratio_list': merger_gas_ratio_list,
                                                              'merger_gassf_ratio_list': merger_gassf_ratio_list,
                                                              'merger_id_list': merger_id_list,
                                                              'merger_snap_list': merger_snap_list,
                                                              'merger_age_list': merger_age_list,
                                                              'merger_z_list': merger_z_list}
                        
                         
                    #------------------------------------
                    # if galaxy stays aligned, or stops being mainline, or stops meeting criteria, ignore
                    else:
                       continue                 
    
    #-------------------
    _analyse_merger_misangles()
    #-------------------  
    
    
    if print_summary:
        print('===================')
        print('NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
        print('===================')
        
    
    #=====================================================================================
    if csv_file: 
        # Converting numpy arrays to lists. When reading, may need to simply convert list back to np.array() (easy)
        class NumpyEncoder(json.JSONEncoder):
            ''' Special json encoder for numpy types '''
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
                  
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Writing to csv...')
            time_start = time.time() 
        
        # Combining all dictionaries
        csv_dict = {'timescale_dict': timescale_dict}
        output_input = {'csv_sample1': csv_sample1,
                        'csv_sample_range': csv_sample_range,
                        'csv_sample2': csv_sample2,
                        'csv_output_in': csv_output_in,
                        'use_angle': use_angle,
                        'use_hmr': use_hmr,
                        'use_proj_angle': use_proj_angle,
                        'lower_mass_limit': lower_mass_limit,
                        'upper_mass_limit': upper_mass_limit,
                        'ETG_or_LTG': ETG_or_LTG,
                        'group_or_field': group_or_field,
                        'merger_threshold_min': merger_threshold_min,
                        'merger_threshold_max': merger_threshold_max,
                        'mySims': sample_input['mySims']}
                        
        csv_dict.update({'output_input': output_input})
        
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%smerger_timescale_tree_%s_r%sr%s_%s_rad%s_proj%s_%s.csv' %(output_dir, csv_sample1, ETG_or_LTG, merger_threshold_min, merger_threshold_max, use_angle, use_hmr, use_proj_angle, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%smerger_timescale_tree_%s_r%sr%s_%s_rad%s_proj%s_%s.csv' %(output_dir, csv_sample1, ETG_or_LTG, merger_threshold_min, merger_threshold_max, use_angle, use_hmr, use_proj_angle, csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
        # Reading JSON file
        """ 
        # Ensuring the sample and output originated together
        csv_timescales = ...
        
        # Loading sample
        dict_timetree = json.load(open('%s/%s.csv' %(sample_dir, csv_timescales), 'r'))
        timescale_dict  = dict_timetree['timescale_dict']
        
        # Loading sample criteria
        timescale_input = dict_timetree['output_input']
    
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print('NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
   
        print('\n===================')
        print('TIMESCALES LOADED:\n %s\n  Snapshots: %s\n  Angle type: %s\n  Angle HMR: %s\n  Projected angle: %s' %(output_input['mySims'][0][0]}, output_input['csv_sample_range'], output_input['use_angle'], output_input['use_hmr'], output_input['use_proj_angle']))
        print('  NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
        
        
        print('\nPLOT:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s\n  Upper mass limit: %s\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
        print('===================')
        """
        



def _plot_merger_timescales(csv_timescales = 'L100_merger_timescale_tree_both_r0.1r1.0_stars_gas_sf_rad2.0_projFalse_'):
    
    
    # Loading sample
    dict_timetree = json.load(open('%s/%s.csv' %(output_dir, csv_timescales), 'r'))
    timescale_dict  = dict_timetree['timescale_dict']
    
    # Loading sample criteria
    timescale_input = dict_timetree['output_input']
    
    
    
    #print(timescale_dict.keys())
    
    
    
    
    
    
    #stelmass_array = []
    #for galaxyid in timescale_dict.keys():
    #    stelmass_array.append(np.log10(float(timescale_dict['%s' %galaxyid]['stelmass_list'][0])))
    #    
    #plt.hist(stelmass_array, bins=50, range=(8.0, 13))
    #plt.show()    

    for galaxyid in tqdm(timescale_dict.keys()):
        
        #if int(timescale_dict['%s' %galaxyid]['DescendantID_list'][-1]) != 12523088:
        #    continue
        
        plt.plot(np.arange(1, len(timescale_dict['%s' %galaxyid]['misangle_list'])+1, 1), timescale_dict['%s' %galaxyid]['misangle_list'])
        print(np.arange(1, len(timescale_dict['%s' %galaxyid]['misangle_list'])+1, 1))
        print(timescale_dict['%s' %galaxyid]['misangle_list'])
        
        
        #plt.text(np.arange(1, len(timescale_dict['%s' %galaxyid]['misangle_list'])+1, 1)[-1], timescale_dict['%s' %galaxyid]['misangle_list'][-1], '%s' %(timescale_dict['%s' %galaxyid]['DescendantID_list'][-1]))
        
        for snap, angle, i in zip(timescale_dict['%s' %galaxyid]['SnapNum_list'], timescale_dict['%s' %galaxyid]['misangle_list'], np.arange(1, len(timescale_dict['%s' %galaxyid]['misangle_list'])+1, 1)):
            
            if snap in timescale_dict['%s' %galaxyid]['merger_snap_list']:
                plt.scatter(i, angle)
            
    plt.show()
    
    """for galaxyid in tqdm(timescale_dict.keys()):
        
        #if int(timescale_dict['%s' %galaxyid]['DescendantID_list'][-1]) != 12523088:
            #continue

        print(timescale_dict['%s' %galaxyid]['misangle_list'])
        print(timescale_dict['%s' %galaxyid]['SnapNum_list'])
        print(timescale_dict['%s' %galaxyid]['GalaxyID_list'])
    
    
        plt.plot(timescale_dict['%s' %galaxyid]['Lookbacktime_list'], timescale_dict['%s' %galaxyid]['misangle_list'])
        
        for snap, angle, age in zip(timescale_dict['%s' %galaxyid]['SnapNum_list'], timescale_dict['%s' %galaxyid]['misangle_list'], timescale_dict['%s' %galaxyid]['Lookbacktime_list']):
            
            if snap in timescale_dict['%s' %galaxyid]['merger_snap_list']:
                plt.scatter(age, angle)
            
    plt.xlim(9, 0)
    plt.show()"""
    



# Plots for how long (time and snaps) misalignments perisit (from aligned -> stable) 
def _plot_time_spent_misaligned(csv_timescales = 'L100_timescale_tree_ETG_stars_gas_sf_rad2.0_projFalse_',
                                #--------------------------
                                plot_type = 'time',            # 'time', 'snap'    
                                #--------------------------
                                showfig       = False,
                                savefig       = True,
                                  file_format = 'pdf',
                                  savefig_txt = '',
                                #--------------------------
                                print_progress = False,
                                debug = False):
    
    
    # Loading sample
    dict_timetree = json.load(open('%s/%s.csv' %(output_dir, csv_timescales), 'r'))
    timescale_dict  = dict_timetree['timescale_dict']
    
    # Loading sample criteria
    timescale_input = dict_timetree['output_input']

    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    if debug:
        print('NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))

    print('\n===================')
    print('TIMESCALES LOADED:\n  %s\n  Snapshots: %s\n  Angle type: %s\n  Angle HMR: %s\n  Projected angle: %s' %(timescale_input['mySims'][0][0], timescale_input['csv_sample_range'], timescale_input['use_angle'], timescale_input['use_hmr'], timescale_input['use_proj_angle']))
    print('  NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
    
    #================================
    # Collect values for plot
    plot_timescale = []
    for GalaxyID in timescale_dict.keys():
        if np.isnan(np.array(timescale_dict['%s' %GalaxyID]['time_end'])) == False:
            
            #print(GalaxyID)
            #print(timescale_dict['%s' %GalaxyID]['kappa_stars_list'])
            #print(timescale_dict['%s' %GalaxyID]['misangle_list'])
            
            if plot_type == 'time':
                plot_timescale.append(float(timescale_dict['%s' %GalaxyID]['time_start']) - float(timescale_dict['%s' %GalaxyID]['time_end']))
            elif plot_type == 'snap':
                plot_timescale.append(abs(float(timescale_dict['%s' %GalaxyID]['SnapNum_start']) - float(timescale_dict['%s' %GalaxyID]['SnapNum_end'])))
            
        
    

    #================================
    # Plotting
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        print('Plotting')
        time_start = time.time()
    
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[7.0, 4.2], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-----------    
    # Plot histogram
    axs.hist(plot_timescale, weights=np.ones(len(plot_timescale))/len(plot_timescale), bins=np.arange(0, 10.1, 0.5), histtype='bar', edgecolor='none', facecolor='b', alpha=0.1)
    bin_count, _, _ = axs.hist(plot_timescale, weights=np.ones(len(plot_timescale))/len(plot_timescale), bins=np.arange(0, 10.1, 0.5), histtype='bar', edgecolor='b', facecolor='none', alpha=1.0)
    
    # Add poisson errors to each bin (sqrt N)
    hist_n, _ = np.histogram(plot_timescale, bins=np.arange(0, 10.1, 0.5), range=(0, 10))
    axs.errorbar(np.arange(0.25, 10.1, 0.5), hist_n/len(plot_timescale), xerr=None, yerr=np.sqrt(hist_n)/len(plot_timescale), ecolor='b', ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol=''))
    axs.set_xlim(0, 10)
    axs.set_xticks(np.arange(0, 10.1, step=1))
    if plot_type == 'time':
        axs.set_xlabel('Max. time spent misaligned [Gyr]')
    if plot_type == 'snap':
        axs.set_xlabel('Max. snapshots spent misaligned')
    axs.set_ylabel('Percentage of galaxies')
    
    
    #-----------
    # Annotations
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    # Add LTG/ETG if specified
    if timescale_input['ETG_or_LTG'] == 'both':
        legend_labels.append('Sample')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('grey')
    else:
        legend_labels.append('%s' %timescale_input['ETG_or_LTG'])
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('grey')
    
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
    
    
    
    #-----------
    # other
    plt.tight_layout()
    
    
    #-----------
    # Savefig
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        print('Finished')
    
    metadata_plot = {'Title': 'NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()),
                     'Subject': str(hist_n)}
    
    if savefig:
        plt.savefig("%s/%srelaxation_timescales_%s_%s_HMR%s_proj%s_m%sm%s_%s_%s.%s" %(fig_dir, timescale_input['csv_sample1'], timescale_input['ETG_or_LTG'], timescale_input['use_angle'], timescale_input['use_hmr'], timescale_input['use_proj_angle'], np.log10(float(timescale_input['lower_mass_limit'])), np.log10(float(timescale_input['upper_mass_limit'])), plot_type, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/%srelaxation_timescales_%s_%s_HMR%s_proj%s_m%sm%s_%s_%s.%s" %(fig_dir, timescale_input['csv_sample1'], timescale_input['ETG_or_LTG'], timescale_input['use_angle'], timescale_input['use_hmr'], timescale_input['use_proj_angle'], np.log10(float(timescale_input['lower_mass_limit'])), np.log10(float(timescale_input['upper_mass_limit'])), plot_type, savefig_txt, file_format))
    if showfig:
        plt.show()
    plt.close()
    

# Goes through existing CSV files to find correlation between misalignment created and relaxation time
def _plot_delta_misalignment(csv_timescales = 'L100_timescale_tree_ETG_stars_gas_sf_rad2.0_projFalse_',
                             #--------------------------
                             plot_type = 'time',            # 'time', 'snap'    
                             #--------------------------
                             showfig       = True,
                             savefig       = False,
                               file_format = 'pdf',
                               savefig_txt = '',
                             #--------------------------
                             print_progress = False,
                             debug = False):
    
    
    # Loading sample
    dict_timetree = json.load(open('%s/%s.csv' %(output_dir, csv_timescales), 'r'))
    timescale_dict  = dict_timetree['timescale_dict']
    
    # Loading sample criteria
    timescale_input = dict_timetree['output_input']

    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    if debug:
        print('NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))

    print('\n===================')
    print('TIMESCALES LOADED:\n  %s\n  Snapshots: %s\n  Angle type: %s\n  Angle HMR: %s\n  Projected angle: %s' %(timescale_input['mySims'][0][0], timescale_input['csv_sample_range'], timescale_input['use_angle'], timescale_input['use_hmr'], timescale_input['use_proj_angle']))
    print('  NUMBER OF MISALIGNMENTS: %s' %len(timescale_dict.keys()))
    
    #================================
    # Change in angle from misaligned state to settle, peak at 90
    
    # Collect values for plot
    plot_timescale = []
    plot_misangle = []
    plot_result = []
    for GalaxyID in timescale_dict.keys():
        if np.isnan(np.array(timescale_dict['%s' %GalaxyID]['time_end'])) == False:
            if len(timescale_dict['%s' %GalaxyID]['misangle_list']) > 2:
                plot_timescale.append(float(timescale_dict['%s' %GalaxyID]['time_start']) - float(timescale_dict['%s' %GalaxyID]['time_end']))
                plot_misangle.append(float(timescale_dict['%s' %GalaxyID]['misangle_list'][1]))
                
                if float(timescale_dict['%s' %GalaxyID]['misangle_list'][-1]) < 30:
                    plot_result.append('b')
                elif float(timescale_dict['%s' %GalaxyID]['misangle_list'][-1]) > 150:
                    plot_result.append('r')
    
    # Plotting
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        print('Plotting')
        time_start = time.time()
        
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[7.0, 4.2], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-----------    
    # Plot scatter
    axs.scatter(plot_misangle, plot_timescale, s=0.3, color=plot_result, alpha=0.9)
    
    
    axs.hist(plot_timescale, weights=np.ones(len(plot_timescale))/len(plot_timescale), bins=np.arange(0, 10.1, 0.5), histtype='bar', edgecolor='none', facecolor='b', alpha=0.1)
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_ylim(0, 10)
    axs.set_xlim(0, 180)
    axs.set_xticks(np.arange(0, 181, step=30))
    axs.set_ylabel('Relaxation time (Gyr)')
    axs.set_xlabel('$\Delta \psi_{\mathrm{3D}}$')
    
    #-----------
    # Annotations
    
    
    #-----------
    ### Legend
    
    
    #-----------
    # other
    plt.tight_layout()
    
    
    #-----------
    # Savefig
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        print('Finished')
    
    metadata_plot = {'Subject': 'none'}
    
    if savefig:
        plt.savefig("%s/%sdelta_misangle_%s_%s.%s" %(fig_dir, timescale_input['csv_sample1'], plot_type, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/%sdelta_misangle_%s_%s.%s" %(fig_dir, timescale_input['csv_sample1'], plot_type, savefig_txt, file_format))
    if showfig:
        plt.show()
    plt.close()
    
    

#=============================
_create_misalignment_time_csv()
#_create_misalignment_merger_csv()

#_plot_merger_timescales()

#_plot_time_spent_misaligned()
#_plot_delta_misalignment()
#=============================



































