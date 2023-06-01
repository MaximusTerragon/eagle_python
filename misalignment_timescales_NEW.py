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


#--------------------------------
# Goes through all csv samples given and collects all galaxies with matching criteria
def _extract_criteria_galaxies(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
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
    
    
    #---------------------- 
    # Load sample csv
    if print_progress:
        print('Cycling through CSV files and extracting galaxies')
        time_start = time.time()
    
    print('===================')
    print('CSV CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %2E M*\n  Upper mass limit: %2E M*\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
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
    

    if print_progress:
        print('NUMBER OF GALAXIES MEETING CRITERIA: ', len(list(galaxy_dict.keys())))


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
        csv_dict = {'galaxy_dict': galaxy_dict}
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
        json.dump(csv_dict, open('%s/%sgalaxy_dict_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, csv_sample1, ETG_or_LTG, use_angle, use_hmr, use_proj_angle, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%sgalaxy_dict_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, csv_sample1, ETG_or_LTG, use_angle, use_hmr, use_proj_angle, csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
        # Reading JSON file
        """ 
        # Ensuring the sample and output originated together
        csv_galaxy_dict = ...
        
        # Loading galaxies meeting criteria
        galaxy_dict_load = json.load(open('%s/%s.csv' %(output_dir, csv_galaxy_dict), 'r'))
        galaxy_dict  = galaxy_dict_load['timescale_dict']
        
        # Loading sample criteria
        galaxy_input = galaxy_dict_load['output_input']
    
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print('NUMBER OF MISALIGNMENTS: %s' %len(galaxy_dict.keys()))
   
        print('\n===================')
        print('GALAXY CRITERIA LOADED:\n %s\n  Snapshots: %s\n  Angle type: %s\n  Angle HMR: %s\n  Projected angle: %s' %(galaxy_input['mySims'][0][0], galaxy_input['csv_sample_range'], galaxy_input['use_angle'], galaxy_input['use_hmr'], galaxy_input['use_proj_angle']))
        print('  NUMBER OF GALAXIES MEETING CRITERIA: %s' %len(galaxy_dict.keys()))
            
        print('\nCRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s\n  Upper mass limit: %s\n  ETG or LTG: %s\n  Group or field: %s' %(galaxy_input['use_angle'], galaxy_input['use_hmr'], galaxy_input['use_proj_angle'], galaxy_input['lower_mass_limit'], galaxy_input['upper_mass_limit'], galaxy_input['ETG_or_LTG'], galaxy_input['group_or_field']))
        print('===================')
        """


#--------------------------------
# Goes through existing CSV files (minor and major) and creates merger tree
def _create_merger_tree_csv(csv_start        = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                            csv_sample       = '_all_sample_misalignment_9.0',
                            csv_sample_minor = '_minor_sample_misalignment_8.0',
                            csv_output_in    = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                            csv_sample_range = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],   # snapnums
                            #--------------------------
                            # Galaxy analysis
                            print_summary = True,
                            #--------------------------
                            csv_file      = False,             # Will write sample to csv file in sample_dir
                              csv_name    = '',               # extra stuff at end
                            #--------------------------
                            print_progress = False,
                            debug = True):

    #================================================  
    # Load sample csv
    if print_progress:
        print('Cycling through CSV files and extracting galaxies')
        time_start = time.time()
    
    print('===================')
    print('CSV CRITERIA:\n  Tree range: %s' %(csv_sample_range))
    print('===================\n')
    
    
    #================================================ 
    # Creating dictionary to create merger tree
    tree_dict = {}
    
    #----------------------
    # Cycling over all the csv samples we want
    for csv_sample_range_i in tqdm(csv_sample_range):
        
        # Ignore last snap
        if int(csv_sample_range_i) == csv_sample_range[-1]:
            continue
        
        
        # Ensuring the sample and output originated together
        csv_sample_load       = csv_start + str(csv_sample_range_i) + csv_sample
        csv_sample_minor_load = csv_start + str(csv_sample_range_i) + csv_sample_minor
        csv_output_load       = csv_sample_load + csv_output_in
        csv_output_minor_load = csv_sample_minor_load + '_'
        
        
        #================================================  
        # Load sample csv
        if print_progress:
            print('Loading initial sample')
            time_start = time.time()
    
        #--------------------------------
        # Creating sample
        dict_sample       = json.load(open('%s/%s.csv' %(sample_dir, csv_sample_load), 'r'))
        dict_sample_minor = json.load(open('%s/%s.csv' %(sample_dir, csv_sample_minor_load), 'r'))
        GroupNum_List     = np.concatenate((np.array(dict_sample['GroupNum']), np.array(dict_sample_minor['GroupNum'])))
        SubGroupNum_List  = np.concatenate((np.array(dict_sample['SubGroupNum']), np.array(dict_sample_minor['SubGroupNum'])))
        GalaxyID_List     = np.concatenate((np.array(dict_sample['GalaxyID']), np.array(dict_sample_minor['GalaxyID'])))
        DescendantID_List = np.concatenate((np.array(dict_sample['DescendantID']), np.array(dict_sample_minor['DescendantID'])))
        SnapNum_List      = np.concatenate((np.array(dict_sample['SnapNum']), np.array(dict_sample_minor['SnapNum'])))
        
        # Loading outputs
        dict_output       = json.load(open('%s/%s.csv' %(output_dir, csv_output_load), 'r'))
        dict_output_minor = json.load(open('%s/%s.csv' %(output_dir, csv_output_minor_load), 'r'))
        all_general = dict_output['all_general']
        all_general.update(dict_output_minor['all_general'])
        
        # Loading sample criteria
        sample_input        = dict_sample['sample_input']
        sample_minor_input  = dict_sample['sample_input']
        output_input        = dict_output['output_input']
        output_minor_input  = dict_output['output_input']
        
        
        #print('NEW SNAP: %s' %dict_sample_minor['SnapNum'])
        
    
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print(sample_input)
            print(GroupNum_List)
            print(SubGroupNum_List)
            print(GalaxyID_List)
            print(DescendantID_List)
            print(SnapNum_List)
   
        print('===================')
        print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %.2f\n' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift']))
        print('SAMPLE MINOR LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %.2f\n' %(sample_minor_input['mySims'][0][0], sample_minor_input['snapNum'], sample_minor_input['Redshift']))
        
        
        #----------------------------
        # Looping over all GalaxyIDs
        merger_tally = 0
        for GalaxyID, DescendantID in zip(GalaxyID_List, DescendantID_List):
            
            # find age
            Lookbacktime = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(output_input['Redshift'])).value
            
            #-----------------------------
            # Create new/update entry for descendantID... this means we have unique entries for each descendant, and within a list of all galaxies taht will become that descendant
            if str(DescendantID) not in tree_dict.keys():
                tree_dict['%s' %DescendantID] = {'%s' %GalaxyID: {}}
                
                if debug:
                    print('New Descendant: ', DescendantID)
                
            elif (str(DescendantID) in tree_dict.keys()):
                tree_dict['%s' %DescendantID].update({'%s' %GalaxyID: {}})
                
                if debug:
                    print('Existing DescendantID: ', DescendantID)
                    print('  Number of constituants: ', len(tree_dict['%s' %DescendantID]), tree_dict['%s' %DescendantID].keys())
                
                merger_tally += 1
                print(merger_tally)
                
            else:
                raise Exception('aaaa something broke-y')
    
            tree_dict['%s' %DescendantID]['%s' %GalaxyID].update({'GalaxyID': GalaxyID, 
                                                                  'DescendantID': DescendantID, 
                                                                  'GroupNum': all_general['%s' %GalaxyID]['GroupNum'],
                                                                  'SubGroupNum': all_general['%s' %GalaxyID]['SubGroupNum'],
                                                                  'SnapNum': output_input['snapNum'],
                                                                  'Redshift': output_input['Redshift'],
                                                                  'Lookbacktime': Lookbacktime,
                                                                  'stelmass': all_general['%s' %GalaxyID]['stelmass'],
                                                                  'gasmass': all_general['%s' %GalaxyID]['gasmass'],
                                                                  'gasmass_sf': all_general['%s' %GalaxyID]['gasmass_sf']})
    
    if debug:
        for DescendantID in tree_dict.keys():
            if len(tree_dict['%s' %DescendantID].keys()) > 6:
                print('DescendantID: ', DescendantID)
                print(' ', tree_dict['%s' %DescendantID].keys())
                
                for GalaxyID in tree_dict['%s' %DescendantID].keys():
                    print(tree_dict['%s' %DescendantID]['%s' %GalaxyID]['stelmass'])
                    print(float(tree_dict['%s' %DescendantID]['%s' %GalaxyID]['stelmass'])/float(tree_dict['%s' %DescendantID]['%s' %(int(DescendantID)+1)]['stelmass']))
        
    
    #===================================
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
        csv_dict = {'tree_dict': tree_dict}
        output_input = {'csv_start': csv_start,
                        'csv_sample': csv_sample,
                        'csv_sample_minor': csv_sample_minor,
                        'csv_output_in': csv_output_in,
                        'csv_sample_range': csv_sample_range,
                        'mySims': sample_input['mySims']}
                        
        csv_dict.update({'output_input': output_input})
        
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%smerger_tree_%s.csv' %(output_dir, csv_start, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%smerger_tree_%s.csv' %(output_dir, csv_start, csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
        # Reading JSON file
        ''' 
        # Ensuring the sample and output originated together
        csv_mergertree = ...
        
        # Loading sample
        merger_tree_dict = json.load(open('%s/%s.csv' %(sample_dir, csv_mergertree), 'r'))
        merger_tree = merger_tree_dict['timescale_dict']
        
        # Loading sample criteria
        merger_tree_input = merger_tree['output_input']
    
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print('NUMBER OF DESCENDANTID: %s' %len(merger_tree.keys()))
   
        print('\n===================')
        print('MERGER TREE LOADED:\n %s\n  Snapshots: %s' %(merger_tree_input['mySims'][0][0], merger_tree_input['csv_sample_range']))
        print('===================')
        
        
        # .keys() gives list of DescendantIDs across all snapshots
        '''
               

#--------------------------------
# Goes through galaxies that meet criteria, analyses the time spend in misaligned state
def _analyse_misalignment_timescales(csv_galaxy_dict = 'L100_galaxy_dict_both_stars_gas_sf_rad2.0_projFalse_',
                                     #--------------------------
                                     # Galaxy analysis
                                     print_summary = True,
                                     #--------------------------
                                     csv_file       = False,             # Will write sample to csv file in sample_dir
                                       csv_name     = '',               # extra stuff at end
                                     #--------------------------
                                     print_progress = False,
                                     debug = False):

    #---------------------- 
    # Load dictionary of galaxies that meet criteria
    if print_progress:
        print('Loading csv of galaxies that meet criteria')
        time_start = time.time()
    
    # Loading galaxies meeting criteria
    galaxy_dict_load  = json.load(open('%s/%s.csv' %(output_dir, csv_galaxy_dict), 'r'))
    galaxy_dict       = galaxy_dict_load['galaxy_dict']
    galaxy_dict_input = galaxy_dict_load['output_input']
    
    csv_sample_range  = [int(i) for i in galaxy_dict_input['csv_sample_range']]


    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))

    print('\n===================')
    print('GALAXY CRITERIA LOADED:\n %s\n  Snapshots: %s\n  Angle type: %s\n  Angle HMR: %s\n  Projected angle: %s' %(galaxy_dict_input['mySims'][0][0], galaxy_dict_input['csv_sample_range'], galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle']))
    print('  NUMBER OF GALAXIES MEETING CRITERIA: %s' %len(galaxy_dict.keys()))
    
    print('\nCRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %2E\n  Upper mass limit: %2E\n  ETG or LTG: %s\n  Group or field: %s' %(galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle'], galaxy_dict_input['lower_mass_limit'], galaxy_dict_input['upper_mass_limit'], galaxy_dict_input['ETG_or_LTG'], galaxy_dict_input['group_or_field']))
    print('===================')
    

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
        
                GalaxyID        = int(GalaxyID)
                DescendantID    = int(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['DescendantID'])
                # if misaligned, skip as we have already included it or we can't constrain when it started (ei. joined sample misaligned)
                if (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) >= 30):
                    continue
                    
                # Start time
                SnapNum_start   = int(SnapNum)
                time_start      = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime'])
                Redshift_start  = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift'])
     
                #-----------------------------------
                # if descendant meets criteria and descendant is root descendant:
                if (str(DescendantID) in galaxy_dict['%s' %(SnapNum+1)]) and (int(DescendantID) == int(GalaxyID)-1):
                    
                    #-----------------------------------
                    # if descendant is NOT aligned:
                    if float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) > 30:
                        
                        if debug:
                            print('\nBECOMES MISALIGNED: %s    %s    %.2E ' %(GalaxyID, SnapNum, float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass'])))
                        
                        # Append initial galaxy conditions
                        SnapNum_list           = [SnapNum]
                        GalaxyID_list          = [GalaxyID]
                        DescendantID_list      = [DescendantID]
                        Lookbacktime_list      = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime']]
                        Redshift_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift']]
                        GroupNum_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['GroupNum']]
                        SubGroupNum_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SubGroupNum']]
                        misangle_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']]
                        stelmass_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']]
                        gasmass_list           = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass']]
                        gasmass_sf_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf']]
                        halfmass_rad_list      = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad']]
                        halfmass_rad_proj_list = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad_proj']]
                        kappa_stars_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_stars']]
                        kappa_gas_sf_list      = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_gas_sf']]
                        
                        
                        # Update Snap, galaxyID, descendant to next one
                        SnapNum_tmp      = galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['SnapNum']
                        GalaxyID_tmp     = galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['GalaxyID']
                        DescendantID_tmp = galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['DescendantID']
                        
                        #-----------------------------------
                        # This will update for as long as galaxy remains misaligned
                        SnapNum_end  = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SnapNum']
                        Redshift_end = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift']
                        time_end     = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime']
                        
                        # misangle of descendant
                        misangle = float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                        
                        
                        #-----------------------------------
                        # While galaxy is misaligned... (30 - 150)... skips if counter-rotating
                        while (misangle >= 30) and (misangle <= 150):
                            
                            if debug:
                                print('  BECAME UNSTABLE MISALIGNED AT ', GalaxyID_tmp, SnapNum_tmp)
                                print('  MISANGLE:      ', misangle)
                            
                            # If current misaligned galaxy out of csv range, break
                            if (int(SnapNum_tmp) == 28):
                                if debug:
                                    print('  a')
                                SnapNum_end  = math.nan
                                Redshift_end = math.nan
                                time_end     = math.nan
                                break
                            
                            # If descendant of current misaligned galaxy not a mainline, break
                            if (int(SnapNum_tmp) < 28) and (int(DescendantID_tmp) != int(GalaxyID_tmp)-1):
                                if debug:
                                    print('  b')
                                SnapNum_end  = math.nan
                                Redshift_end = math.nan
                                time_end     = math.nan
                                break
                            
                            # If descendant of current misaligned galaxy not meet criteria, break
                            if (str(DescendantID_tmp) not in galaxy_dict['%s' %(int(SnapNum_tmp)+1)]):
                                if debug:
                                    print('  c')
                                SnapNum_end  = math.nan
                                Redshift_end = math.nan
                                time_end     = math.nan
                                break
                                
                            
                            #-----------------------------------
                            # Gather current stats
                            SnapNum_list.append(SnapNum_tmp)
                            GalaxyID_list.append(GalaxyID_tmp)
                            DescendantID_list.append(DescendantID_tmp)
                            Lookbacktime_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime'])
                            Redshift_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift'])
                            GroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['GroupNum'])
                            SubGroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SubGroupNum'])
                            misangle_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                            stelmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['stelmass'])
                            gasmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass'])
                            gasmass_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass_sf'])
                            halfmass_rad_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad'])
                            halfmass_rad_proj_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad_proj'])
                            kappa_stars_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_stars'])
                            kappa_gas_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_gas_sf'])
                            
                            
                            #-----------------------------------
                            # Update Snap, GalaxyID, and DescendantID:                           
                            SnapNum_tmp      = int(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum'])
                            GalaxyID_tmp     = int(galaxy_dict['%s' %(int(SnapNum_tmp))]['%s' %DescendantID_tmp]['GalaxyID'])
                            DescendantID_tmp = int(galaxy_dict['%s' %(int(SnapNum_tmp))]['%s' %GalaxyID_tmp]['DescendantID'])
                            
                            # Update misangle
                            misangle = float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                            
                            #-----------------------------------
                            # This will update for as long as galaxy remains misaligned
                            SnapNum_end  = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SnapNum']
                            Redshift_end = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift']
                            time_end     = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime']
                            
                            
                        #===========================================================
                        # Out of while loop, so descendant has become aligned again and meets criteria by default (for instant counter - already met, while loop takes care of rest). If anything not met will have a nan assigned to time_end
                        
                        # Check for galaxies existing whlie loop that aligned descendant
                        if np.isnan(time_end) == False:
                            
                            #-----------------------------------
                            # This will update for as long as galaxy remains misaligned
                            SnapNum_end  = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SnapNum']
                            Redshift_end = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift']
                            time_end     = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime']
                            
                            #-----------------------------------
                            # Gather finishing stats
                            SnapNum_list.append(SnapNum_tmp)
                            GalaxyID_list.append(GalaxyID_tmp)
                            DescendantID_list.append(DescendantID_tmp)
                            Lookbacktime_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime'])
                            Redshift_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift'])
                            GroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['GroupNum'])
                            SubGroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SubGroupNum'])
                            misangle_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                            stelmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['stelmass'])
                            gasmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass'])
                            gasmass_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass_sf'])
                            halfmass_rad_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad'])
                            halfmass_rad_proj_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad_proj'])
                            kappa_stars_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_stars'])
                            kappa_gas_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_gas_sf'])
                            
                            if not debug:
                                print(' ')
                            print('IN SAMPLE:   >>> ', GalaxyID_list[-1], len(SnapNum_list), ' <<<')
                            if len(SnapNum_list) == 2:
                                print('  BECAME COUNTER-ROTATING')
                            else:
                                print('  TRANSITIONED')
                            print('  TIME TAKEN TO RELAX: ', abs(time_start - time_end))
                            print('  ', SnapNum_list)
                            print('  ', GalaxyID_list)
                            print('  ', DescendantID_list)
                            print('  ', misangle_list)
                            print('  ', Lookbacktime_list)
                                
                            
                            #------------------------------------
                            # Update dictionary
                            timescale_dict['%s' %GalaxyID] = {'SnapNum_list': SnapNum_list,
                                                              'GalaxyID_list': GalaxyID_list, 
                                                              'DescendantID_list': DescendantID_list,
                                                              'Lookbacktime_list': Lookbacktime_list,
                                                              'Redshift_list': Redshift_list,
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
        output_input = {'csv_galaxy_dict': csv_galaxy_dict}
        output_input.update(galaxy_dict_input)
                        
        csv_dict.update({'output_input': output_input})
        
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%stimescale_tree_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, galaxy_dict_input['csv_sample1'], galaxy_dict_input['ETG_or_LTG'], galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle'], csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%stimescale_tree_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, galaxy_dict_input['csv_sample1'], galaxy_dict_input['ETG_or_LTG'], galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle'], csv_name))
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


#--------------------------------
# Goes through galaxies that meet criteria, extracts galaxies that became misaligned coinciding within X Gyr of a merger
def _analyse_merger_origin_timescales(csv_galaxy_dict = 'L100_galaxy_dict_both_stars_gas_sf_rad2.0_projFalse_',
                                      csv_merger_tree = 'L100_merger_tree_',
                                      #--------------------------
                                      # Galaxy analysis
                                      print_summary = True,
                                        merger_misaligned_time_pre  = 0.1,             # Gyr, Time before last aligned state, and merger between which the galaxy is misaligned
                                        merger_misaligned_time_post = 2.0,             # Gyr, Time between last aligned state, and merger between which the galaxy is misaligned
                                        merger_threshold_min   = 0.1,             # >= to include
                                        merger_threshold_max   = 1.0,             # <= to include
                                      #--------------------------
                                      csv_file       = True,             # Will write sample to csv file in sample_dir
                                        csv_name     = '',               # extra stuff at end
                                      #--------------------------
                                      print_progress = False,
                                      debug = True):

    #---------------------- 
    # Load dictionary of galaxies that meet criteria
    if print_progress:
        print('Loading csv of galaxies that meet criteria')
        time_start = time.time()
    
    # Loading galaxies meeting criteria
    galaxy_dict_load  = json.load(open('%s/%s.csv' %(output_dir, csv_galaxy_dict), 'r'))
    galaxy_dict       = galaxy_dict_load['galaxy_dict']
    galaxy_dict_input = galaxy_dict_load['output_input']
    
    csv_sample_range = [int(i) for i in galaxy_dict_input['csv_sample_range']]

    # Loading merger tree
    merger_tree_load  = json.load(open('%s/%s.csv' %(output_dir, csv_merger_tree), 'r'))
    merger_tree       = merger_tree_load['tree_dict']     
    merger_tree_input = merger_tree_load['output_input']
    

    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))

    print('\n===================')
    print('GALAXY CRITERIA LOADED:\n %s\n  Snapshots: %s\n  Angle type: %s\n  Angle HMR: %s\n  Projected angle: %s' %(galaxy_dict_input['mySims'][0][0], galaxy_dict_input['csv_sample_range'], galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle']))
    print('  NUMBER OF GALAXIES MEETING CRITERIA: %s' %len(galaxy_dict.keys()))
    
    print('\nGALAXY CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %2E\n  Upper mass limit: %2E\n  ETG or LTG: %s\n  Group or field: %s' %(galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle'], galaxy_dict_input['lower_mass_limit'], galaxy_dict_input['upper_mass_limit'], galaxy_dict_input['ETG_or_LTG'], galaxy_dict_input['group_or_field']))
    print('\nMISALIGNMENT CRITERIA: \n  Delta merger/misaligned: %s Gyr pre, %s Gyr post last aligned\n  Merger thresholds: %s - %s' %(merger_misaligned_time_pre, merger_misaligned_time_post, merger_threshold_min, merger_threshold_max))
    print('===================')
    

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
            
            # Ignore first and last snap (28)
            if (SnapNum == csv_sample_range[0]) or (SnapNum == csv_sample_range[-1]):
                continue
            
            if debug:
                print('NEW SNAP: ', SnapNum)
            
            for GalaxyID in tqdm(galaxy_dict['%s' %SnapNum].keys()):
        
                GalaxyID        = int(GalaxyID)
                DescendantID    = int(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['DescendantID'])
                # if misaligned, skip as we have already included it or we can't constrain when it started (ei. joined sample misaligned)
                if (float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']) >= 30):
                    continue
     
                # Start time
                SnapNum_start   = int(SnapNum)
                time_start      = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime'])
                Redshift_start  = float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift'])
                
                #-----------------------------------
                # if descendant meets criteria and descendant is root descendant:
                if (str(DescendantID) in galaxy_dict['%s' %(SnapNum+1)]) and (int(DescendantID) == int(GalaxyID)-1):
                    
                    #-----------------------------------
                    # if descendant is NOT aligned:
                    if float(galaxy_dict['%s' %(SnapNum+1)]['%s' %DescendantID]['misangle']) > 30:
                        
                        # Galaxies that reach here must be aligned in current SnapNum, become misaligned in SnapNum+1, both meet criteria, and may or may not have undergone a merger meeting criteria in next (or other) snapshots within time specified from last aligned
                        
                        if debug:
                            print('\nBECOMES MISALIGNED: %s    %s    %.2E ' %(GalaxyID, SnapNum, float(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass'])))
                        
                        # Append initial galaxy conditions
                        SnapNum_list           = [SnapNum]
                        GalaxyID_list          = [GalaxyID]
                        DescendantID_list      = [DescendantID]
                        Lookbacktime_list      = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime']]
                        Redshift_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift']]
                        GroupNum_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['GroupNum']]
                        SubGroupNum_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SubGroupNum']]
                        misangle_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['misangle']]
                        stelmass_list          = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['stelmass']]
                        gasmass_list           = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass']]
                        gasmass_sf_list        = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['gasmass_sf']]
                        halfmass_rad_list      = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad']]
                        halfmass_rad_proj_list = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['halfmass_rad_proj']]
                        kappa_stars_list       = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_stars']]
                        kappa_gas_sf_list      = [galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['kappa_gas_sf']]
                        
                        merger_analysis = {}
                        merger_time_criteria = False
                        
                        
                        #===================================#===================================
                        # Analyse any mergers between aligned (GalaxyID) and misaligned (DescendantID)
                        
                        merger_snap_array        = []
                        merger_age_array         = []
                        merger_redshift_array    = []
                        merger_ratio_array       = []
                        merger_gas_ratio_array   = []
                        merger_gassf_ratio_array = []
                        merger_id_array          = []
                        
                        if debug:
                            print('  GalaxyID components: ', merger_tree['%s' %GalaxyID].keys())
                        
                        # Run merger analysis for current galaxy (GalaxyID) with merger between previous galaxy (GalaxyID - 1)
                        for component_GalaxyID in merger_tree['%s' %GalaxyID].keys():
                            
                            if debug:
                                print('     componentID: %s     %.2f M' %(component_GalaxyID, np.log10(float(merger_tree['%s' %GalaxyID]['%s' %component_GalaxyID]['stelmass']))))
                            
                            # If descendant is main line progenitor, ignore
                            if int(GalaxyID) == (int(component_GalaxyID) - 1):
                                if debug:
                                    print('     Mainline, continue')
                                continue
                        
                            # If descendant is bugged, ignore
                            if int(GalaxyID) == (int(component_GalaxyID)):
                                if debug:
                                    print('     Bugged, continue')
                                continue
                            
                            # If current galaxy has no progenitor, ignore
                            if str(int(GalaxyID) + 1) not in merger_tree['%s' %GalaxyID]:
                                if debug:
                                    print('     No progenitor, continue')
                                continue
                            
                            #-------------
                            # Assign primary and component values
                            primary_stelmass    = float(merger_tree['%s' %GalaxyID]['%s' %(int(GalaxyID) + 1)]['stelmass'])
                            primary_gasmass     = float(merger_tree['%s' %GalaxyID]['%s' %(int(GalaxyID) + 1)]['gasmass'])
                            primary_gassfmass   = float(merger_tree['%s' %GalaxyID]['%s' %(int(GalaxyID) + 1)]['gasmass_sf'])
                            component_stelmass  = float(merger_tree['%s' %GalaxyID]['%s' %component_GalaxyID]['stelmass'])
                            component_gasmass   = float(merger_tree['%s' %GalaxyID]['%s' %component_GalaxyID]['gasmass'])
                            component_gassfmass = float(merger_tree['%s' %GalaxyID]['%s' %component_GalaxyID]['gasmass_sf'])
                            
                            # Find stellar mass merger ratio (strictly < 1)
                            merger_ratio = component_stelmass / primary_stelmass 
                        
                            # Find gas ratios
                            gas_ratio   = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)
                            gassf_ratio = (primary_gassfmass + component_gassfmass) / (primary_stelmass + component_stelmass)
                        
                            if merger_ratio > 1.0:
                                merger_ratio = 1 / merger_ratio
                                gas_ratio    = 1 / gas_ratio
                                gassf_ratio  = 1 / gassf_ratio
                        
                            #--------------
                            # Append to lists
                            merger_snap_array.append(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SnapNum'])
                            merger_age_array.append(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Lookbacktime'])
                            merger_redshift_array.append(galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['Redshift'])
                            merger_ratio_array.append(float(merger_ratio))
                            merger_gas_ratio_array.append(float(gas_ratio))
                            merger_gassf_ratio_array.append(float(gassf_ratio))
                            merger_id_array.append(int(component_GalaxyID))
                        
                            if debug:
                                print(' STATS:  \n  GalaxyID: %s  GalaxyID+1: %s  component: %s  component_ratio: %s  snap: %s' %(GalaxyID, int(GalaxyID)-1, component_GalaxyID, merger_ratio, galaxy_dict['%s' %SnapNum]['%s' %GalaxyID]['SnapNum']))
                        
                        # Create and append merger stats
                        if len(merger_snap_array) > 0:
                            merger_analysis.update({'%s' %SnapNum: {'SnapNum_list': merger_snap_array,
                                                                    'Lookbacktime_list': merger_age_array,
                                                                    'Redshift_list': merger_redshift_array,
                                                                    'Ratio_stars_list': merger_ratio_array,
                                                                    'Ratio_gas_list': merger_gas_ratio_array,
                                                                    'Ratio_gassf_list': merger_gassf_ratio_array,
                                                                    'GalaxyID_list': merger_id_array}})
                            if debug:
                                print('HAS MERGER (LAST ALIGNED): ', int(SnapNum))
                              
                        #===================================
                        # TEST MERGER TIME CRITERIA If galaxy has at least one merger (from current aligned until next misaligned):
                        if len(merger_ratio_array) > 0:
                            # if there is a merger meeting min/max criteria within merger_misaligned_time of becoming misaligned:
                            if (np.array(merger_ratio_array).max() >= merger_threshold_min) and (np.array(merger_ratio_array).max() <= merger_threshold_max) and (abs(np.array(merger_age_array)[np.array(merger_ratio_array).argmax()] - time_start) <= merger_misaligned_time_pre):
                                # Galaxies that reach here must be aligned in current SnapNum, become misaligned in SnapNum+1, both meet criteria, must have undergone a merger meeting criteria, within a time meeting criteria from last aligned state but only for this
                                
                                if debug:
                                    print('HAS MERGER CRITERIA ON LAST ALIGNED')
                                
                                merger_time_criteria = True
                        #===================================#===================================
                        
                        
                        
                        #===================================#===================================
                        # Analyse any mergers between aligned (GalaxyID) and misaligned (DescendantID)
                        merger_snap_array        = []
                        merger_age_array         = []
                        merger_redshift_array    = []
                        merger_ratio_array       = []
                        merger_gas_ratio_array   = []
                        merger_gassf_ratio_array = []
                        merger_id_array          = []
                        
                        if debug:
                            print('  DescendantID components: ', merger_tree['%s' %DescendantID].keys())
                        
                        # Run over all component galaxies for next descendant
                        for component_GalaxyID in merger_tree['%s' %DescendantID].keys():
                        
                            if debug:
                                print('     componentID: %s     %.2f M' %(component_GalaxyID, np.log10(float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass']))))
                    
                            # If descendant is main line progenitor, ignore
                            if int(DescendantID) == (int(component_GalaxyID) - 1):
                                if debug:
                                    print('     Mainline, continue')
                                continue
                        
                            # If descendant is bugged, ignore
                            if int(DescendantID) == (int(component_GalaxyID)):
                                if debug:
                                    print('     Bugged, continue')
                                continue
                    
                            # If current galaxy has no progenitor, ignore
                            if str(int(DescendantID) + 1) not in merger_tree['%s' %DescendantID]:
                                if debug:
                                    print('     No progenitor, continue')
                                continue
                                
                            #-------------
                            # Assign primary and component values
                            primary_stelmass    = float(merger_tree['%s' %DescendantID]['%s' %(int(DescendantID) + 1)]['stelmass'])
                            primary_gasmass     = float(merger_tree['%s' %DescendantID]['%s' %(int(DescendantID) + 1)]['gasmass'])
                            primary_gassfmass   = float(merger_tree['%s' %DescendantID]['%s' %(int(DescendantID) + 1)]['gasmass_sf'])
                            component_stelmass  = float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['stelmass'])
                            component_gasmass   = float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['gasmass'])
                            component_gassfmass = float(merger_tree['%s' %DescendantID]['%s' %component_GalaxyID]['gasmass_sf'])
                            
                            # Find stellar mass merger ratio (strictly < 1)
                            merger_ratio = component_stelmass / primary_stelmass 
                        
                            # Find gas ratios
                            gas_ratio   = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)
                            gassf_ratio = (primary_gassfmass + component_gassfmass) / (primary_stelmass + component_stelmass)
                        
                            if merger_ratio > 1.0:
                                merger_ratio = 1 / merger_ratio
                                gas_ratio    = 1 / gas_ratio
                                gassf_ratio  = 1 / gassf_ratio
                        
                            #--------------
                            # Append to lists
                            merger_snap_array.append(galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['SnapNum'])
                            merger_age_array.append(galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['Lookbacktime'])
                            merger_redshift_array.append(galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['Redshift'])
                            merger_ratio_array.append(float(merger_ratio))
                            merger_gas_ratio_array.append(float(gas_ratio))
                            merger_gassf_ratio_array.append(float(gassf_ratio))
                            merger_id_array.append(int(component_GalaxyID))
                        
                            if debug:
                                print(' STATS:  \n  Descendant: %s  GalaxyID: %s  component: %s  component_ratio: %s  snap: %s' %(DescendantID, GalaxyID, component_GalaxyID, merger_ratio, galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['SnapNum']))
                        
                        # Create and append merger stats
                        if len(merger_snap_array) > 0:
                            merger_analysis.update({'%s' %(int(SnapNum)+1): {'SnapNum_list': merger_snap_array,
                                                                             'Lookbacktime_list': merger_age_array,
                                                                             'Redshift_list': merger_redshift_array,
                                                                             'Ratio_stars_list': merger_ratio_array,
                                                                             'Ratio_gas_list': merger_gas_ratio_array,
                                                                             'Ratio_gassf_list': merger_gassf_ratio_array,
                                                                             'GalaxyID_list': merger_id_array}})
                            if debug:
                                print('HAS MERGER: ', int(SnapNum)+1)
                            
                        #===================================
                        # TEST MERGER TIME CRITERIA If galaxy has at least one merger (from current aligned until next misaligned):
                        if len(merger_ratio_array) > 0:
                            # if there is a merger meeting min/max criteria within merger_misaligned_time of becoming misaligned:
                            if (np.array(merger_ratio_array).max() >= merger_threshold_min) and (np.array(merger_ratio_array).max() <= merger_threshold_max) and (abs(np.array(merger_age_array)[np.array(merger_ratio_array).argmax()] - time_start) <= merger_misaligned_time_post):
                                # Galaxies that reach here must be aligned in current SnapNum, become misaligned in SnapNum+1, both meet criteria, must have undergone a merger meeting criteria, within a time meeting criteria from last aligned state but only for this
                                
                                if debug:
                                    print('HAS MERGER CRITERIA ON FIRST MISALIGNED')
                                    
                                merger_time_criteria = True
                        #===================================#===================================
                        
                        
                        #-----------------------------------
                        # Update Snap, galaxyID, descendant to next one
                        SnapNum_tmp      = galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['SnapNum']
                        GalaxyID_tmp     = galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['GalaxyID']
                        DescendantID_tmp = galaxy_dict['%s' %(int(SnapNum)+1)]['%s' %DescendantID]['DescendantID']
                        
                        #-----------------------------------
                        # This will update for as long as galaxy remains misaligned
                        SnapNum_end  = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SnapNum']
                        Redshift_end = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift']
                        time_end     = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime']
                        
                        # misangle of descendant
                        misangle = float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                        
                        
                        #-----------------------------------
                        # While galaxy is misaligned... (30 - 150)... skips if counter-rotating
                        while (misangle >= 30) and (misangle <= 150):
                            
                            if debug:
                                print('  BECAME UNSTABLE MISALIGNED AT ', GalaxyID_tmp, SnapNum_tmp)
                                print('  MISANGLE:      ', misangle)
                            
                            # If current misaligned galaxy out of csv range, break
                            if (int(SnapNum_tmp) == 28):
                                if debug:
                                    print('  a')
                                SnapNum_end  = math.nan
                                Redshift_end = math.nan
                                time_end     = math.nan
                                break
                            
                            # If descendant of current misaligned galaxy not a mainline, break
                            if (int(SnapNum_tmp) < 28) and (int(DescendantID_tmp) != int(GalaxyID_tmp)-1):
                                if debug:
                                    print('  b')
                                SnapNum_end  = math.nan
                                Redshift_end = math.nan
                                time_end     = math.nan
                                break
                            
                            # If descendant of current misaligned galaxy not meet criteria, break
                            if (str(DescendantID_tmp) not in galaxy_dict['%s' %(int(SnapNum_tmp)+1)]):
                                if debug:
                                    print('  c')
                                SnapNum_end  = math.nan
                                Redshift_end = math.nan
                                time_end     = math.nan
                                break
                                
                            
                            #-----------------------------------
                            # Gather current stats
                            SnapNum_list.append(SnapNum_tmp)
                            GalaxyID_list.append(GalaxyID_tmp)
                            DescendantID_list.append(DescendantID_tmp)
                            Lookbacktime_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime'])
                            Redshift_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift'])
                            GroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['GroupNum'])
                            SubGroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SubGroupNum'])
                            misangle_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                            stelmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['stelmass'])
                            gasmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass'])
                            gasmass_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass_sf'])
                            halfmass_rad_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad'])
                            halfmass_rad_proj_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad_proj'])
                            kappa_stars_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_stars'])
                            kappa_gas_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_gas_sf'])
                            
                            
                            
                            #===================================#===================================
                            # Check for merger fitting criteria within time_start:
                            
                            merger_snap_array        = []
                            merger_age_array         = []
                            merger_redshift_array    = []
                            merger_ratio_array       = []
                            merger_gas_ratio_array   = []
                            merger_gassf_ratio_array = []
                            merger_id_array          = []
                            
                            if debug:
                                print('  DescendantID_tmp components: ', merger_tree['%s' %DescendantID_tmp].keys())
                        
                            # Run over all component galaxies for next descendant
                            for component_GalaxyID in merger_tree['%s' %DescendantID_tmp].keys():
                        
                                if debug:
                                    print('     componentID: %s     %.2f M' %(component_GalaxyID, np.log10(float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['stelmass']))))
                    
                                # If descendant is main line progenitor, ignore
                                if int(DescendantID_tmp) == (int(component_GalaxyID) - 1):
                                    if debug:
                                        print('     Mainline, continue')
                                    continue
                        
                                # If descendant is bugged, ignore
                                if int(DescendantID_tmp) == (int(component_GalaxyID)):
                                    if debug:
                                        print('     Bugged, continue')
                                    continue
                                
                                # If current galaxy has no progenitor, ignore
                                if str(int(DescendantID_tmp) + 1) not in merger_tree['%s' %DescendantID_tmp]:
                                    if debug:
                                        print('     No progenitor, continue')
                                    continue
                                    
                                #-------------
                                # Assign primary and component values
                                primary_stelmass    = float(merger_tree['%s' %DescendantID_tmp]['%s' %(int(DescendantID_tmp) + 1)]['stelmass'])
                                primary_gasmass     = float(merger_tree['%s' %DescendantID_tmp]['%s' %(int(DescendantID_tmp) + 1)]['gasmass'])
                                primary_gassfmass   = float(merger_tree['%s' %DescendantID_tmp]['%s' %(int(DescendantID_tmp) + 1)]['gasmass_sf'])
                                component_stelmass  = float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['stelmass'])
                                component_gasmass   = float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['gasmass'])
                                component_gassfmass = float(merger_tree['%s' %DescendantID_tmp]['%s' %component_GalaxyID]['gasmass_sf'])
                                
                                # Find stellar mass merger ratio (strictly < 1)
                                merger_ratio = component_stelmass / primary_stelmass 
                        
                                # Find gas ratios
                                gas_ratio   = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)
                                gassf_ratio = (primary_gassfmass + component_gassfmass) / (primary_stelmass + component_stelmass)
                        
                                if merger_ratio > 1.0:
                                    merger_ratio = 1 / merger_ratio
                                    gas_ratio    = 1 / gas_ratio
                                    gassf_ratio  = 1 / gassf_ratio
                        
                                #--------------
                                # Append to lists
                                merger_snap_array.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum'])
                                merger_age_array.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Lookbacktime'])
                                merger_redshift_array.append(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['Redshift'])
                                merger_ratio_array.append(float(merger_ratio))
                                merger_gas_ratio_array.append(float(gas_ratio))
                                merger_gassf_ratio_array.append(float(gassf_ratio))
                                merger_id_array.append(int(component_GalaxyID))
                        
                                if debug:
                                    print(' STATS:  \n  Descendant: %s  GalaxyID: %s  component: %s  component_ratio: %s  snap: %s' %(DescendantID_tmp, GalaxyID_tmp, component_GalaxyID, merger_ratio, galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum']))
                        
                            # Create and append merger stats
                            if len(merger_ratio_array) > 0:
                                merger_analysis.update({'%s' %(int(SnapNum_tmp)+1): {'SnapNum_list': merger_snap_array,
                                                                                     'Lookbacktime_list': merger_age_array,
                                                                                     'Redshift_list': merger_redshift_array,
                                                                                     'Ratio_stars_list': merger_ratio_array,
                                                                                     'Ratio_gas_list': merger_gas_ratio_array,
                                                                                     'Ratio_gassf_list': merger_gassf_ratio_array,
                                                                                     'GalaxyID_list': merger_id_array}})
                                if debug:
                                    print('HAS MERGER: ', int(SnapNum_tmp)+1)
                            
                            #===================================
                            # TEST MERGER TIME CRITERIA If galaxy has at least one merger (from current aligned until next misaligned):
                            if len(merger_ratio_array) > 0:
                                # if there is a merger meeting min/max criteria within merger_misaligned_time of becoming misaligned:
                                if (np.array(merger_ratio_array).max() >= merger_threshold_min) and (np.array(merger_ratio_array).max() <= merger_threshold_max) and (abs(np.array(merger_age_array)[np.array(merger_ratio_array).argmax()] - time_start) <= merger_misaligned_time_post):
                                    # Galaxies that reach here must be aligned in current SnapNum, become misaligned in SnapNum+1, both meet criteria, must have undergone a merger meeting criteria, within a time meeting criteria from last aligned state but only for this
                                
                                    if debug:
                                        print('HAS MERGER CRITERIA ON NEXT MISALIGNED, ', (int(SnapNum_tmp)+1))
                                    
                                    merger_time_criteria = True
                            #===================================#===================================
                            
                            
                            #-----------------------------------
                            # Update Snap, GalaxyID, and DescendantID:                           
                            SnapNum_tmp      = int(galaxy_dict['%s' %(int(SnapNum_tmp)+1)]['%s' %DescendantID_tmp]['SnapNum'])
                            GalaxyID_tmp     = int(galaxy_dict['%s' %(int(SnapNum_tmp))]['%s' %DescendantID_tmp]['GalaxyID'])
                            DescendantID_tmp = int(galaxy_dict['%s' %(int(SnapNum_tmp))]['%s' %GalaxyID_tmp]['DescendantID'])
                            
                            # Update misangle
                            misangle = float(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                            
                            #-----------------------------------
                            # This will update for as long as galaxy remains misaligned
                            SnapNum_end  = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SnapNum']
                            Redshift_end = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift']
                            time_end     = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime']
                        
                        
                        
                        
                          
                        #===========================================================
                        # Out of while loop, so descendant has become aligned again and meets criteria by default (for instant counter - already met, while loop takes care of rest). If anything not met will have a nan assigned to time_end
                        
                        # Check for galaxies existing whlie loop that aligned descendant, and has a merger that fits both the ratio and time
                        if debug:
                            if (np.isnan(time_end) == False) and (merger_time_criteria == False):
                                print('  DOES NOT MEET MERGER CRITERIA')
                        if (np.isnan(time_end) == False) and (merger_time_criteria == True):
                            
                            #-----------------------------------
                            # This will update for as long as galaxy remains misaligned
                            SnapNum_end  = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SnapNum']
                            Redshift_end = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift']
                            time_end     = galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime']
                            
                            #-----------------------------------
                            # Gather finishing stats
                            SnapNum_list.append(SnapNum_tmp)
                            GalaxyID_list.append(GalaxyID_tmp)
                            DescendantID_list.append(DescendantID_tmp)
                            Lookbacktime_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Lookbacktime'])
                            Redshift_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['Redshift'])
                            GroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['GroupNum'])
                            SubGroupNum_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['SubGroupNum'])
                            misangle_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['misangle'])
                            stelmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['stelmass'])
                            gasmass_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass'])
                            gasmass_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['gasmass_sf'])
                            halfmass_rad_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad'])
                            halfmass_rad_proj_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['halfmass_rad_proj'])
                            kappa_stars_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_stars'])
                            kappa_gas_sf_list.append(galaxy_dict['%s' %SnapNum_tmp]['%s' %GalaxyID_tmp]['kappa_gas_sf'])
                            
                            if not debug:
                                print(' ')
                            print('IN SAMPLE:   >>> ', GalaxyID_list[-1], len(SnapNum_list), ' <<<')
                            if len(SnapNum_list) == 2:
                                print('  BECAME COUNTER-ROTATING')
                            else:
                                print('  TRANSITIONED')
                            print('  TIME TAKEN TO RELAX: ', abs(time_start - time_end))
                            print('  ', SnapNum_list)
                            print('  ', GalaxyID_list)
                            print('  ', DescendantID_list)
                            print('  ', misangle_list)
                            print('  ', Lookbacktime_list)
                            print('  MERGERS: ', merger_analysis.items())
                                
                            
                            #------------------------------------
                            # Update dictionary
                            timescale_dict['%s' %GalaxyID] = {'SnapNum_list': SnapNum_list,
                                                              'GalaxyID_list': GalaxyID_list, 
                                                              'DescendantID_list': DescendantID_list,
                                                              'Lookbacktime_list': Lookbacktime_list,
                                                              'Redshift_list': Redshift_list,
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
                                                              'merger_analysis': merger_analysis}
                            #------------------------------------
                                
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
        output_input = {'csv_galaxy_dict': csv_galaxy_dict,
                        'merger_misaligned_time_pre': merger_misaligned_time_pre,
                        'merger_misaligned_time_post': merger_misaligned_time_post,
                        'merger_threshold_min': merger_threshold_min,
                        'merger_threshold_max': merger_threshold_max}
        output_input.update(output_input)
                        
        csv_dict.update({'output_input': output_input})
        
        
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%smerger_origin_r%sr%s_t%st%s_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, galaxy_dict_input['csv_sample1'], merger_threshold_min, merger_threshold_max, merger_misaligned_time_pre, merger_misaligned_time_post, galaxy_dict_input['ETG_or_LTG'], galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle'], csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%smerger_origin_r%sr%s_t%st%s_%s_%s_rad%s_proj%s_%s.csv' %(output_dir, galaxy_dict_input['csv_sample1'], merger_threshold_min, merger_threshold_max, merger_misaligned_time_pre, merger_misaligned_time_post, galaxy_dict_input['ETG_or_LTG'], galaxy_dict_input['use_angle'], galaxy_dict_input['use_hmr'], galaxy_dict_input['use_proj_angle'], csv_name))
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























#--------------------------------
# Plots for how long (time and snaps) misalignments perisit (from aligned -> stable) 
def _plot_time_spent_misaligned(csv_timescales = 'L100_timescale_tree_both_stars_gas_sf_rad2.0_projFalse_',
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
#_extract_criteria_galaxies()
#_create_merger_tree_csv()

#_analyse_misalignment_timescales()
_analyse_merger_origin_timescales()




#_create_misalignment_time_csv()
#_create_misalignment_merger_csv()

#_plot_merger_timescales()

#_plot_time_spent_misaligned()
#_plot_delta_misalignment()
#=============================



































