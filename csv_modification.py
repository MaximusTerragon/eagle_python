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
answer = input("-----------------\nDirectories?:\n     l local\n     a serpens_snap\n     i snip\n")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================



# Modifies existing csv output file by adding or removing relevant fields
def _modify_sample_csv(csv_sample = '#L12_20_all_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                       #--------------------------   
                       csv_file = True,                       # Will write sample to csv file in sapmle_dir
                         csv_name = '',
                       #--------------------------
                       print_progress = False,
                       debug = False):
                       
    #---------------------------------------------    
    # Load sample csv
    if print_progress:
        print('Loading initial sample')
        time_start = time.time()
        
    
    # Loading sample
    dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    sample_input_old = dict_new['sample_input']
    
    #===============================================
    # prompt confirmation
    answer = input("\nContinue?")
    if answer.lower() in ["y","yes"]:
        
        # Modify sample_input
         print(sample_input_old.keys())
         print(sample_input_old['mySims'])
         sample_input = {'galaxy_mass_min': sample_input_old['galaxy_mass_min'],
                         'galaxy_mass_max': sample_input_old['galaxy_mass_max'],
                         'snapNum': sample_input_old['snapNum'],
                         'Redshift': sample_input_old['Redshift'],
                         'use_satellites': sample_input_old['use_satellites'],
                         'mySims': [[sample_input_old['mySims'][0][0], sample_input_old['mySims'][0][1], 'snap']]}
         
         print(sample_input['mySims'])
         
         
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
                 print('Writing CSV sample')
                 time_start = time.time()
        
             # Combining all dictionaries
             csv_dict = dict_new
             csv_dict['sample_input'] = sample_input
   
             #-----------------------------
             # Writing one massive JSON file
             json.dump(csv_dict, open('%s/%s.csv' %(sample_dir, csv_sample), 'w'), cls=NumpyEncoder)
             print('\n  SAVED: %s/%s.csv' %(sample_dir, csv_sample))
         
         
         
         

         
    elif answer.lower() in ["n","no"]:
         raise Exception('Terminated')
    else:
         raise Exception('Terminated')
    
    
# Modifies existing csv output file by adding or removing relevant fields
def _modify_misalignment_csv(csv_sample = '#L100_28_all_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                             csv_output = '_RadProj_noErr__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                             csv_output2 = 'L12_TEMP_28_all_sample_misalignment_9.0_TEMP_RadProj_noErr__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                             #--------------------------   
                             csv_file = True,                       # Will write sample to csv file in sapmle_dir
                               csv_name = '',
                             #--------------------------
                             print_progress = False,
                             debug = False):

    # Ensuring the sample and output originated together
    csv_output = csv_sample + csv_output 
    
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
    Redshift_List       = np.array(dict_sample['Redshift'])
    HaloMass_List       = np.array(dict_sample['halo_mass'])
    Centre_List         = np.array(dict_sample['centre'])
    MorphoKinem_List    = np.array(dict_sample['MorphoKinem'])
    
        
    # Loading output
    dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    old_general         = dict_output['all_general']
    old_spins           = dict_output['all_spins']
    old_coms            = dict_output['all_coms']
    old_counts          = dict_output['all_counts']
    old_masses          = dict_output['all_masses']
    old_misangles       = dict_output['all_misangles']
    old_misanglesproj   = dict_output['all_misanglesproj']
    old_gasdata         = dict_output['all_gasdata']
    
    
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
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Lower mass limit: %.2E M*\n  Upper mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_min'], sample_input['galaxy_mass_max'], sample_input['use_satellites']))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('===================')
    
    #===============================================
    # prompt confirmation
    answer = input("\nContinue?")
    if answer.lower() in ["y","yes"]:
         #===============================================
         # MAKE MODIFICATIONS HERE
    
         #------------------------
         # Update sims
         """output_input.update(sample_input)
         """

         #------------------------
         # Update lists
    
         """for GalaxyID in old_general.keys():
             old_general['%s' %GalaxyID]['SnapNum'] = 28"""
    
         #------------------------
         # SPLICE CSVS
         """# Loading output
         dict_output2 = json.load(open('%s/%s.csv' %(output_dir, csv_output2), 'r'))
         old2_general         = dict_output2['all_general']
         old2_counts          = dict_output2['all_counts']
         old2_masses          = dict_output2['all_masses']
         old2_misangles       = dict_output2['all_misangles']
         old2_misanglesproj   = dict_output2['all_misanglesproj']
         old2_flags           = dict_output2['all_flags']

         print('OLD:')
         print(old_misangles['12431873']['hmr'])
         print(old_misangles['8230966']['hmr'])
         print(old_misangles['11861302']['hmr'])
    
         # old_ is my incomplete csv array
         # old2_ is my shorter, updated csv array
         for GalaxyID in old2_general.keys():
             old_general['%s' %GalaxyID] = old2_general['%s' %GalaxyID]
             old_counts['%s' %GalaxyID]  = old2_counts['%s' %GalaxyID]
             old_masses['%s' %GalaxyID]  = old2_masses['%s' %GalaxyID]
             old_misangles['%s' %GalaxyID] = old2_misangles['%s' %GalaxyID]
             old_misanglesproj['%s' %GalaxyID] = old2_misanglesproj['%s' %GalaxyID]
             old_flags['%s' %GalaxyID] = old2_flags['%s' %GalaxyID]

    
         print('NEW:')
         print(old_misangles['12431873']['hmr'])
         print(old_misangles['8230966']['hmr'])
         print(old_misangles['11861302']['hmr'])
         """

    
    
    
         # Create _main to analyse smaller galaxies of mass 10^8, append them to sample anyway
         # Check whether we get same results

         #================================================
         # Finish modifications
         all_general         = old_general
         all_spins           = old_spins
         all_coms            = old_coms
         all_counts          = old_counts
         all_masses          = old_masses
         all_misangles       = old_misangles
         all_misanglesproj   = old_misanglesproj
         all_gasdata         = old_gasdata
         all_flags           = old_flags
    
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
             csv_dict = {'all_general': all_general,
                         'all_spins': all_spins,
                         'all_coms': all_coms,
                         'all_counts': all_counts,
                         'all_masses': all_masses,
                         'all_misangles': all_misangles,
                         'all_misanglesproj': all_misanglesproj, 
                         'all_gasdata': all_gasdata,
                         'all_flags': all_flags,
                         'output_input': output_input}
             #csv_dict.update({'function_input': str(inspect.signature(_misalignment_distribution))})
    
             #-----------------------------
             # Writing one massive JSON file
             json.dump(csv_dict, open('%s/MOD_%s.csv' %(output_dir, csv_output), 'w'), cls=NumpyEncoder)
             print('\n  SAVED: %s/MOD_%s.csv' %(output_dir, csv_output))
             if print_progress:
                 print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
             # Reading JSON file
             """ 
             # Ensuring the sample and output originated together
             csv_output = csv_sample + csv_output
    
             # Loading sample
             dict_sample = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
             GroupNum_List       = np.array(dict_sample['GroupNum'])
             SubGroupNum_List    = np.array(dict_sample['SubGroupNum'])
             GalaxyID_List       = np.array(dict_sample['GalaxyID'])
             SnapNum_List        = np.array(dict_sample['SnapNum'])
    
             # Loading output
             dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
             all_general         = dict_output['all_general']
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
             print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_limit'], sample_input['use_satellites']))
             print('  SAMPLE LENGTH: ', len(GroupNum_List))
             print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
             print('\nPLOT:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s\n  Upper mass limit: %s\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
             print('===================')
             """
         #================================================  

         
    elif answer.lower() in ["n","no"]:
         raise Exception('Terminated')
    else:
         raise Exception('Terminated')
    
    
# Modifies existing csv output file by adding or removing relevant fields
def _modify_radial_csv(csv_output = '#L12_radial_ID37445_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                       #--------------------------   
                       csv_file = True,                       # Will write sample to csv file in sapmle_dir
                         csv_name = '',
                       #--------------------------
                       print_progress = False,
                       debug = False):

    
    #================================================  
    # Load sample csv
    if print_progress:
        print('Loading initial sample')
        time_start = time.time()
    
    #--------------------------------    
    # Loading output
    dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    old_general         = dict_output['all_general']
    old_counts          = dict_output['all_counts']
    old_masses          = dict_output['all_masses']
    old_misangles       = dict_output['all_misangles']
    old_misanglesproj   = dict_output['all_misanglesproj']
    old_flags           = dict_output['all_flags']
    
    # Loading sample criteria
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
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('===================')
    
    
    #===============================================
    # MAKE MODIFICATIONS HERE
    
    
    # Update sims
    """mySims = [('RefL0012N0188', 12)]
    output_input.update({'mySims': mySims})
    """

    # Update lists
    




    # Finish modifications
    all_general         = old_general
    all_counts          = old_counts
    all_masses          = old_masses
    all_misangles       = old_misangles
    all_misanglesproj   = old_misanglesproj
    all_flags           = old_flags
    
    #===================================================================
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
        csv_dict = {'all_general': all_general,
                    'all_counts': all_counts,
                    'all_masses': all_masses,
                    'all_misangles': all_misangles,
                    'all_misanglesproj': all_misanglesproj, 
                    'all_flags': all_flags,
                    'output_input': output_input}
        #csv_dict.update({'function_input': str(inspect.signature(_misalignment_distribution))})
    
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/MOD_%s.csv' %(output_dir, csv_output), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/MOD_%s.csv' %(output_dir, csv_output))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
        # Reading JSON file
        """ 
        # Ensuring the sample and output originated together
        csv_output = csv_sample + csv_output
    
        # Loading sample
        dict_sample = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
        GroupNum_List       = np.array(dict_sample['GroupNum'])
        SubGroupNum_List    = np.array(dict_sample['SubGroupNum'])
        GalaxyID_List       = np.array(dict_sample['GalaxyID'])
        SnapNum_List        = np.array(dict_sample['SnapNum'])
    
        # Loading output
        dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
        all_general         = dict_output['all_general']
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
        print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_limit'], sample_input['use_satellites']))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
        print('\nPLOT:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s\n  Upper mass limit: %s\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
        print('===================')
        """
    

#==========================
#_modify_sample_csv()
#_modify_misalignment_csv()
#_modify_radial_csv()
#==========================










