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
import astropy.units as u
from astropy.cosmology import z_at_value, FlatLambdaCDM
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from subhalo_main import Initial_Sample, Initial_Sample_Snip, Subhalo_Extract, Subhalo_Extract_Basic, Subhalo_Analysis
import eagleSqlTools as sql
from graphformat import set_rc_params
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================

        
#--------------------------------
# Creates a sample of galaxies given the inputs. Returns GroupNum, SubGroupNum, SnapNum, and GalaxyID for each galaxy
# SAVED: /samples/L%s_%s_cent_sample_misalignment
# 151
def _sample_misalignment(mySims = [('RefL0100N1504', 100)],
                         #--------------------------
                         galaxy_mass_min    = 10**(9.5),            # Lower mass limit within 30pkpc
                         galaxy_mass_max    = 10**(15),           # Lower mass limit within 30pkpc
                         snapNum            = 28,               # Target snapshot
                         use_satellites     = True,             # Whether to include SubGroupNum =/ 0
                         print_sample       = False,             # Print list of IDs
                         #--------------------------   
                         plot_sample        = False,
                         plot_coords        = False,
                         #-------------------------- 
                         csv_file = True,                       # Will write sample to csv file in sapmle_dir
                            csv_name = '',
                         #--------------------------     
                         print_progress = False,
                         debug = False):
                        
    #=====================================  
    # Create sample
    if print_progress:
        print('Creating sample')
        time_start = time.time()
        
    #-----------------------------------------
    # adjust mySims for serpens
    if (answer == '2') or (answer == '3') or (answer == '4') or (snapNum > 28):
        mySims = [('RefL0100N1504', 100)]
        
    # Extracting all GroupNum, SubGroupNum, GalaxyID, and SnapNum
    if snapNum > 28:     
        sample = Initial_Sample_Snip(tree_dir, mySims, snapNum, galaxy_mass_min, galaxy_mass_max, use_satellites)
    else:
        sample = Initial_Sample(mySims, snapNum, galaxy_mass_min, galaxy_mass_max, use_satellites)
    
    if debug:
        print(sample.GroupNum)
        print(sample.SubGroupNum)
        print(sample.GalaxyID)
        print(sample.SnapNum)
        print(sample.Redshift)
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
    if print_sample:
        print("  ", sample.GroupNum)
    
    print('\n===================')
    print('SAMPLE CREATED:\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(snapNum, sample.Redshift[0], galaxy_mass_min, galaxy_mass_max, use_satellites))
    print("  SAMPLE LENGTH: ", len(sample.GalaxyID))
    print('===================')
    
    
    #===================================== 
    # Creating dictionaries to gather input data
    sample_input = {'galaxy_mass_min': galaxy_mass_min,
                    'galaxy_mass_max': galaxy_mass_max,
                    'snapNum': snapNum,
                    'Redshift': sample.Redshift[0],
                    'use_satellites': use_satellites,
                    'mySims': mySims}
    if debug:
        print(sample_input.items())
    
    
    #===================================== 
    # Plotting histogram
    if plot_sample:
        plt.hist(np.log10(np.array(sample.stelmass)), bins=96, range=(8, 15))
        plt.xticks([8, 9, 10, 11, 12, 13, 14, 15])
        plt.show()
    
    if plot_coords:
        print(sample.centre[:,[0, 1]])
        
        # Plot filaments
        fig, axs = plt.subplots(1, 1, figsize=[7.0, 7.0], sharex=True, sharey=False) 
        
        axs.scatter(sample.centre[:,0], sample.centre[:,1], c='k', s=0.1)
        axs.set_xlim(0, 100)
        axs.set_ylim(0, 100)
        plt.savefig("%s/filament_plot/L%s_%s_subhalo_location.pdf" %(fig_dir, sample_input['mySims'][0][1], sample_input['snapNum']), format='pdf', bbox_inches='tight', dpi=600)    
        print("\n  %s/filament_plot/L%s_%s_subhalo_location.pdf" %(fig_dir, sample_input['mySims'][0][1], sample_input['snapNum']))
        
    
    
    #=====================================
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
        csv_dict = {'GroupNum': sample.GroupNum, 
                    'SubGroupNum': sample.SubGroupNum, 
                    'GalaxyID': sample.GalaxyID,
                    'DescendantID': sample.DescendantID,
                    'SnapNum': sample.SnapNum,
                    'Redshift': sample.Redshift,
                    'halo_mass': sample.halo_mass,
                    'centre': sample.centre,
                    'MorphoKinem': sample.MorphoKinem, 
                    'sample_input':  sample_input}
                    
        #csv_dict.update({'function_input': str(inspect.signature(_misalignment_sample))})
   
        # Writing one massive JSON file
        if use_satellites:
            json.dump(csv_dict, open('%s/L%s_%s_all_sample_misalignment_%s%s.csv' %(sample_dir, str(mySims[0][1]), str(snapNum), np.log10(galaxy_mass_min), csv_name), 'w'), cls=NumpyEncoder)
            print('\n  SAVED: %s/L%s_%s_all_sample_misalignment_%s%s.csv' %(sample_dir, str(mySims[0][1]), str(snapNum), np.log10(galaxy_mass_min), csv_name))
        if not use_satellites:
            json.dump(csv_dict, open('%s/L%s_%s_cent_sample_misalignment_%s%s.csv' %(sample_dir, str(mySims[0][1]), str(snapNum), np.log10(galaxy_mass_min), csv_name), 'w'), cls=NumpyEncoder)
            print('\n  SAVED: %s/L%s_%s_cent_sample_misalignment_%s%s.csv' %(sample_dir, str(mySims[0][1]), str(snapNum), np.log10(galaxy_mass_min), csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
   
        # Reading JSON file
        """ 
        # Loading sample
        dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    
        # Extract GroupNum etc.
        GroupNum_List       = np.array(dict_new['GroupNum'])
        SubGroupNum_List    = np.array(dict_new['SubGroupNum'])
        GalaxyID_List       = np.array(dict_new['GalaxyID'])
        DescendantID_List   = np.array(dict_new['DescendantID'])
        SnapNum_List        = np.array(dict_new['SnapNum'])
        HaloMass_List       = np.array(dict_new['halo_mass'])
        Centre_List         = np.array(dict_new['centre'])
        MorphoKinem_List    = np.array(dict_new['MorphoKinem'])
        
        
        sample_input        = dict_new['sample_input']
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
        print('\nEXTRACT:\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s' %(str(angle_selection), str(spin_hmr), str(find_uncertainties), str(rad_projected)))
        print('===================')
        """
    
  
# Creates a sample of galaxies given the inputs. Returns GroupNum, SubGroupNum, SnapNum, and GalaxyID for each galaxy
# SAVED: /samples/L%s_%s_minor_sample_misalignment
def _sample_misalignment_minor(mySims = [('RefL0012N0188', 12)],
                              #--------------------------
                              galaxy_mass_min    = 10**8,            # Lower mass limit within 30pkpc
                              galaxy_mass_max    = 10**9,           # Lower mass limit within 30pkpc
                              snapNum            = 19,               # Target snapshot
                              use_satellites     = True,             # Whether to include SubGroupNum =/ 0
                              print_sample       = False,             # Print list of IDs
                              #--------------------------   
                              csv_file = True,                       # Will write sample to csv file in sapmle_dir
                                csv_name = '',
                              #--------------------------     
                              print_progress = False,
                              debug = False):
                        
                        
    #=====================================  
    # Create sample
    if print_progress:
        print('Creating sample')
        time_start = time.time()
        
    
    #-----------------------------------------
    # adjust mySims for serpens
    if (answer == '2') or (answer == '3') or (snapNum > 28):
        mySims = [('RefL0100N1504', 100)]
        
    # Extracting all GroupNum, SubGroupNum, GalaxyID, and SnapNum
    if snapNum > 28:
        sample = Initial_Sample_Snip(tree_dir, mySims, snapNum, galaxy_mass_min, galaxy_mass_max, use_satellites)
    else:
        sample = Initial_Sample(mySims, snapNum, galaxy_mass_min, galaxy_mass_max, use_satellites)
        
    if debug:
        print(sample.GroupNum)
        print(sample.SubGroupNum)
        print(sample.GalaxyID)
        print(sample.SnapNum)
        print(sample.Redshift)
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
    if print_sample:
        print("  ", sample.GroupNum)
    
    print('\n===================')
    print('SAMPLE CREATED:\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(snapNum, sample.Redshift[0], galaxy_mass_min, galaxy_mass_max, use_satellites))
    print("  SAMPLE LENGTH: ", len(sample.GalaxyID))
    print('===================')
    
    
    #===================================== 
    # Creating dictionaries to gather input data
    sample_input = {'galaxy_mass_min': galaxy_mass_min,
                    'galaxy_mass_max': galaxy_mass_max,
                    'snapNum': snapNum,
                    'Redshift': sample.Redshift[0],
                    'use_satellites': use_satellites,
                    'mySims': mySims}
    if debug:
        print(sample_input.items())
    
    
    #=====================================
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
        csv_dict = {'GroupNum': sample.GroupNum, 
                    'SubGroupNum': sample.SubGroupNum, 
                    'GalaxyID': sample.GalaxyID,
                    'DescendantID': sample.DescendantID,
                    'SnapNum': sample.SnapNum,
                    'Redshift': sample.Redshift,
                    'halo_mass': sample.halo_mass,
                    'centre': sample.centre,
                    'sample_input':  sample_input}
                    
        #csv_dict.update({'function_input': str(inspect.signature(_misalignment_sample))})
   
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/L%s_%s_minor_sample_misalignment_%s%s.csv' %(sample_dir, str(mySims[0][1]), str(snapNum), np.log10(galaxy_mass_min), csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/L%s_%s_minor_sample_misalignment_%s%s.csv' %(sample_dir, str(mySims[0][1]), str(snapNum), np.log10(galaxy_mass_min), csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
   
        # Reading JSON file
        """ 
        # Loading sample
        dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    
        # Extract GroupNum etc.
        GroupNum_List       = np.array(dict_new['GroupNum'])
        SubGroupNum_List    = np.array(dict_new['SubGroupNum'])
        GalaxyID_List       = np.array(dict_new['GalaxyID'])
        DescendantID_List   = np.array(dict_new['DescendantID'])
        SnapNum_List        = np.array(dict_new['SnapNum'])
        HaloMass_List       = np.array(dict_new['halo_mass'])
        Centre_List         = np.array(dict_new['centre'])
        
        
        sample_input        = dict_new['sample_input']
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print(sample_input)
            print(GroupNum_List)
            print(SubGroupNum_List)
            print(GalaxyID_List)
            print(SnapNum_List)
       
       
        print('\n===================')
        print('SAMPLE MINOR LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_limit'], sample_input['use_satellites']))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('\nEXTRACT:\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s' %(str(angle_selection), str(spin_hmr), str(find_uncertainties), str(rad_projected)))
        print('===================')
        """
    


#--------------------------------
# Will read in existing sample and modify it based on what galaxies ended up being part of the sample . Largely intended for snipshots
def _sample_modify(csv_sample = 'L12_28_all_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                   csv_output = 'L12_28_all_sample_misalignment_9.0_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                   #--------------------------
                   print_modifications = True,
                   #--------------------------
                   csv_file       = True,              # Will write sample to csv file in sample_dir
                     csv_name     = '',                 # extra stuff at end
                   #--------------------------
                   print_progress = False,
                   debug = False):
                   
    
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
    DescendantID_List   = np.array(dict_sample['DescendantID'])
    Redshift_List       = np.array(dict_sample['Redshift'])
    halo_mass_List      = np.array(dict_sample['halo_mass'])
    centre_List         = np.array(dict_sample['centre'])
    MorphoKinem_List    = np.array(dict_sample['MorphoKinem'])
        
    # Loading output
    dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    all_general         = dict_output['all_general']
    all_flags           = dict_output['all_flags']
    
    # Loading sample criteria
    sample_input        = dict_sample['sample_input']
    output_input        = dict_output['output_input']
    trim_stelmass       = sample_input['galaxy_mass_min']
    
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    if debug:
        print(sample_input)
        print(GroupNum_List)
        print(SubGroupNum_List)
        print(GalaxyID_List)
        print(SnapNum_List)
   
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift'], output_input['galaxy_mass_min'], output_input['galaxy_mass_max']))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('===================')
    
    
    #=====================================
    GroupNum_new        = []
    SubGroupNum_new     = []
    GalaxyID_new        = []
    SnapNum_new         = []
    DescendantID_new    = []
    Redshift_new        = []
    halo_mass_new       = []
    centre_new          = []
    MorphoKinem_new      = []
    
    # Cycle over all galaxyIDs in sample. If same galaxy met mass criteria in output, append to new file
    for GroupNum, SubGroupNum, GalaxyID, SnapNum, DescendantID, Redshift, halo_mass, centre, MorphoKinem in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List, DescendantID_List, Redshift_List, halo_mass_List, centre_List, MorphoKinem_List), total=len(GroupNum_List)):
        if all_general['%s' %GalaxyID]['stelmass'] >= trim_stelmass:
            
            # Append to new sample array
            GroupNum_new.append(GroupNum)
            SubGroupNum_new.append(SubGroupNum)
            GalaxyID_new.append(GalaxyID)
            SnapNum_new.append(SnapNum)
            DescendantID_new.append(DescendantID)
            Redshift_new.append(Redshift)
            halo_mass_new.append(halo_mass)
            centre_new.append(centre)
            MorphoKinem_new.append(MorphoKinem)
            
            if debug:
                print('GalaxyID: %s\t met with %s' %(GalaxyID, all_general['%s' %GalaxyID]['stelmass']))

    if print_modifications:
        print('Original sample: ', len(GroupNum_List))
        print('     New sample: ', len(GroupNum_new))
        
    
    #=====================================
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
        csv_dict = {'GroupNum': GroupNum_new, 
                    'SubGroupNum': SubGroupNum_new, 
                    'GalaxyID':  GalaxyID_new,
                    'DescendantID':  DescendantID_new,
                    'SnapNum':  SnapNum_new,
                    'Redshift':  Redshift_new,
                    'halo_mass':  halo_mass_new,
                    'centre':  centre_new,
                    'MorphoKinem':  MorphoKinem_new, 
                    'sample_input':  sample_input}
                    
        #csv_dict.update({'function_input': str(inspect.signature(_misalignment_sample))})
   
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%s%s.csv' %(sample_dir, csv_sample, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s%s.csv' %(sample_dir, csv_sample, csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
 
    


#--------------------------------
# Reads in a sample file, and does all relevant calculations, and exports as csv file
# SAVED: /outputs/%s_%s_%s_%s...
def _analysis_misalignment_distribution(csv_sample = 'L100_151_all_sample_misalignment_9.5',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                        #--------------------------
                                        # Galaxy extraction properties
                                        viewing_axis        = 'z',                  # Which axis to view galaxy from.  DEFAULT 'z'
                                        aperture_rad        = 30,                   # trim all data to this maximum value before calculations [pkpc]
                                        kappa_rad           = 30,                   # calculate kappa for this radius [pkpc]
                                        trim_rad = np.array([30]),                 # keep as 100... will be capped by aperture anyway. Doesn't matter
                                        align_rad = False,                          # keep on False
                                        orientate_to_axis='z',                      # Keep as z
                                        viewing_angle = 0,                          # Keep as 0
                                        #-----------------------------------------------------
                                        # Misalignments we want extracted and at which radii
                                        angle_selection     = ['stars_gas',            # stars_gas     stars_gas_sf
                                                               'stars_gas_sf',         # stars_dm      gas_dm        gas_sf_dm       gas_nsf_dm        
                                                               'gas_sf_gas_nsf',       # gas_sf_gas_nsf
                                                               'stars_dm',
                                                               'gas_dm',
                                                               'gas_sf_dm'],           
                                        spin_hmr            = np.array([1.0, 2.0]),          # multiples of hmr for which to find spin
                                        find_uncertainties  = True,                    # whether to find 2D and 3D uncertainties
                                        rad_projected       = False,                     # whether to use rad in projection or 3D
                                        #--------------------------
                                        # Selection criteria
                                        com_min_distance    = 2.0,         # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                                        min_particles       = 20,          # Minimum particles to find spin.  DEFAULT 10
                                        min_inclination     = 0,           # Minimum inclination toward viewing axis [deg] DEFAULT 0
                                        #--------------------------   
                                        csv_file       = True,             # Will write sample to csv file in sample_dir
                                          csv_name     = '',               # extra stuff at end
                                        #--------------------------
                                        print_progress = False,
                                        print_galaxy   = False,
                                        debug = False):
    
    
    #---------------------------------------------    
    # Load sample csv
    if print_progress:
        print('Loading initial sample')
        time_start = time.time()
        
    
    # Loading sample
    dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    
    # Extract GroupNum etc.
    GroupNum_List       = np.array(dict_new['GroupNum'])
    SubGroupNum_List    = np.array(dict_new['SubGroupNum'])
    GalaxyID_List       = np.array(dict_new['GalaxyID'])
    SnapNum_List        = np.array(dict_new['SnapNum'])
    Redshift_List       = np.array(dict_new['Redshift'])
    HaloMass_List       = np.array(dict_new['halo_mass'])
    Centre_List         = np.array(dict_new['centre'])
    MorphoKinem_List    = np.array(dict_new['MorphoKinem'])
    sample_input        = dict_new['sample_input']
    
    
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    if debug:
        print(sample_input)
        print(GroupNum_List)
        print(SubGroupNum_List)
        print(GalaxyID_List)
        print(SnapNum_List)
        print(HaloMass_List)
        print(Centre_List)
        print(MorphoKinem_List)
       
       
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_min'], sample_input['galaxy_mass_max'], sample_input['use_satellites']))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nEXTRACT:\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s' %(str(angle_selection), str(spin_hmr), str(find_uncertainties), str(rad_projected)))
    print('===================')
    
    
    output_input = {'angle_selection': angle_selection,
                    'spin_hmr': spin_hmr,
                    'find_uncertainties': find_uncertainties,
                    'rad_projected': rad_projected,
                    'viewing_axis': viewing_axis,
                    'aperture_rad': aperture_rad,
                    'kappa_rad': kappa_rad,
                    'com_min_distance': com_min_distance,
                    'min_particles': min_particles,
                    'min_inclination': min_inclination}
    output_input.update(sample_input)
    

    
    
    #---------------------------------------------
    # Empty dictionaries to collect relevant data
    all_flags         = {}          # has reason why galaxy failed sample
    all_general       = {}          # has total masses, kappa, halfmassrad, etc.
    all_coms          = {}          # has all C.o.Ms
    all_spins         = {}          # has all spins
    all_l             = {}          # has all specific l
    all_counts        = {}          # has all the particle count within rad
    all_masses        = {}          # has all the particle mass within rad
    all_totmass       = {}          # has total masses at hmr
    all_sfr           = {}          # has bulk sfr
    all_Z             = {}          # has bulk metallicity
    all_misangles     = {}          # has all 3D angles
    all_misanglesproj = {}          # has all 2D projected angles from 3d when given a viewing axis and viewing_angle = 0
    all_gasdata       = {}
    
    
    #============================================
    # Manual entry
    #GroupNum_List = [3]
    #SubGroupNum_List = [3]
    #GalaxyID_List = [14916079]
    #SnapNum_List = [28]
    #====================
    
    
    #=================================================================== 
    # Run analysis for each individual galaxy in loaded sample
    for GroupNum, SubGroupNum, GalaxyID, SnapNum, Redshift, HaloMass, Centre, MorphoKinem in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List, Redshift_List, HaloMass_List, Centre_List, MorphoKinem_List), total=len(GroupNum_List)):
        
        
        #-----------------------------
        if print_progress:
            print('Extracting particle data Subhalo_Extract()')
            time_start = time.time()
            
        # Initial extraction of galaxy particle data
        galaxy = Subhalo_Extract(sample_input['mySims'], dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, Centre, HaloMass, aperture_rad, viewing_axis, MorphoKinem)
        GroupNum = galaxy.gn

        if debug:
            print(galaxy.gn, galaxy.sgn, galaxy.centre, galaxy.halfmass_rad, galaxy.halfmass_rad_proj)
    
        
        
        #-----------------------------
        # Begin subhalo analysis
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Running particle data analysis Subhalo_Analysis()')
            time_start = time.time()
       
        # Set spin_rad here
        if rad_projected == True:
            spin_rad = np.array(spin_hmr) * galaxy.halfmass_rad_proj
            spin_hmr_tmp = spin_hmr
        
            # Reduce spin_rad array if value exceeds aperture_rad_in... means not all dictionaries will have same number of array spin values
            spin_rad_in_tmp = [x for x in spin_rad if x <= aperture_rad]
            spin_hmr_in_tmp = [x for x in spin_hmr if x*galaxy.halfmass_rad_proj <= aperture_rad]
            
            # Ensure min. rad is >1 pkpc
            spin_rad_in = [x for x in spin_rad_in_tmp if x >= 1.0]
            spin_hmr_in = [x for x in spin_hmr_in_tmp if x*galaxy.halfmass_rad_proj >= 1.0]
        
            if len(spin_hmr_in) != len(spin_hmr_tmp):
                print('Capped spin_rad: %.2f - %.2f - %.2f HMR | Min/Max %.2f / %.2f pkpc' %(min(spin_hmr_in), (max(spin_hmr_in) - min(spin_hmr_in))/(len(spin_hmr_in) - 1), max(spin_hmr_in), min(spin_rad_in), max(spin_rad_in)))
        elif rad_projected == False:
            spin_rad = np.array(spin_hmr) * galaxy.halfmass_rad
            spin_hmr_tmp = spin_hmr
        
            # Reduce spin_rad array if value exceeds aperture_rad_in... means not all dictionaries will have same number of array spin values
            spin_rad_in_tmp = [x for x in spin_rad if x <= aperture_rad]
            spin_hmr_in_tmp = [x for x in spin_hmr if x*galaxy.halfmass_rad <= aperture_rad]
            
            # Ensure min. rad is >1 pkpc
            spin_rad_in = [x for x in spin_rad_in_tmp if x >= 1.0]
            spin_hmr_in = [x for x in spin_hmr_in_tmp if x*galaxy.halfmass_rad >= 1.0]
        
            if len(spin_hmr_in) != len(spin_hmr_tmp):
                print('Capped spin_rad: %.2f - %.2f - %.2f HMR | Min/Max %.2f / %.2f pkpc' %(min(spin_hmr_in), (max(spin_hmr_in) - min(spin_hmr_in))/(len(spin_hmr_in) - 1), max(spin_hmr_in), min(spin_rad_in), max(spin_rad_in)))
        
        # If we want the original values, enter 0 for viewing angle
        subhalo = Subhalo_Analysis(sample_input['mySims'], GroupNum, SubGroupNum, GalaxyID, SnapNum, galaxy.MorphoKinem, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.data_nil,
                                            viewing_axis,
                                            aperture_rad,
                                            kappa_rad, 
                                            trim_rad, 
                                            align_rad,              #align_rad = False
                                            orientate_to_axis,
                                            viewing_angle,
                                            
                                            angle_selection,        
                                            spin_rad_in,
                                            spin_hmr_in,
                                            find_uncertainties,
                                            rad_projected,
                                            
                                            com_min_distance,
                                            min_particles,                                            
                                            min_inclination)
          
        
        #print('GalaxyID asdasd', GalaxyID)
        #print(subhalo.sfr['hmr'])
        #print(np.multiply(subhalo.sfr['gas_sf'], 3.154e+7))
        #print(subhalo.Z['hmr'])
        #print(subhalo.Z['stars'])
        #print(subhalo.Z['gas_sf'])
        
        
        """ FLAGS
        ------------
        #print(subhalo.flags['total_particles'])            # will flag if there are missing particles within aperture_rad
        #print(subhalo.flags['min_particles'])              # will flag if min. particles not met within spin_rad (will find spin if particles exist, but no uncertainties)
        #print(subhalo.flags['min_inclination'])            # will flag if inclination angle not met within spin_rad... all spins and uncertainties still calculated
        #print(subhalo.flags['com_min_distance'])           # will flag if com distance not met within spin_rad... all spins and uncertainties still calculated
        ------------
        """
        
        #--------------------------------
        # Collecting all relevant particle info for galaxy
        all_flags['%s' %str(subhalo.GalaxyID)]          = subhalo.flags
        all_general['%s' %str(subhalo.GalaxyID)]        = subhalo.general
        all_l['%s' %str(subhalo.GalaxyID)]              = subhalo.l
        all_spins['%s' %str(subhalo.GalaxyID)]          = subhalo.spins
        all_coms['%s' %str(subhalo.GalaxyID)]           = subhalo.coms
        all_counts['%s' %str(subhalo.GalaxyID)]         = subhalo.counts
        all_masses['%s' %str(subhalo.GalaxyID)]         = subhalo.masses
        all_totmass['%s' %str(subhalo.GalaxyID)]        = subhalo.tot_mass
        all_sfr['%s' %str(subhalo.GalaxyID)]            = subhalo.sfr
        all_Z['%s' %str(subhalo.GalaxyID)]              = subhalo.Z
        all_misangles['%s' %str(subhalo.GalaxyID)]      = subhalo.mis_angles
        all_misanglesproj['%s' %str(subhalo.GalaxyID)]  = subhalo.mis_angles_proj
        all_gasdata['%s' %str(subhalo.GalaxyID)]        = subhalo.gas_data
        #all_massflow... not added as we can't evaluate it here
        #---------------------------------
          
        if print_galaxy:
            print('|%s| |ID:   %s\t|M*:  %.2e  |HMR:  %.2f  |KAPPA:  %.2f' %(SnapNum, str(subhalo.GalaxyID), subhalo.stelmass, subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'])) 
            
            
    #=====================================
    # End of individual subhalo loop
    
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
                    'all_l': all_l,
                    'all_spins': all_spins,
                    'all_coms': all_coms,
                    'all_counts': all_counts,
                    'all_masses': all_masses,
                    'all_totmass': all_totmass,
                    'all_sfr': all_sfr,
                    'all_Z': all_Z,
                    'all_misangles': all_misangles,
                    'all_misanglesproj': all_misanglesproj, 
                    'all_gasdata': all_gasdata,
                    'all_flags': all_flags,
                    'output_input': output_input}
        #csv_dict.update({'function_input': str(inspect.signature(_misalignment_distribution))})
        
        #-----------------------------
        # File names
        angle_str = ''
        for angle_name in list(angle_selection):
            angle_str = '%s_%s' %(str(angle_str), str(angle_name))
            
        uncertainty_str = 'noErr'    
        if find_uncertainties:
            uncertainty_str = 'Err'   
            
        rad_str = 'Rad'    
        if rad_projected:
            rad_str = 'RadProj'
        
            
        
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%s_%s_%s_%s_%s.csv' %(output_dir, csv_sample, rad_str, uncertainty_str, angle_str, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s_%s_%s_%s_%s.csv' %(output_dir, csv_sample, rad_str, uncertainty_str, angle_str, csv_name))
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
        print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_limit'], sample_input['use_satellites']))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
        print('\nPLOT:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s\n  Upper mass limit: %s\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field))
        print('===================')
        """
        

# Reads in a sample file, and does all relevant calculations, and exports as csv file
# SAVED: /outputs/%s csv_sample
def _analysis_misalignment_minor(csv_sample = 'L100_19_minor_sample_misalignment_8.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                 #--------------------------
                                 # Galaxy extraction properties
                                 aperture_rad        = 30,                   # trim all data to this maximum value before calculations [pkpc]
                                 #--------------------------   
                                 csv_file       = True,             # Will write sample to csv file in sample_dir
                                   csv_name     = '',               # extra stuff at end
                                 #--------------------------
                                 print_progress = False,
                                 print_galaxy   = False,
                                 debug = False):
    
    
    #---------------------------------------------    
    # Load sample csv
    if print_progress:
        print('Loading initial sample')
        time_start = time.time()
        
    
    # Loading sample
    dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    
    # Extract GroupNum etc.
    GroupNum_List       = np.array(dict_new['GroupNum'])
    SubGroupNum_List    = np.array(dict_new['SubGroupNum'])
    GalaxyID_List       = np.array(dict_new['GalaxyID'])
    SnapNum_List        = np.array(dict_new['SnapNum'])
    Redshift_List       = np.array(dict_new['Redshift'])
    HaloMass_List       = np.array(dict_new['halo_mass'])
    Centre_List         = np.array(dict_new['centre'])
    sample_input        = dict_new['sample_input']
    
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    if debug:
        print(sample_input)
        print(GroupNum_List)
        print(SubGroupNum_List)
        print(GalaxyID_List)
        print(SnapNum_List)
        print(HaloMass_List)
        print(Centre_List)
       
       
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_min'], sample_input['galaxy_mass_max'], sample_input['use_satellites']))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nEXTRACT:\n  Masses.') 
    print('===================')
    
    
    output_input = {'aperture_rad': aperture_rad}
    output_input.update(sample_input)
    

    #---------------------------------------------
    # Empty dictionaries to collect relevant data
    all_general       = {}          # has total masses, kappa, halfmassrad, etc.
    
    #============================================
    # Manual entry
    #GroupNum_List = [3]
    #SubGroupNum_List = [3]
    #GalaxyID_List = [14916079]
    #SnapNum_List = [28]
    #====================
    
    
    #=================================================================== 
    # Run analysis for each individual galaxy in loaded sample
    for GroupNum, SubGroupNum, GalaxyID, SnapNum, Redshift, HaloMass, Centre in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List, Redshift_List, HaloMass_List, Centre_List), total=len(GroupNum_List)):
        
        #-----------------------------
        if print_progress:
            print('Running basis analysis')
            time_start = time.time()
            
        # Initial extraction of galaxy particle data
        subhalo = Subhalo_Extract_Basic(sample_input['mySims'], dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, GalaxyID, Centre, HaloMass, aperture_rad)
        GroupNum = galaxy.gn
        
        if debug:
            print(subhalo.gn, subhalo.sgn)
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
       
        
        #--------------------------------
        # Collecting all relevant particle info for galaxy
        all_general['%s' %str(subhalo.GalaxyID)]        = subhalo.general
        #---------------------------------
          
        if print_galaxy:
            print('|%s| |ID:   %s\t|M*:  %.2e  |M_sf:  %.2f' %(SnapNum, str(subhalo.GalaxyID), subhalo.stelmass, subhalo.gasmass_sf)) 
            
    
    #=====================================
    # End of individual subhalo loop
    
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
                    'output_input': output_input}
        #csv_dict.update({'function_input': str(inspect.signature(_misalignment_distribution))})
        
        #-----------------------------        
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%s_%s.csv' %(output_dir, csv_sample, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%s_%s.csv' %(output_dir, csv_sample, csv_name))
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
        print('MINOR SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_limit'], sample_input['use_satellites']))
        print('  MINOR SAMPLE LENGTH: ', len(GroupNum_List))
        print('===================')
        """
        


#===========================    
#_sample_misalignment(snapNum = 190)
#_sample_misalignment_minor(snapNum = int(snap_i))

#_sample_modify()

#_analysis_misalignment_minor()
#_analysis_misalignment_distribution()


#for snap_i in np.arange(19, 28, 1):
#    _sample_misalignment(snapNum = snap_i)
#    _analysis_misalignment_distribution(csv_sample = 'L100_%s_all_sample_misalignment_9.5' %snap_i)
  



    
#===========================


#for snap_i in np.arange(134, 145, 1):
#for snap_i in np.arange(145, 155, 1):
#for snap_i in np.arange(155, 165, 1):
#for snap_i in np.arange(165, 175, 1):
#for snap_i in np.arange(175, 185, 1):
#for snap_i in np.arange(185, 195, 1):
#for snap_i in np.arange(195, 201, 1):
#    _sample_misalignment(snapNum = int(snap_i), galaxy_mass_min    = 10**(9.5), galaxy_mass_max    = 10**(15))
#    _analysis_misalignment_distribution(csv_sample = 'L100_' + str(int(snap_i)) + '_all_sample_misalignment_9.5')
    

#===========================






