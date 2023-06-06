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
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Extract_Basic, Subhalo_Analysis
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
# Creates a sample of galaxies given the inputs. Returns GroupNum, SubGroupNum, SnapNum, and GalaxyID for each galaxy
def _sample_misalignment(mySims = [('RefL0012N0188', 12)],
                         #--------------------------
                         galaxy_mass_min    = 10**9,            # Lower mass limit within 30pkpc
                         galaxy_mass_max    = 10**15,           # Lower mass limit within 30pkpc
                         snapNum            = 28,               # Target snapshot
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
        
    
    # Extracting all GroupNum, SubGroupNum, GalaxyID, and SnapNum
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
    print('SAMPLE CREATED:\n  SnapNum: %s\n  Redshift: %s\n  Min Mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(snapNum, sample.Redshift[0], galaxy_mass_min, galaxy_mass_max, use_satellites))
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
                    'MorphoKinem': sample.MorphoKinem, 
                    'sample_input':  sample_input}
                    
        csv_dict.update({'function_input': str(inspect.signature(_misalignment_sample))})
   
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
def _sample_misalignment_minor(mySims = [('RefL0012N0188', 12)],
                              #--------------------------
                              galaxy_mass_min    = 10**8,            # Lower mass limit within 30pkpc
                              galaxy_mass_max    = 10**9,           # Lower mass limit within 30pkpc
                              snapNum            = 28,               # Target snapshot
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
        
    
    # Extracting all GroupNum, SubGroupNum, GalaxyID, and SnapNum
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
    print('SAMPLE CREATED:\n  SnapNum: %s\n  Redshift: %s\n  Min Mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(snapNum, sample.Redshift[0], galaxy_mass_min, galaxy_mass_max, use_satellites))
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
# Reads in a sample file, and does all relevant calculations, and exports as csv file
def _analysis_misalignment_minor(csv_sample = 'L12_28_minor_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                 #--------------------------
                                 # Galaxy extraction properties
                                 aperture_rad        = 30,                   # trim all data to this maximum value before calculations [pkpc]
                                 #--------------------------   
                                 csv_file       = False,             # Will write sample to csv file in sample_dir
                                   csv_name     = '',               # extra stuff at end
                                 #--------------------------
                                 print_progress = False,
                                 print_galaxy   = True,
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
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_min'], sample_input['use_satellites']))
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
        # Gives: subhalo.general: GroupNum, SubGroupNum, GalaxyID, stelmass, gasmass, gasmass_sf, gasmass_nsf
        
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
        

# Reads in a sample file, and does all relevant calculations, and exports as csv file
def _analysis_misalignment_distribution(csv_sample = 'L12_19_all_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
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
                                        angle_selection     = ['stars_gas',            # stars_gas     stars_gas_sf    stars_gas_nsf
                                                               'stars_gas_sf',         # gas_dm        gas_sf_dm       gas_nsf_dm
                                                               'stars_gas_nsf',        # gas_sf_gas_nsf
                                                               'gas_sf_gas_nsf',
                                                               'stars_dm'],           
                                        spin_hmr            = np.array([1.0, 2.0]),          # multiples of hmr for which to find spin
                                        find_uncertainties  = True,                    # whether to find 2D and 3D uncertainties
                                        rad_projected       = True,                     # whether to use rad in projection or 3D
                                        #--------------------------
                                        # Selection criteria
                                        com_min_distance    = 2.0,         # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                                        min_particles       = 20,          # Minimum particles to find spin.  DEFAULT 20
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
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_limit'], sample_input['use_satellites']))
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
    all_counts        = {}          # has all the particle count within rad
    all_masses        = {}          # has all the particle mass within rad
    all_misangles     = {}          # has all 3D angles
    all_misanglesproj = {}          # has all 2D projected angles from 3d when given a viewing axis and viewing_angle = 0
    
    
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
        galaxy = Subhalo_Extract(sample_input['mySims'], dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, Centre, HaloMass, aperture_rad, viewing_axis)
        # Gives: galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh
        
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
        subhalo = Subhalo_Analysis(sample_input['mySims'], GroupNum, SubGroupNum, GalaxyID, SnapNum, MorphoKinem, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.data_nil,
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
                                            
                                            com_min_distance,
                                            min_particles,                                            
                                            min_inclination)
          
          
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
        all_spins['%s' %str(subhalo.GalaxyID)]          = subhalo.spins
        all_coms['%s' %str(subhalo.GalaxyID)]           = subhalo.coms
        all_counts['%s' %str(subhalo.GalaxyID)]         = subhalo.counts
        all_masses['%s' %str(subhalo.GalaxyID)]         = subhalo.masses
        all_misangles['%s' %str(subhalo.GalaxyID)]      = subhalo.mis_angles
        all_misanglesproj['%s' %str(subhalo.GalaxyID)]  = subhalo.mis_angles_proj
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
                    'all_spins': all_spins,
                    'all_coms': all_coms,
                    'all_counts': all_counts,
                    'all_masses': all_masses,
                    'all_misangles': all_misangles,
                    'all_misanglesproj': all_misanglesproj, 
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
        

#--------------------------------
# Plots singular graphs by reading in existing csv file
def _plot_misalignmentt(csv_sample = 'L100_24_all_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                       csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                       #--------------------------
                       # Galaxy plotting
                       print_summary = True,
                         use_angle          = 'stars_gas_sf',         # Which angles to plot
                         use_hmr            = 2.0,                    # Which HMR to use
                         use_proj_angle     = True,                   # Whether to use projected or absolute angle 10**9
                         lower_mass_limit   = 10**9,            # Whether to plot only certain masses 10**15
                         upper_mass_limit   = 10**15,         
                         ETG_or_LTG         = 'ETG',           # Whether to plot only ETG/LTG
                         group_or_field     = 'both',           # Whether to plot only field/group
                         use_satellites     = False,             # Whether to include SubGroupNum =/ 0
                       #--------------------------
                       showfig       = True,
                       savefig       = False,
                         file_format = 'pdf',
                         savefig_txt = '',
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
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift'], output_input['galaxy_mass_limit'], use_satellites))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s M*\n  Upper mass limit: %s M*\n  ETG or LTG: %s\n  Group or field: %s\n  Use satellites:  %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field, use_satellites))
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
    
    
    #-----------------------------
    # Set definitions
    group_threshold     = 10**13.8
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
    if use_satellites:
        satellite_criteria = 99999999
    if not use_satellites:
        satellite_criteria = 0
    #------------------------------
    def _plot_misalignment_distributions(debug=False):
        # We have use_angle = 'stars_gas_sf', and use_particles = ['stars', 'gas_sf'] 
        
        #=================================
        # Collect values to plot
        plot_angles     = []
        plot_angles_err = []
        
        # Collect other useful values
        catalogue = {'total': {},          # Total galaxies in mass range
                     'sample': {},         # Sample of galaxies
                     'plot': {}}           # Sub-sample that is plotted
        for key in catalogue.keys():
            catalogue[key] = {'all': 0,        # Size of group
                              'group': 0,      # number of group galaxies
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
        
        
        #--------------------------
        # Loop over all galaxies we have available, and analyse output of flags
        for GalaxyID in GalaxyID_List:
            
            #-----------------------------
            # Determine if group or field
            if all_general['%s' %GalaxyID]['halo_mass'] > group_threshold:
                catalogue['total']['group'] += 1
                
                # Determine if criteria met. If it is, it is part of the final sample
                if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                    catalogue['sample']['group'] += 1
                    catalogue['sample']['all'] += 1
                    
                    # Determine if this is a galaxy we want to plot
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                        catalogue['plot']['group'] += 1
                        catalogue['plot']['all'] += 1
                        
                        # Mask correct integer (formatting weird but works)
                        mask_rad = int(np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == use_hmr)[0])
                        
                        # Collect misangle or misangleproj
                        if use_proj_angle:
                            plot_angles.append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_rad])
                        else:
                            plot_angles.append(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_rad])
                else:
                    if debug:
                        print('not included group: ', GalaxyID)
                         
            elif all_general['%s' %GalaxyID]['halo_mass'] <= group_threshold: 
                catalogue['total']['field'] += 1
                
                # Determine if criteria met. If it is, it is part of the final sample
                if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle])and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                    catalogue['sample']['field'] += 1
                    catalogue['sample']['all'] += 1
                    
                    # Determine if this is a galaxy we want to plot
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                        catalogue['plot']['field'] += 1
                        catalogue['plot']['all'] += 1
                        
                        # Mask correct integer (formatting weird but works)
                        mask_rad = int(np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == use_hmr)[0])
                        
                        # Collect misangle or misangleproj
                        if use_proj_angle:
                            plot_angles.append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_rad])
                        else:
                            plot_angles.append(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_rad])
                else:
                    if debug:
                        print('not included group: ', GalaxyID)
                
            #-----------------------------
            # Determine if ETG or field
            if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                catalogue['total']['LTG'] += 1
                
                # Determine if criteria met. If it is, it is part of the final sample
                if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                    catalogue['sample']['LTG'] += 1
                    
                    # Determine if this is a galaxy we want to plot
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                        catalogue['plot']['LTG'] += 1
                else:
                    if debug:
                        print('not included group: ', GalaxyID)
                
            elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                catalogue['total']['ETG'] += 1
                
                # Determine if criteria met. If it is, it is part of the final sample
                if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                    catalogue['sample']['ETG'] += 1
                    
                    # Determine if this is a galaxy we want to plot
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                        catalogue['plot']['ETG'] += 1                        
                else:
                    if debug:
                        print('not included group: ', GalaxyID)
        
        
        assert catalogue['plot']['all'] == len(plot_angles), 'Number of angles collected does not equal number in catalogue... for some reason'
        if debug:
            print(catalogue['total'])
            print(catalogue['sample'])
            print(catalogue['plot'])
        
        
        
        #================================
        # Plotting
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Plotting')
            time_start = time.time()
        
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[7.0, 4.2], sharex=True, sharey=False)
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
        
        """ Some useful quantities:
        catalogue['sample']['all']  = number of galaxies that meet criteria
        catalogue['plot']['all']    = number of galaxies in this particular plot
        """
        
        #-----------
        ### Creating graphs
        # Plot histogram
        axs.hist(plot_angles, weights=np.ones(catalogue['plot']['all'])/catalogue['plot']['all'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color, alpha=0.1)
        bin_count, _, _ = axs.hist(plot_angles, weights=np.ones(catalogue['plot']['all'])/catalogue['plot']['all'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color, facecolor='none', alpha=1.0)
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(plot_angles, bins=np.arange(0, 181, 10), range=(0, 180))
        axs.errorbar(np.arange(5, 181, 10), hist_n/catalogue['plot']['all'], xerr=None, yerr=np.sqrt(hist_n)/catalogue['plot']['all'], ecolor=plot_color, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
        
        
        #-----------
        ### General formatting
        # Axis labels
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol=''))
        axs.set_xlim(0, 180)
        axs.set_xticks(np.arange(0, 181, step=30))
        if use_proj_angle:
            axs.set_xlabel('Misalignment angle, $\psi_{z}$')
        else:
            axs.set_xlabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        axs.set_ylabel('Percentage of galaxies')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
        #-----------
        # Annotations
        axs.axvline(30, ls='--', lw=1, c='k')
        
        
        #-----------
        ### Legend
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
        
        if group_or_field != 'both':
            legend_labels.append('%s-galaxies' %group_or_field)
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
        
        if print_summary:
            print('CATALOGUE:')
            print(catalogue['total'])
            print(catalogue['sample'])
            print(catalogue['plot'])
            print('\nRAW BIN VALUES:')
        aligned_tally           = 0
        aligned_err_tally       = 0
        misaligned_tally        = 0 
        misaligned_err_tally    = 0 
        counter_tally           = 0
        counter_err_tally       = 0
        for i, bin_count_i in enumerate(hist_n):
            if print_summary:
                print('  %i' %bin_count_i, end='')
            
            if i < 3:
                aligned_tally += bin_count_i
                aligned_err_tally += bin_count_i**0.5
            if i >= 3:
                misaligned_tally += bin_count_i
                misaligned_err_tally += bin_count_i**0.5
            if i >= 15:
                counter_tally += bin_count_i
                counter_err_tally += bin_count_i**0.5
            
        if print_summary:    
            print('\n')     # total population includes galaxies that failed sample, so can add to less than 100% (ei. remaining % is galaxies that make up non-sample)
            print('OF TOTAL POPULATION: \t(all galaxies in mass range)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['total']['all'], aligned_err_tally*100/catalogue['total']['all'], misaligned_tally*100/catalogue['total']['all'], misaligned_err_tally*100/catalogue['total']['all'], counter_tally*100/catalogue['total']['all'], counter_err_tally*100/catalogue['total']['all']))
            print('OF TOTAL SAMPLE: \t(no flags, hmr exists, +/- subhalo):\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['sample']['all'], aligned_err_tally*100/catalogue['sample']['all'], misaligned_tally*100/catalogue['sample']['all'], misaligned_err_tally*100/catalogue['sample']['all'], counter_tally*100/catalogue['sample']['all'], counter_err_tally*100/catalogue['sample']['all']))
            print('OF PLOT SAMPLE: \t(specific plot criteria)\n  Aligned:          %.1f ± %.1f %%\n  Misaligned:       %.1f ± %.1f %%\n  Counter-rotating: %.1f ± %.1f %%' %(aligned_tally*100/catalogue['plot']['all'], aligned_err_tally*100/catalogue['plot']['all'], misaligned_tally*100/catalogue['plot']['all'], misaligned_err_tally*100/catalogue['plot']['all'], counter_tally*100/catalogue['plot']['all'], counter_err_tally*100/catalogue['plot']['all']))       
        
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
       
        if savefig:
            plt.savefig("%s/L%s_%s_%s_misalignment_%s_%s_HMR%s_proj%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_limit'])), use_angle, str(use_hmr), use_proj_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, group_or_field, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/L%s_%s_%s_misalignment_%s_%s_HMR%s_proj%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], sat_str, np.log10(float(output_input['galaxy_mass_limit'])), use_angle, str(use_hmr), use_proj_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, group_or_field, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
            
        # Check what is plotted
        """fig, axs = plt.subplots(1, 1, figsize=[7.0, 2.2], sharex=True, sharey=False)
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
# Manually plots a graph tracking share of aligned, misaligned, and counter-rotating systems with z
def _plot_misalignment_z(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                         csv_sample_range = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],   # snapnums
                         csv_sample2 = '_all_sample_misalignment_9.0',
                         csv_output_in = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                         #--------------------------
                         # Galaxy plotting
                         print_summary = True,
                           use_angle          = 'stars_gas_sf',         # Which angles to plot
                           use_hmr            = 2.0,                    # Which HMR to use
                           use_proj_angle     = True,                   # Whether to use projected or absolute angle 10**9
                           lower_mass_limit   = 10**9,            # Whether to plot only certain masses 10**15
                           upper_mass_limit   = 10**15,         
                           ETG_or_LTG         = 'LTG',           # Whether to plot only ETG/LTG
                           group_or_field     = 'both',           # Whether to plot only field/group
                           use_satellites     = False,             # Whether to include SubGroupNum =/ 0
                         #--------------------------
                         showfig       = True,
                         savefig       = False,
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
    print('PLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s M*\n  Upper mass limit: %s M*\n  ETG or LTG: %s\n  Group or field: %s\n  Use satellites:  %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, group_or_field, use_satellites))
    print('===================\n')
    
    #--------------------------------
    # Create arrays to collect data
    plot_dict = {'SnapNum': [],
                 'Redshift': [],
                 'LookbackTime': [],
                 'aligned': [],
                 'aligned_err': [],
                 'misaligned': [],
                 'misaligned_err': [],
                 'counter': [],
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
                catalogue[key] = {'all': 0,        # Size of group
                                  'group': 0,      # number of group galaxies
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
        
        
            #--------------------------
            # Loop over all galaxies we have available, and analyse output of flags
            for GalaxyID in GalaxyID_List:
            
                #-----------------------------
                # Determine if group or field
                if all_general['%s' %GalaxyID]['halo_mass'] > group_threshold:
                    catalogue['total']['group'] += 1
                
                    # Determine if criteria met. If it is, it is part of the final sample
                    if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                        catalogue['sample']['group'] += 1
                        catalogue['sample']['all'] += 1
                    
                        # Determine if this is a galaxy we want to plot
                        if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                            catalogue['plot']['group'] += 1
                            catalogue['plot']['all'] += 1
                        
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == use_hmr)[0])
                        
                            # Collect misangle or misangleproj
                            if use_proj_angle:
                                plot_angles.append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_rad])
                            else:
                                plot_angles.append(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_rad])
                    else:
                        if debug:
                            print('not included group: ', GalaxyID)
                         
                elif all_general['%s' %GalaxyID]['halo_mass'] <= group_threshold: 
                    catalogue['total']['field'] += 1
                
                    # Determine if criteria met. If it is, it is part of the final sample
                    if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle])and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                        catalogue['sample']['field'] += 1
                        catalogue['sample']['all'] += 1
                    
                        # Determine if this is a galaxy we want to plot
                        if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                            catalogue['plot']['field'] += 1
                            catalogue['plot']['all'] += 1
                        
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == use_hmr)[0])
                        
                            # Collect misangle or misangleproj
                            if use_proj_angle:
                                plot_angles.append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_rad])
                            else:
                                plot_angles.append(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_rad])
                    else:
                        if debug:
                            print('not included group: ', GalaxyID)
                
                #-----------------------------
                # Determine if ETG or field
                if all_general['%s' %GalaxyID]['kappa_stars'] > LTG_threshold:
                    catalogue['total']['LTG'] += 1
                
                    # Determine if criteria met. If it is, it is part of the final sample
                    if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                        catalogue['sample']['LTG'] += 1
                    
                        # Determine if this is a galaxy we want to plot
                        if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                            catalogue['plot']['LTG'] += 1
                    else:
                        if debug:
                            print('not included group: ', GalaxyID)
                
                elif all_general['%s' %GalaxyID]['kappa_stars'] <= LTG_threshold:
                    catalogue['total']['ETG'] += 1
                
                    # Determine if criteria met. If it is, it is part of the final sample
                    if (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['total_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_particles'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[0]]) and (use_hmr not in all_flags['%s' %GalaxyID]['min_inclination'][use_particles[1]]) and (use_hmr not in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle]) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria) and (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                        catalogue['sample']['ETG'] += 1
                    
                        # Determine if this is a galaxy we want to plot
                        if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph):
                            catalogue['plot']['ETG'] += 1
                    else:
                        if debug:
                            print('not included group: ', GalaxyID)
        
        
            assert catalogue['plot']['all'] == len(plot_angles), 'Number of angles collected does not equal number in catalogue... for some reason'
            if debug:
                print(catalogue['total'])
                print(catalogue['sample'])
                print(catalogue['plot'])
        
        
            #=====================================
            # Extracting histogram values
            hist_n, _ = np.histogram(plot_angles, bins=np.arange(0, 181, 10), range=(0, 180))
            
            #-------------
            # Print summary
            aligned_tally           = 0
            aligned_err_tally       = 0
            misaligned_tally        = 0 
            misaligned_err_tally    = 0 
            counter_tally           = 0
            counter_err_tally       = 0
            for i, bin_count_i in enumerate(hist_n):
                if i < 3:
                    aligned_tally += bin_count_i
                    aligned_err_tally += bin_count_i**0.5
                if i >= 3:
                    misaligned_tally += bin_count_i
                    misaligned_err_tally += bin_count_i**0.5
                if i >= 15:
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
            plot_dict['aligned_err'].append(aligned_err_tally*100/catalogue['plot']['all'])
            plot_dict['misaligned'].append(misaligned_tally*100/catalogue['plot']['all'])
            plot_dict['misaligned_err'].append(misaligned_err_tally*100/catalogue['plot']['all'])
            plot_dict['counter'].append(counter_tally*100/catalogue['plot']['all'])
            plot_dict['counter_err'].append(counter_err_tally*100/catalogue['plot']['all'] )
            
         
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
        fig, axs = plt.subplots(1, 1, figsize=[6, 5.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        #-----------
        ### Creating graphs
        axs.errorbar(plot_dict['LookbackTime'], plot_dict['aligned'], yerr=plot_dict['aligned_err'], capsize=3, elinewidth=0.7, markeredgewidth=1, color='b')
        axs.errorbar(plot_dict['LookbackTime'], plot_dict['misaligned'], yerr=plot_dict['misaligned_err'], capsize=3, elinewidth=0.7, markeredgewidth=1, color='r')
        axs.errorbar(plot_dict['LookbackTime'], plot_dict['counter'], yerr=plot_dict['counter_err'], capsize=3, elinewidth=0.7, markeredgewidth=1, color='indigo')
        

        #------------------------
        ### General formatting
        # Setting regular axis
        axs.set_xlim(0, 13.5)
        axs.set_xlabel('Lookback time (Gyr)')
        axs.invert_xaxis()
        axs.set_ylim(0, 100)
        axs.set_yticks(np.arange(0, 101, 20))
        axs.set_ylabel('Percentage of galaxies')
        axs.minorticks_on()
        
        
        #-----------
        # Create redshift axis:
        redshiftticks = [0, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20]
        ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
        
        ax_top = axs.twiny()
        ax_top.set_xticks(ageticks)
        ax_top.set_xlim(0, 13.5)
        ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
        ax_top.set_xlabel('Redshift')
        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major')
        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
        ax_top.invert_xaxis()
        
        
        #-----------
        ### Legend
        
        legend_elements = []
        legend_labels = []
        legend_colors = []
        for line_name, line_color in zip(['Aligned', 'Misaligned', 'Counter-rotating'], ['b', 'r', 'indigo']):
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_labels.append(line_name)
            legend_colors.append(line_color)
        
        # Add mass range
        if (lower_mass_limit != 10**9) and (upper_mass_limit != 10**15):
            legend_labels.append('$10 ^{%.1f} - 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit), np.log10(upper_mass_limit)))    
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        elif (lower_mass_limit != 10**9):
            legend_labels.append('$> 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(lower_mass_limit)))    
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        elif (upper_mass_limit != 10**15):
            legend_labels.append('$< 10 ^{%.1f}$ M$_{\odot}$' %(np.log10(upper_mass_limit)))    
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        
        # Add LTG/ETG if specified
        if ETG_or_LTG != 'both':
            legend_labels.append('%s' %ETG_or_LTG)
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        if group_or_field != 'both':
            legend_labels.append('%s-galaxies' %group_or_field)
            legend_elements.append(Line2D([-10], [-10], marker=' ', color='w'))
            legend_colors.append('grey')
        
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        
        
        #-----------
        # other
        plt.tight_layout()
        
        
        if print_summary:    
            print('===================')
            print('SUMMARY DATA:')
            print('  | Redshift | time | Aligned     | Misaligned | Counter-rot |')
            for snap_p, red_p, time_p, ali_p, alierr_p, mis_p, miserr_p, coun_p, counerr_p in zip(plot_dict['SnapNum'], plot_dict['Redshift'], plot_dict['LookbackTime'], plot_dict['aligned'], plot_dict['aligned_err'], plot_dict['misaligned'], plot_dict['misaligned_err'], plot_dict['counter'], plot_dict['counter_err']):
                print('  |   %.2f   | %.2f | %.1f ± %.1f%% | %.1f ± %.1f%% | %.1f ± %.1f%%  |' %(red_p, time_p, ali_p, alierr_p, mis_p, miserr_p, coun_p, counerr_p))
            print('===================')
            
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
            plt.savefig("%s/L%s_ALL_%s_misalignment_summary_%s_%s_HMR%s_proj%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], sat_str, np.log10(float(output_input['galaxy_mass_limit'])), use_angle, str(use_hmr), use_proj_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, group_or_field, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/L%s_ALL_%s_misalignment_summary_%s_%s_HMR%s_proj%s_m%sm%s_morph%s_env%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], sat_str, np.log10(float(output_input['galaxy_mass_limit'])), use_angle, str(use_hmr), use_proj_angle, np.log10(lower_mass_limit), np.log10(upper_mass_limit), ETG_or_LTG, group_or_field, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
            
    
    #---------------------
    _plot_misalignment_z()
    #---------------------
    
    
#===========================    
#_sample_misalignment()
#_sample_misalignment()

#_analysis_misalignment_minor()
#_analysis_misalignment_distribution()

#_plot_misalignment()
#_plot_misalignment_z()
#===========================
    

                              