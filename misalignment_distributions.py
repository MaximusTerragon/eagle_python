import h5py
import numpy as np
import math
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from subhalo_main_new import Initial_Sample, Subhalo_Extract, Subhalo_Analysis
import eagleSqlTools as sql
from graphformat import set_rc_params



# Directories
EAGLE_dir       = '/Users/c22048063/Documents/EAGLE'
dataDir_main    = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
sample_dir      = EAGLE_dir + '/samples'
output_dir      = EAGLE_dir + '/outputs'
fig_dir         = EAGLE_dir + '/plots'


# Directories of data hdf5 file(s)
dataDir_dict = {}
dataDir_dict['10'] = dataDir_main + 'snapshot_010_z003p984/snap_010_z003p984.0.hdf5'
dataDir_dict['11'] = dataDir_main + 'snapshot_011_z003p528/snap_011_z003p528.0.hdf5'
dataDir_dict['12'] = dataDir_main + 'snapshot_012_z003p017/snap_012_z003p017.0.hdf5'
dataDir_dict['13'] = dataDir_main + 'snapshot_013_z002p478/snap_013_z002p478.0.hdf5'
dataDir_dict['14'] = dataDir_main + 'snapshot_014_z002p237/snap_014_z002p237.0.hdf5'
dataDir_dict['15'] = dataDir_main + 'snapshot_015_z002p012/snap_015_z002p012.0.hdf5'
dataDir_dict['16'] = dataDir_main + 'snapshot_016_z001p737/snap_016_z001p737.0.hdf5'
dataDir_dict['17'] = dataDir_main + 'snapshot_017_z001p487/snap_017_z001p487.0.hdf5'
dataDir_dict['18'] = dataDir_main + 'snapshot_018_z001p259/snap_018_z001p259.0.hdf5'
dataDir_dict['19'] = dataDir_main + 'snapshot_019_z001p004/snap_019_z001p004.0.hdf5'
dataDir_dict['20'] = dataDir_main + 'snapshot_020_z000p865/snap_020_z000p865.0.hdf5'
dataDir_dict['21'] = dataDir_main + 'snapshot_021_z000p736/snap_021_z000p736.0.hdf5'
dataDir_dict['22'] = dataDir_main + 'snapshot_022_z000p615/snap_022_z000p615.0.hdf5'
dataDir_dict['23'] = dataDir_main + 'snapshot_023_z000p503/snap_023_z000p503.0.hdf5'
dataDir_dict['24'] = dataDir_main + 'snapshot_024_z000p366/snap_024_z000p366.0.hdf5'
dataDir_dict['25'] = dataDir_main + 'snapshot_025_z000p271/snap_025_z000p271.0.hdf5'
dataDir_dict['26'] = dataDir_main + 'snapshot_026_z000p183/snap_026_z000p183.0.hdf5'
dataDir_dict['27'] = dataDir_main + 'snapshot_027_z000p101/snap_027_z000p101.0.hdf5'
dataDir_dict['28'] = dataDir_main + 'snapshot_028_z000p000/snap_028_z000p000.0.hdf5'
#dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
#dataDir = '/home/universe/spxtd1-shared/RefL0100N1504/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)




# Creates a sample of galaxies given the inputs.
# Returns GroupNum, SubGroupNum, SnapNum, and GalaxyID for each galaxy
def _misalignment_sample(mySims                = [('RefL0012N0188', 12)],
                            galaxy_mass_limit  = 10**9,            # Lower mass limit within 30pkpc
                            snapNum            = 28,               # Target snapshot
                            use_satellites     = 'yes',             # Whether to include SubGroupNum =/ 0
                            print_sample       = False,             # Print list of IDs
                         #--------------------------   
                         csv_file = True,                       # Will write sample to csv file in sapmle_dir
                            csv_name = 'sample_misalignment',
                         #--------------------------     
                         print_progress = False,
                         debug = False):
                        
                        
    #=====================================  
    # Create sample
    if print_progress:
        print('Creating sample')
        time_start = time.time()
        
    
    # Extracting all GroupNum, SubGroupNum, GalaxyID, and SnapNum
    sample = Initial_Sample(mySims, snapNum, galaxy_mass_limit, use_satellites)
    if debug:
        print(sample.GroupNum)
        print(sample.SubGroupNum)
        print(sample.GalaxyID)
        print(sample.SnapNum)
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
    if print_sample:
        print("  ", sample.GroupNum)
    
    print('\n===================\nSAMPLE CREATED:\n  Mass limit: %.2E\n  SnapNum: %s\n  Satellites: %s' %(galaxy_mass_limit, snapNum, use_satellites))
    print("  SAMPLE LENGTH: ", len(sample.GalaxyID))
    
    
    
    #===================================== 
    # Creating dictionaries to gather input data
    sample_input = {'galaxy_mass_limit': galaxy_mass_limit,
                    'snapNum': snapNum,
                    'use_satellites': use_satellites,
                    'mySims': mySims}
    if debug:
        print(sample_input.items())

    
    #=====================================
    if csv_file: 
        # Converting numpy arrays to lists. When reading, may need to simply convert list back to np.array() (easy)
        class NumpyEncoder(json.JSONEncoder):
            """ Special json encoder for numpy types """
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
                    'SnapNum': sample.SnapNum,
                    'sample_input':  sample_input}
                    
        csv_dict.update({'function_input': str(inspect.signature(_misalignment_sample))})
   
        # Writing one massive JSON file
        if use_satellites == 'no':
            json.dump(csv_dict, open('%s/%s_cent_%s_%s.csv' %(sample_dir, str(snapNum), csv_name, np.log10(galaxy_mass_limit)), 'w'), cls=NumpyEncoder)
            print('\nSAVED: %s/%s_cent_%s_%s.csv' %(sample_dir, str(snapNum), csv_name, np.log10(galaxy_mass_limit)))
        else:
            json.dump(csv_dict, open('%s/%s_all_%s_%s.csv' %(sample_dir, str(snapNum), csv_name, np.log10(galaxy_mass_limit)), 'w'), cls=NumpyEncoder)
            print('\nSAVED: %s/%s_all_%s_%s.csv' %(sample_dir, str(snapNum), csv_name, np.log10(galaxy_mass_limit)))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
   
        # Reading JSON file
        """dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
        # example nested dictionaries
        GroupNum_List       = np.array(dict_new['GroupNum'])
        SubGroupNum_List    = np.array(dict_new['SubGroupNum'])
        GalaxyID_List       = np.array(dict_new['GalaxyID'])
        SnapNum_List        = np.array(dict_new['SnapNum'])
        sample_input        = dict_new['sample_input']
        
        print('\n===================\nSAMPLE LOADED:\n  Mass limit: %.2E\n  SnapNum: %s\n  Satellites: %s' %(sample_input['galaxy_mass_limit'], sample_input['snapNum'], sample_input['use_satellites']))
        print("  SAMPLE LENGTH: ", len(GroupNum_List))
        """
    
    
# Reads in a sample file, and does all relevant calculations, and exports as csv file
def _misalignment_distribution(csv_sample = '28_cent_sample_misalignment_9.0',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                                
                                #--------------------------
                                # Galaxy extraction properties
                                viewing_axis        = 'z',                  # Which axis to view galaxy from.  DEFAULT 'z'
                                aperture_rad        = 30,                   # trim all data to this maximum value before calculations [pkpc]
                                kappa_rad           = 30,                   # calculate kappa for this radius [pkpc]
                                trim_hmr = np.array([100]),              # keep as 100... will be capped by aperture anyway. Doesn't matter
                                align_rad = False,                       # keep on False
                                orientate_to_axis='z',                      # Keep as z
                                viewing_angle = 0,                          # Keep as 0
                                #-----------------------------------------------------
                                # Misalignments we want extracted and at which radii
                                angle_selection     = np.array(['stars_gas',            # stars_gas     stars_gas_sf    stars_gas_nsf
                                                                'stars_gas_sf',         # gas_dm        gas_sf_dm       gas_nsf_dm
                                                                'stars_gas_nsf',        # gas_sf_gas_nsf
                                                                'gas_sf_gas_nsf',
                                                                'stars_dm']),           
                                spin_hmr            = np.array([1.0, 2.0]),          # multiples of hmr for which to find spin
                                find_uncertainties  = True,                    # whether to find 2D and 3D uncertainties
                                rad_projected       = True,                     # whether to use rad in projection or 3D
                                #--------------------------
                                # Selection criteria
                                com_min_distance        = 2.0,                    # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                                min_particles           = 20,                     # Minimum particles to find spin.  DEFAULT 20
                                min_inclination         = 0,                      # Minimum inclination toward viewing axis [deg] DEFAULT 0
                                #--------------------------   
                                csv_file = True,                       # Will write sample to csv file in sapmle_dir
                                    csv_name = 'data_misalignment',
                                #--------------------------
                                print_progress = False,
                                debug = False):
    
    
    #---------------------------------------------
    # Load sample csv
    if print_progress:
        print('Loading initial sample')
        time_start = time.time()
        
        
    dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    
    # Extract GroupNum etc.
    GroupNum_List       = np.array(dict_new['GroupNum'])
    SubGroupNum_List    = np.array(dict_new['SubGroupNum'])
    GalaxyID_List       = np.array(dict_new['GalaxyID'])
    SnapNum_List        = np.array(dict_new['SnapNum'])
    sample_input        = dict_new['sample_input']
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    if debug:
        print(sample_input)
        print(GroupNum_List)
        print(SubGroupNum_List)
        print(GalaxyID_List)
        print(SnapNum_List)
       
       
    print('\n===================\nSAMPLE LOADED:\n  %s\n  Mass limit: %.2E\n  SnapNum: %s\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['galaxy_mass_limit'], sample_input['snapNum'], sample_input['use_satellites']))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nEXTRACT:\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s' %(str(angle_selection), str(spin_hmr), str(find_uncertainties), str(rad_projected)))
    print('===================')
    
    #---------------------------------------------
    # Empty dictionaries to collect relevant data
    all_flags         = {}          # has reason why galaxy failed sample
    all_general       = {}          # has total masses, kappa, halfmassrad, etc.
    all_coms          = {}          # has all C.o.Ms
    all_misangles     = {}          # has all 3D angles
    all_particles     = {}          # has all the particle count and mass within rad
    all_misanglesproj = {}          # has all 2D projected angles from 3d when given a viewing axis and viewing_angle = 0
    
    
    # Run analysis for each individual galaxy in loaded sample
    for GroupNum, SubGroupNum, GalaxyID, SnapNum in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List), total=len(GroupNum_List)):
        
        #-----------------------------
        if print_progress:
            print('Extracting particle data Subhalo_Extract()')
            time_start = time.time()
            
        # Initial extraction of galaxy particle data
        galaxy = Subhalo_Extract(sample_input['mySims'], dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, aperture_rad, viewing_axis)
        # Gives: galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, galaxy.halo_mass
        
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
            spin_rad = spin_hmr * galaxy.halfmass_rad_proj
        elif rad_projected == False:
            spin_rad = spin_hmr * galaxy.halfmass_rad
        

        # If we want the original values, enter 0 for viewing angle
        subhalo = Subhalo_Analysis(sample_input['mySims'], GroupNum, SubGroupNum, GalaxyID, SnapNum, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, 
                                            viewing_axis,
                                            aperture_rad,
                                            kappa_rad, 
                                            trim_hmr, 
                                            align_rad,              #align_rad = False
                                            orientate_to_axis,
                                            viewing_angle,
                                            
                                            angle_selection,        
                                            spin_rad,
                                            spin_hmr,
                                            find_uncertainties,
                                            
                                            com_min_distance,
                                            min_particles,                                            
                                            min_inclination)
                                       
        
        print(GroupNum)
        #print(subhalo.flags['total_particles'])
        #print(subhalo.flags['min_particles'])
        #print(subhalo.flags['min_inclination'])
        #print(subhalo.flags['com_min_distance'])
        print(subhalo.mis_angles_proj[viewing_axis]['stars_gas_sf_angle'])
        print(subhalo.mis_angles_proj[viewing_axis]['stars_gas_sf_angle_err'])
        print(' ')
        print(subhalo.counts.items())
        
        
        #clip spin_rad to aperture
        
        """ 
        .flags:     dictionary
            Has list of arrays that will be != if flagged. Contains hmr at failure, or 30pkpc
                ['total_particles']
                    ['stars']       - [hmr]
                    ['gas']         - [hmr]
                    ['gas_sf']      - [hmr]
                    ['gas_nsf']     - [hmr]
                    ['dm']          - [hmr]
                    ['bh']          - [hmr]
                ['min_particles']
                    ['stars']       - [hmr]
                    ['gas']         - [hmr]
                    ['gas_sf']      - [hmr]
                    ['gas_nsf']     - [hmr]
                    ['dm']          - [hmr]
                ['min_inclination']
                    ['stars']       - [hmr]
                    ['gas']         - [hmr]
                    ['gas_sf']      - [hmr]
                    ['gas_nsf']     - [hmr]
                    ['dm']          - [hmr]
                ['com_min_distance']
                    ['stars_gas']   - [hmr]
                    ['stars_gas_sf']- [hmr]
                    ...
                
        .spins:   dictionary
            Has aligned/rotated spin vectors within spin_rad_in's:
                ['rad']     - [pkpc]
                ['hmr']     - multiples of halfmass_rad
                ['stars']   - [unit vector]
                ['gas']     - [unit vector]
                ['gas_sf']  - [unit vector]
                ['gas_nsf'] - [unit vector]
                ['dm']      - unit vector at 30pkpc
        .counts:   dictionary
            Has aligned/rotated particle count and mass within spin_rad_in's:
                ['rad']          - [pkpc]
                ['hmr']     - multiples of halfmass_rad
                ['stars']        - [count]
                ['gas']          - [count]
                ['gas_sf']       - [count]
                ['gas_nsf']      - [count]
                ['dm']           - count at 30pkpc
        .masses:   dictionary
            Has aligned/rotated particle count and mass within spin_rad_in's:
                ['rad']          - [pkpc]
                ['hmr']     - multiples of halfmass_rad
                ['stars']        - [Msun]
                ['gas']          - [Msun]
                ['gas_sf']       - [Msun]
                ['gas_nsf']      - [Msun]
                ['dm']           - Msun at 30pkpc
        .coms:     dictionary
            Has all centres of mass and distances within a spin_rad_in:
                ['rad']          - [pkpc]
                ['hmr']     - multiples of halfmass_rad
                ['stars']          - [x, y, z] [pkpc]
                ['gas']            - [x, y, z] [pkpc]
                ['gas_sf']         - [x, y, z] [pkpc]
                ['gas_nsf']        - [x, y, z] [pkpc]
                ['dm']             - x, y, z [pkpc]  at 30pkpc
        """
    
    
    
    
    
    
    
def _misalignment_plot():
    print('PLOT')    
    
    
    

    
    
    
#===========================    
#_misalignment_sample()
_misalignment_distribution()
#===========================
    

                              