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
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID, ConvertID_noMK, ConvertID_snip, MergerTree
import eagleSqlTools as sql
from graphformat import set_rc_params
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================

""" ID LISTS
Counter-rotating (0 snip)
#ID_list = [16300531, 16049479, 17341514, 15165512, 14531471, 18108607, 13809906] #, 9678375, 16647245, 9628491, 8257780, 8806615, 13851368, 9822659, 9998415, 10883094, 8625363, 10525701, 1784463, 10784741]

1 snip:
#ID_list = [14216102, 8707373, 18447769, 18363467, 17718284, 9110372, 9542932, 9008303, 9216030, 12187581, 9746293, 8707373, 8494196, 10145405]

2 snip:
#ID_list = [13866051, 9777730, 10009377, 8345213, 10443502, 10670173]

3 snip:
#ID_list = [17374402, 8077031, 17480553, 15851557]

4+ snip:
#ID_list = [10438463, 8763330, 3327115]
"""
ID_list = [15851557, 3327115, 10438463, 10670173, 13866051, 17480553, 8077031, 8494196, 8763330, 9777730, 10009377, 10145405, 14216102, 16049479, 17374402, 17718284, 18447769]
#--------------------
# Run analysis on individual galaxies and output individual CSV files
# SAVED: /outputs/L%s_evolution_ID_
def _analysis_evolution(csv_sample = False,              # Whether to read in existing list of galaxies  
                               #--------------------------
                               mySims = [('RefL0012N0188', 12)],
                               GalaxyID_List_target = [30494],               # Will create a csv file for each galaxy
                               snapNumMin           = 19,                    # Highest snapNum to go to
                               snapNumMax           = 28,                    # Highest snapNum to go to
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
                               com_min_distance    = 2.0,               # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                               min_particles       = 20,                # Minimum particles to find spin.  DEFAULT 20
                               min_inclination     = 0,                 # Minimum inclination toward viewing axis [deg] DEFAULT 0
                               #--------------------------   
                               csv_file       = True,                   # Will write sample to csv file in sapmle_dir
                                 csv_name     = '',                     # extra stuff at end
                               #--------------------------
                               print_progress = False,
                               print_galaxy   = True,
                               debug = False):
                             
    
    
    #============================================
    # Extracting GroupNum_List, SubGroupNum_List and SnapNum_List
    if print_progress:
        print('Extracting GroupNum, SubGroupNum, SnapNum lists')
        time_start = time.time()
        
    #-----------------------------------------
    # adjust mySims for serpens
    if (answer == '2') or (answer == '3'):
        mySims = [('RefL0100N1504', 100)]
        
    #---------------------------------------------
    # Use IDs and such from sample
    if csv_sample:
        # Load sample csv
        if print_progress:
            print('Loading initial sample')
            time_start = time.time()
        
    
        # Loading sample
        dict_new = json.load(open('%s/%s.csv' %(sample_dir, csv_sample), 'r'))
    
    
        # Extract GroupNum etc.
        GroupNum_List_target       = np.array(dict_new['GroupNum'])
        SubGroupNum_List_target    = np.array(dict_new['SubGroupNum'])
        GalaxyID_List_target       = np.array(dict_new['GalaxyID'])
        SnapNum_List_target        = np.array(dict_new['SnapNum'])
        Redshift_List_target       = np.array(dict_new['Redshift'])
        sample_input               = dict_new['sample_input']
        mySims                     = sample_input['mySims']
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print(sample_input)
            print(GroupNum_List_target)
            print(SubGroupNum_List_target)
            print(GalaxyID_List_target)
            print(SnapNum_List_target)
       
        print('\n===================')
        print('SAMPLE LOADED:\n  %s\n  GalaxyIDs: %s' %(mySims[0][0], GalaxyID_List_target))
        print('  SAMPLE LENGTH: ', len(GroupNum_List_target))
        print('===================')
                               
    
    #---------------------------------------------
    # If no csv_sample given, use GalaxyID_List
    else:
        # If using snipshots...
        if answer == '3':
            # Extract GroupNum, SubGroupNum, and Snap for each ID
            GroupNum_List_target    = []
            SubGroupNum_List_target = []
            SnapNum_List_target     = []
            Redshift_List_target    = []
        
            for galID in GalaxyID_List_target:
                gn, sgn, snap, z, _, _, _ = ConvertID_snip(tree_dir, galID, mySims)
    
                # Append to arrays
                GroupNum_List_target.append(gn)
                SubGroupNum_List_target.append(sgn)
                SnapNum_List_target.append(snap)
                Redshift_List_target.append(z)
            
            if debug:
                print(GroupNum_List_target)
                print(SubGroupNum_List_target)
                print(GalaxyID_List_target)
                print(SnapNum_List_target)
        
        else:
            # Extract GroupNum, SubGroupNum, and Snap for each ID
            GroupNum_List_target    = []
            SubGroupNum_List_target = []
            SnapNum_List_target     = []
            Redshift_List_target    = []
        
            for galID in GalaxyID_List_target:
                gn, sgn, snap, z, _, _, _ = ConvertID(galID, mySims)
    
                # Append to arrays
                GroupNum_List_target.append(gn)
                SubGroupNum_List_target.append(sgn)
                SnapNum_List_target.append(snap)
                Redshift_List_target.append(z)
            
            if debug:
                print(GroupNum_List_target)
                print(SubGroupNum_List_target)
                print(GalaxyID_List_target)
                print(SnapNum_List_target)
            
        print('\n===================')
        print('SAMPLE INPUT:\n  %s\n  GalaxyIDs: %s' %(mySims[0][0], GalaxyID_List_target))
        print('  SAMPLE LENGTH: ', len(GroupNum_List_target))
        print('===================')
            
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
    #---------------------------------------------
    #We are left with:
    """ 
    GalaxyID_List_target    
    GroupNum_List_target    
    SubGroupNum_List_target 
    SnapNum_List_target    
    Redshift_List_target    
    """
    
    # For the time being ensure target snap = 28
    if (answer == '1') or (answer == '2'):
        assert SnapNum_List_target[-1] == 28, 'Merger tree not configured to work with non z=0 targets just yet'
        assert snapNumMin >= 19, 'Limit of snapshots reached'
        assert snapNumMax <= 28, 'Limit of snapshots reached'
    if answer == '3':
        assert snapNumMin >= 147, 'Limit of snapshots reached'
        assert snapNumMax <= 200, 'Limit of snapshots reached'
    
    
    output_input = {'angle_selection': angle_selection,
                    'spin_hmr': spin_hmr,
                    'find_uncertainties': find_uncertainties,
                    'rad_projected': rad_projected,
                    'viewing_axis': viewing_axis,
                    'aperture_rad': aperture_rad,
                    'kappa_rad': kappa_rad,
                    'com_min_distance': com_min_distance,
                    'min_particles': min_particles,
                    'min_inclination': min_inclination,
                    'snapNumMin': snapNumMin,
                    'snapNumMax': snapNumMax,
                    'mySims': mySims}
                    
    #=================================================================== 
    # For each target galaxy we are interested in...
    for target_GroupNum, target_SubGroupNum, target_GalaxyID, target_SnapNum, target_Redshift in zip(GroupNum_List_target, SubGroupNum_List_target, GalaxyID_List_target, SnapNum_List_target, Redshift_List_target):
    
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Extracting merger tree data MergerTree()')
            time_start = time.time()
          
           
        # Extract merger tree data
        tree = MergerTree(tree_dir, mySims, target_GalaxyID, snapNumMin, snapNumMax)
        
        
        #------------------------------------------------
        # Extract halomass, centres, and morphokinem for tree.main_branch
        GalaxyID_List    = tree.main_branch['GalaxyID']
        GroupNum_List    = tree.main_branch['GroupNumber']
        SubGroupNum_List = tree.main_branch['SubGroupNumber']
        SnapNum_List     = tree.main_branch['snapnum']
        Redshift_List    = tree.main_branch['redshift']
        HaloMass_List    = []
        Centre_List      = []
        MorphoKinem_List = []
        
        
        # Finding halomasses, and nan MK's for snap < 15
        if (answer == '1') or (answer == '2'):
            for galID, snapID in zip(GalaxyID_List, SnapNum_List):
                if snapID >= 15:
                    _, _, _, _, halomass_i, centre_i, morphkinem_i = ConvertID(galID, mySims)
                
                    # Append to arrays
                    HaloMass_List.append(halomass_i) 
                    Centre_List.append(centre_i)
                    MorphoKinem_List.append(morphkinem_i)
                else:
                    _, _, _, _, halomass_i, centre_i, morphkinem_i = ConvertID_noMK(galID, mySims)
                
                    # Append to arrays
                    HaloMass_List.append(halomass_i) 
                    Centre_List.append(centre_i)
                    MorphoKinem_List.append(morphkinem_i)
        elif answer == '3':
            for galID, snapID in zip(GalaxyID_List, SnapNum_List):
                _, _, _, _, halomass_i, centre_i, morphkinem_i = ConvertID_snip(tree_dir, galID, mySims)
            
                # Append to arrays
                HaloMass_List.append(halomass_i) 
                Centre_List.append(centre_i)
                MorphoKinem_List.append(morphkinem_i)
            
            
        if debug:
            print(GroupNum_List)
            print(SubGroupNum_List)
            print(GalaxyID_List)
            print(SnapNum_List)
            
        if print_galaxy:
            for snap_p, id_p, gn_p, sgn_p, morph_p in zip(SnapNum_List, GalaxyID_List, GroupNum_List, SubGroupNum_List, MorphoKinem_List):
                print('|%s| |ID:   %s\t|GN:  %s  |SGN:  %s  |KAPPA:  %.2f' %(snap_p, id_p, gn_p, sgn_p, morph_p[2])) 
                
            
        #----------------------------------
        # Update totals with current target_GalaxyID. This will be the single identifier for each galaxy
        
        # Empty dictionaries to collect relevant data
        total_flags         = {}        # has reason why galaxy failed sample
        total_general       = {}        # has total masses, kappa, halfmassrad, etc.
        total_coms          = {}        # has all C.o.Ms
        total_spins         = {}        # has all spins
        total_counts        = {}        # has all the particle count within rad
        total_masses        = {}        # has all the particle mass within rad
        total_sfr           = {}        # has all bulk SFR
        total_Z             = {}        # has all bulk metallicity
        total_misangles     = {}        # has all 3D angles
        total_misanglesproj = {}        # has all 2D projected angles from 3d when given a viewing axis and viewing_angle = 0
        total_gasdata       = {}        # has all gas particle IDs
        total_massflow      = {}        # has all mass flow from previous snap to current (=math.nan for first)
    
        # Merger tree stuff
        total_allbranches = {}
        total_mainbranch  = {}
        total_mergers     = {}
        
        # Empty dictionaries are created
        total_flags.update({'%s' %target_GalaxyID: {}})
        total_general.update({'%s' %target_GalaxyID: {}})
        total_coms.update({'%s' %target_GalaxyID: {}})
        total_spins.update({'%s' %target_GalaxyID: {}})
        total_counts.update({'%s' %target_GalaxyID: {}})
        total_masses.update({'%s' %target_GalaxyID: {}})
        total_sfr.update({'%s' %target_GalaxyID: {}})
        total_Z.update({'%s' %target_GalaxyID: {}})
        total_misangles.update({'%s' %target_GalaxyID: {}})
        total_misanglesproj.update({'%s' %target_GalaxyID: {}})
        total_gasdata.update({'%s' %target_GalaxyID: {}})
        total_massflow.update({'%s' %target_GalaxyID: {}})
    
        total_allbranches.update({'%s' %target_GalaxyID: tree.all_branches})
        total_mainbranch.update({'%s' %target_GalaxyID: tree.main_branch})
        total_mergers.update({'%s' %target_GalaxyID: tree.mergers})
        #----------------------------------
    
        
        #=================================================================== 
        # Run analysis for each individual galaxy in loaded sample
        for GroupNum, SubGroupNum, GalaxyID, SnapNum, Redshift, HaloMass, Centre_i, MorphoKinem, snap_iteration in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List, Redshift_List, HaloMass_List, Centre_List, MorphoKinem_List, np.arange(len(GroupNum_List))), total=len(GroupNum_List)):
        
            if print_progress:
                 print('Extracting particle data Subhalo_Extract()')
                 time_start = time.time()
    
            # Initial extraction of galaxy particle data
            galaxy = Subhalo_Extract(mySims, dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, Centre_i, HaloMass, aperture_rad, viewing_axis, MorphoKinem)
            GroupNum = galaxy.gn
            # Gives: galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh
            # Gives: subhalo.general: GroupNum, SubGroupNum, GalaxyID, stelmass, gasmass, gasmass_sf, gasmass_nsf
        
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
        
            
            # Skip inflow/outflow for first snap
            if snap_iteration == 0:
                gas_data_old = False
            # If not first snap, use predecessor gasdata
            else:
                gas_data_old = total_gasdata['%s' %target_GalaxyID]['%s' %str(int(GalaxyID)+1)]
            
            
            # If we want the original values, enter 0 for viewing angle
            subhalo = Subhalo_Analysis(mySims, GroupNum, SubGroupNum, GalaxyID, SnapNum, galaxy.MorphoKinem, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.data_nil, 
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
                                                min_inclination,
                                                
                                                gas_data_old)
    
            """ FLAGS 
            ------------
            #print(subhalo.flags['total_particles'])            # will flag if there are missing particles within aperture_rad
            #print(subhalo.flags['min_particles'])              # will flag if min. particles not met within spin_rad (will find spin if particles exist, but no uncertainties)
            #print(subhalo.flags['min_inclination'])            # will flag if inclination angle not met within spin_rad... all spins and uncertainties still calculated
            #print(subhalo.flags['com_min_distance'])           # will flag if com distance not met within spin_rad... all spins and uncertainties still calculated
            ------------
            """
            
            """ INFLOW OUTFLOW 
            print(subhalo.mass_flow['2.0_hmr']['gas']['inflow'])
            print(subhalo.mass_flow['2.0_hmr']['gas']['outflow'])
            print(subhalo.mass_flow['2.0_hmr']['gas']['massloss'])
            """
            #print(subhalo.bh_mdot)
            #print(subhalo.bh_edd)
            #print(subhalo.bh_id)
            
            if print_galaxy:
                print('|%s| |ID:   %s\t|M*:  %.2e  |HMR:  %.2f  |KAPPA:  %.2f' %(SnapNum, str(subhalo.GalaxyID), subhalo.stelmass, subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'])) 
            
            #--------------------------------
            # Collecting all relevant particle info for galaxy

            total_flags['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]         = subhalo.flags
            total_general['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]       = subhalo.general
            total_coms['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]          = subhalo.coms
            total_spins['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]         = subhalo.spins
            total_counts['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]        = subhalo.counts
            total_masses['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]        = subhalo.masses
            total_sfr['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]           = subhalo.sfr
            total_Z['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]             = subhalo.Z
            total_misangles['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]     = subhalo.mis_angles
            total_misanglesproj['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)] = subhalo.mis_angles_proj
            total_gasdata['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]       = subhalo.gas_data
            total_massflow['%s' %target_GalaxyID]['%s' %str(subhalo.GalaxyID)]      = subhalo.mass_flow
            #---------------------------------
                  
                  
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
            csv_dict = {'total_flags': total_flags, 
                        'total_general': total_general, 
                        'total_coms': total_coms,
                        'total_spins': total_spins,
                        'total_counts': total_counts,
                        'total_masses': total_masses,
                        'total_sfr': total_sfr,
                        'total_Z': total_Z,
                        'total_misangles': total_misangles, 
                        'total_misanglesproj': total_misanglesproj, 
                        'total_gasdata': total_gasdata,
                        'total_massflow': total_massflow,
                        'total_allbranches': total_allbranches, 
                        'total_mainbranch': total_mainbranch, 
                        'total_mergers': total_mergers,
                        'output_input': output_input}
            #csv_dict.update({'function_input': str(inspect.signature(_radial_evolution_analysis))})
    
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
            json.dump(csv_dict, open('%s/L%s_evolution_ID%s_%s_%s_%s_%s.csv' %(output_dir, mySims[0][1], GalaxyID, rad_str, uncertainty_str, angle_str, csv_name), 'w'), cls=NumpyEncoder)
            print('\n  SAVED: %s/L%s_evolution_ID%s_%s_%s_%s_%s.csv' %(output_dir, mySims[0][1], GalaxyID, rad_str, uncertainty_str, angle_str, csv_name))
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
    
            # Reading JSON file
            """ 
            # Loading output
            dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    
            total_flags           = dict_output['total_flags']
            total_general         = dict_output['total_general']
            total_spins           = dict_output['total_spins']
            total_counts          = dict_output['total_counts']
            total_masses          = dict_output['total_masses']
            total_coms            = dict_output['total_coms']
            total_misangles       = dict_output['total_misangles']
            total_misanglesproj   = dict_output['total_misanglesproj']
    
            total_allbranches     = dict_output['total_allbranches']
            total_mainbranch      = dict_output['total_mainbranch']
            total_mergers         = dict_output['total_mergers']
    
            # Loading sample criteria
            output_input        = dict_output['output_input']
    
            #---------------------------------
            # Extract GroupNum, SubGroupNum, and Snap for each ID
            GalaxyID_List_target = list(total_general.keys())

            print('\n===================')
            print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Mass limit: %.2E M*\n  Satellites: %s' %(sample_input['mySims'][0][0], sample_input['snapNum'], sample_input['Redshift'], sample_input['galaxy_mass_limit'], sample_input['use_satellites']))
            print('  SAMPLE LENGTH: ', len(GroupNum_List))
            print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
            print('\nPLOT:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Lower mass limit: %s\n  Upper mass limit: %s\n  ETG or LTG: %s\n  Group or field: %s' %(use_angle, use_hmr, use_proj_angle, lower_mass_limit, upper_mass_limit, ETG_or_LTG, cluster_or_field))
            print('===================')
            """


#--------------------
# Will plot evolution of single galaxy but with improved formatting for poster/presentation and with outflows/inflows
# SAVED: /plots/individual_evolution/
def _plot_evolution(csv_output = 'L100_evolution_ID443970_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',   # CSV sample file to load 
                               csv_merger_tree = 'L12_merger_tree_',
                               #--------------------------
                               # Galaxy plotting
                               print_summary = True,
                                 use_angles         = ['stars_gas_sf'],                 # Which angles to plot
                                 use_hmr            = [1, 2],         # Which misangle HMR to plot
                                 use_hmr_frac       = [2],                # Which mass and fraction HMR to plot  
                                 use_proj_angle     = True,                   # Whether to use projected or absolute angle, 'both'
                                 use_uncertainties  = True,                   # Whether to plot uncertainties or not
                                 min_merger_ratio   = 0.05,
                              #-------------------------
                              # Plot settings
                              highlight_criteria = True,       # whether to indicate when criteria not met (but still plot)
                              rad_type_plot      = 'hmr',      # 'rad' whether to use absolute distance or hmr 
                              #--------------------------
                              showfig        = False,
                              savefig        = True,
                                file_format  = 'pdf',
                                savefig_txt  = 'NEW',
                              #--------------------------
                              print_progress = False,
                              debug = False):
    
    
    
    #================================================  
    # Load sample csv
    if print_progress:
        print('Loading output')
        time_start = time.time()
    
    #--------------------------------
    # Loading output
    dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    
    # Loading merger tree
    merger_tree_load  = json.load(open('%s/%s.csv' %(output_dir, csv_merger_tree), 'r'))
    merger_tree       = merger_tree_load['tree_dict']     
    merger_tree_input = merger_tree_load['output_input']
    
    
    #--------------------------------
    total_flags           = dict_output['total_flags']
    total_general         = dict_output['total_general']
    total_spins           = dict_output['total_spins']
    total_counts          = dict_output['total_counts']
    total_masses          = dict_output['total_masses']
    total_sfr             = dict_output['total_sfr']
    total_Z               = dict_output['total_Z']
    total_coms            = dict_output['total_coms']
    total_misangles       = dict_output['total_misangles']
    total_misanglesproj   = dict_output['total_misanglesproj']
    total_massflow        = dict_output['total_massflow']
    
    total_allbranches     = dict_output['total_allbranches']
    total_mainbranch      = dict_output['total_mainbranch']
    total_mergers         = dict_output['total_mergers']
    
    # Loading sample criteria
    output_input        = dict_output['output_input']
    
    assert merger_tree_input['mySims'][0][1] == output_input['mySims'][0][1], 'Wrong merger tree'
    
    
    #---------------------------------
    # Extract GroupNum, SubGroupNum, and Snap for each ID
    GalaxyID_List_target = list(total_general.keys())
        
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  GalaxyIDs: %s' %(output_input['mySims'][0][0], GalaxyID_List_target))
    print('  SAMPLE LENGTH: ', len(GalaxyID_List_target))
    print('\nOUTPUT LOADED:\n  Min Snap: %s\n  Max Snap: %s\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['snapNumMin'], output_input['snapNumMax'],output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle:    %s\n  Uncertainties:      %s\n  Highlight Criteria: %s\n  Rad or HMR:         %s' %(use_angles, use_hmr, use_proj_angle, use_uncertainties, highlight_criteria, rad_type_plot))
    print('===================')
    

    #------------------------------
    # Check if requested plot is possible with loaded data
    for use_angles_i in use_angles:
        assert use_angles_i in output_input['angle_selection'], 'Requested angle %s not in output_input' %use_angles_i
    for use_hmr_i in use_hmr:
        assert use_hmr_i in output_input['spin_hmr'], 'Requested HMR %s not in output_input' %use_hmr_i
    if use_uncertainties:
        assert use_uncertainties == output_input['find_uncertainties'], 'Output file does not contain uncertainties to plot'
    
    #=======================================
    # Iterate over all GalaxyIDList_targets
    for target_GalaxyID in tqdm(np.array(GalaxyID_List_target)):
        
        #---------------------------------------------------
        # Graph initialising and base formatting
        fig, axs = plt.subplots(nrows=5, ncols=1, gridspec_kw={'height_ratios': [4, 4, 2, 2, 2]}, figsize=[5.5, 12.5], sharex=True, sharey=False)
        
    
        #----------------------------
        # Add mergers
        for galaxyID_i, ratio_i, gas_ratio_i, lookbacktime_i, snap_i in zip(total_mergers['%s' %target_GalaxyID]['GalaxyIDs'], total_mergers['%s' %target_GalaxyID]['ratios'], total_mergers['%s' %target_GalaxyID]['gasratios'], total_mainbranch['%s' %target_GalaxyID]['lookbacktime'], total_mainbranch['%s' %target_GalaxyID]['snapnum']):
            
            if len(ratio_i) == 0:
                continue
            
            else:
                if max(ratio_i) >= min_merger_ratio:
                    for ax in axs:
                        ax.axvline(lookbacktime_i, ls='--', color='grey', alpha=0.8, linewidth=2)

                    if snap_i > int(output_input['snapNumMin']):
                        
                        # Manually finding merger masses
                        primary_stelmass  = float(merger_tree['%s' %(int(target_GalaxyID) +  28 - int(snap_i))]['%s' %(int(target_GalaxyID) +  28 - int(snap_i) + 1)]['stelmass'])
                        primary_gasmass   = float(merger_tree['%s' %(int(target_GalaxyID) +  28 - int(snap_i))]['%s' %(int(target_GalaxyID) +  28 - int(snap_i) + 1)]['gasmass'])
                        primary_gassfmass = float(merger_tree['%s' %(int(target_GalaxyID) +  28 - int(snap_i))]['%s' %(int(target_GalaxyID) +  28 - int(snap_i) + 1)]['gasmass_sf'])
                        for mergerID_i in galaxyID_i[np.argmax(np.array(ratio_i))]:
                            # if primary, skip
                            if int(mergerID_i) == (int(target_GalaxyID) +  28 - int(snap_i) + 1):
                                continue
                            
                            secondary_stelmass  = float(merger_tree['%s' %(int(target_GalaxyID) +  28 - int(snap_i))]['%s' %mergerID_i]['stelmass'])
                            secondary_gasmass   = float(merger_tree['%s' %(int(target_GalaxyID) +  28 - int(snap_i))]['%s' %mergerID_i]['gasmass'])
                            secondary_gassfmass = float(merger_tree['%s' %(int(target_GalaxyID) +  28 - int(snap_i))]['%s' %mergerID_i]['gasmass_sf'])
                        
                        # Find merger ratios
                        manual_ratio = secondary_stelmass/primary_stelmass
                        manual_gas_ratio = secondary_gasmass/primary_gasmass
                        manual_gassf_ratio = secondary_gassfmass/primary_gassfmass

                    # Annotate
                    axs[0].text(lookbacktime_i-0.2, 165, '%.2f' %manual_ratio, color='grey')
                    #axs[0].text(lookbacktime_i-0.2, 160, '%.2f' %manual_gas_ratio, color='green')
                    axs[0].text(lookbacktime_i-0.2, 150, '%.2f' %manual_gassf_ratio, color='blue')
        
        #----------------------------
        # Add time spent as satellite
        time_start = 0
        for SubGroupNum_i, lookbacktime_i, snap_i in zip(total_mainbranch['%s' %target_GalaxyID]['SubGroupNumber'], total_mainbranch['%s' %target_GalaxyID]['lookbacktime'], total_mainbranch['%s' %target_GalaxyID]['snapnum']):
            if (SubGroupNum_i == 0) & (time_start == 0):
                continue
            elif (SubGroupNum_i != 0) & (time_start == 0):
                time_start = lookbacktime_i
                time_end = lookbacktime_i
            elif (SubGroupNum_i != 0) & (time_start != 0):
                time_end = lookbacktime_i
                continue
            elif (SubGroupNum_i == 0) & (time_start != 0):
                time_end = lookbacktime_i
                for ax in axs:
                    ax.axvspan(time_start, time_end, facecolor='grey', alpha=0.2)
                
                time_start = 0
                time_end = 0
        
        
        #===========================
        # Loop over each angle type
        for plot_count, use_angle_i in enumerate(use_angles):
            
            #------------------------
            # Setting colors and labels
            if use_angle_i == 'stars_gas':
                plot_color = 'green'
                plot_cmap  = plt.get_cmap('Greens')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-gas'
                use_particles = ['stars', 'gas']
            elif use_angle_i == 'stars_gas_sf':
                plot_color = 'b'
                plot_cmap  = plt.get_cmap('Blues')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-gas$_{sf}$'
                use_particles = ['stars', 'gas_sf']
            elif use_angle_i == 'stars_gas_nsf':
                plot_color = 'indigo'
                plot_cmap  = plt.get_cmap('Purples')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-gas$_{nsf}$'
                use_particles = ['stars', 'gas_nsf']
            elif use_angle_i == 'stars_dm':
                plot_color = 'r'
                plot_cmap  = plt.get_cmap('Reds')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-DM'
                use_particles = ['stars', 'nsf']
            else:
                plot_color = 'brown'
                plot_cmap  = plt.get_cmap('Greys')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = use_angle_i
                raise Exception('Not configured other plot types yet')
            
            if output_input['rad_projected'] == True:
                r50 = '$r_{1/2,z}$'
            if output_input['rad_projected'] == False:
                r50 = '$r_{1/2}$'
            
            # Create some colormaps of things we want
            #colors_blues    = plt.get_cmap('Blues')(np.linspace(0.4, 0.9, len(use_hmr)))
            #colors_reds     = plt.get_cmap('Reds')(np.linspace(0.4, 0.9, len(use_hmr)))
            #colors_greens   = plt.get_cmap('Greens')(np.linspace(0.4, 0.9, len(use_hmr)))
            #colors_spectral = plt.get_cmap('Spectral_r')(np.linspace(0.05, 0.95, len(use_hmr)))
            plot_cmap = plt.get_cmap('tab10')(np.arange(0, len(use_hmr), 1))

            #-----------------------
            # Loop over each rad
            for hmr_i, color_angle in zip(np.flip(use_hmr), plot_cmap):

                # Create empty arrays to plot
                plot_lookbacktime = []
                plot_redshift     = []

                plot_angles         = []
                plot_angles_small   = []
                plot_angles_lo      = []
                plot_angles_hi      = []
                plot_angles_proj        = []
                plot_angles_proj_small  = []
                plot_angles_proj_lo     = []
                plot_angles_proj_hi     = []

                plot_stelmass     = []
                plot_gasmass      = []
                plot_gassfmass    = []
                
                plot_gas_frac     = []
                plot_gassf_frac   = []


                #-------------------------------------
                # Same as taking 'for redshift in XX'
                for GalaxyID_i, lookbacktime_i, redshift_i in zip(np.array(total_mainbranch['%s' %target_GalaxyID]['GalaxyID']), np.array(total_mainbranch['%s' %target_GalaxyID]['lookbacktime']), np.array(total_mainbranch['%s' %target_GalaxyID]['redshift'])):
                       
                    #----------
                    # Extracting misalignment angles
                    if use_proj_angle == True:
                        if (hmr_i in total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles_proj.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                            else:
                                plot_angles_proj.append(math.nan)
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles_proj_small.append(math.nan)
                            plot_angles_proj.append(math.nan)
                            plot_angles_proj_lo.append(math.nan)
                            plot_angles_proj_hi.append(math.nan)   
                    elif use_proj_angle == False:
                        if (hmr_i in total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                            else:
                                plot_angles.append(math.nan)
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles.append(math.nan)
                            plot_angles_small.append(math.nan)
                            plot_angles_lo.append(math.nan)
                            plot_angles_hi.append(math.nan)
                    elif use_proj_angle == 'both':
                        if (hmr_i in total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles_proj.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                            else:
                                plot_angles_proj.append(math.nan)
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles_proj_small.append(math.nan)
                            plot_angles_proj.append(math.nan)
                            plot_angles_proj_lo.append(math.nan)
                            plot_angles_proj_hi.append(math.nan)
                        
                        if (hmr_i in total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                            else:
                                plot_angles.append(math.nan)
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles.append(math.nan)
                            plot_angles_small.append(math.nan)
                            plot_angles_lo.append(math.nan)
                            plot_angles_hi.append(math.nan)
                        

                    #----------
                    # Gather gas masses and fractions
                    if hmr_i in use_hmr_frac:
                        if hmr_i in total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']:
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']) == hmr_i)[0])
                            
                            plot_stelmass.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars'][mask_rad])
                            plot_gasmass.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad])
                            plot_gassfmass.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas_sf'][mask_rad])
                            
                            plot_gas_frac.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad] / (total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars'][mask_rad] + total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad]))
                            plot_gassf_frac.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas_sf'][mask_rad] / (total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars'][mask_rad] + total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad]))
                            
                        else:
                            plot_stelmass.append(math.nan)
                            plot_gasmass.append(math.nan)
                            plot_gassfmass.append(math.nan)
                            plot_gas_frac.append(math.nan)
                            plot_gassf_frac.append(math.nan)

                    
                    #----------
                    # Gather times
                    plot_redshift.append(redshift_i)
                    plot_lookbacktime.append(lookbacktime_i)   
                
                
                if debug:
                    print('Target galaxy: %s' %target_GalaxyID)
                    print('. plot_lookbacktime: ', lookbacktime_plot)
                    print('\n  plot_angles: ', plot_angles)
                    print('  plot_angles_lo: ', plot_angles_lo)
                    print('  plot_angles_hi: ', plot_angles_hi)
                    print('\n  plot_angles_proj: ', plot_angles_proj)
                    print('  plot_angles_proj_lo: ', plot_angles_proj_lo)
                    print('  plot_angles_proj_hi: ', plot_angles_proj_hi)
                    print('\n  plot_stelmass: ', plot_stelmass)
                    print('  plot_gasmass: ', plot_gasmass)
                    print('  plot_gassfmass: ', plot_gassfmass)
                    print('\n  plot_gas_frac: ', plot_gas_frac)
                    print('  plot_gassf_frac: ', plot_gassf_frac)
                    
                #====================================
                ### Plotting
                # Plot 1: Misalignment angles, errors, with time/redshift
                # Plot 2: Stellar mass, gas mass, gas sf mass, with time/redshift
                # Plot 3: Gas fractions with time/redshift
                
                #------------------------
                # PLOT 1
                # Plot scatter and errorbars
                
                if use_proj_angle == True:
                    axs[0].plot(plot_lookbacktime, plot_angles_proj_small, alpha=1, ms=2, ls=':', c='grey', zorder=8)
                    axs[0].plot(plot_lookbacktime, plot_angles_proj, alpha=1, ms=2, ls='-', c=color_angle, zorder=10, label='%.0f%s' %(hmr_i, r50))
                        
                    if use_uncertainties:
                        axs[0].fill_between(plot_lookbacktime, plot_angles_proj_lo, plot_angles_proj_hi, alpha=0.25, color=color_angle, lw=0, zorder=5)
                elif use_proj_angle == False:
                    axs[0].plot(plot_lookbacktime, plot_angles_small, alpha=1, ms=2, ls=':', c='grey', zorder=8)
                    axs[0].plot(plot_lookbacktime, plot_angles, alpha=1, ms=2, ls='-', c=color_angle, zorder=10, label='%.0f%s' %(hmr_i, r50))
                    
                    if use_uncertainties:
                        axs[0].fill_between(plot_lookbacktime, plot_angles_lo, plot_angles_hi, alpha=0.25, color=color_angle, lw=0, zorder=5)
                elif use_proj_angle == 'both':
                    axs[0].plot(plot_lookbacktime, plot_angles_proj_small, alpha=1, ms=2, ls=':', c='grey', zorder=10)
                    axs[0].plot(plot_lookbacktime, plot_angles_proj, alpha=1, ms=2, ls='-', c=color_angle, zorder=10, label='%.0f%s' %(hmr_i, r50))

                    axs[0].plot(plot_lookbacktime, plot_angles_small, alpha=1, ms=2, ls=':', c='grey', zorder=8)
                    axs[0].plot(plot_lookbacktime, plot_angles, alpha=1, ms=2, ls='dashdot', c=color_angle, zorder=10)
                    
                    if use_uncertainties:
                        axs[0].fill_between(plot_lookbacktime, plot_angles_proj_lo, plot_angles_proj_hi, alpha=0.25, color=color_angle, lw=0, zorder=5)
                    
                # Only plot the masses and fractions once, regardless of angle type
                if plot_count == 0:
                    if hmr_i in use_hmr_frac:
                        #------------------------
                        # PLOT 2
                        # Plot masses
                        axs[3].plot(plot_lookbacktime, np.log10(np.array(plot_stelmass)), alpha=1.0, lw=1.2, c='r', label='$M_*$')
                        #axs[3].plot(plot_lookbacktime, np.log10(np.array(plot_gasmass)), alpha=1.0, lw=1.2, c='g', label='$M_{\mathrm{gas}}$')
                        axs[3].plot(plot_lookbacktime, np.log10(np.array(plot_gassfmass)), alpha=1.0, lw=1.2, c='b', label='$M_{\mathrm{SF}}$', markerfacecolor='None')
                    
                
                #----------
                # Lists all angles at a given HMR_i over time
                if debug:
                    print('\nhmr: ', hmr_i )
                    for time_i, angllle_i in zip(plot_lookbacktime, plot_angles):
                        print('Time: %.2f   Angle: %.1f' %(time_i, angllle_i))
        
        
        
        #----------
        # Lists all angles and hmr fr a given time        
        if debug:
            for GalaxyID_i, lookbacktime_i, redshift_i in zip(np.array(total_mainbranch['%s' %target_GalaxyID]['GalaxyID']), np.array(total_mainbranch['%s' %target_GalaxyID]['lookbacktime']), np.array(total_mainbranch['%s' %target_GalaxyID]['redshift'])):
                print('Lookbacktime: %.2f' %lookbacktime_i)
                for i, hmr_i in enumerate(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']):
                    print('HMR: %.2f   Angle: %.1f' %(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr'][i], total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars_gas_sf_angle'][i]))
            
        
        #------------------------
        # PLOT 2 & 3
        # Find bh accretion rate, mass flow, edd
        inflow_rate         = []
        outflow_rate        = []
        stelmassloss_rate   = []
        inflow_Z            = []
        outflow_Z           = []
        insitu_Z            = []
        bh_mdot             = []
        bh_edd              = []

        # Plot inflow for all hmr we care about
        for hmr_i in use_hmr_frac:
            lookbacktime_old = math.nan
            for GalaxyID_i, lookbacktime_i, i,  in zip(np.array(total_mainbranch['%s' %target_GalaxyID]['GalaxyID']), np.array(total_mainbranch['%s' %target_GalaxyID]['lookbacktime']), np.arange(len(total_mainbranch['%s' %target_GalaxyID]['lookbacktime']))):

                # Append and convert bhmdot (as difference of masses rather than mdot)
                if i == 0:
                    bh_mdot.append(math.nan)
                    bh_edd.append(math.nan)
                else:
                    bh_mdot.append(1000* (float(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['bh_mass']) - float(total_general['%s' %str(target_GalaxyID)]['%s' %str(int(GalaxyID_i)+1)]['bh_mass'])) / ((lookbacktime_old - lookbacktime_i) * 1e9))
                    bh_edd.append(np.log10(float(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['bh_edd'])))
                    
                    if debug:
                        print('bh masses')
                        print(float(total_general['%s' %str(target_GalaxyID)]['%s' %str(int(GalaxyID_i)+1)]['bh_mass']))
                        print(float(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['bh_mass']))
                        print((float(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['bh_mass']) - float(total_general['%s' %str(target_GalaxyID)]['%s' %str(int(GalaxyID_i)+1)]['bh_mass'])) / ((lookbacktime_old - lookbacktime_i) * 1e9))
                
                # Append inflow/outflow/stelmassloss
                inflow_rate.append(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas_sf']['inflow'])/((lookbacktime_old - lookbacktime_i) * 1e9))
                outflow_rate.append(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas_sf']['outflow'])/((lookbacktime_old - lookbacktime_i) * 1e9))
                stelmassloss_rate.append(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas_sf']['massloss'])/((lookbacktime_old - lookbacktime_i) * 1e9))
                
                 # Append inflow/outflow/insitu metallicity
                inflow_Z.append(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas_sf']['inflow_Z']))
                outflow_Z.append(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas_sf']['outflow_Z']))
                insitu_Z.append(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas_sf']['insitu_Z']))
                 
                
                if debug:
                    print('delta time', lookbacktime_old - lookbacktime_i)
                    print('RAW MASS LOSSES:')
                    print(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas']['inflow'])
                    print(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas']['outflow'])
                    print(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas']['massloss'])
                    print('MASS LOSS RATE:')
                    print(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas']['inflow'])/((lookbacktime_old - lookbacktime_i) * 1e9))
                    print(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas']['outflow'])/((lookbacktime_old - lookbacktime_i) * 1e9))
                    print(float(total_massflow['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['%s_hmr' %float(hmr_i)]['gas']['massloss'])/((lookbacktime_old - lookbacktime_i) * 1e9))
                
                lookbacktime_old = lookbacktime_i
                
            
            if print_summary:
                print('-- mass flow --')
                for time_i, in_i, out_i, inmetal_i, outmetal_i, insitmetal_i, stel_i, bh_i in zip(plot_lookbacktime, inflow_rate, outflow_rate, stelmassloss_rate, inflow_Z, outflow_Z, insitu_Z, bh_mdot):
                    print('%.2f |  %.2f     %.2f     %.2f  |  %.2f     %.2f     %.2f  |   %.2f' %(time_i, in_i, out_i, stel_i, inmetal_i, outmetal_i, insitmetal_i, bh_i*1000))
                
                
            # Plot mass rates
            axs[1].plot(plot_lookbacktime, inflow_rate, alpha=1.0, lw=1.2, c='g', label='Gas$_{\mathrm{SF}}$ inflow')
            axs[1].plot(plot_lookbacktime, outflow_rate, alpha=1.0, lw=1.2, c='r', label='Gas$_{\mathrm{SF}}$ outflow')
            #axs[1].plot(plot_lookbacktime, stelmassloss_rate, alpha=1.0, lw=1.5, ls='dashdot', label='Stellar mass loss')
            axs[1].plot(plot_lookbacktime, bh_mdot, alpha=1.0, lw=1.2, c='purple', ls='dashdot', label='$\dot{M}_{\mathrm{BH}}$(×$10^3$)')
            axs[2].plot(plot_lookbacktime, bh_edd, alpha=1.0, lw=1.2, c='k', ls='-', label='$\lambda_{\mathrm{Edd}}$')
            
        
        
        #------------------------
        # PLOT 4
        # Find kappas and other _general stats
        kappa_stars       = []
        kappa_gas_sf      = []

        # Same as taking 'for redshift in XX'
        for GalaxyID_i, lookbacktime_i in zip(np.array(total_mainbranch['%s' %target_GalaxyID]['GalaxyID']), np.array(total_mainbranch['%s' %target_GalaxyID]['lookbacktime'])):
            kappa_stars.append(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['kappa_stars'])
            kappa_gas_sf.append(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['kappa_gas_sf'])
                
        # Plot kappas
        axs[4].axhline(0.4, lw=1, ls='--', c='grey', alpha=0.7)
        axs[4].text(7.9, 0.44, ' LTG', color='grey')
        axs[4].text(7.9, 0.26, ' ETG', color='grey')
        axs[4].plot(plot_lookbacktime, kappa_stars, alpha=1.0, lw=1.2, c='r', label='$\kappa_{\mathrm{co}}^*$')
        axs[4].plot(plot_lookbacktime, kappa_gas_sf, alpha=1.0, lw=1.2, c='b', label='$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
                
                
        #=============================
        ### General formatting 
        
        ### Customise legend labels
        axs[0].legend(loc='upper left', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        axs[1].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        axs[2].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        axs[3].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        axs[4].legend(loc='center right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)

        #------------------------
        # Create redshift axis:
        redshiftticks = [0, 0.1, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20]
        ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
        for i, ax in enumerate(axs):
            ax_top = ax.twiny()
            ax_top.set_xticks(ageticks)

            ax.set_xlim(0, 8)
            ax_top.set_xlim(0, 8)

            if i == 0:
                ax.set_ylim(0, 180)

                ax_top.set_xlabel('Redshift')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')

                ax.set_yticks(np.arange(0, 181, 30))
                ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
                ax.set_ylabel('Misalignment angle, $\psi$') 

                #ax.set_title('GalaxyID: %s' %str(target_GalaxyID))
                ax.text(8, 205, 'GalaxyID: %s' %str(target_GalaxyID), fontsize=8)
                ax.invert_xaxis()
                ax_top.invert_xaxis()
                
            if i == 1:
                #ax.set_ylim(0, 20)
                ax.set_ylabel('$\mathrm{log_{10}}(\mathrm{M}/\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')

                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                ax.invert_xaxis()
                ax_top.invert_xaxis()
            
            if i == 2:
                #ax.set_ylim(-4, 0)
                ax.set_ylabel('$\mathrm{log_{10}}(\lambda_{\mathrm{Edd}})$')

                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                ax.invert_xaxis()
                ax_top.invert_xaxis()
                
            if i == 3:
                ax.set_ylim(7.9, 12.1)
                ax.set_ylabel('$\mathrm{log_{10}}(\mathrm{M}/\mathrm{M}_{\odot})$')

                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                ax.invert_xaxis()
                ax_top.invert_xaxis()
            
            if i == 4:
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.1, 0.25))

                ax.set_xlabel('Lookback time (Gyr)')
                ax.set_ylabel('$\kappa_{\mathrm{co}}$')

                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                ax.invert_xaxis()
                ax_top.invert_xaxis()

            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='minor')
            
        #------------------------
        # Other
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.12)
            
            
        #=====================================
        ### Print summary
        
        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        
        metadata_plot = {'Title': 'GalaxyID: %s\nM*: %.2e\nHMR: %.2f\nKappa: %.2f\nTriax: %.2f' %(target_GalaxyID, total_general['%s' %target_GalaxyID]['%s' %target_GalaxyID]['stelmass'], total_general['%s' %target_GalaxyID]['%s' %target_GalaxyID]['halfmass_rad_proj'], total_general['%s' %target_GalaxyID]['%s' %target_GalaxyID]['kappa_stars'], total_general['%s' %target_GalaxyID]['%s' %target_GalaxyID]['triax'])}
        
        
        angle_str = ''
        for angle_name in list(use_angles):
            angle_str = '%s_%s' %(str(angle_str), str(angle_name))
        
        
        if savefig:
            plt.savefig("%s/individual_evolution/L%s_evolution_ID%s_proj%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], target_GalaxyID, use_proj_angle, angle_str, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/individual_evolution/L%s_evolution_ID%s_proj%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], target_GalaxyID, use_proj_angle, angle_str, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
 
#--------------------
# Will plot evolution of single galaxy       
# SAVED: /plots/individual_evolution/                                                                           
def _plot_evolution_old(csv_output = 'L100_evolution_ID401467650_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',   # CSV sample file to load 
                           #--------------------------
                           # Galaxy plotting
                           print_summary = True,
                             use_angles         = ['stars_gas_sf'],                 # Which angles to plot
                             use_hmr            = [1, 2],         # Which misangle HMR to plot
                             use_hmr_frac       = [2],                # Which mass and fraction HMR to plot             
                             use_proj_angle     = False,                   # Whether to use projected or absolute angle, 'both'
                             use_uncertainties  = True,                   # Whether to plot uncertainties or not
                             min_merger_ratio   = 0.05,
                           #-------------------------
                           # Plot settings
                           highlight_criteria = True,       # whether to indicate when criteria not met (but still plot)
                           rad_type_plot      = 'hmr',      # 'rad' whether to use absolute distance or hmr 
                           #--------------------------
                           showfig        = False,
                           savefig        = True,
                             file_format  = 'pdf',
                             savefig_txt  = '',
                           #--------------------------
                           print_progress = False,
                           debug = False):
    
    
    
    #================================================  
    # Load sample csv
    if print_progress:
        print('Loading output')
        time_start = time.time()
    
    #--------------------------------
    # Loading output
    dict_output = json.load(open('%s/%s.csv' %(output_dir, csv_output), 'r'))
    
    total_flags           = dict_output['total_flags']
    total_general         = dict_output['total_general']
    total_spins           = dict_output['total_spins']
    total_counts          = dict_output['total_counts']
    total_masses          = dict_output['total_masses']
    total_coms            = dict_output['total_coms']
    total_misangles       = dict_output['total_misangles']
    total_misanglesproj   = dict_output['total_misanglesproj']
    
    total_allbranches     = dict_output['total_allbranches']
    total_mainbranch      = dict_output['total_mainbranch']
    total_mergers         = dict_output['total_mergers']
    
    # Loading sample criteria
    output_input        = dict_output['output_input']
    
    #---------------------------------
    # Extract GroupNum, SubGroupNum, and Snap for each ID
    GalaxyID_List_target = list(total_general.keys())
        
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  GalaxyIDs: %s' %(output_input['mySims'][0][0], GalaxyID_List_target))
    print('  SAMPLE LENGTH: ', len(GalaxyID_List_target))
    print('\nOUTPUT LOADED:\n  Max Snap: %s\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['snapNumMax'], output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle:    %s\n  Uncertainties:      %s\n  Highlight Criteria: %s\n  Rad or HMR:         %s' %(use_angles, use_hmr, use_proj_angle, use_uncertainties, highlight_criteria, rad_type_plot))
    print('===================')
    

    #------------------------------
    # Check if requested plot is possible with loaded data
    for use_angles_i in use_angles:
        assert use_angles_i in output_input['angle_selection'], 'Requested angle %s not in output_input' %use_angles_i
    for use_hmr_i in use_hmr:
        assert use_hmr_i in output_input['spin_hmr'], 'Requested HMR %s not in output_input' %use_hmr_i
    if use_uncertainties:
        assert use_uncertainties == output_input['find_uncertainties'], 'Output file does not contain uncertainties to plot'
    
    #=======================================
    # Iterate over all GalaxyIDList_targets
    for target_GalaxyID in tqdm(np.array(GalaxyID_List_target)):
        
        
        #---------------------------------------------------
        # Graph initialising and base formatting
        fig, axs = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1.5]}, figsize=[7, 15], sharex=True, sharey=False)
        
    
        #----------------------------
        # Add mergers
        for ratio_i, lookbacktime_i, snap_i in zip(total_mergers['%s' %target_GalaxyID]['ratios'], total_mainbranch['%s' %target_GalaxyID]['lookbacktime'], total_mainbranch['%s' %target_GalaxyID]['snapnum']):
            if len(ratio_i) == 0:
                continue
            else:
                if max(ratio_i) >= min_merger_ratio:
                    for ax in axs:
                        ax.axvline(lookbacktime_i, ls='--', color='grey', alpha=1, linewidth=2)

                    # Annotate
                    axs[0].text(lookbacktime_i-0.2, 170, '%.2f' %max(ratio_i), color='grey')
        
        #----------------------------
        # Add time spent as satellite
        time_start = 0
        for SubGroupNum_i, lookbacktime_i, snap_i in zip(total_mainbranch['%s' %target_GalaxyID]['SubGroupNumber'], total_mainbranch['%s' %target_GalaxyID]['lookbacktime'], total_mainbranch['%s' %target_GalaxyID]['snapnum']):
            if (SubGroupNum_i == 0) & (time_start == 0):
                continue
            elif (SubGroupNum_i != 0) & (time_start == 0):
                time_start = lookbacktime_i
                time_end = lookbacktime_i
            elif (SubGroupNum_i != 0) & (time_start != 0):
                time_end = lookbacktime_i
                continue
            elif (SubGroupNum_i == 0) & (time_start != 0):
                time_end = lookbacktime_i
                for ax in axs:
                    ax.axvspan(time_start, time_end, facecolor='grey', alpha=0.2)
                
                time_start = 0
                time_end = 0
        
        
        #===========================
        # Loop over each angle type
        for plot_count, use_angle_i in enumerate(use_angles):
            
            #------------------------
            # Setting colors and labels
            if use_angle_i == 'stars_gas':
                plot_color = 'green'
                plot_cmap  = plt.get_cmap('Greens')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-gas'
                use_particles = ['stars', 'gas']
            elif use_angle_i == 'stars_gas_sf':
                plot_color = 'b'
                plot_cmap  = plt.get_cmap('Blues')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-gas$_{sf}$'
                use_particles = ['stars', 'gas_sf']
            elif use_angle_i == 'stars_gas_nsf':
                plot_color = 'indigo'
                plot_cmap  = plt.get_cmap('Purples')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-gas$_{nsf}$'
                use_particles = ['stars', 'gas_nsf']
            elif use_angle_i == 'stars_dm':
                plot_color = 'r'
                plot_cmap  = plt.get_cmap('Reds')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = 'Stars-DM'
                use_particles = ['stars', 'nsf']
            else:
                plot_color = 'brown'
                plot_cmap  = plt.get_cmap('Greys')(np.linspace(0.4, 0.9, len(use_hmr)))
                plot_label = use_angle_i
                raise Exception('Not configured other plot types yet')
            
            if output_input['rad_projected'] == True:
                r50 = '$r_{1/2,z}$'
            if output_input['rad_projected'] == False:
                r50 = '$r_{1/2}$'
            
            # Create some colormaps of things we want
            #colors_blues    = plt.get_cmap('Blues')(np.linspace(0.4, 0.9, len(use_hmr)))
            #colors_reds     = plt.get_cmap('Reds')(np.linspace(0.4, 0.9, len(use_hmr)))
            #colors_greens   = plt.get_cmap('Greens')(np.linspace(0.4, 0.9, len(use_hmr)))
            #colors_spectral = plt.get_cmap('Spectral_r')(np.linspace(0.05, 0.95, len(use_hmr)))
            plot_cmap = plt.get_cmap('tab10')(np.arange(0, len(use_hmr), 1))

            #-----------------------
            # Loop over each rad
            for hmr_i, color_angle in zip(np.flip(use_hmr), plot_cmap):

                # Create empty arrays to plot
                plot_lookbacktime = []
                plot_redshift     = []

                plot_angles         = []
                plot_angles_small   = []
                plot_angles_lo      = []
                plot_angles_hi      = []
                plot_angles_proj        = []
                plot_angles_proj_small  = []
                plot_angles_proj_lo     = []
                plot_angles_proj_hi     = []

                plot_stelmass     = []
                plot_gasmass      = []
                plot_gassfmass    = []
                
                plot_gas_frac     = []
                plot_gassf_frac   = []


                #-------------------------------------
                # Same as taking 'for redshift in XX'
                for GalaxyID_i, lookbacktime_i, redshift_i in zip(np.array(total_mainbranch['%s' %target_GalaxyID]['GalaxyID']), np.array(total_mainbranch['%s' %target_GalaxyID]['lookbacktime']), np.array(total_mainbranch['%s' %target_GalaxyID]['redshift'])):
                                    
                    #----------
                    # Extracting misalignment angles
                    if use_proj_angle == True:
                        if (hmr_i in total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles_proj.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                            else:
                                plot_angles_proj.append(math.nan)
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles_proj_small.append(math.nan)
                            plot_angles_proj.append(math.nan)
                            plot_angles_proj_lo.append(math.nan)
                            plot_angles_proj_hi.append(math.nan)   
                    elif use_proj_angle == False:
                        if (hmr_i in total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                            else:
                                plot_angles.append(math.nan)
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles.append(math.nan)
                            plot_angles_small.append(math.nan)
                            plot_angles_lo.append(math.nan)
                            plot_angles_hi.append(math.nan)
                    elif use_proj_angle == 'both':
                        if (hmr_i in total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_inclination'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles_proj.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])                      
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                            else:
                                plot_angles_proj.append(math.nan)
                                plot_angles_proj_small.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_proj_lo.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_proj_hi.append(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles_proj_small.append(math.nan)
                            plot_angles_proj.append(math.nan)
                            plot_angles_proj_lo.append(math.nan)
                            plot_angles_proj_hi.append(math.nan)
                        
                        if (hmr_i in total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']):
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']) == hmr_i)[0])
                            
                            if (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[0]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['min_particles'][use_particles[1]]) and (hmr_i not in total_flags['%s' %target_GalaxyID]['%s' %GalaxyID_i]['com_min_distance'][use_angle_i]) and (total_general['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stelmass'] > 1E9):
                                plot_angles.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                            else:
                                plot_angles.append(math.nan)
                                plot_angles_small.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle' %use_angle_i][mask_rad])
                                plot_angles_lo.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][0])
                                plot_angles_hi.append(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['%s_angle_err' %use_angle_i][mask_rad][1])
                                
                        else:
                            # append nans if hmr_i was trimmed
                            plot_angles.append(math.nan)
                            plot_angles_small.append(math.nan)
                            plot_angles_lo.append(math.nan)
                            plot_angles_hi.append(math.nan)
                        

                    #----------
                    # Gather gas masses and fractions
                    if hmr_i in use_hmr_frac:
                        if hmr_i in total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']:
                            # Mask correct integer (formatting weird but works)
                            mask_rad = int(np.where(np.array(total_misanglesproj['%s' %target_GalaxyID]['%s' %GalaxyID_i][output_input['viewing_axis']]['hmr']) == hmr_i)[0])
                            
                            plot_stelmass.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars'][mask_rad])
                            plot_gasmass.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad])
                            plot_gassfmass.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas_sf'][mask_rad])
                            
                            plot_gas_frac.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad] / (total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars'][mask_rad] + total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad]))
                            plot_gassf_frac.append(total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas_sf'][mask_rad] / (total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars'][mask_rad] + total_masses['%s' %target_GalaxyID]['%s' %GalaxyID_i]['gas'][mask_rad]))
                            
                        else:
                            plot_stelmass.append(math.nan)
                            plot_gasmass.append(math.nan)
                            plot_gassfmass.append(math.nan)
                            plot_gas_frac.append(math.nan)
                            plot_gassf_frac.append(math.nan)

                    
                    #----------
                    # Gather times
                    plot_redshift.append(redshift_i)
                    plot_lookbacktime.append(lookbacktime_i)   
                
                
                if debug:
                    print('Target galaxy: %s' %target_GalaxyID)
                    print('. plot_lookbacktime: ', lookbacktime_plot)
                    print('\n  plot_angles: ', plot_angles)
                    print('  plot_angles_lo: ', plot_angles_lo)
                    print('  plot_angles_hi: ', plot_angles_hi)
                    print('\n  plot_angles_proj: ', plot_angles_proj)
                    print('  plot_angles_proj_lo: ', plot_angles_proj_lo)
                    print('  plot_angles_proj_hi: ', plot_angles_proj_hi)
                    print('\n  plot_stelmass: ', plot_stelmass)
                    print('  plot_gasmass: ', plot_gasmass)
                    print('  plot_gassfmass: ', plot_gassfmass)
                    print('\n  plot_gas_frac: ', plot_gas_frac)
                    print('  plot_gassf_frac: ', plot_gassf_frac)
                    
                #====================================
                ### Plotting
                # Plot 1: Misalignment angles, errors, with time/redshift
                # Plot 2: Stellar mass, gas mass, gas sf mass, with time/redshift
                # Plot 3: Gas fractions with time/redshift
                
                #------------------------
                # PLOT 1
                # Plot scatter and errorbars
                
                if use_proj_angle == True:
                    axs[0].plot(plot_lookbacktime, plot_angles_proj, alpha=1, ms=2, lw=2, ls='-', c=color_angle, zorder=10, label='%.1f %s' %(hmr_i, r50))
                    axs[0].plot(plot_lookbacktime, plot_angles_proj_small, alpha=1, ms=2, lw=1.5, ls=':', c=color_angle, zorder=8)
                    
                    if use_uncertainties:
                        axs[0].fill_between(plot_lookbacktime, plot_angles_proj_lo, plot_angles_proj_hi, alpha=0.25, color=color_angle, lw=0, zorder=5)
                elif use_proj_angle == False:
                    axs[0].plot(plot_lookbacktime, plot_angles, alpha=1, ms=2, lw=2, ls='-', c=color_angle, zorder=10, label='%.1f %s' %(hmr_i, r50))
                    axs[0].plot(plot_lookbacktime, plot_angles_small, alpha=1, ms=2, lw=1.5, ls=':', c=color_angle, zorder=8)
                    
                    if use_uncertainties:
                        axs[0].fill_between(plot_lookbacktime, plot_angles_lo, plot_angles_hi, alpha=0.25, color=color_angle, lw=0, zorder=5)
                elif use_proj_angle == 'both':
                    axs[0].plot(plot_lookbacktime, plot_angles_proj, alpha=1, ms=2, lw=2, ls='-', c=color_angle, zorder=10, label='%.1f %s' %(hmr_i, r50))
                    axs[0].plot(plot_lookbacktime, plot_angles_proj_small, alpha=1, ms=2, lw=2, ls='--', c=color_angle, zorder=10)
                    
                    axs[0].plot(plot_lookbacktime, plot_angles, alpha=1, ms=2, lw=1.5, ls=':', c=color_angle, zorder=10)
                    axs[0].plot(plot_lookbacktime, plot_angles_small, alpha=1, ms=2, lw=1.5, ls=':', c=color_angle, zorder=8)
                    
                    if use_uncertainties:
                        axs[0].fill_between(plot_lookbacktime, plot_angles_proj_lo, plot_angles_proj_hi, alpha=0.25, color=color_angle, lw=0, zorder=5)
                    
                # Only plot the masses and fractions once, regardless of angle type
                if plot_count == 0:
                    if hmr_i in use_hmr_frac:
                        #------------------------
                        # PLOT 2
                        # Plot masses
                        axs[1].plot(plot_lookbacktime, np.log10(np.array(plot_stelmass)), alpha=1.0, lw=1.5, c='r', label='$M_*$')
                        axs[1].plot(plot_lookbacktime, np.log10(np.array(plot_gasmass)), alpha=1.0, lw=1.5, c='g', label='$M_{\mathrm{gas}}$')
                        axs[1].plot(plot_lookbacktime, np.log10(np.array(plot_gassfmass)), alpha=1.0, lw=1.5, c='b', label='$M_{\mathrm{SF}}$', markerfacecolor='None')
                
                        #------------------------
                        # PLOT 3
                        # Plot gas frations
                        axs[2].plot(plot_lookbacktime, plot_gas_frac, alpha=1.0, lw=1.5, c='g', label='$f_{\mathrm{gas}}$')
                        axs[2].plot(plot_lookbacktime, plot_gassf_frac, alpha=1.0, lw=1.5, c='b', label='$f_{\mathrm{SF}}$')
                    
                
                #----------
                # Lists all angles at a given HMR_i over time
                if debug:
                    print('\nhmr: ', hmr_i )
                    for time_i, angllle_i in zip(plot_lookbacktime, plot_angles):
                        print('Time: %.2f   Angle: %.1f' %(time_i, angllle_i))
        
        #----------
        # Lists all angles and hmr fr a given time        
        if debug:
            for GalaxyID_i, lookbacktime_i, redshift_i in zip(np.array(total_mainbranch['%s' %target_GalaxyID]['GalaxyID']), np.array(total_mainbranch['%s' %target_GalaxyID]['lookbacktime']), np.array(total_mainbranch['%s' %target_GalaxyID]['redshift'])):
                print('Lookbacktime: %.2f' %lookbacktime_i)
                for i, hmr_i in enumerate(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr']):
                    print('HMR: %.2f   Angle: %.1f' %(total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['hmr'][i], total_misangles['%s' %target_GalaxyID]['%s' %GalaxyID_i]['stars_gas_sf_angle'][i]))
            
        
        #------------------------
        # PLOT 4
        # Find kappas and other _general stats
        kappa_stars       = []
        kappa_gas_sf      = []

        # Same as taking 'for redshift in XX'
        for GalaxyID_i, lookbacktime_i in zip(np.array(total_mainbranch['%s' %target_GalaxyID]['GalaxyID']), np.array(total_mainbranch['%s' %target_GalaxyID]['lookbacktime'])):
            kappa_stars.append(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['kappa_stars'])
            kappa_gas_sf.append(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['kappa_gas_sf'])
                
        # Plot kappas
        axs[3].axhline(0.4, lw=1, ls='--', c='grey', alpha=0.7)
        axs[3].text(7.4, 0.43, ' LTG', color='grey')
        axs[3].text(7.4, 0.29, ' ETG', color='grey')
        axs[3].plot(plot_lookbacktime, kappa_stars, alpha=1.0, lw=1.5, c='r', label='$\kappa_{\mathrm{co}}^*$')
        axs[3].plot(plot_lookbacktime, kappa_gas_sf, alpha=1.0, lw=1.5, c='b', label='$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
                
                
        #=============================
        ### General formatting 

        ### Customise legend labels
        axs[0].legend(loc='center right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        axs[1].legend(loc='upper right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        axs[2].legend(loc='upper right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        axs[3].legend(loc='upper right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)

        #------------------------
        # Create redshift axis:
        redshiftticks = [0, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20]
        ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
        for i, ax in enumerate(axs):
            ax_top = ax.twiny()
            ax_top.set_xticks(ageticks)

            ax.set_xlim(0, 7.5)
            ax_top.set_xlim(0, 7.5)

            if i == 0:
                ax.set_ylim(0, 180)

                ax_top.set_xlabel('Redshift')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')

                ax.set_yticks(np.arange(0, 181, 30))
                ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
                ax.set_ylabel('Misalignment angle, $\psi$') 

                #ax.set_title('GalaxyID: %s' %str(target_GalaxyID))
                ax.text(7.5, 200, 'GalaxyID: %s' %str(target_GalaxyID))
                ax.invert_xaxis()
                ax_top.invert_xaxis()
            if i == 1:
                ax.set_ylim(7, 13)
                ax.set_ylabel('Mass $\mathrm{log_{10}}(\mathrm{M}/\mathrm{M}_{\odot})$')

                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                ax.invert_xaxis()
                ax_top.invert_xaxis()
            if i == 2:
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.1, 0.25))

                ax.set_ylabel('Mass fraction')

                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                ax.invert_xaxis()
                ax_top.invert_xaxis()

            if i == 3:
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.1, 0.25))

                ax.set_xlabel('Lookback time (Gyr)')
                ax.set_ylabel('$\kappa_{\mathrm{co}}$')

                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                ax.invert_xaxis()
                ax_top.invert_xaxis()

            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='minor')
            
        #------------------------
        # Other
        plt.tight_layout()
            
            
        #=====================================
        ### Print summary
        
        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        
        metadata_plot = {'Title': 'GalaxyID: %s\nM*: %.2e\nHMR: %.2f\nKappa: %.2f' %(target_GalaxyID, total_general['%s' %target_GalaxyID]['%s' %target_GalaxyID]['stelmass'], total_general['%s' %target_GalaxyID]['%s' %target_GalaxyID]['halfmass_rad_proj'], total_general['%s' %target_GalaxyID]['%s' %target_GalaxyID]['kappa_stars'])}
        
        
        angle_str = ''
        for angle_name in list(use_angles):
            angle_str = '%s_%s' %(str(angle_str), str(angle_name))
        
        
        if savefig:
            plt.savefig("%s/individual_evolution/L%s_evolution_ID%s_proj%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], target_GalaxyID, use_proj_angle, angle_str, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/individual_evolution/L%s_evolution_ID%s_proj%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], target_GalaxyID, use_proj_angle, angle_str, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
 



#--------------------
# Will plot evolution of single galaxy but with improved formatting for poster/presentation and with outflows/inflows
# SAVED: /plots/individual_evolution/
# All:
#ID_list = [108988077, 479647060, 21721896, 390595970, 401467650, 182125463, 192213531, 24276812, 116404995, 239808134, 215988755, 86715463, 6972011, 475772617, 374037507, 429352532, 441434976, 1361598, 1403994, 10421872, 17879310, 21200847, 21532243, 21659372, 24053428, 182125501, 274449295, 462956130, 462956130]
# interesting:
#ID_list = [1361598, 1403994, 10421872, 17879310, 21200847, 21532243, 21659372, 24053428, 182125501, 274449295, 462956130, 462956130]
# tim:
#ID_list = [349651696, 462956130, 182125516]
# co-co relaxations, >135 angles
ID_list = [443962, 17386687, 37720520, 74948378, 102011598, 236121860, 239192401, 239568811, 303860577, 323164883, 327004290, 349651696, 374037537, 401953578, 390652869, 271560499, 239520114, 65296062, 86568202, 470037125]
# long trelax >2 GYR
#ID_list = [115659946, 203653117, 216029810, 251900011, 273987842, 300443124, 453139727, 463220955, 390652869, 137732479, 208235276, 370237257, 239924249, 208272775, 334237852, 350073611, 92395081, 175434605, 264298155, 470037125, 444435190]
# Casanueva galaxies:
ID_list = [198707313, 248944532]

def _plot_evolution_snip(csv_tree = 'L100_galaxy_tree_',
                         #--------------------------
                         # Individual galaxies
                         GalaxyID_list = ID_list,             # [ None / ID_list ]
                         #==================================================================================
                         # Highlight criteria settings
                         highlight_criteria    = False,       # NOT WORKING whether to indicate when criteria not met (but still plot)
                           min_particles       = 20,         # [count]
                           max_com             = 2.0,        # [pkpc]
                           min_inclination     = 0,                 # [ 0 / degrees]
                           highlight_satellite = None,       # [ 'satellite' is sgn >= 1 / 'central' is sgn == 1 / None ]
                           redshift_axis       = False,      # Whether to add the redshift axis
                         #-------------------------------------------
                         # Misalignment angles                
                           use_hmr_general          = '2.0',    # [ 1.0 / 2.0 / aperture]      Used for stelmass | APERTURE NOT AVAILABLE FOR sfr ssfr
                           use_hmr_angle            = 1.0,           # [ 1.0 / 2.0 ]                Used for misangle, inc angle, com, counts
                           abs_or_proj              = 'abs',             # [ 'both' / 'abs' / 'proj' ]
                           use_angle                = 'stars_gas_sf',
                           plot_dm_angles           = True,            # Whether to add DM-stars, DM-gas_sf
                             use_dm_uncertainties   = False,              # Whether to plot uncertainties or not
                           misangle_threshold       = 30,                # [ 30 / 45 ]  
                           use_uncertainties        = True,              # Whether to plot uncertainties or not
                         #-------------------------------------------
                         # Merger settings
                           min_merger_ratio   = 0.1,
                         #-------------------------------------------
                         # Angles   [ deg ]
                           plot_angles          = True,
                           plot_inclination     = False,
                         # Inflow (gas) [ Msun/yr ]
                           plot_inflow          = True,
                           plot_outflow         = True,
                           plot_stelmassloss    = False,
                           plot_bh_acc          = False,
                           plot_bh_acc_instant  = False,
                           plot_sfr             = True,
                         # Masses   [ Msun ]
                           plot_halomass        = False,
                           plot_stelmass        = True,
                           plot_gasmass         = False,
                           plot_sfmass          = True,
                           plot_nsfmass         = False,
                           plot_bhmass          = False,
                         # sSFR     [ /yr ]
                           plot_ssfr            = False,
                         # l of stars (specific angular momentum or spin) [ pkpc/kms-1 ]
                           plot_l               = False,
                         # Radius   [ pkpc]
                           plot_radius          = False,
                           plot_radius_sf       = False,
                         # Morphology stars/gas [ ]
                           plot_kappa_stars     = True,
                           plot_kappa_gas       = False,
                           plot_kappa_sf        = True,
                           plot_kappa_nsf       = False,
                           plot_ellip           = True,
                           plot_triax           = False,
                         # Metallicity  [ Z ]
                           plot_Z_stars         = False,
                           plot_Z_gas           = False,
                           plot_Z_inflow        = False,
                           plot_Z_outflow       = False,
                           plot_Z_insitu        = False,
                         # Eddington    [ ]
                           plot_edd             = False,
                         # Luminosity   [ erg/s ]
                           plot_lbol            = False,
                         # Disc velocity    [ km/s ]
                           plot_vcirc           = False,
                         # Torquing time    [ Gyr ]
                           plot_tdyn            = True,
                           plot_ttorque         = True,     
                         #==================================================================================
                         showfig        = True,
                         savefig        = False,
                           file_format  = 'pdf',
                           savefig_txt  = '', 
                         #--------------------------
                         print_progress = False,
                         debug = False):
    
    
    #================================================ 
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    
    #---------------------------
    # Loading files
    if print_progress:
        print('Loading files')
        time_start = time.time()
    
    # Loading mergertree file to establish windows
    f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
    Redshift_tree     = np.array(f['Snapnum_Index']['Redshift'])
    Lookbacktime_tree = np.array(f['Snapnum_Index']['LookbackTime'])
    f.close()
    
    # Load galaxy_tree
    dict_tree = json.load(open('%s/%s.csv' %(output_dir, csv_tree), 'r'))
    galaxy_tree     = dict_tree['galaxy_tree']
    tree_input      = dict_tree['tree_input']
    output_input    = dict_tree['output_input']
    sample_input    = dict_tree['sample_input']
    
    
    #---------------------------
    # Test for required particles
    particle_selection = []         #particle_list_in = []
    compound_selection = []         #angle_selection  = []
    if 'stars_gas' == use_angle:
        if 'stars' not in particle_selection:
            particle_selection.append('stars')
        if 'gas' not in particle_selection:
            particle_selection.append('gas')
        compound_selection.append(['stars', 'gas'])
    if 'stars_gas_sf' == use_angle:
        if 'stars' not in particle_selection:
            particle_selection.append('stars')
        if 'gas_sf' not in particle_selection:
            particle_selection.append('gas_sf')
        compound_selection.append(['stars', 'gas_sf'])
    if 'stars_gas_nsf' == use_angle:
        if 'stars' not in particle_selection:
            particle_selection.append('stars')
        if 'gas_nsf' not in particle_selection:
            particle_selection.append('gas_nsf')
        compound_selection.append(['stars', 'gas_nsf'])
    if 'gas_sf_gas_nsf' == use_angle:
        if 'gas_sf' not in particle_selection:
            particle_selection.append('gas_sf')
        if 'gas_nsf' not in particle_selection:
            particle_selection.append('gas_nsf')
        compound_selection.append(['gas_sf', 'gas_nsf'])
    if 'stars_dm' == use_angle:
        if 'stars' not in particle_selection:
            particle_selection.append('stars')
        if 'dm' not in particle_selection:
            particle_selection.append('dm')
        compound_selection.append(['stars', 'dm'])
    if 'gas_dm' == use_angle:
        if 'gas' not in particle_selection:
            particle_selection.append('gas')
        if 'dm' not in particle_selection:
            particle_selection.append('dm')
        compound_selection.append(['gas', 'dm'])
    if 'gas_sf_dm' == use_angle:
        if 'gas_sf' not in particle_selection:
            particle_selection.append('gas_sf')
        if 'dm' not in particle_selection:
            particle_selection.append('dm')
        compound_selection.append(['gas_sf', 'dm'])
    if 'gas_nsf_dm' == use_angle:
        if 'gas_nsf' not in particle_selection:
            particle_selection.append('gas_nsf')
        if 'dm' not in particle_selection:
            particle_selection.append('dm')
        compound_selection.append(['gas_nsf', 'dm'])
    
    
    #---------------------------
    # Find GalaxyID in tree and only process this
    
    # Load merger tree 
    f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
    
    GalaxyID_list_extract = []
    for GalaxyID_find in GalaxyID_list:
        # Find row
        row_mask, _ = np.where(np.array(f['Histories']['GalaxyID']) == GalaxyID_find)
        row_mask = row_mask[0]
        
        for ID_i in np.array(f['Histories']['GalaxyID'])[row_mask]:
            if str(ID_i) in galaxy_tree.keys():
                GalaxyID_list_extract.append(ID_i)
                print('ID %s found in galaxy_tree' %ID_i)
    f.close()
    
    
    #================================================ 
    # Loop over all galaxies
    for GalaxyID in tqdm(galaxy_tree.keys()):
        
        # If we are looking at individual galaxies, filter them out
        if GalaxyID_list != None:
            if int(GalaxyID) not in GalaxyID_list_extract:
                continue
    
        #------------------------
        # Default graphs: [ angles, massrate, mass, ssfr, radius, kappa, Z, edd, lbol ]
        plot_height_ratios = []
        plot_names         = []
        if plot_angles or plot_inclination:
            plot_height_ratios.append(3)
            plot_names.append('angles')
        if plot_tdyn or plot_ttorque:
            plot_height_ratios.append(2)
            plot_names.append('time')
        if plot_inflow or plot_outflow or plot_stelmassloss or plot_bh_acc or plot_bh_acc_instant or plot_sfr:
            plot_height_ratios.append(2)
            plot_names.append('massrate')
        if plot_halomass or plot_stelmass or plot_gasmass or plot_sfmass or plot_nsfmass or plot_bhmass:
            plot_height_ratios.append(2)
            plot_names.append('mass')
        if plot_ssfr:
            plot_height_ratios.append(2)
            plot_names.append('ssfr')
        if plot_l:
            plot_height_ratios.append(2)
            plot_names.append('l')
        if plot_radius or plot_radius_sf:
            plot_height_ratios.append(2)
            plot_names.append('radius')
        if plot_kappa_stars or plot_kappa_gas or plot_kappa_sf or plot_kappa_nsf or plot_ellip or plot_triax:
            plot_height_ratios.append(2)
            plot_names.append('morphology')
        if plot_Z_stars or plot_Z_gas or plot_Z_inflow or plot_Z_outflow or plot_Z_insitu:
            plot_height_ratios.append(2)
            plot_names.append('Z')
        if plot_edd:
            plot_height_ratios.append(2)
            plot_names.append('edd')
        if plot_lbol:
            plot_height_ratios.append(2)
            plot_names.append('lbol')
        if plot_vcirc:
            plot_height_ratios.append(2)
            plot_names.append('vcirc')
        
        #------------------------
        # Graph initialising and base formatting
        #fig, axs = plt.subplots(nrows=len(plot_height_ratios), ncols=1, gridspec_kw={'height_ratios': plot_height_ratios}, figsize=[5.5, 1*np.sum(np.array(plot_height_ratios))], sharex=True, sharey=False)
        fig, axs = plt.subplots(nrows=len(plot_height_ratios), ncols=1, gridspec_kw={'height_ratios': plot_height_ratios}, figsize=[10/3, 0.5*np.sum(np.array(plot_height_ratios))], sharex=True, sharey=False)
        
        
        #------------------------
        # Create each graph separately
        # Create redshift axis:
        redshiftticks = [0, 0.1, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20]
        ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
        for i, plot_names_i in enumerate(plot_names):
            
            ### Formatting
            ax_top = axs[i].twiny()
            if redshift_axis:
                ax_top.set_xticks(ageticks)
            
            axs[i].set_xlim(8, 0)
            ax_top.set_xlim(8, 0)
            
            # Plot 0 - use_hmr_angle
            if plot_names_i == 'angles':
                # Plot data in galaxy_tree():
                if plot_angles:
                    
                    if use_angle == 'stars_gas':
                        angle_label = 'stars$-$gas'
                    elif use_angle == 'stars_gas_sf':
                        angle_label = 'stars$-$gas$_{\mathrm{SF}}$'
                    elif use_angle == 'gas_sf_gas_nsf':
                        angle_label = 'gas$_{\mathrm{SF}}$$-$gas$_{\mathrm{NSF}}$'
                    elif use_angle == 'stars_dm':
                        angle_label = 'DM$-$stars'
                    elif use_angle == 'gas_dm':
                        angle_label = 'DM$-$gas'
                    elif use_angle == 'gas$_{\mathrm{SF}}$_dm':
                        angle_label = 'DM$-$gas$_{\mathrm{SF}}$'
                    
                    if abs_or_proj == 'abs':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj], alpha=1.0, ms=2, ls='-', lw=0.5, zorder=10, label='%s'%angle_label, color='k')
                        if use_uncertainties:
                            axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,0], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,1], alpha=0.25, lw=0, zorder=5, color='k')
                        
                        if plot_dm_angles:
                            axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj], alpha=1.0, ms=2, ls='dashdot', lw=0.5, zorder=10, label='DM$-$stars', color='r')
                            axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj], alpha=1.0, ms=2, ls=':', lw=0.5, zorder=10, label='DM$-$gas$_{\mathrm{SF}}$', color='b')
                            if use_dm_uncertainties:
                                axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,0], np.array(galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,1], alpha=0.25, lw=0, zorder=5, facecolor='r')
                                axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,0], np.array(galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,1], alpha=0.25, lw=0, zorder=5, facecolor='b')
                    if abs_or_proj == 'proj':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj], alpha=1.0, ms=2, ls=':', lw=0.5, zorder=10, label='%s (projected)'%angle_label)
                        if use_uncertainties:
                            axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,0], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,1], alpha=0.25, lw=0, zorder=5)
                        
                        if plot_dm_angles:
                            axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj], alpha=1.0, ms=2, ls='dashdot', lw=0.5, zorder=10, label='DM$-$stars', color='r')
                            axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj], alpha=1.0, ms=2, ls=':', lw=0.5, zorder=10, label='DM$-$gas$_{\mathrm{SF}}$', color='b')
                            if use_dm_uncertainties:
                                axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,0], np.array(galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,1], alpha=0.25, lw=0, zorder=5, facecolor='r')
                                axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,0], np.array(galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[:,1], alpha=0.25, lw=0, zorder=5, facecolor='b')
                    if abs_or_proj == 'both':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['angle_abs'], alpha=1.0, ms=2, ls='-', lw=0.5, zorder=10, label='%s'%angle_label)
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['angle_proj'], alpha=1.0, ms=2, ls=':', lw=0.5, zorder=10, label='%s (projected)'%angle_label)
                        if use_uncertainties:
                            axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_abs'])[:,0], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_abs'])[:,1], alpha=0.25, lw=0, zorder=5)
                            axs[0].fill_between(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_proj'])[:,0], np.array(galaxy_tree['%s' %GalaxyID][use_angle]['%s_hmr' %use_hmr_angle]['err_proj'])[:,1], alpha=0.25, lw=0, zorder=5)     
                                     
                if plot_inclination:
                    for particle_i in particle_selection:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['%s' %particle_i]['%s_hmr' %use_hmr_angle]['proj_angle'], alpha=1.0, lw=0.8, ls=':', label='%s' %particle_i)
                
                #---------------------
                ### Formatting
                axs[i].set_ylim(0, 180)
                if redshift_axis:
                    ax_top.set_xlabel('Redshift')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')

                axs[i].set_yticks(np.arange(0, 181, 30))
                if redshift_axis:
                    ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
                if not redshift_axis:
                    ax_top.set_xticklabels([])
                axs[i].set_ylabel('Misalignment angle, $\psi$')
                
                #---------------------
                ### Annotation
                axs[i].set_title('GalaxyID ($z=0$): %s' %str(galaxy_tree['%s' %GalaxyID]['GalaxyID'][-1]), size=7, loc='left', pad=3)
                #axs[i].text(8, 185, 'GalaxyID: %s' %str(GalaxyID), fontsize=8)
                axs[i].axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
                axs[i].axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
                #axs[0].grid(lw=0.3)
                #axs[0].grid(lw=0.3)
                
                #---------------------
                ### Legend
                axs[0].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1)
                
                #---------------------
                ### Plot mergers
                metadata_ratios = []
                metadata_gas_ratios = []
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
                            metadata_ratios.append(max(ratio_i))
                            metadata_gas_ratios.append(gas_i[np.argmax(ratio_i)])
                            #axs[i].text(time_i+0.1, 163, '%.2f' %max(ratio_i), color='grey', fontsize=7, zorder=999)
                            #axs[i].text(time_i+0.2, 151, '%.2f' %gas_i[np.argmax(ratio_i)], color='blue', fontsize=7, zorder=999)
            
            # Plot 11
            if plot_names_i == 'time':
                # Plot data in galaxy_tree():
                if plot_tdyn:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_angle]['tdyn'], alpha=1.0, lw=0.5, c='k', label='$t_{\mathrm{dyn}}$')
                if plot_ttorque:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_angle]['ttorque'], alpha=1.0, lw=0.5, c='g', label='$t_{\mathrm{torque}}$')
                
                #---------------------
                ### Formatting
                axs[i].set_ylabel('Time (Gyr)')
                axs[i].set_ylim(bottom=0)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
             
            # Plot 1 - use_hmr_angle
            if plot_names_i == 'massrate':
                # Plot data in galaxy_tree():
                if plot_inflow:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['inflow_rate'], alpha=1.0, lw=0.5, c='g', ls='-', label='inflow')
                if plot_outflow:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['outflow_rate'], alpha=1.0, lw=0.5, c='r', ls='--', label='outflow')
                if plot_stelmassloss:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['stelmassloss_rate'], alpha=1.0, lw=0.8, ls='dashdot', label='stellar mass loss')
                if plot_bh_acc:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], 1000*np.array(galaxy_tree['%s' %GalaxyID]['bh']['mdot']), alpha=1.0, lw=0.8, c='purple', ls=':', label='$\dot{M}_{\mathrm{BH}}$(×$10^3$)')
                if plot_bh_acc_instant:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], 1000*np.array(galaxy_tree['%s' %GalaxyID]['bh']['mdot_instant']), alpha=1.0, lw=0.8, c='k', ls=':', label='$\dot{M}_{\mathrm{BH}}$(×$10^3$) (inst)')
                if plot_sfr:
                    # Plot data in galaxy_tree():
                    if use_hmr_general == 'aperture:':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['ap_sfr'])), alpha=1.0, lw=0.5, c='orange', ls='dashdot', label='SFR')
                    else:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(10*np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['sfr'])), alpha=1.0, lw=0.5, c='orange', ls='dashdot', label='SFR(×$10$)')
                
                #---------------------    
                ### Formatting
                axs[i].set_ylabel('$\dot{M}$ $(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
                #axs[i].set_ylim(0, 30)
                axs[i].set_ylim(bottom=0)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
                            
            # Plot 2 - use_hmr_general 
            if plot_names_i == 'mass':
                # Plot data in galaxy_tree():
                if plot_halomass:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['halomass'])), alpha=1.0, lw=0.5, c='brown', ls='-', label='halo')
                if plot_stelmass:
                    if use_hmr_general == 'aperture':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['stars']['ap_mass'])), alpha=1.0, lw=0.5, c='r', ls='-', label='star')
                    else:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass'])), alpha=1.0, lw=0.5, c='r', ls='-', label='star')
                if plot_gasmass:
                    if use_hmr_general == 'aperture':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['gas']['ap_mass'])), alpha=1.0, lw=0.5, c='g', ls='--', label='gas')
                    else:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['mass'])), alpha=1.0, lw=0.5, c='g', ls='--', label='gas') 
                if plot_sfmass:
                    if use_hmr_general == 'aperture':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['gas_sf']['ap_mass'])), alpha=1.0, lw=0.5, c='b', ls='dashdot', label='gas$_{\mathrm{SF}}$')
                    else:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['gas_sf']['%s_hmr' %use_hmr_general]['mass'])), alpha=1.0, lw=0.5, c='b', ls=':', label='gas$_{\mathrm{SF}}$')
                if plot_nsfmass:
                    if use_hmr_general == 'aperture':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['gas_nsf']['ap_mass'])), alpha=1.0, lw=0.5, c='b', ls=':', label='gas$_{\mathrm{NSF}}$')
                    else:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['gas_nsf']['%s_hmr' %use_hmr_general]['mass'])), alpha=1.0, lw=0.5, c='b', ls=':', label='gas$_{\mathrm{NSF}}$')
                if plot_bhmass:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['bh']['mass'])), alpha=1.0, lw=0.5, c='purple', ls='dashdot', label='BH')

                
                #---------------------
                ### Formatting
                #axs[i].set_ylabel('$\mathrm{log_{10}}(\mathrm{M}/\mathrm{M}_{\odot})$')
                axs[i].set_ylabel('$\mathrm{log_{10}}M$ $(\mathrm{M}_{\odot})$')
                axs[i].set_ylim(bottom=8.5, top=11.5)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
                            
            # Plot 3 - use_hmr_general 
            if plot_names_i == 'ssfr':
                # Plot data in galaxy_tree():
                if use_hmr_general == 'aperture:':
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.divide(np.array(galaxy_tree['%s' %GalaxyID]['ap_sfr']), np.array(galaxy_tree['%s' %GalaxyID]['stars']['ap_mass']))), alpha=1.0, lw=0.5, c='orange')
                else:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['sfr']), np.array(galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass']))), alpha=1.0, lw=0.5, c='orange')
                    
                #---------------------
                ### Formatting
                axs[i].set_ylabel('$\mathrm{sSFR}$ $(\mathrm{yr}^{-1})$')
                axs[i].set_ylim(-12, -7)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
            
            # Plot 4 - use_hmr_angle
            if plot_names_i == 'l':
                # Plot data in galaxy_tree():
                axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_angle]['l'])), alpha=1.0, lw=0.5, c='r', ls='-', label='stars')
                
                #---------------------
                ### Formatting
                axs[i].set_ylabel('$\mathrm{log_{10}}j_{\mathrm{stars}}$ $(\mathrm{pkpc} / \mathrm{km s}^{-1})$')
                axs[i].set_ylim(0, 4)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
                            
            # Plot 5
            if plot_names_i == 'radius':
                # Plot data in galaxy_tree():
                if plot_radius:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['rad'], alpha=1.0, lw=0.5, c='k', label='$r_{\mathrm{1/2}}$')
                if plot_radius_sf:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['rad_sf'], alpha=1.0, lw=0.5, c='b', label='$r_{\mathrm{1/2}^{\mathrm{SF}}}$')
                    
                
                #---------------------
                ### Formatting
                axs[i].set_ylabel('Radius (pkpc)')
                axs[i].set_ylim(bottom=0)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
                            
            # Plot 6 - aperture
            if plot_names_i == 'morphology':
                # Plot data in galaxy_tree():
                if plot_kappa_stars:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['stars']['kappa'], alpha=1.0, lw=0.5, c='r', ls='-', label='$\kappa_{\mathrm{co}}^{\mathrm{*}}$')
                if plot_kappa_gas:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['kappa'], alpha=1.0, lw=0.5, c='g', ls='-', label='$\kappa_{\mathrm{co}}^{\mathrm{gas}}$')
                if plot_kappa_sf:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas_sf']['kappa'], alpha=1.0, lw=0.5, c='b', ls='dashdot', label='$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
                if plot_kappa_nsf:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas_nsf']['kappa'], alpha=1.0, lw=0.5, c='b', ls='-', label='$\kappa_{\mathrm{co}}^{\mathrm{NSF}}$')
                if plot_ellip:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['ellip'], alpha=1.0, lw=0.5, c='purple', ls='--', label='$\epsilon$')
                if plot_triax:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['triax'], alpha=1.0, lw=0.5, c='orange', ls=':', label='$T$')
                    
                
                
                
                
                
                ##
                #print('remove ####')
                #print('mass   rad   radsf.  ellip.  vcirc.  tdyn.  ttorque')
                #for mass_ii, r_ii, rsf_ii, ellip_ii, vcirc_ii, tdyn_ii, ttorque_ii in zip(galaxy_tree['%s' %GalaxyID]['other']['1.0_hmr']['tot_mass'], galaxy_tree['%s' %GalaxyID]['rad'], galaxy_tree['%s' %GalaxyID]['rad_sf'], galaxy_tree['%s' %GalaxyID]['ellip'], galaxy_tree['%s' %GalaxyID]['other']['1.0_hmr']['vcirc'], galaxy_tree['%s' %GalaxyID]['other']['1.0_hmr']['tdyn'], galaxy_tree['%s' %GalaxyID]['other']['1.0_hmr']['ttorque']):
                #    print('%.2e | %.2f | %.2f | %.2f |±| %.1f |±| %.3f | %.3f' %(mass_ii, r_ii, rsf_ii, ellip_ii, vcirc_ii, tdyn_ii, ttorque_ii))
                
                
                
                
                
                
                
                
                
                #---------------------
                ### Annotate
                axs[i].axhline(0.4, lw=0.5, ls=':', c='k', alpha=0.9, zorder=1)
                axs[i].text(7.95, 0.42, 'LTG', color='grey', fontsize=7, alpha=0.9)
                axs[i].text(7.95, 0.30, 'ETG', color='grey', fontsize=7, alpha=0.9)
                
                #---------------------
                ### Formatting
                axs[i].set_ylim(0, 1)
                axs[i].set_yticks(np.arange(0, 1.1, 0.2))
                if not plot_ellip or not plot_triax:
                    axs[i].set_ylabel('$\kappa_{\mathrm{co}}$')
                else:
                    axs[i].set_ylabel('Morphology')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
                            
            # Plot 7 - use_hmr_angle & use_hmr_general
            if plot_names_i == 'Z':
                # Plot data in galaxy_tree():
                if plot_Z_stars:
                    if use_hmr_general == 'aperture':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['stars']['ap_Z'], alpha=1.0, lw=1, c='r', label='$Z_{\mathrm{*}}$')
                    else:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['Z'], alpha=1.0, lw=1, c='r', label='$Z_{\mathrm{*}}$')
                if plot_Z_gas:
                    if use_hmr_general == 'aperture':
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['ap_Z'], alpha=1.0, lw=1, c='g', label='$Z_{\mathrm{gas}}$')
                    else:
                        axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['Z'], alpha=1.0, lw=1, c='g', label='$Z_{\mathrm{gas}}$')
                if plot_Z_inflow:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['inflow_Z'], alpha=1.0, lw=1, c='r', ls='dashdot', label='$Z_{\mathrm{inflow}}$')
                if plot_Z_outflow:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['outflow_Z'], alpha=1.0, lw=1, c='g', ls='dashdot', label='$Z_{\mathrm{outflow}}$')
                if plot_Z_insitu:
                    axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['insitu_Z'], alpha=1.0, lw=1, c='b', ls='dashdot', label='$Z_{\mathrm{insitu}}$')
                
                #---------------------
                ### Formatting
                axs[i].set_ylim(0, 0.1)
                axs[i].set_yticks(np.arange(0, 0.1, 0.025))
                axs[i].set_ylabel('$Z$')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
            
            # Plot 8
            if plot_names_i == 'edd':
                # Plot data in galaxy_tree():
                axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['bh']['edd'])), alpha=1.0, lw=1, c='purple')
                
                #---------------------
                ### Formatting
                axs[i].set_ylim(-6, 0)
                axs[i].set_ylabel('$\mathrm{log_{10}}$ $\lambda_{\mathrm{edd}}$')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                #axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
            
            # Plot 9
            if plot_names_i == 'lbol':
                # Plot data in galaxy_tree():
                axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.log10(np.array(galaxy_tree['%s' %GalaxyID]['bh']['lbol'])), alpha=1.0, lw=1, c='purple')
                
                #---------------------
                ### Formatting
                #axs[i].set_ylim(-6, 0)
                axs[i].set_ylabel('$\mathrm{log_{10}}$ $L_{\mathrm{BH,bol}}$ (erg s$^{-1}$)')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                #axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
            
            # Plot 10
            if plot_names_i == 'vcirc':
                # Plot data in galaxy_tree():
                axs[i].plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_angle]['vcirc'], alpha=1.0, lw=1, c='k')
                
                #---------------------
                ### Formatting
                axs[i].set_ylabel('Disc edge velocity (km s$^{-1}$)')
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', labelbottom=False, labeltop=False)
                ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor')
                
                #---------------------
                ### Legend
                #axs[i].legend(loc='best', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1.2)
                
                #---------------------
                ### Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > min_merger_ratio:
                            #axs[i].axvline(time_i, c='grey', ls='-', lw=3, alpha=0.5, zorder=1)
                            axs[i].axvline(time_i, c='grey', ls='--', lw=0.6, alpha=0.9, zorder=1)
            
            
            
            #---------------------
            axs[i].minorticks_on()
            if redshift_axis:
                axs[i].tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='major')
                axs[i].tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='minor')
            if not redshift_axis:
                axs[i].tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
                axs[i].tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
            if i == len(plot_names)-1:
                axs[i].set_xlabel('Lookback time (Gyr)')
            
            
        #------------------------
        # Other
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        
        
        #=====================================
        ### Print summary
        metadata_plot = {'Title': 'ID: %s\nM*: %.2e\nKappa: %.2f\nM_gas: %.2e\nM_sf: %.2e\nHalo: %.2e\nHMR: %.2f\nHMR_p: %.2f\nMergers: %s\nGasratio: %s' %(GalaxyID, galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass'][-1], galaxy_tree['%s' %GalaxyID]['stars']['kappa'][-1], galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['mass'][-1], galaxy_tree['%s' %GalaxyID]['gas_sf']['%s_hmr' %use_hmr_general]['mass'][-1], galaxy_tree['%s' %GalaxyID]['halomass'][-1], galaxy_tree['%s' %GalaxyID]['rad'][-1], galaxy_tree['%s' %GalaxyID]['radproj'][-1], metadata_ratios, metadata_gas_ratios),
                         'Author': 'abs_or_proj: %s\nuse_hmr_general: %s\nuse_hmr_angle: %s\nmisangle_threshold: %s\nmin_merger_ratio: %s\n' %(abs_or_proj, use_hmr_general, use_hmr_angle, misangle_threshold, min_merger_ratio)}
        
        if savefig:
            plt.savefig("%s/individual_evolution/ID%s_evolution_proj%s_%s_%s.%s" %(fig_dir, str(galaxy_tree['%s' %GalaxyID]['GalaxyID'][-1]), abs_or_proj, use_angle, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/individual_evolution/ID%s_evolution_proj%s_%s_%s.%s" %(fig_dir, str(galaxy_tree['%s' %GalaxyID]['GalaxyID'][-1]), abs_or_proj, use_angle, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        



         
     
#============================
#_analysis_evolution()

#_plot_evolution()

_plot_evolution_snip()
#============================

#for ID_i in [108988077, 479647060, 21721896, 390595970, 401467650, 182125463, 192213531, 24276812, 116404995, 239808134, 215988755, 86715463, 6972011, 475772617, 374037507, 429352532, 441434976]:
#    _plot_evolution_old(csv_output = 'L100_evolution_ID' + str(ID_i) + '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_')
#============================



