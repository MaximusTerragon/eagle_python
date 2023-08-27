import h5py
import numpy as np
import math
import random
import uuid
import hashlib
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
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local\n")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================


#--------------------------------
""" GUIDE TO galaxy_tree['%s' %GalaxyID] VALUES:
We will want to ensure that for whatever we are 
extracting (eg. stars_gas_sf):
    'stars':        entire array ['2.0_hmr']['count'] > 20
    'gas_sf':       entire array ['2.0_hmr']['count'] > 20
    'stars_gas_sf': entire array ['2.0_hmr']['com_proj'] < 2.0

We can then additionall impose limits, ei.:
       -            entire array is >10^14
    'stars':        entire array ['kappa'] > 0.4
    'gas_sf':       entire array ['2.0_hmr']['count'] > 50
    'gas_sf':       entire array ['2.0_hmr']['proj_angle'] > 10, < 170

['2.0_hmr'] will always be there but may have math.nan, so will need
to ensure entire array np.isnan = False

#-------------------------------------
GalaxyID                -
DescendantID            -
GroupNum                -
SubGroupNum             -
SnapNum                 -
Redshift                -
Lookbacktime            -
halomass                - total per subfind
rad                     - [pkpc]
rad_proj                - [pkpc]
merger_ID               - merger stuff
merger_ratio_stars      - peak stellar ratios in past 2 Gyr ish
merger_ratio_gas        - gas ratios at time of peak stellar ratios
gasdata_old             - gets updated and replaced with math.nan

'stars'
    tot_mass            - total subfind mass
    ap_mass             - total aperture mass
    kappa               - kappa
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        Z               - metallicity in radius
        proj_angle      - angle to viewing axis

'gas'
    tot_mass            - total subfind mass
    ap_mass             - total aperture mass
    kappa               - kappa
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        Z               - metallicity in radius
        sfr             - sfr in radius
        proj_angle      - angle to viewing axis
        inflow_rate     - inflow rate at radius [Msun/yr]
        inflow_Z        - inflow metallicity (mass-weighted, but Z is fine)
        outflow_rate    - outflow rate at radius [Msun/yr]
        outflow_Z       - outflow metallicity (mass-weighted, but Z is fine)
        stelmassloss    - stellar mass loss rate [Msun/yr]
        insitu_Z        - metallicity of material that remained (mass-weighted, but Z is fine)

'gas_sf'
    ap_mass             - total aperture mass
    kappa               - kappa
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        sfr             - sfr in radius
        Z               - metallicity in radius
        proj_angle      - angle to viewing axis
        inflow_rate     - inflow rate at radius [Msun/yr]
        inflow_Z        - inflow metallicity (mass-weighted, but Z is fine)
        outflow_rate    - outflow rate at radius [Msun/yr]
        outflow_Z       - outflow metallicity (mass-weighted, but Z is fine)
        stelmassloss    - stellar mass loss rate [Msun/yr]
        insitu_Z        - metallicity of material that remained (mass-weighted, but Z is fine)

'gas_nsf'
    ap_mass             - total aperture mass
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        Z               - metallicity in radius
        proj_angle      - angle to viewing axis
        inflow_rate     - inflow rate at radius [Msun/yr]
        inflow_Z        - inflow metallicity (mass-weighted, but Z is fine)
        outflow_rate    - outflow rate at radius [Msun/yr]
        outflow_Z       - outflow metallicity (mass-weighted, but Z is fine)
        stelmassloss    - stellar mass loss rate [Msun/yr]
        insitu_Z        - metallicity of material that remained (mass-weighted, but Z is fine)

'dm'
    ap_mass             - total aperture mass
    count               - total count in aperture
    proj_angle          - angle to viewing axis

'bh'
    mass                - bh mass of central BH (not particle mass)
    mdot_instant        - mdot of that particle
    mdot                - mdot averaged over snipshot time difference
    edd                 - instantaneous eddington from mdot_instant
    count               - len(mass)

'stars_gas'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'stars_gas_sf'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'stars_gas_nsf'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'gas_sf_gas_nsf'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'stars_dm'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for viewing axis
            
"""
# Goes through all csv samples given and creates giant merger tree, no criteria used
# SAVED: /outputs_snips/%sgalaxy_tree_
def _create_galaxy_tree(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                        csv_sample2 = '_all_sample_misalignment_10.0',
                        csv_sample_range = np.arange(147, 201, 1),   # snapnums
                        csv_output_in = '_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',
                        #--------------------------
                        # Galaxy analysis
                        print_summary  = True,
                        #--------------------------
                        csv_file       = True,             # Will write sample to csv file in sample_dir
                          csv_name     = '',               # extra stuff at end
                        #--------------------------
                        print_progress = False,
                        debug = False):
                        
    #================================================   
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    
    #---------------------- 
    # Loading mergertree file
    if print_progress:
        print('Cycling through CSV files and extracting galaxies')
        time_start = time.time()
    
    f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
    GalaxyID_tree             = np.array(f['Histories']['GalaxyID'])
    DescendantID_tree         = np.array(f['Histories']['DescendantID'])
    Redshift_tree             = np.array(f['Snapnum_Index']['Redshift'])
    Lookbacktime_tree         = np.array(f['Snapnum_Index']['LookbackTime'])
    StellarMass_tree          = np.array(f['Histories']['StellarMass'])
    GasMass_tree              = np.array(f['Histories']['GasMass'])
    StarFormationRate_30_tree = np.array(f['Histories']['StarFormationRate_30'])
    f.close()
    
    
    #================================================ 
    # Creating dictionary to collect all galaxies that meet criteria
    galaxy_tree        = {}
    
    
    #----------------------
    # Cycling over all the csv samples we want
    csv_sample_range = [int(i) for i in csv_sample_range]
    for csv_sample_range_i in csv_sample_range:
        
        # Ensuring the sample and output originated together
        csv_sample = csv_sample1 + str(csv_sample_range_i) + csv_sample2
        csv_output = csv_sample + csv_output_in
        
        
        #================================================  
        # Load sample csv
        if print_progress:
            print('Loading initial sample')
            time_start = time.time()
    
        #-------------------------
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
        all_sfr             = dict_output['all_sfr']
        all_Z               = dict_output['all_Z']
        all_misangles       = dict_output['all_misangles']
        all_misanglesproj   = dict_output['all_misanglesproj']
        all_gasdata         = dict_output['all_gasdata']
        all_flags           = dict_output['all_flags']
        
        # Loading sample criteria
        sample_input        = dict_sample['sample_input']
        output_input        = dict_output['output_input']
        
        
        #-------------------------
        # Find angle galaxy makes with viewing axis
        def _find_angle(vector1, vector2):
            return np.rad2deg(np.arccos(np.clip(np.dot(vector1/np.linalg.norm(vector1), vector2/np.linalg.norm(vector2)), -1.0, 1.0)))     # [deg]
    
        if output_input['viewing_axis'] == 'x':
            viewing_vector = [1., 0, 0]
        elif output_input['viewing_axis'] == 'y':
            viewing_vector = [0, 1., 0]
        elif output_input['viewing_axis'] == 'z':
            viewing_vector = [0, 0, 1.]
    
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
        
        #-------------------------
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
        
        
        #================================================
        # Looping over all GalaxyIDs
        for GalaxyID, SnapNum in tqdm(zip(GalaxyID_List, SnapNum_List), total=len(GalaxyID_List)):
            
            # use mergertree to find exact tree in mergertree
            row_mask = np.where(GalaxyID_tree == GalaxyID)[0][0]
            
            # Find DescendantID
            DescendantID = DescendantID_tree[row_mask][SnapNum]
            Lookbacktime = Lookbacktime_tree[SnapNum]
            Redshift     = Redshift_tree[SnapNum]
            
            # Mask mainbranch of this galaxy
            mainbranch_GalaxyIDs     = GalaxyID_tree[row_mask][csv_sample_range[0]:int(csv_sample_range[-1]+1)]
            mainbranch_DescendantIDs = DescendantID_tree[row_mask][csv_sample_range[0]:int(csv_sample_range[-1]+1)]
            
            # If lowest ID in range already in dict, 
            new_entry = False
            ID_dict   = [i for i in mainbranch_GalaxyIDs if str(i) in galaxy_tree]
            if len(ID_dict) == 1:
                ID_dict = ID_dict[0]
                #new_entry = False
            elif len(ID_dict) > 1:
                raise Exception('Multiple mainbranch IDs already added')
            else:
                ID_dict   = GalaxyID
                new_entry = True
                
           
            #--------------------------------------
            # Find mergers from previous snap to now. Uses only merger tree data, so no gassf 
            
            #Ignoring row_mask, look for duplicate DescendantIDs with same GalaxyID as current galaxy
            merger_mask = [i for i in np.where(np.array(DescendantID_tree)[:,int(SnapNum-1)] == GalaxyID)[0] if i != row_mask]
            if debug:
                print(merger_mask)          # all galaxies that will merge in next snap

            # find peak stelmass of those galaxies
            merger_ID_array    = []
            merger_ratio_array = []
            merger_gas_array   = []
            if len(merger_mask) > 0:
                for mask_i in merger_mask:
                    # Find largest stellar mass of this satellite.
                    mass_mask = np.argmax(StellarMass_tree[mask_i][int(SnapNum-10):int(SnapNum)]) + (SnapNum-10)
                    
                    # Extract secondary properties
                    primary_stelmass   = StellarMass_tree[row_mask][mass_mask]
                    primary_gasmass    = GasMass_tree[row_mask][mass_mask]
                    component_stelmass = StellarMass_tree[mask_i][mass_mask]
                    component_gasmass  = GasMass_tree[mask_i][mass_mask]
                    
                    if primary_stelmass <= 0.0:
                        # Adjust stelmass
                        primary_stelmass   = math.nan
                        primary_gasmass    = math.nan
                        
        
                    # Find ratios
                    merger_ratio = component_stelmass / primary_stelmass 
                    gas_ratio    = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)
                    if debug:
                        print('component stats')
                        print(GalaxyID_tree[mask_i][int(SnapNum-1)])
                        print(primary_stelmass)
                        print(component_stelmass)
                        print(primary_gasmass)
                        print(component_gasmass)
                        print(merger_ratio)
                        print(gas_ratio)

                    # Append
                    merger_ID_array.append(GalaxyID_tree[mask_i][int(SnapNum-1)])
                    merger_ratio_array.append(merger_ratio)
                    merger_gas_array.append(gas_ratio)

            
            #=============================================
            # If first snap, always start new tree
            if new_entry:
                # Create entry
                galaxy_tree['%s' %ID_dict]  = {'GalaxyID': [all_general['%s' %GalaxyID]['GalaxyID']],
                                               'DescendantID': [DescendantID],
                                               'GroupNum': [all_general['%s' %GalaxyID]['GroupNum']],
                                               'SubGroupNum': [all_general['%s' %GalaxyID]['SubGroupNum']],
                                               'SnapNum': [all_general['%s' %GalaxyID]['SnapNum']],
                                               'Redshift': [Redshift],
                                               'Lookbacktime': [Lookbacktime],
                                               # total mass of halo from mergertree
                                               'halomass': [all_general['%s' %GalaxyID]['halo_mass']],
                                               # radii
                                               'rad': [all_general['%s' %GalaxyID]['halfmass_rad']],
                                               'radproj': [all_general['%s' %GalaxyID]['halfmass_rad_proj']],
                                               'ap_sfr': [StarFormationRate_30_tree[row_mask][SnapNum]],
                                               # merger analysis
                                               'merger_ID': [merger_ID_array],
                                               'merger_ratio_stars': [merger_ratio_array],
                                               'merger_ratio_gas': [merger_gas_array],
                                               # gasdata
                                               'gasdata_old': all_gasdata['%s' %GalaxyID],             # update going forward
                                               # specific data
                                               'stars': {},
                                               'gas': {},
                                               'gas_sf': {},
                                               'gas_nsf': {},
                                               'dm': {},
                                               'bh': {},
                                               # angles and coms
                                               'stars_gas': {},
                                               'stars_gas_sf': {},
                                               'stars_gas_nsf': {},
                                               'gas_sf_gas_nsf': {},
                                               'stars_dm': {}} 
                
                if csv_sample_range_i == csv_sample_range[-1]:
                    galaxy_tree['%s' %ID_dict]['gasdata_old'] = math.nan
                
                #------------------                       
                # Create stars
                galaxy_tree['%s' %ID_dict]['stars'] = {'tot_mass': [StellarMass_tree[row_mask][SnapNum]],
                                                       'ap_mass': [all_general['%s' %GalaxyID]['stelmass']],
                                                       'kappa': [all_general['%s' %GalaxyID]['kappa_stars']]}
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['stars'].update({'%s_hmr' %hmr_i: {'mass': [math.nan],
                                                                                      'count': [math.nan],                     
                                                                                      'Z': [math.nan],  
                                                                                      'proj_angle': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['stars'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['stars'][mask_masses]],
                                                                                      'count': [all_counts['%s' %GalaxyID]['stars'][mask_counts]],                     
                                                                                      'Z': [all_Z['%s' %GalaxyID]['stars'][mask_Z]],  
                                                                                      'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['stars'][mask_spins], viewing_vector)]}})
                                                                                                                          
                #------------------                       
                # Create gas
                galaxy_tree['%s' %ID_dict]['gas']   = {'tot_mass': [GasMass_tree[row_mask][SnapNum]],
                                                       'ap_mass': [all_general['%s' %GalaxyID]['gasmass']],
                                                       'kappa': [all_general['%s' %GalaxyID]['kappa_gas']]}
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas'].update({'%s_hmr' %hmr_i: {'mass': [math.nan],
                                                                                    'count': [math.nan],    
                                                                                    'sfr': [math.nan],                      
                                                                                    'Z': [math.nan],  
                                                                                    'proj_angle': [math.nan],
                                                                                    # inflow/outflow and metallicity flow
                                                                                    'inflow_rate': [math.nan],
                                                                                    'inflow_Z': [math.nan],
                                                                                    'outflow_rate': [math.nan],
                                                                                    'outflow_Z': [math.nan],
                                                                                    'stelmassloss_rate': [math.nan],
                                                                                    'insitu_Z': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['gas'][mask_masses]],
                                                                                    'count': [all_counts['%s' %GalaxyID]['gas'][mask_counts]],                     
                                                                                    'Z': [all_Z['%s' %GalaxyID]['gas'][mask_Z]],  
                                                                                    'sfr': [all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr]], 
                                                                                    'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['gas'][mask_spins], viewing_vector)],
                                                                                    # inflow/outflow and metallicity flow
                                                                                    'inflow_rate': [math.nan],
                                                                                    'inflow_Z': [math.nan],
                                                                                    'outflow_rate': [math.nan],
                                                                                    'outflow_Z': [math.nan],
                                                                                    'stelmassloss_rate': [math.nan],
                                                                                    'insitu_Z': [math.nan]}})
                                                                          
                #------------------                       
                # Create gas_sf
                galaxy_tree['%s' %ID_dict]['gas_sf'] = {'ap_mass': [all_general['%s' %GalaxyID]['gasmass_sf']],
                                                        'kappa': [all_general['%s' %GalaxyID]['kappa_gas_sf']]}
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_sf'].update({'%s_hmr' %hmr_i: {'mass': [math.nan],
                                                                                       'count': [math.nan],   
                                                                                       'sfr': [math.nan],                  
                                                                                       'Z': [math.nan],  
                                                                                       'proj_angle': [math.nan],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_sf'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['gas_sf'][mask_masses]],
                                                                                       'count': [all_counts['%s' %GalaxyID]['gas_sf'][mask_counts]],                     
                                                                                       'Z': [all_Z['%s' %GalaxyID]['gas_sf'][mask_Z]],  
                                                                                       'sfr': [all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr]],                 
                                                                                       'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['gas_sf'][mask_spins], viewing_vector)],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                                                                          
                #------------------                       
                # Create gas_nsf
                galaxy_tree['%s' %ID_dict]['gas_nsf'] = {'ap_mass': [all_general['%s' %GalaxyID]['gasmass_nsf']]}
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf'].update({'%s_hmr' %hmr_i: {'mass': [math.nan],
                                                                                        'count': [math.nan],                 
                                                                                        'Z': [math.nan],  
                                                                                        'proj_angle': [math.nan],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['gas_nsf'][mask_masses]],
                                                                                        'count': [all_counts['%s' %GalaxyID]['gas_nsf'][mask_counts]],                     
                                                                                        'Z': [all_Z['%s' %GalaxyID]['gas_nsf'][mask_Z]],                  
                                                                                        'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['gas_nsf'][mask_spins], viewing_vector)],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                                                                          
                #------------------                       
                # Create dm
                galaxy_tree['%s' %ID_dict]['dm']   = {'ap_mass': [all_general['%s' %GalaxyID]['dmmass']],
                                                      'count': [all_counts['%s' %GalaxyID]['dm']],
                                                      'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['dm'][-1], viewing_vector)]}
                    
                #------------------                       
                # Create bh
                if np.isnan(all_general['%s' %GalaxyID]['bh_mass']) == True:
                    count_bh = 0
                else:
                    count_bh = 1
                galaxy_tree['%s' %ID_dict]['bh']   = {'mass': [all_general['%s' %GalaxyID]['bh_mass']],
                                                      'mdot': [math.nan],
                                                      'mdot_instant': [all_general['%s' %GalaxyID]['bh_mdot']],
                                                      'edd': [all_general['%s' %GalaxyID]['bh_edd']],
                                                      'count': [count_bh]}
                            
                #------------------                       
                # Create angles 
                for angle_name, particle_names in zip(['stars_gas', 'stars_gas_sf', 'stars_gas_nsf', 'stars_dm', 'gas_sf_gas_nsf'], [['stars', 'gas'], ['stars', 'gas_sf'], ['stars', 'gas_nsf'], ['stars', 'dm'], ['gas_sf', 'gas_nsf']]):
                    for hmr_i in output_input['spin_hmr']:
                        # if this hmr_i not available
                        if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                            galaxy_tree['%s' %ID_dict][angle_name].update({'%s_hmr' %hmr_i: {'angle_abs': [math.nan],
                                                                                             'err_abs': [[math.nan, math.nan]],
                                                                                             'angle_proj': [math.nan],
                                                                                             'err_proj': [[math.nan, math.nan]],
                                                                                             'com_abs': [math.nan],
                                                                                             'com_proj': [math.nan]}})
                        else:
                            # Creating masks
                            mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            
                            if angle_name != 'stars_dm':
                                com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]][mask_angles], 'abs')
                                com_proj = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]][mask_angles], output_input['viewing_axis'])
                            else:
                                com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]], 'abs')
                                com_proj = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]], output_input['viewing_axis'])                                
                            
                            # Updating
                            galaxy_tree['%s' %ID_dict][angle_name].update({'%s_hmr' %hmr_i: {'angle_abs': [all_misangles['%s' %GalaxyID]['%s_angle' %angle_name][mask_angles]],
                                                                                             'err_abs': [all_misangles['%s' %GalaxyID]['%s_angle_err' %angle_name][mask_angles]],
                                                                                             'angle_proj': [all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %angle_name][mask_angles]],
                                                                                             'err_proj': [all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %angle_name][mask_angles]],
                                                                                             'com_abs': [com_abs],
                                                                                             'com_proj': [com_proj]}})
                            
                
            #=============================================
            # If not creating new entry, append instead
            else:
                # General
                galaxy_tree['%s' %ID_dict]['GalaxyID'].append(all_general['%s' %GalaxyID]['GalaxyID'])
                galaxy_tree['%s' %ID_dict]['DescendantID'].append(DescendantID)
                galaxy_tree['%s' %ID_dict]['GroupNum'].append(all_general['%s' %GalaxyID]['GroupNum'])
                galaxy_tree['%s' %ID_dict]['SubGroupNum'].append(all_general['%s' %GalaxyID]['SubGroupNum'])
                galaxy_tree['%s' %ID_dict]['SnapNum'].append(all_general['%s' %GalaxyID]['SnapNum'])
                galaxy_tree['%s' %ID_dict]['Redshift'].append(Redshift)
                galaxy_tree['%s' %ID_dict]['Lookbacktime'].append(Lookbacktime)
                # total mass of halo from mergertree
                galaxy_tree['%s' %ID_dict]['halomass'].append(all_general['%s' %GalaxyID]['halo_mass'])
                # radii
                galaxy_tree['%s' %ID_dict]['rad'].append(all_general['%s' %GalaxyID]['halfmass_rad'])
                galaxy_tree['%s' %ID_dict]['radproj'].append(all_general['%s' %GalaxyID]['halfmass_rad_proj'])
                galaxy_tree['%s' %ID_dict]['ap_sfr'].append(StarFormationRate_30_tree[row_mask][SnapNum])
                # merger analysis
                galaxy_tree['%s' %ID_dict]['merger_ID'].append(merger_ID_array)
                galaxy_tree['%s' %ID_dict]['merger_ratio_stars'].append(merger_ratio_array)
                galaxy_tree['%s' %ID_dict]['merger_ratio_gas'].append(merger_gas_array)
                
                # time_step
                time_step   = 1e9 * abs(galaxy_tree['%s' %ID_dict]['Lookbacktime'][-1] - galaxy_tree['%s' %ID_dict]['Lookbacktime'][-2])
                
                #------------------                       
                # Updating stars
                galaxy_tree['%s' %ID_dict]['stars']['tot_mass'].append(StellarMass_tree[row_mask][SnapNum])
                galaxy_tree['%s' %ID_dict]['stars']['ap_mass'].append(all_general['%s' %GalaxyID]['stelmass'])
                galaxy_tree['%s' %ID_dict]['stars']['kappa'].append(all_general['%s' %GalaxyID]['kappa_stars'])
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['count'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['stars'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['stars'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['stars'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['stars'][mask_spins], viewing_vector))
                        
                #------------------                       
                # Updating gas
                galaxy_tree['%s' %ID_dict]['gas']['tot_mass'].append(GasMass_tree[row_mask][SnapNum])
                galaxy_tree['%s' %ID_dict]['gas']['ap_mass'].append(all_general['%s' %GalaxyID]['gasmass'])
                galaxy_tree['%s' %ID_dict]['gas']['kappa'].append(all_general['%s' %GalaxyID]['kappa_gas'])
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['count'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['sfr'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['gas'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['gas'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['sfr'].append(all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['gas'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['gas'][mask_spins], viewing_vector))
                        
                        
                        
                        #================================================
                        # Find inflow/outflow
                        # If gasdata_old does not have current hmr_i, update and move on. Else... find inflow/outflow
                        if '%s_hmr'%str(hmr_i) not in galaxy_tree['%s' %ID_dict]['gasdata_old'].keys():
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                        else:
                            gasdata_old = galaxy_tree['%s' %ID_dict]['gasdata_old']['%s_hmr' %hmr_i]['gas']
                            gasdata     = all_gasdata['%s' %GalaxyID]['%s_hmr' %hmr_i]['gas']
                            
                            previous_mass = np.sum(gasdata_old['Mass'])       # M1
                            current_mass  = np.sum(gasdata['Mass'])           # M2
                            inflow_mass   = 0
                            outflow_mass  = 0
                            insitu_mass   = 0
                        
                            #------------------
                            # Check for inflow (use current, run check on previous)
                            inflow_mass_metal = 0
                            insitu_mass_metal = 0
                            for ID_i, mass_i, metal_i in zip(gasdata['ParticleIDs'], gasdata['Mass'], gasdata['Metallicity']):
                            
                                # If ID was within 2hmr of _old, gas particle stayed
                                if ID_i in gasdata_old['ParticleIDs']:
                                    insitu_mass       = insitu_mass + mass_i
                                    insitu_mass_metal = insitu_mass_metal + (mass_i * metal_i)
                                    continue
                                # If ID was NOT within 2hmr of _old, gas particle was accreted
                                else:
                                    inflow_mass = inflow_mass + mass_i
                                    inflow_mass_metal = inflow_mass_metal + (mass_i * metal_i)
                        
                            #------------------
                            # Find metallicity of inflow
                            if inflow_mass != 0:
                                inflow_Z = inflow_mass_metal / inflow_mass
                            else:
                                inflow_Z = math.nan
                        
                            # Find metallicity of insitu
                            if insitu_mass != 0:
                                insitu_Z = insitu_mass_metal / insitu_mass
                            else:
                                insitu_Z = math.nan
                        
                            #------------------    
                            # Check for outflow (use old, run check on current)
                            outflow_mass_metal = 0
                            for ID_i, mass_i, metal_i in zip(gasdata_old['ParticleIDs'], gasdata_old['Mass'], gasdata_old['Metallicity']):
            
                                # If ID will be within 2hmr of current, gas particle stayed
                                if ID_i in gasdata['ParticleIDs']:
                                    continue
                                # If ID will NOT be within 2hmr of current, gas particle was outflowed
                                else:
                                    outflow_mass = outflow_mass + mass_i
                                    outflow_mass_metal = outflow_mass_metal + (mass_i * metal_i)

                            # Find metallicity of outflow
                            if outflow_mass != 0:
                                outflow_Z = outflow_mass_metal / outflow_mass
                            else:
                                outflow_Z = math.nan
                        
                            #------------------  
                            # Left with current_mass = previous_mass + inflow_mass - outflow_mass + stellarmassloss
                            stellarmassloss = current_mass - previous_mass - inflow_mass + outflow_mass
                            
                            # Update
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_rate'].append(inflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].append(inflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].append(outflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].append(outflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(stellarmassloss / time_step)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_Z'].append(insitu_Z)
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        # Update inflow
                        
                #------------------                       
                # Updating gas_sf
                galaxy_tree['%s' %ID_dict]['gas_sf']['ap_mass'].append(all_general['%s' %GalaxyID]['gasmass_sf'])
                galaxy_tree['%s' %ID_dict]['gas_sf']['kappa'].append(all_general['%s' %GalaxyID]['kappa_gas_sf'])
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['count'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['sfr'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['gas_sf'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['gas_sf'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['sfr'].append(all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['gas_sf'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['gas_sf'][mask_spins], viewing_vector))
                        
                        
                        
                        #================================================
                        # Find inflow/outflow
                        # If gasdata_old does not have current hmr_i, update and move on. Else... find inflow/outflow
                        if ('%s_hmr'%str(hmr_i) not in galaxy_tree['%s' %ID_dict]['gasdata_old'].keys()) or (int(galaxy_tree['%s' %ID_dict]['SnapNum'][-1]) - int(galaxy_tree['%s' %ID_dict]['SnapNum'][-2]) != 1):
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                        else:
                            gasdata_old = galaxy_tree['%s' %ID_dict]['gasdata_old']['%s_hmr' %hmr_i]['gas_sf']
                            gasdata     = all_gasdata['%s' %GalaxyID]['%s_hmr' %hmr_i]['gas_sf']
                            
                            previous_mass = np.sum(gasdata_old['Mass'])       # M1
                            current_mass  = np.sum(gasdata['Mass'])           # M2
                            inflow_mass   = 0
                            outflow_mass  = 0
                            insitu_mass   = 0
                        
                            #------------------
                            # Check for inflow (use current, run check on previous)
                            inflow_mass_metal = 0
                            insitu_mass_metal = 0
                            for ID_i, mass_i, metal_i in zip(gasdata['ParticleIDs'], gasdata['Mass'], gasdata['Metallicity']):
                            
                                # If ID was within 2hmr of _old, gas particle stayed
                                if ID_i in gasdata_old['ParticleIDs']:
                                    insitu_mass       = insitu_mass + mass_i
                                    insitu_mass_metal = insitu_mass_metal + (mass_i * metal_i)
                                    continue
                                # If ID was NOT within 2hmr of _old, gas particle was accreted
                                else:
                                    inflow_mass = inflow_mass + mass_i
                                    inflow_mass_metal = inflow_mass_metal + (mass_i * metal_i)
                        
                            #------------------
                            # Find metallicity of inflow
                            if inflow_mass != 0:
                                inflow_Z = inflow_mass_metal / inflow_mass
                            else:
                                inflow_Z = math.nan
                        
                            # Find metallicity of insitu
                            if insitu_mass != 0:
                                insitu_Z = insitu_mass_metal / insitu_mass
                            else:
                                insitu_Z = math.nan
                        
                            #------------------    
                            # Check for outflow (use old, run check on current)
                            outflow_mass_metal = 0
                            for ID_i, mass_i, metal_i in zip(gasdata_old['ParticleIDs'], gasdata_old['Mass'], gasdata_old['Metallicity']):
            
                                # If ID will be within 2hmr of current, gas particle stayed
                                if ID_i in gasdata['ParticleIDs']:
                                    continue
                                # If ID will NOT be within 2hmr of current, gas particle was outflowed
                                else:
                                    outflow_mass = outflow_mass + mass_i
                                    outflow_mass_metal = outflow_mass_metal + (mass_i * metal_i)

                            # Find metallicity of outflow
                            if outflow_mass != 0:
                                outflow_Z = outflow_mass_metal / outflow_mass
                            else:
                                outflow_Z = math.nan
                        
                            #------------------  
                            # Left with current_mass = previous_mass + inflow_mass - outflow_mass + stellarmassloss
                            stellarmassloss = current_mass - previous_mass - inflow_mass + outflow_mass
                            
                            # Update
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_rate'].append(inflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].append(inflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].append(outflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].append(outflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(stellarmassloss / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_Z'].append(insitu_Z)
                 
                #------------------                       
                # Updating gas_nsf
                galaxy_tree['%s' %ID_dict]['gas_nsf']['ap_mass'].append(all_general['%s' %GalaxyID]['gasmass_nsf'])
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['count'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['gas_nsf'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['gas_nsf'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['gas_nsf'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['gas_nsf'][mask_spins], viewing_vector))
                        
                        
                        
                        #================================================
                        # Find inflow/outflow
                        # If gasdata_old does not have current hmr_i, update and move on. Else... find inflow/outflow
                        if '%s_hmr'%str(hmr_i) not in galaxy_tree['%s' %ID_dict]['gasdata_old'].keys():
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                        else:
                            gasdata_old = galaxy_tree['%s' %ID_dict]['gasdata_old']['%s_hmr' %hmr_i]['gas_nsf']
                            gasdata     = all_gasdata['%s' %GalaxyID]['%s_hmr' %hmr_i]['gas_nsf']
                            
                            previous_mass = np.sum(gasdata_old['Mass'])       # M1
                            current_mass  = np.sum(gasdata['Mass'])           # M2
                            inflow_mass   = 0
                            outflow_mass  = 0
                            insitu_mass   = 0
                        
                            #------------------
                            # Check for inflow (use current, run check on previous)
                            inflow_mass_metal = 0
                            insitu_mass_metal = 0
                            for ID_i, mass_i, metal_i in zip(gasdata['ParticleIDs'], gasdata['Mass'], gasdata['Metallicity']):
                            
                                # If ID was within 2hmr of _old, gas particle stayed
                                if ID_i in gasdata_old['ParticleIDs']:
                                    insitu_mass       = insitu_mass + mass_i
                                    insitu_mass_metal = insitu_mass_metal + (mass_i * metal_i)
                                    continue
                                # If ID was NOT within 2hmr of _old, gas particle was accreted
                                else:
                                    inflow_mass = inflow_mass + mass_i
                                    inflow_mass_metal = inflow_mass_metal + (mass_i * metal_i)
                        
                            #------------------
                            # Find metallicity of inflow
                            if inflow_mass != 0:
                                inflow_Z = inflow_mass_metal / inflow_mass
                            else:
                                inflow_Z = math.nan
                        
                            # Find metallicity of insitu
                            if insitu_mass != 0:
                                insitu_Z = insitu_mass_metal / insitu_mass
                            else:
                                insitu_Z = math.nan
                        
                            #------------------    
                            # Check for outflow (use old, run check on current)
                            outflow_mass_metal = 0
                            for ID_i, mass_i, metal_i in zip(gasdata_old['ParticleIDs'], gasdata_old['Mass'], gasdata_old['Metallicity']):
            
                                # If ID will be within 2hmr of current, gas particle stayed
                                if ID_i in gasdata['ParticleIDs']:
                                    continue
                                # If ID will NOT be within 2hmr of current, gas particle was outflowed
                                else:
                                    outflow_mass = outflow_mass + mass_i
                                    outflow_mass_metal = outflow_mass_metal + (mass_i * metal_i)

                            # Find metallicity of outflow
                            if outflow_mass != 0:
                                outflow_Z = outflow_mass_metal / outflow_mass
                            else:
                                outflow_Z = math.nan
                        
                            #------------------  
                            # Left with current_mass = previous_mass + inflow_mass - outflow_mass + stellarmassloss
                            stellarmassloss = current_mass - previous_mass - inflow_mass + outflow_mass
                            
                            # Update
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_rate'].append(inflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].append(inflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].append(outflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].append(outflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(stellarmassloss / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_Z'].append(insitu_Z)                                                                   
                                                                                      
                #------------------                       
                # Updating dm   
                galaxy_tree['%s' %ID_dict]['dm']['ap_mass'].append(all_general['%s' %GalaxyID]['dmmass'])
                galaxy_tree['%s' %ID_dict]['dm']['count'].append(all_counts['%s' %GalaxyID]['dm'])
                galaxy_tree['%s' %ID_dict]['dm']['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['dm'][-1], viewing_vector))
                
                #------------------                       
                # Updating bh                                                                 
                if np.isnan(all_general['%s' %GalaxyID]['bh_mass']) == True:
                    count_bh = 0
                else:
                    count_bh = 1                                                                      
                galaxy_tree['%s' %ID_dict]['bh']['mass'].append(all_general['%s' %GalaxyID]['bh_mass'])
                mdot = (float(galaxy_tree['%s' %ID_dict]['bh']['mass'][-1]) - float(galaxy_tree['%s' %ID_dict]['bh']['mass'][-2])) / time_step
                galaxy_tree['%s' %ID_dict]['bh']['mdot'].append(mdot)
                galaxy_tree['%s' %ID_dict]['bh']['mdot_instant'].append(all_general['%s' %GalaxyID]['bh_mdot'])
                galaxy_tree['%s' %ID_dict]['bh']['edd'].append(all_general['%s' %GalaxyID]['bh_edd'])
                galaxy_tree['%s' %ID_dict]['bh']['count'].append(count_bh)
                
                #------------------                       
                # Update angles 
                for angle_name, particle_names in zip(['stars_gas', 'stars_gas_sf', 'stars_gas_nsf', 'stars_dm', 'gas_sf_gas_nsf'], [['stars', 'gas'], ['stars', 'gas_sf'], ['stars', 'gas_nsf'], ['stars', 'dm'], ['gas_sf', 'gas_nsf']]):
                    for hmr_i in output_input['spin_hmr']:
                        # if this hmr_i not available
                        if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_abs'].append(math.nan)
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_abs'].append([math.nan, math.nan])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_proj'].append(math.nan)
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_proj'].append([math.nan, math.nan])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_abs'].append(math.nan)
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_proj'].append(math.nan)
                            
                        else:
                            # Creating masks
                            mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            
                            if angle_name != 'stars_dm':
                                com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]][mask_angles], 'abs')
                                com_proj = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]][mask_angles], output_input['viewing_axis'])
                            else:
                                com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]], 'abs')
                                com_proj = _evaluate_com(all_coms['%s' %GalaxyID][particle_names[0]][mask_angles], all_coms['%s' %GalaxyID][particle_names[1]], output_input['viewing_axis'])                                
                            
                            # Updating
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_abs'].append(all_misangles['%s' %GalaxyID]['%s_angle' %angle_name][mask_angles])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_abs'].append(all_misangles['%s' %GalaxyID]['%s_angle_err' %angle_name][mask_angles])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_proj'].append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %angle_name][mask_angles])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_proj'].append(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %angle_name][mask_angles])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_abs'].append(com_abs)
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_proj'].append(com_proj)
                                                                                                                                                        
                #------------------  
                # Update gasdata
                if csv_sample_range_i == csv_sample_range[-1]:
                    galaxy_tree['%s' %ID_dict]['gasdata_old'] = math.nan
                else:
                    galaxy_tree['%s' %ID_dict]['gasdata_old'] = all_gasdata['%s' %GalaxyID]             
                
                    
    #-----------------------------------
    # Ensuring all gasdata_old removed before saving and inserting math.nan
    print('PROCESSING...')
    for ID_dict in tqdm(galaxy_tree.keys()):
        # Erasing final gasdata
        galaxy_tree['%s' %ID_dict]['gasdata_old'] = math.nan
        
        # insert nan
        # Inserting math.nan in missing spots
        for index in np.where(np.in1d(np.arange(galaxy_tree['%s' %ID_dict]['SnapNum'][0], galaxy_tree['%s' %ID_dict]['SnapNum'][-1], 1), np.array(galaxy_tree['%s' %ID_dict]['SnapNum'])) == False)[0]:
            
            # General
            galaxy_tree['%s' %ID_dict]['GalaxyID'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['DescendantID'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['GroupNum'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['SubGroupNum'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['SnapNum'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['Redshift'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['Lookbacktime'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['halomass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['rad'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['radproj'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['ap_sfr'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['merger_ID'].insert(index, [])
            galaxy_tree['%s' %ID_dict]['merger_ratio_stars'].insert(index, [])
            galaxy_tree['%s' %ID_dict]['merger_ratio_gas'].insert(index, [])
            
            #------------------                       
            # Updating stars
            galaxy_tree['%s' %ID_dict]['stars']['tot_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['stars']['ap_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['stars']['kappa'].insert(index, math.nan)
            for hmr_i in output_input['spin_hmr']:
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['count'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['proj_angle'].insert(index, math.nan)
                
            #------------------                       
            # Updating gas
            galaxy_tree['%s' %ID_dict]['gas']['tot_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['gas']['ap_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['gas']['kappa'].insert(index, math.nan)
            for hmr_i in output_input['spin_hmr']:
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['count'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['sfr'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['proj_angle'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_Z'].insert(index, math.nan)
                   
            #------------------                       
            # Updating gas_sf
            galaxy_tree['%s' %ID_dict]['gas_sf']['ap_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['gas_sf']['kappa'].insert(index, math.nan)
            for hmr_i in output_input['spin_hmr']:
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['count'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['sfr'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['proj_angle'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_Z'].insert(index, math.nan)
                
            #------------------                       
            # Updating gas_nsf
            galaxy_tree['%s' %ID_dict]['gas_nsf']['ap_mass'].insert(index, math.nan)
            for hmr_i in output_input['spin_hmr']:
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['count'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['proj_angle'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_Z'].insert(index, math.nan)
                                                                                  
            #------------------                       
            # Updating dm   
            galaxy_tree['%s' %ID_dict]['dm']['ap_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['dm']['count'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['dm']['proj_angle'].insert(index, math.nan)
            
            #------------------                       
            # Updating bh                                                                         
            galaxy_tree['%s' %ID_dict]['bh']['mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['mdot'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['mdot_instant'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['edd'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['count'].insert(index, math.nan)
            
            #------------------                       
            # Update angles 
            for angle_name, particle_names in zip(['stars_gas', 'stars_gas_sf', 'stars_gas_nsf', 'stars_dm', 'gas_sf_gas_nsf'], [['stars', 'gas'], ['stars', 'gas_sf'], ['stars', 'gas_nsf'], ['stars', 'dm'], ['gas_sf', 'gas_nsf']]):
                for hmr_i in output_input['spin_hmr']:
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_abs'].insert(index, math.nan)
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_abs'].insert(index, [math.nan, math.nan])
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_proj'].insert(index, math.nan)
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_proj'].insert(index, [math.nan, math.nan])
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_abs'].insert(index, math.nan)
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_proj'].insert(index, math.nan)
    
    if print_summary:
        print('\nTOTAL GALAXY TREES: ', len(galaxy_tree.keys()))
    
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
        csv_dict = {'galaxy_tree': galaxy_tree}
        
        tree_input = {'csv_sample1': csv_sample1,
                      'csv_sample_range': csv_sample_range,
                      'csv_sample2': csv_sample2,
                      'csv_output_in': csv_output_in,
                      'mySims': sample_input['mySims']}
        csv_dict.update({'tree_input': tree_input,
                         'output_input': output_input,
                         'sample_input': sample_input})
        
        
        
        #-----------------------------
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%sgalaxy_tree_%s.csv' %(output_dir, csv_sample1, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/%sgalaxy_tree_%s.csv' %(output_dir, csv_sample1, csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
    
    
#--------------------------------
# Reads in tree and extracts galaxies that meet criteria
# SAVED: 
def _analyse_tree(csv_tree = 'L100_galaxy_tree_',
                  #--------------------------
                  # Galaxy analysis
                  print_summary             = True,
                  print_galaxy              = True,
                    plot_misangle_detection = True,
                  #-----------------------
                  # Individual galaxies
                  force_all_snaps = True,                # KEEP TRUE. force us to use galaxies in which we have all snaps.
                    GalaxyID_list = [401467650],         # [ None / list of IDs to process ]
                  #====================================================================================================
                  # Misalignment must take place in range   
                    max_z = 0.8,                  
                    min_z = None,                         # [ None / value ] min/max z to sample galaxies BECOMING MISALIGNED
                  # Radii to extract
                    use_hmr_general     = 'aperture',    # [ 1.0 / 2.0 / aperture]      Used for stelmass, sfr. ssfr
                    use_hmr_angle       = 2.0,           # [ 1.0 / 2.0 ]                Used for misangle, inc angle, com, counts
                  #--------------------------------------------------------
                  # PROPERTIES TO ALWAYS MEET
                    min_particles     = 20,              # [count]
                    max_com           = 2.0,             # [pkpc]
                    min_inclination   = 10,              # [ 0 / degrees]
                  #--------------------------------------------------------
                  # PROPERTIES TO AVERAGE OVER WHILE MISALIGNED / RELAXING
                  # Satellites, central, or both         [ 'satellite' is sgn >= 1 / 'central' is sgn == 1 / None ]
                    limit_satellites    = 'satellite',
                  # Group / Field / both                 [ None / value ] (halo threshold: 10**14)
                    min_halomass        = None,         
                    max_halomass        = None, 
                  # Stellar mass and SFR                 [ None / value ]
                    min_stelmass        = 1E10,
                    max_stelmass        = 5E10,
                  # SF and quiescent galaxies            [ None / value ]
                    min_sfr             = None,          
                    max_sfr             = None,          # SF limit of ~ 0.1   Msun/yr
                    min_ssfr            = None,          # SF limit of ~ 1e-11 /yr
                    max_ssfr            = None,         
                  # Morphology stars / gas.              [ None / value ] (ETG threshold: 0.4)
                    min_kappa_stars     = None,
                    max_kappa_stars     = None,
                    min_kappa_gas       = None,          # For LTG-style, typically around 0.8+, for ETG-style, 0.6+
                    max_kappa_gas       = None,          # Take these with pinch of salt though
                    min_kappa_sf        = None,         
                    max_kappa_sf        = None,
                    min_kappa_nsf       = None,
                    max_kappa_nsf       = None,
                  # Radius limits                        [ None / value ]
                    min_rad             = None,     
                    max_rad             = None,   
                  # Inflow (gas) | 7.0 is pretty high    [ None / value ]
                    min_inflow          = None,             
                    max_inflow          = None,
                  # BH                                   [ None / value ]
                    min_bh_mass         = None,
                    max_bh_mass         = None,
                    min_edd             = None,          # Uses instantaneous accretion rate
                    max_edd             = None,
                    min_lbol            = None,
                    max_lbol            = None,
                  #--------------------------------------------------------
                  # Misalignment angles                
                    abs_or_proj         = 'abs',         # [ 'abs' / 'proj' ]
                    use_angle           = 'stars_gas_sf',
                    misangle_threshold  = 30,            # [ 30 / 45 ]
                    min_delta_angle     = 10,            # [ None / deg ] Change in angle between successive snapshots from aligned to misaligned
                    latency_time        = 0.24,          # [ None / Gyr ] Consecutive time galaxy must be <30 / >150 to count as finished relaxing
                  #--------------------------------------------------------
                  # Mergers 
                  use_merger_criteria   = True,          # Whether we limit to merger-induced, or any misalignments
                    min_stellar_ratio   = 0.1,           # [ value ]
                    max_stellar_ratio   = 2.0,               
                    min_gas_ratio       = None,          # [ None / value ]
                    max_gas_ratio       = None,          # [ None / value ]
                    max_closest_merger  = 1.0,           # [Gyr] ± max time to closest merger from point of misalignment
                  #--------------------------------------------------------
                  # Temporal selection
                    time_extra       = 0.2,      # [Gyr], extra time before and after misalignment which is also extracted
                    time_no_misangle = None,     # [Gyr], extra time before and after misalignment which has no misalignments. Similar to relax snapshots
                  #====================================================================================================
                  csv_file       = False,             # Will write sample to csv file in sample_dir
                    csv_name     = '',               # extra stuff at end
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
    if GalaxyID_list != None:
        GalaxyID_list_extract = []
        for GalaxyID_find in GalaxyID_list:
            for ID_i in np.arange(GalaxyID_find, GalaxyID_find+65):
                if str(ID_i) in galaxy_tree.keys():
                    GalaxyID_list_extract.append(ID_i)
                    print('ID %s found in galaxy_tree' %ID_i)
    
    
    #----------------------------
    # Loop over all galaxies
    misalignment_tree = {}
    for GalaxyID in tqdm(galaxy_tree.keys()):
        
        #--------------------------------
        # QUICK FILTERS
        
        # If we are looking at individual galaxies, filter them out
        if GalaxyID_list != None:
            if int(GalaxyID) not in GalaxyID_list_extract:
                continue
        
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('QUICK FILTERS')
            time_start = time.time()
        
        # Test if any misalignments even present in z range, else move on
        mask_test_z = np.where((np.array(galaxy_tree['%s' %GalaxyID]['Redshift']) >= (-1 if min_z == None else min_z)) & (np.array(galaxy_tree['%s' %GalaxyID]['Redshift']) <= (999 if max_z == None else max_z)))[0]
        mask_test_z = np.arange(mask_test_z[0], mask_test_z[-1]+1)
        mask_test_z_misangle = np.where(np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj])[mask_test_z] > misangle_threshold)
        if len(mask_test_z_misangle) == 0:
            continue
        
        if print_galaxy:
            print('ID: ', ID_i)
            print('In range %s - %s:\nIndex\tSnap\tz\tTime\tAngLo\tAngle\tAngHi\tRatio' %(max_z, min_z))
            for index, snap_i, time_i, z_i, angle_i, err_i, merger_i in zip(np.array(galaxy_tree['%s' %GalaxyID]['SnapNum'])[mask_test_z] - np.array(galaxy_tree['%s' %GalaxyID]['SnapNum'])[0], np.array(galaxy_tree['%s' %GalaxyID]['SnapNum'])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['Lookbacktime'])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['Redshift'])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], dtype=object)[mask_test_z]):
                print('%s\t%s\t%.2f\t%.2f\t%.1f\t%.1f\t%.1f\t%.2f' %(index, snap_i, z_i, time_i, err_i[0], angle_i, err_i[1], max(merger_i, default=0)))
            print(' ')
            
        
        
        #--------------------------------
        # CHECK 1: establishing a complete window for misalignment (pre-post at z)
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('LOCATING MISALIGNMENTS')
            time_start = time.time()
        
        # Identify indexes of start and ends of individual relaxations
        misalignment_locations = {'misalign': {'index': [],
                                               'snapnum': []},
                                  'relax':    {'index': [],
                                               'snapnum': []}}
                                               
        all_angles = galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj]
        misalignment_started = False
        index_ahead = math.nan
        for index, snap_i, angle_i in zip(np.arange(0, len(all_angles)+1), galaxy_tree['%s' %GalaxyID]['SnapNum'], all_angles):
        
            # If there is a nan inbetween, reset count
            if angle_i == math.nan:
                misalignment_started = False
        
            if index < (len(all_angles)-1):
                # Check for start of misalignment that meets conditions
                if (misalignment_started == False) & (angle_i < misangle_threshold) & (all_angles[index+1] > misangle_threshold) & ((all_angles[index+1] - angle_i) >= (0 if min_delta_angle == None else min_delta_angle)) & (galaxy_tree['%s' %GalaxyID]['Redshift'][index] >= (-1 if min_z == None else min_z)) & (galaxy_tree['%s' %GalaxyID]['Redshift'][index] <= (999 if max_z == None else max_z)):
                    misalignment_started = True
                    misalignment_started_index = index
                    misalignment_started_snap  = snap_i
        
            # If we have begun a misalignment, check how many snaps we need to check for in the future
            if (misalignment_started == True):
                if len(np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index] - (0 if latency_time == None else latency_time)))[0]) == 0:
                    index_ahead = math.nan
                    continue
                else:
                    index_ahead = np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index] - (0 if latency_time == None else latency_time)))[0][0] - snap_i
                if debug:
                    print(snap_i, angle_i, index_ahead)
            
            # If we have begun a misalignment, check if it relaxes within the time
            if np.isnan(index_ahead) == False:
                if (misalignment_started == True) & (index < (len(all_angles)-1-index_ahead)):
                    check_consecutive_index = 0
                    if debug:
                        print('limiting index:', len(all_angles)-1-index_ahead, index)
            
                    # if it relaxes to counter-, ensure it stays counter
                    if (all_angles[index+1] < misangle_threshold):
                        # Loop over all snaps ahead of time
                        for index_ahead_i in np.arange(0, index_ahead+1):
                            if debug:
                                print('ahead: ', all_angles[index+1+index_ahead_i])
                                print(galaxy_tree['%s' %GalaxyID]['SnapNum'][index+1+index_ahead_i])
                            if (all_angles[index+1+index_ahead_i] < misangle_threshold): 
                                check_consecutive_index += 1
                                if debug:
                                    print('   met ', all_angles[index+1+index_ahead_i])
                            else:
                                if debug:
                                    print('   not met ', all_angles[index+1+index_ahead_i])
                    
                        if abs(check_consecutive_index) == index_ahead+1:
                            misalignment_locations['misalign']['index'].append(misalignment_started_index )
                            misalignment_locations['misalign']['snapnum'].append(misalignment_started_snap)
                            misalignment_locations['relax']['index'].append((index+1))
                            misalignment_locations['relax']['snapnum'].append(galaxy_tree['%s' %GalaxyID]['SnapNum'][index+1])
                            misalignment_started = False        
                
            
                    # if it relaxes to co-, ensure it stays regular
                    elif (all_angles[index+1] > (180-misangle_threshold)):
                        # Loop over all snaps ahead of time
                        for index_ahead_i in np.arange(0, index_ahead+1):
                            if debug:
                                print('ahead: ', all_angles[index+1+index_ahead_i])
                                print(galaxy_tree['%s' %GalaxyID]['SnapNum'][index+1+index_ahead_i])
                            if (all_angles[index+1+index_ahead_i] > (180-misangle_threshold)):
                                check_consecutive_index += 1
                                if debug:
                                    print('   met ', all_angles[index+1+index_ahead_i])
                            else:
                                if debug:
                                    print('   not met ', all_angles[index+1+index_ahead_i])
                    
                        if abs(check_consecutive_index) == index_ahead+1:
                            misalignment_locations['misalign']['index'].append(misalignment_started_index )
                            misalignment_locations['misalign']['snapnum'].append(misalignment_started_snap)
                            misalignment_locations['relax']['index'].append((index+1))
                            misalignment_locations['relax']['snapnum'].append(galaxy_tree['%s' %GalaxyID]['SnapNum'][index+1])
                            misalignment_started = False
                    
                    else:
                        continue
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('done')
            time_start = time.time()
        if debug:
            print(misalignment_locations.items())
           
        # Optionally plot detection
        if print_galaxy:
            if len(misalignment_locations['misalign']['index']) > 0:
                print('Snap -\tSnap\tTime -\tTime\tDuration [Gyr]')
                for index_m, index_r, snap_m, snap_r in zip(misalignment_locations['misalign']['index'], misalignment_locations['relax']['index'], misalignment_locations['misalign']['snapnum'], misalignment_locations['relax']['snapnum']):
                    print('%s\t%s\t%.2f\t%.2f\t%.2f' %(snap_m, snap_r, galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r], abs(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m])))
            else:
                print('\n> No misalignments using imposed limits <')
        if plot_misangle_detection:
            # Plot mergers
            for time_i, ratio_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars']):
                #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                if len(ratio_i) > 0:
                    if max(ratio_i) > 0.1:
                        plt.axvline(time_i, c='grey', ls='--', lw=2)
                        plt.text(time_i-0.2, 170, '%.2f' %max(ratio_i), color='grey')
            # Plot misalignment detections
            for index_m, index_r in zip(misalignment_locations['misalign']['index'], misalignment_locations['relax']['index']):
                plt.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]], [90,90], lw=10, color='grey', solid_capstyle="butt", alpha=0.5)
                plt.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-(0 if latency_time==None else latency_time)], [90,90], lw=5, color='lightgrey', solid_capstyle="butt", alpha=0.5)
                plt.text(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], 88, '%.2f' %abs(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m]), fontsize=9)
                plt.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], ymin=85, ymax=95, lw=1, colors='grey', alpha=0.5)
                plt.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r], ymin=85, ymax=95, lw=1, colors='grey', alpha=0.5)
                
            plt.plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], all_angles, 'go-', mec='k', ms=1)
            plt.axhline(misangle_threshold, lw=0.5, c='k')
            plt.axhline(180-misangle_threshold, lw=0.5, c='k')
            plt.ylim(0, 181)
            plt.xlim(8.1, 0)
            plt.show()
        
        # Skip galaxy if no misalignments detected
        if len(misalignment_locations['misalign']['index']) == 0:
            continue
        
        
           
        

        # For each misalignment and relaxation PAIR:
        # Use mergertree to find hypothetical snapnums either side, and their indexes for this galaxy (lower + upper)
        # Use lower + upper to create np.arange(), and ensure these are all in SnapNum list for this galaxy
        
        
        
        
        raise Exception('current break')
        
        
        
        
        
        
    
        
        
        
        
        
        # CHECK 1: establishing a complete window for misalignment (pre-post at z)
        # Find location of misalignments (which is within z range)
        # Find location of relaxations (X consecutive snapshots aligned)
        # For each misalignment and relaxation PAIR:
        # Use mergertree to find hypothetical snapnums either side, and their indexes for this galaxy (lower + upper)
        # Use lower + upper to create np.arange(), and ensure these are all in SnapNum list for this galaxy
        # If all these conditions met, this galaxy can be checked for mergers and inflows
        
        # CHECK 2: if use_merger_criteria
        # Using index of misalignment and merger tree, find index range to seach for mergers
        # Check if there is a merger meeting criteria
        # If met, move on to next check
        
        # CHECK 3: inflows/outflows, other misalignments
        # Check if there are any other misalignments in range
        # If met, move on to next check
        
        # CHECK 4: all in range checks
        # Apply checks over all in range
        
        # CHECK 5:
        # Apply checks to averages in misaligned range
        
        # If this galaxy's particular misalignment passes, append arrays to new misalignment_tree starting from first ID
        # Append also: index start, index relax, relaxation time
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    #if use_hmr_general == 'aperture':
    #    check_mass = (np.array(galaxy_tree['12345']['stars']['ap_mass'])[mask_z] >= (1e9 if min_stelmass == None else min_stelmass)) & (np.array(galaxy_tree['12345']['stars']['ap_mass'])[mask_z] <= (1e15 if max_stelmass == None else max_stelmass))
    #else:
    #    check_mass = (np.array(galaxy_tree['12345']['stars']['%s_hmr' %use_hmr_general]['mass'])[mask_z] >= (1e9 if min_stelmass == None else min_stelmass)) & (np.array(galaxy_tree['12345']['%s_hmr' %use_hmr_general]['mass'])[mask_z] <= (1e15 if max_stelmass == None else max_stelmass))
    #check_count = (np.array(galaxy_tree['12345']['stars']['%s_hmr' %use_hmr_angle]['count'])[mask_z] >= (1 if min_particles == None else min_particles)) 
    #check_inc   = (np.array(galaxy_tree['12345']['stars']['%s_hmr' %use_hmr_angle]['proj_angle'])[mask_z] >= (0 if min_inclination == None else min_inclination))
    
    
    
    
    
    
    
    
    
    
    #------------------------------
    # Test plot
    
    # apply criteria to range of misangle
    # overplot outcome of filters
    
    """#------------
    GalaxyID_insert = 17879310
    for ID_i in np.arange(GalaxyID_insert, GalaxyID_insert+60):
        if str(ID_i) in galaxy_tree.keys():
            print(ID_i)
            GalaxyID = ID_i
            #print(galaxy_tree['%s' %ID_i].keys())
            
    for time_i, ratio_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars']):
        #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
        if len(ratio_i) > 0:
            if max(ratio_i) > 0.1:
                plt.axvline(time_i, c='grey', ls='--', lw=2)
                plt.text(time_i-0.2, 170, '%.2f' %max(ratio_i), color='grey')
    
    for snap_i, ID_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['SnapNum'], galaxy_tree['%s' %GalaxyID]['merger_ID'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
        print(snap_i, ID_i, ratio_i, gas_i)
    
    plt.scatter(galaxy_tree['401467700']['Lookbacktime'], np.full(len(galaxy_tree['401467700']['Lookbacktime']), 170), color='k')
    plt.scatter(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], np.full(len(galaxy_tree['%s' %GalaxyID]['Lookbacktime']), 165), color='r')
    plt.plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['stars_gas_sf']['1.0_hmr']['angle_abs'])
    plt.ylim(0, 180)
    plt.xlim(8, 0)
    plt.show()
    #------------
    """
    
    
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
        csv_dict = {'misalignment_tree': misalignment_tree}
        
        misalignment_input = {'all entries': 0}
        csv_dict.update({'misalignment_input': misalignment_input,
                         'tree_input': tree_input,
                         'output_input': output_input,
                         'sample_input': sample_input})
        
        #-----------------------------
        # Create unique string name for file
        def create_uuid_from_string(val: str):
            hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
            return uuid.UUID(hex=hex_string)

        csv_string = "TrueTrueFalse8E10TrueTrueFalse8E10TrueTrueFalse8E10TrueTrueFalse8E10TrueTrueFalse8E10TrueTrueFalse8E10TrueTrueFalse8E10TrueTrueFalse8E10TrueTrueFalse8E10"
        print(create_uuid_from_string(csv_string))
        
        
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/L100_misalignment_tree_%s_%s.csv' %(output_dir, csv_string, csv_name), 'w'), cls=NumpyEncoder)
        print('\n  SAVED: %s/L100_misalignment_tree_%s_%s.csv' %(output_dir, csv_string, csv_name))
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        
        
        
        
    
    
    
    
    
    

    
#=============================
#_create_galaxy_tree()  

_analyse_tree()
#=============================
    