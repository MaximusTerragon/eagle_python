import h5py
import numpy as np
import scipy
from scipy import stats
import math
import random
import uuid
import hashlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, NullFormatter)
import seaborn as sns
import pandas as pd
from plotbin.sauron_colormap import register_sauron_colormap
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
from graphformat import set_rc_params, lighten_color
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
register_sauron_colormap()
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
ap_sfr					- [Msun/yr] in aperture
rad                     - [pkpc]
radproj                 - [pkpc]
rad_sf  				- [pkpc]
ellip					- 1 is flat, 0 is sphere Typically 0.2-0.3 for ETG, 0.6-0.8 for LTG
triax					- Low values are oblate, high are prolate
disp_ani				- Values of δ > 0 indicate that the velocity dispersion is primarily contributed by disordered motion in the disc plane
disc_to_total			- The mass fraction of stars that are rotationally-supported (which can be considered the ‘disc’ mass fraction) is a simple and intuitive kinematic diagnostic.
rot_to_disp_ratio		- the ratio of rotation and dispersion velocities
merger_ID               - merger stuff
merger_ratio_stars      - peak stellar ratios in past 2 Gyr ish
merger_ratio_gas        - gas ratios at time of peak stellar ratios
gasdata_old             - gets updated and replaced with math.nan

'other'
    '1.0_hmr'
    '2.0_hmr'
        tot_mass        - [Msun] total mass of all masses
        vcirc           - [km/s] circular velocity according to virial theorem
        tdyn            - [Gyr] relaxation time at this radius
        ttorque           	- [Gyr] torquing time at this radius = tdyn/ellip
		
'stars'
    tot_mass            - total subfind mass
    ap_mass             - total aperture mass
    kappa               - kappa
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        Z               - metallicity in radius
		l				- specific angular momentum (spin) in units of [pkpc/kms-1]. 109 M have roughly log(l) of 1.5, 1010.5 have roughly log(l) of 2.5
        proj_angle      - angle to viewing axis

'gas'
    tot_mass            - total subfind mass
    ap_mass             - total aperture mass
    kappa               - kappa
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        sfr             - sfr in radius [Msun/yr]
        Z               - metallicity in radius
		l				- specific angular momentum (spin) in units of [pkpc/kms-1]
        proj_angle      - angle to viewing axis
        inflow_rate     - inflow rate at radius [Msun/yr]
        inflow_Z        - inflow metallicity (mass-weighted, but Z is fine)
        outflow_rate    - outflow rate at radius [Msun/yr]
        outflow_Z       - outflow metallicity (mass-weighted, but Z is fine)
        stelmassloss_rate   - stellar mass loss rate [Msun/yr]
        insitu_Z        - metallicity of material that remained (mass-weighted, but Z is fine)

'gas_sf'
    ap_mass             - total aperture mass
    kappa               - kappa
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        sfr             - sfr in radius [Msun/yr]
        Z               - metallicity in radius
		l				- specific angular momentum (spin) in units of [pkpc/kms-1]
        proj_angle      - angle to viewing axis
        inflow_rate     - inflow rate at radius [Msun/yr]
        inflow_Z        - inflow metallicity (mass-weighted, but Z is fine)
        outflow_rate    - outflow rate at radius [Msun/yr]
        outflow_Z       - outflow metallicity (mass-weighted, but Z is fine)
        stelmassloss_rate   - stellar mass loss rate [Msun/yr]
        insitu_Z        - metallicity of material that remained (mass-weighted, but Z is fine)

'gas_nsf'
    ap_mass             - total aperture mass
    kappa               - kappa
    '1.0_hmr'
    '2.0_hmr'
        mass            - mass in radius
        count           - counts in radius
        Z               - metallicity in radius
		l				- specific angular momentum (spin) in units of [pkpc/kms-1]
        proj_angle      - angle to viewing axis
        inflow_rate     - inflow rate at radius [Msun/yr]
        inflow_Z        - inflow metallicity (mass-weighted, but Z is fine)
        outflow_rate    - outflow rate at radius [Msun/yr]
        outflow_Z       - outflow metallicity (mass-weighted, but Z is fine)
        stelmassloss_rate   - stellar mass loss rate [Msun/yr]
        insitu_Z        - metallicity of material that remained (mass-weighted, but Z is fine)

'dm'
    ap_mass             - total aperture mass
    count               - total count in aperture
    proj_angle          - angle to viewing axis
	l					- specific angular momentum (spin) in units of [pkpc/kms-1] in maximum hmr available

'bh'
    mass                - bh mass of central BH (not particle mass)
    mdot_instant        - [ Msun/yr ] mdot of that particle
    mdot                - [ Msun/yr ] mdot averaged over snipshot time difference
    edd                 - instantaneous eddington from mdot_instant
	lbol				- instantaneous bolometric luminosity [erg/s]
    count               - len(mass)

'stars_gas'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
		angle_halo		- inner stars vs outer gas
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for viewing axis

'stars_gas_sf'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
		angle_halo		- inner stars vs outer gas_sf
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'gas_sf_gas_nsf'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
		angle_halo		- inner gas_sf vs outer gas_nsf
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'stars_dm'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
		angle_halo		- outer stars vs dm halo
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'gas_dm'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
		angle_halo		- outer gas vs dm halo
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis

'gas_sf_dm'
    '1.0_hmr'           value may not exist if pkpc was capped at maximum. If 0 particles appends math.nan, will have values for everything else (even min particle not met)
    '2.0_hmr'
        angle_abs       - angle
        err_abs         - error 
        angle_proj      - projected angle
        err_proj        - error of projected
		angle_halo		- outer gas_sf vs dm halo
        com_abs         - com [pkpc]
        com_proj        - com [pkpc] for  viewing axis
            
"""
# Goes through all csv samples given and creates giant merger tree, no criteria used
# SAVED: /outputs_snips/%sgalaxy_tree_
def _create_galaxy_tree(csv_sample1 = 'L100_',                                 # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                        csv_sample2 = '_all_sample_misalignment_9.5',
                        csv_sample_range = np.arange(134, 201, 1),   # snapnums
                        csv_output_in = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                        #--------------------------
                        # Galaxy analysis
                        print_summary  = True,
                          disc_edge    = 2.0,               # HMR of r_50 SF that we consider the disk
                          merger_lookback_time = 2,      # [ 2 / Gyr ] Maximum time to look back for peak stellar mass
                        #--------------------------
                        csv_file       = True,             # Will write sample to csv file in sample_dir
                          csv_name     = '_NEW_NEW',               # extra stuff at end
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
        all_spinshalo       = dict_output['all_spinshalo']
        all_coms            = dict_output['all_coms']
        all_counts          = dict_output['all_counts']
        all_masses          = dict_output['all_masses']
        all_totmass         = dict_output['all_totmass']
        all_sfr             = dict_output['all_sfr']
        all_Z               = dict_output['all_Z']
        all_l               = dict_output['all_l']
        all_misangles       = dict_output['all_misangles']
        all_misangleshalo   = dict_output['all_misangleshalo']
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
                    # Find last snap up to 2 Gyr ago
                    SnapNum_merger = np.where(Lookbacktime_tree >= (Lookbacktime_tree[SnapNum] + merger_lookback_time))[0][-1]
                    
                    # Find largest stellar mass of this satellite, per method of Rodriguez-Gomez et al. 2015, Qu et al. 2017 (see crain2017)
                    mass_mask = np.argmax(StellarMass_tree[mask_i][int(SnapNum_merger-100):int(SnapNum)]) + (SnapNum_merger-100)
                    
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
                    if merger_ratio > 1:
                        merger_ratio = 1/merger_ratio
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
                                               'ap_sfr': [(3.154e+7*all_general['%s' %GalaxyID]['ap_sfr'])],       
                                               # radii
                                               'rad': [all_general['%s' %GalaxyID]['halfmass_rad']],
                                               'radproj': [all_general['%s' %GalaxyID]['halfmass_rad_proj']],
                                               'rad_sf': [all_general['%s' %GalaxyID]['halfmass_rad_sf']],
                                               # morpho kinem
                                               'ellip': [all_general['%s' %GalaxyID]['ellip']],
                                               'triax': [all_general['%s' %GalaxyID]['triax']],
                                               'disp_ani': [all_general['%s' %GalaxyID]['disp_ani']],
                                               'disc_to_total': [all_general['%s' %GalaxyID]['disc_to_total']],
                                               'rot_to_disp_ratio':  [all_general['%s' %GalaxyID]['rot_to_disp_ratio']],
                                               # merger analysis
                                               'merger_ID': [merger_ID_array],
                                               'merger_ratio_stars': [merger_ratio_array],
                                               'merger_ratio_gas': [merger_gas_array],
                                               # gasdata
                                               'gasdata_old': all_gasdata['%s' %GalaxyID],             # update going forward
                                               # other properties in rad
                                               'other': {},
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
                                               'stars_dm': {},
                                               'gas_dm': {},
                                               'gas_sf_dm': {}}
                
                if csv_sample_range_i == csv_sample_range[-1]:
                    galaxy_tree['%s' %ID_dict]['gasdata_old'] = math.nan
                
                #------------------                       
                # Create other
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if (hmr_i not in all_totmass['%s' %GalaxyID]['hmr']) or (disc_edge not in all_totmass['%s' %GalaxyID]['hmr']):
                        # Updating
                        galaxy_tree['%s' %ID_dict]['other'].update({'%s_hmr' %hmr_i: {'tot_mass': [math.nan],
                                                                                      'vcirc': [math.nan],
                                                                                      'tdyn': [math.nan],
                                                                                      'ttorque': [math.nan]}})                        
                    else:
                        # Creating masks. mask_disc = disc definition, mask_masses = simply the currnet 
                        mask_disc = np.where(np.array(all_totmass['%s' %GalaxyID]['hmr']) == float(disc_edge))[0][0]
                        mask_masses  = np.where(np.array(all_totmass['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # if edge of SF disk larger than current hmr, we cap and use R and M of current hmr. Else we use R and M of SF disk (most commonly done for ETGs)
                        if (disc_edge*float(all_general['%s' %GalaxyID]['halfmass_rad_sf'])) > (hmr_i*float(all_general['%s' %GalaxyID]['halfmass_rad'])):
                            R = hmr_i*float(all_general['%s' %GalaxyID]['halfmass_rad'])            # R of hmr
                            M = float(all_totmass['%s' %GalaxyID]['mass'][mask_masses])             # totmass at hmr edge
                        else:
                            R = disc_edge*float(all_general['%s' %GalaxyID]['halfmass_rad_sf'])     # R of gassf
                            M = float(all_totmass['%s' %GalaxyID]['mass_disc'][mask_disc])          # totmass at disk edge
                        
                        
                        # finding vcirc in [kms/s], tdyn in [Gyr]
                        vcirc = (1/1000 * np.sqrt(np.divide(np.array(M) * 2e30 * 6.67e-11, R * 3.09e19)))
                        tdyn  = 1e-9 * np.divide(2*np.pi * np.array(R) * 3.09e19, vcirc*1000) / 3.154e+7
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['other'].update({'%s_hmr' %hmr_i: {'tot_mass': [M],
                                                                                      'vcirc': [vcirc],
                                                                                      'tdyn': [tdyn],
                                                                                      'ttorque': [tdyn/all_general['%s' %GalaxyID]['ellip']]}})
                        
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
                                                                                      'l': [math.nan],
                                                                                      'proj_angle': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['stars'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['stars'][mask_masses]],
                                                                                      'count': [all_counts['%s' %GalaxyID]['stars'][mask_counts]],                     
                                                                                      'Z': [all_Z['%s' %GalaxyID]['stars'][mask_Z]],                      
                                                                                      'l': [all_l['%s' %GalaxyID]['stars'][mask_l]],  
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
                                                                                    'l': [math.nan],
                                                                                    'proj_angle': [math.nan],
                                                                                    # inflow/outflow and metallicity flow
                                                                                    'inflow_rate': [math.nan],
                                                                                    'inflow_mass': [math.nan],
                                                                                    'inflow_Z': [math.nan],
                                                                                    'outflow_rate': [math.nan],
                                                                                    'outflow_mass': [math.nan],
                                                                                    'outflow_Z': [math.nan],
                                                                                    'stelmassloss_rate': [math.nan],
                                                                                    'insitu_mass': [math.nan],
                                                                                    'insitu_Z': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['gas'][mask_masses]],
                                                                                    'count': [all_counts['%s' %GalaxyID]['gas'][mask_counts]],                     
                                                                                    'Z': [all_Z['%s' %GalaxyID]['gas'][mask_Z]],                       
                                                                                    'l': [all_l['%s' %GalaxyID]['gas'][mask_l]], 
                                                                                    'sfr': [(3.154e+7*all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr])], 
                                                                                    'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['gas'][mask_spins], viewing_vector)],
                                                                                    # inflow/outflow and metallicity flow
                                                                                    'inflow_rate': [math.nan],
                                                                                    'inflow_mass': [math.nan],
                                                                                    'inflow_Z': [math.nan],
                                                                                    'outflow_rate': [math.nan],
                                                                                    'outflow_mass': [math.nan],
                                                                                    'outflow_Z': [math.nan],
                                                                                    'stelmassloss_rate': [math.nan],
                                                                                    'insitu_mass': [math.nan],
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
                                                                                       'l': [math.nan],
                                                                                       'proj_angle': [math.nan],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_mass': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_mass': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_mass': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_sf'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['gas_sf'][mask_masses]],
                                                                                       'count': [all_counts['%s' %GalaxyID]['gas_sf'][mask_counts]],                     
                                                                                       'Z': [all_Z['%s' %GalaxyID]['gas_sf'][mask_Z]],                       
                                                                                       'l': [all_l['%s' %GalaxyID]['gas_sf'][mask_l]], 
                                                                                       'sfr': [3.154e+7*(all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr])],                 
                                                                                       'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['gas_sf'][mask_spins], viewing_vector)],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_mass': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_mass': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_mass': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                                                                          
                #------------------                       
                # Create gas_nsf
                galaxy_tree['%s' %ID_dict]['gas_nsf'] = {'ap_mass': [all_general['%s' %GalaxyID]['gasmass_nsf']],
                                                         'kappa': [all_general['%s' %GalaxyID]['kappa_gas_nsf']]}
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf'].update({'%s_hmr' %hmr_i: {'mass': [math.nan],
                                                                                        'count': [math.nan],                 
                                                                                        'Z': [math.nan],   
                                                                                        'l': [math.nan],
                                                                                        'proj_angle': [math.nan],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_mass': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_mass': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_mass': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf'].update({'%s_hmr' %hmr_i: {'mass': [all_masses['%s' %GalaxyID]['gas_nsf'][mask_masses]],
                                                                                        'count': [all_counts['%s' %GalaxyID]['gas_nsf'][mask_counts]],                     
                                                                                        'Z': [all_Z['%s' %GalaxyID]['gas_nsf'][mask_Z]],                        
                                                                                        'l': [all_l['%s' %GalaxyID]['gas_nsf'][mask_l]],                 
                                                                                        'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['gas_nsf'][mask_spins], viewing_vector)],
                                                                                        # inflow/outflow and metallicity flow
                                                                                        'inflow_rate': [math.nan],
                                                                                        'inflow_mass': [math.nan],
                                                                                        'inflow_Z': [math.nan],
                                                                                        'outflow_rate': [math.nan],
                                                                                        'outflow_mass': [math.nan],
                                                                                        'outflow_Z': [math.nan],
                                                                                        'stelmassloss_rate': [math.nan],
                                                                                        'insitu_mass': [math.nan],
                                                                                        'insitu_Z': [math.nan]}})
                                                                          
                #------------------                       
                # Create dm
                galaxy_tree['%s' %ID_dict]['dm']   = {'ap_mass': [all_general['%s' %GalaxyID]['dmmass']],
                                                      'count': [all_counts['%s' %GalaxyID]['dm']],
                                                      'proj_angle': [_find_angle(all_spins['%s' %GalaxyID]['dm'][-1], viewing_vector)],
                                                      'l': [all_l['%s' %GalaxyID]['dm'][-1]]}
                                                      
                    
                #------------------                       
                # Create bh
                if np.isnan(all_general['%s' %GalaxyID]['bh_mass']) == True:
                    count_bh = 0
                else:
                    count_bh = 1
                galaxy_tree['%s' %ID_dict]['bh']   = {'mass': [all_general['%s' %GalaxyID]['bh_mass']],
                                                      'mdot': [math.nan],
                                                      'mdot_instant': [(3.154e+7*all_general['%s' %GalaxyID]['bh_mdot'])],
                                                      'edd': [all_general['%s' %GalaxyID]['bh_edd']],
                                                      'lbol': [(all_general['%s' %GalaxyID]['bh_mdot'] * (2e30) * (0.1 * (3e8)**2) * (1e7))],
                                                      'count': [count_bh]}
                            
                #------------------                       
                # Create angles (normal and halo)
                for angle_name, particle_names in zip(['stars_gas', 'stars_gas_sf', 'stars_dm', 'gas_dm', 'gas_sf_dm'], [['stars', 'gas'], ['stars', 'gas_sf'], ['stars', 'dm'], ['gas', 'dm'], ['gas_sf', 'dm']]):
                    for hmr_i in output_input['spin_hmr']:
                        # if this hmr_i not available
                        if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                            galaxy_tree['%s' %ID_dict][angle_name].update({'%s_hmr' %hmr_i: {'angle_abs': [math.nan],
                                                                                             'err_abs': [[math.nan, math.nan]],
                                                                                             'angle_proj': [math.nan],
                                                                                             'err_proj': [[math.nan, math.nan]],
                                                                                             'angle_halo': [math.nan],
                                                                                             'com_abs': [math.nan],
                                                                                             'com_proj': [math.nan]}})
                        else:
                            # Creating masks
                            mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_angles_halo = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            
                            if angle_name not in ('stars_dm', 'gas_dm', 'gas_sf_dm'):
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
                                                                                             'angle_halo': [all_misangleshalo['%s' %GalaxyID]['%s_angle' %angle_name][mask_angles_halo]],
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
                galaxy_tree['%s' %ID_dict]['ap_sfr'].append((3.154e+7*all_general['%s' %GalaxyID]['ap_sfr']))
                # radii
                galaxy_tree['%s' %ID_dict]['rad'].append(all_general['%s' %GalaxyID]['halfmass_rad'])
                galaxy_tree['%s' %ID_dict]['radproj'].append(all_general['%s' %GalaxyID]['halfmass_rad_proj'])
                galaxy_tree['%s' %ID_dict]['rad_sf'].append(all_general['%s' %GalaxyID]['halfmass_rad_sf'])
                # morpho kinem
                galaxy_tree['%s' %ID_dict]['ellip'].append(all_general['%s' %GalaxyID]['ellip'])
                galaxy_tree['%s' %ID_dict]['triax'].append(all_general['%s' %GalaxyID]['triax'])
                galaxy_tree['%s' %ID_dict]['disp_ani'].append(all_general['%s' %GalaxyID]['disp_ani'])
                galaxy_tree['%s' %ID_dict]['disc_to_total'].append(all_general['%s' %GalaxyID]['disc_to_total'])
                galaxy_tree['%s' %ID_dict]['rot_to_disp_ratio'].append(all_general['%s' %GalaxyID]['rot_to_disp_ratio'])
                # merger analysis
                galaxy_tree['%s' %ID_dict]['merger_ID'].append(merger_ID_array)
                galaxy_tree['%s' %ID_dict]['merger_ratio_stars'].append(merger_ratio_array)
                galaxy_tree['%s' %ID_dict]['merger_ratio_gas'].append(merger_gas_array)
                
                # time_step
                time_step   = 1e9 * abs(galaxy_tree['%s' %ID_dict]['Lookbacktime'][-1] - galaxy_tree['%s' %ID_dict]['Lookbacktime'][-2])
                
                #------------------        
                # Updating other
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if (hmr_i not in all_totmass['%s' %GalaxyID]['hmr']) or (disc_edge not in all_totmass['%s' %GalaxyID]['hmr']):
                        # Updating
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['tot_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['vcirc'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['tdyn'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['ttorque'].append(math.nan)
                        
                        
                    else:
                        # Creating masks. mask_disc = disc definition, mask_masses (SHOULD BE mask_rad IDEALLY) = simply the currnet rad
                        mask_disc = np.where(np.array(all_totmass['%s' %GalaxyID]['hmr']) == float(disc_edge))[0][0]
                        mask_masses  = np.where(np.array(all_totmass['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # if edge of SF disk larger than current hmr, we use R and M of current hmr. Else we use R and M of SF disk
                        if (disc_edge*float(all_general['%s' %GalaxyID]['halfmass_rad_sf'])) > (hmr_i*float(all_general['%s' %GalaxyID]['halfmass_rad'])):
                            R = hmr_i*float(all_general['%s' %GalaxyID]['halfmass_rad'])
                            M = float(all_totmass['%s' %GalaxyID]['mass'][mask_masses])             # totmass at hmr edge
                        else:
                            R = disc_edge*float(all_general['%s' %GalaxyID]['halfmass_rad_sf'])
                            M = float(all_totmass['%s' %GalaxyID]['mass_disc'][mask_disc])          # totmass at disk edge
                        
                        
                        # finding vcirc in [kms/s], tdyn in [Gyr]
                        vcirc = (1/1000 * np.sqrt(np.divide(np.array(M) * 2e30 * 6.67e-11, R * 3.09e19)))
                        tdyn  = 1e-9 * np.divide(2*np.pi * np.array(R) * 3.09e19, vcirc*1000) / 3.154e+7
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['tot_mass'].append(M)
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['vcirc'].append(vcirc)
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['tdyn'].append(tdyn)
                        galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['ttorque'].append(tdyn/all_general['%s' %GalaxyID]['ellip'])
                        
                
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
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['l'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                    
                        # Updating
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['stars'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['stars'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['stars'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['l'].append(all_l['%s' %GalaxyID]['stars'][mask_l])
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
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['l'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['gas'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['gas'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['sfr'].append((3.154e+7*all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr]))
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['gas'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['l'].append(all_l['%s' %GalaxyID]['gas'][mask_l])
                        galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['gas'][mask_spins], viewing_vector))
                        
                        
                        
                        #================================================
                        # Find inflow/outflow
                        # If gasdata_old does not have current hmr_i, update and move on. Else... find inflow/outflow
                        if '%s_hmr'%str(hmr_i) not in galaxy_tree['%s' %ID_dict]['gasdata_old'].keys():
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_mass'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_mass'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_mass'].append(math.nan)
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
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_mass'].append(inflow_mass)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].append(inflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].append(outflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_mass'].append(outflow_mass)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].append(outflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(stellarmassloss / time_step)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_mass'].append(stellarmassloss)
                            galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_Z'].append(insitu_Z)
                              
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
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['l'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_sfr    = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['gas_sf'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['gas_sf'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['sfr'].append((3.154e+7*all_sfr['%s' %GalaxyID]['gas_sf'][mask_sfr]))
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['gas_sf'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['l'].append(all_l['%s' %GalaxyID]['gas_sf'][mask_l])
                        galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['gas_sf'][mask_spins], viewing_vector))
                        
                        
                        
                        #================================================
                        # Find inflow/outflow
                        # If gasdata_old does not have current hmr_i, update and move on. Else... find inflow/outflow
                        if ('%s_hmr'%str(hmr_i) not in galaxy_tree['%s' %ID_dict]['gasdata_old'].keys()) or (int(galaxy_tree['%s' %ID_dict]['SnapNum'][-1]) - int(galaxy_tree['%s' %ID_dict]['SnapNum'][-2]) != 1):
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_mass'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_mass'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_mass'].append(math.nan)
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
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_mass'].append(inflow_mass)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].append(inflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].append(outflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_mass'].append(outflow_mass)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].append(outflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(stellarmassloss / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_mass'].append(stellarmassloss)
                            galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_Z'].append(insitu_Z)
                 
                #------------------                       
                # Updating gas_nsf
                galaxy_tree['%s' %ID_dict]['gas_nsf']['ap_mass'].append(all_general['%s' %GalaxyID]['gasmass_nsf'])
                galaxy_tree['%s' %ID_dict]['gas_nsf']['kappa'].append(all_general['%s' %GalaxyID]['kappa_gas_nsf'])
                for hmr_i in output_input['spin_hmr']:
                    # if this hmr_i not available
                    if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['count'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['l'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['proj_angle'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_mass'].append(math.nan)
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_Z'].append(math.nan)
                    else:
                        # Creating masks
                        mask_masses = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_Z      = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        mask_l      = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                        
                        # Updating
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['mass'].append(all_masses['%s' %GalaxyID]['gas_nsf'][mask_masses])
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['count'].append(all_counts['%s' %GalaxyID]['gas_nsf'][mask_counts])
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['Z'].append(all_Z['%s' %GalaxyID]['gas_nsf'][mask_Z])
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['l'].append(all_l['%s' %GalaxyID]['gas_nsf'][mask_l])
                        galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['gas_nsf'][mask_spins], viewing_vector))
                        
                        
                        
                        #================================================
                        # Find inflow/outflow
                        # If gasdata_old does not have current hmr_i, update and move on. Else... find inflow/outflow
                        if '%s_hmr'%str(hmr_i) not in galaxy_tree['%s' %ID_dict]['gasdata_old'].keys():
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_mass'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_mass'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(math.nan)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_mass'].append(math.nan)
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
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_mass'].append(inflow_mass)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].append(inflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].append(outflow_mass / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_mass'].append(outflow_mass)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].append(outflow_Z)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].append(stellarmassloss / time_step)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_mass'].append(stellarmassloss)
                            galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_Z'].append(insitu_Z)                                                               
                                                                                      
                #------------------                       
                # Updating dm   
                galaxy_tree['%s' %ID_dict]['dm']['ap_mass'].append(all_general['%s' %GalaxyID]['dmmass'])
                galaxy_tree['%s' %ID_dict]['dm']['count'].append(all_counts['%s' %GalaxyID]['dm'])
                galaxy_tree['%s' %ID_dict]['dm']['proj_angle'].append(_find_angle(all_spins['%s' %GalaxyID]['dm'][-1], viewing_vector))
                galaxy_tree['%s' %ID_dict]['dm']['l'].append(all_l['%s' %GalaxyID]['dm'][-1])
                
                #------------------                       
                # Updating bh                                                                 
                if np.isnan(all_general['%s' %GalaxyID]['bh_mass']) == True:
                    count_bh = 0
                else:
                    count_bh = 1                                                                      
                galaxy_tree['%s' %ID_dict]['bh']['mass'].append(all_general['%s' %GalaxyID]['bh_mass'])
                mdot = (float(galaxy_tree['%s' %ID_dict]['bh']['mass'][-1]) - float(galaxy_tree['%s' %ID_dict]['bh']['mass'][-2])) / time_step
                galaxy_tree['%s' %ID_dict]['bh']['mdot'].append(mdot)
                galaxy_tree['%s' %ID_dict]['bh']['mdot_instant'].append((3.154e+7*all_general['%s' %GalaxyID]['bh_mdot']))
                galaxy_tree['%s' %ID_dict]['bh']['edd'].append(all_general['%s' %GalaxyID]['bh_edd'])
                galaxy_tree['%s' %ID_dict]['bh']['lbol'].append((3.154e+7*all_general['%s' %GalaxyID]['bh_mdot']* (2e30 / 3.154e+7) * (0.1 * (3e8)**2) * (1e7)))
                galaxy_tree['%s' %ID_dict]['bh']['count'].append(count_bh)
                
                #------------------                       
                # Update angles 
                for angle_name, particle_names in zip(['stars_gas', 'stars_gas_sf', 'stars_dm', 'gas_dm', 'gas_sf_dm'], [['stars', 'gas'], ['stars', 'gas_sf'], ['stars', 'dm'], ['gas', 'dm'], ['gas_sf', 'dm']]):
                    for hmr_i in output_input['spin_hmr']:
                        # if this hmr_i not available
                        if hmr_i not in all_counts['%s' %GalaxyID]['hmr']:
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_abs'].append(math.nan)
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_abs'].append([math.nan, math.nan])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_proj'].append(math.nan)
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_proj'].append([math.nan, math.nan])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_halo'].append([math.nan, math.nan])
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_abs'].append(math.nan)
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['com_proj'].append(math.nan)
                            
                        else:
                            # Creating masks
                            mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            mask_angles_halo = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(hmr_i))[0][0]
                            
                            if angle_name not in ('stars_dm', 'gas_dm', 'gas_sf_dm'):
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
                            galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_halo'].append(all_misangleshalo['%s' %GalaxyID]['%s_angle' %angle_name][mask_angles_halo])
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
            galaxy_tree['%s' %ID_dict]['ap_sfr'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['rad'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['radproj'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['rad_sf'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['ellip'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['triax'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['disp_ani'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['disc_to_total'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['rot_to_disp_ratio'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['merger_ID'].insert(index, [])
            galaxy_tree['%s' %ID_dict]['merger_ratio_stars'].insert(index, [])
            galaxy_tree['%s' %ID_dict]['merger_ratio_gas'].insert(index, [])
            
            #------------------                       
            # Updating other
            for hmr_i in output_input['spin_hmr']:
                galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['tot_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['vcirc'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['tdyn'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['other']['%s_hmr' %hmr_i]['ttorque'].insert(index, math.nan)
                
            #------------------                       
            # Updating stars
            galaxy_tree['%s' %ID_dict]['stars']['tot_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['stars']['ap_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['stars']['kappa'].insert(index, math.nan)
            for hmr_i in output_input['spin_hmr']:
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['count'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['stars']['%s_hmr' %hmr_i]['l'].insert(index, math.nan)
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
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['l'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['proj_angle'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['inflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['outflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['stelmassloss_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas']['%s_hmr' %hmr_i]['insitu_mass'].insert(index, math.nan)
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
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['l'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['proj_angle'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['inflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['outflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['stelmassloss_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_sf']['%s_hmr' %hmr_i]['insitu_Z'].insert(index, math.nan)
                
            #------------------                       
            # Updating gas_nsf
            galaxy_tree['%s' %ID_dict]['gas_nsf']['ap_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['gas_nsf']['kappa'].insert(index, math.nan)
            for hmr_i in output_input['spin_hmr']:
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['count'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['l'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['proj_angle'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['inflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['outflow_Z'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['stelmassloss_rate'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_mass'].insert(index, math.nan)
                galaxy_tree['%s' %ID_dict]['gas_nsf']['%s_hmr' %hmr_i]['insitu_Z'].insert(index, math.nan)
                                                                                  
            #------------------                       
            # Updating dm   
            galaxy_tree['%s' %ID_dict]['dm']['ap_mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['dm']['count'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['dm']['proj_angle'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['dm']['l'].insert(index, math.nan)
            
            #------------------                       
            # Updating bh                                                                         
            galaxy_tree['%s' %ID_dict]['bh']['mass'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['mdot'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['mdot_instant'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['edd'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['lbol'].insert(index, math.nan)
            galaxy_tree['%s' %ID_dict]['bh']['count'].insert(index, math.nan)
            
            #------------------                       
            # Update angles 
            for angle_name, particle_names in zip(['stars_gas', 'stars_gas_sf', 'stars_dm', 'gas_dm', 'gas_sf_dm'], [['stars', 'gas'], ['stars', 'gas_sf'], ['stars', 'dm'], ['gas', 'dm'], ['gas_sf', 'dm']]):
                for hmr_i in output_input['spin_hmr']:
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_abs'].insert(index, math.nan)
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_abs'].insert(index, [math.nan, math.nan])
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_proj'].insert(index, math.nan)
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['err_proj'].insert(index, [math.nan, math.nan])
                    galaxy_tree['%s' %ID_dict][angle_name]['%s_hmr' %hmr_i]['angle_halo'].insert(index, math.nan)
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
# SAVED: does not save outputs currently
ID_list = [108988077, 479647060, 21721896, 390595970, 401467650, 182125463, 192213531, 24276812, 116404995, 239808134, 215988755, 86715463, 6972011, 475772617, 374037507, 429352532, 441434976]
ID_list = [1361598, 1403994, 10421872, 17879310, 21200847, 21659372, 24053428, 182125501, 274449295]
ID_list = [21200847, 182125516, 462956141]
def _analyse_tree(csv_tree = 'L100_galaxy_tree_',
                  #--------------------------
                  # Galaxy analysis
                  print_summary             = True,
                  print_galaxy              = False,
                    plot_misangle_detection = False,    # toggle show and save below
                  print_checks              = False,
                  #-----------------------
                  # Individual galaxies
                  force_all_snaps = True,                # KEEP TRUE. force us to use galaxies in which we have all snaps.
                    GalaxyID_list = None,             # [ None / ID_list ]
                  #====================================================================================================
                  # Misalignment must take place in range   
                    max_z = 1.0,                  
                    min_z = None,                        # [ None / value ] min/max z to sample galaxies BECOMING MISALIGNED
                  # Radii to extract
                    use_hmr_general     = '2.0',    # [ 1.0 / 2.0 / aperture]      Used for stelmass | APERTURE NOT AVAILABLE FOR sfr ssfr
                  #------------------------------------------------------------
                  # PROPERTIES TO ALWAYS MEET
                    min_particles     = 20,              # [count]
                    max_com           = 2,             # [pkpc]
                    max_uncertainty   = 30,            # [ None / 30 / 45 ]                  Degrees
                  #------------------------------------------------------------
                  # PROPERTIES TO AVERAGE OVER WHILE MISALIGNED / RELAXING
                  # Satellites, central, or both         [ None / 'satellite' is sgn >= 1 / 'central' is sgn == 1 / None ]
                    limit_satellites    = None,
                  # Group / Field / both                 [ None / value ] (halo threshold: 10**14)
                    min_halomass        = None,     max_halomass        = None, 
                  # Masses and                           [ None / value ]
                    min_stelmass        = None,     max_stelmass        = None,
                    min_gasmass         = None,     max_gasmass         = None,
                    min_sfmass          = None,     max_sfmass          = None,
                    min_nsfmass         = None,     max_nsfmass         = None,
                  # SF and quiescent galaxies            [ None / value ]
                    min_sfr             = None,     max_sfr             = None,     # [ Msun/yr ] SF limit of ~ 0.1
                    min_ssfr            = None,     max_ssfr            = None,     # [ /yr ] SF limit of ~ 1e-10-11
                  # Morphology stars / gas.              [ None / value ] 
                    min_kappa_stars     = None,     max_kappa_stars     = None,     
                    min_kappa_gas       = None,     max_kappa_gas       = None,     # Take these with pinch of salt though
                    min_kappa_sf        = None,     max_kappa_sf        = None,     # >0.7 broadly shows disk per jimenez
                    min_kappa_nsf       = None,     max_kappa_nsf       = None,
                    min_ellip           = None,     max_ellip           = None,     # 0 is perfect sphere, 1 is flat.
                    min_triax           = None,     max_triax           = None,     # Lower value = oblate, high = prolate
                  # Radius limits                        [ None / value ]
                    min_rad             = None,     max_rad             = None,   
                    min_rad_sf          = None,     max_rad_sf          = None,   
                  # l not added
                    # l of stars not added yet = None
                  # Inflow (gas) | 7.0 is pretty high    [ None / value ]
                    min_inflow          = None,     max_inflow          = None,
                  # Metallicity of inflowing gas         [ None / value ]
                    min_inflow_Z        = None,     max_inflow_Z        = None,
                  # BH                                   [ None / value ]
                    force_steady_bh     = False,                                     # NOT WORKING
                    min_bh_mass         = None,     max_bh_mass         = None,
                    min_bh_acc          = None,     max_bh_acc          = None,     # [ Msun/yr ] Uses averaged rate over snapshots
                    min_bh_acc_instant  = None,     max_bh_acc_instant  = None,     # [ Msun/yr ] Uses instantaneous accretion rate
                    min_edd             = None,     max_edd             = None,     # Uses instantaneous accretion rate
                    min_lbol            = None,     max_lbol            = None,     # [ erg/s ] 10^44 good cutoff for AGN. Uses averaged rate over snapshots
                  # Dynamics
                    min_vcirc           = None,     max_vcirc           = None,     # [ km/s ] 200 is roughly ETG
                    min_tdyn            = None,     max_tdyn            = None,     # [ None / Gyr ] Min/max dynamical time
                    min_ttorque         = None,     max_ttorque         = None,     # [ None / Gyr ] Min/max torquing time
                  #------------------------------------------------------------
                  # Misalignment angles                
                    use_hmr_angle        = 1.0,           # [ 1.0 / 2.0 ]                Used for misangle, inc angle, com, counts
                    abs_or_proj          = 'abs',         # [ 'abs' / 'proj' ]
                    min_inclination      = 0,              # [ 0 / degrees]
                    use_angle            = 'stars_gas_sf',
                    misangle_threshold   = 20,            # [ 20 / 30 / 45 ]  Must meet this angle (or 180-misangle_threshold) to be considered misaligned
                    min_delta_angle      = 0,            # [ None / deg ] 10    Change in angle between successive snapshots from aligned to misaligned
                  #------------------------------------------------------------
                  # Mergers 
                  use_merger_criteria   = False,   # [ True / None / False ] Whether we limit to merger-induced, no mergers, or any misalignments
                    min_stellar_ratio   = 0.1,       max_stellar_ratio   = 1/0.1,     # [ value ] -> set to 0 if we dont care, set to 999 if we dont care
                    min_gas_ratio       = None,      max_gas_ratio       = None,    # [ None / value ]
                    max_merger_pre      = 0.2,       max_merger_post     = 0.2,    # [0.2 + 0.5 / Gyr] -/+ max time to closest merger from point of misalignment
                  #------------------------------------------------------------
                  # Temporal selection
                    latency_time     = 0.1,          # [ None / 0.1 Gyr ]   Consecutive time galaxy must be <30 / >150 to count as finished relaxing
                    time_extra       = 0.1,      # [Gyr] 0.1     extra time before and after misalignment which is also extracted
                    time_no_misangle = 0.1,     # [Gyr] 0.1         extra time before and after misalignment which has no misalignments. Similar to relax snapshots
                  #====================================================================================================
                  
                  
                  # Relaxation selection
                    relaxation_type    = ['co-co', 'counter-counter', 'co-counter', 'counter-co', ],        # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                    relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],           # ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'] based off initial vs end kappa
                    misalignment_morph = ['LTG', 'ETG'],                                # ['ETG', 'LTG', 'other'] averages kappa over misalignment. Uses 0.35, 0.45 means as cutoff
                      morph_limits     = [0.4, 0.4],                                                    # [ upper ETG, lower LTG ] bounds
                    peak_misangle      = 30,          # [ None / angle ] Maximum delta from where the galaxy relaxes to. So for co = 0, counter = 180 
                    min_trelax         = None,        
                    max_trelax         = None,        # [ None / Gyr ] Min/max relaxation time
                  #--------------------------
                  csv_file       = False,             # Will write sample to csv file in sample_dir
                    csv_name     = '_',               # extra stuff at end
                  
                  
                  #====================================================================================================
                  # Mergers
                  # Processed after sample, applied to first index_misaligned rather than entire misalignment
                  use_alt_merger_criteria = False,
                    half_window         = 0.3,      # [ 0.2 / +/-Gyr ] window centred on first misaligned snap to look for mergers
                    min_ratio           = 0.1,   
                    merger_lookback_time = 2,       # Gyr, number of years to check for peak stellar mass
                    
                  use_alt_relaxation_morph   = False,     # False / ['ETG-ETG'],
                  
                  use_alt_min_trelax    = False,    # [ False / Gyr ]
                  use_alt_min_tdyn      = False,
                  use_alt_min_ttorque   = False,
                  #====================================================================================================
                    
                  # Plot histogram of sample we ended up selecting
                  _plot_sample_hist             = False,
                    set_bin_width_mass                  = 0.01,      # 0.1
                    
                  #-----------------------------
                  # Plot timescale histogram with current sample
                  _plot_timescale_histogram     = False,   
                    set_bin_limit_trelax                = 5,        # [ None / Gyr ]
                    set_bin_width_trelax                = 0.25,     # [ 0.25 / Gyr ]
                      set_plot_percentage               = False,
                      set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_inset                       = True,     # whether to have smaller second plot
                        add_inset_bestfit               = True,
                    set_thist_ymax_trelax               = 0.45,             # 0.45 / 500  yaxis max
                  _plot_tdyn_histogram          = False,
                    set_bin_limit_tdyn                  = 32,       # [ None / multiples ]
                    set_bin_width_tdyn                  = 1,        # [ multiples ]
                    set_thist_ymax_tdyn                 = 0.35,             # 0.35 / 400 yaxis max
                  _plot_ttorque_histogram       = False,
                    set_bin_limit_ttorque               = 12,       # [ None / multiples ]
                    set_bin_width_ttorque               = 0.5,      # [ multiples ]
                    set_thist_ymax_ttorque              = 0.35,             # 0.35 / 400 yaxis max
                
                  #-----------------------------
                  # Plot stacked misalignments based on current sample
                  _plot_stacked_trelax          = False,           # single
                  _plot_stacked_trelax_2x2      = False,           # 2x2 subplot of co-co, co-counter, ...
                    set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                    set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                    set_plot_extra_time                 = False,              # Plot extra time after relaxation
                    set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                    set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                    set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                      add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                      set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                  _plot_stacked_tdyn            = False,
                  _plot_stacked_tdyn_2x2        = False,
                  _plot_stacked_ttorque         = False,
                  _plot_stacked_ttorque_2x2     = False,
                
                  #-----------------------------
                  # Plots relaxation type vs relaxation time for ETG and LTG.
                  _plot_box_and_whisker_trelax  = False,     # Plots relaxation type vs relaxation time for ETG and LTG. USes relaxation_type and relaxation_morph
                    set_whisker_morphs                  = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                  _plot_box_and_whisker_tdyn    = False,
                  _plot_box_and_whisker_ttorque = False,
                  
                  #-----------------------------
                  # Will plot misalignments within an offset range (peak misangle) for ETGs and LTGs
                  _plot_offset_trelax           = False,                
                  _plot_offset_morph_trelax     = False,                
                    set_offset_morphs                   = ['LTG-LTG', 'ETG-ETG'],           # Can be either relaxation_morph or misalignment_morph
                    set_plot_offset_range               = [30, 150],                         # [min angle , max angle] of peak misangle
                    set_plot_offset_type                = ['co-co'],                     # [ 'co-co', 'co-counter' ]  or False
                      set_plot_offset_log               = True,
                  _plot_offset_tdyn             = False,                
                  _plot_offset_morph_tdyn       = False,     
                  _plot_offset_ttorque          = False,           
                  _plot_offset_morph_ttorque    = False,        
                    
                  #-----------------------------
                  # Will plot scatter of number of mergers with relaxation time, suited for 1010 sample and above
                  _plot_merger_count_trelax     = False,
                    set_min_merger_trelax               = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                    set_plot_merger_count_lim           = 0.1,          # stellar ratio (will also pick reciprocal)
                    add_plot_merger_count_gas           = True,         # will colour by gas ratio
                      set_plot_merger_count_log         = True,
                  _plot_merger_count_tdyn       = False,
                    set_min_merger_tdyn                 = 0,          # [ trelax/dyn ] min relaxation time, as we dont care about short relaxers
                  _plot_merger_count_ttorque    = False,
                    set_min_merger_ttorque              = 0,          # [ trelax/torque ] min relaxation time, as we dont care about short relaxers
                    
                  #-----------------------------
                  # Average halo misalignment with relaxtime
                  _plot_halo_misangle_trelax    = False,
                    set_halo_trelax_resolution          = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                    set_min_halo_trelax                 = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                      add_plot_halo_morph_median        = True,
                      set_plot_halo_misangle_log        = True,
                  _plot_halo_misangle_tdyn      = False,
                    set_min_halo_tdyn                   = 0,          # [ trealx/dyn ] min relaxation time, as we dont care about short relaxers
                  _plot_halo_misangle_ttorque   = False,
                    set_min_halo_ttorque                = 0,          # [ trelax/torque ] min relaxation time, as we dont care about short relaxers
                  _plot_halo_misangle_manual    = False,
                    
                  #-----------------------------
                  # Average halo misalignment before misalignment with fractional occurence
                  _plot_halo_misangle_pre_frac  = False, 
                    set_misanglepre_morph               = True,         # will use a stacked histogram for ETG-ETG, LTG-LTG, ETG-LTG, etc.
                    set_misanglepre_type                = ['co-co', 'co-counter'],           # [ 'co-co', 'co-counter' ]  or False
                  
                  #-----------------------------
                  # Plots stacked bar chart. Will use use_alt_merger_criteria from earlier to find mergers
                  _plot_origins                 = False,
                    use_only_start_morph                = True,          # use only ETG - and LTG - 
                    set_origins_morph                   = ['ETG-ETG', 'ETG-LTG', 'LTG-ETG', 'LTG-LTG'],
                    
                  #-----------------------------
                  _plot_timescale_gas_histogram = False,
                    set_gashist_type                    = ['co-co'],
                    set_gashist_min_trelax              = 0.5,
                      add_plot_gas_morph_median         = False,
                  
                  
                  #-----------------------------
                  # General formatting
                  showfig       = True,
                  savefig       = False,    
                    file_format = 'pdf',
                    savefig_txt = 'txt',     # [ 'manual' / txt ] 'manual' will prompt txt before saving
              #====================================================================================================
              load_csv_file  = '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph',     # [ 'file_name' / False ] load existing misalignment tree                                                                     '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph'                                                                                                                           '_20Thresh_30Peak_normalLatency_anyMergers_hardMorph'                                                                                                            '_20Thresh_30Peak_normalLatency_anyMergers_ETG-ETG'                                                                                                                         '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010' 
              
                plot_annotate  = False,    #  [ False / 'ETG → ETG' r'ETG ($\bar{\kappa}_{\mathrm{co}}^{\mathrm{*}} < 0.35$)'  ]      '$t_{\mathrm{relax}}>3\bar{t}_{\mathrm{torque}}$'             # string of text or False / 'ETG' 
              #====================================================================================================
                  print_progress = False,
                  debug = False):
                  
    
    #================================================ 
    assert (answer == '4') or (answer == '3'), 'Must use snips'
    for i in relaxation_type:
        assert (i == 'co-co') or (i == 'co-counter') or (i == 'counter-co') or (i == 'counter-counter'), 'Incorrect relaxation_type'
    
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
    
    # If loading misalignment_tree(), don't do the whole analysis bit:
    if not load_csv_file:
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
    
    
        #==================================================================================================
        # Loop over all galaxies
        misalignment_tree = {}
        for GalaxyID in tqdm(galaxy_tree.keys()):
        
        
            #=========================================================================
            # CHECK 1: checking if there are any misalignments in range at all
            # If we are looking at individual galaxies, filter them out
            if GalaxyID_list != None:
                if int(GalaxyID) not in GalaxyID_list_extract:
                    continue
            if print_checks:
                print('\n\nID: ', GalaxyID)
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('QUICK FILTERS')
                time_start = time.time()
        
            # Test if any misalignments even present in z range, else move on
            mask_test_z = np.where((np.array(galaxy_tree['%s' %GalaxyID]['Redshift']) >= (-1 if min_z == None else min_z)) & (np.array(galaxy_tree['%s' %GalaxyID]['Redshift']) <= (999 if max_z == None else max_z)))[0]
            if len(mask_test_z) == 0:
                continue
            mask_test_z = np.arange(mask_test_z[0], mask_test_z[-1]+1)
            mask_test_z_misangle = np.where(np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj])[mask_test_z] > misangle_threshold)
        
            if plot_misangle_detection:
                plt.close()
            
                ### Create figure
                fig, axs = plt.subplots(1, 1, figsize=[8, 5], sharex=True, sharey=False)
                plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
                # Plot mergers
                for time_i, ratio_i, gas_i in zip(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], galaxy_tree['%s' %GalaxyID]['merger_ratio_gas']):
                    #print('%.1f\t%.2f' %(time_i, max(ratio_i, default=math.nan)))
                    if len(ratio_i) > 0:
                        if max(ratio_i) > 0.01:
                            axs.axvline(time_i, c='grey', ls='--', lw=1)
                            axs.text(time_i-0.2, 175, '%.2f' %max(ratio_i), color='grey', fontsize=8, zorder=999)
                            axs.text(time_i-0.2, 170, '%.2f' %gas_i[np.argmax(ratio_i)], color='blue', fontsize=8, zorder=999)
                axs.plot(galaxy_tree['%s' %GalaxyID]['Lookbacktime'], galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj], 'ko-', mec='k', lw=0.9, ms=1)
                axs.text(8, 196, 'ID: %s' %GalaxyID, fontsize=8)
                axs.text(8, 190, '%s' %(use_angle), fontsize=8, color='grey')
                axs.text(8, 184, '%s' %(abs_or_proj), fontsize=8, color='grey')
                axs.axhspan(0, misangle_threshold, alpha=0.25, ec=None, fc='grey')
                axs.axhspan(180-misangle_threshold, 180, alpha=0.25, ec=None, fc='grey')
                axs.set_ylim(0, 180)
                axs.set_xlim(8.1, -0.1)
                axs.set_xticks(np.arange(8, -1, -1))
                axs.set_yticks(np.arange(0, 181, 30))
                axs.set_xlabel('Lookback-time (Gyr)')
                axs.set_ylabel('Misalignment angle')
                #axs.minorticks_on()
                #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
                #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
            
        
            #>>>>>>>>>>>>>>>>>>>>>
            if len(mask_test_z_misangle) == 0:
                if print_checks:
                    print('x FAILED CHECK 1: no misalignments in z range: %s - %s' %(max_z, min_z))
                if plot_misangle_detection:
                    if showfig:
                        plt.show()
                continue
            elif print_checks:
                print('  CHECK 1: %s misalignments in z range: %s - %s' %(len(mask_test_z_misangle), max_z, min_z))
            
        
            if print_galaxy:
                print('ID: ', GalaxyID)
                print('In range %s - %s:\nID      Index\tSnap\tz\tTime\tAngLo\tAngle\tAngHi\tRatio\tStelmass' %(max_z, min_z))
                for ID_ii, index, snap_i, time_i, z_i, angle_i, err_i, merger_i, stars_i in zip(np.array(galaxy_tree['%s' %GalaxyID]['GalaxyID'])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['SnapNum'])[mask_test_z] - np.array(galaxy_tree['%s' %GalaxyID]['SnapNum'])[0], np.array(galaxy_tree['%s' %GalaxyID]['SnapNum'])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['Lookbacktime'])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['Redshift'])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj])[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'], dtype=object)[mask_test_z], np.array(galaxy_tree['%s' %GalaxyID]['stars']['ap_mass'])[mask_test_z]):
                    print('%s  %s\t%s\t%.2f\t%.2f\t%.1f\t%.1f\t%.1f\t%.2f\t%.1f' %(ID_ii, index, snap_i, z_i, time_i, err_i[0], angle_i, err_i[1], max(merger_i, default=0), np.log10(stars_i)))
                print(' ')
            #=========================================================================
        
        
            #=========================================================================
            # CHECK 2: establishing a complete window for misalignment (pre-post at z)
            # Identify indexes of start and ends of individual relaxations
            index_dict = {'misalignment_locations': {'misalign': {'index': [],
                                                                  'snapnum': []},
                                                     'relax':    {'index': [],
                                                                  'snapnum': []}}}
            index_dict.update({'plot_height': []})
        
            all_angles = galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj]
            misalignment_started = False
            index_ahead = math.nan
            for index, snap_i, angle_i in zip(np.arange(0, len(all_angles)+1), galaxy_tree['%s' %GalaxyID]['SnapNum'], all_angles):
        
                # If there is a nan inbetween, reset count
                if np.isnan(angle_i) == True:
                    misalignment_started = False
        
                if index < (len(all_angles)-1):
                    # Check for start of misalignment that meets conditions
                
                    # Check for co state
                    if (misalignment_started == False) & (angle_i < misangle_threshold) & (all_angles[index+1] > misangle_threshold) & (abs(all_angles[index+1] - angle_i) >= (0 if min_delta_angle == None else min_delta_angle)) & (galaxy_tree['%s' %GalaxyID]['Redshift'][index] >= (-1 if min_z == None else min_z)) & (galaxy_tree['%s' %GalaxyID]['Redshift'][index] <= (999 if max_z == None else max_z)):
                        misalignment_started = True
                        misalignment_started_index = index
                        misalignment_started_snap  = snap_i
                    # Check for counter state
                    elif (misalignment_started == False) & (angle_i > (180-misangle_threshold)) & (all_angles[index+1] < (180-misangle_threshold)) & (abs(all_angles[index+1] - angle_i) >= (0 if min_delta_angle == None else min_delta_angle)) & (galaxy_tree['%s' %GalaxyID]['Redshift'][index] >= (-1 if min_z == None else min_z)) & (galaxy_tree['%s' %GalaxyID]['Redshift'][index] <= (999 if max_z == None else max_z)):
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
                                index_dict['misalignment_locations']['misalign']['index'].append(misalignment_started_index )
                                index_dict['misalignment_locations']['misalign']['snapnum'].append(misalignment_started_snap)
                                index_dict['misalignment_locations']['relax']['index'].append((index+1))
                                index_dict['misalignment_locations']['relax']['snapnum'].append(galaxy_tree['%s' %GalaxyID]['SnapNum'][index+1])
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
                                index_dict['misalignment_locations']['misalign']['index'].append(misalignment_started_index )
                                index_dict['misalignment_locations']['misalign']['snapnum'].append(misalignment_started_snap)
                                index_dict['misalignment_locations']['relax']['index'].append((index+1))
                                index_dict['misalignment_locations']['relax']['snapnum'].append(galaxy_tree['%s' %GalaxyID]['SnapNum'][index+1])
                                misalignment_started = False
                    
                        else:
                            continue
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('done')
                time_start = time.time()
            if debug:
                print(index_dict['misalignment_locations'].items())
           
            # Optionally plot detection
            if print_galaxy:
                print('MISALIGNMENTS:')
                if len(index_dict['misalignment_locations']['misalign']['index']) > 0:
                    print('Snap -\tSnap\tTime -\tTime\tDuration [Gyr]')
                    for index_m, index_r, snap_m, snap_r in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], index_dict['misalignment_locations']['misalign']['snapnum'], index_dict['misalignment_locations']['relax']['snapnum']):
                        print('%s\t%s\t%.2f\t%.2f\t%.10f' %(snap_m, snap_r, galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r], abs(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m])))
                else:
                    print('\n> No misalignments using imposed limits <')
                print(' ')
            # Green and orange bar, grey bars
            if plot_misangle_detection:
                # Plot misalignment detections
                axs.text(6.5, 196, 'Detected misalignments: %s' %len(index_dict['misalignment_locations']['misalign']['index']), fontsize=8, color='green')
                axs.text(6.5, 190, 'Latency period: %s Gyr' %latency_time, fontsize=8, color='orange')
            
                for index_m, index_r, plot_height in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], np.linspace(180-misangle_threshold-10, misangle_threshold+10, len(index_dict['misalignment_locations']['misalign']['index']))):
                
                    index_dict['plot_height'].append(plot_height)
                
                    axs.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]], [plot_height, plot_height], lw=10, color='green', solid_capstyle="butt", alpha=0.4)
                    axs.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-(0 if latency_time==None else latency_time)], [plot_height, plot_height], lw=10, color='orange', solid_capstyle="butt", alpha=0.4)
                    axs.text(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], plot_height-2, '%.2f' %abs(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m]), fontsize=9)
                    axs.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m], ymin=plot_height-7, ymax=plot_height+7, lw=1.5, colors='k', alpha=0.3)
                    axs.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r], ymin=plot_height-7, ymax=plot_height+7, lw=1.5, colors='k', alpha=0.3)
                
   
            #>>>>>>>>>>>>>>>>>>>>>
            # Skip galaxy if no misalignments detected
            if len(index_dict['misalignment_locations']['misalign']['index']) == 0:
                if print_checks:
                    print('x FAILED CHECK 2: no misalignments meeting algorithm criteria: %s°, latency time: %s Gyr' %(misangle_threshold, latency_time))
                if plot_misangle_detection:
                    if showfig:
                        plt.show()
                continue
            elif print_checks:
                print('  CHECK 2: %s misalignments meeting algorithm criteria: %s°, latency time: %s Gyr' %(len(index_dict['misalignment_locations']['misalign']['index']), misangle_threshold, latency_time))
            
            #=========================================================================
        
        
            #=========================================================================
            # CHECK 3: Ensure we have window meeting minimum time of time_extra. Will throw out if window not complete
            # For each misalignment and relaxation PAIR, find hypothetical snapnums and indexes either side
            index_dict.update({'window_locations': {'misalign': {'index': [],
                                                                 'snapnum': []},
                                                    'relax':    {'index': [],
                                                                 'snapnum': []}}})
            for index_m, index_r, snap_m, snap_r in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], index_dict['misalignment_locations']['misalign']['snapnum'], index_dict['misalignment_locations']['relax']['snapnum']):
                # Find indexes 
                time_extra_snap_m = np.where(Lookbacktime_tree >= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m] + (0 if time_extra == None else time_extra)))[0][-1]
        
                if len(np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r] - (0 if time_extra == None else time_extra)))[0]) == 0:
                    index_dict['window_locations']['misalign']['index'].append(math.nan)
                    index_dict['window_locations']['relax']['index'].append(math.nan)
                    index_dict['window_locations']['misalign']['snapnum'].append(math.nan)
                    index_dict['window_locations']['relax']['snapnum'].append(math.nan)
                    continue
                else:
                    time_extra_snap_r = np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r] - (0 if time_extra == None else time_extra)))[0][0]
                if debug:
                    print(' ', time_extra)
                    print(Lookbacktime_tree[time_extra_snap_m])
                    print(Lookbacktime_tree[time_extra_snap_r])
                    print(time_extra_snap_m)
                    print(time_extra_snap_r)
            
                # Find snap window
                snap_window = np.arange(time_extra_snap_m, time_extra_snap_r+1)
            
                # Check if this window exists in entirety within data we have available
                if set(snap_window).issubset(set(galaxy_tree['%s' %GalaxyID]['SnapNum'])) == False:
                    index_dict['window_locations']['misalign']['index'].append(math.nan)
                    index_dict['window_locations']['relax']['index'].append(math.nan)
                    index_dict['window_locations']['misalign']['snapnum'].append(math.nan)
                    index_dict['window_locations']['relax']['snapnum'].append(math.nan)
                    continue
                else:
                    # Find indexes of window
                    time_extra_index_m, time_extra_index_r = np.where((np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) == time_extra_snap_m) | (np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) == time_extra_snap_r))[0]
            
                    index_dict['window_locations']['misalign']['index'].append(time_extra_index_m)
                    index_dict['window_locations']['relax']['index'].append(time_extra_index_r)
                    index_dict['window_locations']['misalign']['snapnum'].append(time_extra_snap_m)
                    index_dict['window_locations']['relax']['snapnum'].append(time_extra_snap_r)
        
            # Black bars = extract data
            if plot_misangle_detection:
                # Plot misalignment detections
                axs.text(6.5, 184, 'Sample window: %s Gyr' %time_extra, fontsize=8, color='k')
            
                for index_m, index_r, index_window_m, index_window_r, plot_height in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], index_dict['window_locations']['misalign']['index'], index_dict['window_locations']['relax']['index'], index_dict['plot_height']):
                
                    # If its one for which no window exists, skip
                    if np.isnan(index_window_m) == True:
                        continue
                
                    # Plot line from window edge to mis start, and from mis end to window edge
                    axs.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_window_m], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m]], [plot_height, plot_height], lw=1, color='k', solid_capstyle="butt", alpha=0.8)
                    axs.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_window_r]], [plot_height, plot_height], lw=1, color='k', solid_capstyle="butt", alpha=0.8)
                    axs.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_window_m], ymin=plot_height-10, ymax=plot_height+10, lw=2, colors='k', alpha=0.8)
                    axs.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_window_r], ymin=plot_height-10, ymax=plot_height+10, lw=2, colors='k', alpha=0.8)
                
                
            #>>>>>>>>>>>>>>>>>>>>>
            # Skip galaxy if no windows available for misalignments
            if len(index_dict['window_locations']['misalign']['index']) == 0:
                if print_checks:
                    print('x FAILED CHECK 3: incomplete window for ± %s Gyr' %(time_extra))
                if plot_misangle_detection:
                    if showfig:
                        plt.show()
                continue
            elif np.isnan(np.array(index_dict['window_locations']['misalign']['index'])).all() == True:
                if print_checks:
                    print('x FAILED CHECK 3: incomplete window for ± %s Gyr' %(time_extra))
                if plot_misangle_detection:
                    if showfig:
                        plt.show()
                continue
            elif print_checks:
                print('  CHECK 3: %s misalignments matching window for ± %s Gyr' %(np.count_nonzero(~np.isnan(index_dict['window_locations']['misalign']['index'])), time_extra))
            #=========================================================================
        
            
            #=========================================================================
            # CHECK 4: Ensure we have gap between misalignments detected. Will throw out if window not complete
            index_dict.update({'windowgap_locations': {'misalign': {'index': [],
                                                                    'snapnum': []},
                                                       'relax':    {'index': [],
                                                                    'snapnum': []}}})
            for index_m, index_r, snap_m, snap_r, index_window_m, index_window_r, snap_window_m, snap_window_r in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], index_dict['misalignment_locations']['misalign']['snapnum'], index_dict['misalignment_locations']['relax']['snapnum'], index_dict['window_locations']['misalign']['index'], index_dict['window_locations']['relax']['index'], index_dict['window_locations']['misalign']['snapnum'], index_dict['window_locations']['relax']['snapnum']):
            
                # If no window available, skip
                if np.isnan(snap_window_m) == True:
                    index_dict['windowgap_locations']['misalign']['index'].append(math.nan)
                    index_dict['windowgap_locations']['relax']['index'].append(math.nan)
                    index_dict['windowgap_locations']['misalign']['snapnum'].append(math.nan)
                    index_dict['windowgap_locations']['relax']['snapnum'].append(math.nan)
                    continue
                
                # Find indexes 
                time_extra_snap_m = np.where(Lookbacktime_tree >= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m] + (0 if time_no_misangle == None else time_no_misangle)))[0][-1]
        
                if len(np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r] - (0 if time_no_misangle == None else time_no_misangle)))[0]) == 0:
                    index_dict['window_locations']['misalign']['index'].append(math.nan)
                    index_dict['window_locations']['relax']['index'].append(math.nan)
                    index_dict['window_locations']['misalign']['snapnum'].append(math.nan)
                    index_dict['window_locations']['relax']['snapnum'].append(math.nan)
                    continue
                else:
                    time_extra_snap_r = np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r] - (0 if time_no_misangle == None else time_no_misangle)))[0][0]
            
                # Find snap window
                snap_window = np.arange(time_extra_snap_m, time_extra_snap_r+1)
            
                # Check if this window exists in entirety within data we have available
                if set(snap_window).issubset(set(galaxy_tree['%s' %GalaxyID]['SnapNum'])) == False:
                    index_dict['windowgap_locations']['misalign']['index'].append(math.nan)
                    index_dict['windowgap_locations']['relax']['index'].append(math.nan)
                    index_dict['windowgap_locations']['misalign']['snapnum'].append(math.nan)
                    index_dict['windowgap_locations']['relax']['snapnum'].append(math.nan)
                    continue
                else:
                    # Find indexes of window
                    time_extra_index_m, time_extra_index_r = np.where((np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) == time_extra_snap_m) | (np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) == time_extra_snap_r))[0]
                
                    # Loop from start of window to begin of misalignment to ensure no misalignments occur
                    nomisangle = []
                    for index_i in np.arange(time_extra_index_m, index_m+1):
                        # If it started at 30, ensure it stays in 30. If it started at 150, ensure it stays in 150
                        if galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_m] < misangle_threshold:
                            if galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_i] < misangle_threshold:
                                nomisangle.append(True)
                            else:
                                nomisangle.append(False)
                        elif galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_m] > (180-misangle_threshold):
                            if galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_i] > (180-misangle_threshold):
                                nomisangle.append(True)
                            else:
                                nomisangle.append(False)
                    
                        #if galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_i] < misangle_threshold:
                        #    nomisangle.append(True)
                        #else:
                        #    nomisangle.append(False)
                    
                    # if pre criterion met... try post criterion
                    if np.array(nomisangle).all() == True:
                        nomisangle = []
                        for index_i in np.arange(index_r, time_extra_index_r+1):
                            # If relax into 30, ensure it stays in 30. If it relax into 150, ensure it stays in 150
                            if galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_r] < misangle_threshold:
                                if galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_i] < misangle_threshold:
                                    nomisangle.append(True)
                                else:
                                    nomisangle.append(False)
                            elif galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_r] > (180-misangle_threshold):
                                if galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_i] > (180-misangle_threshold):
                                    nomisangle.append(True)
                                else:
                                    nomisangle.append(False)
                        # if post criterion met... append
                        if np.array(nomisangle).all() == True: 
                            index_dict['windowgap_locations']['misalign']['index'].append(time_extra_index_m)
                            index_dict['windowgap_locations']['relax']['index'].append(time_extra_index_r)
                            index_dict['windowgap_locations']['misalign']['snapnum'].append(time_extra_snap_m)
                            index_dict['windowgap_locations']['relax']['snapnum'].append(time_extra_snap_r)
                        else:
                            index_dict['windowgap_locations']['misalign']['index'].append(math.nan)
                            index_dict['windowgap_locations']['relax']['index'].append(math.nan)
                            index_dict['windowgap_locations']['misalign']['snapnum'].append(math.nan)
                            index_dict['windowgap_locations']['relax']['snapnum'].append(math.nan)
                            continue
                    else:
                        index_dict['windowgap_locations']['misalign']['index'].append(math.nan)
                        index_dict['windowgap_locations']['relax']['index'].append(math.nan)
                        index_dict['windowgap_locations']['misalign']['snapnum'].append(math.nan)
                        index_dict['windowgap_locations']['relax']['snapnum'].append(math.nan)
                        continue
        
            # Orange bars = no misalignments  
            if plot_misangle_detection:
                # Plot misalignment detections
                axs.text(4.2, 196, 'Isolation time: %s Gyr' %time_no_misangle, fontsize=8, color='orange')
            
                for index_m, index_r, index_window_m, index_window_r, plot_height in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], index_dict['windowgap_locations']['misalign']['index'], index_dict['windowgap_locations']['relax']['index'], index_dict['plot_height']):
                
                    # If its one for which no window exists, skip
                    if np.isnan(index_window_m) == True:
                        continue
                
                    # Plot line from window edge to mis start, and from mis end to window edge
                    axs.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m]+(0 if time_no_misangle == None else time_no_misangle), galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m]], [plot_height, plot_height], lw=1, color='orange', solid_capstyle="butt", alpha=0.8)
                    axs.plot([galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r], galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-(0 if time_no_misangle == None else time_no_misangle)], [plot_height, plot_height], lw=1, color='orange', solid_capstyle="butt", alpha=0.8)
                    axs.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m]+(0 if time_no_misangle == None else time_no_misangle), ymin=plot_height-4, ymax=plot_height+4, lw=2, colors='orange', alpha=0.8)
                    axs.vlines(x=galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r]-(0 if time_no_misangle == None else time_no_misangle), ymin=plot_height-4, ymax=plot_height+4, lw=2, colors='orange', alpha=0.8)

                
            #>>>>>>>>>>>>>>>>>>>>>
            # Skip galaxy if no windows available for misalignments
            if len(index_dict['windowgap_locations']['misalign']['index']) == 0:
                if print_checks:
                    print('x FAILED CHECK 4: no isolated misalignments/gaps for ± %s Gyr' %(time_no_misangle))
                if plot_misangle_detection:
                    if showfig:
                        plt.show()
                continue
            elif np.isnan(np.array(index_dict['windowgap_locations']['misalign']['index'])).all() == True:
                if print_checks:
                    print('x FAILED CHECK 4: no isolated misalignments/gaps for ± %s Gyr' %(time_no_misangle))
                if plot_misangle_detection:
                    if showfig:
                        plt.show()
                continue
            elif print_checks:
                print('  CHECK 4: %s misalignments with isolated misalignments/gaps for ± %s Gyr' %(np.count_nonzero(~np.isnan(index_dict['window_locations']['misalign']['index'])), time_no_misangle))
            #=========================================================================
        
        
            #=========================================================================
            # CHECK 5: Check mergers in range. Will look for all we have available, will keep even if window incomplete
            # USES FIRST MISALIGNED INDEX NOT LAST ALIGNED
            # For each misalignment and relaxation PAIR, find hypothetical snapnums and indexes either side
            index_dict.update({'merger_locations': {'index': [],
                                                    'snapnum': []}})
            if use_merger_criteria != False:
                for index_m, index_r, snap_m, snap_r, index_window_m, index_window_r, snap_window_m, snap_window_r, index_gap_m, index_gap_r, snap_gap_m, snap_gap_r in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], index_dict['misalignment_locations']['misalign']['snapnum'], index_dict['misalignment_locations']['relax']['snapnum'], index_dict['window_locations']['misalign']['index'], index_dict['window_locations']['relax']['index'], index_dict['window_locations']['misalign']['snapnum'], index_dict['window_locations']['relax']['snapnum'], index_dict['windowgap_locations']['misalign']['index'], index_dict['windowgap_locations']['relax']['index'], index_dict['windowgap_locations']['misalign']['snapnum'], index_dict['windowgap_locations']['relax']['snapnum']):
          
                    # If no window available, skip
                    if np.isnan(snap_window_m) == True:
                        index_dict['merger_locations']['index'].append([])
                        index_dict['merger_locations']['snapnum'].append([])
                        continue
                    # If no misangle check fails, skip
                    elif np.isnan(snap_gap_m) == True:
                        index_dict['merger_locations']['index'].append([])
                        index_dict['merger_locations']['snapnum'].append([])
                        continue
                    elif snap_m == 200:
                        index_dict['merger_locations']['index'].append([])
                        index_dict['merger_locations']['snapnum'].append([])
                        continue
                    
                    
                    # Find indexes 
                    time_extra_snap_b = np.where(Lookbacktime_tree >= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m] + (0 if max_merger_pre == None else max_merger_pre)))[0][-1]
                    if time_extra_snap_b < tree_input['csv_sample_range'][0]:
                        time_extra_snap_b = tree_input['csv_sample_range'][0]
                    if len(np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r] - (0 if max_merger_post == None else max_merger_post)))[0]) == 0:
                        time_extra_snap_a = 200
                    else:
                        time_extra_snap_a = np.where(Lookbacktime_tree <= (galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r] - (0 if max_merger_post == None else max_merger_post)))[0][0]
            
            
                    # Find indexes that we have
                    if len(np.where(np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) <= time_extra_snap_b)[0]) > 0:
                        time_extra_index_b = np.where(np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) <= time_extra_snap_b)[0][-1]
                    else:
                        time_extra_index_b = 0
                    if len(np.where(np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) >= time_extra_snap_a)[0]) > 0:
                        time_extra_index_a = np.where(np.array(galaxy_tree['%s' %GalaxyID]['SnapNum']) >= time_extra_snap_a)[0][0]
                    else:
                        time_extra_index_a = len(galaxy_tree['%s' %GalaxyID]['SnapNum'])-1
                
            
                    # Look over index range to look for mergers that meet stellar and gas criteria
                    gather_merger_index = []
                    gather_merger_snapnum = []
                    for index_i in np.arange(time_extra_index_b, time_extra_index_a+1):
                        if (max(galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'][index_i], default=math.nan) >= (0 if min_stellar_ratio == None else min_stellar_ratio)) & (max(galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'][index_i], default=math.nan) <= (1e10 if max_stellar_ratio == None else max_stellar_ratio)) & (max(galaxy_tree['%s' %GalaxyID]['merger_ratio_gas'][index_i], default=math.nan) >= (0 if min_gas_ratio == None else min_gas_ratio)) & (max(galaxy_tree['%s' %GalaxyID]['merger_ratio_gas'][index_i], default=math.nan) <= (1e10 if max_gas_ratio == None else max_gas_ratio)):
                            gather_merger_index.append(index_i)
                            gather_merger_snapnum.append(galaxy_tree['%s' %GalaxyID]['SnapNum'][index_i])
            
                    # Add to merger locations index
                    index_dict['merger_locations']['index'].append(gather_merger_index)
                    index_dict['merger_locations']['snapnum'].append(gather_merger_snapnum)
            
                # Red lines = mergers
                if plot_misangle_detection:
                    for index_m, index_r, index_window_m, index_gap_m, index_list_merger, plot_height in zip(index_dict['misalignment_locations']['misalign']['index'], index_dict['misalignment_locations']['relax']['index'], index_dict['window_locations']['misalign']['index'], index_dict['windowgap_locations']['misalign']['index'], index_dict['merger_locations']['index'], index_dict['plot_height']):

                        # Plot range
                        axs.vlines(x=(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_m] + (0 if max_merger_pre == None else max_merger_pre)), ymin=plot_height-5, ymax=plot_height+5, lw=2, colors='r', alpha=0.8, zorder=999)
                        axs.vlines(x=(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_r] - (0 if max_merger_post == None else max_merger_post)), ymin=plot_height-5, ymax=plot_height+5, lw=2, colors='r', alpha=0.8, zorder=999)
                    
                        # If its one for which no window exists, skip
                        if np.isnan(index_window_m) == True:
                            continue
                        # If its one for which no window exists, skip
                        if np.isnan(index_gap_m) == True:
                            continue
                        # If no mergers in range, skip
                        if len(index_list_merger) == 0:
                            continue
            
                        # overplot mergers with second line that meet criteria
                        for index_i in index_list_merger:
                            axs.axvline(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_i], c='r', ls='--', lw=1, zorder=999)
                            axs.text(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_i]-0.2, 175, '%.2f' %max(galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'][index_i]), color='r', zorder=999, fontsize=8)
                            axs.text(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_i]-0.2, 170, '%.2f' %galaxy_tree['%s' %GalaxyID]['merger_ratio_gas'][index_i][np.argmax(galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'][index_i])], color='b', zorder=999, fontsize=8)
                    axs.text(4.2, 190, 'Merger window: %s-%s Gyr' %(max_merger_pre, max_merger_post), fontsize=8, color='r')
                
                #>>>>>>>>>>>>>>>>>>>>>
                # Skip galaxy if no mergers meeting criteria in range
                if len(index_dict['merger_locations']['index']) == 0:
                    if print_checks:
                        print('x FAILED CHECK 5: no mergers in range %s-%s Gyr' %(max_merger_pre, max_merger_post))
                    if plot_misangle_detection:
                        if showfig:
                            plt.show()
                    continue
                elif print_checks:
                    print('  CHECK 5: %s misalignments with mergers in range %s-%s Gyr' %(np.count_nonzero(~np.isnan(index_dict['window_locations']['misalign']['index'])), max_merger_pre, max_merger_post))
                
            #=========================================================================
            
        
            #if plot_misangle_detection:
            #    if showfig:
            #       plt.show()
        
        
            #=========================================================================
            # CHECK 6 & 7: all range checks (particle counts, inc angles, com)
        
            if print_checks:
                print('  Meets pre-checks, looping over %s misalignments...' %(np.count_nonzero(~np.isnan(index_dict['window_locations']['misalign']['index']))))
        
            # Identify index of misalignments that have met criteria thus far
            if plot_misangle_detection:
                plot_misangle_accepted_window     = []
                plot_misangle_accepted_window_t   = []
                plot_misangle_accepted_misangle   = []
                plot_misangle_accepted_misangle_t = []
            # Loop over all misalignments, and ignore any misalignments that didn't meet previous conditions
            for misindex_i in np.arange(0, len(index_dict['misalignment_locations']['misalign']['index'])):
            
                if print_checks:
                    print('  \nFor misalignment at: %.2f Gyr...' %(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_dict['misalignment_locations']['misalign']['index'][misindex_i]]))
            
                if (np.isnan(index_dict['misalignment_locations']['misalign']['index'][misindex_i]) == True) | (np.isnan(index_dict['window_locations']['misalign']['index'][misindex_i]) == True) | (np.isnan(index_dict['windowgap_locations']['misalign']['index'][misindex_i]) == True):
                    if print_checks:
                        print('    x Misalignment index, window index, or window gap index nan for current misalignment')
                    continue
            
                # If we want mergers and we dont have any, skip
                if use_merger_criteria == True:
                    if len(index_dict['merger_locations']['index'][misindex_i]) == 0:
                        if print_checks:
                            print('    x No mergers for current misalignment')
                        continue
                # If we want NO mergers and we have some, skip
                if use_merger_criteria == None:
                    if len(index_dict['merger_locations']['index'][misindex_i]) > 0:
                        if print_checks:
                            print('    x Mergers meeting criteria exist for current misalignment')
                        continue
            
                #------------------------------------------------------------------------------------------------
                # Check relaxation properties
                index_start = index_dict['window_locations']['misalign']['index'][misindex_i]
                index_stop  = index_dict['window_locations']['relax']['index'][misindex_i]+1
            
            
                # Filter for relaxation type (co-counter) 
                all_angles = galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj]
                if (all_angles[index_dict['misalignment_locations']['misalign']['index'][misindex_i]] < misangle_threshold) and (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] < misangle_threshold):
                    check = 'co-co'
                elif (all_angles[index_dict['misalignment_locations']['misalign']['index'][misindex_i]] < misangle_threshold) and (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] > (180-misangle_threshold)):
                    check = 'co-counter'
                elif (all_angles[index_dict['misalignment_locations']['misalign']['index'][misindex_i]] > (180-misangle_threshold)) and (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] < misangle_threshold):
                    check = 'counter-co'
                elif (all_angles[index_dict['misalignment_locations']['misalign']['index'][misindex_i]] > (180-misangle_threshold)) and (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] > (180-misangle_threshold)):
                    check = 'counter-counter'
                if check not in relaxation_type:
                    if print_checks:
                        print('    x FAILED RELAXATION TYPE: %.2f -> %.2f\t%s' %(all_angles[index_dict['misalignment_locations']['misalign']['index'][misindex_i]], all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]], check))
                    continue
                elif print_checks:
                    print('    MET RELAXATION TYPE: %.2f -> %.2f\t%s' %(all_angles[index_dict['misalignment_locations']['misalign']['index'][misindex_i]], all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]], check))
                
            
                # Filter for relaxation morphology (ETG-LTG)
                if (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_dict['misalignment_locations']['misalign']['index'][misindex_i]+1])) > morph_limits[1]) and (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_dict['misalignment_locations']['relax']['index'][misindex_i]:index_stop])) > morph_limits[1]):
                    check = 'LTG-LTG'
                elif (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_dict['misalignment_locations']['misalign']['index'][misindex_i]+1])) > morph_limits[1]) and (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_dict['misalignment_locations']['relax']['index'][misindex_i]:index_stop])) < morph_limits[0]):
                    check = 'LTG-ETG'
                elif (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_dict['misalignment_locations']['misalign']['index'][misindex_i]+1])) < morph_limits[0]) and (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_dict['misalignment_locations']['relax']['index'][misindex_i]:index_stop])) > morph_limits[1]):
                    check = 'ETG-LTG'
                elif (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_dict['misalignment_locations']['misalign']['index'][misindex_i]+1])) < morph_limits[0]) and (np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_dict['misalignment_locations']['relax']['index'][misindex_i]:index_stop])) < morph_limits[0]):
                    check = 'ETG-ETG'
                else:
                    check = 'other'
                if check not in relaxation_morph:
                    if print_checks:
                        print('    x FAILED RELAXATION MORPH: %.2f -> %.2f\t%s' %(np.mean(np.array(galaxy_tree['%s' %GalaxyID]['kappa_stars'][index_start:index_dict['misalignment_locations']['misalign']['index'][misindex_i]+1])), np.mean(np.array(galaxy_tree['%s' %GalaxyID]['kappa_stars'][index_dict['misalignment_locations']['relax']['index'][misindex_i]:index_stop+1])), check))
                    continue
                elif print_checks:
                    print('    MET RELAXATION MORPH: %.2f -> %.2f\t%s' %(np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_dict['misalignment_locations']['misalign']['index'][misindex_i]+1])), np.mean(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_dict['misalignment_locations']['relax']['index'][misindex_i]:index_stop+1])), check))
            
            
                # Filter for peak misangle (maximum deviation from where it relaxes to)
                all_angles = galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj]
                check = False
                if (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] < misangle_threshold):
                    # relax to co-
                    check_array = max(all_angles[index_start:index_stop])
                    if max(all_angles[index_start:index_stop]) > (0 if peak_misangle == None else peak_misangle):
                        check = True
                elif (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] > (180-misangle_threshold)):
                    # relax to counter-
                    check_array = max(180-np.array(all_angles[index_start:index_stop]))
                    if max(180-np.array(all_angles[index_start:index_stop])) > (0 if peak_misangle == None else peak_misangle):
                        check = True
                if check == False:
                    if print_checks:
                        print('    x FAILED PEAK ANGLE: \t Δ%.1f° to relax to %s, for limit Δ%s°' %(check_array, ('0°' if (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] < misangle_threshold) else '180°'), (0 if peak_misangle == None else peak_misangle)))
                    continue
                elif print_checks:
                    print('    MET PEAK ANGLE: \t Δ%.1f° to relax to %s, for limit Δ%s°' %(check_array, ('0°' if (all_angles[index_dict['misalignment_locations']['relax']['index'][misindex_i]] < misangle_threshold) else '180°'), (0 if peak_misangle == None else peak_misangle)))
            
            
                # Filter for relaxation time
                check = float(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_dict['misalignment_locations']['misalign']['index'][misindex_i]]) - float(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_dict['misalignment_locations']['relax']['index'][misindex_i]])
                if (check < (0 if min_trelax == None else min_trelax)) or (check > (9999 if max_trelax == None else max_trelax)):
                    if print_checks:
                        print('    x FAILED t_relax SELECTION: \t %.2f Gyr, for limits %s / %s Gyr' %(check_array, min_trelax, max_trelax))
                    continue
                elif print_checks:
                    print('    MET t_relax SELECTION: \t %.2f Gyr, for limits %s / %s Gyr' %(check_array, min_trelax, max_trelax))
            
            
                #------------------------------------------------------------------------------------------------
                # Check window properties which are to be ALWAYS met
                # Loop over window and apply checks for particle counts (using particles we care about), com (using angle we care about), and inclination angle (using particles we care about)
                index_start = index_dict['window_locations']['misalign']['index'][misindex_i]
                index_stop  = index_dict['window_locations']['relax']['index'][misindex_i]+1
            
            
                # Check particle counts
                check = []            
                for particle_i in particle_selection:
                    if particle_i == 'dm':
                        check.append((np.array(galaxy_tree['%s' %GalaxyID]['dm']['count'][index_start:index_stop]) >= min_particles).all())
                    else:
                        check.append((np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_i]['%s_hmr' %use_hmr_angle]['count'][index_start:index_stop]) >= min_particles).all())
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED COUNTS:\t\t min %i, min %i, for limit %s' %(np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[0]]['%s_hmr' %use_hmr_angle]['count'][index_start:index_stop])), np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[1]]['%s_hmr' %use_hmr_angle]['count'][index_start:index_stop])), min_particles))
                    continue
                elif print_checks:
                    print('    MET COUNTS:\t\t min %i, min %i, for limit %s' %(np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[0]]['%s_hmr' %use_hmr_angle]['count'][index_start:index_stop])), np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[1]]['%s_hmr' %use_hmr_angle]['count'][index_start:index_stop])), min_particles))
            
            
                # Check dm particle counts as we include stars-dm angle always
                check = []            
                check.append((np.array(galaxy_tree['%s' %GalaxyID]['dm']['count'][index_start:index_stop]) >= min_particles).all())
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED DM COUNTS:\t min %i, min %i, for limit %s' %(np.min(np.array(galaxy_tree['%s' %GalaxyID]['dm']['count'][index_start:index_stop])), np.min(np.array(galaxy_tree['%s' %GalaxyID]['dm']['count'][index_start:index_stop])), min_particles))
                    continue
                elif print_checks:
                    print('    MET DM COUNTS:\t min %i, min %i, for limit %s' %(np.min(np.array(galaxy_tree['%s' %GalaxyID]['dm']['count'][index_start:index_stop])), np.min(np.array(galaxy_tree['%s' %GalaxyID]['dm']['count'][index_start:index_stop])), min_particles))
                
                
                # Check stelmass doesnt drop randomly
                check = []
                if use_hmr_general == 'aperture':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['stars']['ap_mass'][index_start:index_stop])
                else:
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass'][index_start:index_stop])
                for ii, check_i in enumerate(check_array):
                    if ii == 0:
                        check_i_previous = check_i
                        continue
                    else:
                        # Ensure ratio between stelmasses doesnt drop by half or worse
                        if check_i/check_i_previous >= 0.5:
                            check.append(True)
                        else:
                            check.append(False)
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED SUBGROUPNUM MASS')
                    continue
                if print_checks:
                    print('    MET SUBGROUPNUM MASS')
                
                
                # Check aperture stelmass
                check = []
                check.append((np.log10(np.array(galaxy_tree['%s' %GalaxyID]['stars']['ap_mass'][index_start:index_stop])) >= 9.5).all())
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED APERTURE STEL MASS:\t %.1e, for limit 9.5+ [Msun]' %(np.average(check_array, weights=time_weight)))
                    continue
                if print_checks:
                    print('    MET APERTURE STEL MASS:\t %.1e, for limit 9.5+ [Msun]' %(np.average(check_array, weights=time_weight)))
                
                
                # Check inclination
                check = []            
                for particle_i in particle_selection:
                    check.append((np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_i]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop]) >= (0 if min_inclination == None else min_inclination)).all())
                    check.append((np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_i]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop]) <= 180-(0 if min_inclination == None else min_inclination)).all()) 
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED INCLINATION:\t %.1f° / %.1f°, %.1f° / %.1f°, for limit %s / %s°' %(np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[0]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), np.max(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[0]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[1]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), np.max(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[1]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), min_inclination, 180-min_inclination))
                    continue
                if print_checks:
                    print('    MET INCLINATION:\t %.1f° / %.1f°, %.1f° / %.1f°, for limit %s / %s°' %(np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[0]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), np.max(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[0]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), np.min(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[1]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), np.max(np.array(galaxy_tree['%s' %GalaxyID]['%s' %particle_selection[1]]['%s_hmr' %use_hmr_angle]['proj_angle'][index_start:index_stop])), min_inclination, 180-min_inclination))
                
                
                # Check CoM
                check = [] 
                check.append((np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['com_abs'][index_start:index_stop]) <= max_com).all())
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED CoM:\t max %.1f kpc, for limit %s [kpc]' %(np.max(np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['com_abs'][index_start:index_stop])), min_inclination))
                    continue
                if print_checks:
                    print('    MET CoM:\t\t max %.1f kpc, for limit %s [kpc]' %(np.max(np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['com_abs'][index_start:index_stop])), min_inclination))
                
                
                # Filter for uncertainty
                check = []
                check_array = [max(np.abs(i)) for i in (np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj][index_start:index_stop]) - np.array(galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_start:index_stop])[:,None])]
                check.append((np.array(check_array) <= (9999 if max_uncertainty == None else max_uncertainty)).all())
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED UNCERTAINTY:\t max %i, for limit %s' %(max(check_array), max_uncertainty))
                    continue
                if print_checks:
                    print('    MET UNCERTAINTY:\t max %i, for limit %s' %(max(check_array), max_uncertainty))
                
                
                
                #------------------------------------------------------------------------------------------------
                # Check satellite over misalignment
            
                # Extract indexes to mean over
                index_start = index_dict['misalignment_locations']['misalign']['index'][misindex_i]
                index_stop   = index_dict['misalignment_locations']['relax']['index'][misindex_i]+1
            
                # Check satellite properties
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['SubGroupNum'][index_start:index_stop])
                if limit_satellites == 'central':
                    check.append((check_array == 0).all())
                elif limit_satellites == 'satellite':
                    check.append((check_array >= 1).all())
                else:
                    check.append((check_array >= 0).all())
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED CENTRAL/SAT:\t %i / %i, for limit %s' %(np.min(check_array), np.max(check_array), limit_satellites))
                    continue
                if print_checks:
                    print('    MET CENTRAL/SAT:\t %i / %i, for limit %s' %(np.min(check_array), np.max(check_array), limit_satellites))
                
              
                #------------------------------------------------------------------------------------------------
                # Check time-weighted average over misalignment properties
            
                # Extract indexes to mean over
                index_start = index_dict['misalignment_locations']['misalign']['index'][misindex_i]
                index_stop  = index_dict['misalignment_locations']['relax']['index'][misindex_i]+1
                if len(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_start-1:index_stop-1]) == 0:
                    time_weight = np.ones(len(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_start:index_stop]))
                else:
                    time_weight = np.array(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_start-1:index_stop-1]) - np.array(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_start:index_stop])
             
                
                # Filter for misalignment morphology (ETG, LTG)
                check = False
                if np.average(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_stop]), weights=time_weight) <= morph_limits[0]:
                    check = 'ETG'
                elif np.average(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_stop]), weights=time_weight) >= morph_limits[1]:
                    check = 'LTG'
                else:
                    check = 'other'
                if check not in misalignment_morph:
                    if print_checks:
                        print('    x FAILED MISALIGNMENT MORPH: %.2f\t%s' %(np.average(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_stop]), weights=time_weight), check))
                    continue
                elif print_checks:
                    print('    MET MISALIGNMENT MORPH: %.2f\t%s' %(np.average(np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_stop]), weights=time_weight), check))
            
            
                # Check halomass
                check = []                
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['halomass'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_halomass == None else max_halomass))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_halomass == None else min_halomass))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED HALO MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_halomass == None else min_halomass), (math.nan if max_halomass == None else max_halomass)))
                    continue
                if print_checks:
                    print('    MET HALO MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_halomass == None else min_halomass), (math.nan if max_halomass == None else max_halomass)))
            
            
                # Check stelmass
                check = []
                if use_hmr_general == 'aperture':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['stars']['ap_mass'][index_start:index_stop])
                    window_stelmass = galaxy_tree['%s' %GalaxyID]['stars']['ap_mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                else:
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass'][index_start:index_stop])
                    window_stelmass = galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_stelmass == None else max_stelmass))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_stelmass == None else min_stelmass))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED STEL MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_stelmass == None else min_stelmass), (math.nan if max_stelmass == None else max_stelmass)))
                    continue
                if print_checks:
                    print('    MET STEL MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_stelmass == None else min_stelmass), (math.nan if max_stelmass == None else max_stelmass)))
                window_stelmass_1hmr = galaxy_tree['%s' %GalaxyID]['stars']['1.0_hmr']['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                

                # Check gasmass
                check = []
                if use_hmr_general == 'aperture':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas']['ap_mass'][index_start:index_stop])
                    window_gasmass = galaxy_tree['%s' %GalaxyID]['gas']['ap_mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                else:
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['mass'][index_start:index_stop])
                    window_gasmass = galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_gasmass == None else max_gasmass))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_gasmass == None else min_gasmass))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED GAS MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_gasmass == None else min_gasmass), (math.nan if max_gasmass == None else max_gasmass)))
                    continue
                if print_checks:
                    print('    MET GAS MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_gasmass == None else min_gasmass), (math.nan if max_gasmass == None else max_gasmass)))
                window_gasmass_1hmr = galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                
            
                # Check gasmass sf
                check = []
                if use_hmr_general == 'aperture':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas_sf']['ap_mass'][index_start:index_stop])
                    window_sfmass = galaxy_tree['%s' %GalaxyID]['gas_sf']['ap_mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                else:
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas_sf']['%s_hmr' %use_hmr_general]['mass'][index_start:index_stop])
                    window_sfmass = galaxy_tree['%s' %GalaxyID]['gas_sf']['%s_hmr' %use_hmr_general]['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_sfmass == None else max_sfmass))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_sfmass == None else min_sfmass))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED SF MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_sfmass == None else min_sfmass), (math.nan if max_sfmass == None else max_sfmass)))
                    continue
                if print_checks:
                    print('    MET SF MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_sfmass == None else min_sfmass), (math.nan if max_sfmass == None else max_sfmass)))
                window_sfmass_1hmr = galaxy_tree['%s' %GalaxyID]['gas_sf']['1.0_hmr']['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                
            
                # Check gasmass nsf
                check = []
                if use_hmr_general == 'aperture':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas_nsf']['ap_mass'][index_start:index_stop])
                    window_nsfmass = galaxy_tree['%s' %GalaxyID]['gas_nsf']['ap_mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                else:
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas_nsf']['%s_hmr' %use_hmr_general]['mass'][index_start:index_stop])
                    window_nsfmass = galaxy_tree['%s' %GalaxyID]['gas_nsf']['%s_hmr' %use_hmr_general]['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_nsfmass == None else max_nsfmass))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_nsfmass == None else min_nsfmass))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED NSF MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_nsfmass == None else min_nsfmass), (math.nan if max_nsfmass == None else max_nsfmass)))
                    continue
                if print_checks:
                    print('    MET NSF MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_nsfmass == None else min_nsfmass), (math.nan if max_nsfmass == None else max_nsfmass)))
                
               
                # Check sfr
                check = []
                if use_hmr_general == 'aperture':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['ap_sfr'][index_start:index_stop])
                    window_sfr = galaxy_tree['%s' %GalaxyID]['ap_sfr'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                else:
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['sfr'][index_start:index_stop])
                    window_sfr = galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['sfr'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_sfr == None else max_sfr))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_sfr == None else min_sfr))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED SFR:\t\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_sfr == None else min_sfr), (math.nan if max_sfr == None else max_sfr)))
                    continue
                if print_checks:
                    print('    MET SFR:\t\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_sfr == None else min_sfr), (math.nan if max_sfr == None else max_sfr)))
            
            
                # Check ssfr
                check = []
                if use_hmr_general == 'aperture':
                    check_array = np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['sfr'][index_start:index_stop]), np.array(galaxy_tree['%s' %GalaxyID]['stars']['2.0_hmr']['mass'][index_start:index_stop]))
                    window_ssfr = np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['sfr'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]), np.array(galaxy_tree['%s' %GalaxyID]['stars']['2.0_hmr']['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]))
                else:
                    check_array = np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['sfr'][index_start:index_stop]), np.array(galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass'][index_start:index_stop]))
                    window_ssfr = np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_general]['sfr'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]), np.array(galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_general]['mass'][int(index_dict['window_locations']['misalign']['index'][misindex_i]):int(index_dict['window_locations']['relax']['index'][misindex_i]+1)]))
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_ssfr == None else max_ssfr))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_ssfr == None else min_ssfr))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED sSFR:\t\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_ssfr == None else min_ssfr), (math.nan if max_ssfr == None else max_ssfr)))
                    continue
                if print_checks:
                    print('    MET sSFR:\t\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_ssfr == None else min_ssfr), (math.nan if max_ssfr == None else max_ssfr)))
            
            
                # Check kappa stars
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1.0 if max_kappa_stars == None else max_kappa_stars))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_kappa_stars == None else min_kappa_stars))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED KAPPA STARS:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_stars, max_kappa_stars))
                    continue
                if print_checks:
                    print('    MET KAPPA STARS:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_stars, max_kappa_stars))
                averaged_morphology = np.average(check_array, weights=time_weight)
            
            
                # Check kappa gas
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas']['kappa'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1.0 if max_kappa_gas == None else max_kappa_gas))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_kappa_gas == None else min_kappa_gas))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED KAPPA GAS:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_gas, max_kappa_gas))
                    continue
                if print_checks:
                    print('    MET KAPPA GAS:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_gas, max_kappa_gas))
            
            
                # Check kappa sf
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas_sf']['kappa'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1.0 if max_kappa_sf == None else max_kappa_sf))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_kappa_sf == None else min_kappa_sf))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED KAPPA SF:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_sf, max_kappa_sf))
                    continue
                if print_checks:
                    print('    MET KAPPA SF:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_sf, max_kappa_sf))
            
            
                # Check kappa msf
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas_nsf']['kappa'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1.0 if max_kappa_nsf == None else max_kappa_nsf))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_kappa_nsf == None else min_kappa_nsf))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED KAPPA NSF:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_nsf, max_kappa_nsf))
                    continue
                if print_checks:
                    print('    MET KAPPA NSF:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_kappa_nsf, max_kappa_nsf))
            
            
                # Check ellipticity
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['ellip'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1.0 if max_ellip == None else max_ellip))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_ellip == None else min_ellip))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED ELLIPTICITY:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_ellip, max_ellip))
                    continue
                if print_checks:
                    print('    MET ELLIPTICITY:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_ellip, max_ellip))
            
            
                # Check triaxiality
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['triax'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1.0 if max_triax == None else max_triax))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_triax == None else min_triax))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED TRIAXIALITY:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_triax, max_triax))
                    continue
                if print_checks:
                    print('    MET TRIAXIALITY:\t %.2f / %.2f, for limits %s / %s' %(np.min(check_array), np.max(check_array), min_triax, max_triax))
            
            
                # Check rad
                check = []
                if abs_or_proj == 'abs':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['rad'][index_start:index_stop])
                elif abs_or_proj == 'proj':
                    check_array = np.array(galaxy_tree['%s' %GalaxyID]['radproj'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_rad == None else max_rad))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_rad == None else min_rad))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED RAD:\t\t %.2f / %.2f kpc, for limits %s %s / %s [kpc]' %(np.min(check_array), np.max(check_array), abs_or_proj, min_rad, max_rad))
                    continue
                if print_checks:
                    print('    MET RAD:\t\t %.2f / %.2f kpc, for limits %s %s / %s [kpc]' %(np.min(check_array), np.max(check_array), abs_or_proj, min_rad, max_rad))
            
            
                # Check SF gas radius
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['rad_sf'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_rad_sf == None else max_rad_sf))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_rad_sf == None else min_rad_sf))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED RAD SF:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_rad_sf, max_rad_sf))
                    continue
                if print_checks:
                    print('    MET RAD SF:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_rad_sf, max_rad_sf))
            
            
                # Check inflow of gas
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['inflow_rate'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_inflow == None else max_inflow))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_inflow == None else min_inflow))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED GAS INFLOW:\t %.2f / %.2f, for limits %s / %s [Msun/yr]' %(np.min(check_array), np.max(check_array), min_inflow, max_inflow))
                    continue
                if print_checks:
                    print('    MET GAS INFLOW:\t %.2f / %.2f, for limits %s / %s [Msun/yr]' %(np.min(check_array), np.max(check_array), min_inflow, max_inflow))
            
            
                # Check metallicity of inflow gas 
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['inflow_Z'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_inflow_Z == None else max_inflow_Z))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_inflow_Z == None else min_inflow_Z))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED INFLOW Z:\t %.2f / %.2f, for limits %s / %s [Z]' %(np.min(check_array), np.max(check_array), min_inflow_Z, max_inflow_Z))
                    continue
                if print_checks:
                    print('    MET INFLOW Z:\t %.2f / %.2f, for limits %s / %s [Z]' %(np.min(check_array), np.max(check_array), min_inflow_Z, max_inflow_Z))
            
            
                #------------------------------------------------------------------------------------------------
            
                # Check BH mass
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['bh']['mass'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_bh_mass == None else max_bh_mass))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_bh_mass == None else min_bh_mass))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED BH MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_bh_mass == None else min_bh_mass), (math.nan if max_bh_mass == None else max_bh_mass)))
                    continue
                if print_checks:
                    print('    MET BH MASS:\t %.1e, for limits %.1e / %.1e [Msun]' %(np.average(check_array, weights=time_weight), (math.nan if min_bh_mass == None else min_bh_mass), (math.nan if max_bh_mass == None else max_bh_mass)))
            
            
                # Check BH accretion rate (averaged)
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['bh']['mdot'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_bh_acc == None else max_bh_acc))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_bh_acc == None else min_bh_acc))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED BH ACC (AV):\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_bh_acc == None else min_bh_acc), (math.nan if max_bh_acc == None else max_bh_acc)))
                    continue
                if print_checks:
                    print('    MET BH ACC (AV):\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_bh_acc == None else min_bh_acc), (math.nan if max_bh_acc == None else max_bh_acc)))
            
            
                # Check BH accretion rate (instantaneous)
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['bh']['mdot_instant'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_bh_acc_instant == None else max_bh_acc_instant))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_bh_acc_instant == None else min_bh_acc_instant))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED BH ACC (INST):\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_bh_acc_instant == None else min_bh_acc_instant), (math.nan if max_bh_acc_instant == None else max_bh_acc_instant)))
                    continue
                if print_checks:
                    print('    MET BH ACC (INST):\t %.1e, for limits %.1e / %.1e [Msun/yr]' %(np.average(check_array, weights=time_weight), (math.nan if min_bh_acc_instant == None else min_bh_acc_instant), (math.nan if max_bh_acc_instant == None else max_bh_acc_instant)))
            
            
                # Check BH eddington (instantaneous)
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['bh']['edd'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1 if max_edd == None else max_edd))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_edd == None else min_edd))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED BH EDD (INST):\t %.6f / %.6f, for limits %s / %s [Msun/yr]' %(np.min(check_array), np.max(check_array), min_edd, max_edd))
                    continue
                if print_checks:
                    print('    MET BH EDD (INST):\t %.6f / %.6f, for limits %s / %s [Msun/yr]' %(np.min(check_array), np.max(check_array), min_edd, max_edd))
            
            
                # Check BH luminosity (instantaneous) |     using L = e Mdot c2 -> converting Mdot from [Msun/yr] -> [kg/s] -> [erg/s]
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['bh']['lbol'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e60 if max_lbol == None else max_lbol))
                check.append(np.average(check_array, weights=time_weight) >= (0 if min_lbol == None else min_lbol))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED BH Lbol (INST):\t %.1e / %.1e, for limits %s / %s [erg/s]' %(np.min(check_array), np.max(check_array), min_lbol, max_lbol))
                    continue
                if print_checks:
                    print('    MET BH Lbol (INST):\t %.1e / %.1e, for limits %s / %s [erg/s]' %(np.min(check_array), np.max(check_array), min_lbol, max_lbol))
            
            
                #------------------------------------------------------------------------------------------------
            
                # Check vcirc
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_general]['vcirc'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_vcirc == None else max_vcirc))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_vcirc == None else min_vcirc))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED v_circ:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_vcirc, max_vcirc))
                    continue
                if print_checks:
                    print('    MET v_circ:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_vcirc, max_vcirc))
            
            
                # Check dynamical time
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_general]['tdyn'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_tdyn == None else max_tdyn))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_tdyn == None else min_tdyn))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED t_dyn:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_tdyn, max_tdyn))
                    continue
                if print_checks:
                    print('    MET t_dyn:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_tdyn, max_tdyn))
            
            
                # Check torquing time
                check = []
                check_array = np.array(galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_general]['ttorque'][index_start:index_stop])
                check.append(np.average(check_array, weights=time_weight) <= (1e20 if max_ttorque == None else max_ttorque))
                check.append(np.average(check_array, weights=time_weight) >= (0.0 if min_ttorque == None else min_ttorque))
                if np.array(check).all() == False:
                    if print_checks:
                        print('    x FAILED t_torque:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_ttorque, max_ttorque))
                    continue
                if print_checks:
                    print('    MET t_torque:\t %.2f / %.2f kpc, for limits %s / %s [kpc]' %(np.min(check_array), np.max(check_array), min_ttorque, max_ttorque))
            
            
            
                #================================================================================
                # If this galaxy's particular misalignment passes, append arrays WINDOWS to new misalignment_tree starting from first ID
                index_start = index_dict['window_locations']['misalign']['index'][misindex_i]
                index_stop   = index_dict['window_locations']['relax']['index'][misindex_i]+1
            
                ID_i = galaxy_tree['%s' %GalaxyID]['GalaxyID'][index_start:index_stop][0]
            
                # Add stats we limited by
                misalignment_tree.update({'%s' %ID_i: {'GalaxyID': galaxy_tree['%s' %GalaxyID]['GalaxyID'][index_start:index_stop],
                                                        'SnapNum': galaxy_tree['%s' %GalaxyID]['SnapNum'][index_start:index_stop],
                                                        'Redshift': galaxy_tree['%s' %GalaxyID]['Redshift'][index_start:index_stop],
                                                        'Lookbacktime': galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_start:index_stop],
            
                                                        'SubGroupNum': galaxy_tree['%s' %GalaxyID]['SubGroupNum'][index_start:index_stop],
                                                        'halomass': galaxy_tree['%s' %GalaxyID]['halomass'][index_start:index_stop],
                                                        'stelmass': window_stelmass,
                                                        'stelmass_1hmr': window_stelmass_1hmr,
                                                        'gasmass': window_gasmass,
                                                        'gasmass_1hmr': window_gasmass_1hmr,
                                                        'sfmass': window_sfmass,
                                                        'sfmass_1hmr': window_sfmass_1hmr,
                                                        'nsfmass': window_nsfmass,
                                                        'dmmass': galaxy_tree['%s' %GalaxyID]['dm']['ap_mass'][index_start:index_stop],
                                                                                                                 
                                                        'sfr': window_sfr,
                                                        'ssfr': window_ssfr,
                                                    
                                                        'stars_l': galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_angle]['l'][index_start:index_stop],
                                                                                                                 
                                                        'stars_Z': galaxy_tree['%s' %GalaxyID]['stars']['%s_hmr' %use_hmr_angle]['Z'][index_start:index_stop],
                                                        'gas_Z': galaxy_tree['%s' %GalaxyID]['gas']['%s_hmr' %use_hmr_angle]['Z'][index_start:index_stop],
                                                        'sf_Z': galaxy_tree['%s' %GalaxyID]['gas_sf']['%s_hmr' %use_hmr_angle]['Z'][index_start:index_stop],
                                                        'nsf_Z': galaxy_tree['%s' %GalaxyID]['gas_nsf']['%s_hmr' %use_hmr_angle]['Z'][index_start:index_stop],
                                                                                                                 
                                                        'kappa_stars': galaxy_tree['%s' %GalaxyID]['stars']['kappa'][index_start:index_stop],
                                                        'kappa_gas': galaxy_tree['%s' %GalaxyID]['gas']['kappa'][index_start:index_stop],
                                                        'kappa_sf': galaxy_tree['%s' %GalaxyID]['gas_sf']['kappa'][index_start:index_stop],
                                                        'kappa_nsf': galaxy_tree['%s' %GalaxyID]['gas_nsf']['kappa'][index_start:index_stop],
                                                        'ellip': galaxy_tree['%s' %GalaxyID]['ellip'][index_start:index_stop],
                                                        'triax': galaxy_tree['%s' %GalaxyID]['triax'][index_start:index_stop],
                                                        'disp_ani': galaxy_tree['%s' %GalaxyID]['disp_ani'][index_start:index_stop],
                                                        'disc_to_total': galaxy_tree['%s' %GalaxyID]['disc_to_total'][index_start:index_stop],
                                                        'rot_to_disp_ratio': galaxy_tree['%s' %GalaxyID]['rot_to_disp_ratio'][index_start:index_stop],
                                                                                                                 
                                                        'rad': galaxy_tree['%s' %GalaxyID]['rad'][index_start:index_stop],
                                                        'radproj': galaxy_tree['%s' %GalaxyID]['radproj'][index_start:index_stop],
                                                        'rad_sf': galaxy_tree['%s' %GalaxyID]['rad_sf'][index_start:index_stop],
                                                    
                                                        'vcirc': galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_angle]['vcirc'][index_start:index_stop],
                                                        'tdyn': galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_angle]['tdyn'][index_start:index_stop],
                                                        'ttorque': galaxy_tree['%s' %GalaxyID]['other']['%s_hmr' %use_hmr_angle]['ttorque'][index_start:index_stop],
                                                                                                                 
                                                        'inflow_mass_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['inflow_mass'][index_start:index_stop],
                                                        'inflow_mass_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['inflow_mass'][index_start:index_stop],
                                                        'outflow_mass_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['outflow_mass'][index_start:index_stop],
                                                        'outflow_mass_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['outflow_mass'][index_start:index_stop],
                                                        'insitu_mass_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['insitu_mass'][index_start:index_stop],
                                                        'insitu_mass_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['insitu_mass'][index_start:index_stop],
                                                        'inflow_rate_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['inflow_rate'][index_start:index_stop],
                                                        'inflow_rate_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['inflow_rate'][index_start:index_stop],
                                                        'outflow_rate_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['outflow_rate'][index_start:index_stop],
                                                        'outflow_rate_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['outflow_rate'][index_start:index_stop],
                                                        'stelmassloss_rate_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['stelmassloss_rate'][index_start:index_stop],
                                                        'stelmassloss_rate_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['stelmassloss_rate'][index_start:index_stop],
                                                        's_inflow_rate_1hmr': np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['inflow_rate'][index_start:index_stop]), np.array(galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['mass'][index_start:index_stop])),
                                                        's_inflow_rate_2hmr': np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['inflow_rate'][index_start:index_stop]), np.array(galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['mass'][index_start:index_stop])),
                                                        's_outflow_rate_1hmr': np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['outflow_rate'][index_start:index_stop]), np.array(galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['mass'][index_start:index_stop])),
                                                        's_outflow_rate_2hmr': np.divide(np.array(galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['outflow_rate'][index_start:index_stop]), np.array(galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['mass'][index_start:index_stop])),
                                                        'inflow_Z_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['inflow_Z'][index_start:index_stop],
                                                        'inflow_Z_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['inflow_Z'][index_start:index_stop],
                                                        'outflow_Z_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['outflow_Z'][index_start:index_stop],
                                                        'outflow_Z_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['outflow_Z'][index_start:index_stop],
                                                        'insitu_Z_1hmr': galaxy_tree['%s' %GalaxyID]['gas']['1.0_hmr']['insitu_Z'][index_start:index_stop],
                                                        'insitu_Z_2hmr': galaxy_tree['%s' %GalaxyID]['gas']['2.0_hmr']['insitu_Z'][index_start:index_stop],
                                                                                                                 
                                                        'bh_mass': galaxy_tree['%s' %GalaxyID]['bh']['mass'][index_start:index_stop],
                                                        'bh_mdot_av': galaxy_tree['%s' %GalaxyID]['bh']['mdot'][index_start:index_stop],
                                                        'bh_mdot_inst': galaxy_tree['%s' %GalaxyID]['bh']['mdot_instant'][index_start:index_stop],
                                                        'bh_edd': galaxy_tree['%s' %GalaxyID]['bh']['edd'][index_start:index_stop],
                                                        'bh_lbol': np.array(galaxy_tree['%s' %GalaxyID]['bh']['mdot_instant'][index_start:index_stop]) * (2e30 / 3.154e+7) * (0.1 * (3e8)**2) * (1e7),
                                                                                                                  
                                                        '%s' %use_angle: galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_start:index_stop],
                                                        '%s_err' %use_angle: galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj][index_start:index_stop],
                                                        '%s_halo' %use_angle: galaxy_tree['%s' %GalaxyID]['%s' %use_angle]['%s_hmr' %use_hmr_angle]['angle_halo'][index_start:index_stop],
                                                        'stars_dm': galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_start:index_stop],
                                                        'stars_dm_err': galaxy_tree['%s' %GalaxyID]['stars_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj][index_start:index_stop],
                                                        'gas_dm': galaxy_tree['%s' %GalaxyID]['gas_dm']['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_start:index_stop],
                                                        'gas_dm_err': galaxy_tree['%s' %GalaxyID]['gas_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj][index_start:index_stop],
                                                        'gas_sf_dm': galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['angle_%s' %abs_or_proj][index_start:index_stop],
                                                        'gas_sf_dm_err': galaxy_tree['%s' %GalaxyID]['gas_sf_dm']['%s_hmr' %use_hmr_angle]['err_%s' %abs_or_proj][index_start:index_stop],
                                                                                                                 
                                                        'merger_ratio_stars': galaxy_tree['%s' %GalaxyID]['merger_ratio_stars'][index_start:index_stop],
                                                        'merger_ratio_gas': galaxy_tree['%s' %GalaxyID]['merger_ratio_gas'][index_start:index_stop]}})
                                                    
                                                                                                                 
                # Add indexes adjusted to window: index_m = index_m - index_window_m, and relaxation time
                # Can access time of first misaligned as time_to_stack = misalignment_tree['12345']['Lookbacktime'][int(misalignment_tree['12345']['index_m'] + 1)]
                index_misalignment_start = int(index_dict['misalignment_locations']['misalign']['index'][misindex_i]) - int(index_dict['window_locations']['misalign']['index'][misindex_i])
                index_misalignment_end   = int(index_dict['misalignment_locations']['relax']['index'][misindex_i]) - int(index_dict['window_locations']['misalign']['index'][misindex_i])
                if use_merger_criteria:
                    index_merger_locations = np.array(index_dict['merger_locations']['index'][misindex_i]) - int(index_dict['window_locations']['misalign']['index'][misindex_i])
                else:
                    index_merger_locations = [] 
                relaxation_time_entry    = float(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_dict['misalignment_locations']['misalign']['index'][misindex_i]]) - float(galaxy_tree['%s' %GalaxyID]['Lookbacktime'][index_dict['misalignment_locations']['relax']['index'][misindex_i]])
                misalignment_tree['%s' %ID_i].update({'index_s': index_misalignment_start,       # last in stable regime
                                                      'index_r': index_misalignment_end,        # first back in stable regime (so +1 when we use a range eg. [index_m:index_r+1])
                                                      'index_merger': index_merger_locations,   # index of merger that meets criteria
                                                      'relaxation_time': relaxation_time_entry})
                relaxation_tdyn_entry    = misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
                relaxation_ttorque_entry = misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
                misalignment_tree['%s' %ID_i].update({'relaxation_tdyn': relaxation_tdyn_entry,
                                                      'relaxation_ttorque': relaxation_ttorque_entry})
                
                                                                                             
                # Find relaxation type
                if (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_s']] < misangle_threshold) and (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_r']] < misangle_threshold):
                    misalignment_tree['%s' %ID_i].update({'relaxation_type': 'co-co'})    
                elif (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_s']] < misangle_threshold) and (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_r']] > (180-misangle_threshold)):
                    misalignment_tree['%s' %ID_i].update({'relaxation_type': 'co-counter'})    
                elif (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_s']] > (180-misangle_threshold)) and (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_r']] < misangle_threshold):
                    misalignment_tree['%s' %ID_i].update({'relaxation_type': 'counter-co'})    
                elif (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_s']] > (180-misangle_threshold)) and (misalignment_tree['%s' %ID_i]['%s' %use_angle][misalignment_tree['%s' %ID_i]['index_r']] > (180-misangle_threshold)):
                    misalignment_tree['%s' %ID_i].update({'relaxation_type': 'counter-counter'})    
                
                
                # Find first truely misaligned snip, if none return first back
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    # relax to co-
                    index_m = np.where(np.array(misalignment_tree['%s' %ID_i][use_angle]) > (0 if peak_misangle == None else peak_misangle))[0][0]
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    # relax to counter-
                    index_m = np.where((180-np.array(misalignment_tree['%s' %ID_i][use_angle])) > (0 if peak_misangle == None else peak_misangle))[0][0]
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    # relax to co-
                    index_m = np.where(np.array(misalignment_tree['%s' %ID_i][use_angle]) > (0 if peak_misangle == None else peak_misangle))[0][0]
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    # relax to counter-
                    index_m = np.where((180-np.array(misalignment_tree['%s' %ID_i][use_angle])) > (0 if peak_misangle == None else peak_misangle))[0][0]
                misalignment_tree['%s' %ID_i].update({'index_m': index_m})
                
                
                # Find index of peak misalignment from where it relaxes to (-co or -counter)
                if (misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_r']] < misangle_threshold):
                    # relax to co-
                    misalignment_tree['%s' %ID_i].update({'index_peak': misalignment_tree['%s' %ID_i]['index_s'] + np.argmax(np.array(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))})
                    misalignment_tree['%s' %ID_i].update({'angle_peak': np.max(np.array(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))})
                elif (misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_r']] > (180-misangle_threshold)):
                    # relax to counter-
                    misalignment_tree['%s' %ID_i].update({'index_peak': misalignment_tree['%s' %ID_i]['index_s'] + np.argmax(180 - np.array(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))})
                    misalignment_tree['%s' %ID_i].update({'angle_peak': np.max(180 - np.array(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))})
                
            
                # Find relaxation morphology type (ETG-ETG)
                if (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][0:misalignment_tree['%s' %ID_i]['index_s']+1])) > morph_limits[1]) and (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_r']:])) > morph_limits[1]):
                    misalignment_tree['%s' %ID_i].update({'relaxation_morph': 'LTG-LTG'})   
                elif (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][0:misalignment_tree['%s' %ID_i]['index_s']+1])) > morph_limits[1]) and (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_r']:])) < morph_limits[0]):
                    misalignment_tree['%s' %ID_i].update({'relaxation_morph': 'LTG-ETG'})  
                elif (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][0:misalignment_tree['%s' %ID_i]['index_s']+1])) < morph_limits[0]) and (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_r']:])) > morph_limits[1]):
                    misalignment_tree['%s' %ID_i].update({'relaxation_morph': 'ETG-LTG'})  
                elif (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][0:misalignment_tree['%s' %ID_i]['index_s']+1])) < morph_limits[0]) and (np.mean(np.array(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_r']:])) < morph_limits[0]):
                    misalignment_tree['%s' %ID_i].update({'relaxation_morph': 'ETG-ETG'})  
                else:
                    misalignment_tree['%s' %ID_i].update({'relaxation_morph': 'other'})   
        
            
                # Find averaged morphology
                if averaged_morphology <= morph_limits[0]:
                    misalignment_tree['%s' %ID_i].update({'misalignment_morph': 'ETG'}) 
                elif averaged_morphology >= morph_limits[1]:
                    misalignment_tree['%s' %ID_i].update({'misalignment_morph': 'LTG'}) 
                else:
                    misalignment_tree['%s' %ID_i].update({'misalignment_morph': 'other'}) 
            
            
                #--------------------------------------
                # Collect misalignment selection
                if plot_misangle_detection:
                    print('\n\t\t\tADDED TO SAMPLE | Duration: %.2f Gyr' %relaxation_time_entry)
                    plot_misangle_accepted_window.append(misalignment_tree['%s' %ID_i][use_angle])
                    plot_misangle_accepted_window_t.append(misalignment_tree['%s' %ID_i]['Lookbacktime'])
                    plot_misangle_accepted_misangle.append(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
                    plot_misangle_accepted_misangle_t.append(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
                
        
            # PLot final misalignment selection        
            if plot_misangle_detection:
                if len(plot_misangle_accepted_window) > 0:
                    # Plot each accepted misangle of this galaxy
                    for time_window_i, angle_window_i, time_misangle_i, angle_misangle_i in zip(plot_misangle_accepted_window_t, plot_misangle_accepted_window, plot_misangle_accepted_misangle_t, plot_misangle_accepted_misangle):
                        p = axs.plot(time_window_i, angle_window_i, lw=7, alpha=0.4, zorder=1)
                        axs.plot(time_misangle_i, angle_misangle_i, 'o-', c=p[0].get_color(), mec='k', ms=1.3, zorder=999)
                
                    axs.text(1.8, 196, 'Window extracted', fontsize=8, color='lightblue')
                    axs.text(1.8, 190, 'Misalignment extracted', fontsize=8, color='r')
                    if savefig:
                        plt.savefig("%s/individual_misalignments/L%s_misalignment_ID%s_%s_%s%s.%s" %(fig_dir, output_input['mySims'][0][1], galaxy_tree['%s' %GalaxyID]['GalaxyID'][index_start:index_stop][0], use_angle, abs_or_proj, savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
                        print("\n  SAVED: %s/individual_misalignments/L%s_misalignment_ID%s_%s_%s%s.%s" %(fig_dir, output_input['mySims'][0][1], galaxy_tree['%s' %GalaxyID]['GalaxyID'][index_start:index_stop][0], use_angle, abs_or_proj, savefig_txt, file_format))
                    if showfig:
                        plt.show()
                    plt.close()
        
               
    #===================================================================================================
    # Load previous csv if asked for
    if load_csv_file:
        dict_tree = json.load(open('%s/L100_misalignment_tree_%s.csv' %(output_dir, load_csv_file), 'r'))
        misalignment_input = dict_tree['misalignment_input']
        sample_input       = dict_tree['sample_input']
        output_input       = dict_tree['output_input']
        misalignment_tree  = dict_tree['misalignment_tree']
        
        relaxation_type    = dict_tree['misalignment_input']['relaxation_type']
        relaxation_morph   = dict_tree['misalignment_input']['relaxation_morph']
        misalignment_morph = dict_tree['misalignment_input']['misalignment_morph']
        morph_limits       = dict_tree['misalignment_input']['morph_limits']
        peak_misangle      = dict_tree['misalignment_input']['peak_misangle']
        min_trelax         = dict_tree['misalignment_input']['min_trelax']        
        max_trelax         = dict_tree['misalignment_input']['max_trelax']
        
        savefig_txt = load_csv_file
    

    """ # Extract list of IDs matching criteria
    collect_IDs = []
    for ID_i in misalignment_tree.keys():
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 5:
            collect_IDs.append(ID_i)
    print('-------------------------------------------')
    print('Number of >5 ttorque relaxations:  ', len(collect_IDs))
    print(collect_IDs)
    """
    
    # Apply a merger criteria
    if use_alt_relaxation_morph:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] in use_alt_relaxation_morph:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    
    # Apply a min trelax criteria
    if use_alt_min_trelax:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_time'] > use_alt_min_trelax:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    if use_alt_min_tdyn:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > use_alt_min_tdyn:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    if use_alt_min_ttorque:
        misalignment_tree_new = {}
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > use_alt_min_ttorque:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
    
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
    
    
    # Apply a merger criteria
    if use_alt_merger_criteria:
        # Loading mergertree file to establish windows
        f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
        GalaxyID_tree             = np.array(f['Histories']['GalaxyID'])
        DescendantID_tree         = np.array(f['Histories']['DescendantID'])
        Lookbacktime_tree         = np.array(f['Snapnum_Index']['LookbackTime'])
        StellarMass_tree          = np.array(f['Histories']['StellarMass'])
        GasMass_tree              = np.array(f['Histories']['GasMass'])
        f.close()
        
        misalignment_tree_new = {}
        tally_minor = 0
        tally_major = 0
        tally_sample = len(misalignment_tree.keys())
        for ID_i in misalignment_tree.keys():
            """  
            index_merger_window = np.where(np.absolute(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1]) < half_window)[0]
    
            meets_criteria = False
            for merger_i in np.array(misalignment_tree['%s' %ID_i]['merger_ratio_stars'])[index_merger_window]:
                if len(merger_i) > 0:
                    if max(merger_i) > min_ratio:
                        meets_criteria = True
            if meets_criteria == False:
                continue
            else:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
            """
                
            #---------------------------------------------------------
            # Find location of begin of misalignment in merger tree
            (row_i, snap_i) = np.where(GalaxyID_tree == int(misalignment_tree['%s' %ID_i]['GalaxyID'][misalignment_tree['%s' %ID_i]['index_s']+1]))
            row_mask  = row_i[0]
            snap_mask = snap_i[0]
            
            # Find window limits of mergers [SnapNum_merger_min:SnapNum_merger_max]
            SnapNum_merger_min = 1 + np.where(Lookbacktime_tree >= (Lookbacktime_tree[snap_i] + half_window))[0][-1]
            if len(np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0]) > 0:
                SnapNum_merger_max = np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0][0]
            else:
                SnapNum_merger_max = snap_mask
            
            # List of all elligible descendants
            GalaxyID_list       = np.array(GalaxyID_tree)[row_mask, SnapNum_merger_min:SnapNum_merger_max]
            

            merger_ID_array_array    = []
            merger_ratio_array_array = []
            merger_gas_array_array   = []
            for SnapNum_i, GalaxyID_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), GalaxyID_list):
                if int(GalaxyID_i) == -1:
                    continue
                
                merger_mask = [i for i in np.where(np.array(DescendantID_tree)[:,int(SnapNum_i-1)] == GalaxyID_i)[0] if i != row_mask]
                
                # If misalignment found, its position is given by i in merger_mask, SnapNum_i
                merger_ID_array    = []
                merger_ratio_array = []
                merger_gas_array   = []
                if len(merger_mask) > 0:
                    # find peak stelmass of those galaxies
                    for mask_i in merger_mask:
                        # Find last snap up to 2 Gyr ago
                        SnapNum_merger = np.where(Lookbacktime_tree >= (Lookbacktime_tree[SnapNum_i] + merger_lookback_time))[0][-1]
                
                        # Find largest stellar mass of this satellite, per method of Rodriguez-Gomez et al. 2015, Qu et al. 2017 (see crain2017)
                        mass_mask = np.argmax(StellarMass_tree[mask_i][int(SnapNum_merger-100):int(SnapNum_i)]) + (SnapNum_merger-100)
                
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
                        if merger_ratio > 1:
                            merger_ratio = 1/merger_ratio
                        gas_ratio    = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)

                        # Append
                        merger_ID_array.append(GalaxyID_tree[mask_i][int(SnapNum_i-1)])
                        merger_ratio_array.append(merger_ratio)
                        merger_gas_array.append(gas_ratio)
                        
                merger_ID_array_array.append(merger_ID_array)
                merger_ratio_array_array.append(merger_ratio_array)
                merger_gas_array_array.append(merger_gas_array)      
            if debug:
                print(misalignment_tree['%s' %ID_i]['SnapNum'])
                print(misalignment_tree['%s' %ID_i]['merger_ratio_stars'])
                for snap_i, star_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), merger_ratio_array_array):
                    print(snap_i, star_i)
                    
                    
            meets_criteria = False
            for merger_i in merger_ratio_array_array:
                if len(merger_i) > 0:
                    if max(merger_i) > min_ratio:
                        meets_criteria = True
            if meets_criteria == False:
                continue
            else:
                misalignment_tree_new['%s' %ID_i] = misalignment_tree['%s' %ID_i]
                    
                
            merger_count = 0
            for merger_i in merger_ratio_array_array:
                if len(merger_i) > 0:
                    if (max(merger_i) > min_ratio):
                        if merger_count == 0:
                            if 0.3 > max(merger_i) > 0.1:
                                tally_minor += 1
                            if max(merger_i) > 0.3:
                                tally_major += 1
                        merger_count += 1
                        
                        
        misalignment_tree = misalignment_tree_new
        misalignment_tree_new = 0
        
        print('======================================')
        print('Using merger criteria, half_window = %.1f Gyr, min_ratio = %.1f' %(half_window, min_ratio))
        print('Original anyMerger sample:   %i\t' %(tally_sample))
        print('        Number of mergers:   %i\t%.2f %%' %(tally_minor+tally_major, (tally_minor+tally_major)*100/tally_sample))
        print('                 ...major:   %i\t%.2f %%' %(tally_major, (tally_major*100/tally_sample)))
        print('                 ...minor:   %i\t%.2f %%' %(tally_minor, (tally_minor*100/tally_sample)))
        
    """ 
    # Extract specific misalignment
    for ID_i in misalignment_tree.keys():
        if int(ID_i) in range(251899973, 251899973+80):            
            print('Found misalignment: %s\tLookbacktime: %.2f' %(ID_i, misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_m']]))
            print('\t%s' %misalignment_tree['%s' %ID_i]['relaxation_type'])
            print('\t%.2f Gyr' %misalignment_tree['%s' %ID_i]['relaxation_time'])
            print('\t%.2f trelax/tdyn' %misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            print('\t%.2f trelax/ttorque' %misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            print('\t%s' %misalignment_tree['%s' %ID_i]['relaxation_morph'])
            print('\t%.2f peak offset angle' %misalignment_tree['%s' %ID_i]['angle_peak'])
            print('\t%.2e Stellar mass, %.2e gas mass, %.1f pkpc hmr' %(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1], misalignment_tree['%s' %ID_i]['gasmass'][misalignment_tree['%s' %ID_i]['index_s']+1], misalignment_tree['%s' %ID_i]['rad'][misalignment_tree['%s' %ID_i]['index_s']+1]))
            for i, time_i, snap_i, angle_i in zip(np.arange(0, len(misalignment_tree['%s' %ID_i]['Lookbacktime'])), misalignment_tree['%s' %ID_i]['Lookbacktime'], misalignment_tree['%s' %ID_i]['SnapNum'], misalignment_tree['%s' %ID_i]['stars_gas_sf']):
                print('\t\t%.2f\t%i\t%.1f' %(time_i, snap_i, angle_i), misalignment_tree['%s' %ID_i]['merger_ratio_stars'][i], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][i], misalignment_tree['%s' %ID_i]['sfmass'][i])
            print(' ')
    """
    
    #------------------------------------------------ 
    plt.close()
    # Summary and K-S KS test
    if print_summary:
        summary_dict = {'trelax':       {'array': [],               # relaxation times in [Gyr]
                                         'co-co': [],
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []},     
                        'tdyn':         {'array': [],               # trelax/tdyn multiples
                                         'co-co': [],
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []},     
                        'ttorque':      {'array': [],               # trelax/ttorque multiples
                                         'co-co': [],
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []},    
                        'ID':           {'array': [],               # lists all IDs of galaxy types, for quicker calculations
                                         'co-co': [],               
                                         'counter-counter': [],
                                         'co-counter': [],
                                         'counter-co': [],
                                         'ETG': [],
                                         'ETG-ETG': [],
                                         'ETG-LTG': [],
                                         'LTG': [],
                                         'LTG-LTG': [],
                                         'LTG-ETG': []}}     
        

        for ID_i in misalignment_tree.keys():
            summary_dict['trelax']['array'].append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            summary_dict['tdyn']['array'].append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            summary_dict['ttorque']['array'].append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            summary_dict['ID']['array'].append(ID_i)
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                summary_dict['trelax']['co-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['co-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['co-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['co-co'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                summary_dict['trelax']['co-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['co-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['co-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['co-counter'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                summary_dict['trelax']['counter-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['counter-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['counter-co'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['counter-co'].append(ID_i) 
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                summary_dict['trelax']['counter-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['counter-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['counter-counter'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['counter-counter'].append(ID_i) 
            
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                summary_dict['trelax']['ETG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['ETG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['ETG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['ETG-ETG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                summary_dict['trelax']['LTG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['LTG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['LTG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['LTG-LTG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                summary_dict['trelax']['ETG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['ETG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['ETG-LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['ETG-LTG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                summary_dict['trelax']['LTG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['LTG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['LTG-ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['LTG-ETG'].append(ID_i)
                
            if misalignment_tree['%s' %ID_i]['misalignment_morph'] == 'ETG':
                summary_dict['trelax']['ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['ETG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['ETG'].append(ID_i)
            elif misalignment_tree['%s' %ID_i]['misalignment_morph'] == 'LTG':
                summary_dict['trelax']['LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']) 
                summary_dict['tdyn']['LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ttorque']['LTG'].append(misalignment_tree['%s' %ID_i]['relaxation_time']/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                summary_dict['ID']['LTG'].append(ID_i)
        

        # Average timescales
        mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
        median_timescale = np.median(np.array(summary_dict['trelax']['array']))
        std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
        # Average tdyn
        mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
        median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
        std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
        # Average ttorque
        mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
        median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
        std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
        
        # KS test on ETG and LTG   
        if (len(summary_dict['trelax']['ETG']) > 0) and (len(summary_dict['trelax']['LTG']) > 0):
            res1_trelax = stats.ks_2samp(summary_dict['trelax']['ETG'], summary_dict['trelax']['LTG'])
        if (len(summary_dict['trelax']['ETG-ETG']) > 0) and (len(summary_dict['trelax']['LTG-LTG']) > 0):
            res2_trelax = stats.ks_2samp(summary_dict['trelax']['ETG-ETG'], summary_dict['trelax']['LTG-LTG'])
        if (len(summary_dict['tdyn']['ETG']) > 0) and (len(summary_dict['tdyn']['LTG']) > 0):
            res1_tdyn = stats.ks_2samp(summary_dict['tdyn']['ETG'], summary_dict['tdyn']['LTG'])
        if (len(summary_dict['tdyn']['ETG-ETG']) > 0) and (len(summary_dict['tdyn']['LTG-LTG']) > 0):
            res2_tdyn = stats.ks_2samp(summary_dict['tdyn']['ETG-ETG'], summary_dict['tdyn']['LTG-LTG'])
        if (len(summary_dict['ttorque']['ETG']) > 0) and (len(summary_dict['ttorque']['LTG']) > 0):
            res1_ttorque = stats.ks_2samp(summary_dict['ttorque']['ETG'], summary_dict['ttorque']['LTG'])
        if (len(summary_dict['ttorque']['ETG-ETG']) > 0) and (len(summary_dict['ttorque']['LTG-LTG']) > 0):
            res2_ttorque = stats.ks_2samp(summary_dict['ttorque']['ETG-ETG'], summary_dict['ttorque']['LTG-LTG'])
            
        
        
        
        print('\nChecking if ETG_ETG or just ETG is good enough:')
        print('length of ETG array:', len(summary_dict['trelax']['ETG']), len(summary_dict['ID']['ETG']))
        print('length of ETG_ETG array:', len(summary_dict['trelax']['ETG-ETG']), len(summary_dict['ID']['ETG-ETG']))
        print('\tnumber of IDs in common:            ', len([ID_i for ID_i in summary_dict['ID']['ETG'] if ID_i in summary_dict['ID']['ETG-ETG']]))
        print('\tnumber of IDs in ETG but not ETG_ETG', len([ID_i for ID_i in summary_dict['ID']['ETG'] if ID_i not in summary_dict['ID']['ETG-ETG']]))
        print('\tnumber of IDs in ETG_ETG but not ETG', len([ID_i for ID_i in summary_dict['ID']['ETG-ETG'] if ID_i not in summary_dict['ID']['ETG']]))
        
        print('length of LTG array:', len(summary_dict['trelax']['LTG']), len(summary_dict['ID']['LTG']))
        print('length of LTG_LTG array:', len(summary_dict['trelax']['LTG-LTG']), len(summary_dict['ID']['LTG-LTG']))
        print('\tnumber of IDs in common:            ', len([ID_i for ID_i in summary_dict['ID']['LTG'] if ID_i in summary_dict['ID']['LTG-LTG']]))
        print('\tnumber of IDs in LTG but not LTG_LTG', len([ID_i for ID_i in summary_dict['ID']['LTG'] if ID_i not in summary_dict['ID']['LTG-LTG']]))
        print('\tnumber of IDs in LTG_LTG but not LTG', len([ID_i for ID_i in summary_dict['ID']['LTG-LTG'] if ID_i not in summary_dict['ID']['LTG']]))
        print('\n======================================')
        print('NUMBER OF MISALIGNMENTS RECORDED: ', len(misalignment_tree.keys()))    
        print('   co-co: %s \tcounter-counter: %s \tco-counter: %s \tcounter-co: %s' %(('n/a' if 'co-co' not in relaxation_type else len(summary_dict['ID']['co-co'])), ('n/a' if 'counter-counter' not in relaxation_type else len(summary_dict['ID']['counter-counter'])), ('n/a' if 'co-counter' not in relaxation_type else len(summary_dict['ID']['co-counter'])), ('n/a' if 'counter-co' not in relaxation_type else len(summary_dict['ID']['counter-co']))))
        print('   ETG-ETG: %s \tLTG-LTG: %s \tETG-LTG: %s \t\tLTG-ETG: %s' %(('n/a' if 'ETG-ETG' not in relaxation_morph else len(summary_dict['ID']['ETG-ETG'])), ('n/a' if 'LTG-LTG' not in relaxation_morph else len(summary_dict['ID']['LTG-LTG'])), ('n/a' if 'ETG-LTG' not in relaxation_morph else len(summary_dict['ID']['ETG-LTG'])), ('n/a' if 'LTG-ETG' not in relaxation_morph else len(summary_dict['ID']['LTG-ETG']))))
        print('   ETG: %s \tLTG: %s' %(('n/a' if 'ETG' not in misalignment_morph else len(summary_dict['ID']['ETG'])), ('n/a' if 'LTG' not in misalignment_morph else len(summary_dict['ID']['LTG']))))
        print(' ')
        print('Long relaxations:')
        print('   trelax > 1 Gyr:\t%i\t%.2f%%   |    trelax > 2 Gyr:\t%i\t%.2f%%' %((np.array(summary_dict['trelax']['array']) > 1).sum(), 100*(np.array(summary_dict['trelax']['array']) > 1).sum()/len(misalignment_tree.keys()), (np.array(summary_dict['trelax']['array']) > 2).sum(), 100*(np.array(summary_dict['trelax']['array']) > 2).sum()/len(misalignment_tree.keys())))
        print('      ETG: %i   LTG: %i\t        | ETG: %i   LTG: %i\t' %((np.array(summary_dict['trelax']['ETG']) > 1).sum(), (np.array(summary_dict['trelax']['LTG']) > 1).sum(), (np.array(summary_dict['trelax']['ETG']) > 2).sum(), (np.array(summary_dict['trelax']['LTG']) > 2).sum()))
        print('  ETG-ETG: %i   LTG-LTG: %i\t\t    | ETG-ETG: %i   LTG-LTG: %i\t' %((np.array(summary_dict['trelax']['ETG-ETG']) > 1).sum(), (np.array(summary_dict['trelax']['LTG-LTG']) > 1).sum(), (np.array(summary_dict['trelax']['ETG-ETG']) > 2).sum(), (np.array(summary_dict['trelax']['LTG-LTG']) > 2).sum()))
        print('  ETG-LTG: %i   LTG-ETG: %i\t\t    | ETG-LTG: %i   LTG-ETG: %i\t' %((np.array(summary_dict['trelax']['ETG-LTG']) > 1).sum(), (np.array(summary_dict['trelax']['LTG-ETG']) > 1).sum(), (np.array(summary_dict['trelax']['ETG-LTG']) > 2).sum(), (np.array(summary_dict['trelax']['LTG-ETG']) > 2).sum()))
        print('   tdyn   > 10   :\t%i\t%.2f%%   |    tdyn   > 20   :\t%i\t%.2f%%' %((np.array(summary_dict['tdyn']['array']) > 10).sum(), 100*(np.array(summary_dict['tdyn']['array']) > 10).sum()/len(misalignment_tree.keys()), (np.array(summary_dict['tdyn']['array']) > 20).sum(), 100*(np.array(summary_dict['tdyn']['array']) > 20).sum()/len(misalignment_tree.keys())))
        print('      ETG: %i   LTG: %i\t\t\t        | ETG: %i   LTG: %i\t' %((np.array(summary_dict['tdyn']['ETG']) > 10).sum(), (np.array(summary_dict['tdyn']['LTG']) > 10).sum(), (np.array(summary_dict['tdyn']['ETG']) > 20).sum(), (np.array(summary_dict['tdyn']['LTG']) > 20).sum()))
        print('  ETG-ETG: %i   LTG-LTG: %i\t\t    | ETG-ETG: %i   LTG-LTG: %i\t' %((np.array(summary_dict['tdyn']['ETG-ETG']) > 10).sum(), (np.array(summary_dict['tdyn']['LTG-LTG']) > 10).sum(), (np.array(summary_dict['tdyn']['ETG-ETG']) > 20).sum(), (np.array(summary_dict['tdyn']['LTG-LTG']) > 20).sum()))
        print('  ETG-LTG: %i   LTG-ETG: %i\t\t    | ETG-LTG: %i   LTG-ETG: %i\t' %((np.array(summary_dict['tdyn']['ETG-LTG']) > 10).sum(), (np.array(summary_dict['tdyn']['LTG-ETG']) > 10).sum(), (np.array(summary_dict['tdyn']['ETG-LTG']) > 20).sum(), (np.array(summary_dict['tdyn']['LTG-ETG']) > 20).sum()))
        print('   torque > 3   :\t%i\t%.2f%%    |    torque > 5   :\t%i\t%.2f%%' %((np.array(summary_dict['ttorque']['array']) > 3).sum(), 100*(np.array(summary_dict['ttorque']['array']) > 3).sum()/len(misalignment_tree.keys()), (np.array(summary_dict['ttorque']['array']) > 5).sum(), 100*(np.array(summary_dict['ttorque']['array']) > 5).sum()/len(misalignment_tree.keys())))
        print('      ETG: %i   LTG: %i\t\t\t        | ETG: %i   LTG: %i\t' %((np.array(summary_dict['ttorque']['ETG']) > 3).sum(), (np.array(summary_dict['ttorque']['LTG']) > 3).sum(), (np.array(summary_dict['ttorque']['ETG']) > 5).sum(), (np.array(summary_dict['ttorque']['LTG']) > 5).sum()))
        print('  ETG-ETG: %i   LTG-LTG: %i\t\t    | ETG-ETG: %i   LTG-LTG: %i\t' %((np.array(summary_dict['ttorque']['ETG-ETG']) > 3).sum(), (np.array(summary_dict['ttorque']['LTG-LTG']) > 3).sum(), (np.array(summary_dict['ttorque']['ETG-ETG']) > 5).sum(), (np.array(summary_dict['ttorque']['LTG-LTG']) > 5).sum()))
        print('  ETG-LTG: %i   LTG-ETG: %i\t\t    | ETG-LTG: %i   LTG-ETG: %i\t' %((np.array(summary_dict['ttorque']['ETG-LTG']) > 3).sum(), (np.array(summary_dict['ttorque']['LTG-ETG']) > 3).sum(), (np.array(summary_dict['ttorque']['ETG-LTG']) > 5).sum(), (np.array(summary_dict['ttorque']['LTG-ETG']) > 5).sum()))
        print(' ')
        print('Relaxation timescales:')
        print('   (Gyr)     all    ETG    LTG  ETG-ETG  LTG-LTG  ETG-LTG  LTG-ETG')
        print('   Mean:    %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['ETG']) == 0 else np.mean(np.array(summary_dict['trelax']['ETG']))), (math.nan if len(summary_dict['trelax']['LTG']) == 0 else np.mean(np.array(summary_dict['trelax']['LTG']))), (math.nan if len(summary_dict['trelax']['ETG-ETG']) == 0 else np.mean(np.array(summary_dict['trelax']['ETG-ETG']))), (math.nan if len(summary_dict['trelax']['LTG-LTG']) == 0 else np.mean(np.array(summary_dict['trelax']['LTG-LTG']))), (math.nan if len(summary_dict['trelax']['ETG-LTG']) == 0 else np.mean(np.array(summary_dict['trelax']['ETG-LTG']))), (math.nan if len(summary_dict['trelax']['LTG-ETG']) == 0 else np.mean(np.array(summary_dict['trelax']['LTG-ETG'])))))   
        print('   Median:  %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['ETG']) == 0 else np.median(np.array(summary_dict['trelax']['ETG']))), (math.nan if len(summary_dict['trelax']['LTG']) == 0 else np.median(np.array(summary_dict['trelax']['LTG']))), (math.nan if len(summary_dict['trelax']['ETG-ETG']) == 0 else np.median(np.array(summary_dict['trelax']['ETG-ETG']))), (math.nan if len(summary_dict['trelax']['LTG-LTG']) == 0 else np.median(np.array(summary_dict['trelax']['LTG-LTG']))), (math.nan if len(summary_dict['trelax']['ETG-LTG']) == 0 else np.median(np.array(summary_dict['trelax']['ETG-LTG']))), (math.nan if len(summary_dict['trelax']['LTG-ETG']) == 0 else np.median(np.array(summary_dict['trelax']['LTG-ETG'])))))   
        print('   std:     %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['ETG']) == 0 else np.std(np.array(summary_dict['trelax']['ETG']))), (math.nan if len(summary_dict['trelax']['LTG']) == 0 else np.std(np.array(summary_dict['trelax']['LTG']))), (math.nan if len(summary_dict['trelax']['ETG-ETG']) == 0 else np.std(np.array(summary_dict['trelax']['ETG-ETG']))), (math.nan if len(summary_dict['trelax']['LTG-LTG']) == 0 else np.std(np.array(summary_dict['trelax']['LTG-LTG']))), (math.nan if len(summary_dict['trelax']['ETG-LTG']) == 0 else np.std(np.array(summary_dict['trelax']['ETG-LTG']))), (math.nan if len(summary_dict['trelax']['LTG-ETG']) == 0 else np.std(np.array(summary_dict['trelax']['LTG-ETG'])))))
        if (len(summary_dict['ID']['ETG']) > 0) and (len(summary_dict['ID']['LTG']) > 0):
            print('K-S TEST FOR ETG and LTG general:    %s %s' %(len(summary_dict['ID']['ETG']), len(summary_dict['ID']['LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res1_trelax.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG']) + len(summary_dict['ID']['LTG']))/(len(summary_dict['ID']['ETG'])*len(summary_dict['ID']['LTG']))))))
            print('   p-value: %s' %res1_trelax.pvalue)
        if (len(summary_dict['ID']['ETG-ETG']) > 0) and (len(summary_dict['ID']['LTG-LTG']) > 0):
            print('K-S TEST FOR ETG-ETG and LTG-LTG general:    %s %s' %(len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res2_trelax.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG-ETG']) + len(summary_dict['ID']['LTG-LTG']))/(len(summary_dict['ID']['ETG-ETG'])*len(summary_dict['ID']['LTG-LTG']))))))
            print('   p-value: %s' %res2_trelax.pvalue)
        print(' ')
        print('trelax/tdyn multiples:')
        print('             all    ETG    LTG  ETG-ETG  LTG-LTG  ETG-LTG  LTG-ETG')
        print('   Mean:    %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['ETG']) == 0 else np.mean(np.array(summary_dict['tdyn']['ETG']))), (math.nan if len(summary_dict['tdyn']['LTG']) == 0 else np.mean(np.array(summary_dict['tdyn']['LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-ETG']) == 0 else np.mean(np.array(summary_dict['tdyn']['ETG-ETG']))), (math.nan if len(summary_dict['tdyn']['LTG-LTG']) == 0 else np.mean(np.array(summary_dict['tdyn']['LTG-LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-LTG']) == 0 else np.mean(np.array(summary_dict['tdyn']['ETG-LTG']))), (math.nan if len(summary_dict['tdyn']['LTG-ETG']) == 0 else np.mean(np.array(summary_dict['tdyn']['LTG-ETG'])))))   
        print('   Median:  %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['ETG']) == 0 else np.median(np.array(summary_dict['tdyn']['ETG']))), (math.nan if len(summary_dict['tdyn']['LTG']) == 0 else np.median(np.array(summary_dict['tdyn']['LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-ETG']) == 0 else np.median(np.array(summary_dict['tdyn']['ETG-ETG']))), (math.nan if len(summary_dict['tdyn']['LTG-LTG']) == 0 else np.median(np.array(summary_dict['tdyn']['LTG-LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-LTG']) == 0 else np.median(np.array(summary_dict['tdyn']['ETG-LTG']))), (math.nan if len(summary_dict['tdyn']['LTG-ETG']) == 0 else np.median(np.array(summary_dict['tdyn']['LTG-ETG'])))))   
        print('   std:     %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(std_tdyn, (math.nan if len(summary_dict['tdyn']['ETG']) == 0 else np.std(np.array(summary_dict['tdyn']['ETG']))), (math.nan if len(summary_dict['tdyn']['LTG']) == 0 else np.std(np.array(summary_dict['tdyn']['LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-ETG']) == 0 else np.std(np.array(summary_dict['tdyn']['ETG-ETG']))), (math.nan if len(summary_dict['tdyn']['LTG-LTG']) == 0 else np.std(np.array(summary_dict['tdyn']['LTG-LTG']))), (math.nan if len(summary_dict['tdyn']['ETG-LTG']) == 0 else np.std(np.array(summary_dict['tdyn']['ETG-LTG']))), (math.nan if len(summary_dict['tdyn']['LTG-ETG']) == 0 else np.std(np.array(summary_dict['tdyn']['LTG-ETG'])))))
        if (len(summary_dict['ID']['ETG']) > 0) and (len(summary_dict['ID']['LTG']) > 0):
            print('K-S TEST FOR ETG and LTG general:    %s %s' %(len(summary_dict['ID']['ETG']), len(summary_dict['ID']['LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res1_tdyn.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG']) + len(summary_dict['ID']['LTG']))/(len(summary_dict['ID']['ETG'])*len(summary_dict['ID']['LTG']))))))
            print('   p-value: %s' %res1_tdyn.pvalue)
        if (len(summary_dict['ID']['ETG-ETG']) > 0) and (len(summary_dict['ID']['LTG-LTG']) > 0):
            print('K-S TEST FOR ETG-ETG and LTG-LTG general:    %s %s' %(len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res2_tdyn.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG-ETG']) + len(summary_dict['ID']['LTG-LTG']))/(len(summary_dict['ID']['ETG-ETG'])*len(summary_dict['ID']['LTG-LTG']))))))
            print('   p-value: %s' %res2_tdyn.pvalue)
        print(' ')    
        print('trelax/ttorque multiples:')
        print('             all    ETG    LTG  ETG-ETG  LTG-LTG  ETG-LTG  LTG-ETG')
        print('   Mean:    %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['ETG']) == 0 else np.mean(np.array(summary_dict['ttorque']['ETG']))), (math.nan if len(summary_dict['ttorque']['LTG']) == 0 else np.mean(np.array(summary_dict['ttorque']['LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-ETG']) == 0 else np.mean(np.array(summary_dict['ttorque']['ETG-ETG']))), (math.nan if len(summary_dict['ttorque']['LTG-LTG']) == 0 else np.mean(np.array(summary_dict['ttorque']['LTG-LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-LTG']) == 0 else np.mean(np.array(summary_dict['ttorque']['ETG-LTG']))), (math.nan if len(summary_dict['ttorque']['LTG-ETG']) == 0 else np.mean(np.array(summary_dict['ttorque']['LTG-ETG'])))))   
        print('   Median:  %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['ETG']) == 0 else np.median(np.array(summary_dict['ttorque']['ETG']))), (math.nan if len(summary_dict['ttorque']['LTG']) == 0 else np.median(np.array(summary_dict['ttorque']['LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-ETG']) == 0 else np.median(np.array(summary_dict['ttorque']['ETG-ETG']))), (math.nan if len(summary_dict['ttorque']['LTG-LTG']) == 0 else np.median(np.array(summary_dict['ttorque']['LTG-LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-LTG']) == 0 else np.median(np.array(summary_dict['ttorque']['ETG-LTG']))), (math.nan if len(summary_dict['ttorque']['LTG-ETG']) == 0 else np.median(np.array(summary_dict['ttorque']['LTG-ETG'])))))   
        print('   std:     %.2f   %.2f   %.2f   %.2f     %.2f     %.2f     %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['ETG']) == 0 else np.std(np.array(summary_dict['ttorque']['ETG']))), (math.nan if len(summary_dict['ttorque']['LTG']) == 0 else np.std(np.array(summary_dict['ttorque']['LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-ETG']) == 0 else np.std(np.array(summary_dict['ttorque']['ETG-ETG']))), (math.nan if len(summary_dict['ttorque']['LTG-LTG']) == 0 else np.std(np.array(summary_dict['ttorque']['LTG-LTG']))), (math.nan if len(summary_dict['ttorque']['ETG-LTG']) == 0 else np.std(np.array(summary_dict['ttorque']['ETG-LTG']))), (math.nan if len(summary_dict['ttorque']['LTG-ETG']) == 0 else np.std(np.array(summary_dict['ttorque']['LTG-ETG'])))))
        if (len(summary_dict['ID']['ETG']) > 0) and (len(summary_dict['ID']['LTG']) > 0):
            print('K-S TEST FOR ETG and LTG general:    %s %s' %(len(summary_dict['ID']['ETG']), len(summary_dict['ID']['LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res1_ttorque.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG']) + len(summary_dict['ID']['LTG']))/(len(summary_dict['ID']['ETG'])*len(summary_dict['ID']['LTG']))))))
            print('   p-value: %s' %res1_ttorque.pvalue)
        if (len(summary_dict['ID']['ETG-ETG']) > 0) and (len(summary_dict['ID']['LTG-LTG']) > 0):
            print('K-S TEST FOR ETG-ETG and LTG-LTG general:    %s %s' %(len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG'])))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res2_ttorque.statistic, (1.358*np.sqrt((len(summary_dict['ID']['ETG-ETG']) + len(summary_dict['ID']['LTG-LTG']))/(len(summary_dict['ID']['ETG-ETG'])*len(summary_dict['ID']['LTG-LTG']))))))
            print('   p-value: %s' %res2_ttorque.pvalue)
    print('======================================')
   
        
    if len(misalignment_tree.keys()) == 0:
        print('Insufficient galaxies meeting criteria')
        quit()
    
    """ GUIDE TO misalignment_tree['%s' %GalaxyID] VALUES:

    GalaxyID is first ID in window of misalignment.
    Arrays contain no missing entries

    Mergers searched for from first misaligned [index_m+1], to first relax [index_r] -> for range from relax to relax do [index_m:index_r+1]
    Values specified for general (eg. stelmass) use_hmr_general (typically 2.0)

    'GalaxyID'				-
    'SnapNum'				-
    'Redshift'				-
    'Lookbacktime'			- [ Gyr ]

    'SubGroupNum'			- 
    'halomass'				- [ Msun ]
    'stelmass'				- [ Msun ]
    'gasmass'				- [ Msun ]
    'sfmass'				- [ Msun ]
    'nsfmass'				- [ Msun ]				
    'dmmass'				- [ Msun ] in 30pkpc

    'sfr'					- [ Msun/yr]
    'ssfr'					- [ /yr ]

    'stars_l'               - [ pkpc/kms-1 ]
    
    'stars_Z'				-
    'gas_Z'					-
    'sf_Z'					-	
    'nsf_Z'					-

    'kappa_stars'			-
    'kappa_gas'				-
    'kappa_sf'				-
    'kappa_nsf'				-
    'ellip'					-
    'triax'					-
    'disp_ani'				-
    'disc_to_total'			-
    'rot_to_disp_ratio'		-

    'rad'					- [ pkpc ]
    'radproj'				- [ pkpc ]
    'rad_sf'				- [ pkpc ]

    'vcirc'					- [ km/s ]
    'tdyn'					- [ Gyr ]
    'ttorque'				- [ Gyr ] 


    'inflow_mass_1hmr'			- [ Msun ]
    'inflow_mass_2hmr'			- [ Msun ]
    'outflow_mass_1hmr'			- [ Msun ]
    'outflow_mass_2hmr'			- [ Msun ]
    'insitu_mass_1hmr'			- [ Msun ]
    'insitu_mass_2hmr'			- [ Msun ]
    'inflow_rate_1hmr'			- [ Msun/yr ]
    'inflow_rate_2hmr'			- [ Msun/yr ]
    'outflow_rate_1hmr'			- [ Msun/yr ]
    'outflow_rate_2hmr'			- [ Msun/yr ]
    'stelmassloss_rate_1hmr'	- [ Msun/yr ]
    'stelmassloss_rate_2hmr'	- [ Msun/yr ]
    's_inflow_rate_1hmr'		- [ /yr ]
    's_inflow_rate_2hmr'		- [ /yr ]
    's_outflow_rate_1hmr'		- [ /yr ]
    's_outflow_rate_2hmr'		- [ /yr ]
    'inflow_Z_1hmr'				-
    'inflow_Z_2hmr'				-
    'outflow_Z_1hmr'			-
    'outflow_Z_2hmr'			-
    'insitu_Z_1hmr'				-
    'insitu_Z_2hmr'				-

    'bh_mass'				- [ Msun ]
    'bh_mdot_av'			- [ Msun/yr ]
    'bh_mdot_inst'			- [ Msun/yr ]
    'bh_edd'				-
    'bh_lbol'				- [ erg/s ]

    'stars_gas_sf'			- all in degrees
    'stars_gas_sf_err'		-
    'stars_gas_sf_halo'		- inner stars vs outer gas_sf
    'stars_dm'				- not checked for inclination
    'stars_dm_err'			-
    'gas_dm'				- not checked for inclination
    'gas_dm_err'			-
    'gas_sf_dm'				- not checked for inclination
    'gas_sf_dm_err'			-

    'merger_ratio_stars'	- Includes ALL mergers in window
    'merger_ratio_gas'		- ^

    # single values:
    'index_s'				- Last before leaving stable regime, adjusted for current window
    'index_m'				- First index to be misaligned by more than 30 degrees
    'index_r'				- First to be relaxed back in stable regime
    'index_merger' 			- Index of mergers that met locations. Use this to sample 'merger_ratio_stars'
    'relaxation_time' 		- [ Gyr ] Time between index_m and index_r
    'relaxation_tdyn'		- [fraction] trelax/tdyn between index_m and index_r
    'relaxation_ttorque'	- [fraction] trelax/ttorque between index_m and index_r
    'relaxation_type'		- co-co, counter-counter, co-counter, counter-co
    'relaxation_morph'		- ETG-ETG, LTG-LTG, ETG-LTG, LTG-ETG, other					BASED OFF MORPH_LIMITS
    'misalignment_morph'	- ETG, LTG, other
    'angle_peak'			- peak misalignment angle from where it relaxes to (-co (0 deg), -counter (180 deg))
    'index_peak'			- index of the above w.r.t array
	
    
    """
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
                  
        # Combining all dictionaries
        csv_dict = {'misalignment_tree': misalignment_tree}
        
        misalignment_input = {'force_all_snaps': force_all_snaps,
                              'max_z': max_z,
                              'min_z': min_z,
                              'use_hmr_general': use_hmr_general,
                              'max_com': max_com,
                              'max_uncertainty': max_uncertainty,
                              'min_inclination': min_inclination,
                              'limit_satellites': limit_satellites,
                              'min_halomass': min_halomass,                 'max_halomass': max_halomass,
                              'min_stelmass': min_stelmass,                 'max_stelmass': max_stelmass,
                              'min_gasmass ': min_gasmass,                  'max_gasmass ': max_gasmass,
                              'min_sfmass ': min_gasmass,                   'max_sfmass ': max_gasmass,
                              'min_nsfmass ': min_gasmass,                  'max_nsfmass ': max_gasmass,
                              'min_sfr': min_sfr,                           'max_sfr': max_sfr,
                              'min_ssfr ': min_ssfr,                        'max_ssfr ': max_ssfr,
                              'min_kappa_stars': min_kappa_stars,           'max_kappa_stars': max_kappa_stars,
                              'min_kappa_gas': min_kappa_gas,               'max_kappa_gas': max_kappa_gas,
                              'min_kappa_sf': min_kappa_sf,                 'max_kappa_sf': max_kappa_sf,
                              'min_kappa_nsf': min_kappa_nsf,               'max_kappa_nsf': max_kappa_nsf,
                              'min_ellip': min_ellip,                       'max_ellip': max_ellip,
                              'min_triax': min_triax,                       'max_triax': max_triax,
                              'min_rad': min_rad,                           'max_rad': max_rad,
                              'min_rad_sf': min_rad_sf,                     'max_rad_sf': max_rad_sf,
                              'min_inflow': min_inflow,                     'max_inflow': max_inflow,
                              'min_inflow_Z': min_inflow_Z,                 'max_inflow_Z ': max_inflow_Z ,
                              'force_steady_bh': force_steady_bh,
                              'min_bh_mass': min_bh_mass,                   'max_bh_mass': max_bh_mass,
                              'min_bh_acc': min_bh_acc,                     'max_bh_acc': max_bh_acc,
                              'min_bh_acc_instant': min_bh_acc_instant,     'max_bh_acc_instant': max_bh_acc_instant,
                              'min_edd': min_edd,                           'max_edd': max_edd,
                              'min_lbol': min_lbol,                         'max_lbol': max_lbol,
                              'min_vcirc': min_vcirc,                       'max_vcirc': max_vcirc,
                              'min_tdyn': min_tdyn,                         'max_tdyn': max_tdyn,
                              'min_ttorque': min_ttorque,                   'max_ttorque': max_ttorque,
                              'use_hmr_angle': use_hmr_angle,
                              'abs_or_proj': abs_or_proj,
                              'min_particles': min_particles,
                              'use_angle': use_angle,
                              'misangle_threshold': misangle_threshold,
                              'min_delta_angle': min_delta_angle,
                              'use_merger_criteria': use_merger_criteria,
                              'min_stellar_ratio': min_stellar_ratio,       'max_stellar_ratio': max_stellar_ratio,
                              'min_gas_ratio': min_gas_ratio,               'max_gas_ratio': max_gas_ratio,
                              'max_merger_pre': max_merger_pre,             'max_merger_post': max_merger_post,
                              'latency_time': latency_time,
                              'time_extra': time_extra,
                              'time_no_misangle': time_no_misangle,
                              'relaxation_type': relaxation_type,
                              'relaxation_morph': relaxation_morph,
                              'misalignment_morph': misalignment_morph, 
                              'morph_limits': morph_limits,   
                              'peak_misangle': peak_misangle,
                              'min_trelax': min_trelax,        
                              'max_trelax': max_trelax}    
        
        
        csv_dict.update({'misalignment_input': misalignment_input,
                         'tree_input': tree_input,
                         'output_input': output_input,
                         'sample_input': sample_input})
        
        #-----------------------------
        csv_answer = input("\n  > Choose csv name L100_misalignment_tree_...  '_' ... or 'n' to cancel       ")
        if csv_answer != 'n':
            # Writing one massive JSON file
            json.dump(csv_dict, open('%s/L100_misalignment_tree_%s%s.csv' %(output_dir, csv_answer, csv_name), 'w'), cls=NumpyEncoder)
            print('\n  SAVED: %s/L100_misalignment_tree_%s%s.csv' %(output_dir, csv_answer, csv_name))
    
    # clear memory
    galaxy_tree = 0 
    
    
    #==================================================================================================
    # PLOTS
    
    #-------------------------
    # Plot sample histogram of misalignments extracted
    if _plot_sample_hist:
        # Gather data, average stelmass over misalignment
        stelmass_plot = []
        for ID_i in misalignment_tree.keys():
            stelmass_plot.append(np.log10(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
        
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        axs.hist(stelmass_plot, bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), histtype='bar', edgecolor='none', facecolor='b', alpha=0.1)
        axs.hist(stelmass_plot, bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), histtype='bar', edgecolor='b', facecolor='none', alpha=1.0)
        
        print('Median stellar mass: \t%.2e.' %(10**np.median(stelmass_plot)))
        
        #-------------
        ### Formatting
        axs.set_xlabel(r'log$_{10}$ M$_{*}$ ($2r_{50}$) [M$_{\odot}$]')
        axs.set_ylabel('Galaxies in sample')
        axs.set_xticks(np.arange(9, 12.1, 0.5))
        axs.set_xlim(9, 12.1)
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        # savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/time_spent_misaligned/%ssample_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/time_spent_misaligned/%ssample_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
            

    #-------------------------
    # Plot timescale histogram of current criteria
    if _plot_timescale_histogram:
        # Gather data
        relaxationtime_plot = []
        co_co_array           = []
        co_counter_array      = []
        counter_co_array      = []
        counter_counter_array = []
        collect_array         = []
        for ID_i in misalignment_tree.keys():
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
                
                
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                if misalignment_tree['%s' %ID_i]['angle_peak'] > 135:
                    collect_array.append(ID_i)
        
        #print('-------------------------------------------------------------')
        #print('Number of >135 co-co misalignments: ', len(collect_array))
        #print(collect_array)
        print('\nMax trelax:  %.2f Gyr' %max(relaxationtime_plot))  
        
        
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
                
        if set_bin_limit_trelax == None:
            set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
        if set_plot_histogram_log:
            set_plot_relaxation_type = False 
        
        #-------------
        ### Plot histogram
        if set_plot_relaxation_type:
            if set_plot_percentage:
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
            
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
            else:
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
            
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        else:
            if set_plot_percentage:
                axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
                axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
            else:
                axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
                axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        #-------------
        ### Inset second axes
        if add_inset:
            axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
            
            if set_plot_percentage:
                axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
            else: 
                axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
            
            print('bin_count of hist:', bin_count)
            
            # Add uncertainty 
            raise Exception('do below')
            
            
            
            #should be able to use bin_count here i think...
            
            if set_plot_percentage:
                axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
                axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
            else:
                axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
                axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
            
            
            
            
            
            
            
            
            
            
            
            axins.set_yscale('log')
            axins.set_xlim(0, set_bin_limit_trelax)
            axins.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
            axins.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)', fontsize = 5)
            if set_plot_percentage:
                axins.set_ylim(0.0002, set_thist_ymax_trelax)
                axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
                axins.set_ylabel('Percentage of misalignments', fontsize=5)
            else:
                axins.set_ylim(0.5, set_thist_ymax_trelax)
                axins.set_ylabel('Number of misalignments', fontsize=5)
                
                
                
            #-----------
            # Best-fit data
            mask = np.where(np.array(bin_count) > 0)[0]
            yerr = np.sqrt(np.array(bin_count))[mask]
            
            print('bin_count, error:')
            print(bin_count)
            print(yerr)
            
            #slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
            
            
            # Define linear bestfit
            def func(x, a, b):
                return (a*x) + b
            
            popt, pcov = scipy.optimize.curve_fit(f=func, xdata=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax)[mask][1:]. ydata=set_bin_width_trelax)[mask][1:], np.log10(np.array(bin_count)[mask])[1:], sigma=yerr)
            print('\nauhbbebofjn', popt)
            slope = popt[0]
            intercept = popt[1]
            
            

            print('Best fit line: \tfrac = %.2f x 10^(%.2f t)' %(10**intercept, slope))
            
            
            
            
            if add_inset_bestfit:
                axins.plot([0.1 , 5], [(10**intercept) * (10**(slope*0.1)), (10**intercept) * (10**(slope*5))], lw=0.7, ls='--', alpha=0.8, c='purple', label='best-fit')
                #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
        
        
        #-----------
        ### General formatting
        # Axis labels
        if set_plot_histogram_log:
            axs.set_yscale('log')
        if set_plot_percentage:
            axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
        axs.set_xlim(0, set_bin_limit_trelax)
        axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
        if set_plot_percentage:
            axs.set_ylabel('Percentage of misalignments')
        else:
            axs.set_ylabel('Number of misalignments')
        axs.set_ylim(0, set_thist_ymax_trelax)
        
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
    
        # Add any selection criteria if given
        if min_halomass != None or max_halomass != None:
            if min_halomass == 1E14:
                legend_labels.append('Group')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('coral')
            elif max_halomass == 1E14:
                legend_labels.append('Field')  
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('coral')  
        if limit_satellites != None:
            legend_labels.append('%ss' %limit_satellites)
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('coral')
        if min_stelmass != None or max_stelmass != None:
            legend_labels.append('%1e>M$_{*}$>%1e' %(0 if max_stelmass == None else max_stelmass, 0 if min_stelmass == None else min_stelmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_gasmass != None or max_gasmass != None:
            legend_labels.append('%1e>M$_{\mathrm{gas}}$>%1e' %(0 if max_gasmass == None else max_gasmass, 0 if min_gasmass == None else min_gasmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')    
        if min_sfmass != None or max_sfmass != None:
            legend_labels.append('%1e>M$_{\mathrm{SF}}$>%1e' %(0 if max_sfmass == None else max_sfmass, 0 if min_sfmass == None else min_sfmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_nsfmass != None or max_nsfmass != None:
            legend_labels.append('%1e>M$_{\mathrm{SF}}$>%1e' %(0 if max_nsfmass == None else max_nsfmass, 0 if min_nsfmass == None else min_nsfmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_sfr != None or max_sfr != None:
            legend_labels.append('%1e>SFR>%1e' %(0 if max_sfr == None else max_sfr, 0 if min_sfr == None else min_sfr))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('teal')
        if min_ssfr != None or max_ssfr != None:
            legend_labels.append('%.1f>sSFR>%.1f' %(0 if max_ssfr == None else max_ssfr, 0 if min_ssfr == None else min_ssfr))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('teal')
        if min_kappa_stars != None or max_kappa_stars != None:
            if min_kappa_stars != None:
                legend_labels.append('$\kappa_{\mathrm{co}}^{\mathrm{*}}>$%s' %(0 if min_kappa_stars == None else min_kappa_stars))
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('cornflowerblue')
            if max_kappa_stars != None:
                legend_labels.append('$\kappa_{\mathrm{co}}^{\mathrm{*}}<$%s' %(0 if max_kappa_stars == None else max_kappa_stars))
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('cornflowerblue')
        if min_kappa_gas != None or max_kappa_gas != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{gas}}>$%.1f' %(0 if max_kappa_gas == None else max_kappa_gas, 0 if min_kappa_gas == None else min_kappa_gas))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_kappa_sf != None or max_kappa_sf != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{SF}}>$%.1f' %(0 if max_kappa_sf == None else max_kappa_sf, 0 if min_kappa_sf == None else min_kappa_sf))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_kappa_nsf != None or max_kappa_nsf != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{NSF}}>$%.1f' %(0 if max_kappa_nsf == None else max_kappa_nsf, 0 if min_kappa_nsf == None else min_kappa_nsf))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_rad != None or max_rad != None:
            legend_labels.append('%.1f>R>%.1f' %(0 if max_rad == None else max_rad, 0 if min_rad == None else min_rad))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('peru')
        if min_inflow != None or max_inflow != None:
            legend_labels.append('%.1f>inflow>%.1f' %(0 if max_inflow == None else max_inflow, 0 if min_inflow == None else min_inflow))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('darkgreen')
        if min_inflow_Z != None or max_inflow_Z != None:
            legend_labels.append('%.1f>inflow Z>%.1f' %(0 if max_inflow_Z == None else max_inflow_Z, 0 if min_inflow_Z == None else min_inflow_Z))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('darkgreen')
        
        # add z
        #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
        #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        #legend_colors.append('k')
        
        if set_plot_relaxation_type:
            if 'co-co' in relaxation_type:
                legend_labels.append('     co → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C0')
            if 'counter-counter' in relaxation_type:
                legend_labels.append('counter → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C1')
            if 'co-counter' in relaxation_type:
                legend_labels.append('     co → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C2')
            if 'counter-co' in relaxation_type:
                legend_labels.append('counter → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C3')
        if add_inset:
            ncol=2
        else:
            ncol=1
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #-----------
        ### other
        plt.tight_layout()
        
        
        #-----------
        ### Savefig
        if add_inset:
            savefig_txt_2 = savefig_txt + '_inset'
        else:
            savefig_txt_2 = savefig_txt
            
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr\nmax: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale, max(relaxationtime_plot)),
                         'Producer': str(hist_n)}
                         
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/time_spent_misaligned/%stime_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/time_spent_misaligned/%stime_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    # tdyn histogram    
    if _plot_tdyn_histogram:
        # Gather data
        relaxationtime_plot = []
        co_co_array           = []
        co_counter_array      = []
        counter_co_array      = []
        counter_counter_array = []
        for ID_i in misalignment_tree.keys():
            # append average tdyn over misalignment
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
                
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 20:
                print('\tFound >20 tdyn:   ID: %s\tMass: %.2e Msun' %(ID_i, np.mean(misalignment_tree['%s' %ID_i]['stelmass'])))
                print('\t  type: %s, morph: %s, time: %.2f, tdyn: %.2f, ttorque: %.2f' %(misalignment_tree['%s' %ID_i]['relaxation_type'], misalignment_tree['%s' %ID_i]['relaxation_morph'], misalignment_tree['%s' %ID_i]['relaxation_time'], misalignment_tree['%s' %ID_i]['relaxation_tdyn'], misalignment_tree['%s' %ID_i]['relaxation_ttorque']))
            
        print('\nMax tdyn/trelax:  %.2f' %max(relaxationtime_plot)) 
                
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
                
        if set_bin_limit_tdyn == None:
            set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
        if set_plot_histogram_log:
            set_plot_relaxation_type = False 
        
        #-------------
        ### Plot histogram
        if set_plot_relaxation_type:
            if set_plot_percentage:
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
            
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
            else:
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
            
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        else:
            if set_plot_percentage:
                axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
                axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn+0.5*set_bin_width_tdyn, set_bin_width_tdyn), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
            else:
                axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
                axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn+0.5*set_bin_width_tdyn, set_bin_width_tdyn), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        #-------------
        ### Inset second axes
        if add_inset:
            axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
            
            if set_plot_percentage:
                axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
            else:
                axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
            
            print('bin_count of hist:', bin_count)
            
            axins.set_yscale('log')
            axins.set_xlim(0, set_bin_limit_tdyn)
            axins.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
            axins.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$', fontsize=5)
            
            if set_plot_percentage:
                axins.set_ylim(0.0002, set_thist_ymax_tdyn)
                axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
                axins.set_ylabel('Percentage of misalignments', fontsize=5)
            else:
                axins.set_ylim(0.5, set_thist_ymax_tdyn)
                axins.set_ylabel('Number of misalignments', fontsize=5)
                
            #-------------
            # Best-fit data
            mask = np.where(np.array(bin_count) > 0)[0]
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
            print('Best fit line: \tfrac = %.2f x 10^(%.2f t)' %(10**intercept, slope))
            
            if add_inset_bestfit:
                axins.plot([0.1 , 50], [(10**intercept) * (10**(slope*0.1)), (10**intercept) * (10**(slope*50))], lw=0.7, ls='--', alpha=0.8, c='purple', label='best-fit')
                #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
            
            
        #-----------
        ### General formatting
        # Axis labels
        if set_plot_histogram_log:
            axs.set_yscale('log')
        if set_plot_percentage:
            axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
        axs.set_xlim(0, set_bin_limit_tdyn)
        axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=2))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        if set_plot_percentage:
            axs.set_ylabel('Percentage of misalignments')
        else:
            axs.set_ylabel('Number of misalignments')
        axs.set_ylim(0, set_thist_ymax_tdyn)
    
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
    
        # Add any selection criteria if given
        if min_halomass != None or max_halomass != None:
            if min_halomass == 1E14:
                legend_labels.append('Group')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('coral')
            elif max_halomass == 1E14:
                legend_labels.append('Field')  
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('coral')  
        if limit_satellites != None:
            legend_labels.append('%ss' %limit_satellites)
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('coral')
        if min_stelmass != None or max_stelmass != None:
            legend_labels.append('%1e>M$_{*}$>%1e' %(0 if max_stelmass == None else max_stelmass, 0 if min_stelmass == None else min_stelmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_gasmass != None or max_gasmass != None:
            legend_labels.append('%1e>M$_{\mathrm{gas}}$>%1e' %(0 if max_gasmass == None else max_gasmass, 0 if min_gasmass == None else min_gasmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')    
        if min_sfmass != None or max_sfmass != None:
            legend_labels.append('%1e>M$_{\mathrm{SF}}$>%1e' %(0 if max_sfmass == None else max_sfmass, 0 if min_sfmass == None else min_sfmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_nsfmass != None or max_nsfmass != None:
            legend_labels.append('%1e>M$_{\mathrm{SF}}$>%1e' %(0 if max_nsfmass == None else max_nsfmass, 0 if min_nsfmass == None else min_nsfmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_sfr != None or max_sfr != None:
            legend_labels.append('%1e>SFR>%1e' %(0 if max_sfr == None else max_sfr, 0 if min_sfr == None else min_sfr))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('teal')
        if min_ssfr != None or max_ssfr != None:
            legend_labels.append('%.1f>sSFR>%.1f' %(0 if max_ssfr == None else max_ssfr, 0 if min_ssfr == None else min_ssfr))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('teal')
        if min_kappa_stars != None or max_kappa_stars != None:
            if min_kappa_stars != None:
                legend_labels.append('$\kappa_{\mathrm{co}}^{\mathrm{*}}>$%s' %(0 if min_kappa_stars == None else min_kappa_stars))
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('cornflowerblue')
            if max_kappa_stars != None:
                legend_labels.append('$\kappa_{\mathrm{co}}^{\mathrm{*}}<$%s' %(0 if max_kappa_stars == None else max_kappa_stars))
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('cornflowerblue')
        if min_kappa_gas != None or max_kappa_gas != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{gas}}>$%.1f' %(0 if max_kappa_gas == None else max_kappa_gas, 0 if min_kappa_gas == None else min_kappa_gas))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_kappa_sf != None or max_kappa_sf != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{SF}}>$%.1f' %(0 if max_kappa_sf == None else max_kappa_sf, 0 if min_kappa_sf == None else min_kappa_sf))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_kappa_nsf != None or max_kappa_nsf != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{NSF}}>$%.1f' %(0 if max_kappa_nsf == None else max_kappa_nsf, 0 if min_kappa_nsf == None else min_kappa_nsf))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_rad != None or max_rad != None:
            legend_labels.append('%.1f>R>%.1f' %(0 if max_rad == None else max_rad, 0 if min_rad == None else min_rad))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('peru')
        if min_inflow != None or max_inflow != None:
            legend_labels.append('%.1f>inflow>%.1f' %(0 if max_inflow == None else max_inflow, 0 if min_inflow == None else min_inflow))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('darkgreen')
        if min_inflow_Z != None or max_inflow_Z != None:
            legend_labels.append('%.1f>inflow Z>%.1f' %(0 if max_inflow_Z == None else max_inflow_Z, 0 if min_inflow_Z == None else min_inflow_Z))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('darkgreen')
        
        # add z
        #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
        #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        #legend_colors.append('k')
                
        if set_plot_relaxation_type:
            if 'co-co' in relaxation_type:
                legend_labels.append('     co → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C0')
            if 'counter-counter' in relaxation_type:
                legend_labels.append('counter → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C1')
            if 'co-counter' in relaxation_type:
                legend_labels.append('     co → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C2')
            if 'counter-co' in relaxation_type:
                legend_labels.append('counter → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C3')
        if add_inset:
            ncol=2
        else:
            ncol=1
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        plt.tight_layout()
        
        
        #-----------
        ### Savefig
        if add_inset:
            savefig_txt_2 = savefig_txt + '_inset'
        else:
            savefig_txt_2 = savefig_txt
            
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f\nmax: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn, max(relaxationtime_plot)),
                         'Producer': str(hist_n)}
                         
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/time_spent_misaligned/%stdyn_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/time_spent_misaligned/%stdyn_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    # ttorque histogram
    if _plot_ttorque_histogram:
        # Gather data
        relaxationtime_plot = []
        co_co_array           = []
        co_counter_array      = []
        counter_co_array      = []
        counter_counter_array = []
        for ID_i in misalignment_tree.keys():
            # append average ttorque over misalignment
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
        print('\nMax tdyn/ttorque:  %.2f' %max(relaxationtime_plot)) 
        
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
                
        if set_bin_limit_ttorque == None:
            set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
        if set_plot_histogram_log:
            set_plot_relaxation_type = False 
        
        #-------------
        ### Plot histogram
        if set_plot_relaxation_type:
            if set_plot_percentage:
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(summary_dict['ID']['co-co']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['co-counter']))/len(relaxationtime_plot), np.ones(len(summary_dict['ID']['counter-co']))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
            
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
            else:
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
                axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
            
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        else:
            if set_plot_percentage:
                axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
                axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque+0.5*set_bin_width_ttorque, set_bin_width_ttorque), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
            else:
                axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
        
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
                axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque+0.5*set_bin_width_ttorque, set_bin_width_ttorque), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        
        #-------------
        ### Inset second axes
        if add_inset:
            axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
            
            if set_plot_percentage:
                axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
            else:
                axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
                bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
            
            print('bin_count of hist:', bin_count)
            
            axins.set_yscale('log')
            axins.set_xlim(0, set_bin_limit_ttorque)
            axins.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
            axins.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$', fontsize=5)
            if set_plot_percentage:
                axins.set_ylim(0.0002, set_thist_ymax_ttorque)
                axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
                axins.set_ylabel('Percentage of misalignments', fontsize=5)
            else:
                axins.set_ylim(0.5, set_thist_ymax_ttorque)
                axins.set_ylabel('Percentage of misalignments', fontsize=5)
            
            #-------------
            # Best-fit data
            mask = np.where(np.array(bin_count) > 0)[0]
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
            print('Best fit line: \tfrac = %.2f x 10^(%.2f t)' %(10**intercept, slope))
            
            if add_inset_bestfit:
                axins.plot([0.1 , 25], [(10**intercept) * (10**(slope*0.1)), (10**intercept) * (10**(slope*25))], lw=0.7, ls='--', alpha=0.8, c='purple', label='best-fit')
                #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
            
            
        #-----------
        ### General formatting
        # Axis labels
        if set_plot_histogram_log:
            axs.set_yscale('log')
        if set_plot_percentage:
            axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
        axs.set_xlim(0, set_bin_limit_ttorque)
        axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=1))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        if set_plot_percentage:
            axs.set_ylabel('Percentage of misalignments')
        else:
            axs.set_ylabel('Number of misalignments')
        axs.set_ylim(0, set_thist_ymax_ttorque)
    
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
    
        # Add any selection criteria if given
        if min_halomass != None or max_halomass != None:
            if min_halomass == 1E14:
                legend_labels.append('Group')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('coral')
            elif max_halomass == 1E14:
                legend_labels.append('Field')  
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('coral')  
        if limit_satellites != None:
            legend_labels.append('%ss' %limit_satellites)
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('coral')
        if min_stelmass != None or max_stelmass != None:
            legend_labels.append('%1e>M$_{*}$>%1e' %(0 if max_stelmass == None else max_stelmass, 0 if min_stelmass == None else min_stelmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_gasmass != None or max_gasmass != None:
            legend_labels.append('%1e>M$_{\mathrm{gas}}$>%1e' %(0 if max_gasmass == None else max_gasmass, 0 if min_gasmass == None else min_gasmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')    
        if min_sfmass != None or max_sfmass != None:
            legend_labels.append('%1e>M$_{\mathrm{SF}}$>%1e' %(0 if max_sfmass == None else max_sfmass, 0 if min_sfmass == None else min_sfmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_nsfmass != None or max_nsfmass != None:
            legend_labels.append('%1e>M$_{\mathrm{SF}}$>%1e' %(0 if max_nsfmass == None else max_nsfmass, 0 if min_nsfmass == None else min_nsfmass))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
        if min_sfr != None or max_sfr != None:
            legend_labels.append('%1e>SFR>%1e' %(0 if max_sfr == None else max_sfr, 0 if min_sfr == None else min_sfr))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('teal')
        if min_ssfr != None or max_ssfr != None:
            legend_labels.append('%.1f>sSFR>%.1f' %(0 if max_ssfr == None else max_ssfr, 0 if min_ssfr == None else min_ssfr))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('teal')
        if min_kappa_stars != None or max_kappa_stars != None:
            if min_kappa_stars != None:
                legend_labels.append('$\kappa_{\mathrm{co}}^{\mathrm{*}}>$%s' %(0 if min_kappa_stars == None else min_kappa_stars))
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('cornflowerblue')
            if max_kappa_stars != None:
                legend_labels.append('$\kappa_{\mathrm{co}}^{\mathrm{*}}<$%s' %(0 if max_kappa_stars == None else max_kappa_stars))
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('cornflowerblue')
        if min_kappa_gas != None or max_kappa_gas != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{gas}}>$%.1f' %(0 if max_kappa_gas == None else max_kappa_gas, 0 if min_kappa_gas == None else min_kappa_gas))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_kappa_sf != None or max_kappa_sf != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{SF}}>$%.1f' %(0 if max_kappa_sf == None else max_kappa_sf, 0 if min_kappa_sf == None else min_kappa_sf))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_kappa_nsf != None or max_kappa_nsf != None:
            legend_labels.append('%.1f$>\kappa_{\mathrm{co}}^{\mathrm{NSF}}>$%.1f' %(0 if max_kappa_nsf == None else max_kappa_nsf, 0 if min_kappa_nsf == None else min_kappa_nsf))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
        if min_rad != None or max_rad != None:
            legend_labels.append('%.1f>R>%.1f' %(0 if max_rad == None else max_rad, 0 if min_rad == None else min_rad))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('peru')
        if min_inflow != None or max_inflow != None:
            legend_labels.append('%.1f>inflow>%.1f' %(0 if max_inflow == None else max_inflow, 0 if min_inflow == None else min_inflow))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('darkgreen')
        if min_inflow_Z != None or max_inflow_Z != None:
            legend_labels.append('%.1f>inflow Z>%.1f' %(0 if max_inflow_Z == None else max_inflow_Z, 0 if min_inflow_Z == None else min_inflow_Z))
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('darkgreen')
        
        # add z
        #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
        #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        #legend_colors.append('k')
                
        if set_plot_relaxation_type:
            if 'co-co' in relaxation_type:
                legend_labels.append('     co → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C0')
            if 'counter-counter' in relaxation_type:
                legend_labels.append('counter → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C1')
            if 'co-counter' in relaxation_type:
                legend_labels.append('     co → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C2')
            if 'counter-co' in relaxation_type:
                legend_labels.append('counter → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C3')
        if add_inset:
            ncol=2
        else:
            ncol=1
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        plt.tight_layout()
        
        
        #-----------
        ### Savefig
        if add_inset:
            savefig_txt_2 = savefig_txt + '_inset'
        else:
            savefig_txt_2 = savefig_txt
            
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f\nmax: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque, max(relaxationtime_plot)),
                         'Producer': str(hist_n)}
                         
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/time_spent_misaligned/%sttorque_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/time_spent_misaligned/%sttorque_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    #-------------------------
    # Plot single stacked of current criteria
    if _plot_stacked_trelax:
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Creating colormaps to mark mergers
        merger_colormap = plt.get_cmap('Blues', 5)
        merger_normalize = colors.Normalize(vmin=0, vmax=1)
        timescale_colormap = plt.get_cmap('inferno')
        timescale_normalize = colors.Normalize(vmin=0, vmax=6)
        scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
        
        #-----------
        ### Loop over all windows and plot them
        ID_plot     = []
        ID_collect  = []        # flexible array used to extract weird relaxations
        time_collect = []
        diff_co_co           = []
        diff_co_counter      = []
        diff_counter_counter = []
        diff_counter_co      = []
        scatter_x = []
        scatter_y = []
        scatter_c = []
        scatter_s = []
        
        stats_lines = {'co-co': {'time': [],
                                 'angles': []},
                       'co-counter': {'time': [],
                                      'angles': []},
                       'counter-counter': {'time': [],
                                           'angles': []},
                       'counter-co': {'time': [],
                                      'angles': []}}
        
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
                continue
                
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            if set_plot_type == 'time':
                timeaxis_plot = -1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']])
            elif set_plot_type == 'raw_time':
                timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - (0)
            elif set_plot_type == 'snap':
                timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - misalignment_tree['%s' %ID_i]['SnapNum'][misalignment_tree['%s' %ID_i]['index_s']]
            elif set_plot_type == 'raw_snap':
                timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - (0)
            
            
            #-------------
            # Plot stacked
            line_color = 'k'
            alpha = 0.2
            if set_plot_relaxation_type_stacked:
                alpha = 0.1
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    line_color='C0'
                    append_angle = 10
                    diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                    
                    # collect IDs of weird relaxations
                    #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                    #    ID_collect.append(ID_i)
                        
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    line_color='C2'
                    append_angle = 170
                    diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    line_color='C3'
                    append_angle = 10
                    diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    line_color='C1'
                    append_angle = 170
                    diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            
            
            #-------------
            # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
            timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
            angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
            # Add missing 120Mya time slots until end reached
            while timeaxis_stats[-1] < set_bin_limit_trelax:
                timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1]+0.120)
                angles_stats   = np.append(angles_stats, append_angle)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
            
            
            #-------------
            # pick out long relaxersß
            if misalignment_tree['%s' %ID_i]['relaxation_time'] > 2:
                ID_collect.append(ID_i)
                            
            
            #-------------
            # format and plot line
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.02)
                c     = lighten_color(line_color, (misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.03)
                c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/2), 0.02)
                c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/3), 0.02)
                c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
                
                
                
            #axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
            axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c='k', alpha=1)
            
            ### Annotate
            if set_add_GalaxyIDs:
                axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
            
            # Plot mergers (some may be missing if they are out of window)
            if set_plot_merger_limit != None:
                for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                    if len(ratio_i) > 0:
                        if max(ratio_i) >= set_plot_merger_limit:
                            scatter_x.append(time_i)
                            scatter_y.append(angle_i)
                            scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                            scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
        
        #print('-------------------------------------------------------------')
        #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
        #print('  List of >2 Gyr trelax ', len(ID_collect))
        #print(ID_collect)
        
        #--------------
        # Find mean/median and 1 sigma behaviour for each relaxation type
        if add_stacked_median:
            if len(stats_lines['co-co']['time']) != 0:
                line_color='C0'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_co))+0.01, 5.09, np.median(diff_co_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-co']['time'])[mask]
                    current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
                
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['co-counter']['time']) != 0:
                line_color='C2'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_counter))+0.01, 5.09, np.median(diff_co_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-counter']['time']) != 0:
                line_color='C1'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 5.09, np.median(diff_counter_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-co']['time']) != 0:
                line_color='C3'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_co))+0.01, 5.09, np.median(diff_counter_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                     
        #-----------
        # Add threshold
        axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
        
        
        #ßfor ID_i in ID_plot:
        #    print(' ')
        #    print(' ID: ', ID_i)
        #    for time_i, angle_i, merger_i in zip(misalignment_tree['%s' %ID_i]['Lookbacktime'], misalignment_tree['%s' %ID_i][use_angle], misalignment_tree['%s' %ID_i]['merger_ratio_stars']):
        #        print('%.2f\t%.1f\t' %(time_i, angle_i), merger_i)
        print(' remaining sample aaaaaa: ', len(ID_plot))
        
        
        
        #-----------
        ### Formatting
        axs.set_ylim(0, 180)
        axs.set_yticks(np.arange(0, 181, 30))
        axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        if set_plot_type == 'time':
            axs.set_xlim(0-3*time_extra, set_bin_limit_trelax)
            axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, 1))
            axs.set_xlabel('Time since misalignment (Gyr)')
        elif set_plot_type == 'raw_time':
            axs.set_xlim(8, 0)
            axs.set_xticks(np.arange(8, -0.1, -1))
            axs.set_xlabel('Lookbacktime (Gyr)')
        elif set_plot_type == 'snap':
            axs.set_xlim(-10, 70)
            axs.set_xticks(np.arange(-10, 71, 10))
            axs.set_xlabel('Snapshots since misalignment')
        elif set_plot_type == 'raw_snap':
            axs.set_xlim(140, 200)
            axs.set_xticks(np.arange(140, 201, 5))
            axs.set_xlabel('Snapshots')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### Annotations
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            axs.axvline(0, ls='-', lw=1, c='grey')
          
        #-----------
        ### Customise legend labels
        if set_plot_merger_limit != None:
            axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
            
            plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
            plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
            plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
            plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
            plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
            plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
            
            legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
            axs.add_artist(legend1)
        if set_plot_relaxation_type_stacked:
            legend_elements = []
            legend_labels = []
            legend_colors = []
            if 'co-co' in set_stacked_relaxation_type:
                legend_labels.append('co → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C0')
            if 'counter-counter' in set_stacked_relaxation_type:
                legend_labels.append('counter → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C1')
            if 'co-counter' in set_stacked_relaxation_type:
                legend_labels.append('     co → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C2')
            if 'counter-co' in set_stacked_relaxation_type:
                legend_labels.append('counter → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C3')
                
            if set_plot_merger_limit != None:
                loc = [0.62, 0.35]
            else:
                loc = 'upper right'
            legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            axs.add_artist(legend2)
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        # Average timescales
        mean_co_co   = np.mean(np.array(summary_dict['trelax']['co-co']))
        median_co_co = np.median(np.array(summary_dict['trelax']['co-co']))
        std_co_co    = np.std(np.array(summary_dict['trelax']['co-co']))
        mean_co_counter   = np.mean(np.array(summary_dict['trelax']['co-counter']))
        median_co_counter = np.median(np.array(summary_dict['trelax']['co-counter']))
        std_co_counter    = np.std(np.array(summary_dict['trelax']['co-counter']))
        mean_counter_counter   = np.mean(np.array(summary_dict['trelax']['counter-counter']))
        median_counter_counter = np.median(np.array(summary_dict['trelax']['counter-counter']))
        std_counter_counter    = np.std(np.array(summary_dict['trelax']['counter-counter']))
        mean_counter_co   = np.mean(np.array(summary_dict['trelax']['counter-co']))
        median_counter_co = np.median(np.array(summary_dict['trelax']['counter-co']))
        std_counter_co    = np.std(np.array(summary_dict['trelax']['counter-co']))
        print('Relaxation timescales:')
        print('   (Gyr)     all   co-co   co-counter   counter-counter   counter-co')
        print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else mean_counter_co)))   
        print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else median_counter_co)))   
        print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else std_counter_co)))
        
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/stacked_misalignments/trelax_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/stacked_misalignments/trelax_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if _plot_stacked_trelax_2x2:
        # Graph initialising and base formatting
        fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Creating colormaps to mark mergers
        merger_colormap = plt.get_cmap('Blues', 5)
        merger_normalize = colors.Normalize(vmin=0, vmax=1)
        timescale_colormap = plt.get_cmap('inferno')
        timescale_normalize = colors.Normalize(vmin=0, vmax=6)
        scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
        
        #-----------
        ### Loop over all windows and plot them
        ID_plot     = []
        ID_collect  = []        # flexible array used to extract weird relaxations
        diff_co_co           = []
        diff_co_counter      = []
        diff_counter_counter = []
        diff_counter_co      = []
        scatter_x = []
        scatter_y = []
        scatter_c = []
        scatter_s = []
        
        stats_lines = {'co-co': {'time': [],
                                 'angles': []},
                       'co-counter': {'time': [],
                                      'angles': []},
                       'counter-counter': {'time': [],
                                           'angles': []},
                       'counter-co': {'time': [],
                                      'angles': []}}
        
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
                continue
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            if set_plot_type == 'time':
                timeaxis_plot = -1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']])
            elif set_plot_type == 'raw_time':
                timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - (0)
            elif set_plot_type == 'snap':
                timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - misalignment_tree['%s' %ID_i]['SnapNum'][misalignment_tree['%s' %ID_i]['index_s']]
            elif set_plot_type == 'raw_snap':
                timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - (0)
            
            
            #-------------
            # Plot stacked
            line_color = 'k'
            alpha = 0.2
            if set_plot_relaxation_type_stacked:
                alpha = 0.1
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    line_color='C0'
                    append_angle = 10
                    ax = ax_co_co
                    diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                    
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    line_color='C2'
                    append_angle = 170
                    ax = ax_co_counter
                    diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    line_color='C3'
                    append_angle = 10
                    ax = ax_counter_co
                    diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    line_color='C1'
                    append_angle = 170
                    ax = ax_counter_counter
                    diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            
            
            #-------------
            # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
            timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
            angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
            # Add missing 120Mya time slots until end reached
            while timeaxis_stats[-1] < set_bin_limit_trelax:
                timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1]+0.120)
                angles_stats   = np.append(angles_stats, append_angle)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
            
            
            #-------------
            # pick out long relaxersß
            if misalignment_tree['%s' %ID_i]['relaxation_time'] > 2:
                ID_collect.append(ID_i)
                            
            
            #-------------
            # format and plot line
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.02)
                c     = lighten_color(line_color, (misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.03)
                c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/2), 0.02)
                c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/3), 0.02)
                c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
                
                
            ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            ### Annotate
            if set_add_GalaxyIDs:
                ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
            
            # Plot mergers (some may be missing if they are out of window)
            if set_plot_merger_limit != None:
                for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                    if len(ratio_i) > 0:
                        if max(ratio_i) >= set_plot_merger_limit:
                            scatter_x.append(time_i)
                            scatter_y.append(angle_i)
                            scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                            scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
        
        print('-------------------------------------------------------------')
        #print('  List of >2 Gyr trelax ', len(ID_collect))
        #print(ID_collect)
        
        #--------------
        # Find mean/median and 1 sigma behaviour for each relaxation type
        if add_stacked_median:
            if len(stats_lines['co-co']['time']) != 0:
                line_color='C0'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_co))+0.01, 5.09, np.median(diff_co_co))
                
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-co']['time'])[mask]
                    current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
                
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['co-counter']['time']) != 0:
                line_color='C2'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_counter))+0.01, 5.09, np.median(diff_co_counter))
                
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-counter']['time']) != 0:
                line_color='C1'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 5.09, np.median(diff_counter_counter))
                
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-co']['time']) != 0:
                line_color='C3'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_co))+0.01, 5.09, np.median(diff_counter_co))
                
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        
                 
        #-----------
        # Add threshold
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
        
        
        #-----------
        ### Formatting
        ax_co_co.set_ylim(0, 180)
        ax_co_co.set_yticks(np.arange(0, 181, 30))
        ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
        if set_plot_type == 'time':
            ax_counter_counter.set_xlim(0-3*time_extra, set_bin_limit_trelax)
            ax_counter_counter.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, 1))
            ax_counter_co.set_xlabel('Time since misalignment (Gyr)')
            ax_counter_counter.set_xlabel('Time since misalignment (Gyr)')
            #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
        elif set_plot_type == 'raw_time':
            ax_counter_counter.set_xlim(8, 0)
            ax_counter_counter.set_xticks(np.arange(8, -0.1, -1))
            ax_counter_counter.set_xlabel('Lookbacktime (Gyr)')
            ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
        elif set_plot_type == 'snap':
            ax_counter_counter.set_xlim(-10, 70)
            ax_counter_counter.set_xticks(np.arange(-10, 71, 10))
            ax_counter_counter.set_xlabel('Snapshots since misalignment')
            ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
        elif set_plot_type == 'raw_snap':
            ax_counter_counter.set_xlim(140, 200)
            ax_counter_counter.set_xticks(np.arange(140, 201, 5))
            ax_counter_counter.set_xlabel('Snapshots')
            ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### Annotations
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            if (set_plot_type == 'time') or (set_plot_type == 'snap'):
                ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
          
        #-----------
        ### Customise legend labels
        if set_plot_merger_limit != None:
            axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
            
            plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
            plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
            plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
            plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
            plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
            plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
            
            legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
            axs.add_artist(legend1)
        if set_plot_relaxation_type_stacked:
            if set_plot_merger_limit != None:
                loc = [0.62, 0.35]
            else:
                loc = 'upper right'
                
            if 'co-co' in set_stacked_relaxation_type:
                legend_labels = ['co → co']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C0']
                legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_co_co.add_artist(legend2)
            if 'counter-counter' in set_stacked_relaxation_type:
                legend_labels = ['counter → counter']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C1']
                legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_counter_counter.add_artist(legend2)
            if 'co-counter' in set_stacked_relaxation_type:
                legend_labels = ['     co → counter']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C2']
                legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_co_counter.add_artist(legend2)
            if 'counter-co' in set_stacked_relaxation_type:
                legend_labels = ['counter → co']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C3']
                legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_counter_co.add_artist(legend2)
                
        #-----------
        ### title
        if plot_annotate:
            ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
            
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        # Average timescales
        mean_co_co   = np.mean(np.array(summary_dict['trelax']['co-co']))
        median_co_co = np.median(np.array(summary_dict['trelax']['co-co']))
        std_co_co    = np.std(np.array(summary_dict['trelax']['co-co']))
        mean_co_counter   = np.mean(np.array(summary_dict['trelax']['co-counter']))
        median_co_counter = np.median(np.array(summary_dict['trelax']['co-counter']))
        std_co_counter    = np.std(np.array(summary_dict['trelax']['co-counter']))
        mean_counter_counter   = np.mean(np.array(summary_dict['trelax']['counter-counter']))
        median_counter_counter = np.median(np.array(summary_dict['trelax']['counter-counter']))
        std_counter_counter    = np.std(np.array(summary_dict['trelax']['counter-counter']))
        mean_counter_co   = np.mean(np.array(summary_dict['trelax']['counter-co']))
        median_counter_co = np.median(np.array(summary_dict['trelax']['counter-co']))
        std_counter_co    = np.std(np.array(summary_dict['trelax']['counter-co']))
        print('Relaxation timescales:')
        print('   (Gyr)     all   co-co   co-counter   counter-counter   counter-co')
        print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else mean_counter_co)))   
        print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else median_counter_co)))   
        print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else std_counter_co)))
        
        
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/stacked_misalignments/trelax_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/stacked_misalignments/trelax_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    # Plot single stacked of current criteria
    if _plot_stacked_tdyn:
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Creating colormaps to mark mergers
        merger_colormap = plt.get_cmap('Blues', 5)
        merger_normalize = colors.Normalize(vmin=0, vmax=1)
        timescale_colormap = plt.get_cmap('inferno')
        timescale_normalize = colors.Normalize(vmin=0, vmax=6)
        scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
        
        #-----------
        ### Loop over all windows and plot them
        ID_plot     = []
        ID_collect  = []        # flexible array used to extract weird relaxations
        time_collect = []
        diff_co_co           = []
        diff_co_counter      = []
        diff_counter_counter = []
        diff_counter_co      = []
        scatter_x = []
        scatter_y = []
        scatter_c = []
        scatter_s = []
        
        stats_lines = {'co-co': {'time': [],
                                 'angles': []},
                       'co-counter': {'time': [],
                                      'angles': []},
                       'counter-counter': {'time': [],
                                           'angles': []},
                       'counter-co': {'time': [],
                                      'angles': []}}
        
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
                continue
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
            
            
            #-------------
            # Plot stacked
            line_color = 'k'
            alpha = 0.2
            if set_plot_relaxation_type_stacked:
                alpha = 0.1
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    line_color='C0'
                    append_angle = 10
                    diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                    
                    # collect IDs of weird relaxations
                    #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                    #    ID_collect.append(ID_i)
                        
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    line_color='C2'
                    append_angle = 170
                    diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    line_color='C3'
                    append_angle = 10
                    diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    line_color='C1'
                    append_angle = 170
                    diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            
            
            #-------------
            # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
            timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
                        
            angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
            # Add missing 120Mya time slots until end reached
            while timeaxis_stats[-1] < set_bin_limit_tdyn:
                timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
                angles_stats   = np.append(angles_stats, append_angle)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
            
            
            #-------------
            # pick out long relaxersß
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 20:
                ID_collect.append(ID_i)
                            
            
            #-------------
            # format and plot line
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.02)
                c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.03)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/2), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/3), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5))
            
                
                
            axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
            
            ### Annotate
            if set_add_GalaxyIDs:
                axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
            
            # Plot mergers (some may be missing if they are out of window)
            if set_plot_merger_limit != None:
                for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                    if len(ratio_i) > 0:
                        if max(ratio_i) >= set_plot_merger_limit:
                            scatter_x.append(time_i)
                            scatter_y.append(angle_i)
                            scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                            scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
        
        print('-------------------------------------------------------------')
        #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
        print('  List of >20 trelax/tdyn ', len(ID_collect))
        print(ID_collect)
        print('asugaigfaiuf')
        print('median: ', np.median(time_collect))
        
        
        #--------------
        # Find mean/median and 1 sigma behaviour for each relaxation type
        if add_stacked_median:
            if len(stats_lines['co-co']['time']) != 0:
                line_color='C0'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_co))+0.01, 32.09, np.median(diff_co_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                            
                
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-co']['time'])[mask]
                    current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
                
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['co-counter']['time']) != 0:
                line_color='C2'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_counter))+0.01, 32.09, np.median(diff_co_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-counter']['time']) != 0:
                line_color='C1'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 32.09, np.median(diff_counter_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-co']['time']) != 0:
                line_color='C3'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_co))+0.01, 32.09, np.median(diff_counter_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                     
        #-----------
        # Add threshold
        axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
        
        
        #-----------
        ### Formatting
        axs.set_ylim(0, 180)
        axs.set_yticks(np.arange(0, 181, 30))
        axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        axs.set_xlim(-1, set_bin_limit_tdyn)
        axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, 2))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        
        #-----------
        ### Annotations
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            axs.axvline(0, ls='-', lw=1, c='k')
          
        #-----------
        ### Customise legend labels
        if set_plot_merger_limit != None:
            axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
            
            plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
            plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
            plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
            plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
            plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
            plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
            
            legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
            axs.add_artist(legend1)
        if set_plot_relaxation_type_stacked:
            legend_elements = []
            legend_labels = []
            legend_colors = []
            if 'co-co' in set_stacked_relaxation_type:
                legend_labels.append('co → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C0')
            if 'counter-counter' in set_stacked_relaxation_type:
                legend_labels.append('counter → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C1')
            if 'co-counter' in set_stacked_relaxation_type:
                legend_labels.append('     co → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C2')
            if 'counter-co' in set_stacked_relaxation_type:
                legend_labels.append('counter → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C3')
                
            if set_plot_merger_limit != None:
                loc = [0.62, 0.35]
            else:
                loc = 'upper right'
            legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            axs.add_artist(legend2)
        
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        # Average timescales
        mean_co_co   = np.mean(np.array(summary_dict['tdyn']['co-co']))
        median_co_co = np.median(np.array(summary_dict['tdyn']['co-co']))
        std_co_co    = np.std(np.array(summary_dict['tdyn']['co-co']))
        mean_co_counter   = np.mean(np.array(summary_dict['tdyn']['co-counter']))
        median_co_counter = np.median(np.array(summary_dict['tdyn']['co-counter']))
        std_co_counter    = np.std(np.array(summary_dict['tdyn']['co-counter']))
        mean_counter_counter   = np.mean(np.array(summary_dict['tdyn']['counter-counter']))
        median_counter_counter = np.median(np.array(summary_dict['tdyn']['counter-counter']))
        std_counter_counter    = np.std(np.array(summary_dict['tdyn']['counter-counter']))
        mean_counter_co   = np.mean(np.array(summary_dict['tdyn']['counter-co']))
        median_counter_co = np.median(np.array(summary_dict['tdyn']['counter-co']))
        std_counter_co    = np.std(np.array(summary_dict['tdyn']['counter-co']))
        print('trelax/tdyn multiples:')
        print('             all   co-co   co-counter   counter-counter   counter-co')
        print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else mean_counter_co)))   
        print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else median_counter_co)))   
        print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else std_counter_co)))
        
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/stacked_misalignments/tdyn_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/stacked_misalignments/tdyn_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if _plot_stacked_tdyn_2x2:
        # Graph initialising and base formatting
        fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        
        # Creating colormaps to mark mergers
        merger_colormap = plt.get_cmap('Blues', 5)
        merger_normalize = colors.Normalize(vmin=0, vmax=1)
        timescale_colormap = plt.get_cmap('inferno')
        timescale_normalize = colors.Normalize(vmin=0, vmax=6)
        scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
        
        #-----------
        ### Loop over all windows and plot them
        ID_plot     = []
        ID_collect  = []        # flexible array used to extract weird relaxations
        diff_co_co           = []
        diff_co_counter      = []
        diff_counter_counter = []
        diff_counter_co      = []
        scatter_x = []
        scatter_y = []
        scatter_c = []
        scatter_s = []
        
        stats_lines = {'co-co': {'time': [],
                                 'angles': []},
                       'co-counter': {'time': [],
                                      'angles': []},
                       'counter-counter': {'time': [],
                                           'angles': []},
                       'counter-co': {'time': [],
                                      'angles': []}}
        
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
                continue
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
            
            #-------------
            # Plot stacked
            line_color = 'k'
            alpha = 0.2
            if set_plot_relaxation_type_stacked:
                alpha = 0.1
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    line_color='C0'
                    append_angle = 10
                    ax = ax_co_co
                    diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                    
                    # collect IDs of weird relaxations
                    #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                    #    ID_collect.append(ID_i)     
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    line_color='C2'
                    append_angle = 170
                    ax = ax_co_counter
                    diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    line_color='C3'
                    append_angle = 10
                    ax = ax_counter_co
                    diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    line_color='C1'
                    append_angle = 170
                    ax = ax_counter_counter
                    diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            
            
            #-------------
            # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
            timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
            angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
            # Add missing 120Mya time slots until end reached
            while timeaxis_stats[-1] < set_bin_limit_tdyn:
                timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
                angles_stats   = np.append(angles_stats, append_angle)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
            
            
            #-------------
            # pick out long relaxersß
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 10:
                ID_collect.append(ID_i)
                            
            
            #-------------
            # format and plot line
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.02)
                c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.03)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/2), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/3), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5))
                
                
                
                
            ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            ### Annotate
            if set_add_GalaxyIDs:
                ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
            
            # Plot mergers (some may be missing if they are out of window)
            if set_plot_merger_limit != None:
                for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                    if len(ratio_i) > 0:
                        if max(ratio_i) >= set_plot_merger_limit:
                            scatter_x.append(time_i)
                            scatter_y.append(angle_i)
                            scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                            scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
        
        print('-------------------------------------------------------------')
        #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
        #print('  List of >10 trelax/tdyn ', len(ID_collect))
        #print(ID_collect)
        
        #--------------
        # Find mean/median and 1 sigma behaviour for each relaxation type
        if add_stacked_median:
            if len(stats_lines['co-co']['time']) != 0:
                line_color='C0'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_co))+0.01, 32.09, np.median(diff_co_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-co']['time'])[mask]
                    current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
                
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['co-counter']['time']) != 0:
                line_color='C2'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_counter))+0.01, 32.09, np.median(diff_co_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
            
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-counter']['time']) != 0:
                line_color='C1'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 32.09, np.median(diff_counter_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
            
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-co']['time']) != 0:
                line_color='C3'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_co))+0.01, 32.09, np.median(diff_counter_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                
        #-----------
        # Add threshold
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
        
        
        #-----------
        ### Formatting
        ax_co_co.set_ylim(0, 180)
        ax_co_co.set_yticks(np.arange(0, 181, 30))
        ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
        ax_counter_counter.set_xlim(-1, set_bin_limit_tdyn)
        ax_counter_counter.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, 2))
        ax_counter_counter.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        ax_counter_co.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### Annotations
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            if (set_plot_type == 'time') or (set_plot_type == 'snap'):
                ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
        
        #-----------
        ### title
        if plot_annotate:
            ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
            
        #-----------
        ### Customise legend labels
        if set_plot_merger_limit != None:
            axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
            
            plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
            plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
            plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
            plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
            plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
            plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
            
            legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
            axs.add_artist(legend1)
        if set_plot_relaxation_type_stacked:
            if set_plot_merger_limit != None:
                loc = [0.62, 0.35]
            else:
                loc = 'upper right'
                
            if 'co-co' in set_stacked_relaxation_type:
                legend_labels = ['co → co']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C0']
                legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_co_co.add_artist(legend2)
            if 'counter-counter' in set_stacked_relaxation_type:
                legend_labels = ['counter → counter']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C1']
                legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_counter_counter.add_artist(legend2)
            if 'co-counter' in set_stacked_relaxation_type:
                legend_labels = ['     co → counter']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C2']
                legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_co_counter.add_artist(legend2)
            if 'counter-co' in set_stacked_relaxation_type:
                legend_labels = ['counter → co']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C3']
                legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_counter_co.add_artist(legend2)
                
            
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        # Average timescales
        mean_co_co   = np.mean(np.array(summary_dict['tdyn']['co-co']))
        median_co_co = np.median(np.array(summary_dict['tdyn']['co-co']))
        std_co_co    = np.std(np.array(summary_dict['tdyn']['co-co']))
        mean_co_counter   = np.mean(np.array(summary_dict['tdyn']['co-counter']))
        median_co_counter = np.median(np.array(summary_dict['tdyn']['co-counter']))
        std_co_counter    = np.std(np.array(summary_dict['tdyn']['co-counter']))
        mean_counter_counter   = np.mean(np.array(summary_dict['tdyn']['counter-counter']))
        median_counter_counter = np.median(np.array(summary_dict['tdyn']['counter-counter']))
        std_counter_counter    = np.std(np.array(summary_dict['tdyn']['counter-counter']))
        mean_counter_co   = np.mean(np.array(summary_dict['tdyn']['counter-co']))
        median_counter_co = np.median(np.array(summary_dict['tdyn']['counter-co']))
        std_counter_co    = np.std(np.array(summary_dict['tdyn']['counter-co']))
        print('trelax/tdyn multiples:')
        print('             all   co-co   co-counter   counter-counter   counter-co')
        print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else mean_counter_co)))   
        print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else median_counter_co)))   
        print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else std_counter_co)))
        
        
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/stacked_misalignments/tdyn_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/stacked_misalignments/tdyn_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    # Plot single stacked of current criteria
    if _plot_stacked_ttorque:
        # Graph initialising and base formatting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Creating colormaps to mark mergers
        merger_colormap = plt.get_cmap('Blues', 5)
        merger_normalize = colors.Normalize(vmin=0, vmax=1)
        timescale_colormap = plt.get_cmap('inferno')
        timescale_normalize = colors.Normalize(vmin=0, vmax=6)
        scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
        
        #-----------
        ### Loop over all windows and plot them
        ID_plot     = []
        ID_collect  = []        # flexible array used to extract weird relaxations
        time_collect = []
        diff_co_co           = []
        diff_co_counter      = []
        diff_counter_counter = []
        diff_counter_co      = []
        scatter_x = []
        scatter_y = []
        scatter_c = []
        scatter_s = []
        
        stats_lines = {'co-co': {'time': [],
                                 'angles': []},
                       'co-counter': {'time': [],
                                      'angles': []},
                       'counter-counter': {'time': [],
                                           'angles': []},
                       'counter-co': {'time': [],
                                      'angles': []}}
        
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
                continue
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
            
            
            #-------------
            # Plot stacked
            line_color = 'k'
            alpha = 0.2
            if set_plot_relaxation_type_stacked:
                alpha = 0.1
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    line_color='C0'
                    append_angle = 10
                    diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                    
                    # collect IDs of weird relaxations
                    #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                    #    ID_collect.append(ID_i)
                        
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    line_color='C2'
                    append_angle = 170
                    diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    line_color='C3'
                    append_angle = 10
                    diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    line_color='C1'
                    append_angle = 170
                    diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            
            
            #-------------
            # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
            timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
            angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
            # Add missing 120Mya time slots until end reached
            while timeaxis_stats[-1] < set_bin_limit_ttorque:
                timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
                angles_stats   = np.append(angles_stats, append_angle)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
            
            
            #-------------
            # pick out long relaxersß
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 10:
                ID_collect.append(ID_i)
                            
            
            #-------------
            # format and plot line
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.02)
                c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.03)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/2), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/3), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5))
            
                
                
            axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
            
            ### Annotate
            if set_add_GalaxyIDs:
                axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
            
            # Plot mergers (some may be missing if they are out of window)
            if set_plot_merger_limit != None:
                for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                    if len(ratio_i) > 0:
                        if max(ratio_i) >= set_plot_merger_limit:
                            scatter_x.append(time_i)
                            scatter_y.append(angle_i)
                            scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                            scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
        
        #print('-------------------------------------------------------------')
        #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
        print('  List of >10 trelax/ttorque ', len(ID_collect))
        print(ID_collect)
        print('asugaigfaiuf')
        print('median: ', np.median(time_collect))
        
        
        #--------------
        # Find mean/median and 1 sigma behaviour for each relaxation type
        if add_stacked_median:
            if len(stats_lines['co-co']['time']) != 0:
                line_color='C0'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_co))+0.01, 12.09, np.median(diff_co_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                            
                
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-co']['time'])[mask]
                    current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
                
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['co-counter']['time']) != 0:
                line_color='C2'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_counter))+0.01, 12.09, np.median(diff_co_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-counter']['time']) != 0:
                line_color='C1'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 12.09, np.median(diff_counter_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-co']['time']) != 0:
                line_color='C3'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_co))+0.01, 12.09, np.median(diff_counter_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                     
        #-----------
        # Add threshold
        axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
        
        
        #-----------
        ### Formatting
        axs.set_ylim(0, 180)
        axs.set_yticks(np.arange(0, 181, 30))
        axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        axs.set_xlim(-0.5, set_bin_limit_ttorque)
        axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, 1))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
            
        #-----------
        ### Annotations
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            axs.axvline(0, ls='-', lw=1, c='k')
          
        #-----------
        ### Customise legend labels
        if set_plot_merger_limit != None:
            axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
            
            plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
            plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
            plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
            plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
            plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
            plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
            
            legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
            axs.add_artist(legend1)
        if set_plot_relaxation_type_stacked:
            legend_elements = []
            legend_labels = []
            legend_colors = []
            if 'co-co' in set_stacked_relaxation_type:
                legend_labels.append('co → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C0')
            if 'counter-counter' in set_stacked_relaxation_type:
                legend_labels.append('counter → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C1')
            if 'co-counter' in set_stacked_relaxation_type:
                legend_labels.append('     co → counter')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C2')
            if 'counter-co' in set_stacked_relaxation_type:
                legend_labels.append('counter → co')
                legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
                legend_colors.append('C3')
                
            if set_plot_merger_limit != None:
                loc = [0.62, 0.35]
            else:
                loc = 'upper right'
            legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            axs.add_artist(legend2)
        
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        # Average timescales
        mean_co_co   = np.mean(np.array(summary_dict['ttorque']['co-co']))
        median_co_co = np.median(np.array(summary_dict['ttorque']['co-co']))
        std_co_co    = np.std(np.array(summary_dict['ttorque']['co-co']))
        mean_co_counter   = np.mean(np.array(summary_dict['ttorque']['co-counter']))
        median_co_counter = np.median(np.array(summary_dict['ttorque']['co-counter']))
        std_co_counter    = np.std(np.array(summary_dict['ttorque']['co-counter']))
        mean_counter_counter   = np.mean(np.array(summary_dict['ttorque']['counter-counter']))
        median_counter_counter = np.median(np.array(summary_dict['ttorque']['counter-counter']))
        std_counter_counter    = np.std(np.array(summary_dict['ttorque']['counter-counter']))
        mean_counter_co   = np.mean(np.array(summary_dict['ttorque']['counter-co']))
        median_counter_co = np.median(np.array(summary_dict['ttorque']['counter-co']))
        std_counter_co    = np.std(np.array(summary_dict['ttorque']['counter-co']))
        print('trelax/ttorque multiples:')
        print('             all   co-co   co-counter   counter-counter   counter-co')
        print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else mean_counter_co)))   
        print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else median_counter_co)))   
        print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else std_counter_co)))
        
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/stacked_misalignments/ttorque_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/stacked_misalignments/ttorque_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if _plot_stacked_ttorque_2x2:
        # Graph initialising and base formatting
        fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        
        # Creating colormaps to mark mergers
        merger_colormap = plt.get_cmap('Blues', 5)
        merger_normalize = colors.Normalize(vmin=0, vmax=1)
        timescale_colormap = plt.get_cmap('inferno')
        timescale_normalize = colors.Normalize(vmin=0, vmax=6)
        scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
        
        #-----------
        ### Loop over all windows and plot them
        ID_plot     = []
        ID_collect  = []        # flexible array used to extract weird relaxations
        diff_co_co           = []
        diff_co_counter      = []
        diff_counter_counter = []
        diff_counter_co      = []
        scatter_x = []
        scatter_y = []
        scatter_c = []
        scatter_s = []
        
        stats_lines = {'co-co': {'time': [],
                                 'angles': []},
                       'co-counter': {'time': [],
                                      'angles': []},
                       'counter-counter': {'time': [],
                                           'angles': []},
                       'counter-co': {'time': [],
                                      'angles': []}}
        
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
                continue
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
            
            #-------------
            # Plot stacked
            line_color = 'k'
            alpha = 0.2
            if set_plot_relaxation_type_stacked:
                alpha = 0.1
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    line_color='C0'
                    append_angle = 10
                    ax = ax_co_co
                    diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                    
                    # collect IDs of weird relaxations
                    #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                    #    ID_collect.append(ID_i)
                        
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    line_color='C2'
                    append_angle = 170
                    ax = ax_co_counter
                    diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    line_color='C3'
                    append_angle = 10
                    ax = ax_counter_co
                    diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    line_color='C1'
                    append_angle = 170
                    ax = ax_counter_counter
                    diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            
            
            #-------------
            # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
            timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
            angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
            # Add missing 120Mya time slots until end reached
            while timeaxis_stats[-1] < set_bin_limit_tdyn:
                timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
                angles_stats   = np.append(angles_stats, append_angle)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
            stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
            
            
            #-------------
            # pick out long relaxersß
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 5:
                ID_collect.append(ID_i)
                            
            
            #-------------
            # format and plot line
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.02)
                c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.03)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/2), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/3), 0.02)
                c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5))
                
                
                
                
            ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            ### Annotate
            if set_add_GalaxyIDs:
                ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
            
            # Plot mergers (some may be missing if they are out of window)
            if set_plot_merger_limit != None:
                for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                    if len(ratio_i) > 0:
                        if max(ratio_i) >= set_plot_merger_limit:
                            scatter_x.append(time_i)
                            scatter_y.append(angle_i)
                            scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                            scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
        
        print('-------------------------------------------------------------')
        #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
        #print('  List of >5 trelax/ttorque ', len(ID_collect))
        #print(ID_collect)
        
        #--------------
        # Find mean/median and 1 sigma behaviour for each relaxation type
        if add_stacked_median:
            if len(stats_lines['co-co']['time']) != 0:
                line_color='C0'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_co))+0.01, 12.09, np.median(diff_co_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                
                
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-co']['time'])[mask]
                    current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
                
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['co-counter']['time']) != 0:
                line_color='C2'
                line_color='k'
                bins = np.arange((-1*np.median(diff_co_counter))+0.01, 12.09, np.median(diff_co_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-counter']['time']) != 0:
                line_color='C1'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 12.09, np.median(diff_counter_counter))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
                
                #----------
                # plot
                ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            if len(stats_lines['counter-co']['time']) != 0:
                line_color='C3'
                line_color='k'
                bins = np.arange((-1*np.median(diff_counter_co))+0.01, 12.09, np.median(diff_counter_co))
            
                # Create mask that is essentially the bins
                digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
                bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
            
                # Collect median (50%), and 1sigma around that (16% and 84%)
                median_array = []
                median_upper = []
                median_lower = []
                use_percentiles = 16        # 1 sigma
                for bin_i in np.arange(1, len(bins_count)+1, 1):
                    # extract values in current bin
                    mask = digitized == bin_i
                    current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                    current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                                
                    median_array.append(np.percentile(current_angles, 50))
                    median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                    median_lower.append(np.percentile(current_angles, use_percentiles))
            
                #----------
                # remove 'excess'
                mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
                mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
                mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
                
                #----------
                # plot
                ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
                ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
                ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
                #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                
        #-----------
        # Add threshold
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
            ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
        
        
        #-----------
        ### Formatting
        ax_co_co.set_ylim(0, 180)
        ax_co_co.set_yticks(np.arange(0, 181, 30))
        ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
        #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
        ax_counter_counter.set_xlim(-0.5, set_bin_limit_ttorque)
        ax_counter_counter.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, 2))
        ax_counter_counter.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        ax_counter_co.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### Annotations
        for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
            if (set_plot_type == 'time') or (set_plot_type == 'snap'):
                ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
        
        #-----------
        ### title
        if plot_annotate:
            ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
          
        #-----------
        ### Customise legend labels
        if set_plot_merger_limit != None:
            axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
            
            plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
            plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
            plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
            plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
            plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
            plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
            
            legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
            axs.add_artist(legend1)
        if set_plot_relaxation_type_stacked:
            if set_plot_merger_limit != None:
                loc = [0.62, 0.35]
            else:
                loc = 'upper right'
                
            if 'co-co' in set_stacked_relaxation_type:
                legend_labels = ['co → co']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C0']
                legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_co_co.add_artist(legend2)
            if 'counter-counter' in set_stacked_relaxation_type:
                legend_labels = ['counter → counter']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C1']
                legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_counter_counter.add_artist(legend2)
            if 'co-counter' in set_stacked_relaxation_type:
                legend_labels = ['     co → counter']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C2']
                legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_co_counter.add_artist(legend2)
            if 'counter-co' in set_stacked_relaxation_type:
                legend_labels = ['counter → co']
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                legend_colors = ['C3']
                legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
                ax_counter_co.add_artist(legend2)
                
            
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        # Average timescales
        mean_co_co   = np.mean(np.array(summary_dict['ttorque']['co-co']))
        median_co_co = np.median(np.array(summary_dict['ttorque']['co-co']))
        std_co_co    = np.std(np.array(summary_dict['ttorque']['co-co']))
        mean_co_counter   = np.mean(np.array(summary_dict['ttorque']['co-counter']))
        median_co_counter = np.median(np.array(summary_dict['ttorque']['co-counter']))
        std_co_counter    = np.std(np.array(summary_dict['ttorque']['co-counter']))
        mean_counter_counter   = np.mean(np.array(summary_dict['ttorque']['counter-counter']))
        median_counter_counter = np.median(np.array(summary_dict['ttorque']['counter-counter']))
        std_counter_counter    = np.std(np.array(summary_dict['ttorque']['counter-counter']))
        mean_counter_co   = np.mean(np.array(summary_dict['ttorque']['counter-co']))
        median_counter_co = np.median(np.array(summary_dict['ttorque']['counter-co']))
        std_counter_co    = np.std(np.array(summary_dict['ttorque']['counter-co']))
        print('trelax/ttorque multiples:')
        print('             all   co-co   co-counter   counter-counter   counter-co')
        print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else mean_counter_co)))   
        print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else median_counter_co)))   
        print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else std_counter_co)))
        
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/stacked_misalignments/ttorque_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/stacked_misalignments/ttorque_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    #-------------------------
    # Plot box and whisker of relaxation distributions
    if _plot_box_and_whisker_trelax:
        # Gather data
        relaxationtime_plot  = []
        relaxationtype_plot  = []
        relaxationmorph_plot = []
        for ID_i in misalignment_tree.keys():
            
            # only plot morphs we care about (default ETG-ETG, LTG-LTG)
            if 'ETG' in set_whisker_morphs:
                if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                    continue
            elif 'ETG-ETG' in set_whisker_morphs: 
                if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                    continue
            
            # Gather data
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('co\n ↓ \nco')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('co\n ↓ \ncounter')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('counter\n ↓ \nco')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('counter\n ↓ \ncounter')
            
            if 'ETG' in set_whisker_morphs:
                relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
            elif 'ETG-ETG' in set_whisker_morphs: 
                #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                    relaxationmorph_plot.append('ETG → ETG')
                if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                    relaxationmorph_plot.append('LTG → LTG')
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
        
        #----------------------
        # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
        print('\n--------------------------------------')
        print('trelax')
        for relaxation_type_i in relaxation_type:
            
            if relaxation_type_i == 'co-co':
                relaxation_type_i = 'co\n ↓ \nco'
            if relaxation_type_i == 'co-counter':
                relaxation_type_i = 'co\n ↓ \ncounter'
            if relaxation_type_i == 'counter-co':
                relaxation_type_i = 'counter\n ↓ \nco'
            if relaxation_type_i == 'counter-counter':
                relaxation_type_i = 'counter\n ↓ \ncounter'
            
            # Select only relaxation morphs
            if 'ETG' in set_whisker_morphs:
                df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
                df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
            elif 'ETG-ETG' in set_whisker_morphs:    
                df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
                df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
            
            # KS test relaxation types between ETG and LTG
            if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

                res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
                
                if relaxation_type_i == 'co\n ↓ \nco':
                    relaxation_type_i = 'co-co'
                if relaxation_type_i == 'co\n ↓ \ncounter':
                    relaxation_type_i = 'co-counter'
                if relaxation_type_i == 'counter\n ↓ \nco':
                    relaxation_type_i = 'counter-co'
                if relaxation_type_i == 'counter\n ↓ \ncounter':
                    relaxation_type_i = 'counter-counter'
            
                if 'ETG' in set_whisker_morphs:
                    print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
                elif 'ETG-ETG' in set_whisker_morphs:  
                    print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
                print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
                print('   p-value: %s' %res.pvalue)
            else:
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
        print('--------------------------------------')
        
        
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        

        if 'ETG' in set_whisker_morphs:
            order = ['ETG', 'LTG']
        elif 'ETG-ETG' in set_whisker_morphs:   
            order = ['ETG → ETG', 'LTG → LTG']
        #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
        #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
        sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-------------
        ### Formatting
        axs.set_xlim(left=0)
        #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
        
        #print(max(relaxationtime_plot))
        
        
        #------------
        # Legend
        axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #-----------
        # savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
        
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/violinplot_relaxation_morph/trelax_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/violinplot_relaxation_morph/trelax_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
        
        df = 0
    # tdyn
    if _plot_box_and_whisker_tdyn:
        # Gather data
        relaxationtime_plot  = []
        relaxationtype_plot  = []
        relaxationmorph_plot = []
        for ID_i in misalignment_tree.keys():
            
            # only plot morphs we care about (default ETG-ETG, LTG-LTG)
            if 'ETG' in set_whisker_morphs:
                if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                    continue
            elif 'ETG-ETG' in set_whisker_morphs: 
                if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                    continue
            
            # Gather data
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('co\n ↓ \nco')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('co\n ↓ \ncounter')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('counter\n ↓ \nco')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('counter\n ↓ \ncounter')
            
            if 'ETG' in set_whisker_morphs:
                relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
            elif 'ETG-ETG' in set_whisker_morphs: 
                #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                    relaxationmorph_plot.append('ETG → ETG')
                if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                    relaxationmorph_plot.append('LTG → LTG')
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
        
        #----------------------
        # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
        print('\n--------------------------------------')
        print('tdyn')
        for relaxation_type_i in relaxation_type:
            
            if relaxation_type_i == 'co-co':
                relaxation_type_i = 'co\n ↓ \nco'
            if relaxation_type_i == 'co-counter':
                relaxation_type_i = 'co\n ↓ \ncounter'
            if relaxation_type_i == 'counter-co':
                relaxation_type_i = 'counter\n ↓ \nco'
            if relaxation_type_i == 'counter-counter':
                relaxation_type_i = 'counter\n ↓ \ncounter'
            
            # Select only relaxation morphs
            if 'ETG' in set_whisker_morphs:
                df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
                df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
            elif 'ETG-ETG' in set_whisker_morphs:    
                df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
                df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
            
            # KS test relaxation types between ETG and LTG
            if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

                res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
                
                if relaxation_type_i == 'co\n ↓ \nco':
                    relaxation_type_i = 'co-co'
                if relaxation_type_i == 'co\n ↓ \ncounter':
                    relaxation_type_i = 'co-counter'
                if relaxation_type_i == 'counter\n ↓ \nco':
                    relaxation_type_i = 'counter-co'
                if relaxation_type_i == 'counter\n ↓ \ncounter':
                    relaxation_type_i = 'counter-counter'
            
                if 'ETG' in set_whisker_morphs:
                    print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
                elif 'ETG-ETG' in set_whisker_morphs:  
                    print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
                print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
                print('   p-value: %s' %res.pvalue)
            else:
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
        print('--------------------------------------')
        
        
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        

        if 'ETG' in set_whisker_morphs:
            order = ['ETG', 'LTG']
        elif 'ETG-ETG' in set_whisker_morphs:   
            order = ['ETG → ETG', 'LTG → LTG']
        #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
        #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
        sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-------------
        ### Formatting
        axs.set_xlim(left=0)
        #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        
        #------------
        # Legend
        axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #-----------
        # savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
        
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/violinplot_relaxation_morph/tdyn_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/violinplot_relaxation_morph/tdyn_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
        
        df = 0
    # ttorque 
    if _plot_box_and_whisker_ttorque:
        # Gather data
        relaxationtime_plot  = []
        relaxationtype_plot  = []
        relaxationmorph_plot = []
        for ID_i in misalignment_tree.keys():
            
            # only plot morphs we care about (default ETG-ETG, LTG-LTG)
            if 'ETG' in set_whisker_morphs:
                if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                    continue
            elif 'ETG-ETG' in set_whisker_morphs: 
                if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                    continue
            
            # Gather data
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('co\n ↓ \nco')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('co\n ↓ \ncounter')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('counter\n ↓ \nco')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('counter\n ↓ \ncounter')
            
            if 'ETG' in set_whisker_morphs:
                relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
            elif 'ETG-ETG' in set_whisker_morphs: 
                #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                    relaxationmorph_plot.append('ETG → ETG')
                if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                    relaxationmorph_plot.append('LTG → LTG')
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
        
        #----------------------
        # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
        print('\n--------------------------------------')
        print('ttorque')
        for relaxation_type_i in relaxation_type:
            
            if relaxation_type_i == 'co-co':
                relaxation_type_i = 'co\n ↓ \nco'
            if relaxation_type_i == 'co-counter':
                relaxation_type_i = 'co\n ↓ \ncounter'
            if relaxation_type_i == 'counter-co':
                relaxation_type_i = 'counter\n ↓ \nco'
            if relaxation_type_i == 'counter-counter':
                relaxation_type_i = 'counter\n ↓ \ncounter'
            
            # Select only relaxation morphs
            if 'ETG' in set_whisker_morphs:
                df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
                df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
            elif 'ETG-ETG' in set_whisker_morphs:    
                df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
                df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
            
            # KS test relaxation types between ETG and LTG
            if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

                res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
                
                if relaxation_type_i == 'co\n ↓ \nco':
                    relaxation_type_i = 'co-co'
                if relaxation_type_i == 'co\n ↓ \ncounter':
                    relaxation_type_i = 'co-counter'
                if relaxation_type_i == 'counter\n ↓ \nco':
                    relaxation_type_i = 'counter-co'
                if relaxation_type_i == 'counter\n ↓ \ncounter':
                    relaxation_type_i = 'counter-counter'
            
                if 'ETG' in set_whisker_morphs:
                    print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
                elif 'ETG-ETG' in set_whisker_morphs:  
                    print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
                print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
                print('   p-value: %s' %res.pvalue)
            else:
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
        print('--------------------------------------')
        
        
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        

        if 'ETG' in set_whisker_morphs:
            order = ['ETG', 'LTG']
        elif 'ETG-ETG' in set_whisker_morphs:   
            order = ['ETG → ETG', 'LTG → LTG']
        #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
        #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
        sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-------------
        ### Formatting
        axs.set_xlim(left=0)
        #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        
        #------------
        # Legend
        axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #-----------
        # savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(min_particles, max_com, min_inclination, set_plot_merger_limit, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
        
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/violinplot_relaxation_morph/ttorque_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/violinplot_relaxation_morph/ttorque_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
        
        df = 0
    
    
    #-------------------------
    # Plot delta angle, trelax. Looks at peak angle from 180
    if _plot_offset_trelax:
        relaxationtime_plot  = []
        angles_plot = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # check offset angle within range            
            if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
                continue
            if set_plot_offset_type:
                if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                    continue
                                
            # Add angles
            angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            ID_plot.append(ID_i)
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
                
        #-------------
        # Plotting
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        bin_width = 15
        bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
        c = 'k'
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.05, c='dimgrey', marker='.', alpha=1, zorder=1)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c=c, ls='-', zorder=7)
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
        ### Plot histograms
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        
        #-----------
        ### General formatting
        # Axis labels
        ax.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
        ax.set_xlabel('Peak misalignment angle')
        ax_histy.set_xlabel('Count')
        ax_histx.set_ylabel('Count')
        if set_plot_offset_log:
            ax.set_yscale('log')
            ax.set_yticks([0.1, 1, 10])
            ax.set_yticklabels(['0.1', '1', '10'])
        else:
            ax.set_ylim(0, 3)
        
        ax.set_xticks(bins)
        ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.set_xticks([1, 10, 100, 1000])
        ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
        
        #------------
        ### Add title
        if set_plot_offset_type:
            if 'co-co' in set_plot_offset_type:
                ax_histx.set_title(r'co → co', size=7, loc='left', pad=3)
            if 'counter-counter' in set_plot_offset_type:
                ax_histx.set_title(r'counter → counter', size=7, loc='left', pad=3)
            if 'co-counter' in set_plot_offset_type:
                ax_histx.set_title(r'co → counter', size=7, loc='left', pad=3)
            if 'counter-co' in set_plot_offset_type:
                ax_histx.set_title(r'counter → co', size=7, loc='left', pad=3)
                
        
        #------------
        # Legend
        #ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #------------
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/offset_angle/trelax_offset_angle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/offset_angle/trelax_offset_angle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if _plot_offset_morph_trelax:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        angles_plot = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # check offset angle within range            
            if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
                continue
            # check morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_offset_morphs:
                continue
            if set_plot_offset_type:
                if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                    continue
                                
            # Add angles
            angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            ID_plot.append(ID_i)
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
                
        #-------------
        # Plotting
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        bin_width = 15
        bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
        c = 'k'
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.05, c='darkgrey', edgecolor='k', marker='.', alpha=1, zorder=20)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=99, label='sample')
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
        ### Plot histograms
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        #-------------
        ### Plot by morphology
        for offset_morph_i in set_offset_morphs:
            
            if offset_morph_i == 'ETG-ETG':
                offset_morph_i = 'ETG → ETG'
                c = 'r'
                ls = '--'
            elif offset_morph_i == 'LTG-LTG':
                offset_morph_i = 'LTG → LTG'
                c = 'b'
                ls = 'dashdot'
            
            # Dataframe of morphs matching this
            df_morph = df.loc[df['Morphology'] == offset_morph_i]
            
            # Bin hist data, find sigma percentiles
            binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
            bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
            
            #-------------------
            ### Plot scatter
            #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
            
            ### Plot upper, median, and lower sigma
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
            ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.9)
            #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
            
            
            ### Plot histograms
            ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
            ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
        
        #-----------
        ### General formatting
        # Axis labels
        ax.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
        ax.set_xlabel('Peak misalignment angle')
        ax_histy.set_xlabel('Count')
        ax_histx.set_ylabel('Count')
        if set_plot_offset_log:
            ax.set_yscale('log')
            ax.set_ylim(0.1, 10)
            ax.set_yticks([0.1, 1, 10])
            ax.set_yticklabels(['0.1', '1', '10'])
        else:
            ax.set_ylim(0, 3)
        ax.set_xticks(bins)
        ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
        
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.set_xticks([1, 10, 100, 1000])
        ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
        
        #------------
        ### Add title
        if set_plot_offset_type:
            if 'co-co' in set_plot_offset_type:
                ax_histx.set_title(r'co → co', size=7, loc='left', pad=3)
            if 'counter-counter' in set_plot_offset_type:
                ax_histx.set_title(r'counter → counter', size=7, loc='left', pad=3)
            if 'co-counter' in set_plot_offset_type:
                ax_histx.set_title(r'co → counter', size=7, loc='left', pad=3)
            if 'counter-co' in set_plot_offset_type:
                ax_histx.set_title(r'counter → co', size=7, loc='left', pad=3)
                
        
        #------------
        # Legend
        ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #------------
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/offset_angle/trelax_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/offset_angle/trelax_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    # Plot delta angle, tdyn. Looks at peak angle from 180
    if _plot_offset_tdyn:
        relaxationtime_plot  = []
        angles_plot = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # check offset angle within range            
            if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
                continue
            if set_plot_offset_type:
                if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                    continue
                                
            # Add angles
            angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            ID_plot.append(ID_i)
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Peak misalignment angle': angles_plot,  'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
                
        #-------------
        # Plotting
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        bin_width = 15
        bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
        c = 'k'
        
        
        #-------------
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.05, c='dimgrey', marker='.', alpha=1, zorder=1)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c=c, ls='-', zorder=7)
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
        ### Plot histograms
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        
        #-----------
        ### General formatting
        # Axis labels
        ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        ax.set_xlabel('Peak misalignment angle')
        ax_histy.set_xlabel('Count')
        ax_histx.set_ylabel('Count')
        if set_plot_offset_log:
            ax.set_yscale('log')
            ax.set_ylim(0.3, 25)
            ax.set_yticks([1, 10])
            ax.set_yticklabels(['1', '10'])
        else:
            ax.set_ylim(0, 15)
        ax.set_xticks(bins)
        ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
        
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.set_xticks([1, 10, 100, 1000])
        ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
        
        #------------
        ### Add title
        if set_plot_offset_type:
            if 'co-co' in set_plot_offset_type:
                ax_histx.set_title(r'co → co', size=7, loc='left', pad=3)
            if 'counter-counter' in set_plot_offset_type:
                ax_histx.set_title(r'counter → counter', size=7, loc='left', pad=3)
            if 'co-counter' in set_plot_offset_type:
                ax_histx.set_title(r'co → counter', size=7, loc='left', pad=3)
            if 'counter-co' in set_plot_offset_type:
                ax_histx.set_title(r'counter → co', size=7, loc='left', pad=3)
                
        
        #------------
        # Legend
        #ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #------------
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/offset_angle/tdyn_offset_angle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/offset_angle/tdyn_offset_angle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if _plot_offset_morph_tdyn:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        angles_plot = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # check offset angle within range            
            if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
                continue
            # check morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_offset_morphs:
                continue
            if set_plot_offset_type:
                if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                    continue
                                
            # Add angles
            angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            ID_plot.append(ID_i)
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
                
        #-------------
        # Plotting
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        bin_width = 15
        bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
        c = 'k'
        
        #-------------
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.05, c='darkgrey', edgecolor='k', marker='.', alpha=1, zorder=20)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=99, label='sample')
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
        ### Plot histograms
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        
        
        #-------------
        ### Plot by morphology
        for offset_morph_i in set_offset_morphs:
            
            if offset_morph_i == 'ETG-ETG':
                offset_morph_i = 'ETG → ETG'
                c = 'r'
                ls = '--'
            elif offset_morph_i == 'LTG-LTG':
                offset_morph_i = 'LTG → LTG'
                c = 'b'
                ls = 'dashdot'
            
            # Dataframe of morphs matching this
            df_morph = df.loc[df['Morphology'] == offset_morph_i]
            
            # Bin hist data, find sigma percentiles
            binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
            bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
            
            #-------------------
            ### Plot scatter
            #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
            
            ### Plot upper, median, and lower sigma
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
            ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.9)
            #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
            
            
            ### Plot histograms
            ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
            ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
        
        #-----------
        ### General formatting
        # Axis labels
        ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        ax.set_xlabel('Peak misalignment angle')
        ax_histy.set_xlabel('Count')
        ax_histx.set_ylabel('Count')
        if set_plot_offset_log:
            ax.set_yscale('log')
            ax.set_ylim(0.5, 35)
            ax.set_yticks([1, 10])
            ax.set_yticklabels(['1', '10'])
        else:
            ax.set_ylim(0, 15)
        ax.set_xticks(bins)
        ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
        
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.set_xticks([1, 10, 100, 1000])
        ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
        
        #------------
        ### Add title
        if set_plot_offset_type:
            if 'co-co' in set_plot_offset_type:
                ax_histx.set_title(r'co → co', size=7, loc='left', pad=3)
            if 'counter-counter' in set_plot_offset_type:
                ax_histx.set_title(r'counter → counter', size=7, loc='left', pad=3)
            if 'co-counter' in set_plot_offset_type:
                ax_histx.set_title(r'co → counter', size=7, loc='left', pad=3)
            if 'counter-co' in set_plot_offset_type:
                ax_histx.set_title(r'counter → co', size=7, loc='left', pad=3)
                
        
        #------------
        # Legend
        ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #------------
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/offset_angle/tdyn_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/offset_angle/tdyn_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    # Plot delta angle, ttorque. Looks at peak angle from 180
    if _plot_offset_ttorque:
        relaxationtime_plot  = []
        angles_plot = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # check offset angle within range            
            if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
                continue
            if set_plot_offset_type:
                if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                    continue
                                
            # Add angles
            angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            ID_plot.append(ID_i)
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
                
        #-------------
        # Plotting
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        bin_width = 15
        bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
        c = 'k'
        
        #-------------
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
            
        #-------------------
        ### Plot scatter
        ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.05, c='dimgrey', marker='.', alpha=1, zorder=1)
            
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c=c, ls='-', zorder=7)
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
            
            
        ### Plot histograms
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
        
        #-----------
        ### General formatting
        # Axis labels
        ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        ax.set_xlabel('Peak misalignment angle')
        ax_histy.set_xlabel('Count')
        ax_histx.set_ylabel('Count')
        if set_plot_offset_log:
            ax.set_yscale('log')
            ax.set_ylim(0.1, 25)
            ax.set_yticks([0.1, 1, 10])
            ax.set_yticklabels(['0.1', '1', '10'])
        else:
            ax.set_ylim(0, 10)
        ax.set_xticks(bins)
        ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
        
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.set_xticks([1, 10, 100, 1000])
        ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
        
        #------------
        ### Add title
        if set_plot_offset_type:
            if 'co-co' in set_plot_offset_type:
                ax_histx.set_title(r'co → co', size=7, loc='left', pad=3)
            if 'counter-counter' in set_plot_offset_type:
                ax_histx.set_title(r'counter → counter', size=7, loc='left', pad=3)
            if 'co-counter' in set_plot_offset_type:
                ax_histx.set_title(r'co → counter', size=7, loc='left', pad=3)
            if 'counter-co' in set_plot_offset_type:
                ax_histx.set_title(r'counter → co', size=7, loc='left', pad=3)
                
        
        #------------
        # Legend
        #ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #------------
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/offset_angle/ttorque_offset_angle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/offset_angle/ttorque_offset_angle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if _plot_offset_morph_ttorque:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        angles_plot = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # check offset angle within range            
            if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
                continue
            # check morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_offset_morphs:
                continue
            if set_plot_offset_type:
                if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                    continue
                                
            # Add angles
            angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            ID_plot.append(ID_i)
        
        # Collect data into dataframe
        df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
                
        #-------------
        # Plotting
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        bin_width = 15
        bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
        c = 'k'
        
        #-------------
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
            
        #-------------------
        ### Plot scatter
        ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.05, c='darkgrey', edgecolor='k', marker='.', alpha=1, zorder=20)
            
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=99, label='sample')
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
            
            
        ### Plot histograms
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
        ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
        ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
        #-------------
        ### Plot by morphology
        for offset_morph_i in set_offset_morphs:
            
            if offset_morph_i == 'ETG-ETG':
                offset_morph_i = 'ETG → ETG'
                c = 'r'
                ls = '--'
            elif offset_morph_i == 'LTG-LTG':
                offset_morph_i = 'LTG → LTG'
                c = 'b'
                ls = 'dashdot'
            
            # Dataframe of morphs matching this
            df_morph = df.loc[df['Morphology'] == offset_morph_i]
            
            # Bin hist data, find sigma percentiles
            binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
            bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
            
            #-------------------
            ### Plot scatter
            #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
            
            ### Plot upper, median, and lower sigma
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
            ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.9)
            #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
            
            
            ### Plot histograms
            ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
            ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor='none', linewidth=0.8, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
            
        
        #-----------
        ### General formatting
        # Axis labels
        ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        ax.set_xlabel('Peak misalignment angle')
        ax_histy.set_xlabel('Count')
        ax_histx.set_ylabel('Count')
        if set_plot_offset_log:
            ax.set_yscale('log')
            ax.set_ylim(0.2, 25)
            ax.set_yticks([0.2, 1, 10])
            ax.set_yticklabels(['0.2', '1', '10'])
        else:
            ax.set_ylim(0, 10)
        ax.set_xticks(bins)
        ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
        
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.set_xticks([1, 10, 100, 1000])
        ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
        
        #------------
        ### Add title
        if set_plot_offset_type:
            if 'co-co' in set_plot_offset_type:
                ax_histx.set_title(r'co → co', size=7, loc='left', pad=3)
            if 'counter-counter' in set_plot_offset_type:
                ax_histx.set_title(r'counter → counter', size=7, loc='left', pad=3)
            if 'co-counter' in set_plot_offset_type:
                ax_histx.set_title(r'co → counter', size=7, loc='left', pad=3)
            if 'counter-co' in set_plot_offset_type:
                ax_histx.set_title(r'counter → co', size=7, loc='left', pad=3)
                
        
        #------------
        # Legend
        ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #------------
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/offset_angle/ttorque_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/offset_angle/ttorque_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    #-------------------------
    # Number of mergers within window with relaxation time (suited for mass of 1010 and above)
    if _plot_merger_count_trelax:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        mergercount_plot     = []
        mergerstarratio_plot = []
        mergergasratio_plot  = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # remove misalignments that are too short
            if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_merger_trelax:
                continue
            
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            
            # Find mergers within window
            merger_count = 0
            ratio_star_list = []
            ratio_gas_list  = [] 
            for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
                if len(merger_i) > 0:
                    for merger_ii, gas_ii in zip(merger_i, gas_i):
                        if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                            merger_count += 1
                            ratio_star_list.append(merger_ii)
                            ratio_gas_list.append(gas_ii)
            
            # Append number of mergers, and average stellar and gas ratio of these mergers
            mergercount_plot.append(merger_count)
            mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
            mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
            
                      
        # Collect data into dataframe
        df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
        
        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
        print('\n--------------------------------------')
        print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
        print('NUMBER OF MERGERS > 0.1 - RELAXATION TIME SPEARMAN:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
        
        df_0       = df.loc[(df['Number of mergers'] == 0)]
        df_mergers = df.loc[(df['Number of mergers'] > 0)]
        
        im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
        axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
        
        plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')
        
        
        #-------------
        ### Formatting
        axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
        axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
        if set_plot_merger_count_log:
            axs.set_xscale('log')
            axs.set_xlim(0.1, 6)
            axs.set_xticks([0.1, 1, 10])
            axs.set_xticklabels(['0.1', '1', '10'])
        else:
            axs.set_xlim(0.5, 6)
            axs.set_xticks(np.arange(1, 6.1))
        axs.set_ylim(-0.3, 3.3)
        axs.set_yticks(np.arange(0, 3.1, 1))
        #axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### title
        axs.set_title(r'$\rho$ = %.2f p-value = %.1e' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #------------
        # Legend
        axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/number_mergers_relaxtime/trelax_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/number_mergers_relaxtime/trelax_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    # tdyn
    if _plot_merger_count_tdyn:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        mergercount_plot     = []
        mergerstarratio_plot = []
        mergergasratio_plot  = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # remove misalignments that are too short
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] <= set_min_merger_tdyn:
                continue
            
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            
            # Find mergers within window
            merger_count = 0
            ratio_star_list = []
            ratio_gas_list  = [] 
            for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
                if len(merger_i) > 0:
                    for merger_ii, gas_ii in zip(merger_i, gas_i):
                        if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                            merger_count += 1
                            ratio_star_list.append(merger_ii)
                            ratio_gas_list.append(gas_ii)
            
            # Append number of mergers, and average stellar and gas ratio of these mergers
            mergercount_plot.append(merger_count)
            mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
            mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
            
                            
        # Collect data into dataframe
        df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
        
        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
        print('\n--------------------------------------')
        print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
        print('NUMBER OF MERGERS > 0.1 - RELAXATION TDYN SPEARMAN:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
        
        df_0       = df.loc[(df['Number of mergers'] == 0)]
        df_mergers = df.loc[(df['Number of mergers'] > 0)]
        
        im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
        axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
        
        plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')
        
        
        #-------------
        ### Formatting
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
        if set_plot_merger_count_log:
            axs.set_xscale('log')
            axs.set_xticks([0.1, 1, 10])
            axs.set_xticklabels(['0.1', '1', '10'])
            axs.set_xlim(0.6, 25)
        else:
            axs.set_xlim(0, 20)
            axs.set_xticks(np.arange(4, 20.1, 4))
        axs.set_ylim(-0.3, 3.3)
        axs.set_yticks(np.arange(0, 3.1, 1))
        #axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### title
        axs.set_title(r'$\rho$ = %.2f p-value = %.2f' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #------------
        # Legend
        axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/number_mergers_relaxtime/tdyn_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/number_mergers_relaxtime/tdyn_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    # ttorque   
    if _plot_merger_count_ttorque:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        mergercount_plot     = []
        mergerstarratio_plot = []
        mergergasratio_plot  = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # remove misalignments that are too short
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] <= set_min_merger_ttorque:
                continue
            
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            
            # Find mergers within window
            merger_count = 0
            ratio_star_list = []
            ratio_gas_list  = [] 
            for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
                if len(merger_i) > 0:
                    for merger_ii, gas_ii in zip(merger_i, gas_i):
                        if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                            merger_count += 1
                            ratio_star_list.append(merger_ii)
                            ratio_gas_list.append(gas_ii)
            
            # Append number of mergers, and average stellar and gas ratio of these mergers
            mergercount_plot.append(merger_count)
            mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
            mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
            
                            
        # Collect data into dataframe
        df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
        
        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
        print('\n--------------------------------------')
        print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
        print('NUMBER OF MERGERS > 0.1 - RELAXATION TTORQUE SPEARMAN:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
        
        df_0       = df.loc[(df['Number of mergers'] == 0)]
        df_mergers = df.loc[(df['Number of mergers'] > 0)]
        
        im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
        axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
        
        plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')
    
        
        #-------------
        ### Formatting
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
        if set_plot_merger_count_log:
            axs.set_xscale('log')
            axs.set_xticks([0.1, 1, 10])
            axs.set_xticklabels(['0.1', '1', '10'])
            axs.set_xlim(0.6, 15)
        else:
            axs.set_xlim(0.5, 12)
            axs.set_xticks(np.arange(2, 12.1, 2))
        axs.set_ylim(-0.3, 3.3)
        axs.set_yticks(np.arange(0, 3.1, 1))
        #axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### title
        axs.set_title(r'$\rho$ = %.2f p-value = %.2f' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #------------
        # Legend
        axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
        
        #------------
        ### other
        plt.tight_layout()
        
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/number_mergers_relaxtime/ttorque_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/number_mergers_relaxtime/ttorque_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    #-------------------------
    # Plots average DM-stars misangle over relaxation with trelax, coloured by average morphology
    if _plot_halo_misangle_trelax:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        relaxationkappa_plot = []
        halomisangle_plot    = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            
            # remove misalignments that are too below resolution
            if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
                continue
            
            # remove misalignments that are too short
            if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_halo_trelax:
                continue
                
            # Collect relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
            # Collect average stellar-DM misalignment angle
            halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
                            
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
        
        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
        print('\n--------------------------------------')
        print('Size of halo-misangle trelax plot: ', len(ID_plot))
        print('Stars-DM misangle vs trelax:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        #-------------
        # Plotting scatter
        fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
        
        im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
        plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
        
        
        #---------------
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
        
        # Plot average line for total sample
        ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        if add_plot_halo_morph_median:
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
            ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
            ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
        
        
        #-------------
        # Plotting hist
        #bins = np.arange(0, 181, 30)
        #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
        
        
        #-------------
        # Plotting average kappa
        
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        
        ### Plot upper, median, and lower sigma
        ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
        
        
        #-------------
        ### Formatting
        ax_scatter.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
        #ax_hist.set_ylabel('Count')
        ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
        ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
        
        if set_plot_halo_misangle_log:
            ax_scatter.set_yscale('log')
            ax_scatter.set_ylim(0.1, 10)
            ax_scatter.set_yticks([0.1, 1, 10])
            ax_scatter.set_yticklabels(['0.1', '1', '10'])
        else:
            ax_scatter.set_ylim(0, 6)
            ax_scatter.set_yticks(np.arange(0, 6.1, 1))
        #ax_hist.set_yscale('log')
        #ax_hist.set_ylim(bottom=0)
        ax_line.set_ylim(0.1, 0.6)
        ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
        ax_scatter.set_xlim(0, 180)
        ax_scatter.set_xticks(np.arange(0, 180.1, 30))
        for ax in [ax_scatter, ax_line]:
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
            
        #-----------
        ### title
        if plot_annotate:
            ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #------------
        # Legend
        ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
        #------------
        ### other
        #plt.tight_layout()
    
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/halo_misangle_relaxtime/trelax_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/halo_misangle_relaxtime/trelax_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    # tdyn
    if _plot_halo_misangle_tdyn:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        relaxationkappa_plot = []
        halomisangle_plot    = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            
            # remove misalignments that are too below resolution
            if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
                continue
            
            # remove misalignments that are too short
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] <= set_min_halo_tdyn:
                continue
                
                
            # Collect relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
            # Collect average stellar-DM misalignment angle
            halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
                            
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
        
        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
        print('\n--------------------------------------')
        print('Size of halo-misangle tdyn plot: ', len(ID_plot))
        print('Stars-DM misangle vs tdyn:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        #-------------
        # Plotting scatter
        fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
        
        im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
        plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
        
        
        #--------------
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
        
        # Plot average line
        ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        if add_plot_halo_morph_median:
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
            ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
            ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
        
        
        #-------------
        # Plotting hist
        #bins = np.arange(0, 181, 30)
        #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
        
        
        #-------------
        # Plotting average kappa
        
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        
        ### Plot upper, median, and lower sigma
        ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
        
        
        #-------------
        ### Formatting
        ax_scatter.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        #ax_hist.set_ylabel('Count')
        ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
        ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
        if set_plot_halo_misangle_log:
            ax_scatter.set_yscale('log')
            ax_scatter.set_ylim(0.3, 25)
            ax_scatter.set_yticks([1, 10])
            ax_scatter.set_yticklabels(['1', '10'])
        else:
            ax_scatter.set_ylim(0, 15)
        #ax_hist.set_yscale('log')
        #ax_hist.set_ylim(bottom=0)
        ax_line.set_ylim(0.1, 0.6)
        ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
        ax_scatter.set_xlim(0, 180)
        ax_scatter.set_xticks(np.arange(0, 180.1, 30))
        for ax in [ax_scatter, ax_line]:
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
            
        #-----------
        ### title
        if plot_annotate:
            ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #------------
        # Legend
        ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
        #------------
        ### other
        #plt.tight_layout()
    
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/halo_misangle_relaxtime/tdyn_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/halo_misangle_relaxtime/tdyn_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    #ttorque    
    if _plot_halo_misangle_ttorque:
        relaxationtime_plot  = []
        relaxationmorph_plot = []
        relaxationkappa_plot = []
        halomisangle_plot    = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            
            # remove misalignments that are too below resolution
            if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
                continue
            
            # remove misalignments that are too short
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] <= set_min_halo_ttorque:
                continue
                
            # Collect relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
            # Collect average stellar-DM misalignment angle
            halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
                            
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
        
        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
        print('\n--------------------------------------')
        print('Size of halo-misangle ttorque plot: ', len(ID_plot))
        print('Stars-DM misangle vs ttorque:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        #-------------
        # Plotting scatter
        fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
        
        #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
        plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
        
        #--------------
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
        
        # Plot average line
        ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101)
        
        if add_plot_halo_morph_median:
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
            ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
            ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
            
        
        #-------------
        # Plotting hist
        #bins = np.arange(0, 181, 30)
        #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
        
        
        #-------------
        # Plotting average kappa
        
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        
        ### Plot upper, median, and lower sigma
        ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
        
        
        #-------------
        ### Formatting
        ax_scatter.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        #ax_hist.set_ylabel('Count')
        ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
        ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
        if set_plot_halo_misangle_log:
            ax_scatter.set_yscale('log')
            ax_scatter.set_ylim(0.1, 25)
            ax_scatter.set_yticks([0.1, 1, 10])
            ax_scatter.set_yticklabels(['0.1', '1', '10'])
        else:
            ax_scatter.set_ylim(0, 10)
        #ax_hist.set_yscale('log')
        #ax_hist.set_ylim(bottom=0)
        ax_line.set_ylim(0.1, 0.6)
        ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
        ax_scatter.set_xlim(0, 180)
        ax_scatter.set_xticks(np.arange(0, 180.1, 30))
        for ax in [ax_scatter, ax_line]:
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
            
        #-----------
        ### title
        if plot_annotate:
            ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #------------
        # Legend
        ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
        #------------
        ### other
        #plt.tight_layout()
    
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/halo_misangle_relaxtime/ttorque_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/halo_misangle_relaxtime/ttorque_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    # Manual
    if _plot_halo_misangle_manual:
        relaxationtime_plot  = []
        relaxationtorque_plot  = []
        relaxationmorph_plot = []
        relaxationkappa_plot = []
        halomisangle_plot    = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            
            # remove misalignments that are too below resolution
            if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
                continue
            
            # remove misalignments that are too short
            if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_halo_trelax:
                continue
                
            # Collect relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            relaxationtorque_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
            # Collect average stellar-DM misalignment angle
            halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
            
                            
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation torque': relaxationtorque_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
        
        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
        print('\n--------------------------------------')
        print('Size of halo-misangle trelax plot: ', len(ID_plot))
        print('--------------------------------------')
        
        #-------------
        # Plotting scatter
        fig, (ax_scatter1, ax_scatter2, ax_line) = plt.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [2.5, 2.5, 1]}, figsize=[10/3, 4.0], sharex=True, sharey=False, layout='constrained')
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
        
        im1 = ax_scatter1.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=0.9)
        plt.colorbar(im1, ax=ax_scatter1, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
        
        im2 = ax_scatter2.scatter(df['DM-stars angle'], df['Relaxation torque'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=0.9)
        plt.colorbar(im2, ax=ax_scatter2, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
        
        
        #---------------
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        print(bin_medians)
        
        # Plot average line for total sample
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='--', zorder=101)
        ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=101, label='sample')
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=1, c='k', ls='--', zorder=101)

        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation torque'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        print(bin_medians)
        
        # Plot average line for total sample
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='--', zorder=101)
        ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=101, label='sample')
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=1, c='k', ls='--', zorder=101)
        
        if add_plot_halo_morph_median:
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
            #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='b', alpha=0.2, zorder=6)
            ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
            #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='b', ls='--', zorder=101)
            #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='b', ls='--', zorder=101)
        
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
            #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='r', alpha=0.2, zorder=6)
            ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$')
            #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='r', ls='--', zorder=101)
            #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='r', ls='--', zorder=101)
        
            #-------------
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation torque'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
            #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='b', alpha=0.2, zorder=6)
            ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
            #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='b', ls='--', zorder=101)
            #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='b', ls='--', zorder=101)
        
            # Bin hist data, find sigma percentiles
            bin_width = 15
            bins = np.arange(0, 181, bin_width)
            binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation torque'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
            #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='r', alpha=0.2, zorder=6)
            ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$')
            #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='r', ls='--', zorder=101)
            #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='r', ls='--', zorder=101)
            
        
        #-------------
        # Plotting hist
        #bins = np.arange(0, 181, 30)
        #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
        #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
        
        
        #-------------
        # Plotting average kappa
        
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        
        ### Plot upper, median, and lower sigma
        #ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
        ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=7)
        
        
        #-------------
        ### Formatting
        ax_scatter1.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
        ax_scatter2.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        #ax_hist.set_ylabel('Count')
        ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
        ax_line.set_xlabel('Average stellar-DM misalignment angle')
        if set_plot_halo_misangle_log:
            ax_scatter1.set_yscale('log')
            ax_scatter1.set_ylim(0.1, 10)
            ax_scatter1.set_yticks([0.1, 1, 10])
            ax_scatter1.set_yticklabels(['0.1', '1', '10'])
            ax_scatter2.set_yscale('log')
            ax_scatter2.set_ylim(0.1, 25)
            ax_scatter2.set_yticks([0.1, 1, 10])
            ax_scatter2.set_yticklabels(['0.1', '1', '10'])
        else:
            ax_scatter1.set_ylim(0, 6)
            ax_scatter1.set_yticks(np.arange(0, 6.1, 1))
            ax_scatter2.set_ylim(0, 10)
        #ax_hist.set_yscale('log')
        #ax_hist.set_ylim(bottom=0)
        ax_line.set_ylim(0.1, 0.6)
        ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
        ax_scatter1.set_xlim(0, 180)
        ax_scatter1.set_xticks(np.arange(0, 180.1, 30))
        for ax in [ax_scatter1, ax_scatter2, ax_line]:
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
            
        #-----------
        ### title
        if plot_annotate:
            ax_scatter1.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #------------
        # Legend
        ax_scatter1.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
        ax_scatter2.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
        #------------
        ### other
        #plt.tight_layout()
    
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/halo_misangle_relaxtime/both_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/halo_misangle_relaxtime/both_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    #-------------------------
    # Plots hist of average DM-stars misangle co-co preceeding misalignment with fraction
    if _plot_halo_misangle_pre_frac:
        halomisangle_plot    = []
        relaxationmorph_plot = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            
            # If not a co-co or co-counter, skip
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                continue
            
            
            # Collect average stellar-DM misangle before misalignment.
            halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][0:misalignment_tree['%s' %ID_i]['index_s']+1]))
            ID_plot.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
                            
        # Collect data into dataframe
        df = pd.DataFrame(data={'DM-stars angle': halomisangle_plot, 'Relaxation morph': relaxationmorph_plot, 'GalaxyIDs': ID_plot})
        ETG_ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
        LTG_LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
        ETG_LTG_df = df.loc[(df['Relaxation morph'] == 'ETG → LTG')]
        LTG_ETG_df = df.loc[(df['Relaxation morph'] == 'LTG → ETG')]
        
        #-------------
        # Plotting ongoing fraction histogram
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        colors = ['orangered', 'orange', 'cornflowerblue', 'mediumblue']
        
        if set_misanglepre_morph:
            axs.hist([ETG_ETG_df['DM-stars angle'], ETG_LTG_df['DM-stars angle'], LTG_ETG_df['DM-stars angle'], LTG_LTG_df['DM-stars angle']], weights=(np.ones(len(ETG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(ETG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs'])), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', color = colors, alpha=0.5, stacked=True)
            hist_n, _, _ = axs.hist([ETG_ETG_df['DM-stars angle'], ETG_LTG_df['DM-stars angle'], LTG_ETG_df['DM-stars angle'], LTG_LTG_df['DM-stars angle']], weights=(np.ones(len(ETG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(ETG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs'])), bins=np.arange(0, 181, 10), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
        else:
            axs.hist(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', alpha=0.5)
            hist_n, _, _ = axs.hist(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7)
            
        print('Hist bins morphology: ', hist_n)
        hist_n, _ = np.histogram(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), range=(0, 181))
        print('Hist bins total: ', hist_n)
            
        #-------------
        ### Formatting
        axs.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ pre-instability')
        axs.set_xlim(0, 180)
        axs.set_xticks(np.arange(0, 181, step=30))
        axs.set_ylabel('Percentage of\nmisalignments')
        axs.set_ylim(0, 0.2)
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
        #axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #------------
        # Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        
        
        if set_misanglepre_morph:
            legend_labels.append('ETG → ETG')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orangered')
                
            legend_labels.append('ETG → LTG')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
                
            legend_labels.append('LTG → ETG')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('cornflowerblue')
                
            legend_labels.append('LTG → LTG')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('mediumblue')
            
            axs.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            
        #------------
        ### other
        #plt.tight_layout()
    
        #-----------
        # savefig
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/halo_premisangle_fraction/halopremisangle_fraction_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/halo_premisangle_fraction/halopremisangle_fraction_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    #-------------------------
    # Plots stacked histograms for different morphologies, and fraction caused by major, minor, or other origins
    if _plot_origins:
        # Loading mergertree file to establish windows
        f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
        GalaxyID_tree             = np.array(f['Histories']['GalaxyID'])
        DescendantID_tree         = np.array(f['Histories']['DescendantID'])
        Lookbacktime_tree         = np.array(f['Snapnum_Index']['LookbackTime'])
        StellarMass_tree          = np.array(f['Histories']['StellarMass'])
        GasMass_tree              = np.array(f['Histories']['GasMass'])
        f.close()
        
        tally_minor = []
        tally_major = []
        tally_other = []
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_origins_morph:
                continue
                
            
            #---------------------------------------------------------
            # Find location of begin of misalignment in merger tree
            (row_i, snap_i) = np.where(GalaxyID_tree == int(misalignment_tree['%s' %ID_i]['GalaxyID'][misalignment_tree['%s' %ID_i]['index_s']+1]))
            row_mask  = row_i[0]
            snap_mask = snap_i[0]
            
            # Find window limits of mergers [SnapNum_merger_min:SnapNum_merger_max]
            SnapNum_merger_min = 1 + np.where(Lookbacktime_tree >= (Lookbacktime_tree[snap_i] + half_window))[0][-1]
            if len(np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0]) > 0:
                SnapNum_merger_max = np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0][0]
            else:
                SnapNum_merger_max = snap_mask
            
            # List of all elligible descendants
            GalaxyID_list       = np.array(GalaxyID_tree)[row_mask, SnapNum_merger_min:SnapNum_merger_max]
            

            merger_ID_array_array    = []
            merger_ratio_array_array = []
            merger_gas_array_array   = []
            for SnapNum_i, GalaxyID_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), GalaxyID_list):
                if int(GalaxyID_i) == -1:
                    continue
                
                merger_mask = [i for i in np.where(np.array(DescendantID_tree)[:,int(SnapNum_i-1)] == GalaxyID_i)[0] if i != row_mask]
                
                # If misalignment found, its position is given by i in merger_mask, SnapNum_i
                merger_ID_array    = []
                merger_ratio_array = []
                merger_gas_array   = []
                if len(merger_mask) > 0:
                    # find peak stelmass of those galaxies
                    for mask_i in merger_mask:
                        # Find last snap up to 2 Gyr ago
                        SnapNum_merger = np.where(Lookbacktime_tree >= (Lookbacktime_tree[SnapNum_i] + merger_lookback_time))[0][-1]
                
                        # Find largest stellar mass of this satellite, per method of Rodriguez-Gomez et al. 2015, Qu et al. 2017 (see crain2017)
                        mass_mask = np.argmax(StellarMass_tree[mask_i][int(SnapNum_merger-100):int(SnapNum_i)]) + (SnapNum_merger-100)
                
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
                        if merger_ratio > 1:
                            merger_ratio = 1/merger_ratio
                        gas_ratio    = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)

                        # Append
                        merger_ID_array.append(GalaxyID_tree[mask_i][int(SnapNum_i-1)])
                        merger_ratio_array.append(merger_ratio)
                        merger_gas_array.append(gas_ratio)
                        
                merger_ID_array_array.append(merger_ID_array)
                merger_ratio_array_array.append(merger_ratio_array)
                merger_gas_array_array.append(merger_gas_array)      
            if debug:
                print(misalignment_tree['%s' %ID_i]['SnapNum'])
                print(misalignment_tree['%s' %ID_i]['merger_ratio_stars'])
                for snap_i, star_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), merger_ratio_array_array):
                    print(snap_i, star_i)
                    
                
            merger_count = 0
            for merger_i in merger_ratio_array_array:
                if len(merger_i) > 0:
                    if (max(merger_i) > min_ratio):
                        if merger_count == 0:
                            if 0.3 > max(merger_i) > 0.1:
                                tally_minor.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                            if max(merger_i) > 0.3:
                                tally_major.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                        merger_count += 1
            if merger_count == 0:
                tally_other.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                        
        
        print('\n======================================')
        print('Using merger criteria, half_window = %.1f Gyr, min_ratio = %.1f' %(half_window, min_ratio))
        print('\tmajor ', len(tally_major))
        print('\tminor ', len(tally_minor))
        print('\tother ', len(tally_other))
        print('\ttotal ', (len(tally_major)+len(tally_minor)+len(tally_other)))
        
        
        # left with tallo_minor = ['ETG-ETG', 'LTG-ETG', 'ETG-ETG', etc..]
        plot_dict = {'ETG-ETG': {}, 'LTG-ETG': {}, 'ETG-LTG': {}, 'LTG-LTG': {}}
        for dict_i in plot_dict.keys():
            plot_dict[dict_i] = {'array': [],
                                 'major': [],
                                 'minor': [],
                                 'other': []}
        
        for morph_i in set_origins_morph:
            plot_dict['%s' %morph_i]['major'] = len([i for i in tally_major if i == morph_i])
            plot_dict['%s' %morph_i]['minor'] = len([i for i in tally_minor if i == morph_i])
            plot_dict['%s' %morph_i]['other'] = len([i for i in tally_other if i == morph_i])
            
            temp_array = []
            temp_array.extend([i for i in tally_major if i == morph_i])
            temp_array.extend([i for i in tally_minor if i == morph_i])
            temp_array.extend([i for i in tally_other if i == morph_i])
            plot_dict['%s' %morph_i]['array'] = temp_array
            
            
            
        #---------------------
        ### Plotting
        if not use_only_start_morph:
            #-----------
            # Plotting ongoing fraction histogram
            fig, ((ax_ETG_ETG, ax_ETG_LTG), (ax_LTG_ETG, ax_LTG_LTG), (ax_spare_1, ax_spare_2)) = plt.subplots(3, 2, figsize=[10/3, 10/3], gridspec_kw={'height_ratios': [1, 1, 0.1]}, sharex=True, sharey=False)
            plt.subplots_adjust(wspace=0, hspace=0)
            
            #-----------
            # Add pie plots
            colors = ['royalblue', 'orange', 'orangered']
            for morph_i in set_origins_morph:
                y = np.array([plot_dict['%s' %morph_i]['other'], plot_dict['%s' %morph_i]['minor'], plot_dict['%s' %morph_i]['major']])
                mylabels = ['%s' %(' ' if plot_dict['%s' %morph_i]['other'] == 0 else plot_dict['%s' %morph_i]['other']), '%s' %(' ' if plot_dict['%s' %morph_i]['minor'] == 0 else plot_dict['%s' %morph_i]['minor']), '%s' %(' ' if plot_dict['%s' %morph_i]['major'] == 0 else plot_dict['%s' %morph_i]['major'])]
            
                if morph_i == 'ETG-ETG':
                    ax_ETG_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                    ax_ETG_ETG.set_title('ETG → ETG', loc='center', pad=0, fontsize=8)
                if morph_i == 'ETG-LTG':
                    ax_ETG_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                    ax_ETG_LTG.set_title('ETG → LTG', loc='center', pad=0, fontsize=8)
                if morph_i == 'LTG-ETG':
                    ax_LTG_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                    ax_spare_1.set_title('\nLTG → ETG', loc='center', y=-1, fontsize=8)
                if morph_i == 'LTG-LTG':
                    ax_LTG_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                    ax_spare_2.set_title('\nLTG → LTG', loc='center', y=-1, fontsize=8)
            ax_spare_1.axis('off')
            ax_spare_2.axis('off')
        
                
            #------------
            # Add legend
            legend_labels   = []
            legend_elements = []
            legend_colors   = []

            legend_labels.append('Major\nmerger')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
            legend_labels.append('Minor\nmerger')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
            legend_labels.append('Other')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        
            ax_LTG_LTG.legend(handles=legend_elements, labels=legend_labels, loc='lower left', bbox_to_anchor=(0.9, 0.7), labelspacing=1, frameon=False, labelcolor=legend_colors, handlelength=0, ncol=1)
            
            #-----------
            # savefig
            if savefig:
                if savefig_txt == 'manual':
                    savefig_txt = input('\n  -> Enter savefig_txt:   ')
                plt.savefig("%s/misangle_origins/origins_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
                print("\n  SAVED: %s/misangle_origins/origins_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format)) 
            if showfig:
                plt.show()
            plt.close()
            
        else:
            #-----------
            # Plotting ongoing fraction histogram
            fig, (ax_ETG, ax_LTG) = plt.subplots(1, 2, figsize=[10/3, 2.5], sharex=True, sharey=False)
            plt.subplots_adjust(wspace=0, hspace=0)
            
            #-----------
            # Add pie plots
            #colors = ['orange', 'cornflowerblue', 'blue']
            colors = ['royalblue', 'orange', 'orangered']
            for morph_i in ['ETG', 'LTG']:
                
                y = np.array([plot_dict['%s-ETG' %morph_i]['other']+plot_dict['%s-LTG' %morph_i]['other'], plot_dict['%s-ETG' %morph_i]['minor']+plot_dict['%s-LTG' %morph_i]['minor'], plot_dict['%s-ETG' %morph_i]['major']+plot_dict['%s-LTG' %morph_i]['major']])
                mylabels = ['%s' %(' ' if y[0] == 0 else y[0]), '%s' %(' ' if y[1] == 0 else y[1]), '%s' %(' ' if y[2] == 0 else y[2])]
                
                if morph_i == 'ETG':
                    ax_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                    ax_ETG.set_title('ETG', loc='center', pad=0, fontsize=8)
                if morph_i == 'LTG':
                    ax_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                    ax_LTG.set_title('LTG', loc='center', pad=0, fontsize=8)
            
                
            #------------
            # Add legend
            legend_labels   = []
            legend_elements = []
            legend_colors   = []

            legend_labels.append('Major\nmerger')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orangered')
            legend_labels.append('Minor\nmerger')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('orange')
            legend_labels.append('other')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('royalblue')
        
            ax_LTG.legend(handles=legend_elements, labels=legend_labels, loc='center left', bbox_to_anchor=(0.9, 0.5), labelspacing=1, frameon=False, labelcolor=legend_colors, handlelength=0, ncol=1)
            
            #-----------
            # savefig
            if savefig:
                if savefig_txt == 'manual':
                    savefig_txt = input('\n  -> Enter savefig_txt:   ')
                plt.savefig("%s/misangle_origins/origins_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
                print("\n  SAVED: %s/misangle_origins/origins_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format)) 
            if showfig:
                plt.show()
            plt.close()
        
        
        
        """# Plotting ongoing fraction histogram
        fig, axs = plt.subplots(1, 1, figsize=[2.5, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        
        #------------
        # plotting bar
        bottom = np.zeros(4)
        weight_counts = {
            'Major': np.array([plot_dict['ETG-ETG']['major']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['major']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['major']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['major']/len(plot_dict['LTG-LTG']['array'])]),
            'Minor': np.array([plot_dict['ETG-ETG']['minor']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['minor']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['minor']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['minor']/len(plot_dict['LTG-LTG']['array'])]),
            'Other': np.array([plot_dict['ETG-ETG']['other']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['other']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['other']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['other']/len(plot_dict['LTG-LTG']['array'])])}
        bar_x = []
        for morph_i in set_origins_morph:
            if morph_i == 'ETG-ETG':
                bar_x.append('ETG →\nETG')
            if morph_i == 'LTG-ETG':
                bar_x.append('LTG →\nETG')
            if morph_i == 'ETG-LTG':
                bar_x.append('ETG →\nLTG')
            if morph_i == 'LTG-LTG':
                bar_x.append('LTG →\nLTG')
              
        for boolean, weight_count in weight_counts.items():
            axs.bar(bar_x, weight_count, width=0.5, bottom=bottom, edgecolor='none', alpha=0.5, label=boolean)
            axs.bar(bar_x, weight_count, width=0.5, bottom=bottom, facecolor='none', edgecolor='k', alpha=0.9, lw=0.7)
            bottom += weight_count
        
        
        #-------------
        ### Formatting
        axs.set_xlabel('Relaxation type')
        axs.set_ylabel('Percentage of misalignments')
        #axs.set_ylim(0, 1)
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
        #axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        
        #------------
        # Legend
        axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, labelspacing=0.1, handlelength=0.5)
        """
        
        
    #-------------------------
    # Plot timescale histogram of current criteria
    if _plot_timescale_gas_histogram:
        # Gather data
        relaxationtime_plot  = []
        relaxationtype_plot  = []
        relaxationkappa_plot = []
        gas_fraction_array   = []
        gassf_fraction_array = []
        ID_plot              = []
        for ID_i in misalignment_tree.keys():
            
            if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
                continue
        
            if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
                
                ID_plot.append(ID_i)
                relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
                
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                    relaxationtype_plot.append('C0')
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                    relaxationtype_plot.append('C1')
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                    relaxationtype_plot.append('C2')
                if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                    relaxationtype_plot.append('C3')
                    
                
                # take average gas fraction while unstable
                gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
                gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1], np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])))
                
                # Collect average kappa during misalignment
                relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
                
                
        # Collect data into dataframe
        df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

        
        #-------------
        ### Plotting scatter
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        
        #-------------
        # Colourbar for kappa
        norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
        
        im1 = axs.scatter(df['Gas fraction'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        plt.colorbar(im1, ax=axs, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
        
        
        #---------------
        # Bin hist data, find sigma percentiles
        bin_width = 0.1
        bins = np.arange(0, 1.1, bin_width)
        binned_data_arg = np.digitize(df['Gas fraction'], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
        
        # Plot average line for total sample
        axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        if add_plot_gas_morph_median:
            # Bin hist data, find sigma percentiles
            binned_data_arg = np.digitize(df['Gas fraction'][df['Relaxation kappa'] > 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
            axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        
            # Bin hist data, find sigma percentiles
            binned_data_arg = np.digitize(df['Gas fraction'][df['Relaxation kappa'] < 0.4], bins=bins)
            bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            # Plot average line for total sample
            #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
            axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
        
        #-------------
        ### Formatting
        axs.set_ylabel('$t_{\mathrm{relax}}$ (Gyr)')
        axs.set_xlabel('$f_{\mathrm{gas}}(<r_{50})$')
        axs.set_ylim(0, 4.5)
        axs.set_yticks(np.arange(0, 4.1, 1))
        axs.set_xlim(0, 1.1)
        axs.set_xticks(np.arange(0, 1.11, 0.2))
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        ### Savefig
        savefig_txt_2 = savefig_txt    
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/time_spent_misaligned_gas/scatter_trelax_gas_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/time_spent_misaligned_gas/scatter_trelax_gas_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format)) 
        if showfig:
            plt.show()
        plt.close()


        #===================================================================================
        gas_1_df = df.loc[(df['Gas fraction'] < 0.2)]
        gas_2_df = df.loc[(df['Gas fraction'] > 0.2) & (df['Gas fraction'] < 0.5)]
        gas_3_df = df.loc[(df['Gas fraction'] > 0.5)]
        
        
        #print('-------------------------------------------------------------')
        print('Number of coco relaxations: ', len(ID_plot))
        print('\tGas fractions:')
        print('\t0.0 - 0.2:    ', len(gas_1_df))
        print('\t0.2 - 0.5:    ', len(gas_2_df))
        print('\t0.5 +    :    ', len(gas_3_df))
        
        
        #-------------
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
                
        if set_bin_limit_trelax == None:
            set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
        if set_plot_histogram_log:
            set_plot_relaxation_type = False 
        
        #-------------
        ### Plot histogram
        axs.hist(gas_1_df['Relaxation time'], weights=np.ones(len(gas_1_df['Relaxation time']))/len(gas_1_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.9, lw=1.0)
        axs.hist(gas_2_df['Relaxation time'], weights=np.ones(len(gas_2_df['Relaxation time']))/len(gas_2_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.9, lw=1.0)
        axs.hist(gas_3_df['Relaxation time'], weights=np.ones(len(gas_3_df['Relaxation time']))/len(gas_3_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.9, lw=1.0)
        
        
        # Find bin values (sqrt N)
        hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
            
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
        axs.set_xlim(0, set_bin_limit_trelax)
        axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axs.set_xlabel('$t_{\mathrm{relax}}$ (Gyr)')
        axs.set_ylabel('Percentage of misalignments')
        axs.set_ylim(bottom=0.001)
        
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        
        legend_labels.append(r'$0.2<f_{\mathrm{gas}}$')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C0')
        
        legend_labels.append(r'$0.2<f_{\mathrm{gas}}<0.5$')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C1')
        
        legend_labels.append(r'$f_{\mathrm{gas}}>0.5$')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C2')
        
        ncol=1
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
        
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #-----------
        ### other
        plt.tight_layout()
        
        
        #-----------
        ### Savefig
        savefig_txt_2 = savefig_txt    
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/time_spent_misaligned_gas/%strelax_gas_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/time_spent_misaligned_gas/%strelax_gas_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt_2, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    #-------------------------
    ### OLD PLOTS
    """ 
    #-----------------------------
    # Same as spearman but more specific and has errors
    plot_delta_timescale  = False,                # Plot time spent misaligned as a function of misalignment angle. Will plot the 'peak_misangle' with time
    
    """
    # Plot delta angle. Looks at peak angle from 180
    """if plot_delta_timescale:
        relaxationtime_plot = []
        angles_plot = []
        err_plot_l  = []
        err_plot_u  = []
        other_plot  = []
        ID_plot     = []
        for ID_i in misalignment_tree.keys():
            # Find index of peak misalignment from where it relaxes to (-co or -counter)
            all_angles = np.array(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])
            index_peak = misalignment_tree['%s' %ID_i]['index_peak']
            angle_peak = misalignment_tree['%s' %ID_i]['angle_peak']
            
            angles_plot.append(angle_peak)
            #angles_plot.append(misalignment_tree['%s' %ID_i][use_angle][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_peak])
            err_plot_l.append(misalignment_tree['%s' %ID_i]['%s_err' %use_angle][index_peak][0] - misalignment_tree['%s' %ID_i][use_angle][index_peak])
            err_plot_u.append(misalignment_tree['%s' %ID_i]['%s_err' %use_angle][index_peak][1] - misalignment_tree['%s' %ID_i][use_angle][index_peak])
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            other_plot.append(np.average(np.array(misalignment_tree['%s' %ID_i][plot_delta_color][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']]), weights=np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']-1:misalignment_tree['%s' %ID_i]['index_r']-1]) - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']])))
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            
        
        #-------------
        # Stats
        res = stats.spearmanr(angles_plot, relaxationtime_plot)
        print('\n--------------------------------------')
        print('DELTA TIMESCALE SPEARMAN:')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Normalise colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=20, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
        
        axs.errorbar(relaxationtime_plot, angles_plot, yerr=[np.abs(err_plot_l), err_plot_u], ls='none', fmt='o', capsize=3, elinewidth=0.7, markeredgewidth=1, color='k', ecolor='grey', ms=1, alpha=0.5)
        im1 = axs.scatter(relaxationtime_plot, angles_plot, c=other_plot, s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3)
        
        #-------------
        # Colorbar
        cax = plt.axes([0.98, 0.20, 0.03, 0.71])
        plt.colorbar(mapper, cax=cax, label='time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$', extend='max')
        
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_yticks(np.arange(0, 181, 30))
        axs.set_ylim(0, 180)
        axs.set_xlim(0, set_bin_limit_trelax)
        axs.set_xlabel('Relaxation time (Gyr)')
        axs.set_ylabel('Peak misalignment angle')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        axs.set_title(r'$\rho$ = %s, p-value = %s' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
        
        #-----------
        ### Annotations
        if set_add_GalaxyIDs:
            for ID_plot_i, x_i, y_i in zip(ID_plot, angles_plot, relaxationtime_plot):
                axs.text(x_i+5, y_i+0.1, '%s' %ID_plot_i, fontsize=7)
                
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/delta_misangle_t_relax/trelax_delta_misangle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/delta_misangle_t_relax/trelax_delta_misangle_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    """
    
    """ 
    #-----------------------------
    # Plot scatter and extract spearman from it
    plot_spearman         = False,                     # Available variables: halomass, stelmass, gasmass, sfmass, nsfmass, dmmass,   sfr, ssfr,   stars_Z, gas_Z, inflow_Z, outflow_Z
                                                      #                      kappa_stars, kappa_gas, kappa_sf,   ellip, triax,   rad, radproj, rad_sf
                                                      #                      inflow_rate, outflow_rate, stelmassloss_rate,   s_inflow_rate, s_outflow_rate,   inflow_cum, outflow_cum, 
                                                      #                      angle_peak,   number_of_mergers,  relaxation_time
      plot_spearman_x     = 'stelmass',        # [ array name ]
      plot_spearman_y     = 'sfmass',            # [ array name ]
      plot_spearman_c     = 'kappa_stars',               # [ None / array name]
        inflow_skip       = 0.5,                      # [ Gyr ] time to ignore inflow, after which to sample. Relaxations below this time are automatically filetred out
    #-----------------------------
    # Plots trelax - tdyn, which should be proportional
    plot_trelax_tdyn      = False,
    # Plots log(ellip) - log(trelax/tdyn), which should have a (rough) gradient of -1 
    plot_ellip_trelaxtdyn = False,
    """
    # Plot spearman
    """if plot_spearman:
        
        # Collect datapoints
        x_array = []
        y_array = []
        c_array = []
        for ID_i in misalignment_tree.keys():
            
            #--------------
            # IF we're using inflows, skip galaxies that dont have enough misalignment snipshots
            if not set(['inflow_rate', 'inflow_cum', 's_inflow_rate', 's_inflow_cum', 'outflow_rate']).isdisjoint([plot_spearman_x, plot_spearman_y, plot_spearman_c]):
                if misalignment_tree['%s' %ID_i]['relaxation_time'] < inflow_skip:
                    continue
            
            #--------------
            # Isolate inflow_rate, inflow_cum, s_inflow_rate, s_inflow_cum, outflow_rate, peak_angle, number_of_mergers
            if (plot_spearman_x not in misalignment_tree['%s' %ID_i].keys()) or (plot_spearman_x == 'inflow_rate') or (plot_spearman_x == 'outflow_rate') or (plot_spearman_x == 's_inflow_rate') or (plot_spearman_x == 's_outflow_rate'):
            
                if plot_spearman_x == 'inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_x == 'outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_x == 'inflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.sum(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_x == 'outflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.sum(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_x == 's_inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_x == 's_outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_x == 'number_of_mergers':
                    # count mergers during relaxation
                    check = 0
                    for merger_i in misalignment_tree['%s' %ID_i]['merger_ratio_stars'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]:
                        if len(merger_i) > 0:
                            if max(merger_i) > 0.1:
                                check += 1
                
                    x_array.append(check)
            else:
                if plot_spearman_x in ['angle_peak', 'relaxation_time', 'relaxation_type', 'relaxation_morph', 'misalignment_morph']:
                    x_array.append(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_x]) 
                else:
                    x_array.append(np.mean(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_x][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                
            if (plot_spearman_y not in misalignment_tree['%s' %ID_i].keys()) or (plot_spearman_y == 'inflow_rate') or (plot_spearman_y == 'outflow_rate') or (plot_spearman_y == 's_inflow_rate') or (plot_spearman_x == 's_outflow_rate'):
            
                if plot_spearman_y == 'inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_y == 'outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_y == 'inflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.sum(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_y == 'outflow_cum':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.sum(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_y == 's_inflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                elif plot_spearman_y == 's_outflow_rate':
                    # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                    index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                    index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                
                elif plot_spearman_y == 'number_of_mergers':
                    # count mergers during relaxation
                    check = 0
                    for merger_i in misalignment_tree['%s' %ID_i]['merger_ratio_stars'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]:
                        if len(merger_i) > 0:
                            if max(merger_i) > 0.1:
                                check += 1
                
                    y_array.append(check)
            else:
                if plot_spearman_y in ['angle_peak', 'relaxation_time', 'relaxation_type', 'relaxation_morph', 'misalignment_morph']:
                    y_array.append(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_y]) 
                else:
                    y_array.append(np.mean(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_y][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                    
            if plot_spearman_c != None:
                if (plot_spearman_c not in misalignment_tree['%s' %ID_i].keys()) or (plot_spearman_c == 'inflow_rate') or (plot_spearman_c == 'outflow_rate') or (plot_spearman_c == 's_inflow_rate') or (plot_spearman_x == 's_outflow_rate'):
                
                    if plot_spearman_c == 'inflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    elif plot_spearman_c == 'outflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    
                    elif plot_spearman_c == 'inflow_cum':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.sum(misalignment_tree['%s' %ID_i]['inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    elif plot_spearman_c == 'outflow_cum':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.sum(misalignment_tree['%s' %ID_i]['outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    
                    elif plot_spearman_c == 's_inflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_inflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    elif plot_spearman_c == 's_outflow_rate':
                        # Find indexes within misalignment for which AT LEAST inflow_skip time has passed:
                        index_start = np.where(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][0] <= -inflow_skip)[0][0]
                        index_stop  = len(misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])

                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['s_outflow_rate'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1][index_start:index_stop]))
                    
                    elif plot_spearman_c == 'number_of_mergers':
                        # count mergers during relaxation
                        check = 0
                        for merger_i in misalignment_tree['%s' %ID_i]['merger_ratio_stars'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]:
                            if len(merger_i) > 0:
                                if max(merger_i) > 0.1:
                                    check += 1
                    
                        c_array.append(check)  
                else:
                    if plot_spearman_c in ['angle_peak', 'relaxation_time', 'relaxation_type', 'relaxation_morph', 'misalignment_morph']:
                        c_array.append(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_c]) 
                    else:
                        c_array.append(np.mean(misalignment_tree['%s' %ID_i]['%s' %plot_spearman_c][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1]))
                
        
        #-------------
        # Making logs of values that need them
        if plot_spearman_x in ['halomass', 'stelmass', 'gasmass', 'sfmass', 'nsfmass', 'dmmass', 'sfr', 'ssfr', 'inflow_rate', 'outflow_rate', 's_inflow_rate', 's_outflow_cum', 's_inflow_rate', 's_outflow_rate', 'stelmassloss_rate']:
            x_array = np.log10(np.array(x_array))
        if plot_spearman_y in ['halomass', 'stelmass', 'gasmass', 'sfmass', 'nsfmass', 'dmmass', 'sfr', 'ssfr', 'inflow_rate', 'outflow_rate', 's_inflow_rate', 's_outflow_cum', 's_inflow_rate', 's_outflow_rate', 'stelmassloss_rate']:
            y_array = np.log10(np.array(y_array))
        if plot_spearman_c != None:
            if plot_spearman_c in ['halomass', 'stelmass', 'gasmass', 'sfmass', 'nsfmass', 'dmmass', 'sfr', 'ssfr', 'inflow_rate', 'outflow_rate', 's_inflow_rate', 's_outflow_cum', 's_inflow_rate', 's_outflow_rate', 'stelmassloss_rate']:
                c_array = np.log10(np.array(c_array))
        
        
        #------------
        # Stats
        res = stats.spearmanr(x_array, y_array)
        print('\n--------------------------------------')
        print('SPEARMAN:  %s  -  %s' %(plot_spearman_x, plot_spearman_y))
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------')
    
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
         #-----------
        # Colormap
        if plot_spearman_c != None:
            
            if plot_spearman_c == 'halomass':
                vmin = 11
                vmax = 15
                label = 'log$_{10}$ M$_{\mathrm{halo}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'stelmass':
                vmin = 8
                vmax = 12.5
                label = 'log$_{10}$ M$_{\mathrm{*}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'gasmass':
                vmin = 7.5
                vmax = 11
                label = 'log$_{10}$ M$_{\mathrm{gas}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'sfmass':
                vmin = 7.5
                vmax = 11
                label = 'log$_{10}$ M$_{\mathrm{SF}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'nsfmass':
                vmin = 7
                vmax = 11
                label = 'log$_{10}$ M$_{\mathrm{NSF}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'dmmass':
                vmin = 8
                vmax = 13.5
                label = 'log$_{10}$ M$_{\mathrm{DM}}$ (M$_{\odot}$)'
            elif plot_spearman_c == 'sfr':
                vmin = -5
                vmax = 2
                label = 'log$_{10}$ SFR $(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'ssfr':
                vmin = -12
                vmax = -8
                label = 'log$_{10}$ sSFR $(\mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'stars_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{*}}$'
            elif plot_spearman_c == 'gas_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{gas}}$'
            elif plot_spearman_c == 'inflow_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{inflow}}$'
            elif plot_spearman_c == 'outflow_Z':
                vmin = 0
                vmax = max(c_array)
                label = '$Z_{\mathrm{outflow}}$'
            elif plot_spearman_c == 'kappa_stars':
                vmin = 0.2
                vmax = 0.8
                label = '$\kappa_{\mathrm{co}}^{\mathrm{*}}$'
            elif plot_spearman_c == 'kappa_gas':
                vmin = 0
                vmax = 1
                label = '$\kappa_{\mathrm{co}}^{\mathrm{gas}}$'
            elif plot_spearman_c == 'kappa_sf':
                vmin = 0
                vmax = 1
                label = '$\kappa_{\mathrm{co}}^{\mathrm{SF}}$'
            elif plot_spearman_c == 'kappa_nsf':
                vmin = 0
                vmax = 1
                label = '$\kappa_{\mathrm{co}}^{\mathrm{NSF}}$'
            elif plot_spearman_c == 'ellip':
                vmin = 0
                vmax = 1
                label = '$\epsilon$'
            elif plot_spearman_c == 'triax':
                vmin = 0
                vmax = 1
                label = '$T$'
            elif plot_spearman_c == 'rad':
                vmin = 0
                vmax = 15
                label = '$r_{1/2}$ (pkpc)'
            elif plot_spearman_c == 'radproj':
                vmin = 0
                vmax = 15
                label = '$r_{1/2,\mathrm{z}}$ (pkpc)'
            elif plot_spearman_c == 'rad_sf':
                vmin = 0
                vmax = 15
                label = '$r_{1/2,\mathrm{SF}}$ (pkpc)'
            elif plot_spearman_c == 'inflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'outflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged gas outflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'stelmassloss_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged stellar mass-loss\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$'
            elif plot_spearman_c == 's_inflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged specific gas\ninflow $\mathrm{yr}^{-1})$'
            elif plot_spearman_c == 's_outflow_rate':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ time-averaged specific gas\noutflow $\mathrm{yr}^{-1})$'
            elif plot_spearman_c == 'inflow_cum':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ cumulative gas inflow\n$(\mathrm{M}_{\odot})$'
            elif plot_spearman_c == 'outflow_cum':
                vmin = min(c_array)
                vmax = max(c_array)
                label = 'log$_{10}$ cumulative gas outflow\n$(\mathrm{M}_{\odot})$'
            elif plot_spearman_c == 'angle_peak':
                vmin = 0
                vmax = 180
                label = 'Peak misalignment angle'
            elif plot_spearman_c == 'number_of_mergers':
                vmin = 0
                vmax = max(c_array)
                label = 'Number of mergers'
            elif plot_spearman_c == 'relaxation_time':
                vmin = 0
                vmax = 4
                label = 'Relaxation time (Gyr)'
            
            #-------------
            # Normalise colormap
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
            
            #-------------
            # Colorbar
            #cax = plt.axes([0.98, 0.20, 0.03, 0.71])
            plt.colorbar(mapper, ax=axs, label=label, extend='max')
    
         #-----------
        # Plot scatter
        if plot_spearman_c != None:
            axs.scatter(x_array, y_array, c=c_array, s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.5)
        else:
            axs.scatter(x_array, y_array, c=c_array, s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.5)
    
        
        #-----------
        ### General formatting
        if plot_spearman_x != None and plot_spearman_y != None:
            
            # x-axis labels
            if plot_spearman_x == 'halomass':
                axs.set_xticks(np.arange(11, 15.1, 1))
                axs.set_xlim(11, 15)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{halo}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'stelmass':
                axs.set_xticks(np.arange(9, 12.5, 1))
                axs.set_xlim(9, 12.5)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{*}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'gasmass':
                axs.set_xticks(np.arange(7, 11.1, 1))
                axs.set_xlim(7.5, 11)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{gas}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'sfmass':
                axs.set_xticks(np.arange(7.5, 11.1, 1))
                axs.set_xlim(7.5, 12)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{SF}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'nsfmass':
                axs.set_xticks(np.arange(7, 11.1, 1))
                axs.set_xlim(7, 11)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{NSF}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'dmmass':
                axs.set_xticks(np.arange(8, 13.5, 1))
                axs.set_xlim(8, 13.5)
                axs.set_xlabel('log$_{10}$ M$_{\mathrm{DM}}$ (M$_{\odot}$)')
            elif plot_spearman_x == 'sfr':
                axs.set_xticks(np.arange(-2, 3.1, 1))
                axs.set_xlim(-2, 3)
                axs.set_xlabel('log$_{10}$ SFR $(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'ssfr':
                axs.set_xticks(np.arange(-12, -7.9, 1))
                axs.set_xlim(-12, -8)
                axs.set_xlabel('log$_{10}$ sSFR $(\mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'stars_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{*}}$')
            elif plot_spearman_x == 'gas_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{gas}}$')
            elif plot_spearman_x == 'inflow_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{inflow}}$')
            elif plot_spearman_x == 'outflow_Z':
                #axs.set_xticks(np.arange(0, max(x_array), 0.02))
                #axs.set_xlim(0, max(x_array))
                axs.set_xlabel('$Z_{\mathrm{outflow}}$')
            elif plot_spearman_x == 'kappa_stars':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{*}}$')
            elif plot_spearman_x == 'kappa_gas':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{gas}}$')
            elif plot_spearman_x == 'kappa_sf':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
            elif plot_spearman_x == 'kappa_stars':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\kappa_{\mathrm{co}}^{\mathrm{NSF}}$')
            elif plot_spearman_x == 'ellip':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$\epsilon$')
            elif plot_spearman_x == 'triax':
                axs.set_xticks(np.arange(0, 1.1, 0.2))
                axs.set_xlim(0, 1)
                axs.set_xlabel('$T$')
            elif plot_spearman_x == 'rad':
                axs.set_xticks(np.arange(0, 15.1, 3))
                axs.set_xlim(0, 15)
                axs.set_xlabel('$r_{1/2}$ (pkpc)')
            elif plot_spearman_x == 'radproj':
                axs.set_xticks(np.arange(0, 15.1, 3))
                axs.set_xlim(0, 15)
                axs.set_xlabel('$r_{1/2,\mathrm{z}}$ (pkpc)')
            elif plot_spearman_x == 'rad_sf':
                axs.set_xticks(np.arange(0, 15.1, 3))
                axs.set_xlim(0, 15)
                axs.set_xlabel('$r_{1/2,\mathrm{SF}}$ (pkpc)')
            elif plot_spearman_x == 'inflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'outflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged gas outflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'stelmassloss_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged stellar mass-loss\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_x == 's_inflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged specific\ngas inflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_x == 's_outflow_rate':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ time-averaged specific\ngas outflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_x == 'inflow_cum':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ cumulative gas inflow\n$(\mathrm{M}_{\odot})$')
            elif plot_spearman_x == 'outflow_cum':
                #axs.set_xticks(np.arange(0, 15.1, 3))
                #axs.set_xlim(0, 15)
                axs.set_xlabel('log$_{10}$ cumulative gas outflow\n$(\mathrm{M}_{\odot})$')
            elif plot_spearman_x == 'angle_peak':
                axs.set_xticks(np.arange(0, 181, 30))
                axs.set_xlim(0, 180)
                axs.set_xlabel('Peak misalignment angle')
            elif plot_spearman_x == 'number_of_mergers':
                #axs.set_xticks(np.arange(0, max(c_array)+0.1, 1))
                #axs.set_xlim(0, max(c_array))
                axs.set_xlabel('Number of mergers')
            elif plot_spearman_x == 'relaxation_time':
                axs.set_xticks(np.arange(0, 4.1, 0.5))
                axs.set_xlim(0, 4)
                axs.set_xlabel('Relaxation time (Gyr)')
                
            # y-axis labels
            if plot_spearman_y == 'halomass':
                axs.set_yticks(np.arange(11, 15.1, 1))
                axs.set_ylim(11, 15)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{halo}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'stelmass':
                axs.set_yticks(np.arange(9, 12.5, 1))
                axs.set_ylim(9, 12.5)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{*}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'gasmass':
                axs.set_yticks(np.arange(7, 11.1, 1))
                axs.set_ylim(7.5, 11)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{gas}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'sfmass':
                axs.set_yticks(np.arange(7, 11.1, 1))
                axs.set_ylim(7.5, 11)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{SF}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'nsfmass':
                axs.set_yticks(np.arange(7.5, 11.1, 1))
                axs.set_ylim(7.5, 11)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{NSF}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'dmmass':
                axs.set_yticks(np.arange(8, 13.5, 1))
                axs.set_ylim(8, 13.5)
                axs.set_ylabel('log$_{10}$ M$_{\mathrm{DM}}$ (M$_{\odot}$)')
            elif plot_spearman_y == 'sfr':
                axs.set_yticks(np.arange(-2, 3.1, 1))
                axs.set_ylim(-2, 3)
                axs.set_ylabel('log$_{10}$ SFR $(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'ssfr':
                axs.set_yticks(np.arange(-12, -7.9, 1))
                axs.set_ylim(-12, -8)
                axs.set_ylabel('log$_{10}$ sSFR $(\mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'stars_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{*}}$')
            elif plot_spearman_y == 'gas_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{gas}}$')
            elif plot_spearman_y == 'inflow_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{inflow}}$')
            elif plot_spearman_y == 'outflow_Z':
                #axs.set_yticks(np.arange(0, max(x_array), 0.02))
                #axs.set_ylim(0, max(x_array))
                axs.set_ylabel('$Z_{\mathrm{outflow}}$')
            elif plot_spearman_y == 'kappa_stars':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{*}}$')
            elif plot_spearman_y == 'kappa_gas':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{gas}}$')
            elif plot_spearman_y == 'kappa_sf':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{SF}}$')
            elif plot_spearman_y == 'kappa_stars':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\kappa_{\mathrm{co}}^{\mathrm{NSF}}$')
            elif plot_spearman_y == 'ellip':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$\epsilon$')
            elif plot_spearman_y == 'triax':
                axs.set_yticks(np.arange(0, 1.1, 0.2))
                axs.set_ylim(0, 1)
                axs.set_ylabel('$T$')
            elif plot_spearman_y == 'rad':
                axs.set_yticks(np.arange(0, 15.1, 3))
                axs.set_ylim(0, 15)
                axs.set_ylabel('$r_{1/2}$ (pkpc)')
            elif plot_spearman_y == 'radproj':
                axs.set_yticks(np.arange(0, 15.1, 3))
                axs.set_ylim(0, 15)
                axs.set_ylabel('$r_{1/2,\mathrm{z}}$ (pkpc)')
            elif plot_spearman_y == 'rad_sf':
                axs.set_yticks(np.arange(0, 15.1, 3))
                axs.set_ylim(0, 15)
                axs.set_ylabel('$r_{1/2,\mathrm{SF}}$ (pkpc)')
            elif plot_spearman_y == 'inflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged gas inflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'outflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged gas outflow\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'stelmassloss_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged stellar mass-loss\n$(\mathrm{M}_{\odot} \mathrm{yr}^{-1})$')
            elif plot_spearman_y == 's_inflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged specific gas\ninflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_y == 's_outflow_rate':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ time-averaged specific gas\noutflow $\mathrm{yr}^{-1})$')
            elif plot_spearman_y == 'inflow_cum':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ cumulative gas inflow\n$(\mathrm{M}_{\odot}$)')
            elif plot_spearman_y == 'outflow_cum':
                #axs.set_yticks(np.arange(0, 15.1, 3))
                #axs.set_ylim(0, 15)
                axs.set_ylabel('log$_{10}$ cumulative gas outflow\n$(\mathrm{M}_{\odot}$()')
            elif plot_spearman_y == 'angle_peak':
                axs.set_yticks(np.arange(0, 181, 30))
                axs.set_ylim(0, 180)
                axs.set_ylabel('Peak misalignment angle')
            elif plot_spearman_y == 'number_of_mergers':
                #axs.set_yticks(np.arange(0, max(c_array)+0.1, 1))
                #axs.set_ylim(0, max(c_array))
                axs.set_ylabel('Number of mergers')
            elif plot_spearman_y == 'relaxation_time':
                axs.set_yticks(np.arange(0, 4.1, 0.5))
                axs.set_ylim(0, 4)
                axs.set_ylabel('Relaxation time (Gyr)')
            
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        #-----------
        ### other
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        plt.tight_layout()
        
        #-----------
        #### Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/spearman_plots/%s%s-%s_%s_%s.%s" %(fig_dir, 'L100_', plot_spearman_x, plot_spearman_y, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/spearman_plots/%s%s-%sdelta_misangle_%s_%s.%s" %(fig_dir, 'L100_', plot_spearman_x, plot_spearman_y, len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    """
    # Plots trelax - tdyn, which should be proportional
    """if plot_trelax_tdyn:  
        relaxationtime_plot = []
        dyntime_plot        = []
        ID_plot             = []
        for ID_i in misalignment_tree.keys():
            # Actual relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            # Average the dynamical time and ellip
            dyntime_plot.append(np.mean(np.array(misalignment_tree['%s' %ID_i]['relaxation_tdyn'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            

        #-------------
        # Stats
        res = stats.spearmanr(relaxationtime_plot, dyntime_plot)
        print('\n--------------------------------------')
        print('SPEARMAN:  trelax/tdyn - ellip')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------') 
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        axs.scatter(relaxationtime_plot, dyntime_plot, s=10, zorder=99, c='k', edgecolors='k', linewidths=0.3, alpha=0.7)
        
        #------------
        # Annotation (gradient line of -1)
        print('\n   add bestfit?')
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(0, 5)
        axs.set_xlabel('Relaxation time (Gyr)')
        axs.set_yticks(np.arange(0, 2, 0.2))
        axs.set_ylim(0, 2)
        axs.set_ylabel('Dynamical time (Gyr)')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        #-----------
        ### Annotations
        #if set_add_GalaxyIDs:
        #    for ID_plot_i, x_i, y_i in zip(ID_plot, angles_plot, relaxationtime_plot):
        #        axs.text(x_i+5, y_i+0.1, '%s' %ID_plot_i, fontsize=7)
        
        #-----------
        ### title
        if plot_annotate:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        #### Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/tdyn_plots/%stdyn_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/tdyn_plots/%stdyn_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    """
    # Plots log(ellip) - log(trelax/tdyn), which should have a (rough) gradient of -1 
    """if plot_ellip_trelaxtdyn:
        relaxationtime_plot = []
        dyntime_plot        = []
        ellip_plot          = []
        ID_plot             = []
        for ID_i in misalignment_tree.keys():
            # Actual relaxation time
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            # Average the dynamical time and ellip
            dyntime_plot.append(np.mean(np.array(misalignment_tree['%s' %ID_i]['relaxation_tdyn'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
            ellip_plot.append(np.mean(np.array(misalignment_tree['%s' %ID_i]['ellip'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']+1])))
            
            ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
            
        #-------------
        # Log values
        y_array = np.log10(np.divide(np.array(relaxationtime_plot), np.array(dyntime_plot)))
        ellip_plot = np.log10(np.array(ellip_plot))

        #-------------
        # Stats
        res = stats.spearmanr(ellip_plot, y_array)
        print('\n--------------------------------------')
        print('SPEARMAN:  trelax/tdyn - ellip')
        print('   ρ:       %.2f' %res.correlation)
        print('   p-value: %s' %res.pvalue)
        print('--------------------------------------') 
        
        
        #-------------
        # Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        axs.scatter(ellip_plot, y_array, s=10, zorder=99, c='k', edgecolors='k', linewidths=0.3, alpha=0.7)
        
        #------------
        # Annotation (gradient line of -1)
        print('\n   CHECK ANNOTATION')
        axs.plot([-1, 0], [0, -1], c='grey', alpha=0.5, ls='-')
        axs.text(0.5, -0.5, 'gradient = -1', rotation=45, c='grey')
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(-1, 0)
        axs.set_xlabel('log$_{10}$ $\epsilon$')
        #axs.set_yticks(np.arange(0, 181, 30))
        #axs.set_ylim(0, 180)
        axs.set_ylabel('log$_{10}$ $t_{\mathrm{relax}}/t_{\mathrm{dyn}}$')
        axs.minorticks_on()
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
        #-----------
        ### Annotations
        #if set_add_GalaxyIDs:
        #    for ID_plot_i, x_i, y_i in zip(ID_plot, angles_plot, relaxationtime_plot):
        #        axs.text(x_i+5, y_i+0.1, '%s' %ID_plot_i, fontsize=7)
        
        #-----------
        ### other
        plt.tight_layout()
        
        #-----------
        # Savefig
        metadata_plot = {'Title': '%s, %s\nThreshold: %s\nMin delta:%s\nLatency time: %s\nMin ratio: %s\nUSE MERGERS: %s\nMax closest merger: %s-%s\nTime extra: %s\nTime no misangle: %s\nStelmass: %s/%s\n kappa*: %s/%s\n outflow: %s/%s' %(abs_or_proj, use_angle, misangle_threshold, min_delta_angle, latency_time, min_stellar_ratio, use_merger_criteria, max_merger_pre, max_merger_post, time_extra, time_no_misangle, min_stelmass, max_stelmass, min_kappa_stars, max_kappa_stars, min_inflow, max_inflow),
                         'Author': 'Min particles: %s\nMax CoM: %s\nMin inc: %s\nPlot ratio limit: %s\n\nrho: %.2f\np-value %.2e\n\n# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(min_particles, max_com, min_inclination, set_plot_merger_limit, res.correlation, res.pvalue, len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                         
        if savefig:
            if savefig_txt == 'manual':
                savefig_txt = input('\n  -> Enter savefig_txt:   ')
            plt.savefig("%s/tdyn_plots/%sellip_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/tdyn_plots/%sellip_trelax_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    """
    
    
    
        

#=============================
#_create_galaxy_tree()  
#_analyse_tree()
#=============================


    