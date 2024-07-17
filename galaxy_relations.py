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
import eagleSqlTools as sql
from graphformat import set_rc_params
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================


#--------------------------------
""" 
Purpose
-------
Will find useful particle data when given stars, gas

Calling function
----------------
Subhalo_Analysis(sample_input['mySims'], GroupNum, SubGroupNum, GalaxyID, SnapNum, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, 
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

If find_uncertainties = False, will append NaNs to uncertainties... so can
        safely plot.

Input Parameters
----------------

galaxy. values all from above function

viewing_axis:   'x', 'y', 'z'
    Defaults to 'z', speeds up process to only find
    uncertainties in this viewing axis
aperture_rad:    value [pkpc]
    Will trim data to this maximum value for quicker processing.
    Usually 30.
kappa_rad:   False or value [pkpc]
    Will calculate kappa for this radius from centre
    of galaxy. Usually 30
trim_hmr:    array [multiples of rad]
    Will trim the output data to this radius. This is
    used for render and 2dhisto
align_rad:   False or value [pkpc]
    Will orientate the galaxy based on the stellar 
    spin vector within this radius. Usually 30
orientate_to_axis:  'x', 'y', 'z'
    When align_rad_in == value, will orientate galaxy
    to this axis. 
viewing_angle:
    Will rotate output particle data by this angle
        
angle_selection:
    Will speed up process to find only specific angles' uncertainties. 
    Automated process.   
        ['stars_gas',            # stars_gas     stars_gas_sf    stars_gas_nsf
         'stars_gas_sf',         # stars_dm      gas_dm        gas_sf_dm       gas_nsf_dm
         'stars_gas_nsf',        # gas_sf_gas_nsf
         'gas_sf_gas_nsf',
         'stars_dm',
         'gas_sf_dm']
spin_rad:    array [pkpc] 
    When given a list of values, for example:
    galaxy.halfmass_rad*np.arange(0.5, 10.5, 0.5)
    will calculate spin values within these values
spin_hmr:    array .
    When given a list of values, for example:
    np.arange(0.5, 10.5, 0.5)
    will calculate spin values within these values
find_uncertainties:     Boolean
    If True, will create rand_spins to find estimate of
    uncertainty for 3D angles
        
com_min_distance: [pkpc]
    Will initiate flag if this not met within spin_rad
    (if c.o.m > 2.0, in 2HMR)
    Will flag but continue to extract all values 
min_particles:
    Minimum gas_sf (and gas_nsf, stars) particle within spin_rad.
    Will flag but continue to extract all analytical values
        uncertainties not calculated if not met
min_inclination:
    Minimum and maximum spin inclination for a given rad.
    Will flag but continue to extract all values 
        
Output Parameters
-----------------

.general:       dictionary
    'GroupNum', 'SubGroupNum', 'GalaxyID', 'SnapNum', 'halo_mass', 
        'stelmass', 'gasmass', 'gasmass_sf', 'gasmass_nsf', 'dmmass', 
        'ap_sfr', 'bh_id', 'bh_mass', 'bh_mdot', 'bh_edd', 'halfmass_rad', 
        'halfmass_rad_proj', 'viewing_axis',
        'kappa_stars' - 30 kpc
        'kappa_gas'   - 2.0 hmr
        'kappa_gas_sf'
        'kappa_gas_nsf'
        'ellip', 'triax', 'kappa_stars', 'disp_ani', 'disc_to_total', 'rot_to_disp_ratio'

.flags:     dictionary
    Has list of arrays that will be != if flagged. Contains hmr at failure, or 30pkpc
        ['total_particles']
            will flag if there are missing particles within aperture_rad
            ['stars']       - [hmr]
            ['gas']         - [hmr]
            ['gas_sf']      - [hmr]
            ['gas_nsf']     - [hmr]
            ['dm']          - [hmr]
            ['bh']          - [hmr]
        ['min_particles']
            will flag if min. particles not met within spin_rad (will find spin if particles exist, but no uncertainties)
            ['stars']       - [hmr]
            ['gas']         - [hmr]
            ['gas_sf']      - [hmr]
            ['gas_nsf']     - [hmr]
            ['dm']          - [hmr]
        ['min_inclination']
            will flag if inclination angle not met within spin_rad... all spins and uncertainties still calculated
            ['stars']       - [hmr]
            ['gas']         - [hmr]
            ['gas_sf']      - [hmr]
            ['gas_nsf']     - [hmr]
            ['dm']          - [hmr]
        ['com_min_distance']
            will flag if com distance not met within spin_rad... all spins and uncertainties still calculated
            ['stars_gas']   - [hmr]
            ['stars_gas_sf']- [hmr]
            ... for all angle_selection ...
              
.data:    dictionary
    Has aligned/rotated values for 'stars', 'gas', 'gas_sf', 'gas_nsf':
        [hmr]                       - multiples of hmr, ei. '1.0' that data was trimmed to
            ['Coordinates']         - [pkpc]
            ['Velocity']            - [pkm/s]
            ['Mass']                - [Msun]
            ['StarFormationRate']   - [Msun/s] (i think)
            ['GroupNumber']         - int array 
            ['SubGroupNumber']      - int array
        
    if trim_hmr_in has a value, coordinates lying outside
    these values will be trimmed in final output, but still
    used in calculations.

.l:   dictionary
    Specific l, in units of [pkpc/kms-1]. 109 M have roughly log(l) of 1.5, 1010.5 have roughly log(l) of 2.5
        ['rad']     - [pkpc]
        ['hmr']     - multiples of halfmass_rad
        ['stars']   - [unit vector]
        ['gas']     - [unit vector]
        ['gas_sf']  - [unit vector]
        ['gas_nsf'] - [unit vector]
        ['dm']      - [unit vector]
.spins:   dictionary
    Has aligned/rotated spin vectors within spin_rad_in's:
        ['rad']     - [pkpc]
        ['hmr']     - multiples of halfmass_rad
        ['stars']   - [unit vector]
        ['gas']     - [unit vector]
        ['gas_sf']  - [unit vector]
        ['gas_nsf'] - [unit vector]
        ['dm']      - [unit vector]
.counts:   dictionary
    Has aligned/rotated particle count and mass within spin_rad_in's:
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['stars']        - [count]
        ['gas']          - [count]
        ['gas_sf']       - [count]
        ['gas_nsf']      - [count]
        ['dm']           - count at 30pkpc
.masses:   dictionary
    Has aligned/rotated particle count and mass within spin_rad_in's:
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['stars']        - [Msun]
        ['gas']          - [Msun]
        ['gas_sf']       - [Msun]
        ['gas_nsf']      - [Msun]
        ['dm']           - Msun at 30pkpc
.tot_mass:      dictionary
    Has total mass within hmr:
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['mass']         - [Msun]
        ['mass_disc']    - [Msun]   mass found within sf hmr multiples
.Z:   dictionary        
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['stars']        - mass-weighted metallicity
        ['gas']          - mass-weighted metallicity
        ['gas_sf']       - mass-weighted metallicity
        ['gas_nsf']      - mass-weighted metallicity
.sfr:       dictionary
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['gas_sf']       - total SFR
.coms:     dictionary
    Has all centres of mass and distances within a spin_rad_in:
        ['rad']          - [pkpc]
        ['hmr']     - multiples of halfmass_rad
        ['stars']          - [x, y, z] [pkpc] distance in 3D
        ['gas']            - [x, y, z] [pkpc]
        ['gas_sf']         - [x, y, z] [pkpc]
        ['gas_nsf']        - [x, y, z] [pkpc]
        ['dm']             - x, y, z [pkpc]  at 30pkpc
        ['adjust']         - [x, y, z] [pkpc] the value added on to each of the above from centering to stellar COM from 30pkpc COM
.mis_angles:     dictionary
    Has aligned/rotated misalignment angles between stars 
    and X within spin_rad_in's. Errors given by iterations,
    which defults to 500.
        ['rad']            - [pkpc]
        ['hmr']            - multiples of halfmass_rad
        ['stars_gas_angle']             - [deg]                -
        ['stars_gas_sf_angle']          - [deg]                .
        ['stars_gas_nsf_angle']         - [deg]                .
        ['gas_sf_gas_nsf_angle']        - [deg]                .
        ['stars_dm_angle']              - [deg]                .
        ['gas_dm_angle']                - [deg]                .
        ['gas_sf_dm_angle']             - [deg]                .
        ['gas_nsf_dm_angle']            - [deg]                .
        ['stars_gas_angle_err']         - [lo, hi] [deg]       .
        ['stars_gas_sf_angle_err']      - [lo, hi] [deg]       .
        ['stars_gas_nsf_angle_err']     - [lo, hi] [deg]       (assuming it was passed into angle_selection and not flagged)
        ['gas_sf_gas_nsf_angle_err']    - [lo, hi] [deg]       .
        ['stars_dm_angle_err']          - [lo, hi] [deg]       . 
        ['gas_dm_angle_err']            - [lo, hi] [deg]       .
        ['gas_sf_dm_angle_err']         - [lo, hi] [deg]       .
        ['gas_nsf_dm_angle_err']        - [lo, hi] [deg]       ^
.mis_angles_proj                    dictionary
    Has projected misalignment angles. Errors given by iterations,
    which defults to 500.
        ['x']
        ['y']
        ['z']
            ['rad']            - [pkpc]
            ['hmr']            - multiples of halfmass_rad
            ['stars_gas_angle']             - [deg]                -
            ['stars_gas_sf_angle']          - [deg]                .
            ['stars_gas_nsf_angle']         - [deg]                .
            ['gas_sf_gas_nsf_angle']        - [deg]                .
            ['stars_dm_angle']              - [deg]                .
            ['gas_dm_angle']                - [deg]                .
            ['gas_sf_dm_angle']             - [deg]                .
            ['gas_nsf_dm_angle']            - [deg]                .
            ['stars_gas_angle_err']         - [lo, hi] [deg]       -
            ['stars_gas_sf_angle_err']      - [lo, hi] [deg]       .
            ['stars_gas_nsf_angle_err']     - [lo, hi] [deg]       (assuming it was passed into angle_selection and not flagged)
            ['gas_sf_gas_nsf_angle_err']    - [lo, hi] [deg]       .
            ['stars_dm_angle_err']          - [lo, hi] [deg]       . 
            ['gas_dm_angle_err']            - [lo, hi] [deg]       .
            ['gas_sf_dm_angle_err']         - [lo, hi] [deg]       .
            ['gas_nsf_dm_angle_err']        - [lo, hi] [deg]       ^
.gas_data:      dictionary
    Has particle data of hmr requested
        ['1.0_hmr']
        ['2.0_hmr']
            ['gas']
            ['gas_sf']
            ['gas_nsf']
                ['ParticleIDs']     - Particle IDs
                ['Mass']            - Particle masses
                ['Metallicity']     - Particle metallicities
.mass_flow:     dictionary
    Has inflow/outflow data if gas_data_old provided. 
    math.nan if not
        ['1.0_hmr']
        ['2.0_hmr']
            ['gas']                     - Inflow/outflow of all gas
            ['gas_sf']                  - Inflow/outflow of only SF gas
            ['gas_nsf']                 - Inflow/outflow of only NSF gas
                ['inflow']              - Accurate inflow                       [Msun]
                ['outflow']             - Accurate outflow                      [Msun]
                ['massloss']            - Whats left... does not take into account gas particles switching between modes    [Msun]
                ['inflow_Z']            - Inflow metallicity                    [Mass-wighted metallicity]
                ['outflow_Z']           - Outflow metallicity                   [Mass-wighted metallicity]
                ['insitu_Z']            - Whats left... metallicity             [Mass-wighted metallicity]
        
"""
# Plots singular graphs of relations by reading in existing csv file
# SAVED: /plots/other/
def _plot_relations(csv_sample = 'L100_27_all_sample_misalignment_9.5',     # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum
                       csv_output = '_RadProj_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_',
                       #--------------------------
                       use_hmr_values = 'aperture',            # [ 1.0 / 2.0 / 'aperture' ] value to extract general values from
                       # CTRL+F -> 'INPUT VALUES HERE' ... and change teh axis
                       #--------------------------
                       # Selection criteria
                       print_summary = True,
                         use_angle          = 'stars_gas_sf',         # Which angles to plot
                         use_hmr            = 1.0,                    # Which HMR to use for criteria
                         use_proj_angle     = False,                   # Whether to use projected or absolute angle 10**9
                           min_inc_angle    = 0,                     # min. degrees of either spin vector to z-axis, if use_proj_angle
                           min_particles    = 20,               # [ 20 ] number of particles
                           min_com          = 2.0,              # [ 2.0 ] pkpc
                           max_uncertainty  = 30,            # [ None / 30 / 45 ]                  Degrees
                         lower_mass_limit   = 10**9.5,            # Whether to plot only certain masses 10**15
                         upper_mass_limit   = 10**15,         
                         ETG_or_LTG         = 'both',           # Whether to plot only ETG/LTG/both
                         cluster_or_field   = 'both',           # Whether to plot only field/cluster/both
                         use_satellites     = True,             # Whether to include SubGroupNum =/ 0
                       #--------------------------
                       misangle_threshold = 30,             # what we classify as misaligned
                       #--------------------------
                       showfig       = True,
                       savefig       = True,
                         file_format = 'pdf',
                         savefig_txt = 'bh_mass_centrals',
                       #--------------------------
                       print_progress = False,
                       debug = False):
                        
                        
                        
    # Ensuring the sample and output originated together
    csv_output = csv_sample + csv_output 
    csv_output
    
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
    all_l               = dict_output['all_l']
    all_spins           = dict_output['all_spins']
    all_coms            = dict_output['all_coms']
    all_counts          = dict_output['all_counts']
    all_masses          = dict_output['all_masses']
    all_totmass         = dict_output['all_totmass']
    all_sfr             = dict_output['all_sfr']
    all_Z               = dict_output['all_Z']
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
    print('SAMPLE LOADED:\n  %s\n  SnapNum: %s\n  Redshift: %s\n  Min mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(output_input['mySims'][0][0], output_input['snapNum'], output_input['Redshift'], output_input['galaxy_mass_min'], output_input['galaxy_mass_max'], use_satellites))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
    print('\nPLOT CRITERIA:\n  Angle: %s\n  HMR: %s\n  Projected angle: %s\n  Min. inclination: %s\n  Min particles: %s\n  Min COM: %.1f pkpc\n  Min Mass: %.2E M*\n  Max limit: %.2E M*\n  ETG or LTG: %s\n  Cluster or field: %s\n  Use satellites:  %s' %(use_angle, use_hmr, use_proj_angle, min_inc_angle, min_particles, min_com, lower_mass_limit, upper_mass_limit, ETG_or_LTG, cluster_or_field, use_satellites))
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
    
    # Set projection angle criteria
    if not use_proj_angle:
        min_inc_angle = 0
    max_inc_angle = 180 - min_inc_angle
    if output_input['viewing_axis'] == 'x':
        viewing_vector = [1., 0, 0]
    elif output_input['viewing_axis'] == 'y':
        viewing_vector = [0, 1., 0]
    elif output_input['viewing_axis'] == 'z':
        viewing_vector = [0, 0, 1.]
    else:
        raise Exception('Cant read viewing_axis')
        
    #-----------------------------
    # Set definitions
    cluster_threshold     = 1e14
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
        
    # Setting cluster lower and upper boundaries based on inputs
    if cluster_or_field == 'both':
        lower_halo = 0
        upper_halo = 10**16
    elif cluster_or_field == 'cluster':
        lower_halo = cluster_threshold
        upper_halo = 10**16
    elif cluster_or_field == 'field':
        lower_halo = 0
        upper_halo = cluster_threshold
    
    # Setting satellite criteria
    if use_satellites:
        satellite_criteria = 99999999
    if not use_satellites:
        satellite_criteria = 0
    
    collect_ID = []
    #------------------------------
    def _plot_relations_plot(debug=False):
        # We have use_angle = 'stars_gas_sf', and use_particles = ['stars', 'gas_sf'] 
        
        #=================================
        # Collect values to plot
        value_x = []
        value_y = []
        value_c = []
        value_s = []
        
        array_ETG = []
        array_LTG = []
        array_ETG_mass = []
        array_LTG_mass = []
        
        # Find angle galaxy makes with viewing axis
        def _find_angle(vector1, vector2):
            return np.rad2deg(np.arccos(np.clip(np.dot(vector1/np.linalg.norm(vector1), vector2/np.linalg.norm(vector2)), -1.0, 1.0)))     # [deg]
        
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
        
        #--------------------------
        # Loop over all galaxies we have available, and analyse output of flags
        for GalaxyID in GalaxyID_List:
            
            #-----------------------------
            # Check if galaxy meets criteria
            
            # check if hmr exists 
            if (use_hmr in all_misangles['%s' %GalaxyID]['hmr']):
                # creating masks
                mask_counts = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_coms   = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_spins  = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                mask_angles = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(use_hmr))[0][0]
                
                # find particle counts
                if use_particles[0] == 'dm':
                    count_1 = all_counts['%s' %GalaxyID][use_particles[0]]
                else:
                    count_1 = all_counts['%s' %GalaxyID][use_particles[0]][mask_counts]
                if use_particles[1] == 'dm':
                    count_2 = all_counts['%s' %GalaxyID][use_particles[1]]
                else:
                    count_2 = all_counts['%s' %GalaxyID][use_particles[1]][mask_counts]
                
                # find inclination angle(s)
                inc_angle_1 = _find_angle(all_spins['%s' %GalaxyID][use_particles[0]][mask_spins], viewing_vector)
                inc_angle_2 = _find_angle(all_spins['%s' %GalaxyID][use_particles[1]][mask_spins], viewing_vector)
                
                # find CoMs = com_abs
                if use_angle != 'stars_dm':
                    com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]][mask_angles], 'abs')
                else:
                    com_abs  = _evaluate_com(all_coms['%s' %GalaxyID][use_particles[0]][mask_angles], all_coms['%s' %GalaxyID][use_particles[1]], 'abs')
                
                #--------------
                # Determine if this is a galaxy we want to plot and meets the remaining criteria (stellar mass, halo mass, kappa, uncertainty, satellite)
                if use_proj_angle:
                    max_error = max(np.abs((np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle][mask_angles]) - all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles])))
                else:
                    max_error = max(np.abs((np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle][mask_angles]) - all_misangles['%s' %GalaxyID]['%s_angle' %use_angle][mask_angles])))
                
                
                # applying selection criteria for min_inc_angle, min_com, min_particles
                if (count_1 >= min_particles) and (count_2 >= min_particles) and (com_abs <= min_com) and (inc_angle_1 >= min_inc_angle) and (inc_angle_1 <= max_inc_angle) and (inc_angle_2 >= min_inc_angle) and (inc_angle_2 <= max_inc_angle) and (max_error <= (999 if max_uncertainty == None else max_uncertainty)):
                    
                    # applying stelmass, environment, morphology, satellite criteria
                    if (all_general['%s' %GalaxyID]['stelmass'] >= lower_mass_limit) and (all_general['%s' %GalaxyID]['stelmass'] <= upper_mass_limit) and (all_general['%s' %GalaxyID]['halo_mass'] >= lower_halo) and (all_general['%s' %GalaxyID]['halo_mass'] <= upper_halo) and (all_general['%s' %GalaxyID]['kappa_stars'] >= lower_morph) and (all_general['%s' %GalaxyID]['kappa_stars'] <= upper_morph) and (all_general['%s' %GalaxyID]['SubGroupNum'] <= satellite_criteria):
                        
                        #-----------------
                        # COLLECT ID OF INTERESTED GALAXIES aaa
                        if all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle][mask_angles] > 170:
                            collect_ID.append(GalaxyID)
                        
                        #-----------------
                        # Extracting values
                        
                        use_hmr_values
                        
                        # Creating masks
                        if use_hmr_values == 'aperture':
                            # aperture
                            """
                            all_general['%s' %GalaxyID]['GroupNum', 'SubGroupNum', 'GalaxyID', 'SnapNum', 'halo_mass', 
                                                        'stelmass', 'gasmass', 'gasmass_sf', 'gasmass_nsf', 'dmmass', 
                                                        'ap_sfr', 'bh_id', 'bh_mass', 'bh_mdot', 'bh_edd', 'halfmass_rad', 
                                                        'halfmass_rad_proj', 'viewing_axis',
                                                        'kappa_stars' - 30 kpc
                                                        'kappa_gas'   - 2.0 hmr
                                                        'kappa_gas_sf'
                                                        'kappa_gas_nsf'
                                                        'ellip', 'triax', 'kappa_stars', 'disp_ani', 'disc_to_total', 'rot_to_disp_ratio']
                            """
                        
                            #========================================================
                            # INPUT VALUES HERE     value_x.append(np.log10(np.array(all_general['%s' %GalaxyID]['stelmass'])))
                            legend_label = 'in aperture\nz=0.1'
                            
                            xlabel = 'Stellarmass'
                            value_x.append(np.log10(np.array(all_general['%s' %GalaxyID]['stelmass'])))
                            
                            #ylabel = 'gassf / stelmass + gassf'
                            #value_y.append(3.154e+7*np.array(all_general['%s' %GalaxyID]['ap_sfr']))
                            #value_y.append(np.divide(3.154e+7*np.array(all_general['%s' %GalaxyID]['ap_sfr']), np.array(all_general['%s' %GalaxyID]['stelmass'])))
                            #value_y.append(np.divide(all_general['%s' %GalaxyID]['gasmass_sf'], all_general['%s' %GalaxyID]['gasmass_sf'] + all_general['%s' %GalaxyID]['stelmass']))
                            
                            ylabel = 'CentralBHmass'
                            value_y.append(np.log10(np.array(all_general['%s' %GalaxyID]['bh_mass'])))
                            
                            
                            clabel = 'kappa'
                            value_c.append(all_general['%s' %GalaxyID]['kappa_stars'])
                            
                            #value_s.append()
                            #========================================================
                        else:
                            
                            # value
                            value_mask_l       = np.where(np.array(all_l['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_spins   = np.where(np.array(all_spins['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_counts  = np.where(np.array(all_counts['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_masses  = np.where(np.array(all_masses['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_totmass = np.where(np.array(all_totmass['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_Z       = np.where(np.array(all_Z['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_sfr     = np.where(np.array(all_sfr['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_coms    = np.where(np.array(all_coms['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            value_mask_angles  = np.where(np.array(all_misangles['%s' %GalaxyID]['hmr']) == float(use_hmr_values))[0][0]
                            
                            #========================================================
                            # INPUT VALUES HERE     value_x.append(np.log10(np.array(all_masses['%s' %GalaxyID]['stars'][value_mask_masses])))
                            #value_x.append(np.log10(np.array(all_masses['%s' %GalaxyID]['stars'][value_mask_masses])))
                            
                            legend_label = 'in %.1f\nz=%.1f' %(float(use_hmr_values), output_input['Redshift'])
                            
                            xlabel = 'stellar mass'
                            value_x.append(np.log10(np.array(all_masses['%s' %GalaxyID]['stars'][value_mask_masses])))
                            
                            ylabel = 'gassf / stelmass + gassf'
                            #value_y.append(all_general['%s' %GalaxyID]['kappa_stars'])
                            value_y.append(np.divide(np.array(all_masses['%s' %GalaxyID]['gas_sf'][value_mask_masses]), np.array(all_masses['%s' %GalaxyID]['gas_sf'][value_mask_masses]) + np.array(all_masses['%s' %GalaxyID]['stars'][value_mask_masses])))
                            
                            clabel = 'kappa'
                            value_c.append(all_general['%s' %GalaxyID]['kappa_stars'])
                            
                            #value_s.append(math.nan)
                            #========================================================
                        
                        
                        if all_general['%s' %GalaxyID]['kappa_stars'] > 0.5:
                            array_LTG.append(all_general['%s' %GalaxyID]['halfmass_rad_sf'])
                        if all_general['%s' %GalaxyID]['kappa_stars'] < 0.3:
                            array_ETG.append(all_general['%s' %GalaxyID]['halfmass_rad_sf'])
                            
                        
                        
                        
        #============================================    
        # END OF ALL GALAXIES + PLOTTING
        print('Final sample length: ', len(value_x))
        #print(np.median(array_LTG))
        #print(np.median(array_ETG))
                
        # COLLECT ID -> CTRL+F 'COLLECT ID OF INTERESTED GALAXIES aaa'
        #print('\tcollect_ID of counter-rotatos:')
        #print(collect_ID)
        #print(len(collect_ID))
        
        # COLLECT VALUES OF INTEREST -> CTRL+F 'INPUT VALUES HERE'
        
        
        #------------------------
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        #-----------
        ### Creating graphs
        # no colourmap
        if len(value_c) == 0:
            axs.scatter(value_x, value_y, s=1, c='k', alpha=0.5)
            
            # mask LTGs
            #mask = np.where(np.array(value_y) < 0.4)[0]            
                        
            #hist_tot, _ = np.histogram(np.array(value_x), range=[9, 12], bins=20)
            #hist_ETG, _ = np.histogram(np.array(value_x)[mask], range=[9, 12], bins=20)
            
            #axs.plot(np.linspace(9, 12, 20)+0.1, np.divide(hist_ETG, hist_tot))
            
        # colourmap
        else:
            #-------------
            vmin = 0.2
            vmax = 0.6
            
            # Normalise colormap
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm)
        
            #-------------
            # Colorbar
            fig.colorbar(mapper, ax=axs, label=clabel)      #, extend='max'
            
            # Scatter
            axs.scatter(value_x, value_y, c=value_c, s=1, norm=norm, cmap='Spectral', zorder=99, linewidths=0.3, alpha=0.5)
            
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9, 11.5)
        #axs.set_xticks(np.arange(0, 181, step=30))
        axs.set_ylim(5, 10)
        #axs.set_yscale('log')
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
        #-----------
        # Annotations
        #axs.axvline(misangle_threshold, ls='--', lw=1, c='k')
        #axs.text()
        
        #-----------
        ### Legend   
        legend_elements = [Line2D([0], [0], marker=' ', color='w')]
        legend_labels = [legend_label]
        legend_colors = ['k']
        legend1 = axs.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        axs.add_artist(legend1)
        
        #-----------
        # other
        plt.tight_layout()
        
        #-----------
        # Savefig       
        
        metadata_plot = {'Title': 'redshift: %s\nsample size: %s' %(output_input['Redshift'], len(value_x))}
        
        if savefig:
            plt.savefig(     "%s/other/L%s_%s_%s_%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], xlabel, ylabel, ETG_or_LTG, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/other/L%s_%s_%s_%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], output_input['snapNum'], xlabel, ylabel, ETG_or_LTG, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
    #---------------------------------
    _plot_relations_plot()
    #---------------------------------
    




#===========================    
#_plot_relations()
#_plot_relations(ETG_or_LTG = 'ETG')
#_plot_relations(ETG_or_LTG = 'LTG')

#_plot_relations(ETG_or_LTG = 'ETG')
#_plot_relations(ETG_or_LTG = 'LTG')

_plot_relations(csv_sample = 'L100_195_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_')

#===========================






