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
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID
import eagleSqlTools as sql
from graphformat import set_rc_params


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n      local\n      serpens_snap\n      snip\n")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================



"""  
DESCRIPTION
-----------
- When fed a single or list of galaxies into manual_GroupNumList, will plot the misalignment angle between angle_type_in (default is stars_gas_sf) for a radial distribution given by spin_rad_in
- Radial steps can be changed
- 2D/3D can be selected
- Can plot hmr or as [pkpc]


SAMPLE:
------
	Min. sf particle count of 20 within 2 HMR  
    •	(gas_sf_min_particles = 20)
	2D projected angle in ‘z’
    •	(plot_2D_3D = ‘2D’, viewing_axis = ‘z’)
	C.o.M max distance of 2.0 pkpc 
    •	(com_min_distance = 2.0)
	Stars gas sf used
    •	Plot_angle_type = ‘stars_gas_sf’


"""
#==========================
# Run analysis on individual galaxies and output individual CSV files
# SAVED: /outputs/L%s_radial_ID
def _radial_analysis(csv_sample = False,              # Whether to read in existing list of galaxies  
                       #--------------------------
                       mySims = [('RefL0012N0188', 12)],
                       GalaxyID_List = [37445],               # Will create a csv file for each galaxy
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
                       spin_hmr            = np.arange(0.5, 10.1, 0.5),          # multiples of hmr for which to find spin
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
        GroupNum_List       = np.array(dict_new['GroupNum'])
        SubGroupNum_List    = np.array(dict_new['SubGroupNum'])
        GalaxyID_List       = np.array(dict_new['GalaxyID'])
        SnapNum_List        = np.array(dict_new['SnapNum'])
        Redshift_List       = np.array(dict_new['Redshift'])
        HaloMass_List       = np.array(dict_new['halo_mass'])
        Centre_List         = np.array(dict_new['centre'])
        MorphoKinem_List    = np.array(dict_new['MorphoKinem'])
        sample_input        = dict_new['sample_input']
        mySims              = sample_input['mySims']
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
        if debug:
            print(sample_input)
            print(GroupNum_List)
            print(SubGroupNum_List)
            print(GalaxyID_List)
            print(SnapNum_List)
       
        print('\n===================')
        print('SAMPLE LOADED:\n  %s\n  GalaxyIDs: %s' %(mySims[0][0], GalaxyID_List))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('===================')
        
    #---------------------------------------------
    # If no csv_sample given, use GalaxyID_List
    else:
        # Extract GroupNum, SubGroupNum, and Snap for each ID
        GroupNum_List    = []
        SubGroupNum_List = []
        SnapNum_List     = []
        Redshift_List    = []
        HaloMass_List    = []
        Centre_List      = []
        MorphoKinem_List = []
        
        for galID in GalaxyID_List:
            gn, sgn, snap, z, halomass_i, centre_i, morphkinem_i = ConvertID(galID, mySims)
    
            # Append to arrays
            GroupNum_List.append(gn)
            SubGroupNum_List.append(sgn)
            SnapNum_List.append(snap)
            Redshift_List.append(z)
            HaloMass_List.append(halomass_i) 
            Centre_List.append(centre_i)
            MorphoKinem_List.append(morphkinem_i)
            
        if debug:
            print(GroupNum_List)
            print(SubGroupNum_List)
            print(GalaxyID_List)
            print(SnapNum_List)
            
        print('\n===================')
        print('SAMPLE INPUT:\n  %s\n  GalaxyIDs: %s' %(mySims[0][0], GalaxyID_List))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('===================')
        
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
    
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
    #all_gasdata       = {}          # has all particle IDs inside 2hmr
    
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
                    'mySims': mySims}
    
    
    #=================================================================== 
    # Run analysis for each individual galaxy in loaded sample
    for GroupNum, SubGroupNum, GalaxyID, SnapNum, Redshift, HaloMass, Centre_i, MorphoKinem in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List, Redshift_List, HaloMass_List, Centre_List, MorphoKinem_List), total=len(GroupNum_List)):
   
        if print_progress:
             print('Extracting particle data Subhalo_Extract()')
             time_start = time.time()
    
        # Initial extraction of galaxy particle data
        galaxy = Subhalo_Extract(mySims, dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, Centre_i, HaloMass, aperture_rad, viewing_axis)
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
        subhalo = Subhalo_Analysis(mySims, GroupNum, SubGroupNum, GalaxyID, SnapNum, MorphoKinem, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.data_nil, 
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
    

        """ FLAGS
        ------------
        #print(subhalo.flags['total_particles'])            # will flag if there are missing particles within aperture_rad
        #print(subhalo.flags['min_particles'])              # will flag if min. particles not met within spin_rad (will find spin if particles exist, but no uncertainties)
        #print(subhalo.flags['min_inclination'])            # will flag if inclination angle not met within spin_rad... all spins and uncertainties still calculated
        #print(subhalo.flags['com_min_distance'])           # will flag if com distance not met within spin_rad... all spins and uncertainties still calculated
        ------------
        """
        
        if print_galaxy:
            print('|%s| |ID:   %s\t|M*:  %.2e  |HMR:  %.2f  |KAPPA:  %.2f' %(SnapNum, str(subhalo.GalaxyID), subhalo.stelmass, subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'])) 
        
        #--------------------------------
        # Collecting all relevant particle info for galaxy
        all_flags['%s' %str(subhalo.GalaxyID)]          = subhalo.flags
        all_general['%s' %str(subhalo.GalaxyID)]        = subhalo.general
        all_coms['%s' %str(subhalo.GalaxyID)]           = subhalo.coms
        all_spins['%s' %str(subhalo.GalaxyID)]          = subhalo.spins
        all_counts['%s' %str(subhalo.GalaxyID)]         = subhalo.counts
        all_masses['%s' %str(subhalo.GalaxyID)]         = subhalo.masses
        all_misangles['%s' %str(subhalo.GalaxyID)]      = subhalo.mis_angles
        all_misanglesproj['%s' %str(subhalo.GalaxyID)]  = subhalo.mis_angles_proj
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
            csv_dict = {'all_general': all_general,
                        'all_coms': all_coms,
                        'all_spins': all_spins,
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
            json.dump(csv_dict, open('%s/L%s_radial_ID%s_%s_%s_%s_%s.csv' %(output_dir, mySims[0][1], GalaxyID, rad_str, uncertainty_str, angle_str, csv_name), 'w'), cls=NumpyEncoder)
            print('\n  SAVED: %s/L%s_radial_ID%s_%s_%s_%s_%s.csv' %(output_dir, mySims[0][1], GalaxyID, rad_str, uncertainty_str, angle_str, csv_name))
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
          

#=========================
# Plot galaxies fed into from a CSV file | Can also take misalignment sample files (will check if criteria met)
# SAVED: /plots/individual_radial/
def _radial_plot(csv_output = 'L12_radial_ID37445_RadProj_Err__stars_gas_stars_gas_sf_stars_gas_nsf_gas_sf_gas_nsf_stars_dm_',   # CSV sample file to load GroupNum, SubGroupNum, GalaxyID, SnapNum):
                 #--------------------------
                 # Galaxy plotting
                 print_summary = True,
                    use_angles         = ['stars_gas',
                                          'stars_gas_sf',
                                          'stars_dm'],                 # Which angles to plot
                    use_hmr            = np.array([1.0]),         # DOESNT WORK 
                    use_proj_angle     = 'both',                   # Whether to use projected or absolute angle
                    use_uncertainties  = True,                   # Whether to plot uncertainties or not
                 #-------------------------
                 # Plot settings
                 highlight_criteria = True,       # whether to indicate when criteria not met (but still plot)
                 rad_type_plot      = 'hmr',      # 'rad' whether to use absolute distance or hmr 
                 #--------------------------
                 showfig        = True,
                 savefig        = True,
                   file_format  = 'pdf',
                   savefig_txt  = '_bbb',
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
    all_general         = dict_output['all_general']
    all_counts          = dict_output['all_counts']
    all_masses          = dict_output['all_masses']
    all_misangles       = dict_output['all_misangles']
    all_misanglesproj   = dict_output['all_misanglesproj']
    all_flags           = dict_output['all_flags']
    
    # Loading sample criteria
    output_input        = dict_output['output_input']
    
    
    # Extract GroupNum, SubGroupNum, and Snap for each ID
    GalaxyID_List  = list(all_general.keys())
    GroupNum_List    = []
    SubGroupNum_List = []
    SnapNum_List     = []
    Redshift_List    = []
    HaloMass_List    = []
    Centre_List      = []
    MorphoKinem_List = []
    
    for galID in GalaxyID_List:
        gn, sgn, snap, z, halomass_i, centre_i, morphkinem_i = ConvertID(galID, output_input['mySims'])

        # Append to arrays
        GroupNum_List.append(gn)
        SubGroupNum_List.append(sgn)
        SnapNum_List.append(snap)
        Redshift_List.append(z)
        HaloMass_List.append(halomass_i) 
        Centre_List.append(centre_i)
        MorphoKinem_List.append(morphkinem_i)
        
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    if debug:
        print(GroupNum_List)
        print(SubGroupNum_List)
        print(GalaxyID_List)
        print(SnapNum_List)
        
    print('\n===================')
    print('SAMPLE LOADED:\n  %s\n  GalaxyIDs: %s' %(output_input['mySims'][0][0], GalaxyID_List))
    print('  SAMPLE LENGTH: ', len(GroupNum_List))
    print('\nOUTPUT LOADED:\n  Viewing axis: %s\n  Angles: %s\n  HMR: %s\n  Uncertainties: %s\n  Using projected radius: %s\n  COM min distance: %s\n  Min. particles: %s\n  Min. inclination: %s' %(output_input['viewing_axis'], output_input['angle_selection'], output_input['spin_hmr'], output_input['find_uncertainties'], output_input['rad_projected'], output_input['com_min_distance'], output_input['min_particles'], output_input['min_inclination']))
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
    
    # Create particle list of interested values (used later for flag), and plot labels:
    use_particles = []         
    if 'stars_gas' in use_angles:
        if 'stars' not in use_particles:
            use_particles.append('stars')
        if 'gas' not in use_particles:
            use_particles.append('gas')
    if 'stars_gas_sf' in use_angles:
        if 'stars' not in use_particles:
            use_particles.append('stars')
        if 'gas_sf' not in use_particles:
            use_particles.append('gas_sf')
    if 'stars_gas_nsf' in use_angles:
        if 'stars' not in use_particles:
            use_particles.append('stars')
        if 'gas_nsf' not in use_particles:
            use_particles.append('gas_nsf')
    if 'gas_sf_gas_nsf' in use_angles:
        if 'gas_sf' not in use_particles:
            use_particles.append('gas_sf')
        if 'gas_nsf' not in use_particles:
            use_particles.append('gas_nsf')
    if 'stars_dm' in use_angles:
        if 'stars' not in use_particles:
            use_particles.append('stars')
        if 'dm' not in use_particles:
            use_particles.append('dm')
    if 'gas_dm' in use_angles:
        if 'gas' not in use_particles:
            use_particles.append('gas')
        if 'dm' not in use_particles:
            use_particles.append('dm')
    if 'gas_sf_dm' in use_angles:
        if 'gas_sf' not in use_particles:
            use_particles.append('gas_sf')
        if 'dm' not in use_particles:
            use_particles.append('dm')
    if 'gas_nsf_dm' in use_angles:
        if 'gas_nsf' not in use_particles:
            use_particles.append('gas_nsf')
        if 'dm' not in use_particles:
            use_particles.append('dm')
    
    
    #=================================================================== 
    # Run analysis for each individual galaxy in loaded sample
    for GroupNum, SubGroupNum, GalaxyID, SnapNum, Redshift, HaloMass, Centre, MorphoKinem in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List, Redshift_List, HaloMass_List, Centre_List, MorphoKinem_List), total=len(GroupNum_List)):
        
        #---------------------------------------------------
        # Graph initialising and base formatting
        fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]}, figsize=[7, 8], sharex=True, sharey=False)

        
        #----------------------
        # Highlighting selection criteria when broken
        if highlight_criteria:
            # Min. particles and inclination angle
            for parttype_name in use_particles:
                
                #---------------------
                # Highlight min. particles
                if len(all_flags['%s' %GalaxyID]['min_particles'][parttype_name]) > 0:
                    for hmr_i in all_flags['%s' %GalaxyID]['min_particles'][parttype_name]:
                        if rad_type_plot == 'hmr':
                            axs[0].axvline(hmr_i, c='grey', alpha=0.2, lw=20)
                        elif rad_type_plot == 'rad':
                            axs[0].axvline(hmr_i*all_general['%s' %GalaxyID]['halfmass_rad_proj'], c='grey', alpha=0.1, lw=20)
                
                #---------------------
                # Highlight inclination angle
                if len(all_flags['%s' %GalaxyID]['min_inclination'][parttype_name]) > 0:
                    for hmr_i in all_flags['%s' %GalaxyID]['min_inclination'][parttype_name]:
                        if rad_type_plot == 'hmr':
                            axs[0].axvline(hmr_i, c='indigo', alpha=0.2, lw=20)
                        elif rad_type_plot == 'rad':
                            axs[0].axvline(hmr_i*all_general['%s' %GalaxyID]['halfmass_rad_proj'], c='grey', alpha=0.1, lw=20)  
            
            # COM distance    
            for use_angle_i in use_angles:
        
                #---------------------
                # Highlight COM min distance
                if len(all_flags['%s' %GalaxyID]['com_min_distance'][use_angle_i]) > 0:
                    for hmr_i in all_flags['%s' %GalaxyID]['com_min_distance'][use_angle_i]:
                        if rad_type_plot == 'hmr':
                            axs[0].axvline(hmr_i, c='r', alpha=0.2, lw=20)
                        elif rad_type_plot == 'rad':
                            axs[0].axvline(hmr_i*all_general['%s' %GalaxyID]['halfmass_rad_proj'], c='grey', alpha=0.1, lw=20)
                        
                        
        #========================================               
        # Plot each angle type individually
        for use_angle_i in use_angles:
            
            #------------------------
            # Setting colors and labels
            if use_angle_i == 'stars_gas':
                plot_color = 'green'
                plot_label = 'Stars-gas'
            elif use_angle_i == 'stars_gas_sf':
                plot_color = 'b'
                plot_label = 'Stars-gas$_{\mathrm{sf}}$'
            elif use_angle_i == 'stars_gas_nsf':
                plot_color = 'indigo'
                plot_label = 'Stars-gas$_{\mathrm{nsf}}$'
            elif use_angle_i == 'stars_dm':
                plot_color = 'r'
                plot_label = 'Stars-DM'
            else:
                plot_color = 'brown'
                plot_label = use_angle_i
                 
            
            #------------------------
            # Collect values to plot
            
            # Radii for plot
            if rad_type_plot == 'hmr':
                plot_rad = np.array(all_misangles['%s' %GalaxyID]['hmr'])
            elif rad_type_plot == 'rad':
                plot_rad = np.array(all_misangles['%s' %GalaxyID]['rad'])
            
            # Angles for plot
            if use_proj_angle == True:
                plot_angles    = np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle_i])
                plot_angles_lo = np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle_i])[:,0]
                plot_angles_hi = np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle_i])[:,1]
            elif use_proj_angle == False:
                plot_angles    = np.array(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle_i])
                plot_angles_lo = np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle_i])[:,0]
                plot_angles_hi = np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle_i])[:,1]
            elif use_proj_angle == 'both':
                plot_angles    = np.array(all_misangles['%s' %GalaxyID]['%s_angle' %use_angle_i])
                plot_angles_lo = np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle_i])[:,0]
                plot_angles_hi = np.array(all_misangles['%s' %GalaxyID]['%s_angle_err' %use_angle_i])[:,1]
                plot_angles_proj    = np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle' %use_angle_i])
                plot_angles_lo_proj = np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle_i])[:,0]
                plot_angles_hi_proj = np.array(all_misanglesproj['%s' %GalaxyID][output_input['viewing_axis']]['%s_angle_err' %use_angle_i])[:,1]
                
            
            if debug:
                print('Plot rad:', plot_rad)
                print('Angles:', plot_angles)
                print('Errors Lo:', plot_angles_lo)
                print('Errors Hi:', plot_angles_hi)
        
            #-----------------------
            # Plot scatter and errorbars            
            
            if use_proj_angle == 'both':
                axs[0].fill_between(plot_rad, plot_angles_lo_proj, plot_angles_hi_proj, facecolor=plot_color, alpha=0.2)
                axs[0].errorbar(plot_rad, plot_angles, yerr=[abs(plot_angles_lo-plot_angles), abs(plot_angles_hi-plot_angles)], ecolor=plot_color, ls='none', capsize=3, elinewidth=0.7, markeredgewidth=1)
                #axs[0].fill_between(plot_rad, plot_angles_lo_proj, plot_angles_hi_proj, facecolor=plot_color, alpha=0.2)
                axs[0].plot(plot_rad, plot_angles_proj, label=plot_label, c=plot_color, alpha=1.0, ms=2, lw=1.5)
                axs[0].plot(plot_rad, plot_angles, c=plot_color, alpha=1.0, ms=2, lw=1.5, ls=':')
            else:
                axs[0].fill_between(plot_rad, plot_angles_lo, plot_angles_hi, facecolor=plot_color, alpha=0.2)
                axs[0].plot(plot_rad, plot_angles, label=plot_label, c=plot_color, alpha=1.0, ms=2, lw=1.5)
            
                 
                
        #================================
        # Mass fractions
        plot_stars_gas_frac    = np.divide(np.array(all_masses['%s' %GalaxyID]['gas']), np.array(all_masses['%s' %GalaxyID]['stars']) + np.array(all_masses['%s' %GalaxyID]['gas']))
        plot_stars_gas_sf_frac = np.divide(np.array(all_masses['%s' %GalaxyID]['gas_sf']), np.array(all_masses['%s' %GalaxyID]['stars']) + np.array(all_masses['%s' %GalaxyID]['gas']))
        plot_gas_ratio         = np.divide(np.array(all_masses['%s' %GalaxyID]['gas_sf']), np.array(all_masses['%s' %GalaxyID]['gas']))
        
        if debug:
            print('Gas mass fraction: ', plot_stars_gas_frac)
            print('Gas_sf mass fraction: ', plot_stars_gas_sf_frac)
            print('Gas_sf:Gas ratio: ', plot_gas_ratio)
        
        # Plot mass fractions
        axs[1].plot(plot_rad, np.log10(plot_stars_gas_frac), alpha=0.8, lw=2, c='green', label='$f_{\mathrm{gas}}$')
        axs[1].plot(plot_rad, np.log10(plot_stars_gas_sf_frac), alpha=0.8, lw=2, c='b', label='$f_{\mathrm{SF}}$')
        #axs[1].plot(plot_rad, plot_gas_ratio, c='k', ls='--', label='Gas$_{SF}$/Gas')
        

        #-===============================
        ### General formatting
        
        # Axis labels
        axs[0].set_yticks(np.arange(0, 181, 30))
        #axs[1].set_yticks(np.arange(0, 1.1, 0.25))
        #axs[1].set_yticklabels(['0', '', '', '', '1'])
        #axs[1].set_yscale('log')
        axs[1].set_ylim(-3, 0)
        axs[1].set_yticks([-3, -2, -1, 0])
        axs[0].set_ylim(0, 180)
        axs[1].set_ylabel('log$_{10}$(Mass fraction)')
        axs[0].set_xlim(0, max(plot_rad))
        axs[0].set_xticks(np.arange(0, max(plot_rad)+1, 1))
        if rad_type_plot == 'hmr':
            axs[1].set_xlabel('Stellar half-mass radius, $r_{1/2,z}$')
        if rad_type_plot == 'rad':
            axs[1].set_xlabel('Radial distance from centre [pkpc]')
        axs[0].set_ylabel('Misalignment angle, $\psi$')     
               
        #-----------
        # Annotations
        axs[0].text(0, 185, 'GalaxyID: %s' %GalaxyID)
    
        #-----------
        # Legend
        axs[0].plot(-10, -10, ls=':', c='k', label='$\psi_{\mathrm{3D}}$')
        axs[0].plot(-10, -10, ls='-', c='k', label='$\psi_{z}$')
        axs[0].legend(loc='center right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=1)
        axs[1].legend(loc='lower right', frameon=False, labelspacing=0.1, labelcolor='linecolor', handlelength=0)
        
        #-----------
        # Other
        axs[0].grid(alpha=0.3)
        axs[1].grid(alpha=0.3)
        plt.tight_layout()
    
    
        #=====================================
        ### Print summary
        
        #-----------
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        metadata_plot = {'Title': 'GalaxyID: %s\nM*: %.2e\nHMR: %.2f\nKappa: %.2f\nTriax: %.2f' %(GalaxyID, all_general['%s' %GalaxyID]['stelmass'], all_general['%s' %GalaxyID]['halfmass_rad_proj'], all_general['%s' %GalaxyID]['kappa_stars'], all_general['%s' %GalaxyID]['triax'])}
        
        angle_str = ''
        for angle_name in list(use_angles):
            angle_str = '%s_%s' %(str(angle_str), str(angle_name))
        
        
        if savefig:
            plt.savefig("%s/individual_radial/L%s_radial_ID%s_proj%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], GalaxyID, use_proj_angle, angle_str, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/individual_radial/L%s_radial_ID%s_proj%s_%s_%s.%s" %(fig_dir, output_input['mySims'][0][1], GalaxyID, use_proj_angle, angle_str, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
        

#=========================== 
#_radial_analysis()

#----------------
_radial_plot()   
#===========================



