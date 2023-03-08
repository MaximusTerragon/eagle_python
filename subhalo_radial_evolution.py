import h5py
import numpy as np
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import astropy.units as u
import astropy.cosmology.units as cu
import csv
import json
import time
import math
from datetime import datetime
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from subhalo_main import Subhalo_Extract, Subhalo, MergerTree, ConvertID, ConvertGN
import eagleSqlTools as sql
from graphformat import graphformat


# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
#mySims = np.array([('RefL0100N1504', 100)])   
snapNum = 28

# Directories of data hdf5 file(s)
dataDir_main = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
dataDir_dict = {}
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

 


"""  
DESCRIPTION
-----------
- When fed a single GalaxyID, will plot the radial evolution of galaxy with increasing radius over multiple snapshots
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
#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
# 3748
# 37445
def plot_radial_evolution(manual_GalaxyIDList_target = np.array([37445]),       # AT Z=0 leave empty if ignore
                               manual_GroupNumList_target = np.array([]),           # AT Z=0 manually enter galaxy gns we want. -1 for nothing
                               snapNum_target      = 28,                            #Snap number of the target 
                               SubGroupNum_target  = 0,
                               snapNumMax          = 15,                    # maximum snap number we are interested in
                                 galaxy_mass_limit = 10**9.5,                         # for print 
                                 csv_load       = False,              # .csv file will ALL data
                                   csv_load_name = '',                  #FIND IN LINUX, mac is weird
                             kappa_rad_in    = 30,                          # calculate kappa for this radius [pkpc]    
                             aperture_rad_in = 30,                          # trim all data to this maximum value before calculations
                             trim_rad_in     = np.array([100]),             # keep as 100, doesn't matter as capped by aperture anyway
                             align_rad_in    = False,                       # keep on False   
                             orientate_to_axis='z',                         # keep as z
                             viewing_angle=0,                               #keep as 0
                                     find_uncertainties      = True,                    # whether to find 2D and 3D uncertainties
                                     viewing_axis            = 'z',                     # Which axis to view galaxy from.  DEFAULT 'z'
                                     com_min_distance        = 2.0,                     # [pkpc] min distance between sfgas and stars. Min radius of spin_rad_in used
                                     gas_sf_min_particles    = 20,                      # Minimum gas sf particles to use galaxy.  DEFAULT 100
                                     min_inclination         = 0,                       # Minimum inclination toward viewing axis [deg] DEFAULT 0
                                     projected_or_abs        = 'projected',              # 'projected' or 'abs'
                                     spin_hmr_in             = np.array([1.0, 2.0, 3.0, 5.0]),    # multiples of HMR
                                     angle_type_in           = np.array(['stars_gas', 'stars_gas_sf']),       # analytical results for constituent particles will be found. ei., stars_gas_sf will mean stars and gas_sf properties will be found, and the angles between them                                                           
                                     plot_2D_3D           = '2D',                # whether to plot 2D or 3D angle
                                     rad_type_plot        = 'hmr',               # 'rad' whether to use absolute distance or hmr
                                     plot_angle_type_in   = np.array(['stars_gas_sf']),         # angle types to plot
                             plot_single_rad = True,                    # keep on true
                             root_file = '/Users/c22048063/Documents/EAGLE/plots',
                             file_format = 'png',
                               print_galaxy       = False,
                               print_galaxy_short = True,
                               print_progress     = True,
                               csv_file = False,                     # whether to create a csv file of used data
                                 csv_name = 'data_radial',          # name of .csv file
                               savefig = True,
                               showfig = True,  
                                 savefigtxt = '_new', 
                               debug = False):         
          
    time_start = time.time()  
    

    # Load existing data from csv file
    if csv_load:
        print('NOT CONFIGURED')
    
    
    if not csv_load:
        # Creating list (usually just single galaxy)
        if len(manual_GroupNumList_target) > 0:
            # Converting names of variables for consistency
            GroupNumList_target    = manual_GroupNumList_target
            SubGroupNumList_target = np.full(len(manual_GroupNumList_target), SubGroupNum_target)
            snapNumList_target     = np.full(len(manaul_GroupNumList_target), snapNum_target)
            
            GalaxyIDList_target = []
            for gn, sgn, snap in zip(GroupNumList_target, SubGroupNumList_target, snapNumList_target):
                galID = ConvertGN(gn, sgn, snap, mySims)
                GalaxyIDList_target.append(galID)
            
        elif len(manual_GalaxyIDList_target) > 0:
            GalaxyIDList_target    = manual_GalaxyIDList_target
            snapNumList_target     = np.full(len(manual_GalaxyIDList_target), snapNum_target)
            
            # Extract GroupNum, SubGroupNum, and Snap for each ID
            GroupNumList_target    = []
            SubGroupNumList_target = []
            for galID in manual_GalaxyIDList_target:
                gn, sgn, snap = ConvertID(galID, mySims)
                
                # Append to arrays
                GroupNumList_target.append(gn)
                SubGroupNumList_target.append(sgn)
    
        """ We are left with:
        GroupNumList_target     
        SubGroupNumList_target
        snapNumList_target
        GalaxyIDList_target        
        """
        
        #-------------------------------------------------------------------
        # Automating some later variables to avoid putting them in manually

        # making particle_list_in and angle_selection obsolete:
        particle_list_in = []
        angle_selection  = []
        if 'stars_gas' in angle_type_in:
            if 'stars' not in particle_list_in:
                particle_list_in.append('stars')
            if 'gas' not in particle_list_in:
                particle_list_in.append('gas')
            angle_selection.append(['stars', 'gas'])
        if 'stars_gas_sf' in angle_type_in:
            if 'stars' not in particle_list_in:
                particle_list_in.append('stars')
            if 'gas_sf' not in particle_list_in:
                particle_list_in.append('gas_sf')
            angle_selection.append(['stars', 'gas_sf'])
        if 'stars_gas_nsf' in angle_type_in:
            if 'stars' not in particle_list_in:
                particle_list_in.append('stars')
            if 'gas_nsf' not in particle_list_in:
                particle_list_in.append('gas_nsf')
            angle_selection.append(['stars', 'gas_nsf'])
        if 'gas_sf_gas_nsf' in angle_type_in:
            if 'gas_sf' not in particle_list_in:
                particle_list_in.append('gas_sf')
            if 'gas_nsf' not in particle_list_in:
                particle_list_in.append('gas_nsf')
            angle_selection.append(['gas_sf', 'gas_nsf'])
        #-------------------------------------------------------------------

        
        #=================================================================== 
        # For each target galaxy we are interested in...
        for target_GroupNum, target_SubGroupNum, target_GalaxyID, target_snapNum in zip(np.array(GroupNumList_target), np.array(SubGroupNumList_target), GalaxyIDList_target, snapNumList_target):
            # Empty dictionaries to collect relevant data
            all_flags         = {}          # has reason why galaxy failed sample
            all_general       = {}          # has total masses, kappa, halfmassrad
            all_coms          = {}
            all_misangles     = {}          # has all 3d angles
            all_particles     = {}          # has all the particle count and mass within rad
            all_misanglesproj = {}   # has all 2d projected angles from 3d when given a viewing axis and viewing_angle = 0
        
            if print_progress:
                print('Extracting merger tree data MergerTree()')
                time_start = time.time()
            
            # Extract merger tree data
            tree = MergerTree(mySims, target_GalaxyID, snapNumMax) 
           
            
            #plt.plot(tree.main_branch['lookbacktime'], np.log10(tree.main_branch['totalstelmass']))
            #plt.plot(tree.main_branch['lookbacktime'], np.log10(tree.main_branch['totalgasmass']))
            #plt.xlim(0, 13.5)
            #plt.ylim(6, 11)
            #plt.gca().invert_xaxis()
            #plt.show()
            
            
            if print_galaxy_short:
                print('ID\tSNAP\tz\tGN\tSGN')
                for ids, snap, red, gn, sgn in zip(tree.main_branch['GalaxyID'], tree.main_branch['snapnum'], tree.main_branch['redshift'], tree.main_branch['GroupNumber'], tree.main_branch['SubGroupNumber']):
                    print('%i\t%i\t%.2f\t%i\t%i' %(ids, snap, red, gn, sgn))

            #==================================================================================================
            # tree.main_branch['GroupNumber'] and tree.main_branch['SubGroupNumber'] have all the gns and sgns
            for GroupNum, SubGroupNum, snapNum in tqdm(zip(tree.main_branch['GroupNumber'], tree.main_branch['SubGroupNumber'], tree.main_branch['snapnum']), total=len(tree.main_branch['GroupNumber'])):
                
                if print_progress:
                    print('Extracting particle data Subhalo_Extract()')
                    time_start = time.time()
    
                # Initial extraction of galaxy data
                galaxy = Subhalo_Extract(mySims, dataDir_dict['%s' %str(snapNum)], snapNum, GroupNum, SubGroupNum, aperture_rad_in, viewing_axis)
    
                #-------------------------------------------------------------------
                # Automating some later variables to avoid putting them in manually
                if projected_or_abs == 'projected':
                    use_rad = galaxy.halfmass_rad_proj
                elif projected_or_abs == 'abs':
                    use_rad = galaxy.halfmass_rad
        
        
                spin_rad = spin_hmr_in * use_rad                                          #pkpc
                spin_rad = [x for x in spin_rad if x <= aperture_rad_in]
                spin_hmr = [x for x in spin_hmr_in if x*use_rad <= aperture_rad_in]
                if len(spin_rad) != len(spin_hmr_in):
                    print('Capped spin_rad (%s pkpc) at aperture radius (%s pkpc)' %(max(spin_rad), aperture_rad_in))
                 
                trim_rad = trim_rad_in
                aperture_rad = aperture_rad_in
        
                if kappa_rad_in == 'rad':
                    kappa_rad = use_rad
                elif kappa_rad_in == 'tworad':
                    kappa_rad = 2*use_rad
                else:
                    kappa_rad = kappa_rad_in
                if align_rad_in == 'rad':
                    align_rad = use_rad
                elif align_rad_in == 'tworad':
                    align_rad = use_rad
                else:
                    align_rad = align_rad_in  
                #------------------------------------------------------------------
        
                if print_progress:
                    print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                    print('Running particle data analysis Subhalo()')
                    time_start = time.time()
        
                # Galaxy will be rotated to calc_kappa_rad's stellar spin value
                with np.errstate(divide='ignore', invalid='ignore'):
                    # subhalo.halfmass_rad_proj will be either projected or absolute, use this
                    subhalo = Subhalo(galaxy.gn, galaxy.sgn, galaxy.GalaxyID, galaxy.stelmass, galaxy.gasmass, galaxy.halfmass_rad, use_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, galaxy.MorphoKinem,
                                                        angle_selection,
                                                        viewing_angle,
                                                        spin_rad,
                                                        spin_hmr,
                                                        trim_rad, 
                                                        kappa_rad, 
                                                        aperture_rad,
                                                        align_rad,              #align_rad = False
                                                        orientate_to_axis,
                                                        viewing_axis,
                                                        com_min_distance,
                                                        gas_sf_min_particles,
                                                        particle_list_in,
                                                        angle_type_in,
                                                        find_uncertainties,
                                                        min_inclination,
                                                        quiet=True)
                
                print('STELMASS: %f\t%f\tDIFF: %f' %(galaxy.stelmass, subhalo.stelmass, galaxy.stelmass-subhalo.stelmass))
                
                #--------------------------------
                # Collecting all relevant particle info for galaxy
                all_general['%s' %str(subhalo.GalaxyID)]       = subhalo.general
                all_flags['%s' %str(subhalo.GalaxyID)]         = subhalo.flags
                all_particles['%s' %str(subhalo.GalaxyID)]     = subhalo.particles
                all_coms['%s' %str(subhalo.GalaxyID)]          = subhalo.coms
                all_misangles['%s' %str(subhalo.GalaxyID)]     = subhalo.mis_angles
                all_misanglesproj['%s' %str(subhalo.GalaxyID)] = subhalo.mis_angles_proj
                #---------------------------------
                
                print('  FLAGS:\t', subhalo.flags)
                if len(subhalo.flags) > 0:
                    continue
                    
                # Print galaxy properties
                if print_galaxy == True:
                    print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
                    print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))       # [Msun]
                    print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad)             # [pkpc]
                    print('KAPPA:                  %.2f' %subhalo.kappa)
                    print('KAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)
                    print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
                    mask = np.where(np.array(subhalo.coms['hmr'] == min(spin_hmr_in)))
                    print('C.O.M %s HMR STARS-SF [pkpc]:  %.2f' %(str(min(spin_hmr_in)), subhalo.coms['stars_gas_sf'][int(mask[0])]))
                elif print_galaxy_short == True:
                    print('|SNAP:\t%s\t|GN:\t%s\t|ID:\t%s\t|HMR:\t%.2f\t|KAPPA / SF:\t%.2f  %.2f' %(str(snapNum), str(subhalo.gn), str(subhalo.GalaxyID), subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'], subhalo.general['kappa_gas_sf'])) 
            
                
                #===================================================================
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
            
                    # Combining all dictionaries
                    csv_dict = {'all_flags': all_flags, 'all_general': all_general, 'all_misangles': all_misangles, 'all_misanglesproj': all_misanglesproj, 'all_coms': all_coms, 'all_particles': all_particles}
                    csv_dict.update({'function_input': str(inspect.signature(plot_radial_misalignment))})
    
                    # Writing one massive JSON file
                    json.dump(csv_dict, open('%s/%s_%s.csv' %(root_file, csv_name, str(datetime.now())), 'w'), cls=NumpyEncoder)
    
                    """# Reading JSON file
                    dict_new = json.load(open('%s/%s.csv' %(root_file, csv_name), 'r'))
                    # example nested dictionaries
                    new_general = dict_new['all_general']
                    new_misanglesproj = dict_new['all_misanglesproj']
                    # example accessing function input
                    function_input = dict_new['function_input']"""

                
            #==========================
            # End of particle data loop
            
            """ DATA WE HAVE:
            -----------------
            tree
            .target_GalaxyID:       int
                Galaxy ID at z=0 that we care about
            .sim:
                Simulation that we used
            .TopLeafID:
                The oldest galaxy ID in the main branch. All galaxies
                on the main branch will have the same TopLeafID
            .all_branches:      dict
                Has raw data from all branches, including main branch
                ['redshift']        - redshift list (has duplicates, but ordered largest-> smallest)
                ['snapnum']         - 
                ['GalaxyID']        - Unique ID of specific galaxy
                ['DescendantID']    - DescendantID of the galaxy next up, this has been selected to lie on the main branch
                ['TopLeafID']       - Oldest galaxy ID of branch. Can differ, but all on main have same as .TopLeafID
                ['GroupNumber']     - Will mostly be the same as interested galaxy
                ['SubGroupNumber']  - Will mostly be satellites
                ['stelmass']        - stellar mass in 30pkpc
                ['gasmass']         - gasmass in 30pkpc
                ['totalstelmass']   - total Subhalo stellar mass
            .main_branch        dict
                Has raw data from only main branch
                ['redshift']        - redshift list (no duplicates, large -> small)
                ['lookbacktime']        - lookback time
                ['snapnum']         - 
                ['GalaxyID']        - Unique ID of specific galaxy
                ['DescendantID']    - DescendantID of the galaxy next up, this has been selected to lie on the main branch
                ['TopLeafID']       - Oldest galaxy ID of branch. Can differ, but all on main have same as .TopLeafID
                ['GroupNumber']     - Will mostly be the same as interested galaxy
                ['SubGroupNumber']  - Will mostly be satellites
                ['stelmass']        - stellar mass in 30pkpc
                ['gasmass']         - gasmass in 30pkpc
                ['totalstelmass']   - total Subhalo stellar mass
            .mergers            dict
                Has complete data on all mergers, ratios, and IDs. Merger data will be entered
                for snapshot immediately AFTER both galaxies do not appear.

                For example: 
                    Snap    ID              Mass
                    27      35787 264545    10^7 10^4
                    28      35786           10^7...
                                    ... Here the merger takes place at Snap28
                ['redshift']        - Redshift (unique)
                ['snapnum']         - Snapnums (unique)
                ['ratios']          - Array of mass ratios at this redshift
                ['gasratios']       - Array of gas ratios at this redshift
                ['GalaxyIDs']       - Array of type [Primary GalaxyID, Secondary GalaxyID]

            
            For each GalaxyID in tree.main_branch:
              all_general[GalaxyID]
              all_flags[GalaxyID]
              all_particles[GalaxyID]
              all_coms[GalaxyID]
              all_misangles[GalaxyID]
              all_misanglesproj[GalaxyID]
            """
            
    
            def _plot_rad(quiet=1, debug=False):
                # Plots 3D projected misalignment angle from a viewing axis
                if plot_2D_3D == '2D':
                    
                    # Initialise figure
                    graphformat(8, 11, 11, 9, 11, 6.5, 3.75)
                    fig, axs = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [1.5, 1, 1, 1]}, figsize=[3.5, 7], sharex=True, sharey=False)                    
                    
                    
                    # Find kappas and other _general stats
                    kappa_stars       = []
                    kappa_gas         = []
                    
                    # Same as taking 'for redshift in XX'
                    for GalaxyID_i, lookbacktime_i in zip(tree.main_branch['GalaxyID'], tree.main_branch['lookbacktime']):
                        if len(all_flags['%s' %str(GalaxyID_i)]) == 0:
                            kappa_stars.append(all_general['%s' %str(GalaxyID_i)]['kappa_stars'])
                            kappa_gas.append(all_general['%s' %str(GalaxyID_i)]['kappa_gas_sf'])
                        
                        
                    #---------------------------
                    # Loop over each angle type
                    for angle_type_in_i in plot_angle_type_in:
                        # angletype_in_i will be dotted for total gas, and line for SF gas
                        if angle_type_in_i == 'stars_gas':
                            ls = '--'
                        elif angle_type_in_i == 'stars_gas_sf':
                            ls = '-'
                        
                        
                        # Create some colormaps of things we want
                        colors_blues    = plt.get_cmap('Blues_r')(np.linspace(0, 0.7, len(spin_hmr_in)))
                        colors_reds     = plt.get_cmap('Reds_r')(np.linspace(0, 0.7, len(spin_hmr_in)))
                        colors_greens   = plt.get_cmap('Greens_r')(np.linspace(0, 0.7, len(spin_hmr_in)))
                        colors_spectral = plt.get_cmap('Spectral')(np.linspace(0.1, 0.9, len(spin_hmr_in)))
                        
                        
                        #-----------------------
                        # Loop over each rad
                        for hmr_i, color_b, color_r, color_g, color_s in zip(spin_hmr_in, colors_blues, colors_reds, colors_greens, colors_spectral):
                            
                            GalaxyID_plot     = []
                            GalaxyID_notplot  = []
                            lookbacktime_plot = []
                            
                            misangle_plot     = []
                            misangle_err_lo_plot = []
                            misangle_err_hi_plot = []
                            
                            stelmass_plot     = []
                            gasmass_plot      = []
                            gassfmass_plot    = []
                            
                            gas_frac_plot     = []
                            gas_sf_frac_plot  = []
                            
                            
                            #-------------------------------------
                            # Same as taking 'for redshift in XX'
                            for GalaxyID_i, lookbacktime_i in zip(tree.main_branch['GalaxyID'], tree.main_branch['lookbacktime']):
                                
                                print('hmr_i', hmr_i)

                                
                                
                                if len(all_flags['%s' %str(GalaxyID_i)]) == 0:
                                    # Mask correct integer (formatting weird but works)

                                    print(GalaxyID_i)
                                    print(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['hmr'][0])
                                    print(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['hmr'][1])
                                    print(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['hmr'][2])
                                    print(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['hmr'][3])
                                    
                                    print(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['hmr'])
                                    print(np.where(np.array(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['hmr']) == hmr_i)[0])
                                    
                                    
                                    # remove ceil
                                    
                                    
                                    
                                    mask_rad = int(np.where(np.ceil(np.array(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['hmr'])) == hmr_i)[0])
                                
                                    misangle_plot.append(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['%s_angle' %angle_type_in_i][mask_rad])
                                    misangle_err_lo_plot.append(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['%s_angle_err' %angle_type_in_i][mask_rad][0])
                                    misangle_err_hi_plot.append(all_misanglesproj['%s' %str(GalaxyID_i)][viewing_axis]['%s_angle_err' %angle_type_in_i][mask_rad][1])
                                    
                                    stelmass_plot.append(all_particles['%s' %str(GalaxyID_i)]['stars_mass'][mask_rad])
                                    gasmass_plot.append(all_particles['%s' %str(GalaxyID_i)]['gas_mass'][mask_rad])
                                    gassfmass_plot.append(all_particles['%s' %str(GalaxyID_i)]['gas_sf_mass'][mask_rad])
                                    
                                    gas_frac_plot.append(all_particles['%s' %str(GalaxyID_i)]['gas_mass'][mask_rad]  / all_particles['%s' %str(GalaxyID_i)]['stars_mass'][mask_rad])
                                    gas_sf_frac_plot.append(all_particles['%s' %str(GalaxyID_i)]['gas_sf_mass'][mask_rad]  / all_particles['%s' %str(GalaxyID_i)]['stars_mass'][mask_rad])
                                    
                                    lookbacktime_plot.append(lookbacktime_i)
                                    GalaxyID_plot.append(GalaxyID_i)
                                    
                                else:
                                    print('VOID ID: %s\t\t%s' %(str(GalaxyID_i), str(lookbacktime_i)))
                                    print(all_flags['%s' %str(GalaxyID_i)])
                                    
                                    misangle_plot.append(math.nan)
                                    misangle_err_lo_plot.append(math.nan)
                                    misangle_err_hi_plot.append(math.nan)
                                    
                                    stelmass_plot.append(math.nan)
                                    #gasmass_plot.append(math.nan)
                                    gassfmass_plot.append(math.nan)
                                    
                                    gas_frac_plot.append(math.nan)
                                    gas_sf_frac_plot.append(math.nan)
                                    
                                    lookbacktime_plot.append(lookbacktime_i)
                                    GalaxyID_notplot.append(GalaxyID_i)
                                    
                               
                            if debug == True:
                                print('\nGalaxyID plot    ', GalaxyID_plot)
                                print('GalaxyID not plot', GalaxyID_notplot)
                                print(' ')
                                print('lookbacktime_plot', lookbacktime_plot)
                                print(' ')
                                print('misangle_plot       ', misangle_plot)
                                print('misangle_err_lo_plot', misangle_err_lo_plot)
                                print('misangle_err_hi_plot', misangle_err_hi_plot)
                                print(' ')
                                print('stelmass_plot ', stelmass_plot)
                                print('gasmass_plot  ', gasmass_plot)
                                print('gassfmass_plot', gassfmass_plot)
                                print(' ')
                                print('gas_frac_plot   ', gas_frac_plot)
                                print('gas_sf_frac_plot', gas_sf_frac_plot)
                            
                            
                            #========================
                            # Plotting
                            # Plot 1: Misalignment angles, errors, with time/redshift
                            # Plot 2: Stellar mass, gas mass, gas sf mass, with time/redshift
                            # Plot 3: Gas fractions with time/redshift
                            
                            #------------------------
                            # PLOT 1
                            # Plot scatter and errorbars
                            if angle_type_in_i == 'stars_gas':
                                axs[0].plot(lookbacktime_plot, misangle_plot, label='Stars-gas$_{Total}$', alpha=1.0, ms=2, lw=1, ls=ls, c=color_s)
                            if angle_type_in_i == 'stars_gas_sf':
                                axs[0].plot(lookbacktime_plot, misangle_plot, label='Stars-gas$_{SF}$', alpha=1.0, ms=2, lw=1, ls=ls, c=color_s)
                            axs[0].fill_between(lookbacktime_plot, misangle_err_lo_plot, misangle_err_hi_plot, alpha=0.3, color=color_s)
                            
                            
                            
                            #------------------------
                            # PLOT 2
                            # Plot masses
                            axs[1].plot(lookbacktime_plot, np.log10(stelmass_plot), alpha=1.0, lw=1, c=color_r, label='Stars')
                            #axs[1].plot(lookbacktime_plot, np.log10(gasmass_plot), alpha=1.0, lw=1, c=color_b, label='Gas$_{Total}$')
                            axs[1].plot(lookbacktime_plot, np.log10(gassfmass_plot), alpha=1.0, lw=1, c=color_g, label='Gas$_{SF}$')
                            
                            
                            #------------------------
                            # PLOT 3
                            # Plot gas frations
                            axs[2].plot(lookbacktime_plot, gas_frac_plot, alpha=1.0, lw=1, c=color_b, label='Gas$_{Total}$')
                            axs[2].plot(lookbacktime_plot, gas_sf_frac_plot, alpha=1.0, lw=1, c=color_g, label='Gas$_{SF}$')
                        
                    #------------------------
                    # PLOT 4
                    # Plot kappas
                    axs[3].plot(lookbacktime_plot, kappa_stars, alpha=1.0, lw=1, c='r', label='\u03BA$_{Stars}$')
                    axs[3].plot(lookbacktime_plot, kappa_gas, alpha=1.0, lw=1, c='b', label='\u03BA$_{Gas}$')
                    
                    #=====================
                    # Add mergers
                    print(' ')
                    print(tree.mergers['snapnum'])
                    print(tree.mergers['ratios'])
                    print(' ')
                    for ratio_i, lookbacktime_i, snap_i in zip(tree.mergers['ratios'], tree.main_branch['lookbacktime'], tree.main_branch['snapnum']):
                        
                        if len(ratio_i) == 0:
                            next
                        else:
                            if max(ratio_i) >= 0.01:
                                for ax in axs:
                                    ax.axvline(lookbacktime_i, ls='-', color='grey', alpha=min(0.5, max(ratio_i)*10), linewidth=3)
                        
                                # Annotate
                                axs[0].text(lookbacktime_i, 180, '%.2f' %max(ratio_i), fontsize=7)
                    
                    
                    ### General formatting 
                    
                    ### Customise legend labels
                    colors_blues    = plt.get_cmap('Blues_r')(np.linspace(0.2, 0.7, len(spin_hmr_in)))
                    colors_reds     = plt.get_cmap('Reds_r')(np.linspace(0.2, 0.7, len(spin_hmr_in)))
                    colors_greens   = plt.get_cmap('Greens_r')(np.linspace(0.2, 0.7, len(spin_hmr_in)))
                    colors_spectral = plt.get_cmap('Spectral')(np.linspace(0.1, 0.9, len(spin_hmr_in)))
                    
                    legend_elements_0 = [Line2D([0], [0], marker=' ', linestyle='-', color='k'), Line2D([0], [0], marker=' ', linestyle='--', color='k')]
                    labels_0          = ['Stars-gas$_{Total}$', 'Stars-gas$_{SF}$']
                    
                    legend_elements_05 = []
                    labels_05          = []
                    for hmr_i, color_b, color_r, color_g, color_s in zip(spin_hmr_in, colors_blues, colors_reds, colors_greens, colors_spectral):
                        legend_elements_05.append(Line2D([0], [0], marker=' ', color=color_s))
                        labels_05.append('%s r$_{HMR}$' %str(hmr_i))
                    
                    legend_elements_1 = [Line2D([0], [0], marker=' ', color='w'), Line2D([0], [0], marker=' ', color='w'), Line2D([0], [0], marker=' ', color='w')]
                    labels_1          = ['Stars', 'Gas$_{Total}$', 'Gas$_{SF}$']
                    labels_color_1    = [plt.get_cmap('Reds_r')([0.5]), plt.get_cmap('Blues_r')([0.5]), plt.get_cmap('Greens_r')([0.5])]
                    
                    legend_elements_2 = [Line2D([0], [0], marker=' ', color='w'), Line2D([0], [0], marker=' ', color='w')]
                    labels_2          = ['Gas$_{Total}$', 'Gas$_{SF}$']
                    labels_color_2    = [plt.get_cmap('Blues_r')([0.5]), plt.get_cmap('Greens_r')([0.5])]
                    
                    #axs[0].add_artist(plt.legend(handles=legend_elements_0, labels=labels_0, loc='upper left', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=1))
                    axs[0].legend(handles=legend_elements_05, labels=labels_05, loc='upper left', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=0)
                    axs[1].legend(handles=legend_elements_1, labels=labels_1, loc='upper left', frameon=False, labelspacing=0.1, fontsize=8, labelcolor=labels_color_1, handlelength=0)
                    axs[2].legend(handles=legend_elements_2, labels=labels_2, loc='upper left', frameon=False, labelspacing=0.1, fontsize=8, labelcolor=labels_color_2, handlelength=0)
                    axs[3].legend(loc='upper left', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=0)
                    
                    
                    #if rad_type_plot == 'hmr':
                        #
                    #if rad_type_plot == 'rad':
                        #
                    axs[0].set_xlim(0, 13.5)    
                    axs[3].set_xlabel('Lookback time (Gyr)')
                    axs[0].set_ylim(0, 180)
                    axs[0].set_yticks(np.arange(0, 181, 30))
                    axs[0].set_ylabel('PA misalignment')
                    axs[1].set_ylim(6, 12)
                    axs[1].set_ylabel('M$_{\odot}$ [log$_{10}$]')
                    axs[2].set_ylim(0, 1)
                    axs[2].set_ylabel('Mass fraction')
                    axs[2].set_yticks(np.arange(0, 1.1, 0.25))
                    axs[3].set_ylim(0, 1)
                    axs[3].set_ylabel('kappa')
                    axs[3].set_yticks(np.arange(0, 1.1, 0.25))
                    
                                        
                    axs[0].set_title('GalaxyID: %s' %str(target_GalaxyID))
                    for ax in axs:
                        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
                        

                    # other
                    #axs[0].grid(alpha=0.3)
                    #axs[1].grid(alpha=0.3)
                    plt.gca().invert_xaxis()
                    plt.tight_layout()

                    # savefig
                    if savefig == True:
                        plt.savefig('%s/Radial2D_EVOLUTION_id%s_mass%s_%s_part%s_com%s_ax%s%s.%s' %(str(root_file), str(target_GalaxyID), str('%.2f' %np.log10(float(all_general['%s' %str(target_GalaxyID)]['stelmass']))), angle_type_in, str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt, file_format), format='%s' %file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
                    if showfig == True:
                        plt.show()
                    plt.close()
                       
                    
                # Plots analytical misalignment angle in 3D space
                if plot_2D_3D == '3D':
                    print('  NOT CONFIGURED  ')
                    for GroupNum in GroupNumList:
                        # Initialise figure
                        graphformat(8, 11, 11, 9, 11, 4.5, 3.75)
                        fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(3.15, 3.15), sharex=True, sharey=False)
            
                        plot_count = 0
            
                        for angle_type_in_i in angle_type_in:
                            # Collect values to plot
                            rad_points    = []
                            gas_sf_frac   = []
                            pa_points     = []
                            pa_points_lo  = []
                            pa_points_hi  = []
                            GroupNumPlot       = []
                            GroupNumNotPlot    = []
        
                            # If galaxy not flagged, use galaxy
                            if len(all_flags['%s' %str(GroupNum)]) == 0:
                                for i in np.arange(0, len(all_misangles['%s' %str(GroupNum)]['%s' %rad_type_plot]), 1):
                                    rad_points.append(all_misangles['%s' %str(GroupNum)]['%s' %rad_type_plot][i])
                                    pa_points.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %angle_type_in_i][i])
                    
                                    # lower and higher, where error is [lo, hi] in _misanglesproj[...]
                                    pa_points_lo.append(all_misangles['%s' %str(GroupNum)]['%s_angle_err' %angle_type_in_i][i][0])
                                    pa_points_hi.append(all_misangles['%s' %str(GroupNum)]['%s_angle_err' %angle_type_in_i][i][1])
                        
                                    if plot_count == 0:
                                        # Gas sf fraction
                                        gas_sf_frac.append(all_particles['%s' %str(GroupNum)]['gas_sf_mass'][i]  / all_particles['%s' %str(GroupNum)]['gas_mass'][i])
                        
                    
                                GroupNumPlot.append(GroupNum)
                            else:
                                print('VOID: flagged')
                                print(all_flags['%s' %str(GroupNum)].items())
                                GroupNumNotPlot.append(GroupNum)
            
                
                            if debug == True:
                                print('\nrad ', rad_points)
                                print('proj', pa_points)
                                print('lo', pa_points_lo)
                                print('hi', pa_points_hi)
        
        
        
                            # Plot scatter and errorbars
                            #plt.errorbar(rad_points, pa_points, xerr=None, yerr=pa_points_err, label='2D projected', alpha=0.8, ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                            if angle_type_in_i == 'stars_gas':
                                axs[0].plot(rad_points, pa_points, label='Gas', alpha=1.0, ms=2, lw=1)
                            if angle_type_in_i == 'stars_gas_sf':
                                axs[0].plot(rad_points, pa_points, label='SF gas', alpha=1.0, ms=2, lw=1)
                            axs[0].fill_between(rad_points, pa_points_lo, pa_points_hi, alpha=0.3)
            
                            if plot_count == 0:
                                # Plot star forming fraction
                                axs[1].plot(rad_points, gas_sf_frac, alpha=1.0, lw=1, c='k')
            
           
                    ### General formatting 
                    if rad_type_plot == 'hmr':
                        axs[0].set_xlim(0, max(spin_hmr_in))
                        axs[0].set_xticks(np.arange(0, max(spin_hmr_in)+1, 1))
                        axs[1].set_xlabel('Stellar half-mass radius')
                    if rad_type_plot == 'rad':
                        axs[0].set_xlim(0, max(spin_hmr_in * float(all_general['%s' %str(GroupNum)]['halfmass_rad_proj'])))
                        axs[0].set_xticks(np.arange(0, max(spin_hmr_in * float(all_general['%s' %str(GroupNum)]['halfmass_rad_proj']))+1, 5))
                        axs[1].set_xlabel('Radial distance from centre [pkpc]')
            
                    axs[0].set_ylabel('Stellar-gas misalignment')
                    axs[1].set_ylabel('f$_{gas_{sf}/gas_{tot}}$')
                    axs[0].set_yticks(np.arange(0, 181, 30))
                    axs[1].set_yticks(np.arange(0, 1.1, 0.25))
                    axs[1].set_yticklabels(['0', '', '', '', '1'])
                    axs[0].set_ylim(0, 180)
                    axs[1].set_ylim(0, 1)
                    axs[0].set_title('GalaxyID: %i' %subhalo.GalaxyID)
        
                    # Annotations
        
                    # Legend
                    axs[0].legend(loc='lower right', frameon=False, labelspacing=0.1, fontsize=9, labelcolor='linecolor', handlelength=0)
                    for ax in axs:
                        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
                    # Other
                    plt.tight_layout()
                    axs[0].grid(alpha=0.3)
                    axs[1].grid(alpha=0.3)
        
                    # Savefig
                    if savefig == True:
                        plt.savefig('%s/Radial3D_gn%s_id%s_mass%s_%s_part%s_com%s_%s.%s' %(str(root_file), str(all_general['%s' %str(GroupNum)]['gn']), str(all_general['%s' %str(GroupNum)]['GalaxyID']), str('%.2f' %np.log10(float(all_general['%s' %str(GroupNum)]['stelmass']))), angle_type_in, str(gas_sf_min_particles), str(com_min_distance), savefigtxt, file_format), format='%s' %file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
                    if showfig == True:
                        plt.show()
                    plt.close()
        
                    plot_count = plot_count+1
                
                
            #-------------
            _plot_rad()
            #-------------
       
        
#------------------------  
plot_radial_evolution()
#------------------------  
