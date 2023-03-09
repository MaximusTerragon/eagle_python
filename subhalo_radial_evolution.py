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
from astropy.cosmology import z_at_value, FlatLambdaCDM
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
                                 csv_load       = True,              # .csv file will ALL data
                                   csv_load_name = 'data_radial_evolution_2023-03-09 20:13:14.854317',                  #FIND IN LINUX, mac is weird
                             kappa_rad_in    = 30,                          # calculate kappa for this radius [pkpc]    
                             aperture_rad_in = 30,                          # trim all data to this maximum value before calculations
                             trim_hmr_in     = np.array([100]),             # keep as 100, doesn't matter as capped by aperture anyway
                             align_rad_in    = False,                       # keep on False   
                             orientate_to_axis='z',                         # keep as z
                             viewing_angle=0,                               #keep as 0
                                     find_uncertainties      = True,                    # whether to find 2D and 3D uncertainties
                                     viewing_axis            = 'z',                     # Which axis to view galaxy from.  DEFAULT 'z'
                                     com_min_distance        = 2.0,                     # [pkpc] min distance between sfgas and stars. Min radius of spin_rad_in used
                                     gas_sf_min_particles    = 20,                      # Minimum gas sf particles to use galaxy.  DEFAULT 100
                                     min_inclination         = 0,                       # Minimum inclination toward viewing axis [deg] DEFAULT 0
                                     projected_or_abs        = 'projected',              # 'projected' or 'abs'
                                     spin_hmr_in             = np.array([1.0, 1.5, 2.0, 2.5, 3.0]),    # multiples of HMR
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
                                 csv_name = 'data_radial_evolution',          # name of .csv file
                               savefig = False,
                               showfig = True,  
                                 savefigtxt = '', 
                               debug = False):         
          
    time_start = time.time()  
    

    # Load existing data from csv file
    if csv_load:
        # Load existing sample
        dict_new = json.load(open(r'%s/%s.csv' %(root_file, csv_load_name), 'r'))
        
        total_misangles       = dict_new['total_misangles']
        total_coms            = dict_new['total_coms']
        total_particles       = dict_new['total_particles']
        total_misanglesproj   = dict_new['total_misanglesproj']
        total_general         = dict_new['total_general']
        total_flags           = dict_new['total_flags']
        
        total_allbranches     = dict_new['total_allbranches']
        total_mainbranch      = dict_new['total_mainbranch']
        total_mergers         = dict_new['total_mergers']

        # these will all be lists, they need to be transformed into arrays
        print('LOADED CSV:')
        print(dict_new['function_input'])
        
        GalaxyIDList_target = list(total_general.keys())
        
    
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

        # Empty dictionaries to collect relevant data
        total_flags         = {}
        total_general       = {}
        total_coms          = {}
        total_misangles     = {}
        total_particles     = {}
        total_misanglesproj = {}
        total_allbranches = {}
        total_mainbranch  = {}
        total_mergers     = {}
        
        
        #=================================================================== 
        # For each target galaxy we are interested in...
        for target_GroupNum, target_SubGroupNum, target_GalaxyID, target_snapNum in zip(np.array(GroupNumList_target), np.array(SubGroupNumList_target), GalaxyIDList_target, snapNumList_target):
        
            if print_progress:
                print('Extracting merger tree data MergerTree()')
                time_start = time.time()
            
            # Extract merger tree data
            tree = MergerTree(mySims, target_GalaxyID, snapNumMax) 
        
            
            #----------------------------------
            # Update totals with current target_GalaxyID. This will be the single identifier for each galaxy
            # Empty dictionaries are created
            total_flags.update({'%s' %str(tree.target_GalaxyID): {}})
            total_general.update({'%s' %str(tree.target_GalaxyID): {}})
            total_coms.update({'%s' %str(tree.target_GalaxyID): {}})
            total_particles.update({'%s' %str(tree.target_GalaxyID): {}})
            total_misangles.update({'%s' %str(tree.target_GalaxyID): {}})
            total_misanglesproj.update({'%s' %str(tree.target_GalaxyID): {}})
            
            total_allbranches.update({'%s' %str(tree.target_GalaxyID): tree.all_branches})
            total_mainbranch.update({'%s' %str(tree.target_GalaxyID): tree.main_branch})
            total_mergers.update({'%s' %str(tree.target_GalaxyID): tree.mergers})
            #----------------------------------
            
            
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
                 
                trim_hmr = trim_hmr_in
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
                                                        trim_hmr, 
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
                total_general['%s' %str(tree.target_GalaxyID)]['%s' %str(subhalo.GalaxyID)]       = subhalo.general
                total_flags['%s' %str(tree.target_GalaxyID)]['%s' %str(subhalo.GalaxyID)]         = subhalo.flags
                total_particles['%s' %str(tree.target_GalaxyID)]['%s' %str(subhalo.GalaxyID)]     = subhalo.particles
                total_coms['%s' %str(tree.target_GalaxyID)]['%s' %str(subhalo.GalaxyID)]          = subhalo.coms
                total_misangles['%s' %str(tree.target_GalaxyID)]['%s' %str(subhalo.GalaxyID)]     = subhalo.mis_angles
                total_misanglesproj['%s' %str(tree.target_GalaxyID)]['%s' %str(subhalo.GalaxyID)] = subhalo.mis_angles_proj
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
            csv_dict = {'total_flags': total_flags, 'total_general': total_general, 'total_misangles': total_misangles, 'total_misanglesproj': total_misanglesproj, 'total_coms': total_coms, 'total_particles': total_particles, 'total_allbranches': total_allbranches, 'total_mainbranch': total_mainbranch, 'total_mergers': total_mergers}
            csv_dict.update({'function_input': str(inspect.signature(plot_radial_evolution))})

            # Writing one massive JSON file
            json.dump(csv_dict, open('%s/%s_%s.csv' %(root_file, csv_name, str(datetime.now())), 'w'), cls=NumpyEncoder)

            """# Reading JSON file
            dict_new = json.load(open('%s/%s.csv' %(root_file, csv_name), 'r'))
            # example nested dictionaries
            new_general = dict_new['total_general']
            new_misanglesproj = dict_new['total_misanglesproj']
            # example accessing function input
            function_input = dict_new['function_input']"""


    #==========================
    """ Structure of dicts:
    total_flags
        [target_GalaxyID]
            [subhalo.GalaxyID]
                [array of flags if there are any. Len=0 if none]
    
    total_general
        [target_GalaxyID]
            [subhalo.GalaxyID]
                ['GalaxyID']
                ['stelmass']
                ['kappa_stars']
                etc
    total_coms
    total_particles
    total_misangles
    total_misanglesproj
    
    total_allbranches
        [target_GalaxyID]
            ['redshift']
            ['snapnum']
            ['stelmass']
            etc.
    total_mainbranch
    total_mergers
            
    """

    def _plot_rad(quiet=1, debug=False):
        
        # Plots 3D projected misalignment angle from a viewing axis
        if plot_2D_3D == '2D':
            for target_GalaxyID in np.array(GalaxyIDList_target):
                # Initialise figure
                graphformat(8, 11, 11, 9, 11, 6.5, 3.75)
                fig, axs = plt.subplots(nrows=5, ncols=1, gridspec_kw={'height_ratios': [1.3, 0.7, 0.7, 0.7, 0.7]}, figsize=[3.5, 8], sharex=True, sharey=False)                    
    
    
                #=====================
                # Add mergers
                for ratio_i, lookbacktime_i, snap_i in zip(np.array(total_mergers['%s' %str(target_GalaxyID)]['ratios']), np.array(total_mainbranch['%s' %str(target_GalaxyID)]['lookbacktime']), np.array(total_mainbranch['%s' %str(target_GalaxyID)]['snapnum'])):
                    if len(ratio_i) == 0:
                        next
                    else:
                        if max(ratio_i) >= 0.01:
                            for ax in axs:
                                ax.axvline(lookbacktime_i, ls='-', color='grey', alpha=min(0.5, max(ratio_i)*10), linewidth=3)
        
                            # Annotate
                            axs[0].text(lookbacktime_i, 170, ' %.2f' %max(ratio_i), fontsize=7, color='grey')
                
                 
                #---------------------------
                # Loop over each angle type
                for angle_type_in_i in plot_angle_type_in:
                    # angletype_in_i will be dotted for total gas, and line for SF gas
                    if angle_type_in_i == 'stars_gas':
                        ls = '--'
                    elif angle_type_in_i == 'stars_gas_sf':
                        ls = '-'
        
        
                    # Create some colormaps of things we want
                    colors_blues    = plt.get_cmap('Blues_r')(np.linspace(0.05, 0.5, len(spin_hmr_in)))
                    colors_reds     = plt.get_cmap('Reds_r')(np.linspace(0.05, 0.5, len(spin_hmr_in)))
                    colors_greens   = plt.get_cmap('Greens_r')(np.linspace(0.05, 0.5, len(spin_hmr_in)))
                    colors_spectral = plt.get_cmap('Purples_r')(np.linspace(0.05, 0.5, len(spin_hmr_in)))
        
        
                    #-----------------------
                    # Loop over each rad
                    for hmr_i, color_b, color_r, color_g, color_s in zip(spin_hmr_in, colors_blues, colors_reds, colors_greens, colors_spectral):
            
                        GalaxyID_plot     = []
                        GalaxyID_notplot  = []
                        lookbacktime_plot = []
                        redshift_plot     = []
            
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
                        for GalaxyID_i, lookbacktime_i, redshift_i in zip(np.array(total_mainbranch['%s' %str(target_GalaxyID)]['GalaxyID']), np.array(total_mainbranch['%s' %str(target_GalaxyID)]['lookbacktime']), np.array(total_mainbranch['%s' %str(target_GalaxyID)]['redshift'])):
                            
                            if len(np.array(total_flags['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)])) == 0:
                                # Mask correct integer (formatting weird but works)
                                mask_rad = int(np.where(np.array(total_misanglesproj['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)][viewing_axis]['hmr']) == hmr_i)[0])
                                mask_rad = int(np.where(np.array(total_misanglesproj['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)][viewing_axis]['hmr']) == hmr_i)[0])
                
                                misangle_plot.append(total_misanglesproj['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)][viewing_axis]['%s_angle' %angle_type_in_i][mask_rad])
                                misangle_err_lo_plot.append(total_misanglesproj['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)][viewing_axis]['%s_angle_err' %angle_type_in_i][mask_rad][0])
                                misangle_err_hi_plot.append(total_misanglesproj['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)][viewing_axis]['%s_angle_err' %angle_type_in_i][mask_rad][1])
                    
                                stelmass_plot.append(total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['stars_mass'][mask_rad])
                                gasmass_plot.append(total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['gas_mass'][mask_rad])
                                gassfmass_plot.append(total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['gas_sf_mass'][mask_rad])
                    
                                gas_frac_plot.append(total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['gas_mass'][mask_rad]  / (total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['stars_mass'][mask_rad] + total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['gas_mass'][mask_rad]))
                                gas_sf_frac_plot.append(total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['gas_sf_mass'][mask_rad]  / (total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['stars_mass'][mask_rad] + total_particles['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['gas_mass'][mask_rad]))
                    
                                redshift_plot.append(redshift_i)
                                lookbacktime_plot.append(lookbacktime_i)
                                GalaxyID_plot.append(GalaxyID_i)
                    
                            else:
                                print('VOID ID: %s\t\t%s' %(str(GalaxyID_i), str(lookbacktime_i)))
                                print(all_flags['%s' %str(GalaxyID_i)])
                    
                                misangle_plot.append(math.nan)
                                misangle_err_lo_plot.append(math.nan)
                                misangle_err_hi_plot.append(math.nan)
                    
                                stelmass_plot.append(math.nan)
                                gasmass_plot.append(math.nan)
                                gassfmass_plot.append(math.nan)
                    
                                gas_frac_plot.append(math.nan)
                                gas_sf_frac_plot.append(math.nan)
                    
                                redshift_plot.append(redshift_i)
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
                            axs[0].plot(lookbacktime_plot, misangle_plot, label='%s r$_{HMR}$' %str(hmr_i), alpha=0.8, ms=2, lw=1, ls=ls, c=color_s)
                        if angle_type_in_i == 'stars_gas_sf':
                            axs[0].plot(lookbacktime_plot, misangle_plot, label='%s r$_{HMR}$' %str(hmr_i), alpha=0.8, ms=2, lw=1, ls=ls, c=color_s)
                        axs[0].fill_between(lookbacktime_plot, misangle_err_lo_plot, misangle_err_hi_plot, alpha=0.2, color=color_s)
            
            
                        #------------------------
                        # PLOT 2
                        # Plot masses
                        axs[1].plot(lookbacktime_plot, np.log10(stelmass_plot), alpha=1.0, lw=1, c=color_r, label='Stars')
                        #axs[1].plot(lookbacktime_plot, np.log10(gasmass_plot), alpha=1.0, lw=1, c=color_b, label='Gas$_{Total}$')
                        axs[1].plot(lookbacktime_plot, np.log10(gassfmass_plot), alpha=1.0, lw=1, c=color_g, label='Gas$_{SF}$')
            
            
                        #------------------------
                        # PLOT 3
                        # Plot gas frations
                        axs[2].plot(lookbacktime_plot, gas_frac_plot, alpha=1.0, lw=1, c=color_b, label='%s r$_{HMR}$' %str(hmr_i))
                        axs[3].plot(lookbacktime_plot, gas_sf_frac_plot, alpha=1.0, lw=1, c=color_g, label='%s r$_{HMR}$' %str(hmr_i))
        
                #------------------------
                # PLOT 4
    
                # Find kappas and other _general stats
                kappa_stars       = []
                kappa_gas         = []
    
                # Same as taking 'for redshift in XX'
                for GalaxyID_i, lookbacktime_i in zip(np.array(total_mainbranch['%s' %str(target_GalaxyID)]['GalaxyID']), np.array(total_mainbranch['%s' %str(target_GalaxyID)]['lookbacktime'])):
                    if len(total_flags['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]) == 0:
                        kappa_stars.append(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['kappa_stars'])
                        kappa_gas.append(total_general['%s' %str(target_GalaxyID)]['%s' %str(GalaxyID_i)]['kappa_gas_sf'])
                    else:
                        kappa_stars.append(math.nan)
                        kappa_gas.append(math.nan)
        
    
                # Plot kappas
                axs[4].axhline(0.4, lw=1, ls='--', c='grey', alpha=0.7)
                axs[4].text(13.4, 0.41, ' LTG', fontsize=7, color='grey')
                axs[4].text(13.4, 0.29, ' ETG', fontsize=7, color='grey')
                axs[4].plot(lookbacktime_plot, kappa_stars, alpha=1.0, lw=1, c='r', label='\u03BA$_{Stars}$')
                axs[4].plot(lookbacktime_plot, kappa_gas, alpha=1.0, lw=1, c='b', label='\u03BA$_{Gas}$')
    
                #=============================
                ### General formatting 
    
                ### Customise legend labels
                legend_elements_1 = [Line2D([0], [0], marker=' ', color='w'), Line2D([0], [0], marker=' ', color='w')]
                labels_1          = ['Stars', 'Gas$_{SF}$']
                labels_color_1    = [plt.get_cmap('Reds_r')([0.5]), plt.get_cmap('Greens_r')([0.5])]
    
                #axs[0].add_artist(plt.legend(handles=legend_elements_0, labels=labels_0, loc='upper left', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=1))
                axs[0].legend(loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=0)
                axs[1].legend(handles=legend_elements_1, labels=labels_1, loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor=labels_color_1, handlelength=0)
                axs[2].legend(loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=0)
                axs[3].legend(loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=0)
                axs[4].legend(loc='lower right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor='linecolor', handlelength=0)
    
    
                # Create redshift axis:
                redshiftticks = [0, 0.2, 0.5, 1, 1.5, 2, 5, 10, 20]
                ageticks = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(redshiftticks)).value
                for i, ax in enumerate(axs):
                    ax_top = ax.twiny()
                    ax_top.set_xticks(ageticks)

                    ax.set_xlim(0, 13.5)
                    ax_top.set_xlim(0, 13.5)
        
                    if i == 0:
                        ax.set_ylim(0, 180)
            
                        ax_top.set_xlabel('Redshift')
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', length=3, width=0.8)
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor', length=1.8, width=0.8)
            
                        ax.set_yticks(np.arange(0, 181, 30))
                        ax_top.set_xticklabels(['{:g}'.format(z) for z in redshiftticks])
                        ax.set_ylabel('PA misalignment')
            
                        ax.set_title('GalaxyID: %s' %str(target_GalaxyID))
                        ax.invert_xaxis()
                        ax_top.invert_xaxis()
                    if i == 1:
                        ax.set_ylim(7, 13)
                        ax.set_ylabel('Mass [log$_{10}$M$_{\odot}$]')
            
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', length=3, width=0.8, labelbottom=False, labeltop=False)
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor', length=1.8, width=0.8)
                        ax.invert_xaxis()
                        ax_top.invert_xaxis()
                    if i == 2:
                        ax.set_ylim(0, 1)
                        ax.set_yticks(np.arange(0, 1.1, 0.25))
            
                        ax.set_ylabel('Gas mass\nfraction')
            
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', length=3, width=0.8, labelbottom=False, labeltop=False)
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor', length=1.8, width=0.8)
                        ax.invert_xaxis()
                        ax_top.invert_xaxis()
                    if i == 3:
                        ax.set_ylim(0, 1)
                        ax.set_yticks(np.arange(0, 1.1, 0.25))
            
                        ax.set_ylabel('Gas$_{SF}$ mass\nfraction')
            
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', length=3, width=0.8, labelbottom=False, labeltop=False)
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor', length=1.8, width=0.8)
                        ax.invert_xaxis()
                        ax_top.invert_xaxis()
                    if i == 4:
                        ax.set_ylim(0, 1)
                        ax.set_yticks(np.arange(0, 1.1, 0.25))
            
                        ax.set_xlabel('Lookback time (Gyr)')
                        ax.set_ylabel('\u03BA')
            
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='major', length=3, width=0.8, labelbottom=False, labeltop=False)
                        ax_top.tick_params(axis='both', direction='in', top=True, bottom=False, left=False, right=False, which='minor', length=1.8, width=0.8)
                        ax.invert_xaxis()
                        ax_top.invert_xaxis()
        
                    ax.minorticks_on()
                    ax.tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='major', width=0.8, length=3)
                    ax.tick_params(axis='both', direction='in', top=False, bottom=True, left=True, right=True, which='minor', length=1.8, width=0.8)
        
                  
    
                # other
                #axs[0].grid(alpha=0.3)
                #axs[1].grid(alpha=0.3)
                plt.tight_layout()

                # savefig
                if savefig == True:
                    plt.savefig('%s/Radial2D_EVOLUTION_id%s_mass%s_%s_part%s_com%s_ax%s%s.%s' %(str(root_file), str(target_GalaxyID), str('%.2f' %np.log10(float(total_general['%s' %str(target_GalaxyID)]['%s' %str(target_GalaxyID)]['stelmass']))), angle_type_in, str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt, file_format), format='%s' %file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
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
