import h5py
import numpy as np
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import csv
import json
import time
import math
from datetime import datetime
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from subhalo_main import Subhalo_Extract, Subhalo, ConvertID, ConvertGN
import eagleSqlTools as sql
from graphformat import graphformat


# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   

# Directories of data hdf5 file(s)
dataDir_main = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
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
 
""" 
Will extract all galaxies with specified stellar
mass greater than given, and return gn + sgn
"""
class Sample:
    def __init__(self, sim, snapNum, mstarLimit, satellite):
        # Allowing these attributes to be called from the object
        self.mstar_limit = mstarLimit
        self.sim         = sim
        self.snapNum     = snapNum
        
        myData = self.samplesize(satellite)
        
        self.GroupNum    = myData['GroupNumber']
        self.SubGroupNum = myData['SubGroupNumber']
        
    def samplesize(self, satellite):
        # This uses the eagleSqlTools module to connect to the database with your username and password.
        # If the password is not given, the module will prompt for it.
        con = sql.connect('lms192', password='dhuKAP62')
        
        if satellite == 'yes':
            for sim_name, sim_size in self.sim:
                #print(sim_name)
            
                # Construct and execute query for each simulation. This query returns properties for a single galaxy
                myQuery = 'SELECT \
                             SH.GroupNumber, \
                             SH.SubGroupNumber \
                           FROM \
                             %s_Subhalo as SH, \
                             %s_Aperture as AP \
                           WHERE \
        			         SH.SnapNum = %i \
                             and AP.Mass_Star >= %f \
                             and AP.ApertureSize = 30 \
                             and SH.GalaxyID = AP.GalaxyID \
                           ORDER BY \
        			         AP.Mass_Star desc'%(sim_name, sim_name, self.snapNum, self.mstar_limit)
            
            # Execute query.
            myData = sql.execute_query(con, myQuery)
            
        else:
            for sim_name, sim_size in self.sim:
                #print(sim_name)
            
                # Construct and execute query for each simulation. This query returns properties for a single galaxy
                myQuery = 'SELECT \
                             SH.GroupNumber, \
                             SH.SubGroupNumber \
                           FROM \
                             %s_Subhalo as SH, \
                             %s_Aperture as AP \
                           WHERE \
        			         SH.SnapNum = %i \
                             and AP.Mass_star >= %f \
                             and SH.SubGroupNumber = 0 \
                             and AP.ApertureSize = 30 \
                             and SH.GalaxyID = AP.GalaxyID \
                           ORDER BY \
        			         AP.Mass_Star desc'%(sim_name, sim_name, self.snapNum, self.mstar_limit)
    
            # Execute query.
            myData = sql.execute_query(con, myQuery)

        return myData


""" 
DESCRIPTION
-----------
- Connects to the SQL database to sample galaxies with stellar mass of 10^9 (about 1000 star particles see Casanueva)
- Finds the total angular momentum of a specific particle type within a given radius, outputs this as a table
- Currently just takes ‘2D + axis’ or ‘3D’ as options for angle, with a mass requirement. 
   - Can also manually take a manual_GroupNumList which will override the above. This list is kicked out as a print ‘Subhalos plotted’ from subhalo_pa.py


SAMPLE:
------
	Min. galaxy mass 1e9   (galaxy_mass_limit = 1e9)
	Min. sf particle count of 20 within 2 HMR.  (gas_sf_min_particles = 20)
	2D projected angle within 2 HMR (plot_2D_3D = ‘2D’)
	Viewing axis z
	C.o.M max distance of 2.0 pkpc
        •  	Misangle_2D_stars_gas_sf_rad2_part20_com2.0_axz.jpeg


"""
#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
def plot_misalignment_angle(manual_GalaxyIDList   = [],           # manually enter galaxy IDs we want
                              manual_GroupNumList = [],           # manually enter galaxy gns we want
                              SubGroupNum       = 0,
                              snapNum           = 28,
                                galaxy_mass_limit = 10**9.0,                              # for use in SAMPLE
                            kappa_rad_in        = 30,                               # calculate kappa for this radius [pkpc]
                            aperture_rad_in     = 30,                               # trim all data to this maximum value before calculations
                            align_rad_in        = False,                            # keep on False
                            trim_hmr_in         = np.array([100]),                  # keep as 100... will be capped by aperture anyway. Doesn't matter
                            orientate_to_axis='z',                              # Keep as z
                            viewing_angle = 0,                                    # Keep as 0
                                    find_uncertainties      = False,                    # whether to find 2D and 3D uncertainties
                                    spin_hmr_in             = np.array([2.0]),                  # multiples of hmr. Will use lowest value for spin
                                    viewing_axis            = 'z',                     # Which axis to view galaxy from.  DEFAULT 'z'
                                    com_min_distance        = 2.0,                     # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                                    gas_sf_min_particles    = 20,                     # Minimum gas sf particles to use galaxy.  DEFAULT 100
                                    min_inclination         = 0,                          # Minimum inclination toward viewing axis [deg] DEFAULT 0
                                    projected_or_abs        = 'projected',              # 'projected' or 'abs'
                                    angle_type_in           = np.array(['stars_gas_sf']),       # analytical results for constituent particles will be found. ei., stars_gas_sf will mean stars and gas_sf properties will be found, and the angles between them                                                           
                            plot_single = False,                      # keep on true
                            plot_morphology = True,                     # whether to create separate plots for ETGs/LTGs
                                    plot_2D_3D              = '2D',                     #or use '3D'. DEFAULT 2D
                            root_file = '/Users/c22048063/Documents/EAGLE/plots',
                            file_format = 'pdf',
                              print_galaxy       = False,
                              print_galaxy_short = True,
                              print_progress     = False,
                              csv_load       = False,              # .csv file will ALL data
                                csv_load_name = 'data_misalignment_2023-02-28 10:45:11.358603',       #FIND IN LINUX, mac is weird
                              csv_file           = False,              # .csv file will ALL data
                                csv_name = 'data_misalignment',
                              showfig   = True,
                              savefig   = True,  
                                savefig_txt = '_pdf',            #extra savefile txt
                              debug = False):            
    
    
    time_start = time.time()  
    
    
    #-----------------------------------------
    # Load our sample
    if csv_load:
        
        # Load existing sample
        dict_new = json.load(open(r'%s/%s.csv' %(root_file, csv_load_name), 'r'))
        
        all_misangles       = dict_new['all_misangles']
        all_coms            = dict_new['all_coms']
        all_particles       = dict_new['all_particles']
        all_misanglesproj   = dict_new['all_misanglesproj']
        all_general         = dict_new['all_general']
        all_flags           = dict_new['all_flags']

        # these will all be lists, they need to be transformed into arrays
        print('LOADED CSV:')
        print(dict_new['function_input'])
        
        GroupNumList = all_general.keys()
        SubGroupNumList = np.full(len(GroupNumList), SubGroupNum)
        
    #------------------------------------------
    if not csv_load:
         # use manual input if values given, else use sample with mstar_limit
         if print_progress:
             print('Creating sample')
             time_start = time.time()
           
         if len(manual_GroupNumList) > 0:
             # Converting names of variables for consistency
             GroupNumList    = manual_GroupNumList
             SubGroupNumList = np.full(len(manual_GroupNumList), SubGroupNum)
         elif len(manual_GalaxyIDList) > 0:
             # Extract GroupNum, SubGroupNum, and Snap for each ID
             GroupNumList    = []
             SubGroupNumList = []
             for galID in manual_GalaxyIDList:
                 gn, sgn, snap = ConvertID(galID, mySims)
        
                 # Append to arrays
                 GroupNumList.append(gn)
                 SubGroupNumList.append(sgn)
         else:
             # creates a list of applicable gn (and sgn) to sample. To include satellite galaxies, use 'yes'
             sample = Sample(mySims, snapNum, galaxy_mass_limit, 'no')
             print("Sample length: ", len(sample.GroupNum))
             print("  ", sample.GroupNum)
             
             GroupNumList = sample.GroupNum
             SubGroupNumList = sample.SubGroupNum
    
    
    
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
         all_flags         = {}          # has reason why galaxy failed sample
         all_general       = {}          # has total masses, kappa, halfmassrad
         all_coms          = {}
         all_misangles     = {}          # has all 3d angles
         all_particles     = {}          # has all the particle count and mass within rad
         all_misanglesproj = {}   # has all 2d projected angles from 3d when given a viewing axis and viewing_angle = 0
    
    
         for GroupNum, SubGroupNum in tqdm(zip(GroupNumList, SubGroupNumList), total=len(GroupNumList)): 
        
             if print_progress:
                 print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
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
            
             spin_rad = spin_hmr_in * use_rad                            #pkpc
             spin_hmr = spin_hmr_in
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
                 align_rad = 2*use_rad
             else:
                 align_rad = align_rad_in  
             #------------------------------------------------------------------
        
        
             if print_progress:
                 print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                 print('Running particle data analysis Subhalo()')
                 time_start = time.time()
            
             # If we want the original values, enter 0 for viewing angle
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
                                            
             if len(subhalo.flags) > 0:
                 print('SUBHALO SKIPPED: ', subhalo.flags)
            
        
             #--------------------------------
             # Collecting all relevant particle info for galaxy
             all_general['%s' %str(subhalo.gn)]       = subhalo.general
             all_flags['%s' %str(subhalo.gn)]         = subhalo.flags
             all_particles['%s' %str(subhalo.gn)]     = subhalo.particles
             all_coms['%s' %str(subhalo.gn)]          = subhalo.coms
             all_misangles['%s' %str(subhalo.gn)]     = subhalo.mis_angles
             all_misanglesproj['%s' %str(subhalo.gn)] = subhalo.mis_angles_proj
             #---------------------------------
        
             # Print galaxy properties
             if print_galaxy == True:
                 print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
                 print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))       # [Msun]
                 print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad_proj)             # [pkpc]
                 print('KAPPA:                  %.2f' %subhalo.kappa_stars)
                 print('KAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)
                 print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
                 mask = np.where(np.array(subhalo.coms['hmr'] == min(spin_hmr_in)))
                 print('C.O.M %s HMR STARS-SF [pkpc]:  %.2f' %(str(min(spin_hmr_in)), subhalo.coms['stars_gas_sf'][int(mask[0])]))
             elif print_galaxy_short == True:
                 print('GN:\t%s\t|ID:\t%s\t|HMR:\t%.2f\t|KAPPA:\t%.2f' %(str(subhalo.gn), str(subhalo.GalaxyID), subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'])) 
                 
    
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
                       
             # Combining all dictionaries
             csv_dict = {'all_general': all_general, 'all_misangles': all_misangles, 'all_misanglesproj': all_misanglesproj, 'all_coms': all_coms, 'all_particles': all_particles, 'all_flags': all_flags}
             csv_dict.update({'function_input': str(inspect.signature(plot_misalignment_angle))})
        
             # Writing one massive JSON file
             json.dump(csv_dict, open('%s/%s_%s_%s.csv' %(root_file, str(snapNum), csv_name, str(datetime.now())), 'w'), cls=NumpyEncoder)
        
             # Reading JSON file
             """dict_new = json.load(open('%s/%s.csv' %(root_file, csv_name), 'r'))
             # example nested dictionaries
             new_general = dict_new['all_general']
             new_misanglesproj = dict_new['all_misanglesproj']
             # example accessing function input
             function_input = dict_new['function_input']
             print(dict_new['all_general']['4'].items())"""
        
         
    
        
    #=====================================  
    def _plot_single(quiet=0, debug=False):
        
        raise Exception('Do not use. Works but not tested with updated stuff')
        
        for angle_type_in_i in angle_type_in:
            # Collect values to plot
            misalignment_angle = []
            projected_angle    = []
            GroupNumPlot       = []
            GroupNumNotPlot    = []
            
            
            for GroupNum in GroupNumList:
                # If galaxy not flagged, use galaxy
                if len(all_flags['%s' %str(GroupNum)]) == 0:
                    # Mask correct integer (formatting weird but works)
                    mask_rad = int(np.where(np.array(all_misangles['%s' %GroupNum]['hmr']) == min(spin_hmr_in))[0])
                    
                    misalignment_angle.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %angle_type_in_i][mask_rad])
                    projected_angle.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %angle_type_in_i][mask_rad])
                    
                    GroupNumPlot.append(GroupNum)
                else:
                    GroupNumNotPlot.append(GroupNum)
                    
            
            if debug == True:
                print('\nmisalignment angle: ', misalignment_angle)
                print('projected angle: %s ' %viewing_axis, projected_angle)
                
            
            # Print statements
            print('\nInitial sample:    ', len(GroupNumList))
            if not quiet:
                print(' ', GroupNumList)
            print('\nFinal sample:   ', len(GroupNumPlot))
            if not quiet:
                print(' ', GroupNumPlot)  
            print('\nNot in sample:   ', len(GroupNumNotPlot)) 
            if not quiet:
                print(' ', GroupNumNotPlot)
            print('\n\n==========================================')
            print('Spin radius, axis:   %.1f [hmr], %s' %(min(spin_hmr_in), viewing_axis))
            print('Min. stellar mass:   %.1f [log10 M*]' %np.log10(galaxy_mass_limit))
            print('Min. sf particles:   %i' %gas_sf_min_particles)
            print('Min. c.o.m distance: %.1f [pkpc]' %com_min_distance)
            print('---------------------------------------')
            print('Initial sample:  ', len(GroupNumList))
            print('Final sample:    ', len(GroupNumPlot))
            print('Not in sample:   ', len(GroupNumNotPlot)) 
            print('==========================================')
            
            #------------------------------------------------
            # Graph initialising and base formatting
            graphformat(8, 11, 11, 9, 11, 4.5, 3.75)
            fig, ax = plt.subplots(1, 1, figsize=[3.75, 3.75])
        
            # Labels
            if 'stars_gas' in angle_type_in:
                label = ['Gas']
                plot_color = 'dodgerblue'
            if 'stars_gas_sf' in angle_type_in:
                label = ['SF gas']
                plot_color = 'darkorange'
            if 'stars_gas_nsf' in angle_type_in:
                label = ['NSF gas']
                plot_color = 'indigo'
            
            
            if plot_2D_3D == '2D':
                # Plot data as histogram (outer lines + fill)
                plt.hist(projected_angle, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color, alpha=0.1)
                plt.hist(projected_angle, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color, facecolor='none', alpha=1.0)
                
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(projected_angle, bins=np.arange(0, 181, 10), range=(0, 180))
                plt.errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot), ecolor=plot_color, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
                
            elif plot_2D_3D == '3D':
                # Plot data as histogram
                plt.hist(misalignment_angle, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color, alpha=0.1)
                plt.hist(misalignment_angle, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color, facecolor='none', alpha=1.0)
                
                
                
                # Add poisson errors to each bin (sqrt N)graphformat(8, 11, 11, 9, 11,
                hist_n, _ = np.histogram(misalignment_angle, bins=np.arange(0, 181, 10), range=(0, 180))
                plt.errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot), ecolor=plot_color, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
            
            
            ### General formatting
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, step=30))
            if plot_2D_3D == '2D':
                ax.set_xlabel('Stellar-gas PA misalignment')   #3D projected $\Psi$$_{gas-star}$
            elif plot_2D_3D == '3D':
                ax.set_xlabel('Stellar-gas misalignment')
            ax.set_ylabel('Percentage of galaxies')
            ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
            
            
            """if plot_2D_3D == '2D':
                plt.suptitle("L%s: %s Misalignment\nmass: %.1f, type: 2D, hmr: %s, ax: %s, \nparticles: %s, com: %s, galaxies: %s/%s" %(str(mySims[0][1]), angle_type_in_i, np.log10(galaxy_mass_limit), str(min(spin_hmr_in)), viewing_axis, str(gas_sf_min_particles), str(com_min_distance), len(GroupNumPlot), len(sample.GroupNum)))
            elif plot_2D_3D == '3D':
                plt.suptitle("L%s: %s Misalignment\nmass: %.1f, type: 3D, hmr: %s, \nparticles: %s, com: %s, galaxies: %s/%s" %(str(mySims[0][1]), angle_type_in_i, np.log10(galaxy_mass_limit), str(min(spin_hmr_in)), str(gas_sf_min_particles), str(com_min_distance), len(GroupNumPlot), len(sample.GroupNum)))
            """
            
            # Annotations
            ax.axvline(30, ls='--', lw=1, c='k')
            
            # Legend
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            ax.legend(handles=legend_elements, labels=label, loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor=[plot_color], handlelength=0)
              
            # other 
            plt.tight_layout()
            
            # Savefig
            if savefig == True:
                if plot_2D_3D == '2D':
                    plt.savefig("%s/%s_MisAngleHist_2D_mass%s_%s_rad%s_part%s_com%s_ax%s%s.%s" %(root_file, str(snapNum), np.log10(galaxy_mass_limit), angle_type_in_i, str(min(spin_hmr_in)), str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefig_txt, file_format), format=file_format, bbox_inches='tight', pad_inches=0.2, dpi=300)
                elif plot_2D_3D == '3D':
                    plt.savefig("%s/%s_MisAngleHist_3D_mass%s_%s_rad%s_part%s_com%s%s.%s" %(root_file, str(snapNum), np.log10(galaxy_mass_limit), angle_type_in_i, str(min(spin_hmr_in)), str(gas_sf_min_particles), str(com_min_distance), savefig_txt, file_format), format=file_format, bbox_inches='tight', pad_inches=0.2, dpi=300)
            if showfig == True:
                plt.show()
            plt.close()
            
    #=====================================  
    def _plot_morphology(quiet=0, debug=False):
        for angle_type_in_i in angle_type_in:
            # Collect values to plot
            misalignment_angle_LTG = []
            projected_angle_LTG    = []
            misalignment_angle_ETG = []
            projected_angle_ETG    = []
            GroupNumPlot_LTG       = []
            GroupNumPlot_ETG       = []
            GroupNumNotPlot    = []
            
            
            for GroupNum in GroupNumList:
                # If galaxy not flagged, use galaxy
                if len(all_flags['%s' %str(GroupNum)]) == 0:
                    # Mask correct integer (formatting weird but works)
                    mask_rad = int(np.where(np.array(all_misangles['%s' %GroupNum]['hmr']) == min(spin_hmr_in))[0])
                    
                    if all_general['%s' %str(GroupNum)]['kappa_stars'] >= 0.40:
                        misalignment_angle_LTG.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %angle_type_in_i][mask_rad])
                        projected_angle_LTG.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %angle_type_in_i][mask_rad])
                    
                        GroupNumPlot_LTG.append(GroupNum)
                    else:
                        misalignment_angle_ETG.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %angle_type_in_i][mask_rad])
                        projected_angle_ETG.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %angle_type_in_i][mask_rad])
                    
                        GroupNumPlot_ETG.append(GroupNum)
                else:
                    GroupNumNotPlot.append(GroupNum)
                    
            
            if debug == True:
                print('\nmisalignment angle: ', misalignment_angle)
                print('projected angle: %s ' %viewing_axis, projected_angle)
                
            
            # Print statements
            print('\nInitial sample:    ', len(GroupNumList))
            if not quiet:
                print(' ', GroupNumList)
            print('\nFinal sample ETG:   ', len(GroupNumPlot_ETG))
            if not quiet:
                print(' ', GroupNumPlot_ETG)  
            print('\nFinal sample LTG:   ', len(GroupNumPlot_LTG))
            if not quiet:
                print(' ', GroupNumPlot_LTG)
            print('\nNot in sample:   ', len(GroupNumNotPlot)) 
            if not quiet:
                print(' ', GroupNumNotPlot)
            print('\n\n==========================================')
            print('Spin radius, axis:   %.1f [hmr], %s' %(min(spin_hmr_in), viewing_axis))
            print('Min. stellar mass:   %.1f [log10 M*]' %np.log10(galaxy_mass_limit))
            print('Min. sf particles:   %i' %gas_sf_min_particles)
            print('Min. c.o.m distance: %.1f [pkpc]' %com_min_distance)
            print('---------------------------------------')
            print('Initial sample:  ', len(GroupNumList))
            print('Final sample:    ', len(GroupNumPlot_ETG)+len(GroupNumPlot_LTG))
            print('Not in sample:   ', len(GroupNumNotPlot)) 
            print('==========================================')
            
            #------------------------------------------------
            # Graph initialising and base formatting
            graphformat(8, 9, 9, 9, 9, 4.5, 3.75)
            fig, axs = plt.subplots(1, 2, figsize=[7.5, 2.55], sharex=True, sharey=False)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
            # Labels
            if 'stars_gas' in angle_type_in:
                label = ['Gas']
                plot_color = 'dodgerblue'
            if 'stars_gas_sf' in angle_type_in:
                label_ETG = ['ETG']
                label_LTG = ['LTG']
                plot_color_ETG = 'b'
                plot_color_LTG = 'darkorange'
            if 'stars_gas_nsf' in angle_type_in:
                label = ['NSF gas']
                plot_color = 'indigo'
            
            
            if plot_2D_3D == '2D':
                # ETGs
                # Plot data as histogram (outer lines + fill)
                axs[0].hist(projected_angle_ETG, weights=np.ones(len(GroupNumPlot_ETG))/len(GroupNumPlot_ETG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color_ETG, alpha=0.1)
                bin_count, _, _ = axs[0].hist(projected_angle_ETG, weights=np.ones(len(GroupNumPlot_ETG))/len(GroupNumPlot_ETG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color_ETG, facecolor='none', alpha=1.0)
                print('ETG bin count')
                print(bin_count*100)
                
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(projected_angle_ETG, bins=np.arange(0, 181, 10), range=(0, 180))
                axs[0].errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot_ETG), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot_ETG), ecolor=plot_color_ETG, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
                
                # LTGs
                # Plot data as histogram (outer lines + fill)
                axs[1].hist(projected_angle_LTG, weights=np.ones(len(GroupNumPlot_LTG))/len(GroupNumPlot_LTG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color_LTG, alpha=0.1)
                bin_count, _, _ = axs[1].hist(projected_angle_LTG, weights=np.ones(len(GroupNumPlot_LTG))/len(GroupNumPlot_LTG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color_LTG, facecolor='none', alpha=1.0)
                print('LTG bin count')
                print(bin_count*100)
                
                # Add poisson errors to each bin (sqrt N)
                hist_n, _ = np.histogram(projected_angle_LTG, bins=np.arange(0, 181, 10), range=(0, 180))
                axs[1].errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot_LTG), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot_LTG), ecolor=plot_color_LTG, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
                
            elif plot_2D_3D == '3D':
                # ETGs
                # Plot data as histogram
                axs[0].hist(misalignment_angle_ETG, weights=np.ones(len(GroupNumPlot_ETG))/len(GroupNumPlot_ETG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color_ETG, alpha=0.1)
                axs[0].hist(misalignment_angle_ETG, weights=np.ones(len(GroupNumPlot_ETG))/len(GroupNumPlot_ETG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color_ETG, facecolor='none', alpha=1.0)
                
                # Add poisson errors to each bin (sqrt N)graphformat(8, 11, 11, 9, 11,
                hist_n, _ = np.histogram(misalignment_angle_ETG, bins=np.arange(0, 181, 10), range=(0, 180))
                axs[0].errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot_ETG), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot_ETG), ecolor=plot_color_ETG, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
            
                # LTGs
                # Plot data as histogram
                axs[1].hist(misalignment_angle_LTG, weights=np.ones(len(GroupNumPlot_LTG))/len(GroupNumPlot_LTG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color_LTG, alpha=0.1)
                axs[1].hist(misalignment_angle_LTG, weights=np.ones(len(GroupNumPlot_LTG))/len(GroupNumPlot_LTG), bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color_LTG, facecolor='none', alpha=1.0)
                
                # Add poisson errors to each bin (sqrt N)graphformat(8, 11, 11, 9, 11,
                hist_n, _ = np.histogram(misalignment_angle_LTG, bins=np.arange(0, 181, 10), range=(0, 180))
                axs[1].errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot_LTG), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot_LTG), ecolor=plot_color_LTG, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
            
                
            ### General formatting
            
            for index, ax in enumerate(axs):
                ax.yaxis.set_major_formatter(PercentFormatter(1, symbol=''))
                ax.set_xlim(0, 180)
                #ax.set_ylim(0, 1)
                ax.set_xticks(np.arange(0, 181, step=30))
                if plot_2D_3D == '2D':
                    ax.set_xlabel('Stellar-gas PA misalignment')   #3D projected $\Psi$$_{gas-star}$
                elif plot_2D_3D == '3D':
                    ax.set_xlabel('Stellar-gas misalignment')
                    ax.set_ylabel('Percentage of galaxies')
                ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
                
                
            
                """if plot_2D_3D == '2D':
                    plt.suptitle("L%s: %s Misalignment\nmass: %.1f, type: 2D, hmr: %s, ax: %s, \nparticles: %s, com: %s, galaxies: %s/%s" %(str(mySims[0][1]), angle_type_in_i, np.log10(galaxy_mass_limit), str(min(spin_hmr_in)), viewing_axis, str(gas_sf_min_particles), str(com_min_distance), len(GroupNumPlot), len(sample.GroupNum)))
                elif plot_2D_3D == '3D':
                    plt.suptitle("L%s: %s Misalignment\nmass: %.1f, type: 3D, hmr: %s, \nparticles: %s, com: %s, galaxies: %s/%s" %(str(mySims[0][1]), angle_type_in_i, np.log10(galaxy_mass_limit), str(min(spin_hmr_in)), str(gas_sf_min_particles), str(com_min_distance), len(GroupNumPlot), len(sample.GroupNum)))
                """
            
                # Annotations
                ax.axvline(30, ls='--', lw=1, c='k')
            
                # Legend
                legend_elements = [Line2D([0], [0], marker=' ', color='w')]
                axs[0].legend(handles=legend_elements, labels=label_ETG, loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor=[plot_color_ETG], handlelength=0)
                axs[1].legend(handles=legend_elements, labels=label_LTG, loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor=[plot_color_LTG], handlelength=0)
                
              
            # other 
            plt.tight_layout()
            
            # Savefig
            if savefig == True:
                if plot_2D_3D == '2D':
                    plt.savefig("%s/%s_MisAngleHist_morph_2D_mass%s_%s_rad%s_part%s_com%s_ax%s%s.%s" %(root_file, str(snapNum), np.log10(galaxy_mass_limit), angle_type_in_i, str(min(spin_hmr_in)), str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefig_txt, file_format), format=file_format, bbox_inches='tight', pad_inches=0.2, dpi=300)
                elif plot_2D_3D == '3D':
                    plt.savefig("%s/%s_MisAngleHist_morph_3D_mass%s_%s_rad%s_part%s_com%s%s.%s" %(root_file, str(snapNum), np.log10(galaxy_mass_limit), angle_type_in_i, str(min(spin_hmr_in)), str(gas_sf_min_particles), str(com_min_distance), savefig_txt, file_format), format=file_format, bbox_inches='tight', pad_inches=0.2, dpi=300)
            if showfig == True:
                plt.show()
            plt.close()    
        
            
            
        
        
            
    #-------------------------
    if plot_single == True:
        _plot_single()
    if plot_morphology == True:
        _plot_morphology()
    #------------------------- 
    
    
        
#------------------------  
plot_misalignment_angle()
#------------------------  

