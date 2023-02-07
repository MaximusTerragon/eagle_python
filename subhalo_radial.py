import h5py
import numpy as np
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import astropy.units as u
import csv
import json
import time
import math
from datetime import datetime
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from subhalo_main import Subhalo_Extract, Subhalo
import eagleSqlTools as sql
from graphformat import graphformat


# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28

# Directories of data hdf5 file(s)
#dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
 

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
#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
def plot_radial_misalignment(manual_GroupNumList = np.array([1]),           # manually enter galaxy gns we want
                               SubGroupNum = 0,
                               galaxy_mass_limit = 10**9.5,                         # for print 
                             kappa_rad_in    = 30,                          # calculate kappa for this radius [pkpc]    
                             aperture_rad_in = 30,                          # trim all data to this maximum value before calculations
                             trim_rad_in     = np.array([100]),             # keep as 100, doesn't matter as capped by aperture anyway
                             align_rad_in    = False,                       # keep on False   
                             orientate_to_axis='z',                         # keep as z
                             viewing_angle=0,                               #keep as 0
                                     find_uncertainties      = True,                    # whether to find 2D and 3D uncertainties
                                     spin_rad_in             = np.arange(0.25, 10.01, 0.25),    # multiples of rad
                                     viewing_axis            = 'z',                     # Which axis to view galaxy from.  DEFAULT 'z'
                                     com_min_distance        = 2.0,                     # [pkpc] min distance between sfgas and stars. Min radius of spin_rad_in used
                                     gas_sf_min_particles    = 20,                     # Minimum gas sf particles to use galaxy.  DEFAULT 100
                                     angle_type_in           = np.array(['stars_gas', 'stars_gas_sf']),       # analytical results for constituent particles will be found. ei., stars_gas_sf will mean stars and gas_sf properties will be found, and the angles between them                                                           
                            plot_single = True,                    # whether to create single plots. Keep on TRUE
                                     plot_2D_3D           = '2D',                # whether to plot 2D or 3D angle
                                     rad_type_plot        = 'hmr',               # 'rad' whether to use absolute distance or hmr 
                             root_file = '/Users/c22048063/Documents/EAGLE/plots',
                             file_format = 'png',
                               print_galaxy       = False,
                               print_galaxy_short = True,
                               print_progress     = True,
                               csv_file = False,                     # whether to create a csv file of used data
                                 csv_name = 'data_radial',          # name of .csv file
                               savefig = True,
                               showfig = True,  
                                 savefigtxt = '', 
                               debug = False):         
          
    time_start = time.time()   
    
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
        
    
    #=================================================================== 
    GroupNumList = manual_GroupNumList
    for GroupNum in tqdm(GroupNumList):
        
        if print_progress:
            print('Extracting particle data Subhalo_Extract()')
            time_start = time.time()
        
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)
        
        #-------------------------------------------------------------------
        # Automating some later variables to avoid putting them in manually
        spin_rad = spin_rad_in * galaxy.halfmass_rad
        spin_rad = [x for x in spin_rad if x <= aperture_rad_in]
        if len(spin_rad) != len(spin_rad_in):
            print('Capped spin_rad (%s pkpc) at aperture radius (%s pkpc)' %(max(spin_rad), aperture_rad_in))
            
        trim_rad = trim_rad_in
        aperture_rad = aperture_rad_in
            
        if kappa_rad_in == 'rad':
            kappa_rad = galaxy.halfmass_rad
        elif kappa_rad_in == 'tworad':
            kappa_rad = 2*galaxy.halfmass_rad
        else:
            kappa_rad = kappa_rad_in
        if align_rad_in == 'rad':
            align_rad = galaxy.halfmass_rad
        elif align_rad_in == 'tworad':
            align_rad = 2*galaxy.halfmass_rad
        else:
            align_rad = align_rad_in  
        #------------------------------------------------------------------
        
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Running particle data analysis Subhalo()')
            time_start = time.time()
            
        # Galaxy will be rotated to calc_kappa_rad's stellar spin value
        with np.errstate(divide='ignore', invalid='ignore'):
            subhalo = Subhalo(galaxy.gn, galaxy.sgn, galaxy.stelmass, galaxy.gasmass, galaxy.GalaxyID, galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas,
                                                angle_selection,
                                                viewing_angle,
                                                spin_rad,
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
                                                quiet=True)
        
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
            print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad)             # [pkpc]
            print('KAPPA:                  %.2f' %subhalo.kappa)
            print('KAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)
            print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
            mask = np.where(np.array(subhalo.coms['hmr'] == min(spin_rad_in)))
            print('C.O.M %s HMR STARS-SF [pkpc]:  %.2f' %(str(min(spin_rad_in)), subhalo.coms['stars_gas_sf'][int(mask[0])]))
        elif print_galaxy_short == True:
            print('GN:\t%s\t|HMR:\t%.2f\t|KAPPA / SF:\t%.2f  %.2f' %(str(subhalo.gn), subhalo.halfmass_rad, subhalo.kappa, subhalo.kappa_gas_sf)) 
                
        
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
            
        #===================================================================
        # Plot for a single galaxy showing how misalignment angle varies with increasing radius
        def _plot_single(quiet=1, debug=False):
            
            # Initialise figure
            graphformat(8, 11, 11, 9, 11, 3.15, 3.15)
            fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(3.15, 3.15), sharex=True, sharey=False)
            
            
            # Plots 3D projected misalignment angle from a viewing axis
            if plot_2D_3D == '2D':
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
                    
                    
                    for GroupNum in GroupNumList:
                        # If galaxy not flagged, use galaxy
                        if len(all_flags['%s' %str(GroupNum)]) == 0:
                            for i in np.arange(0, len(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s' %rad_type_plot]), 1):
                                rad_points.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s' %rad_type_plot][i])
                                pa_points.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle' %angle_type_in_i][i])
                            
                                # lower and higher, where error is [lo, hi] in _misanglesproj[...]
                                pa_points_lo.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %angle_type_in_i][i][0])
                                pa_points_hi.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %angle_type_in_i][i][1])
                                
                                if plot_count == 0:
                                    # Gas sf fraction
                                    gas_sf_frac.append(all_particles['%s' %str(subhalo.gn)]['gas_sf_mass'][i]  / all_particles['%s' %str(subhalo.gn)]['gas_mass'][i])
                                
                            
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
                    
                    plot_count = plot_count+1
                    
                
                ### General formatting 
                if rad_type_plot == 'hmr':
                    axs[0].set_xlim(0, max(spin_rad_in))
                    axs[0].set_xticks(np.arange(0, max(spin_rad_in)+1, 1))
                    axs[1].set_xlabel('Stellar half-mass radius')
                if rad_type_plot == 'rad':
                    axs[0].set_xlim(0, max(spin_rad_in*subhalo.halfmass_rad))
                    axs[0].set_xticks(np.arange(0, max(spin_rad_in*subhalo.halfmass_rad)+1, 5))
                    axs[1].set_xlabel('Radial distance from centre [pkpc]')
            
                axs[0].set_ylabel('Stellar-gas PA misalignment')
                axs[1].set_ylabel('f$_{gas_{sf}/gas_{tot}}$')
                axs[0].set_yticks(np.arange(0, 181, 30))
                axs[1].set_yticks(np.arange(0, 1.1, 0.25))
                axs[1].set_yticklabels(['0', '', '', '', '1'])
                axs[0].set_ylim(0, 180)
                axs[1].set_ylim(0, 1)
                #axs[0].set_title('Radial 2D\nGroupNum %s: %s, particles: %i, com: %.1f, ax: %s' %(str(subhalo.gn), angle_type_in, gas_sf_min_particles, com_min_distance, viewing_axis))
                
                axs[0].set_title('GalaxyID: %i' %subhalo.GalaxyID)
                axs[0].legend(loc='lower right', frameon=False, labelspacing=0.1, fontsize=9, labelcolor='linecolor', handlelength=0)
                for ax in axs:
                    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
                
                # other
                axs[0].grid(alpha=0.3)
                axs[1].grid(alpha=0.3)
                plt.tight_layout()
                
                # savefig
                if savefig == True:
                    plt.savefig('%s/Radial2D_gn%s_id%s_mass%s_%s_part%s_com%s_ax%s%s.%s' %(str(root_file), str(subhalo.gn), str(subhalo.GalaxyID), str('%.2f' %np.log10(subhalo.stelmass)), angle_type_in, str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt, file_format), format='%s' %file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
                if showfig == True:
                    plt.show()
                plt.close()
                
                
            # Plots analytical misalignment angle in 3D space
            if plot_2D_3D == '3D':
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
                    
                    
                    for GroupNum in GroupNumList:
                        # If galaxy not flagged, use galaxy
                        if len(all_flags['%s' %str(GroupNum)]) == 0:
                            for i in np.arange(0, len(all_misangles['%s' %str(subhalo.gn)]['%s' %rad_type_plot]), 1):
                                rad_points.append(all_misangles['%s' %str(subhalo.gn)]['%s' %rad_type_plot][i])
                                pa_points.append(all_misangles['%s' %str(subhalo.gn)]['%s_angle' %angle_type_in_i][i])
                            
                                # lower and higher, where error is [lo, hi] in _misanglesproj[...]
                                pa_points_lo.append(all_misangles['%s' %str(subhalo.gn)]['%s_angle_err' %angle_type_in_i][i][0])
                                pa_points_hi.append(all_misangles['%s' %str(subhalo.gn)]['%s_angle_err' %angle_type_in_i][i][1])
                                
                                if plot_count == 0:
                                    # Gas sf fraction
                                    gas_sf_frac.append(all_particles['%s' %str(subhalo.gn)]['gas_sf_mass'][i]  / all_particles['%s' %str(subhalo.gn)]['gas_mass'][i])
                                
                            
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
                    
                    plot_count = plot_count+1
                    
                
                ### General formatting 
                if rad_type_plot == 'hmr':
                    axs[0].set_xlim(0, max(spin_rad_in))
                    axs[0].set_xticks(np.arange(0, max(spin_rad_in)+1, 1))
                    axs[1].set_xlabel('Stellar half-mass radius')
                if rad_type_plot == 'rad':
                    axs[0].set_xlim(0, max(spin_rad_in*subhalo.halfmass_rad))
                    axs[0].set_xticks(np.arange(0, max(spin_rad_in*subhalo.halfmass_rad)+1, 5))
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
                    plt.savefig('%s/Radial3D_gn%s_id%s_mass%s_%s_part%s_com%s_%s.%s' %(str(root_file), str(subhalo.gn), str(subhalo.GalaxyID), str('%.2f' %np.log10(subhalo.stelmass)), angle_type_in, str(gas_sf_min_particles), str(com_min_distance), savefigtxt, file_format), format='%s' %file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
                if showfig == True:
                    plt.show()
                plt.close()
                
                
            
    #-------------------------
    if plot_single == True:
        _plot_single()
    #-------------------------
      

    
                
                
          
        
#------------------------  
plot_radial_misalignment()
#------------------------  
