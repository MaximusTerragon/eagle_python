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
- When fed a single or list of galaxies into manual_GroupNumList, will plot the misalignment angle between plot_angle_type (default is stars_gas_sf) for a radial distribution given by spin_rad_in
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
def plot_radial_misalignment(manual_GroupNumList = np.array([4]),           # manually enter galaxy gns we want
                               SubGroupNum = 0,
                               galaxy_mass_limit            = 10**9.5,                         # for print 
                                     spin_rad_in            = np.arange(0.5, 10.1, 0.25),    # multiples of rad
                                     kappa_rad_in           = 30,                          # calculate kappa for this radius [pkpc]     
                                     trim_rad_in            = np.array([100]),             # keep as 100
                                     angle_selection        = [['stars', 'gas'], ['stars', 'gas_sf']],       # list of angles to find analytically [[ , ], [ , ] ...]
                               align_rad_in      = False,                           # keep on False   
                               orientate_to_axis = 'z',                             # keep as z
                               viewing_angle     = 0,                               #keep as 0  
                             plot_single = True,                    # whether to create single plots. Keep on TRUE
                               viewing_axis         = 'z',          # DEFUALT z
                               com_min_distance     = 10.0,         # minimum distance between stars and gas_sf c.o.m
                               gas_sf_min_particles = 20,           # minimum gas sf particles to use galaxy
                               plot_2D_3D           = '2D',                # whether to plot 2D or 3D angle
                                     plot_angle_type        = ['stars_gas_sf', 'stars_gas'],         
                                     rad_type_plot          = 'hmr',               # 'rad' whether to use absolute distance or hmr 
                             root_file = '/Users/c22048063/Documents/EAGLE/trial_plots',
                               print_galaxy       = False,
                               print_galaxy_short = True,
                               csv_file = False,                     # whether to create a csv file of used data
                                 csv_name = 'data_radial',          # name of .csv file
                               savefig = True,
                               showfig = True,  
                                 savefigtxt = '_withgas', 
                               debug = False):         
                            
    # create dictionaries
    all_misangles     = {}
    all_coms          = {}
    all_particles     = {}
    all_misanglesproj = {}
    all_general       = {}
    
    GroupNumList = manual_GroupNumList
    
    for GroupNum in tqdm(GroupNumList):
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)
        
        # Detect keywords of rad, tworad. All in [pkpc]
        spin_rad = spin_rad_in * galaxy.halfmass_rad
        trim_rad = trim_rad_in
            
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
    
        # Galaxy will be rotated to calc_kappa_rad's stellar spin value
        with np.errstate(divide='ignore', invalid='ignore'):
            subhalo = Subhalo(galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas,
                                            angle_selection,
                                            viewing_angle,
                                            spin_rad,
                                            trim_rad, 
                                            kappa_rad, 
                                            align_rad,              #align_rad = False
                                            orientate_to_axis,
                                            quiet=True)
        
        # Print galaxy properties
        if print_galaxy == True:
            print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
            print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))       # [Msun]
            print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad)             # [pkpc]
            print('KAPPA:                  %.2f' %subhalo.kappa)
            print('KAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)
            print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
            mask = np.where(np.array(subhalo.coms['hmr'] == use_angle_in))
            print('C.O.M %s HMR STARS-SF [pkpc]:  %.2f' %(str(use_angle_in), subhalo.coms['stars_gas_sf'][int(mask[0])]))
            print(' HALF-\tANGLES (STARS-)\t\tPARTICLE COUNT\t\t\tMASS')
            print(' RAD\tGAS\tSF\tNSF\tSF-NSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
            for i in np.arange(0, len(spin_rad_in), 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    print(' %.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.mis_angles['hmr'][i], subhalo.mis_angles['stars_gas_angle'][i], subhalo.mis_angles['stars_gas_sf_angle'][i], subhalo.mis_angles['stars_gas_nsf_angle'][i], subhalo.mis_angles['gas_sf_gas_nsf_angle'][i], subhalo.particles['stars'][i], subhalo.particles['gas'][i], subhalo.particles['gas_sf'][i], subhalo.particles['gas_nsf'][i], np.log10(subhalo.particles['stars_mass'][i]), np.log10(subhalo.particles['gas_mass'][i]), np.log10(subhalo.particles['gas_sf_mass'][i]), np.log10(subhalo.particles['gas_nsf_mass'][i])))        
            print('CENTRE [pMpc]:      [%.5f,\t%.5f,\t%.5f]' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))        # [pkpc]
            print('PERC VEL [pkm/s]:   [%.5f,\t%.5f,\t%.5f]' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))  # [pkm/s]
            #print('VIEWING ANGLES: ', end='')
        elif print_galaxy_short == True:
            print('GN:\t%s\t|HMR:\t%.2f\t|KAPPA / SF:\t%.2f  %.2f' %(str(subhalo.gn), subhalo.halfmass_rad, subhalo.kappa, subhalo.kappa_gas_sf)) 
            
        
        #--------------------------------
        # Collecting all relevant particle info for galaxy
        all_misangles['%s' %str(GroupNum)] = subhalo.mis_angles
        all_coms['%s' %str(GroupNum)] = subhalo.coms
        all_particles['%s' %str(GroupNum)] = subhalo.particles
        all_misanglesproj['%s' %str(subhalo.gn)] = subhalo.mis_angles_proj
        
        all_general.update({'%s' %str(subhalo.gn): {'gn':[], 'stelmass':[], 'gasmass':[], 'gasmass_sf':[], 'gasmass_nsf':[], 'halfmass_rad':[], 'kappa':[], 'kappa_gas':[], 'kappa_gas_sf':[], 'kappa_gas_nsf':[]}})
        
        all_general['%s' %str(subhalo.gn)]['gn']            = subhalo.gn
        all_general['%s' %str(subhalo.gn)]['stelmass']      = subhalo.stelmass
        all_general['%s' %str(subhalo.gn)]['gasmass']       = subhalo.gasmass
        all_general['%s' %str(subhalo.gn)]['gasmass_sf']    = subhalo.gasmass_sf
        all_general['%s' %str(subhalo.gn)]['gasmass_nsf']   = subhalo.gasmass_nsf
        all_general['%s' %str(subhalo.gn)]['halfmass_rad']  = subhalo.halfmass_rad
        all_general['%s' %str(subhalo.gn)]['kappa']         = subhalo.kappa
        all_general['%s' %str(subhalo.gn)]['kappa_gas']     = subhalo.kappa_gas
        all_general['%s' %str(subhalo.gn)]['kappa_gas_sf']  = subhalo.kappa_gas_sf
        all_general['%s' %str(subhalo.gn)]['kappa_gas_nsf'] = subhalo.kappa_gas_nsf
        #---------------------------------
        
        
        # Plot for a single galaxy showing how misalignment angle varies with increasing radius
        def _plot_single(quiet=1, debug=False):
            
            # Initialise figure
            graphformat(8, 11, 11, 11, 11, 3.75, 3)
            fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(4.5, 5), sharex=True, sharey=False)
            
            axs[0].set_xlim(min(spin_rad_in), max(spin_rad_in))
            axs[0].set_xticks(np.arange(min(spin_rad_in), max(spin_rad_in)+1, 1))
            
            
            # formatting 
            if rad_type_plot == 'hmr':
                axs[1].set_xlabel('Halfmass rad')
            elif rad_type_plot == 'rad':
                axs[1].set_xlabel('Distance from centre [pkpc]')
            axs[0].set_ylabel('3D projected angle')
            axs[1].set_ylabel('f$_{gas_{sf}/gas_{tot}}$')
            axs[0].set_ylim(0, 180)
            axs[1].set_ylim(0, 1)
            
            
            # Plots 3D projected misalignment angle from a viewing axis
            if plot_2D_3D == '2D':
                plot_count = 0
                for plot_angle_type_i in plot_angle_type:
                    # Collect values to plot
                    rad_points    = []
                    gas_sf_frac   = []
                    pa_points     = []
                    pa_points_lo  = []
                    pa_points_hi  = []
                    
                    for i in np.arange(0, len(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s' %rad_type_plot]), 1):
                    
                        # min. sf particle requirement
                        if all_particles['%s' %str(GroupNum)]['gas_sf'][i] >= gas_sf_min_particles:
                    
                            # min. com distance requirement
                            if all_coms['%s' %str(GroupNum)]['stars_gas_sf'][i] <= com_min_distance:
                                rad_points.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s' %rad_type_plot][i])
                                pa_points.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle' %plot_angle_type_i][i])
                            
                                # lower and higher, where error is [lo, hi] in _misanglesproj[...]
                                pa_points_lo.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %plot_angle_type_i][i][0])
                                pa_points_hi.append(all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %plot_angle_type_i][i][1])
                                
                                if plot_count == 0:
                                    # Gas sf fraction
                                    gas_sf_frac.append(all_particles['%s' %str(subhalo.gn)]['gas_sf_mass'][i]  / all_particles['%s' %str(subhalo.gn)]['gas_mass'][i])
                
                    if debug == True:
                        print('\nrad ', rad_points)
                        print('proj', pa_points)
                        print('lo', pa_points_lo)
                        print('hi', pa_points_hi)
                
                    # Plot scatter and errorbars
                    #plt.errorbar(rad_points, pa_points, xerr=None, yerr=pa_points_err, label='2D projected', alpha=0.8, ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                    if plot_angle_type_i == 'stars_gas':
                        axs[0].plot(rad_points, pa_points, label='Total gas', alpha=1.0, ms=2, lw=1)
                    if plot_angle_type_i == 'stars_gas_sf':
                        axs[0].plot(rad_points, pa_points, label='Star-forming gas', alpha=1.0, ms=2, lw=1)
                    axs[0].fill_between(rad_points, pa_points_lo, pa_points_hi, alpha=0.3)
                    
                    if plot_count == 0:
                        # Plot star forming fraction
                        axs[1].plot(rad_points, gas_sf_frac)
                    
                    plot_count = plot_count+1
                    
                
                # Formatting 
                axs[0].set_title('Radial 2D\nGroupNum %s: %s, particles: %i, com: %.1f, ax: %s' %(str(subhalo.gn), plot_angle_type, gas_sf_min_particles, com_min_distance, viewing_axis))
                axs[0].legend()
                plt.tight_layout()
                
                # savefig
                if savefig == True:
                    plt.savefig('%s/Radial2D_gn%s_%s_part%s_com%s_ax%s%s.jpeg' %(str(root_file), str(subhalo.gn), plot_angle_type, str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
                if showfig == True:
                    plt.show()
                plt.close()
                
                
            # Plots analytical misalignment angle in 3D space
            if plot_2D_3D == '3D':
                # Collect values to plot
                rad_points    = []
                pa_points     = []
                pa_points_lo  = []
                pa_points_hi  = []
                
                for i in np.arange(0, len(all_misangles['%s' %str(subhalo.gn)][viewing_axis]['%s' %rad_type_plot]), 1):
                    
                    # min. sf particle requirement
                    if all_particles['%s' %str(GroupNum)]['gas_sf'][i] >= gas_sf_min_particles:
                    
                        # min. com distance requirement
                        if all_coms['%s' %str(GroupNum)]['stars_gas_sf'][i] <= com_min_distance:
                            rad_points    = all_misangles['%s' %str(subhalo.gn)][viewing_axis]['%s' %rad_type_plot][i]
                            pa_points     = all_misangles['%s' %str(subhalo.gn)][viewing_axis]['%s_angle' %plot_angle_type][i]
                            
                            # lower and higher, where error is [lo, hi] in _misanglesproj[...]
                            pa_points_lo.append(all_misangles['%s' %str(subhalo.gn)]['%s_angle_err' %plot_angle_type][i][0])
                            pa_points_hi.append(all_misangles['%s' %str(subhalo.gn)]['%s_angle_err' %plot_angle_type][i][1])
                        
                
                if debug == True:
                    print(all_misangles['%s' %str(subhalo.gn)][viewing_axis].items())
                    print(all_particles['%s' %str(GroupNum)]['gas_sf'])
                    
                    print('\nrad ', rad_points)
                    print('proj', pa_points)
                    print('lo', pa_points_lo)
                    print('hi', pa_points_hi)
                
                
                # Plot scatter and errorbars
                #plt.errorbar(rad_points, pa_points, xerr=None, yerr=pa_points_err, label='3D projected', alpha=0.8, ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                plt.plot(rad_points, pa_points, label='3D misalignment', alpha=1.0, ms=2, lw=1)
                plt.fill_between(rad_points, pa_points_lo, pa_points_hi, alpha=0.3, facecolor='grey')
                
                # Formatting 
                plt.title('Radial 3D\nGroupNum %s: %s, particles: %i, com: %.1f' %(str(subhalo.gn), plot_angle_type, gas_sf_min_particles, com_min_distance))
                plt.legend()
                plt.tight_layout()
                
                # savefig
                if savefig == True:
                    plt.savefig('%s/Radial3D_gn%s_%s_part%s_com%s_%s.jpeg' %(str(root_file), str(subhalo.gn), plot_angle_type, str(gas_sf_min_particles), str(com_min_distance), savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
                if showfig == True:
                    plt.show()
                plt.close()
            
    #-------------------------
    if plot_single == True:
        _plot_single()
    #-------------------------
      

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
        csv_dict = {'all_general': all_general, 'all_misangles': all_misangles, 'all_misanglesproj': all_misanglesproj, 'all_coms': all_coms, 'all_particles': all_particles}
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
                
                
          
        
#------------------------  
plot_radial_misalignment()
#------------------------  
