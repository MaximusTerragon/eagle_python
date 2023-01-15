import h5py
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from subhalo_main import Subhalo_Extract, Subhalo
import eagleSqlTools as sql
from graphformat import graphformat

# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'

# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28

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
                             %s_Subhalo as SH \
                           WHERE \
        			         SH.SnapNum = 28 \
                             and SH.MassType_Star >= %f \
                           ORDER BY \
        			         SH.MassType_Star desc'%(sim_name, self.mstar_limit)
            
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
                             %s_Subhalo as SH \
                           WHERE \
        			         SH.SnapNum = 28 \
                             and SH.MassType_Star >= %f \
                             and SH.SubGroupNumber = 0 \
                           ORDER BY \
        			         SH.MassType_Star desc'%(sim_name, self.mstar_limit)
    
            # Execute query.
            myData = sql.execute_query(con, myQuery)

        return myData


def plot_misalignment_angle(galaxy_mass_limit = 1e9,
                            manual_GroupNumList = [],           # manually enter galaxy gns we want
                              use_angle_in    = 2.0,              # multiples of rad
                            plot_angle_type = np.array(['stars_gas_sf']),  #np.array(['stars_gas', 'stars_gas_sf', 'stars_gas_nsf']),
                            plot_2D_3D      = '2D',             #or use '3D'. DEFAULT 2D
                            viewing_axis    = 'z',              #will ignore if '3D' above. DEFAULT z
                              gas_sf_min_particles = 20,         # minimum gas sf particles to use galaxy
                              com_min_distance  = 2.0,              # minimum distance between stars and gas_sf c.o.m
                            plot_single     = True,                        # whether to create single plots
                                savefig   = False,
                                showfig   = True,  
                                savefig_txt = '',            #extra savefile txt                    
                            spin_rad_in     = np.array([2.0]), #np.arange(1, 3, 0.5),    # multiples of rad
                            trim_rad_in     = np.array([100]),                 # keep as 100
                            kappa_rad_in    = 30,                              # calculate kappa for this radius [pkpc]
                            align_rad_in    = False,                           # keep on False              
                            root_file = 'trial_plots',
                              print_galaxy       = False,
                              print_galaxy_short = False,
                            orientate_to_axis = 'z',     # keep as z
                            viewing_angle = 0):            # keep as 0
                            
    # creates a list of applicable gn (and sgn) to sample. To include satellite galaxies, use 'yes'
    sample = Sample(mySims, snapNum, galaxy_mass_limit, 'no')
    
    all_misangles     = {}
    all_coms          = {}
    all_particles     = {}
    all_misanglesproj = {}
    all_general       = {}
    
    if len(manual_GroupNumList) > 0:
        GroupNumList = manual_GroupNumList
    else:
        GroupNumList = sample.GroupNum
    
    SubGroupNum = 0
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
        subhalo = Subhalo(galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas,
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
        
        all_general = {'%s' %str(subhalo.gn): {'stelmass':[], 'gasmass':[], 'gasmass_sf':[], 'gasmass_nsf':[], 'halfmass_rad':[], 'kappa':[], 'kappa_gas':[], 'kappa_gas_sf':[], 'kappa_gas_nsf':[]}}
        
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
        
        
    def _plot_single(quiet=1):
        for plot_angle_type_i in plot_angle_type:
            # Collect values to plot
            misalignment_angle = []
            projected_angle    = []
            GroupNumPlot       = []
            GroupNumNotPlot    = []
            
            # Selection criteria
            mask = np.where(all_misangles['%s' %str(GroupNumList[0])]['hmr'] == use_angle_in)
            for GroupNum in GroupNumList:
                
                # min. sf particle requirement
                mask_sf = np.where(all_particles['%s' %str(GroupNumList[0])]['hmr'] == use_angle_in)
                if all_particles['%s' %str(GroupNum)]['gas_sf'][int(mask_sf[0])] >= gas_sf_min_particles:
                    
                    # min. com distance requirement
                    mask_com = np.where(all_coms['%s' %str(GroupNumList[0])]['hmr'] == use_angle_in)
                    if all_coms['%s' %str(GroupNum)]['stars_gas_sf'][int(mask_com[0])] <= com_min_distance:
                        misalignment_angle.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %plot_angle_type_i][int(mask[0])])
                        projected_angle.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %plot_angle_type_i][int(mask[0])])
                        GroupNumPlot.append(GroupNum)
                    
                    else:
                        GroupNumNotPlot.append([GroupNum, 'com: %.2f' %all_coms['%s' %str(GroupNum)]['stars_gas_sf'][int(mask_com[0])]])
                else:
                    GroupNumNotPlot.append([GroupNum, 'sf part: %i' %all_particles['%s' %str(GroupNum)]['gas_sf'][int(mask_sf[0])]])
        
            
            # Print statements
            print('\nInitial sample:    ', len(GroupNumList))
            print(' ', GroupNumList)
            print('\nFinal sample:   ', len(GroupNumPlot))
            print(' ', GroupNumPlot)  
            print('\nNot in sample:   ', len(GroupNumNotPlot)) 
            print(' ', GroupNumNotPlot)
            print('\n\n==========================================')
            print('Spin radius, axis:   %.1f [pkpc], %s' %(use_angle_in, viewing_axis))
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
            graphformat(8, 11, 11, 11, 11, 5, 5)
            fig, ax = plt.subplots(1, 1, figsize=[8, 4])
        
            
            #GROUPNUMLIST USED AS SAMPLE SIZE, CHANGE THIS
            if plot_2D_3D == '2D':
                # Plot data as histogram
                plt.hist(projected_angle, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='b', facecolor='none', alpha=0.8, label=plot_angle_type_i)
                hist_n, _ = np.histogram(projected_angle, bins=np.arange(0, 181, 10), range=(0, 180))
            
                # Add poisson errors to each bin (sqrt N)
                plt.errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot), ecolor='k', ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
                
            elif plot_2D_3D == '3D':
                # Plot data as histogram
                plt.hist(misalignment_angle, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='b', facecolor='none', alpha=0.8, label=plot_angle_type_i)
                hist_n, _ = np.histogram(misalignment_angle, bins=np.arange(0, 181, 10), range=(0, 180))
            
                # Add poisson errors to each bin (sqrt N)
                plt.errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot), ecolor='k', ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
            
            
            if not quiet:
                print('\nmisalignment angle: ', misalignment_angle)
                print('projected angle: %s ' %viewing_axis, projected_angle)
            
            # General formatting
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 190, step=30))
            if plot_2D_3D == '2D':
                ax.set_xlabel('3D projected $\Psi$$_{gas-star}$')
            elif plot_2D_3D == '3D':
                ax.set_xlabel('3D $\Psi$$_{gas-star}$')
            ax.set_ylabel('Percentage of galaxies')
            ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    
            # Annotations
            ax.axvline(30, ls='--', lw=0.5, c='k')
            if plot_2D_3D == '2D':
                plt.suptitle("L%s: %s Misalignment\ntype: 2D, hmr: %s, ax: %s, particles: %s, com: %s, galaxies: %s/%s" %(str(mySims[0][1]), plot_angle_type_i, str(use_angle_in), viewing_axis, str(gas_sf_min_particles), str(com_min_distance), len(GroupNumPlot), len(sample.GroupNum)))
            elif plot_2D_3D == '3D':
                plt.suptitle("L%s: %s Misalignment\ntype: 3D, hmr: %s, particles: %s, com: %s, galaxies: %s/%s" %(str(mySims[0][1]), plot_angle_type_i, str(use_angle_in), str(gas_sf_min_particles), str(com_min_distance), len(GroupNumPlot), len(sample.GroupNum)))
            plt.legend()
    
            if savefig == True:
                if plot_2D_3D == '2D':
                    plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/Misangle_2D_%s_rad%s_part%s_com%s_ax%s%s.jpeg" %(plot_angle_type_i, str(int(use_angle_in)), str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefig_txt), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
                elif plot_2D_3D == '3D':
                    plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/Misangle_3D_%s_rad%s_part%s_com%s%s.jpeg" %(plot_angle_type_i, str(int(use_angle_in)), str(gas_sf_min_particles), str(com_min_distance), savefig_txt), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
            if showfig == True:
                plt.show()
            plt.close()
            

    #-------------------------
    if plot_single == True:
        _plot_single()
    #-------------------------    
        
#------------------------  
plot_misalignment_angle()
#------------------------  

