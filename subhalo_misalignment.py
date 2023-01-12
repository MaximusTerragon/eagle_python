import h5py
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
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
                            plot_2D_3D      = '2D',             #or use '3D'
                            viewing_axis    = 'x',              #will ignore if '3D' above
                              plot_single     = True,                        # whether to create single plots
                              plot_together   = False,                       # or overlapping
                                savefig   = False,
                                showfig   = True,  
                                savefig_txt = '_nolim',            #extra savefile txt                    
                            spin_rad_in     = np.arange(1, 3, 0.5),    # multiples of rad
                            trim_rad_in     = np.array([100]),                 # keep as 100
                            kappa_rad_in    = 30,                              # calculate kappa for this radius [pkpc]
                            align_rad_in    = False,                           # keep on False              
                            root_file = 'trial_plots',
                            print_galaxy  = True,
                            orientate_to_axis = 'z',     # keep as z
                            viewing_angle = 0):            # keep as 0
                            
    # creates a list of applicable gn (and sgn) to sample. To include satellite galaxies, use 'yes'
    sample = Sample(mySims, snapNum, galaxy_mass_limit, 'no')
    print('Number of subhalos in sample with M > %e: %i\n' %(galaxy_mass_limit, len(sample.GroupNum)))   
    
    all_misangles = {}
    all_particles = {}
    all_misanglesproj = {}
    all_general = {}
    
    if len(manual_GroupNumList) > 0:
        GroupNumList = manual_GroupNumList
    else:
        GroupNumList = sample.GroupNum
    
    SubGroupNum = 0
    for GroupNum in GroupNumList:
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
            print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
            print(' HALF-\tANGLES (STARS-)\t\tPARTICLE COUNT\t\t\tMASS')
            print(' RAD\tGAS\tSF\tNSF\tSF-NSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
            for i in np.arange(0, len(spin_rad_in), 1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    print(' %.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.mis_angles['hmr'][i], subhalo.mis_angles['stars_gas_angle'][i], subhalo.mis_angles['stars_gas_sf_angle'][i], subhalo.mis_angles['stars_gas_nsf_angle'][i], subhalo.mis_angles['gas_sf_gas_nsf_angle'][i], subhalo.particles['stars'][i], subhalo.particles['gas'][i], subhalo.particles['gas_sf'][i], subhalo.particles['gas_nsf'][i], np.log10(subhalo.particles['stars_mass'][i]), np.log10(subhalo.particles['gas_mass'][i]), np.log10(subhalo.particles['gas_sf_mass'][i]), np.log10(subhalo.particles['gas_nsf_mass'][i])))        
            print('CENTRE [pMpc]:      [%.5f,\t%.5f,\t%.5f]' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))        # [pkpc]
            print('PERC VEL [pkm/s]:   [%.5f,\t%.5f,\t%.5f]' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))  # [pkm/s]
            #print('VIEWING ANGLES: ', end='')
        
        
        #--------------------------------
        # Collecting all relevant particle info for galaxy
        all_misangles['%s' %str(GroupNum)] = subhalo.mis_angles
        all_particles['%s' %str(GroupNum)] = subhalo.particles
        all_misanglesproj['%s' %str(subhalo.gn)] = subhalo.mis_angles_proj
        
        all_general = {'%s' %str(subhalo.gn): {'stelmass':[], 'gasmass':[], 'gasmass_sf':[], 'gasmass_nsf':[], 'halfmass_rad':[], 'kappa':[]}}
        
        all_general['%s' %str(subhalo.gn)]['stelmass']     = subhalo.stelmass
        all_general['%s' %str(subhalo.gn)]['gasmass']      = subhalo.gasmass
        all_general['%s' %str(subhalo.gn)]['gasmass_sf']   = subhalo.gasmass_sf
        all_general['%s' %str(subhalo.gn)]['gasmass_nsf']  = subhalo.gasmass_nsf
        all_general['%s' %str(subhalo.gn)]['halfmass_rad'] = subhalo.halfmass_rad
        all_general['%s' %str(subhalo.gn)]['kappa']        = subhalo.kappa
        #---------------------------------
        
    # Print statements pre-extra conditions or filters
    print('Initial sample:', GroupNumList)
    
    def _plot_single(quiet=1):
        for plot_angle_type_i in plot_angle_type:
            # Collect values to plot
            misalignment_angle = []
            mask = np.where(all_misangles['1']['hmr'] == use_angle_in)
            for GroupNum in GroupNumList:
                misalignment_angle.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %plot_angle_type_i][int(mask[0])])
        
            # Graph initialising and base formatting
            graphformat(8, 11, 11, 11, 11, 5, 5)
            fig, ax = plt.subplots(1, 1, figsize=[8, 4])
        
            # Plot data as histogram
            #MISALIGNMENT_ANGLE USED AS SAMPLE SIZE, CHANGE THIS
            plt.hist(misalignment_angle, weights=np.ones(len(misalignment_angle))/len(misalignment_angle), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='b', facecolor='none', alpha=0.8, label=plot_angle_type_i)
            hist_n, _ = np.histogram(misalignment_angle, bins=np.arange(0, 181, 10), range=(0, 180))
            
            # Add poisson errors to each bin (sqrt N)
            plt.errorbar(np.arange(5, 181, 10), hist_n/len(misalignment_angle), xerr=None, yerr=np.sqrt(hist_n)/len(misalignment_angle), ecolor='k', ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
            
            
            # General formatting
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 190, step=30))
            ax.set_xlabel('3D $\Psi$$_{gas-star}$')
            ax.set_ylabel('Percentage of Galaxies')
            ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    
            # Annotations
            ax.axvline(30, ls='--', lw=0.5, c='k')
            plt.suptitle("L%s: %s Misalignment\ntype: 3D, hmr: %s, galaxies: %s/%s" %(str(mySims[0][1]), plot_angle_type_i, str(use_angle_in), len(GroupNumList), len(sample.GroupNum)))
            plt.legend()
    
            if savefig == True:
                plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/Misangle_3D_%s_rad%s%s.jpeg" %(plot_angle_type_i, str(int(use_angle_in)), savefig_txt), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
            if showfig == True:
                plt.show()
            plt.close()
            
    def _plot_together(quiet=1):
        # Graph initialising and base formatting
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig, ax = plt.subplots(1, 1, figsize=[8, 4])
        
        for plot_angle_type_i in plot_angle_type:        
            # Collect values to plot
            misalignment_angle = []
            mask = np.where(all_misangles['1']['hmr'] == use_angle_in)
            for GroupNum in GroupNumList:
                misalignment_angle.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %plot_angle_type_i][int(mask[0])])
        
            # Plot data as histogram
            plt.hist(misalignment_angle, bins=np.arange(0, 181, 10), histtype='bar', edgecolor='black', alpha=0.5, label=plot_angle_type_i)
        
        # General formatting
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 190, step=30))
        ax.set_ylim(0, 9)
        ax.set_xlabel('3D $\Psi$$_{gas-star}$')
        ax.set_ylabel('Number')
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    
        # Annotations
        ax.axvline(30, ls='--', lw=0.5, c='k')
        plt.suptitle("L%s: Gas - Star Misalignments, type: 3D, hmr: %s" %(str(mySims[0][1]), str(use_angle_in)))
        plt.legend()
    
        if savefig == True:
            plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/Misangle_3D_rad%s%s.jpeg" %(str(int(use_angle_in)), savefig_txt), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
        if showfig == True:
            plt.show()
        plt.close()
    
    #-------------------------
    if plot_single == True:
        _plot_single()
    if plot_together == True:
        _plot_together()
    #-------------------------    
        
#------------------------  
plot_misalignment_angle()
#------------------------  

