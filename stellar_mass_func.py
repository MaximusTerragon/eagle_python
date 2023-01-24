import h5py
import numpy as np
import pandas as pd
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import csv
import json
from datetime import datetime
from tqdm import tqdm
import eagleSqlTools as sql
from subhalo_main import Subhalo_Extract, Subhalo
from graphformat import graphformat


# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28

# Directories of data hdf5 file(s)
#dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
 

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
        			         SH.SnapNum = %i \
                             and SH.MassType_Star >= %f \
                           ORDER BY \
        			         SH.MassType_Star desc'%(sim_name, snapNum, self.mstar_limit)
            
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


""" 
PURPOSE
-------

Create a stellar mass function plot of a given simulation,
and can optionally import an existing sample group from
csv_load to compare this to.
"""
def _stellar_mass_func(galaxy_mass_limit = 10**9,               # Mass limit of sample
                       sql_mass_limit    = 10**8,               # Mass limit of SQL query    
                        SubGroupNum       = 0,
                               spin_rad_in          = np.array([2.0]),              #np.arange(1, 3, 0.5),    # multiples of hmr
                               use_angle_in         = 2.0,                          # hmr we are interested in filtering multiples of rad
                               gas_sf_min_particles = 20,                           # minimum gas sf particles to use galaxy
                               hist_bin_width       = 0.2,                          # log10 Msun
                               com_min_distance     = 2.0,                          # minimum distance between stars and gas_sf c.o.m
                               plot_angle_type      = np.array(['stars_gas_sf']),   #np.array(['stars_gas', 'stars_gas_sf', 'stars_gas_nsf']),
                         kappa_rad_in         = 30,                               # calculate kappa for this radius [pkpc]
                         angle_selection      = [['stars', 'gas_sf']],            # list of angles to find analytically [[ , ], [ , ] ...]
                         trim_rad_in          = np.array([100]),                  # keep as 100
                         align_rad_in         = False,                        # keep on False
                         orientate_to_axis = 'z',                              # Keep as z
                         viewing_angle = 0,                                    # Keep as 0
                       plot_single     = True,                        # whether to create single plots. KEEP ON TRUE
                         root_file = '/Users/c22048063/Documents/EAGLE/trial_plots',
                         csv_load           = True,              # .csv file will ALL data
                           csv_name = 'data_misalignment_2023-01-23 14:31:20.481086',       #FIND IN LINUX, mac is weird
                         showfig   = True,
                         savefig   = False,  
                           savefigtxt = '',            #extra savefile txt
                         quiet = True,
                         debug = False):
            
                
    sim_name, sim_size = mySims[0]
    
    # Initialise figure
    plt.figure()
    graphformat(8, 11, 11, 11, 11, 3.75, 3)
    
    #-----------------------------------------
    # Construct and execute query for each simulation. This query returns the number of galaxies 
    for sim_name, sim_size in mySims:
        con = sql.connect('lms192', password='dhuKAP62')
        
    	# for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width). 
        myQuery = 'SELECT \
                     %f+floor(log10(SH.MassType_Star)/%f)*%f as mass, \
                     count(*) as num \
                   FROM \
                     %s_SubHalo as SH \
                   WHERE \
			         SH.MassType_Star > %f and \
                     SH.SnapNum = %i \
                   GROUP BY \
			         %f+floor(log10(SH.MassType_Star)/%f)*%f \
                   ORDER BY \
			         mass'%(hist_bin_width/2, hist_bin_width, hist_bin_width, sim_name, sql_mass_limit, snapNum, hist_bin_width/2, hist_bin_width, hist_bin_width)
                    
        # Execute query.
        myData 	= sql.execute_query(con, myQuery)
        
        if not quiet:
            print('SQL number:  ', myData['num'][:])
        
        # Normalize by volume and bin width.
        hist = myData['num'][:] / (float(sim_size))**3.
        hist = hist / hist_bin_width
        
        plt.plot(myData['mass'], np.log10(hist), label=sim_name, linewidth=2)

    #-----------------------------------------
    # Load our sample
    if csv_load:
        
        # Load existing sample
        dict_new = json.load(open(r'%s/%s.csv' %(root_file, csv_name), 'r'))
        
        all_misangles       = dict_new['all_misangles']
        all_coms            = dict_new['all_coms']
        all_particles       = dict_new['all_particles']
        all_misanglesproj   = dict_new['all_general']
        all_general         = dict_new['all_general']

        # these will all be lists, they need to be transformed into arrays
        print('LOADED CSV:')
        print(dict_new['function_input'])
          
    if not csv_load:
        # creates a list of applicable gn (and sgn) to sample. To include satellite galaxies, use 'yes'
        sample = Sample(mySims, snapNum, galaxy_mass_limit, 'no')
        GroupNumList = sample.GroupNum
        
        # Create dictionaries to collect for csv
        all_misangles     = {}
        all_coms          = {}
        all_particles     = {}
        all_misanglesproj = {}
        all_general       = {}
        
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
                                                angle_selection,
                                                viewing_angle,
                                                spin_rad,
                                                trim_rad, 
                                                kappa_rad, 
                                                align_rad,              #align_rad = False
                                                orientate_to_axis,
                                                quiet=True)
            
            
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
            
        #add sample requirements
    
    
    # Collect values to plot
    stelmass_hist      = []
    GroupNumPlot       = []
    GroupNumNotPlot    = []
    
    # Pick first item in list to make hmr masks
    mask_GroupNum = list(all_general.keys())[0]
    
    # Selection criteria
    mask = np.where(np.array(all_misangles['%s' %mask_GroupNum]['hmr']) == use_angle_in)
    for GroupNum in all_general.keys():
        
        # min. sf particle requirement
        mask_sf = np.where(np.array(all_particles['%s' %mask_GroupNum]['hmr']) == use_angle_in)
        if np.array(all_particles['%s' %str(GroupNum)]['gas_sf'])[int(mask_sf[0])] >= gas_sf_min_particles:
            
            # min. com distance requirement
            mask_com = np.where(np.array(all_coms['%s' %mask_GroupNum]['hmr']) == use_angle_in)
            if np.array(all_coms['%s' %str(GroupNum)]['stars_gas_sf'])[int(mask_com[0])] <= com_min_distance:
                stelmass_hist.append(float(all_general['%s' %str(GroupNum)]['stelmass']))
                GroupNumPlot.append(GroupNum)
            
            else:
                GroupNumNotPlot.append([GroupNum, 'com: %.2f' %all_coms['%s' %str(GroupNum)]['stars_gas_sf'][int(mask_com[0])]])
        else:
            GroupNumNotPlot.append([GroupNum, 'sf part: %i' %all_particles['%s' %str(GroupNum)]['gas_sf'][int(mask_sf[0])]])
    
    # Print statements
    if not quiet:
        print('\nFinal sample:   ', len(GroupNumPlot))
        print(' ', GroupNumPlot)  
        print('\nNot in sample:   ', len(GroupNumNotPlot)) 
        print(' ', GroupNumNotPlot)
    else:
        print('---------------------------------------')
        print('Final sample:    ', len(GroupNumPlot))
        print(' ', GroupNumPlot)  
        print('Not in sample:   ', len(GroupNumNotPlot)) 
        print(' ', GroupNumNotPlot)
        print('==========================================')    
    
    
    # Create histogram of sample    
    hist_sample, _ = np.histogram((hist_bin_width/2)+np.floor(np.log10(stelmass_hist)/hist_bin_width)*hist_bin_width , bins=np.arange(np.log10(galaxy_mass_limit)+(hist_bin_width/2), np.log10(10**15), hist_bin_width))
    hist_sample = hist_sample[:] / (float(sim_size))**3
    hist_sample = hist_sample / hist_bin_width                      # why?
    hist_bins   = np.arange(np.log10(galaxy_mass_limit)+(hist_bin_width/2), np.log10(10**15)-hist_bin_width, hist_bin_width)
    
    # Masking out nans
    with np.errstate(divide='ignore', invalid='ignore'):
        hist_mask = np.isfinite(np.log10(hist_sample))
    hist_sample = hist_sample[hist_mask]
    hist_bins   = hist_bins[hist_mask]

    # Plot
    plt.plot(hist_bins, np.log10(hist_sample), label='sample', ls='--', linewidth=2)
    
    
    # Formatting
    plt.xlim(7, 12.5)
    plt.ylim(-5, -0.5)
    plt.xlabel(r'log$_{10}$ M$_{*}$ [M$_{\odot}$]')
    plt.ylabel(r'log$_{10}$ dn/dlog$_{10}$(M$_{*}$) [cMpc$^{-3}$]')
    
    # Title
    plt.title('StellarMassFunc:\n sample vs total')
    plt.tight_layout()
    plt.legend()
    
    # savefig
    if savefig == True:
        plt.savefig('%s/stellarMassFunc_%s.jpeg' %(str(root_file), savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
    if showfig == True:
        plt.show()
    plt.close()
    
#----------------------
_stellar_mass_func()
#----------------------