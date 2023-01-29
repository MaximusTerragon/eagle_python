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
                             %s_Subhalo as SH, \
                             %s_Aperture as AP \
                           WHERE \
        			         SH.SnapNum = %i \
                             and AP.Mass_Star >= %f \
                             and AP.ApertureSize = 30 \
                             and SH.GalaxyID = AP.GalaxyID \
                           ORDER BY \
        			         AP.Mass_Star desc'%(sim_name, sim_name, snapNum, self.mstar_limit)
            
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
        			         AP.Mass_Star desc'%(sim_name, sim_name, snapNum, self.mstar_limit)
    
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
                       kappa_rad_in        = 30,                               # calculate kappa for this radius [pkpc]
                       aperture_rad_in     = 30,                               # trim all data to this maximum value before calculations
                       align_rad_in        = False,                            # keep on False
                       trim_rad_in         = np.array([100]),                  # keep as 100... will be capped by aperture anyway. Doesn't matter
                       orientate_to_axis='z',                              # Keep as z
                       viewing_angle = 0,                                    # Keep as 0
                               find_uncertainties      = False,                         # whether to find 2D and 3D uncertainties
                               spin_rad_in             = np.array([2.0]),               # SAMPLE multiples of hmr. Will use lowest value for spin
                               viewing_axis            = 'z',                           # SAMPLE Which axis to view galaxy from.  DEFAULT 'z'
                               com_min_distance        = 2.0,                           # SAMPLE [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                               gas_sf_min_particles    = 20,                            # SAMPLE Minimum gas sf particles to use galaxy.  DEFAULT 100
                               angle_type_in           = np.array(['stars_gas_sf']),    # SAMPLE analytical results for constituent particles will be found. ei., stars_gas_sf will mean stars and gas_sf properties will be found, and the angles between them                                                           
                       plot_single     = True,                        # whether to create single plots. KEEP ON TRUE
                               hist_bin_width          = 0.2,         # Msun
                         print_progress = True,
                         root_file = '/Users/c22048063/Documents/EAGLE/plots',
                         file_format = 'png',
                         csv_load       = False,              # .csv file will ALL data
                           csv_name = 'data_misalignment_2023-01-23 14:31:20.481086',       #FIND IN LINUX, mac is weird
                         showfig   = True,
                         savefig   = True,  
                           savefigtxt = '',            #extra savefile txt
                         quiet = True,
                         debug = False):
            
                
    sim_name, sim_size = mySims[0]
    
    # Initialise figure
    # Graph initialising and base formatting
    graphformat(8, 11, 11, 11, 11, 3.15, 3.15)
    fig, ax = plt.subplots(1, 1, figsize=[3.15, 2.9])
    
    #-----------------------------------------
    # Construct and execute query for each simulation. This query returns the number of galaxies 
    for sim_name, sim_size in mySims:
        con = sql.connect('lms192', password='dhuKAP62')
        
    	# for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width). 
        myQuery = 'SELECT \
                     %f+floor(log10(AP.Mass_Star)/%f)*%f as mass, \
                     count(*) as num \
                   FROM \
                     %s_SubHalo as SH, \
                     %s_Aperture as AP \
                   WHERE \
                     SH.SnapNum = %i \
			         and AP.Mass_Star >= %f \
                     and AP.ApertureSize = 30 \
                     and SH.GalaxyID = AP.GalaxyID \
                   GROUP BY \
			         %f+floor(log10(AP.Mass_Star)/%f)*%f \
                   ORDER BY \
			         mass'%(hist_bin_width/2, hist_bin_width, hist_bin_width, sim_name, sim_name, snapNum, sql_mass_limit, hist_bin_width/2, hist_bin_width, hist_bin_width)
                    
        # Execute query.
        myData 	= sql.execute_query(con, myQuery)
        
        if not quiet:
            print('SQL number:  ', myData['num'][:])
        
        # Normalize by volume and bin width.
        hist = myData['num'][:] / (float(sim_size))**3.
        hist = hist / hist_bin_width
        
        ax.plot(myData['mass'], np.log10(hist), label=sim_name, linewidth=1, c='indigo')

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
        all_flags           = dict_new['all_general']

        # these will all be lists, they need to be transformed into arrays
        print('LOADED CSV:')
        print(dict_new['function_input'])
          
    if not csv_load:
        # creates a list of applicable gn (and sgn) to sample. To include satellite galaxies, use 'yes'
        sample = Sample(mySims, snapNum, galaxy_mass_limit, 'no')
        GroupNumList = sample.GroupNum
        
        # Create dictionaries to collect for csv
        all_flags         = {}
        all_misangles     = {}
        all_coms          = {}
        all_particles     = {}
        all_misanglesproj = {}
        all_general       = {}
        
        for GroupNum in tqdm(GroupNumList):
            # Initial extraction of galaxy data
            galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)
            
            #-------------------------------------------------------------------
            # Automating some later variables to avoid putting them in manually
    
            # Use spin_rad_in as a way to trim data. This variable swap is from older version but allows future use of trim_rad_in
            trim_rad_in = spin_rad_in
    
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
            
            spin_rad = spin_rad_in * galaxy.halfmass_rad        #pkpc
            trim_rad = trim_rad_in                              #rads
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
            
            # Galaxy will be rotated to calc_kappa_rad's stellar spin value
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
        
            
            #-------------------------------------------------------------
            # Assign particle data once per galaxy from extraction
            all_general['%s' %str(subhalo.gn)]       = subhalo.general
            all_flags['%s' %str(subhalo.gn)]         = subhalo.flags
            all_particles['%s' %str(subhalo.gn)]     = subhalo.particles
            all_coms['%s' %str(subhalo.gn)]          = subhalo.coms
            all_misangles['%s' %str(subhalo.gn)]     = subhalo.mis_angles
            all_misanglesproj['%s' %str(subhalo.gn)] = subhalo.mis_angles_proj
                
            #-------------------------------------------------------------------    
            
            
        #add sample requirements
    
    
    # Collect values to plot
    stelmass_hist      = []
    GroupNumPlot       = []
    GroupNumNotPlot    = []
    
    for GroupNum in GroupNumList:
        # If galaxy not flagged, use galaxy
        if len(all_flags['%s' %str(GroupNum)]) == 0:
            # If not flagged in extraction process...
            stelmass_hist.append(float(all_general['%s' %str(GroupNum)]['stelmass']))
            GroupNumPlot.append(GroupNum)
            
        else:
            GroupNumNotPlot.append(GroupNum)
    
    # Print statements
    if not quiet:
        print('\nFinal sample:   ', len(GroupNumPlot))
        print(' ', GroupNumPlot)  
        print('\nNot in sample:   ', len(GroupNumNotPlot)) 
        print(' ', GroupNumNotPlot)
    else:
        print('---------------------------------------')
        print('Initial sample:  ', len(GroupNumList))
        print(' ', GroupNumList)
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


    #------------------------------------------------    
    # Plot
    ax.plot(hist_bins, np.log10(hist_sample), label='Sample selection', ls='--', linewidth=1, c='r')
    
    ### General formatting
    plt.xlim(7, 12.5)
    plt.ylim(-5, -0.5)
    plt.xlabel(r'log$_{10}$ M$_{*}$ [M$_{\odot}$]')
    plt.ylabel(r'log$_{10}$ dn/dlog$_{10}$(M$_{*}$) [cMpc$^{-3}$]')
    plt.xticks(np.arange(7, 12.5, 1))
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
    
    # Annotations
    
    # Legend
    ax.legend(loc='upper right', frameon=False, labelspacing=0.1, fontsize=9, labelcolor='linecolor', handlelength=0)
    
    # Other
    plt.tight_layout()
    
    # Savefig
    if savefig == True:
        plt.savefig('%s/stellarMassFunc_%s.%s' %(str(root_file), savefigtxt, file_format), format=file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
    if showfig == True:
        plt.show()
    plt.close()
    
#----------------------
_stellar_mass_func()
#----------------------