import h5py
import numpy as np
import math
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
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
from graphformat import graphformat, set_rc_params


# Directories
EAGLE_dir       = '/Users/c22048063/Documents/EAGLE'
dataDir_main    = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
# Directories serpens
#EAGLE_dir       = '/home/user/c22048063/Documents/EAGLE'
#dataDir_main   = '/home/universe/spxtd1-shared/RefL0100N1504/'


# Other directories
sample_dir      = EAGLE_dir + '/samples'
output_dir      = EAGLE_dir + '/outputs'
fig_dir         = EAGLE_dir + '/plots'

# Directories of data hdf5 file(s)
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
PURPOSE
-------
- Can visualise a galaxy as a scatter graph in 3D and render side-on images of it
- Can do this for X particles or all of them
- Render boxsize uses fixed aperture (say 100pkpc), but can also trim particle data to within x*HMR
- Galaxy is automatically centred, and velocity adjusted
- Can also orientate the galaxy to a given axis, based on the angular momentum vector calculated at a given distance. This is used in the calculation of kappa at 30pkpc

"""
#1, 2, 4, 3, 6, 5, 7, 9, 16, 14, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
#3748
#37445, 37446, 37447
#3748, 20455, 37445, 30494, 43163, 40124, 44545, 48383, 57647, 55343, 51640, 46366, 53904, 52782, 56522, 59467, 49986, 61119, 62355, 63199, 61831


def galaxy_render(csv_sample = False,              # False, Whether to read in existing list of galaxies  
                    #--------------------------
                    mySims = [('RefL0012N0188', 12)],
                    GalaxyID_List = [37445],
                    #--------------------------
                    # Galaxy extraction properties
                    kappa_rad            = 30,          # calculate kappa for this radius [pkpc]
                    viewing_angle        = 0,           # Keep as 0
                    #--------------------------
                    # Visualisation properties
                    boxradius           = 30,                  # boxradius of render [kpc], 'rad', 'tworad'
                    particles           = 5000,
                    viewing_axis        = 'z',                  # Which axis to view galaxy from.  DEFAULT 'z'
                    aperture_rad        = 30,                   # trim all data to this maximum value before calculations [pkpc]
                    trim_hmr            = np.array([30]),           # WILL PLOT LOWEST VALUE. trim particles # multiples of hmr
                    align_rad           = False,                          # True/False
                    #=====================================================
                    # Misalignments we want extracted and at which radii  
                    angle_selection     = ['stars_gas',                     # stars_gas     stars_gas_sf    stars_gas_nsf
                                           'stars_gas_sf',                  # gas_dm        gas_sf_dm       gas_nsf_dm
                                           'stars_gas_nsf',                 # gas_sf_gas_nsf
                                           'gas_sf_gas_nsf',
                                           'stars_dm'],           
                    spin_hmr            = np.array([0.2]),                  # multiples of hmr for which to find spin. Will plot lowest value
                    rad_projected       = True,                             # whether to use rad in projection or 3D
                    #--------------------------
                    # Plot options
                    plot_spin_vectors   = True,
                    centre_of_pot       = True,                             # Plot most bound object (default centre)
                    centre_of_mass      = False,                            # Plot total centre of mass
                    axis                = True,                             # Plot small axis below galaxy
                    #--------------------------
                    # Particle properties
                    stars               = True,
                    gas                 = False,
                    gas_sf              = True,
                    gas_nsf             = True,    
                    dark_matter         = False,
                    black_holes         = True,
                    #=====================================================
                    showfig      = True,
                    savefig      = False,
                      savefigtxt = '',                # added txt to append to end of savefile
                    #--------------------------
                    print_progress = False,
                    print_galaxy   = True,
                    debug = False):


    #=========================================
    # Properties that don't change
    spin_vector_rad      = spin_hmr[0]
    min_inclination      = 0           # Minimum inclination toward viewing axis [deg] DEFAULT 0
    find_uncertainties   = False       # whether to find 2D and 3D uncertainties
    com_min_distance     = 10000       # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
    min_particles        = 0           # Minimum gas sf particles to use galaxy.  DEFAULT 100
    orientate_to_axis    ='z'          # Keep as z
                  
    if print_progress:
        print('Extracting GroupNum, SubGroupNum, SnapNum lists')
        time_start = time.time()
        
    
    #-----------------------------------------
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
        
    #---------------------------------------
    # If no csv_sample given, use GalaxyID_List
    else:
        # Extract GroupNum, SubGroupNum, and Snap for each ID
        GroupNum_List    = []
        SubGroupNum_List = []
        SnapNum_List     = []
        Redshift_List    = []
        for galID in GalaxyID_List:
            gn, sgn, snap, z = ConvertID(galID, mySims)
    
            # Append to arrays
            GroupNum_List.append(gn)
            SubGroupNum_List.append(sgn)
            SnapNum_List.append(snap)
            Redshift_List.append(z)
            
        if debug:
            print(GroupNum_List)
            print(SubGroupNum_List)
            print(GalaxyID_List)
            print(SnapNum_List)
            
        print('\n===================')
        print('SAMPLE INPUT:\n  %s\n  GalaxyIDs: %s' %(mySims, GalaxyID_List))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('===================')
              
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))

    # Run analysis for each individual galaxy in loaded sample
    for GroupNum, SubGroupNum, GalaxyID, SnapNum in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List), total=len(GroupNum_List)):
        
        #-----------------------------
        if print_progress:
            print('Extracting particle data Subhalo_Extract()')
            time_start = time.time()
            
        # Initial extraction of galaxy particle data
        galaxy = Subhalo_Extract(mySims, dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, aperture_rad, viewing_axis)
        # Gives: galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, galaxy.halo_mass
        
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
            
            # Reduce spin_rad array if value exceeds aperture_rad... means not all dictionaries will have same number of array spin values
            spin_rad = [x for x in spin_rad if x <= aperture_rad]
            spin_hmr = [x for x in spin_hmr if x*galaxy.halfmass_rad_proj <= aperture_rad]
            
            if len(spin_hmr) != len(spin_hmr_tmp):
                print('Capped spin_rad (%s pkpc) at aperture radius (%s pkpc)' %(spin_rad, aperture_rad))
        elif rad_projected == False:
            spin_rad = np.array(spin_hmr) * galaxy.halfmass_rad
            spin_hmr_tmp = spin_hmr
            
            # Reduce spin_rad array if value exceeds aperture_rad... means not all dictionaries will have same number of array spin values
            spin_rad = [x for x in spin_rad if x <= aperture_rad]
            spin_hmr = [x for x in spin_hmr if x*galaxy.halfmass_rad <= aperture_rad]
            
            if len(spin_hmr) != len(spin_hmr_tmp):
                print('Capped spin_rad (%s pkpc) at aperture radius (%s pkpc)' %(max(spin_rad), aperture_rad))
        
        
        # If we want the original values, enter 0 for viewing angle
        subhalo = Subhalo_Analysis(mySims, GroupNum, SubGroupNum, GalaxyID, SnapNum, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, 
                                            viewing_axis,
                                            aperture_rad,
                                            kappa_rad, 
                                            trim_hmr, 
                                            align_rad,              #align_rad = False
                                            orientate_to_axis,
                                            viewing_angle,
                                            
                                            angle_selection,        
                                            spin_rad,
                                            spin_hmr,
                                            find_uncertainties,
                                            
                                            com_min_distance,
                                            min_particles,                                            
                                            min_inclination)
    
        
        
        if print_galaxy:
            print('ID:\t%s\t|M*:  %.2e  |HMR:  %.2f  |KAPPA:  %.2f' %(str(subhalo.GalaxyID), subhalo.stelmass, subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'])) 
        
    
        #===========================================
        # Graph initialising and base formatting
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig = plt.figure() 
        ax = Axes3D(fig, auto_add_to_figure=False, box_aspect=[1,1,1])
        fig.add_axes(ax, computed_zorder=False)
        
        
        def plot_rand_scatter(dict_name, part_type, color, debug=False):
            # Plot formatting
            ax.set_facecolor('xkcd:black')
            ax.w_xaxis.pane.fill = False
            ax.w_yaxis.pane.fill = False
            ax.w_zaxis.pane.fill = False
            ax.grid(False)
            ax.set_xlim(-boxradius, boxradius)
            ax.set_ylim(-boxradius, boxradius)
            ax.set_zlim(-boxradius, boxradius)
            
            # Keep same seed
            np.random.seed(42)
            
            # Selecting N (particles) sets of coordinates
            if dict_name[part_type]['Coordinates'].shape[0] <= particles:
                coords = dict_name[part_type]['Coordinates']
            else:
                coords = dict_name[part_type]['Coordinates'][np.random.choice(dict_name[part_type]['Coordinates'].shape[0], particles, replace=False), :]
            
            # Plot scatter
            if part_type == 'bh':
                bh_size = dict_name[part_type]['Mass']
                ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=(bh_size/8e5)**(1/3), alpha=1, c=color, zorder=4)
            else:
                ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.02, alpha=0.9, c=color, zorder=4)
           
        def plot_spin_vector(dict_name, part_type, rad, color, debug=False):
            # Plot formatting
            ax.set_facecolor('xkcd:black')
            ax.w_xaxis.pane.fill = False
            ax.w_yaxis.pane.fill = False
            ax.w_zaxis.pane.fill = False
            ax.grid(False)
            ax.set_xlim(-boxradius, boxradius)
            ax.set_ylim(-boxradius, boxradius)
            ax.set_zlim(-boxradius, boxradius)
            
            if part_type == 'dm':
                arrow = dict_name[part_type]
            
                # Plot original stars spin vector
                ax.quiver(0, 0, 0, arrow[0]*boxradius*0.6, arrow[1]*boxradius*0.6, arrow[2]*boxradius*0.6, color=color, alpha=1, linewidth=1, zorder=50)
            else:
                mask = np.where(dict_name['hmr'] == rad)
                        
                arrow = dict_name[part_type][int(min(mask))]
            
                # Plot original stars spin vector
                ax.quiver(0, 0, 0, arrow[0]*boxradius*0.6, arrow[1]*boxradius*0.6, arrow[2]*boxradius*0.6, color=color, alpha=1, linewidth=1, zorder=50)

        def plot_coms(dict_name, part_type, rad, color, debug=False):
            # Plot formatting
            ax.set_facecolor('xkcd:black')
            ax.w_xaxis.pane.fill = False
            ax.w_yaxis.pane.fill = False
            ax.w_zaxis.pane.fill = False
            ax.grid(False)
            ax.set_xlim(-boxradius, boxradius)
            ax.set_ylim(-boxradius, boxradius)
            ax.set_zlim(-boxradius, boxradius)
            
            if part_type == 'dm':
                com = dict_name[part_type]
                
                # Plot COM
                ax.scatter(com[0], com[1], com[2], color=color, alpha=1, s=3, zorder=10)
                
            else:
                mask = np.where(dict_name['hmr'] == rad)
                
                com = dict_name[part_type][int(min(mask))]
                
                # Plot COM
                ax.scatter(com[0], com[1], com[2], color=color, alpha=1, s=3, zorder=10)
                
            
        #--------------------------------------------
        # Buttons 
        class Index:
            def plot_clear_button(self, event):
                ax.cla()
                # Plot formatting
                ax.set_facecolor('xkcd:black')
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False
                ax.w_zaxis.pane.fill = False
                ax.grid(False)
                ax.set_xlim(-boxradius, boxradius)
                ax.set_ylim(-boxradius, boxradius)
                ax.set_zlim(-boxradius, boxradius)
                fig.canvas.draw_idle()
            def stars_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'stars', 'lightyellow')
                fig.canvas.draw_idle()
            def stars_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'stars', spin_vector_rad, 'r')
                fig.canvas.draw_idle()
            def gas_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'gas', 'lime')
                fig.canvas.draw_idle()    
            def gas_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'gas', spin_vector_rad, 'forestgreen')
                fig.canvas.draw_idle()
            def gas_sf_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'gas_sf', 'cyan')
                fig.canvas.draw_idle()
            def gas_sf_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'gas_sf', spin_vector_rad, 'darkturquoise')
                fig.canvas.draw_idle()
            def gas_nsf_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'gas_nsf', 'royalblue')
                fig.canvas.draw_idle()
            def gas_nsf_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'gas_nsf', spin_vector_rad, 'blue')   
                fig.canvas.draw_idle()
            def dm_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'dm', 'saddlebrown')
                fig.canvas.draw_idle()
            def dm_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'dm', aperture_rad, 'maroon') 
                fig.canvas.draw_idle()
            def bh_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'bh', 'blueviolet')
                fig.canvas.draw_idle()
            def com_button(self, event):
                plot_coms(subhalo.coms, 'stars', spin_vector_rad, 'r')
                plot_coms(subhalo.coms, 'gas', spin_vector_rad, 'forestgreen')
                plot_coms(subhalo.coms, 'gas_sf', spin_vector_rad, 'darkturquoise')
                plot_coms(subhalo.coms, 'gas_nsf', spin_vector_rad, 'blue')
                plot_coms(subhalo.coms, 'dm', spin_vector_rad, 'maroon')
                
                fig.canvas.draw_idle()
            def cop_button(self, event):
                ax.scatter(0, 0, 0, c='pink', s=20, zorder=10, marker='x')
                fig.canvas.draw_idle()    
            def draw_hmr(self, event):
                # Plot 1 and 2 HMR projected rad 
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = subhalo.halfmass_rad*np.cos(u)*np.sin(v)
                y = subhalo.halfmass_rad*np.sin(u)*np.sin(v)
                z = subhalo.halfmass_rad*np.cos(v)
            
                ax.plot_wireframe(x, y, z, color="w", alpha=0.3, linewidth=0.5)
            
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = 2*subhalo.halfmass_rad*np.cos(u)*np.sin(v)
                y = 2*subhalo.halfmass_rad*np.sin(u)*np.sin(v)
                z = 2*subhalo.halfmass_rad*np.cos(v)
            
                ax.plot_wireframe(x, y, z, color="w", alpha=0.3, linewidth=0.5)
            
                fig.canvas.draw_idle()
            def draw_hmr_proj(self, event):
                # Plot 1 and 2 HMR projected rad 
                radius_circle = Circle((0, 0), subhalo.halfmass_rad_proj, linewidth=1, edgecolor='w', alpha=0.7, facecolor=None, fill=False)
                radius_circle2 = Circle((0, 0), 2*subhalo.halfmass_rad_proj, linewidth=1, edgecolor='w', alpha=0.7, facecolor=None, fill=False)
                ax.add_patch(radius_circle)
                ax.add_patch(radius_circle2)
                art3d.pathpatch_2d_to_3d(radius_circle, z=0, zdir=viewing_axis)
                art3d.pathpatch_2d_to_3d(radius_circle2, z=0, zdir=viewing_axis)
            
                fig.canvas.draw_idle()    
            def draw_aperture(self, event):
                # Plot 1 and 2 HMR projected rad 
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = aperture_rad*np.cos(u)*np.sin(v)
                y = aperture_rad*np.sin(u)*np.sin(v)
                z = aperture_rad*np.cos(v)
            
                ax.plot_wireframe(x, y, z, color="r", alpha=0.3, linewidth=0.5)
            
                fig.canvas.draw_idle()    
            def load_region(self, event):
                load_region_cMpc = 0.15
                load_region = 0.5 * load_region_cMpc * u.Mpc.to(u.kpc) * 0.6777**-1. * 1.0**1.
            
                # Plotting verticies of loaded region
                ax.scatter(-load_region, -load_region, -load_region, s=3, c='lime')
                ax.scatter(-load_region, -load_region,  load_region, s=3, c='lime')
                ax.scatter( load_region, -load_region, -load_region, s=3, c='lime')
                ax.scatter( load_region, -load_region,  load_region, s=3, c='lime')
                ax.scatter(-load_region,  load_region, -load_region, s=3, c='lime')
                ax.scatter(-load_region,  load_region,  load_region, s=3, c='lime')
                ax.scatter( load_region,  load_region, -load_region, s=3, c='lime')
                ax.scatter( load_region,  load_region,  load_region, s=3, c='lime')
            
                fig.canvas.draw_idle()
            def auto_rotate(self, event):
                fig.canvas.draw_idle()
                for ii in np.arange(ax.azim, ax.azim+360, 1):
                    ax.view_init(elev=ax.elev, azim=ii)
                    plt.pause(0.01)
                    ax.set_zlim(-boxradius, boxradius)
                    fig.canvas.draw_idle()
            def view_x(self, event):
                fig.canvas.draw_idle()
                ax.view_init(elev=0, azim=0)
                ax.set_zlim(-boxradius, boxradius)
                fig.canvas.draw_idle()
            def view_y(self, event):
                fig.canvas.draw_idle()
                ax.view_init(elev=0, azim=90)
                ax.set_zlim(-boxradius, boxradius)
                fig.canvas.draw_idle()
            def view_z(self, event):
                fig.canvas.draw_idle()
                ax.view_init(elev=90, azim=0)
                ax.set_zlim(-boxradius, boxradius)
                fig.canvas.draw_idle()
            
        callback = Index()     
    
        #-----------
        bclear  = Button(fig.add_axes([0.01, 0.88, 0.12, 0.03]), 'CLEAR', color='red', hovercolor='red')
        #-----------
        bstars    = Button(fig.add_axes([0.01, 0.96, 0.10, 0.03]), 'STARS', color='yellow', hovercolor='yellow')
        bstars_v  = Button(fig.add_axes([0.11, 0.96, 0.02, 0.03]), '⬆', color='yellow', hovercolor='yellow') 
        bgas      = Button(fig.add_axes([0.13, 0.96, 0.10, 0.03]), 'GAS', color='lime', hovercolor='lime')
        bgas_v    = Button(fig.add_axes([0.23, 0.96, 0.02, 0.03]), '⬆', color='lime', hovercolor='lime')
        bgassf    = Button(fig.add_axes([0.25, 0.96, 0.10, 0.03]), 'GAS SF', color='cyan', hovercolor='cyan')
        bgassf_v  = Button(fig.add_axes([0.35, 0.96, 0.02, 0.03]), '⬆', color='cyan', hovercolor='cyan')
        bgasnsf   = Button(fig.add_axes([0.37, 0.96, 0.10, 0.03]), 'GAS NSF', color='royalblue', hovercolor='royalblue')
        bgasnsf_v = Button(fig.add_axes([0.47, 0.96, 0.02, 0.03]), '⬆', color='royalblue', hovercolor='royalblue')
        bdm       = Button(fig.add_axes([0.49, 0.96, 0.10, 0.03]), 'DM', color='saddlebrown', hovercolor='saddlebrown')
        bdm_v     = Button(fig.add_axes([0.59, 0.96, 0.02, 0.03]), '⬆', color='saddlebrown', hovercolor='saddlebrown')
        bbh     = Button(fig.add_axes([0.61, 0.96, 0.10, 0.03]), 'BH', color='blueviolet', hovercolor='blueviolet')
        #-----------
        bcom    = Button(fig.add_axes([0.01, 0.92, 0.12, 0.03]), 'C.O.M')
        bcop    = Button(fig.add_axes([0.13, 0.92, 0.12, 0.03]), 'C.O.P')
        bhmr    = Button(fig.add_axes([0.25, 0.92, 0.12, 0.03]), 'HMR')
        bhmrpro = Button(fig.add_axes([0.37, 0.92, 0.12, 0.03]), 'HMR P')
        bapert  = Button(fig.add_axes([0.49, 0.92, 0.12, 0.03]), 'APERT.')
        bregion = Button(fig.add_axes([0.61, 0.92, 0.12, 0.03]), 'LOADED')
        #-----------
        brotate = Button(fig.add_axes([0.81, 0.96, 0.18, 0.03]), 'ROTATE 360', color='limegreen', hovercolor='darkgreen')
        bviewx  = Button(fig.add_axes([0.81, 0.92, 0.05, 0.03]), 'x')
        bviewy  = Button(fig.add_axes([0.87, 0.92, 0.05, 0.03]), 'y')
        bviewz  = Button(fig.add_axes([0.93, 0.92, 0.05, 0.03]), 'z')
    
        #-----------
        bclear.on_clicked(callback.plot_clear_button)
        #-----------
        bstars.on_clicked(callback.stars_button)
        bstars_v.on_clicked(callback.stars_v_button)
        bgas.on_clicked(callback.gas_button)
        bgas_v.on_clicked(callback.gas_v_button)
        bgassf.on_clicked(callback.gas_sf_button)
        bgassf_v.on_clicked(callback.gas_sf_v_button)
        bgasnsf.on_clicked(callback.gas_nsf_button)
        bgasnsf_v.on_clicked(callback.gas_nsf_v_button)
        bdm.on_clicked(callback.dm_button)
        bdm_v.on_clicked(callback.dm_v_button)
        bbh.on_clicked(callback.bh_button)
        #-----------
        bcom.on_clicked(callback.com_button)
        bcop.on_clicked(callback.cop_button)
        bhmr.on_clicked(callback.draw_hmr)
        bhmrpro.on_clicked(callback.draw_hmr_proj)
        bapert.on_clicked(callback.draw_aperture)
        bregion.on_clicked(callback.load_region)
        #-----------
        brotate.on_clicked(callback.auto_rotate)
        bviewx.on_clicked(callback.view_x)
        bviewy.on_clicked(callback.view_y)
        bviewz.on_clicked(callback.view_z)
        #--------------------------------------------
        
        
        
        
        #--------------------------------------------
        # Plot scatters and spin vectors, COMs 
        if stars:
            plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'stars', 'lightyellow')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'stars', spin_vector_rad, 'r')
            if centre_of_mass:
                plot_coms(subhalo.coms, 'stars', spin_vector_rad, 'r')
        if gas:
            plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'gas', 'lime')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'gas', spin_vector_rad, 'forestgreen')
            if centre_of_mass:
                plot_coms(subhalo.coms, 'gas', spin_vector_rad, 'forestgreen')
        if gas_sf:
            plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'gas_sf', 'cyan')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'gas_sf', spin_vector_rad, 'darkturquoise')
            if centre_of_mass:
                plot_coms(subhalo.coms, 'gas_sf', spin_vector_rad, 'darkturquoise')
        if gas_nsf:
            plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'gas_nsf', 'royalblue')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'gas_nsf', spin_vector_rad, 'blue')  
            if centre_of_mass:
                plot_coms(subhalo.coms, 'gas_nsf', spin_vector_rad, 'blue')
        if dark_matter:
            plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'dm', 'saddlebrown')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'dm', aperture_rad, 'maroon') 
            if centre_of_mass:
                plot_coms(subhalo.coms, 'dm', spin_vector_rad, 'maroon')
        if black_holes:
            plot_rand_scatter(subhalo.data['%s' %str(trim_hmr[0])], 'bh', 'blueviolet')

        # Plot centre of potential (0,0,0) and mass
        if centre_of_pot == True:
            ax.scatter(0, 0, 0, c='pink', s=3, zorder=10)
            
        # Plot axis
        if axis == True:
            ax.quiver(0, 0, -boxradius, boxradius/3, 0, 0, color='r', linewidth=0.5)
            ax.quiver(0, 0, -boxradius, 0, boxradius/3, 0, color='g', linewidth=0.5)
            ax.quiver(0, 0, -boxradius, 0, 0, boxradius/3, color='b', linewidth=0.5)
                        
        # Viewing angle
        ax.view_init(90, 0)
        
        # Formatting 
        ax.set_xlabel('x-pos [pkpc]')
        ax.set_ylabel('y-pos [pkpc]')
        ax.set_zlabel('z-pos [pkpc]')
        ax.spines['bottom'].set_color('red')
        ax.spines['top'].set_color('red')
        ax.xaxis.label.set_color('grey')
        ax.yaxis.label.set_color('grey')
        ax.zaxis.label.set_color('grey')
        ax.tick_params(axis='x', colors='grey')
        ax.tick_params(axis='y', colors='grey')
        ax.tick_params(axis='z', colors='grey')
        
        
        # Savefig
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Finished')
        
        metadata_plot = {'Author': 'ID: %s' %GalaxyID,
                         'Title': 'Particles: %s' %particles}
       
        particle_txt = ''
        if stars:
            particle_txt += '_stars'
        if gas:
            particle_txt += '_gas'
        if gas_sf:
            particle_txt += '_gasSF'
        if gas_nsf:
            particle_txt += '_gasNSF'
        if dark_matter:
            particle_txt += '_dm'
        if black_holes:
            particle_txt += '_bh'
        
        if savefig:
            plt.savefig("%s/L%s_render_ID%s_%s_%s.%s" %(fig_dir, mySims[0][1], GalaxyID, particle_txt, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', pad_inches=0.1, dpi=600)    
            print('\n  SAVED:%s/L%s_render_ID%s_%s_%s.%s' %(fig_dir, mySims[0][1], GalaxyID, particle_txt, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
        
#--------------      
galaxy_render()
#--------------
