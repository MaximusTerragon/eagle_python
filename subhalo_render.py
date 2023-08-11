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
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID, ConvertID_snip
import eagleSqlTools as sql
from graphformat import set_rc_params
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local\n")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================
 
    
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
ID_list = [3748, 20455, 37445, 30494, 43163, 40124, 44545, 48383, 57647, 55343, 51640, 46366, 53904, 52782, 56522, 59467, 49986, 61119, 62355, 63199, 61831]


#====================================
# Will visualise all particles belonging to a SINGLE galaxy/subhalo
# SAVED:%s/individual_render/
def galaxy_render(csv_sample = False,              # False, Whether to read in existing list of galaxies  
                    #--------------------------
                    mySims = [('RefL0012N0188', 12)],
                    GalaxyID_List = [20464],
                    #--------------------------
                    # Galaxy extraction properties
                    kappa_rad            = 30,          # calculate kappa for this radius [pkpc]
                    viewing_angle        = 0,           # Keep as 0
                    #--------------------------
                    # Visualisation properties
                    boxradius           = 50,                           # size of box of render [kpc], 'rad', 'tworad'
                    particles           = 5000,                         # number of random particles to plot
                    viewing_axis        = 'z',                          # Which axis to view galaxy from.  DEFAULT 'z'
                    aperture_rad        = 30,                           # calculations radius limit [pkpc]
                    trim_rad            = np.array([100]),           # largest radius in pkpc to plot | 2.0_hmr, rad_projected=True
                    align_rad           = False,                          # False/Value
                    mask_sgn            = False,                        # False = plot all nearby subhalos too
                    #=====================================================
                    # Misalignments we want extracted and at which radii  
                    angle_selection     = ['stars_gas',                     # stars_gas     stars_gas_sf    stars_gas_nsf
                                           'stars_gas_sf',                  # gas_dm        gas_sf_dm       gas_nsf_dm
                                           'stars_gas_nsf',                 # gas_sf_gas_nsf
                                           'gas_sf_gas_nsf',
                                           'stars_dm'],           
                    spin_hmr            = np.array([2.0]),                  # multiples of hmr for which to find spin. Will plot lowest value
                    rad_projected       = True,                             # whether to use rad in projection or 3D
                    #--------------------------
                    # Plot options
                    plot_spin_vectors   = True,
                    centre_of_pot       = True,                             # Plot most bound object 
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
                    #--------------------------
                    color_metallicity   = False,
                    #=====================================================
                    showfig       = True,
                    savefig       = False,
                      savefig_txt = '',                # added txt to append to end of savefile
                      file_format = 'pdf',
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
    # adjust mySims for serpens
    if (answer == '2') or (answer == '3') or (answer == '4'):
        mySims = [('RefL0100N1504', 100)]
    
    
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
        Redshift_List       = np.array(dict_new['Redshift'])
        HaloMass_List       = np.array(dict_new['halo_mass'])
        Centre_List         = np.array(dict_new['centre'])
        MorphoKinem_List    = np.array(dict_new['MorphoKinem'])
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
        
    # If no csv_sample given, use GalaxyID_List
    else:
        # If using snipshots...
        if answer == '3':
            # Extract GroupNum, SubGroupNum, and Snap for each ID
            GroupNum_List    = []
            SubGroupNum_List = []
            SnapNum_List     = []
            Redshift_List    = []
            HaloMass_List    = []
            Centre_List      = []
            MorphoKinem_List = []
            
            for galID in GalaxyID_List:
                gn, sgn, snap, z, halomass_i, centre_i, morphkinem_i = ConvertID_snip(tree_dir, galID, mySims)
    
                # Append to arrays
                GroupNum_List.append(gn)
                SubGroupNum_List.append(sgn)
                SnapNum_List.append(snap)
                Redshift_List.append(z)
                HaloMass_List.append(halomass_i) 
                Centre_List.append(centre_i)
                MorphoKinem_List.append(morphkinem_i)
            
            if debug:
                print(GroupNum_List)
                print(SubGroupNum_List)
                print(GalaxyID_List)
                print(SnapNum_List)
        
        # Else...
        else:
            # Extract GroupNum, SubGroupNum, and Snap for each ID
            GroupNum_List    = []
            SubGroupNum_List = []
            SnapNum_List     = []
            Redshift_List    = []
            HaloMass_List    = []
            Centre_List      = []
            MorphoKinem_List = []
         
            for galID in GalaxyID_List:
                gn, sgn, snap, z, halomass_i, centre_i, morphkinem_i = ConvertID(galID, mySims)
    
                # Append to arrays
                GroupNum_List.append(gn)
                SubGroupNum_List.append(sgn)
                SnapNum_List.append(snap)
                Redshift_List.append(z)
                HaloMass_List.append(halomass_i) 
                Centre_List.append(centre_i)
                MorphoKinem_List.append(morphkinem_i)
            
            if debug:
                print(GroupNum_List)
                print(SubGroupNum_List)
                print(GalaxyID_List)
                print(SnapNum_List)
            
        print('\n===================')
        print('SAMPLE INPUT:\n  %s\n  GalaxyIDs: %s' %(mySims[0][0], GalaxyID_List))
        print('  SAMPLE LENGTH: ', len(GroupNum_List))
        print('===================')
        
              
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))

    # Run analysis for each individual galaxy in loaded sample
    for GroupNum, SubGroupNum, GalaxyID, SnapNum, Redshift, HaloMass, Centre_i, MorphoKinem in tqdm(zip(GroupNum_List, SubGroupNum_List, GalaxyID_List, SnapNum_List, Redshift_List, HaloMass_List, Centre_List, MorphoKinem_List), total=len(GroupNum_List)):
        
        #-----------------------------
        if print_progress:
            print('Extracting particle data Subhalo_Extract()')
            time_start = time.time()
        
        
        # Initial extraction of galaxy particle data
        galaxy = Subhalo_Extract(mySims, dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, Centre_i, HaloMass, aperture_rad, viewing_axis, mask_sgn = mask_sgn)
        GroupNum = galaxy.gn
        
        # Gives: galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh
        # Gives: subhalo.general: GroupNum, SubGroupNum, GalaxyID, stelmass, gasmass, gasmass_sf, gasmass_nsf
        """mask_sf        = np.nonzero(galaxy.data_nil['gas']['StarFormationRate'])          
        mask_nsf       = np.where(galaxy.data_nil['gas']['StarFormationRate'] == 0)
        
        # Create dataset of star-forming and non-star-forming gas
        gas_sf = {}
        gas_nsf = {}
        for arr in galaxy.data_nil['gas'].keys():
            gas_sf[arr]  = galaxy.data_nil['gas'][arr][mask_sf]
            gas_nsf[arr] = galaxy.data_nil['gas'][arr][mask_nsf]
            
        
        print(gas_sf['StarFormationRate'])
        print('Gas: ', len(galaxy.data_nil['gas']['Mass']))
        print('SF:  ', len(gas_sf['Mass']))
        print('NSF: ', len(gas_nsf['Mass']))
        """
        
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
        
            # Reduce spin_rad array if value exceeds aperture_rad_in... means not all dictionaries will have same number of array spin values
            spin_rad_in_tmp = [x for x in spin_rad if x <= aperture_rad]
            spin_hmr_in_tmp = [x for x in spin_hmr if x*galaxy.halfmass_rad_proj <= aperture_rad]
            
            # Ensure min. rad is >1 pkpc
            spin_rad_in = [x for x in spin_rad_in_tmp if x >= 1.0]
            spin_hmr_in = [x for x in spin_hmr_in_tmp if x*galaxy.halfmass_rad_proj >= 1.0]
        
            if len(spin_hmr_in) != len(spin_hmr_tmp):
                print('Capped spin_rad: %.2f - %.2f - %.2f HMR | Min/Max %.2f / %.2f pkpc' %(min(spin_hmr_in), (max(spin_hmr_in) - min(spin_hmr_in))/(len(spin_hmr_in) - 1), max(spin_hmr_in), min(spin_rad_in), max(spin_rad_in)))
        elif rad_projected == False:
            spin_rad = np.array(spin_hmr) * galaxy.halfmass_rad
            spin_hmr_tmp = spin_hmr
        
            # Reduce spin_rad array if value exceeds aperture_rad_in... means not all dictionaries will have same number of array spin values
            spin_rad_in_tmp = [x for x in spin_rad if x <= aperture_rad]
            spin_hmr_in_tmp = [x for x in spin_hmr if x*galaxy.halfmass_rad <= aperture_rad]
            
            # Ensure min. rad is >1 pkpc
            spin_rad_in = [x for x in spin_rad_in_tmp if x >= 1.0]
            spin_hmr_in = [x for x in spin_hmr_in_tmp if x*galaxy.halfmass_rad >= 1.0]
        
            if len(spin_hmr_in) != len(spin_hmr_tmp):
                print('Capped spin_rad: %.2f - %.2f - %.2f HMR | Min/Max %.2f / %.2f pkpc' %(min(spin_hmr_in), (max(spin_hmr_in) - min(spin_hmr_in))/(len(spin_hmr_in) - 1), max(spin_hmr_in), min(spin_rad_in), max(spin_rad_in)))  
        
        
        # If we want the original values, enter 0 for viewing angle
        subhalo = Subhalo_Analysis(mySims, GroupNum, SubGroupNum, GalaxyID, SnapNum, MorphoKinem, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.data_nil, 
                                            viewing_axis,
                                            aperture_rad,
                                            kappa_rad, 
                                            trim_rad, 
                                            align_rad,              #align_rad = False
                                            orientate_to_axis,
                                            viewing_angle,
                                            
                                            angle_selection,        
                                            spin_rad_in,
                                            spin_hmr_in,
                                            find_uncertainties,
                                            rad_projected,
                                            
                                            com_min_distance,
                                            min_particles,                                            
                                            min_inclination)
                        
        if print_galaxy:
            print('|Combined particle properties within %s pkpc:' %aperture_rad)
            print('|%s| |ID:   %s\t|M*:  %.2e  |HMR:  %.2f  |KAPPA:  %.2f' %(SnapNum, str(subhalo.GalaxyID), subhalo.stelmass, subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'])) 
        
        # INFLOW OUTFLOW 
        #print(subhalo.mass_flow['2.0_hmr']['gas_sf']['inflow'])
        #print(subhalo.mass_flow['2.0_hmr']['gas_sf']['insitu_Z'])
        #print(subhalo.sfr['hmr'])
        #print(np.multiply(3.154e+7, subhalo.sfr['gas_sf']))
        #print(subhalo.Z['hmr'])
        #print(subhalo.Z['stars'])
        #print(subhalo.Z['gas_sf'])
        
        #print(subhalo.bh_mdot)
        #print(subhalo.bh_edd)
        #print(subhalo.bh_id)
        
        #print(len(subhalo.gas_data['2.0_hmr']['gas']['ParticleIDs']))
        #print(subhalo.gas_data['2.0_hmr']['gas']['Total_mass'])
        #print(len(subhalo.gas_data['2.0_hmr']['gas_sf']['ParticleIDs']))
        #print(subhalo.gas_data['2.0_hmr']['gas_sf']['Total_mass'])
        #print(subhalo.gas_data['2.0_hmr']['gas_sf']['ParticleIDs'])
        #print(subhalo.gas_data['2.0_hmr']['gas_sf'].keys())
        
        #print(subhalo.data['%s' %str(trim_rad[0])]['gas_sf']['ParticleIDs'][35])
        # 8035248386127
        #if 8035248386127.0 in subhalo.data['%s' %str(trim_rad[0])]['gas_sf']['ParticleIDs']:
            #print('True')
        
        
        #===========================================
        # Graph initialising and base formatting
        def graphformat(size1, size2, size3, size4, size5, width, height):
            plt.rcParams['text.usetex'] = False
            plt.rcParams["font.family"] = "DeJavu Serif"
            plt.rcParams['font.serif'] = ['Times New Roman']
    
            # General graph font size formatting
            plt.rc('font', size=size1)          # controls default text sizes
    
            plt.rc('figure', titlesize=size2)   # fontsize of the figure title
    
            plt.rc('axes', titlesize=size3)     # fontsize of the axes title
            plt.rc('axes', labelsize=size3)     # fontsize of the x and y labels
    
            plt.rc('xtick', labelsize=size4)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=size4)    # fontsize of the tick labels

            plt.rc('legend', fontsize=size5)    # legend fontsize

            plt.rc('figure', figsize=(width, height))  # figure size [inches]
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig = plt.figure() 
        if (answer == '2') or (answer == '3'):
            ax = Axes3D(fig, box_aspect=[1,1,1])
        else:
            ax = Axes3D(fig, auto_add_to_figure=False, box_aspect=[1,1,1])
        fig.add_axes(ax, computed_zorder=False)
        
        
        def plot_rand_scatter(dict_name, part_type, color, metallicity=color_metallicity, debug=False):
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
                if (part_type == 'gas') | (part_type == 'gas_sf') | (part_type == 'gas_nsf') | (part_type == 'stars'):
                    metals = dict_name[part_type]['Metallicity']
            else:
                coords = dict_name[part_type]['Coordinates'][np.random.choice(dict_name[part_type]['Coordinates'].shape[0], particles, replace=False), :]
                if (part_type == 'gas') | (part_type == 'gas_sf') | (part_type == 'gas_nsf') | (part_type == 'stars'):
                    metals = dict_name[part_type]['Metallicity'][np.random.choice(dict_name[part_type]['Coordinates'].shape[0], particles, replace=False)]
            
            # Plot scatter
            if part_type == 'bh':
                bh_size = dict_name[part_type]['Mass']
                ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=(bh_size/8e5)**(1/3), alpha=1, c=color, zorder=4)
            else:
                if (part_type == 'gas') | (part_type == 'gas_sf') | (part_type == 'gas_nsf') | (part_type == 'stars'):
                    if metallicity:
                        if part_type == 'gas':
                            ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.02, alpha=0.9, c=metals, zorder=4, cmap='YlGn_r')
                        if part_type == 'gas_sf':
                            ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.02, alpha=0.9, c=metals, zorder=4, cmap='Blues_r')
                        if part_type == 'gas_nsf':
                            ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.02, alpha=0.9, c=metals, zorder=4, cmap='Purples_r')
                        if part_type == 'stars':
                            ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.02, alpha=0.9, c=metals, zorder=4, cmap='YlOrRd_r')
                        
                    else:
                        ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.02, alpha=0.9, c=color, zorder=4)
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
            

            mask = np.where(dict_name['hmr'] == rad)
                    
            arrow = dict_name[part_type][int(min(mask))]
            point = subhalo.coms['adjust'][int(min(mask))]
        
            # Plot original stars spin vector
            ax.quiver(point[0], point[1], point[2], arrow[0]*boxradius*0.6, arrow[1]*boxradius*0.6, arrow[2]*boxradius*0.6, color=color, alpha=1, linewidth=1, zorder=50)

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
                
                com = dict_name[part_type][int(min(mask))] + dict_name['adjust'][int(min(mask))]
                
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
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'stars', 'lightyellow')
                fig.canvas.draw_idle()
            def stars_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'stars', spin_vector_rad, 'r')
                fig.canvas.draw_idle()
            def gas_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'gas', 'lime')
                fig.canvas.draw_idle()    
            def gas_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'gas', spin_vector_rad, 'forestgreen')
                fig.canvas.draw_idle()
            def gas_sf_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'gas_sf', 'cyan')
                fig.canvas.draw_idle()
            def gas_sf_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'gas_sf', spin_vector_rad, 'darkturquoise')
                fig.canvas.draw_idle()
            def gas_nsf_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'gas_nsf', 'royalblue')
                fig.canvas.draw_idle()
            def gas_nsf_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'gas_nsf', spin_vector_rad, 'blue')   
                fig.canvas.draw_idle()
            def dm_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'dm', 'saddlebrown')
                fig.canvas.draw_idle()
            def dm_v_button(self, event):
                plot_spin_vector(subhalo.spins, 'dm', spin_vector_rad, 'maroon') 
                fig.canvas.draw_idle()
            def bh_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'bh', 'blueviolet')
                fig.canvas.draw_idle()
            def com_button(self, event):
                plot_coms(subhalo.coms, 'stars', spin_vector_rad, 'r')
                plot_coms(subhalo.coms, 'gas', spin_vector_rad, 'forestgreen')
                plot_coms(subhalo.coms, 'gas_sf', spin_vector_rad, 'darkturquoise')
                plot_coms(subhalo.coms, 'gas_nsf', spin_vector_rad, 'blue')
                plot_coms(subhalo.coms, 'dm', spin_vector_rad, 'maroon')
                
                fig.canvas.draw_idle()
            def cop_button(self, event):
                ax.scatter(galaxy.stars_com[0], galaxy.stars_com[1], galaxy.stars_com[2], c='pink', s=20, zorder=10, marker='x')
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
                #radius_circle3 = Circle((0, 0), 0.5*subhalo.halfmass_rad_proj, linewidth=1, edgecolor='w', alpha=0.7, facecolor=None, fill=False)
                ax.add_patch(radius_circle)
                ax.add_patch(radius_circle2)
                #ax.add_patch(radius_circle3)
                art3d.pathpatch_2d_to_3d(radius_circle, z=0, zdir=viewing_axis)
                art3d.pathpatch_2d_to_3d(radius_circle2, z=0, zdir=viewing_axis)
                #art3d.pathpatch_2d_to_3d(radius_circle3, z=0, zdir=viewing_axis)
            
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
                    ax.set_zlim(ax.get_ylim())
                    fig.canvas.draw_idle()
            def view_x(self, event):
                fig.canvas.draw_idle()
                ax.view_init(elev=0, azim=0)
                ax.set_zlim(ax.get_ylim())
                fig.canvas.draw_idle()
            def view_y(self, event):
                fig.canvas.draw_idle()
                ax.view_init(elev=0, azim=90)
                ax.set_zlim(ax.get_ylim())
                fig.canvas.draw_idle()
            def view_z(self, event):
                fig.canvas.draw_idle()
                ax.view_init(elev=90, azim=0)
                ax.set_zlim(ax.get_ylim())
                fig.canvas.draw_idle()
            def zoom_plus(self, event):
                fig.canvas.draw_idle()
                current_lim = ax.get_ylim()
                ax.set_xlim(current_lim[0]*0.8, current_lim[1]*0.8)
                ax.set_ylim(current_lim[0]*0.8, current_lim[1]*0.8)
                ax.set_zlim(current_lim[0]*0.8, current_lim[1]*0.8)
                fig.canvas.draw_idle()
            def zoom_minus(self, event):
                fig.canvas.draw_idle()
                current_lim = ax.get_ylim()
                ax.set_xlim(current_lim[0]/0.8, current_lim[1]/0.8)
                ax.set_ylim(current_lim[0]/0.8, current_lim[1]/0.8)
                ax.set_zlim(current_lim[0]/0.8, current_lim[1]/0.8)
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
        bplus   = Button(fig.add_axes([0.93, 0.10, 0.05, 0.05]), '+')
        bminus  = Button(fig.add_axes([0.93, 0.05, 0.05, 0.05]), '-')
    
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
        #-----------
        bplus.on_clicked(callback.zoom_plus)
        bminus.on_clicked(callback.zoom_minus)
        #--------------------------------------------
        
        
        
        
        #--------------------------------------------
        # Plot scatters and spin vectors, COMs 
        if stars:
            plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'stars', 'lightyellow')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'stars', spin_vector_rad, 'r')
            if centre_of_mass:
                plot_coms(subhalo.coms, 'stars', spin_vector_rad, 'r')
        if gas:
            plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'gas', 'lime')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'gas', spin_vector_rad, 'forestgreen')
            if centre_of_mass:
                plot_coms(subhalo.coms, 'gas', spin_vector_rad, 'forestgreen')
        if gas_sf:
            plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'gas_sf', 'cyan')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'gas_sf', spin_vector_rad, 'darkturquoise')
            if centre_of_mass:
                plot_coms(subhalo.coms, 'gas_sf', spin_vector_rad, 'darkturquoise')
        if gas_nsf:
            plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'gas_nsf', 'royalblue')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'gas_nsf', spin_vector_rad, 'blue')  
            if centre_of_mass:
                plot_coms(subhalo.coms, 'gas_nsf', spin_vector_rad, 'blue')
        if dark_matter:
            plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'dm', 'saddlebrown')
            if plot_spin_vectors:
                plot_spin_vector(subhalo.spins, 'dm', aperture_rad, 'maroon') 
            if centre_of_mass:
                plot_coms(subhalo.coms, 'dm', spin_vector_rad, 'maroon')
        if black_holes:
            plot_rand_scatter(subhalo.data['%s' %str(trim_rad[-1])], 'bh', 'blueviolet')
            
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
            plt.savefig("%s/individual_render/L%s_render_ID%s_%s_%s_%s.%s" %(fig_dir, mySims[0][1], GalaxyID, SnapNum, particle_txt, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
            print('\n  SAVED:%s/individual_render/L%s_render_ID%s_%s_%s_%s.%s' %(fig_dir, mySims[0][1], GalaxyID, SnapNum, particle_txt, savefig_txt, file_format))
        if showfig:
            plt.show()
        plt.close()
        
        
#--------------      
galaxy_render()
#--------------
