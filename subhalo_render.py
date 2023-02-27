import h5py
import numpy as np
import pandas as pd
import random
import math
import csv
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from subhalo_main import Subhalo_Extract, Subhalo
from graphformat import graphformat


# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28

# Directories of data hdf5 file(s)
#dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
  
    
""" 
PURPOSE
-------
- Can visualise a galaxy as a scatter graph in 3D and render side-on images of it
- Can do this for X particles or all of them
- Render boxsize uses fixed aperture (say 100pkpc), but can also trim particle data to within x*HMR
- Galaxy is automatically centred, and velocity adjusted
- Can also orientate the galaxy to a given axis, based on the angular momentum vector calculated at a given distance. This is used in the calculation of kappa at 30pkpc

"""
#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
#1, 2, 4, 3, 6, 5, 7, 9, 16, 14, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
def galaxy_render(manual_GroupNumList = np.array([1, 2, 4, 3, 6, 5, 7, 9, 16, 14, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21]),
                    SubGroupNum       = 0, 
                  spin_rad_in           = np.array([2.0]),              # multiples of rad
                  kappa_rad_in          = 30,                           # Calculate kappa for this radius [pkpc]
                  aperture_rad_in       = 30,                           # trim all data to this maximum value
                  align_rad_in          = False,                              # Align galaxy to stellar vector in. this radius [pkpc]
                  projected_or_abs      = 'projected',                  # 'projected' or 'abs'
                  orientate_to_axis='z',                                      # Keep as 'z'
                  viewing_angle=0,                                            # Keep as 0
                    minangle  = 0,
                    maxangle  = 0, 
                    stepangle = 30,
                  plot_spin_vectors = True,
                    spin_vector_rad = 2.0,            # radius of spinvector to display (in hmr)
                  centre_of_pot     = True,           # Plot most bound object (default centre)
                  centre_of_mass    = False,           # Plot total centre of mass
                  axis              = True,           # Plot small axis below galaxy
                      boxradius_in          = 30,                  # boxradius of render [kpc], 'rad', 'tworad'
                      particles             = 5000,
                      trim_rad_in           = np.array([1000]),     # WILL PLOT LOWEST VALUE. trim particles # multiples of rad, num [pkpc], found in subhalo.data[hmr]    
                      stars                 = True,
                      gas_sf                = True,
                      gas_nsf               = True,    
                      dark_matter           = True,
                      black_holes           = True,
                  find_uncertainties = False,                   # LEAVE THESE. whether to find 2D and 3D uncertainties
                  viewing_axis = 'x',                           # Which axis to view galaxy from.  DEFAULT 'z'
                  com_min_distance = 10000,                      # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                  gas_sf_min_particles = 0,                     # Minimum gas sf particles to use galaxy.  DEFAULT 100
                  min_inclination = 0,                          # Minimum inclination toward viewing axis [deg] DEFAULT 0
                  angle_type_in = ['stars_gas', 'stars_gas_sf', 'stars_gas_nsf'],             # PA fits and PA misalignment angles to be found ['stars_gas', 'stars_gas_sf', 'stars_gas_nsf', 'gas_sf_gas_nsf']. Particles making up this data will be automatically found, ei. stars_gas_sf = stars and gas_sf   
                  root_file = '/Users/c22048063/Documents/EAGLE/trial_plots',        # 'trial_plots' or 'plots'
                    print_galaxy       = False,              # print galaxy stats in chat
                    print_galaxy_short = True,
                    txt_file           = False,              # create a txt file with print data
                    showfig        = True,
                    savefig        = False,
                      savefigtxt       = '',                # added txt to append to end of savefile
                    debug = False):
        
        
    # Converting names of variables for consistency
    GroupNumList = manual_GroupNumList    
    
    #-----------------------------------------------------------
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
    
    
    for GroupNum in tqdm(GroupNumList):         
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum, aperture_rad_in, viewing_axis)
        
        if projected_or_abs == 'projected':
            use_rad = galaxy.halfmass_rad_proj
        elif projected_or_abs == 'abs':
            use_rad = galaxy.halfmass_rad
            
        # Detect keywords of rad, tworad. All in [pkpc]
        spin_rad = spin_rad_in * use_rad                            #pkpc
        trim_rad = trim_rad_in
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
        if boxradius_in == 'rad':
            boxradius = use_rad
        elif boxradius_in == 'tworad':
            boxradius = 2*use_rad
        else:
            boxradius = boxradius_in
            
        # Galaxy will be rotated to calc_kappa_rad's stellar spin value. subhalo.halfmass_rad_proj will be EITHER abs or projected
        subhalo = Subhalo(galaxy.gn, galaxy.sgn, galaxy.GalaxyID, galaxy.stelmass, galaxy.gasmass, galaxy.halfmass_rad, use_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, galaxy.MorphoKinem,
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
                                            min_inclination,
                                            quiet=True)
        
        
        # Add filter to skip pafitting of galaxy if any basic condition not met
        if len(subhalo.flags) != 0:
            print('Subhalo skipped')
            print(subhalo.flags)
            continue
        
        if debug == True:
            print(' ')
            print(subhalo.mis_angles.items()) 
            print(' ')                       
                                        
        # Print galaxy properties
        if print_galaxy == True:
            print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
            print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))       # [Msun]
            print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad_proj)             # [pkpc]
            print('KAPPA:                  %.2f' %subhalo.kappa)
            print('KAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)
            print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
            mask = np.where(np.array(subhalo.coms['hmr'] == min(spin_rad_in)))
            print('C.O.M %s HMR STARS-SF [pkpc]:  %.2f' %(str(min(spin_rad_in)), subhalo.coms['stars_gas_sf'][int(mask[0])]))
        elif print_galaxy_short == True:
            print('GN:\t%s\t|HMR:\t%.2f\t|KAPPA / SF:\t%.2f  %.2f' %(str(subhalo.gn), subhalo.halfmass_rad_proj, subhalo.general['kappa_stars'], subhalo.general['kappa_gas_sf'])) 
             
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
            
            mask = np.where(dict_name['hmr'] == rad)
                        
            arrow = dict_name[part_type][int(min(mask))]
            
            # Plot original stars spin vector
            ax.quiver(0, 0, 0, arrow[0]*boxradius*0.6, arrow[1]*boxradius*0.6, arrow[2]*boxradius*0.6, color=color, alpha=1, linewidth=1, zorder=50)

        #--------------------------------------------
        # Buttons for stars, gas_sf, gas_nsf
        class Index:
            def stars_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'stars', 'lightyellow')
                plot_spin_vector(subhalo.spins, 'stars', spin_vector_rad, 'r')
                fig.canvas.draw_idle()
            def gas_sf_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'gas_sf', 'cyan')
                plot_spin_vector(subhalo.spins, 'gas_sf', spin_vector_rad, 'darkturquoise')
                fig.canvas.draw_idle()
            def gas_nsf_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'gas_nsf', 'royalblue')
                plot_spin_vector(subhalo.spins, 'gas_nsf', spin_vector_rad, 'blue')   
                fig.canvas.draw_idle()
            def dm_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'dm', 'saddlebrown')
                fig.canvas.draw_idle()
            def bh_button(self, event):
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'bh', 'blueviolet')
                fig.canvas.draw_idle()
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
            def com_button(self, event):
                ax.scatter(subhalo.centre_mass[0] - subhalo.centre[0], subhalo.centre_mass[1] - subhalo.centre[1], subhalo.centre_mass[2] - subhalo.centre[2], c='purple', marker='x', s=20, zorder=10)
                fig.canvas.draw_idle()
            def cop_button(self, event):
                ax.scatter(0, 0, 0, c='pink', s=20, zorder=10, marker='x')
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
         
        bclear  = Button(fig.add_axes([0.01, 0.88, 0.12, 0.03]), 'CLEAR', color='red', hovercolor='red')
        bstars  = Button(fig.add_axes([0.01, 0.96, 0.12, 0.03]), 'STARS', color='yellow', hovercolor='yellow')
        bgassf  = Button(fig.add_axes([0.13, 0.96, 0.12, 0.03]), 'GAS SF', color='cyan', hovercolor='cyan')
        bgasnsf = Button(fig.add_axes([0.25, 0.96, 0.12, 0.03]), 'GAS NSF', color='royalblue', hovercolor='royalblue')
        bdm     = Button(fig.add_axes([0.37, 0.96, 0.12, 0.03]), 'DM', color='saddlebrown', hovercolor='saddlebrown')
        bbh     = Button(fig.add_axes([0.49, 0.96, 0.12, 0.03]), 'BH', color='blueviolet', hovercolor='blueviolet')
        bcom    = Button(fig.add_axes([0.01, 0.92, 0.12, 0.03]), 'C.O.M')
        bcop    = Button(fig.add_axes([0.13, 0.92, 0.12, 0.03]), 'C.O.P')
        brotate = Button(fig.add_axes([0.81, 0.96, 0.18, 0.03]), 'ROTATE 360', color='limegreen', hovercolor='darkgreen')
        bregion = Button(fig.add_axes([0.25, 0.92, 0.18, 0.03]), 'LOAD REGION')
        bviewx  = Button(fig.add_axes([0.81, 0.92, 0.05, 0.03]), 'x')
        bviewy  = Button(fig.add_axes([0.87, 0.92, 0.05, 0.03]), 'y')
        bviewz  = Button(fig.add_axes([0.93, 0.92, 0.05, 0.03]), 'z')
        
        bstars.on_clicked(callback.stars_button)
        bgassf.on_clicked(callback.gas_sf_button)
        bgasnsf.on_clicked(callback.gas_nsf_button)
        bdm.on_clicked(callback.dm_button)
        bbh.on_clicked(callback.bh_button)
        bclear.on_clicked(callback.plot_clear_button)
        bcom.on_clicked(callback.com_button)
        bcop.on_clicked(callback.cop_button)
        brotate.on_clicked(callback.auto_rotate)
        bregion.on_clicked(callback.load_region)
        bviewx.on_clicked(callback.view_x)
        bviewy.on_clicked(callback.view_y)
        bviewz.on_clicked(callback.view_z)
        #--------------------------------------------
        
        # Plot scatters and spin vectors   
        if not align_rad:
            if stars:
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'stars', 'lightyellow')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins, 'stars', spin_vector_rad, 'r')
                    plot_spin_vector(subhalo.spins, 'gas', spin_vector_rad, 'lime')
            if gas_sf:
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'gas_sf', 'cyan')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins, 'gas_sf', spin_vector_rad, 'darkturquoise')
            if gas_nsf:
                plot_rand_scatter(subhalo.data['%s' %str(trim_rad_in[0])], 'gas_nsf', 'royalblue')  
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins, 'gas_nsf', spin_vector_rad, 'blue')   
        elif align_rad:
            if stars:
                plot_rand_scatter(subhalo.data_align['%s' %str(trim_rad_in[0])], 'stars', 'lightyellow')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins_align, 'stars', spin_vector_rad, 'r')
            if gas_sf:
                plot_rand_scatter(subhalo.data_align['%s' %str(trim_rad_in[0])], 'gas_sf', 'cyan')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins_align, 'gas_sf', spin_vector_rad, 'teal')
            if gas_nsf:
                plot_rand_scatter(subhalo.data_align['%s' %str(trim_rad_in[0])], 'gas_nsf', 'royalblue')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins_align, 'gas_nsf', spin_vector_rad, 'blue')
        
        # Plot centre of potential (0,0,0) and mass
        if centre_of_pot == True:
            ax.scatter(0, 0, 0, c='pink', s=3, zorder=10)
        if centre_of_mass == True:
            ax.scatter(subhalo.centre_mass[0] - subhalo.centre[0], subhalo.centre_mass[1] - subhalo.centre[1], subhalo.centre_mass[2] - subhalo.centre[2], c='purple', s=3, zorder=10)
        
        # Plot axis
        if axis == True:
            ax.quiver(0, 0, -boxradius, boxradius/3, 0, 0, color='r', linewidth=0.5)
            ax.quiver(0, 0, -boxradius, 0, boxradius/3, 0, color='g', linewidth=0.5)
            ax.quiver(0, 0, -boxradius, 0, 0, boxradius/3, color='b', linewidth=0.5)
                          
        
        
        
            
        
        
        
        for ii in np.arange(minangle, maxangle+1, stepangle):
            #print(ii , end=' ')                 # [deg]
            ax.view_init(0, ii)
            
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

            if savefig:
                if (stars == True) & (gas_sf == True) & (gas_nsf == True) & (align_rad_in == False):
                    plt.savefig("%s/galaxy_%s/render_all_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == True) & (gas_sf == False) & (gas_nsf == False) & (align_rad_in == False):
                    plt.savefig("%s/galaxy_%s/render_s_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == False) & (gas_sf == True) & (gas_nsf == True) & (align_rad_in == False):
                    plt.savefig("%s/galaxy_%s/render_sf_nsf_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == False) & (gas_sf == True) & (gas_nsf == False) & (align_rad_in == False):
                    plt.savefig("%s/galaxy_%s/render_sf_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == False) & (gas_sf == False) & (gas_nsf == True) & (align_rad_in == False):
                    plt.savefig("%s/galaxy_%s/render_nsf_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == True) & (gas_sf == True) & (gas_nsf == True):
                    plt.savefig("%s/galaxy_%s/render_rot_all_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == True) & (gas_sf == False) & (gas_nsf == False):
                    plt.savefig("%s/galaxy_%s/render_rot_s_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == False) & (gas_sf == True) & (gas_nsf == False):
                    plt.savefig("%s/galaxy_%s/render_rot_sf_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)
                elif (stars == False) & (gas_sf == False) & (gas_nsf == True):
                    plt.savefig("%s/galaxy_%s/render_rot_nsf_rad%s_spin%s_angle%s%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in[0]), str(spin_vector_rad), str(ii), savefigtxt), dpi=300)

        if showfig:
            plt.show()
            
        plt.close()
        
        # Create txt file with output for that galaxy
        if txt_file == True:
            dash = '-' * 100
            f = open("%s/galaxy_%s/render_gn%s.txt" %(str(root_file), str(GroupNum), str(GroupNum)), 'w+')
            f.write(dash)
            f.write('\nGROUP NUMBER:           %s' %str(subhalo.gn))
            f.write('\nSUBGROUP NUMBER:        %s' %str(subhalo.sgn))
            f.write('\n' + dash)
            f.write('\nTOTAL STELLAR MASS:     %.3f   \t[Msun]' %np.log10(subhalo.stelmass))     # [pMsun]
            f.write('\nTOTAL GAS MASS:         %.3f   \t[Msun]' %np.log10(subhalo.gasmass))      # [pMsun]
            f.write('\nTOTAL SF GAS MASS:      %.3f   \t[Msun]' %np.log10(subhalo.gasmass_sf))   # [pMsun]
            f.write('\nTOTAL NON-SF GAS MASS:  %.3f   \t[Msun]' %np.log10(subhalo.gasmass_nsf))  # [pMsun]
            f.write('\n' + dash)
            f.write('\nHALFMASS RAD:           %.3f   \t[pkpc]' %subhalo.halfmass_rad)   # [pkpc]
            f.write('\n' + dash)
            f.write('\nKAPPA:                  %.2f' %subhalo.kappa)
            f.write('\nKAPPA GAS:              %.2f' %subhalo.kappa_gas)   
            f.write('\nKAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)  
            f.write('\nKAPPA GAS NSF:          %.2f' %subhalo.kappa_gas_nsf)  
            f.write('\nKAPPA RADIUS CALC:      %s     \t\t[pkpc]' %str(kappa_rad_in))
            f.write('\n' + dash)
            f.write('\nCENTRE OF MASS [pkpc]:')
            f.write('\nHALF-\tCOORDINATE')
            f.write('\nRAD\tSTARS\t\t\tGAS\t\t\tSF\t\t\tNSF')
            i = 0
            while i < len(subhalo.coms['rad']):
                with np.errstate(divide="ignore"):
                    f.write('\n%.1f\t[%.2f, %.2f, %.2f]\t[%.2f, %.2f, %.2f]\t[%.2f, %.2f, %.2f]\t[%.2f, %.2f, %.2f]' %(subhalo.coms['hmr'][i], subhalo.coms['stars'][i][0], subhalo.coms['stars'][i][1], subhalo.coms['stars'][i][2], subhalo.coms['gas'][i][0], subhalo.coms['gas'][i][1], subhalo.coms['gas'][i][2], subhalo.coms['gas_sf'][i][0], subhalo.coms['gas_sf'][i][1], subhalo.coms['gas_sf'][i][2], subhalo.coms['gas_nsf'][i][0], subhalo.coms['gas_nsf'][i][1], subhalo.coms['gas_nsf'][i][2]))
                i += 1
            f.write('\n\nCENTRE OF MASS DISTANCE [pkpc]:')
            f.write('\nHALF-\tDISTANCE (STARS-)')
            f.write('\nRAD\tGAS\tSF\tNSF\tSF-NSF')
            i = 0
            while i < len(subhalo.coms['rad']):
                with np.errstate(divide="ignore"):
                    f.write('\n%.1f\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.coms['hmr'][i], subhalo.coms['stars_gas'][i], subhalo.coms['stars_gas_sf'][i], subhalo.coms['stars_gas_nsf'][i], subhalo.coms['gas_sf_gas_nsf'][i]))        
                i += 1
            f.write('\n' + dash)
            f.write('\nMISALIGNMENT ANGLES [deg]:')
            f.write('\nHALF-\tANGLES (STARS-)\t\t\tPARTICLE COUNT\t\t\tMASS')
            f.write('\nRAD\tGAS\tSF\tNSF\tSF-NSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
            i = 0
            while i < len(subhalo.mis_angles['rad']):
                with np.errstate(divide="ignore"):
                    f.write('\n%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.mis_angles['hmr'][i], subhalo.mis_angles['stars_gas_angle'][i], subhalo.mis_angles['stars_gas_sf_angle'][i], subhalo.mis_angles['stars_gas_nsf_angle'][i], subhalo.mis_angles['gas_sf_gas_nsf_angle'][i], subhalo.particles['stars'][i], subhalo.particles['gas'][i], subhalo.particles['gas_sf'][i], subhalo.particles['gas_nsf'][i], np.log10(subhalo.particles['stars_mass'][i]), np.log10(subhalo.particles['gas_mass'][i]), np.log10(subhalo.particles['gas_sf_mass'][i]), np.log10(subhalo.particles['gas_nsf_mass'][i])))        
                i += 1
            f.write('\n' + dash)
            f.write('\nLOCATION OF CENTRE:   [%.5f,\t%.5f,\t%.5f]\t[pMpc]' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))                                 # [pMpc]
            f.write('\nPERC VELOCITY:        [%.5f,\t%.5f,\t%.5f]\t[pkm/s]' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))
            f.write('\n' + dash)
            
            f.close()
        
        print('')
        
#--------------      
galaxy_render()
#--------------
