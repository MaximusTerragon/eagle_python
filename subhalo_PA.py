import h5py
import numpy as np
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from astropy.constants import G
from read_dataset_tools import read_dataset, read_dataset_dm_mass, read_header
from pafit.fit_kinematic_pa import fit_kinematic_pa
from plotbin.sauron_colormap import register_sauron_colormap
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from subhalo_main import Subhalo_Extract, Subhalo
import eagleSqlTools as sql
from graphformat import graphformat


# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'

# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28

""" 
Purpose
-------
Will create 2dhist to mimic flux pixels, mass-weight 
the velocity and find mean within bins.
            
Calling function
----------------
_, _, _, xbins, ybins, vel_weighted = _weight_histo(subhalo.gn, 
                                                    root_file,
                                                    subhalo.data, 
                                                    'stars',
                                                    viewing_angle,
                                                    viewing_axis,
                                                    resolution, 
                                                    boxradius,
                                                    trim_rad_in)
                   
Input Parameters
----------------
        
gn: int
    Groupnumber for filesave
root_file:  
    root file ei. 'trial_plots'
data_type:
    subhalo.data or subhalo.data_align
particle_type:
    'stars', 'gas', 'gas_sf', 'gas_nsf'
viewing_angle:  float or int
    viewing angle for filesave
viewing_axis:   'x', 'y', 'z'
    Default: 'x'

    Axis FROM WHICH to view the galaxy
    from.
resolution: float
    Default: 1 [pkpc]

    Minimum resolution of bins
boxradius:  float [pkpc]
    2d edges of 2dhistogram
trim_rad:   float [pkpc]
    Radius of particle data that was trimmed.
    This may be 'tworad', in which case the 
    data will have been trimmed by this radius
    but the boxradius may stay unchanged.
quiet: boolean
    Mutes the prints
plot: boolean
    Can plot and savefig


Output Parameters
-----------------

points: [[x, y], [x, y], ...]
    x and y coordinates of CENTRE of bin
    in units of whatever was put in
points_num: [n1, n2, n3, ...]
    Returns number of particles in the bin
points_vel: [v1, v2, v3, ...]
    mass-weighted mean velocity of the bins
    in whatever unit was put in
xbins, ybins: 
    Used in pcolormesh
vel_weighted:
    Velocity weighted bins of 2dhisto, used
    in pcolormesh
"""
def _weight_histo(gn,
                  root_file,
                  data_type, 
                  particle_type,
                  viewing_angle,
                  viewing_axis, 
                  resolution, 
                  boxradius,
                  trim_rad_in,
                  quiet=1, 
                  plot=0):
                 
    # Assign x, y, and z values for histogram2d depending on viewing axis
    if viewing_axis == 'x':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[0] for row in data_type[particle_type]['Velocity']*-1.]
    elif viewing_axis == 'y':
        x_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[1] for row in data_type[particle_type]['Velocity']]
    elif viewing_axis == 'z':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        vel   = [row[2] for row in data_type[particle_type]['Velocity']]
    
    # Find number of pixels along histogram
    pixel = math.ceil(boxradius*2 / resolution)
    
    # Histogram to find counts in each bin
    counts, xbins, ybins = np.histogram2d(y_new, x_new, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
    
    # Histogram to find total mass-weighted velocity in those bins, and account for number of values in each bin
    vel_weighted, _, _ = np.histogram2d(y_new, x_new, weights=vel*data_type[particle_type]['Mass']/np.mean(data_type[particle_type]['Mass']), bins=(xbins, ybins), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
    vel_weighted       = np.divide(vel_weighted, counts, out=np.zeros_like(counts), where=counts!=0)
    
    if not quiet:
        # Print mean mass-weighted velocity of each bin
        print(vel_weighted)
    
    if plot:
        # Plot 2d histogram
        plt.figure()
        graphformat(8, 11, 11, 11, 11, 5, 4)
        im = plt.pcolormesh(xbins, ybins, vel_weighted, cmap='coolwarm', vmin=-150, vmax=150)
        plt.colorbar(im, label='mass-weighted mean velocity', extend='both')
        
        # Formatting
        plt.xlim(-boxradius, boxradius)
        plt.ylim(-boxradius, boxradius)
        
        plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/2dhist_%s_%s_%s_angle_%s.jpeg' %(str(root_file), str(gn), str(particle_type), str(trim_rad_in), str(viewing_axis), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
    # Convert histogram2d data into coords + value
    i, j = np.nonzero(counts)                                     # find indicies of non-0 bins
    a = np.array((xbins[:-1] + 0.5*(xbins[1] - xbins[0]))[i])     # convert bin to centred coords (x, y)
    b = np.array((ybins[:-1] + 0.5*(ybins[1] - ybins[0]))[j])
    
    # Stack coordinates and remove 0 bins
    points_vel = vel_weighted[np.nonzero(counts)]
    points_num = counts[np.nonzero(counts)]            # number of particles within bins
    points     = np.column_stack((b, a))
        
    if not quiet:
        print(points)
        print(points_num)
        print(points_vel)
    
    return points, points_num, points_vel, xbins, ybins, vel_weighted
    
    
""" 
Purpose
-------
Will create vornoi plot using the 2dhist, and output mean vel
values within the bin

Calling function
----------------
points_stars, vel_bin_stars, vor = _voronoi_tessalate(subhalo.gn, 
                                                      root_file,
                                                      subhalo.data, 
                                                      'stars',
                                                      viewing_angle,
                                                      viewing_axis,
                                                      resolution,
                                                      target_particles, 
                                                      boxradius,
                                                      trim_rad_in)
               
Input Parameters
----------------
        
gn: int
    Groupnumber for filesave
root_file:  
    root file ei. 'trial_plots'
data_type:
    subhalo.data or subhalo.data_align
particle_type:
    'stars', 'gas', 'gas_sf', 'gas_nsf'
viewing_angle:  float or int
    viewing angle for filesave
viewing_axis:   'x', 'y', 'z'
    Default: 'x'

    Axis FROM WHICH to view the galaxy
    from.
resolution: float
    Default: 1 [pkpc]

    Minimum resolution of bins
target_particles: int
    Min. number of particles to vornoi bin    
boxradius:  float [pkpc]
    2d edges of 2dhistogram
trim_rad:   float [pkpc]
    Radius of particle data that was trimmed.
    This may be 'tworad', in which case the 
    data will have been trimmed by this radius
    but the boxradius may stay unchanged.
quiet: boolean
    Mutes the prints
plot: boolean
    Can plot and savefig


Output Parameters
-----------------
        
points: [[x, y], [x, y], ...]
    x and y coordinates of CENTRE of bin
    in units of whatever was put in. Will 
    not include infinity points.
vel_bin: [v1, v2, v3, ...]
    mass-weighted mean velocity that was
    grouped by min. particle count
vor: something
    voronoi tessalation details to be 
    fed into the vornoi_plot_2d

"""
def _voronoi_tessalate(gn,
                       root_file,
                       data_type,
                       particle_type,
                       viewing_angle,
                       viewing_axis,
                       resolution,
                       target_particles,
                       boxradius,
                       trim_rad, 
                       quiet=1, 
                       plot=0):
                                  
    # Assign x, y, and z values for histogram2d depending on viewing axis
    if viewing_axis == 'x':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[0] for row in data_type[particle_type]['Velocity']*-1.]
    elif viewing_axis == 'y':
        x_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[1] for row in data_type[particle_type]['Velocity']]
    elif viewing_axis == 'z':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        vel   = [row[2] for row in data_type[particle_type]['Velocity']]
    
    # Find number of pixels along histogram
    pixel = math.ceil(boxradius*2 / resolution)
    
    # Histogram to find counts in each bin
    counts, xbins, ybins = np.histogram2d(y_new, x_new, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
    
    # Histogram to find total mass-weighted velocity in those bins, and account for number of values in each bin
    vel_weighted, _, _ = np.histogram2d(y_new, x_new, weights=vel*data_type[particle_type]['Mass']/np.mean(data_type[particle_type]['Mass']), bins=(xbins, ybins), range=[[-boxradius, boxradius], [-boxradius, boxradius]])

    # Convert histogram2d data into coords + value
    i, j = np.nonzero(counts)                                     # find indicies of non-0 bins
    a = np.array((xbins[:-1] + 0.5*(xbins[1] - xbins[0]))[i])     # convert bin to centred coords (x, y)
    b = np.array((ybins[:-1] + 0.5*(ybins[1] - ybins[0]))[j])
    
    # Stack coordinates and remove 0 bins
    points_vel = vel_weighted[np.nonzero(counts)]
    points_num = counts[np.nonzero(counts)]           # number of particles within bins
    points     = np.column_stack((b, a))

    # Call 2d voronoi binning (don't plot)
    if plot:
        plt.figure()
        graphformat(8, 11, 11, 11, 11, 5, 4)
        _, x_gen, y_gen, _, _, bin_count, vel, _, _ = voronoi_2d_binning(points[:,0], points[:,1], points_vel, points_num, target_particles, plot=1, quiet=quiet, pixelsize=(2*boxradius/pixel), sn_func=None)
        plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/vorbin_output_%s_%s_%s_angle_%s.jpeg' %(str(root_file), str(gn), str(particle_type), str(trim_rad_in), str(viewing_axis), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
    if not plot:
        _, x_gen, y_gen, _, _, bin_count, vel, _, _ = voronoi_2d_binning(points[:,0], points[:,1], points_vel, points_num, target_particles, plot=0, quiet=quiet, pixelsize=(2*boxradius/pixel), sn_func=None)
    
    if not quiet:
        # print number of particles in each bin
        print(bin_count)

    # Create tessalation, append points at infinity to color plot edges
    points     = np.column_stack((x_gen, y_gen))   
    vel_bin    = np.divide(vel, bin_count)     # find mean in each square bin (total velocity / total particles in voronoi bins)
    vor        = Voronoi(points_inf)
    
    return points, vel_bin, vor
    
         
""" 
Purpose
-------
        
"""
#1, 4, 7, 16
#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
def velocity_projection(GroupNumList = np.array([4, 7, 16]), 
                        SubGroupNum  = 0,
                        minangle     = 0,
                        maxangle     = 360,
                        stepangle    = 30,
                        spin_rad_in      = np.arange(0.5, 10.5, 0.5),   # multiples of rad
                        trim_rad_in      = 100,                         # trim particles <radius, False, 'rad', 'tworad', num [pkpc]
                        kappa_rad_in     = 30,                          # calculate kappa for this radius [pkpc]
                        align_rad_in     = False,                       # align galaxy to stellar vector in. this radius [pkpc]
                        boxradius_in     = 40,                          # boxradius of 2dhisto
                        vel_minmax       = 200
                        resolution       = 1,           # bin size [pkpc]
                        target_particles = 10,          # target voronoi bins
                        viewing_axis     = 'x',         # Which axis to view galaxy from
                        root_file = 'trial_plots',      # 'trial_plots' or 'plots'
                        print_galaxy     = True,        # print galaxy stats in chat
                        txt_file         = True, 
                        particle_list_in = ['stars', 'gas', 'gas_sf', 'gas_nsf']
                        orientate_to_axis        = 'z',  # Keep as 'z'
                        viewing_angle            = 0,    # Keep as 0
                        plot_2dhist_graph        = False,
                        plot_voronoi_graph       = False,
                        plot_2dhist_pafit_graph  = False,
                        plot_voronoi_pafit_graph = False,
                        pa_compare               = False,       # plot the voronoi and 2dhist data pa_fit comparison for single galaxy
                        mis_pa_compare           = False,      # plot misangle - pafit for all selected galaxies
                        mis_angle_histo          = False):     # plot histogram of pafit misangles
    
    # Empty dictionaries to collect relevant data
    all_general   = {}          # has total masses, kappa, halfmassrad
    all_misangles = {}          # has all 3d angles
    all_particles = {}          # has mass and particle count data
    all_pafit     = {}    
        
    for GroupNum in GroupNumList:
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)
            
        # Detect keywords of rad, tworad. All in [pkpc]
        if trim_rad_in == 'rad':
            trim_rad = galaxy.halfmass_rad
        elif trim_rad_in == 'tworad':
            trim_rad = 2*galaxy.halfmass_rad
        else:
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
            
        if boxradius_in == 'rad':
            boxradius = galaxy.halfmass_rad
        elif boxradius_in == 'tworad':
            boxradius = 2*galaxy.halfmass_rad
        else:
            boxradius = boxradius_in
                
        
        # int count to broadcast print statements of each galaxy
        i = 0    
        # Use the data we have called to find a variety of properties
        for viewing_angle in np.arange(minangle, maxangle+1, stepangle):
            # If we want the original values, enter 0 for viewing angle
            subhalo = Subhalo(galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas,
                                                viewing_angle,
                                                spin_rad,
                                                trim_rad, 
                                                kappa_rad, 
                                                align_rad,              #align_rad = False
                                                orientate_to_axis,
                                                quiet=True)
                
            # Print galaxy properties for first call
            if i == 0:
                if print_galaxy == True:
                    print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
                    print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))
                    print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad)        
                    print('KAPPA:                  %.2f' %subhalo.kappa)
                    print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
                    print(' HALF-\tANGLES (STARS-)\t\tPARTICLE COUNT\t\t\tMASS')
                    print(' RAD\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
                    for i in [1, 3, len(spin_rad_in)-1]:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            print(' %.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.mis_angles['hmr'][i], subhalo.mis_angles['gas'][i], subhalo.mis_angles['gas_sf'][i], subhalo.mis_angles['gas_nsf'][i], subhalo.particles['stars'][i], subhalo.particles['gas'][i], subhalo.particles['gas_sf'][i], subhalo.particles['gas_nsf'][i], np.log10(subhalo.particles['stars_mass'][i]), np.log10(subhalo.particles['gas_mass'][i]), np.log10(subhalo.particles['gas_sf_mass'][i]), np.log10(subhalo.particles['gas_nsf_mass'][i])))        
                    print('CENTRE [pMpc]:      [%.5f,\t%.5f,\t%.5f]' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))        # [pkpc]
                    print('PERC VEL [pkm/s]:   [%.5f,\t%.5f,\t%.5f]' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))  # [pkm/s]
                    #print('VIEWING ANGLES: ', end='')
                    
            # Print galaxy properties
            if print_galaxy == True:
                print('VIEWING ANGLE: %s' %str(viewing_angle), end=' ')
            
            
            # Function to plot 2dhist-fed data 
            def _plot_2dhist():
                # Initialise figure
                graphformat(8, 11, 11, 11, 11, 3.75, 3)
                fig, axs = plt.subplots(nrows=0, ncols=len(particle_list_in), figsize=(4, 4*len(particle_list_in)), sharex=True, sharey=True)
                
                """# Hist stars  
                _, _, _, xbins, ybins, vel_weighted = _weight_histo(subhalo.gn, root_file, subhalo.data, 'stars', viewing_angle, viewing_axis, resolution, boxradius, trim_rad)
                im = axs[0,0].pcolormesh(xbins, ybins, vel_weighted, cmap='coolwarm', vmin=-vel_minmax, vmax=vel_minmax)
                axs[0,0].set_title('Stars')
                
                # Hist gas
                _, _, _, xbins, ybins, vel_weighted = _weight_histo(subhalo.gn, root_file, subhalo.data, 'gas', viewing_angle, viewing_axis, resolution, boxradius, trim_rad)
                axs[0,1].pcolormesh(xbins, ybins, vel_weighted, cmap='coolwarm', vmin=-vel_minmax, vmax=vel_minmax)
                axs[0,1].set_title('Total gas')
                
                # Hist gas_sf
                _, _, _, xbins, ybins, vel_weighted = _weight_histo(subhalo.gn, root_file, subhalo.data, 'gas_sf', viewing_angle, viewing_axis, resolution, boxradius, trim_rad)
                axs[1,0].pcolormesh(xbins, ybins, vel_weighted, cmap='coolwarm', vmin=-vel_minmax, vmax=vel_minmax)
                axs[1,0].set_title('Starforming gas')
                
                # Hist gas_nsf
                _, _, _, xbins, ybins, vel_weighted = _weight_histo(subhalo.gn, root_file, subhalo.data, 'gas_nsf', viewing_angle, viewing_axis, resolution, boxradius, trim_rad)
                axs[1,1].pcolormesh(xbins, ybins, vel_weighted, cmap='coolwarm', vmin=-vel_minmax, vmax=vel_minmax)
                axs[1,1].set_title('Non-starforming gas')
                """
                """# Graph formatting
                for ax in axs:
                    ax.set_xlim(-boxradius, boxradius)
                    ax.set_ylim(-boxradius, boxradius)
                axs[1,0].set_xlabel('x-axis [pkpc]')
                axs[1,1].set_xlabel('x-axis [pkpc]')
                axs[0,0].set_ylabel('y-axis [pkpc]')
                axs[1,0].set_ylabel('y-axis [pkpc]')
                """
                
                ### 2D HISTO ROUTINE
                j = 0
                for particle_list_in_i in particle_list_in:
                    _, _, _, xbins, ybins, vel_weighted = _weight_histo(subhalo.gn, root_file, subhalo.data, particle_list_in_i, viewing_angle, viewing_axis, resolution, boxradius, trim_rad)
                    im = axs[j].pcolormesh(xbins, ybins, vel_weighted, cmap='coolwarm', vmin=-vel_minmax, vmax=vel_minmax)
                    
                    # Graph formatting 
                    axs[j].set_xlabel('x-axis [pkpc]')
                    axs[j].set_xlim(-boxradius, boxradius)
                    axs[j].set_ylim(-boxradius, boxradius)
                    
                    if particle_list_in_i == 'stars':
                        axs[j].set_title('Stars')
                    elif particle_list_in_i == 'gas':
                        axs[j].set_title('Total gas')
                    elif particle_list_in_i == 'gas_sf':
                        axs[j].set_title('Starforming gas')
                    elif particle_list_in_i == 'gas_nsf':
                        axs[j].set_title('Non-Starforming gas')
                    
                    j = j + 1
                    
                # Graph formatting
                axs[0].set_ylabel('y-axis [pkpc]')
                
                # Annotation
                axs[0].text(-boxradius, boxradius+1, 'resolution: %s, trim_rad: %s, hmr: %s' %(str(resolution), str(trim_rad_in), str(subhalo.halfmass_rad)), fontsize=8)

                # Colorbar
                cax = plt.axes([0.92, 0.11, 0.015, 0.77])
                plt.colorbar(im, cax=cax, label='mass-weighted mean velocity [km/s]', extend='both')
            
                plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/2dhist_%s_%s_%s_angle_%s.jpeg' %(str(root_file), str(subhalo.gn), str(trim_rad_in), str(viewing_axis), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.3)                
                plt.close()
                
            #----------------------------------
            # Plot 2dhist of stars and gas
            if plot_2dhist_graph == True:
                _plot_2dhist()
            #----------------------------------
            

            # Function to plot voronoi-fed data    
            def _plot_voronoi():
                # Initialise figure
                graphformat(8, 11, 11, 11, 11, 3.75, 3)
                fig, axs = plt.subplots(nrows=0, ncols=len(particle_list_in), figsize=(4, 4*len(particle_list_in)), sharex=True, sharey=True)
                
                """# Tessalate stars 
                points_stars, vel_bin_stars, vor = _voronoi_tessalate(subhalo.gn, root_file, subhalo.data, 'stars', viewing_angle, viewing_axis, resolution, target_particles, boxradius, trim_rad_in)
                
                # normalize chosen colormap
                norm = mpl.colors.Normalize(vmin=-vel_minmax, vmax=vel_minmax, clip=True)
                mapper = cm.ScalarMappable(norm=norm, cmap='sauron')         #cmap=cm.coolwarm), cmap='sauron'
                
                # plot Voronoi diagram, and fill finite regions with color mapped from vel value
                voronoi_plot_2d(vor, ax=axs[0,0], show_points=False, show_vertices=False, line_width=0, s=1)
                for r in range(len(vor.point_region)):
                    region = vor.regions[vor.point_region[r]]
                    if not -1 in region:
                        polygon = [vor.vertices[i] for i in region]
                        ax1.fill(*zip(*polygon), color=mapper.to_rgba(vel_bin_stars[r]))    
            
                # Tessalate gas   
                points_gas, vel_bin_gas, vor = _voronoi_tessalate(subhalo.gn, root_file, subhalo.data, 'gas', viewing_angle, viewing_axis, resolution, target_particles, boxradius, trim_rad_in)
                
                # plot Voronoi diagram, and fill finite regions with color mapped from vel value
                voronoi_plot_2d(vor, ax=ax2, show_points=False, show_vertices=False, line_width=0, s=1)
                for r in range(len(vor.point_region)):
                    region = vor.regions[vor.point_region[r]]
                    if not -1 in region:
                        polygon = [vor.vertices[i] for i in region]
                        ax2.fill(*zip(*polygon), color=mapper.to_rgba(vel_bin_gas[r]))
                """
                """# Graph formatting
                for ax in [ax1, ax2]:
                    ax.set_xlim(-boxradius, boxradius)
                    ax.set_ylim(-boxradius, boxradius)
                    ax.set_xlabel('x-axis [pkpc]')
                ax1.set_ylabel('y-axis [pkpc]')
                ax1.set_title('Stars')
                ax2.set_title('Gas')
                """
                
                ### VORONOI TESSALATION ROUTINE
                j = 0
                for particle_list_in_i in particle_list_in:
                    points_particle, vel_bin_particle, vor = _voronoi_tessalate(subhalo.gn, root_file, subhalo.data, particle_list_in_i, viewing_angle, viewing_axis, resolution, target_particles, boxradius, trim_rad_in)
                
                    # normalize chosen colormap
                    norm = mpl.colors.Normalize(vmin=-vel_minmax, vmax=vel_minmax, clip=True)
                    mapper = cm.ScalarMappable(norm=norm, cmap='sauron')         #cmap=cm.coolwarm), cmap='sauron'
                
                    # plot Voronoi diagram, and fill finite regions with color mapped from vel value
                    voronoi_plot_2d(vor, ax=axs[j], show_points=False, show_vertices=False, line_width=0, s=1)
                    for r in range(len(vor.point_region)):
                        region = vor.regions[vor.point_region[r]]
                        if not -1 in region:
                            polygon = [vor.vertices[i] for i in region]
                            ax1.fill(*zip(*polygon), color=mapper.to_rgba(vel_bin_particle[r]))
                            
                    # Graph formatting 
                    axs[j].set_xlabel('x-axis [pkpc]')
                    axs[j].set_xlim(-boxradius, boxradius)
                    axs[j].set_ylim(-boxradius, boxradius)
                    
                    if particle_list_in_i == 'stars':
                        axs[j].set_title('Stars')
                    elif particle_list_in_i == 'gas':
                        axs[j].set_title('Total gas')
                    elif particle_list_in_i == 'gas_sf':
                        axs[j].set_title('Starforming gas')
                    elif particle_list_in_i == 'gas_nsf':
                        axs[j].set_title('Non-Starforming gas')        
                    
                    j = j + 1
                
                # Graph formatting
                axs[0].set_ylabel('y-axis [pkpc]')
                
                # Annotation
                axs[0].text(-boxradius, boxradius+1, 'resolution: %s, trim_rad: %s, target particles: %s, hmr: %s' %(str(resolution), str(trim_rad_in), str(target_particles), str(subhalo.halfmass_rad)), fontsize=8)
                            
                # Colorbar
                cax = plt.axes([0.92, 0.11, 0.015, 0.77])
                plt.colorbar(mapper, cax=cax, label='mass-weighted mean velocity [km/s]', extend='both')
        
                plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/voronoi_%s_%s_%s_angle_%s.jpeg' %(str(root_file), str(subhalo.gn), str(trim_rad_in), str(viewing_axis), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.3)                
                plt.close()
               
            #---------------------------------   
            if plot_voronoi_graph == True:
                _plot_voronoi()
            #--------------------------------- 
            
            
            
            
            
            # Function to plot pa_fit 2dhist-fed data    
            def _pa_fit_2dhist(plot=plot_2dhist_pafit_graph, quiet=1):
                # Extract points
                points_stars, _, vel_bin_stars, _, _, _ = _weight_histo(subhalo.gn, subhalo.viewing_angle, subhalo.stars_coords*1000, subhalo.stars_vel*u.Mpc.to(u.km), subhalo.stars_mass)
                points_gas, _, vel_bin_gas, _, _, _ = _weight_histo(subhalo.gn, subhalo.viewing_angle, subhalo.gas_coords*1000, subhalo.gas_vel*u.Mpc.to(u.km), subhalo.gas_mass)
                
                # Run pa_fit on 2dhist
                if plot:
                    angle_stars, angle_err_stars, velsyst_gas = fit_kinematic_pa(points_stars[:,0], points_stars[:,1], vel_bin_stars, quiet=1, plot=1)
                    plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/PA_stars_2dhist_%s_%s_%s.jpeg' %(str(root_file), str(subhalo.gn), str(boxradius_in), str(calc_spin_rad_in), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.3)
                    plt.close()
                    
                    angle_gas, angle_err_gas, velsyst_gas = fit_kinematic_pa(points_gas[:,0], points_gas[:,1], vel_bin_gas, quiet=1, plot=1)
                    plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/PA_gas_2dhist_%s_%s_%s.jpeg' %(str(root_file), str(subhalo.gn), str(boxradius_in), str(calc_spin_rad_in), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.3)
                    plt.close()
                if not plot:
                    angle_stars, angle_err_stars, velsyst_gas = fit_kinematic_pa(points_stars[:,0], points_stars[:,1], vel_bin_stars, quiet=1, plot=1)
                    angle_gas, angle_err_gas, velsyst_gas = fit_kinematic_pa(points_gas[:,0], points_gas[:,1], vel_bin_gas, quiet=1, plot=1)
                    
                # Find misalignment angle from pa
                pa_fit = abs(angle_stars - angle_gas)
                if pa_fit >= 180:
                    pa_fit = 360 - pa_fit
                pa_fit_error = angle_err_stars + angle_err_gas
                
                if not quiet:
                    print("PA stars: %.1f" %angle_stars)
                    print("PA gas:   %.1f" %angle_gas)
                    print("PA: %.1f +/- %.1f" %(pa_fit, pa_fit_error))
                
                return pa_fit, pa_fit_error
                
            # Function to pa_fit voronoi-fed data
            def _pa_fit_voronoi(plot=plot_voronoi_pafit_graph, quiet=1):
                # Extract points
                points_stars, vel_bin_stars, vor = _voronoi_tessalate(subhalo.gn, subhalo.viewing_angle, subhalo.stars_coords*1000, subhalo.stars_vel*u.Mpc.to(u.km), subhalo.stars_mass)
                points_gas, vel_bin_gas, vor = _voronoi_tessalate(subhalo.gn, subhalo.viewing_angle, subhalo.gas_coords*1000, subhalo.gas_vel*u.Mpc.to(u.km), subhalo.gas_mass)
                
                # Run pa_fit on voronoi
                if plot:       
                    angle_stars, angle_err_stars, velsyst_gas = fit_kinematic_pa(points_stars[:,0], points_stars[:,1], vel_bin_stars, quiet=1, plot=1)
                    #plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/pa_fits/PA_stars_voronoi_%s_%s_%s.jpeg' %(str(root_file), str(subhalo.gn), str(boxradius_in), str(calc_spin_rad_in), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.2)
                    plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/PA_stars_voronoi_%s_%s_%s.jpeg' %(str(root_file), str(subhalo.gn), str(boxradius_in), str(calc_spin_rad_in), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.3)
                    
                    plt.close()
                    angle_gas, angle_err_gas, velsyst_gas = fit_kinematic_pa(points_gas[:,0], points_gas[:,1], vel_bin_gas, quiet=1, plot=plot)
                    #plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/pa_fits/PA_gas_voronoi_%s_%s_%s.jpeg' %(str(root_file), str(subhalo.gn), str(boxradius_in), str(calc_spin_rad_in), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.2)
                    plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/PA_gas_voronoi_%s_%s_%s.jpeg' %(str(root_file), str(subhalo.gn), str(boxradius_in), str(calc_spin_rad_in), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.3)
                    
                    plt.close()
                if not plot:
                    angle_stars, angle_err_stars, velsyst_gas = fit_kinematic_pa(points_stars[:,0], points_stars[:,1], vel_bin_stars, quiet=1, plot=0)
                    angle_gas, angle_err_gas, velsyst_gas = fit_kinematic_pa(points_gas[:,0], points_gas[:,1], vel_bin_gas, quiet=1, plot=plot)
                                   
                # Find misalignment angle from pa
                pa_fit = abs(angle_stars - angle_gas)
                if pa_fit >= 180:
                    pa_fit = 360 - pa_fit
                pa_fit_error = angle_err_stars + angle_err_gas
                
                if not quiet:
                    print("PA: %.1f +/- %.1f" %(pa_fit, pa_fit_error))
                
                return pa_fit, pa_fit_error            
                

            
            
            #-------------------------------- 
            
            # Run PA fit routine once per galaxy to append to galaxy list
            if i == 0:
                # PA fit routine
                #pa_fit, pa_fit_error = _pa_fit_voronoi()
                pa_fit, pa_fit_error = _pa_fit_2dhist()
                
                # Append to data
                pa_fit_list       = np.append(pa_fit_list, pa_fit)
                pa_fit_error_list = np.append(pa_fit_error_list, pa_fit_error)
                
            # Call function to track difference between 2dhist and voronoi
            if pa_compare == 1:
                # PA fit routine
                pa_fit, pa_fit_error = _pa_fit_2dhist()
                pa_fit_2dhist_list      = np.append(pa_fit_2dhist_list, pa_fit)
                pa_fit_2dhist_err_list  = np.append(pa_fit_2dhist_err_list, pa_fit_error)
                
                # PA fit routine
                pa_fit, pa_fit_error = _pa_fit_voronoi()
                pa_fit_voronoi_list     = np.append(pa_fit_voronoi_list, pa_fit)
                pa_fit_voronoi_err_list = np.append(pa_fit_voronoi_err_list, pa_fit_error)
                    
            i = i + 1
            
        #---------------------------------
        # Start of once per galaxy
        print('PA ANGLE [deg]: %.1f +/- %.1f' %(pa_fit, pa_fit_error))             # [deg]
        print('')
        
        # Collect values for each galaxy
        gn_list        = np.append(gn_list, subhalo.gn)
        stelmass_list  = np.append(stelmass_list, subhalo.stelmass)
        kappa_list     = np.append(kappa_list, subhalo_al.kappa) 
        mis_angle_list = np.append(mis_angle_list, subhalo.mis_angle)            

        # Plot pa_fit routine difference between voronoi and 2dhist
        if pa_compare == True:
            plt.figure()
            plt.scatter(np.arange(minangle, maxangle+1, stepangle), pa_fit_2dhist_list, c='b', label='2dhist')
            plt.scatter(np.arange(minangle, maxangle+1, stepangle), pa_fit_voronoi_list, c='r', label='voronoi')
            plt.errorbar(np.arange(minangle, maxangle+1, stepangle), pa_fit_2dhist_list, xerr=None, yerr=pa_fit_2dhist_err_list, c='b', alpha=0.5)
            plt.errorbar(np.arange(minangle, maxangle+1, stepangle), pa_fit_voronoi_list, xerr=None, yerr=pa_fit_voronoi_err_list, c='r', alpha=0.5)
            
            # Formatting
            plt.axhline(subhalo.mis_angle, c='k', ls='--')
            plt.xlabel('Viewing angle')
            plt.ylabel('fit_kinematic_pa')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig('/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/PA_comparison_%s_%s.jpeg' %(str(root_file), str(subhalo.gn), str(boxradius_in), str(calc_spin_rad_in)), dpi=300, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            
            
        # Create txt file with output for that galaxy
        if txt_file == True:
            dash = '-' * 60
            f = open("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/velproj_%s_%s_NEW.txt" %(str(root_file), str(GroupNum), str(calc_spin_rad_in), str(calc_kappa_rad_in)), 'w+')
            f.write(dash)
            f.write('\nGROUP NUMBER:     %s\n' %str(subhalo.gn))
            f.write('\nSUBGROUP NUMBER:  %s\n' %str(subhalo.sgn))
            f.write(dash)
            f.write('\nSTELLAR MASS:     %.3f   [Msun]' %np.log10(subhalo.stelmass))     # [pMsun]
            f.write('\nGAS MASS:         %.3f   [Msun]' %np.log10(subhalo.gasmass))      # [pMsun]
            f.write('\nSF GAS MASS:      %.3f   [Msun]' %np.log10(subhalo.gasmass_sf))   # [pMsun]
            f.write('\nNON-SF GAS MASS:  %.3f   [Msun]' %np.log10(subhalo.gasmass_nsf))  # [pMsun]
            f.write('\n' + dash)
            f.write('\nHALFMASS RAD:     %.3f   [pkpc]' %(1000*subhalo.halfmass_rad))   # [pkpc]
            f.write('\n' + dash)
            f.write('\nKAPPA:             %.2f' %subhalo_al.kappa)  
            f.write('\nKAPPA RADIUS CALC:   %s    [pkpc]' %str(calc_kappa_rad_in))
            f.write('\n' + dash)
            f.write('\nMISALIGNMENT ANGLES [deg]:')
            f.write('\n\tRAD\tGAS\tSF\tNSF')
            i = 0
            while i < len(calc_spin_rad_list_in):
                f.write('\n\t%.1f\t%.1f\t%.1f\t%.1f' %(calc_spin_rad_list_in[i], spin_gas_list[i], spin_sf_list[i], spin_nsf_list[i]))
                i += 1
            f.write('\n' + dash)
            f.write('\nSPIN RADIUS CALC [pMpc]:    %s' %str(calc_spin_rad_in))
            f.write('\nPAFIT [deg]:                %.1f +/- %.1f' %(pa_fit, pa_fit_error))  
            f.write('\nSPIN RADIUS CALC [pMpc]:    %s' %str(calc_spin_rad_in))
            f.write('\nPAFIT RADIUS CALC [pkpc]:   %s' %str(boxradius_in))
            f.write('\nPIXEL RESOLUTION [pkpc]:    %s' %str(resolution))
            f.write('\nVORONOI TARGET PARTICLES:   %s particles' %str(target_particles))
            f.write('\n' + dash)
            f.write('\nCENTRE:          [%.5f,\t%.5f,\t%.5f]\t[pMpc]\n' %(subhalo.centre[0], subhalo_al.centre[1], subhalo_al.centre[2]))                                 # [pMpc]
            f.write('PERC VELOCITY:   [%.5f,\t%.5f,\t%.5f]\t[pkm/s]\n' %(subhalo.perc_vel[0]*u.Mpc.to(u.km), subhalo.perc_vel[1]*u.Mpc.to(u.km), subhalo.perc_vel[2]*u.Mpc.to(u.km))) # [pkm/s]
            
            f.close()
            

    #------------------------------------
    # Start of GroupNum loop
    
    # Append misalignment angles once per galaxy
    data_catalogue['GroupNum'] = gn_list
    data_catalogue['Stelmass'] = stelmass_list
    data_catalogue['Kappa']    = kappa_list
    data_catalogue['MisAngle'] = mis_angle_list
    data_catalogue['PA']       = pa_fit_list
    data_catalogue['PA_error'] = pa_fit_error_list
    #print(data_catalogue)
    
    # Function to plot 3D angle vs 2D projected angle from pa_fit
    if mis_pa_compare == True:
        plt.figure()
        graphformat(8, 11, 11, 11, 11, 5, 5)
        plt.scatter(data_catalogue['MisAngle'], data_catalogue['PA'], c='k', label='M$_{*}$ > 10E9')
        plt.errorbar(data_catalogue['MisAngle'], data_catalogue['PA'], xerr=None, yerr=data_catalogue['PA_error'], ecolor='k', ls='none')
        
        # Plot straight line (expected)
        plt.plot([0, 180], [0, 180], c='k', ls='--')
        
        # Formatting
        plt.xlabel('3D $\Psi$$_{gas-star}$')
        plt.ylabel('fit_kinematic_pa')
        plt.xlim(0, 180)
        #plt.ylim(0, 180)
        plt.legend()
        plt.tight_layout()
        plt.title('3D angle - PA fit 2rad')
        
        plt.savefig('/Users/c22048063/Documents/EAGLE/%s/angle_compare_%s_%s.jpeg' %(str(root_file), str(boxradius_in), str(calc_spin_rad_in)), dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # Plot to create histogram of pafit angles for galaxy selection
    if mis_angle_histo == True:
        
        # Graph initialising and base formatting
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig, ax = plt.subplots(1, 1, figsize=[8, 4])

        # Plot data as histogram
        plt.hist(data_catalogue['PA'], bins=np.arange(0, 181, 10), histtype='bar', edgecolor='black', facecolor='dodgerblue', alpha=0.8)

        # General formatting
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 180, step=30))
        ax.set_ylim(0, 9)
        ax.set_xlabel('$\Psi$$_{gas-star}$')
        ax.set_ylabel('Number')
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

        # Annotations
        ax.text(160, 8, "z = 0", fontsize=10)
        ax.axvline(30, ls='--', lw=0.5, c='k')
        plt.suptitle("L%s: Misalignment angle"%str(mySims[0][1]))
    
        plt.savefig("/Users/c22048063/Documents/EAGLE/%s/Misalignment_PAfit_%s_%s.jpeg" %(str(root_file), str(boxradius_in), str(calc_spin_rad_in)), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.close()
        
        
        