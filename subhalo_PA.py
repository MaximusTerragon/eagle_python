import h5py
import numpy as np
import random
import math
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
import json
import time 
from datetime import datetime
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from astropy.constants import G
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import stats
from read_dataset_tools import read_dataset, read_dataset_dm_mass, read_header
from pafit.fit_kinematic_pa import fit_kinematic_pa
from plotbin.sauron_colormap import register_sauron_colormap
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from subhalo_main import Subhalo_Extract, Subhalo
import eagleSqlTools as sql
from graphformat import graphformat


time_start_main = time.time()

# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28

# Directories of data hdf5 file(s)
#dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
 
 
# Register stuff
register_sauron_colormap()

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
                  plot=0,
                  colormap='coolwarm',
                  debug=False):
                 
    # Assign x, y, and z values for histogram2d depending on viewing axis
    if viewing_axis == 'x':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[0] for row in data_type[particle_type]['Velocity']*-1.]
    elif viewing_axis == 'y':
        x_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[1] for row in data_type[particle_type]['Velocity']*-1.]
    elif viewing_axis == 'z':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        vel   = [row[2] for row in data_type[particle_type]['Velocity']*-1.]
    
    # Find number of pixels along histogram
    pixel = math.ceil(boxradius*2 / resolution)
    
    # Histogram to find counts in each bin
    counts, xbins, ybins = np.histogram2d(y_new, x_new, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
    
    # Histogram to find total mass-weighted velocity in those bins, and account for number of values in each bin
    vel_weighted, _, _ = np.histogram2d(y_new, x_new, weights=vel*data_type[particle_type]['Mass']/np.mean(data_type[particle_type]['Mass']), bins=(xbins, ybins), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
    vel_weighted       = np.divide(vel_weighted, counts, out=np.zeros_like(counts), where=counts!=0)
    
    # Find standard deviaion of each bin
    vel_std, _, _, _ = stats.binned_statistic_2d(y_new, x_new, vel*data_type[particle_type]['Mass']/np.mean(data_type[particle_type]['Mass']), statistic='std', bins=(xbins, ybins))
    
    if debug:
        # Print mean mass-weighted velocity of each bin
        print(vel_weighted)
        print('\nvarience', vel_std)
    
    if plot:
        # Plot 2d histogram
        plt.figure()
        graphformat(8, 11, 11, 11, 11, 5, 4)
        im = plt.pcolormesh(xbins, ybins, vel_weighted, cmap=colormap, vmin=-200, vmax=200)
        plt.colorbar(im, label='mass-weighted mean velocity', extend='both')
        
        # Formatting
        plt.xlim(-boxradius, boxradius)
        plt.ylim(-boxradius, boxradius)
        
        plt.show()
        #plt.savefig('%s/gn_%s_2dhist_%s_%s_%s_angle_%s.jpeg' %(str(root_file), str(gn), str(particle_type), str(trim_rad_in), str(viewing_axis), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
    # Convert histogram2d data into coords + value
    i, j = np.nonzero(counts)                                     # find indicies of non-0 bins
    a = np.array((xbins[:-1] + 0.5*(xbins[1] - xbins[0]))[i])     # convert bin to centred coords (x, y)
    b = np.array((ybins[:-1] + 0.5*(ybins[1] - ybins[0]))[j])
    
    # Stack coordinates and remove 0 bins
    points_vel = vel_weighted[np.nonzero(counts)]
    points_num = counts[np.nonzero(counts)]            # number of particles within bins
    points     = np.column_stack((b, a))
        
    if debug:
        print(' ')
        print(points)
        print(points_num)
        print(points_vel)
    
    
    # plot std
    #im3 = plt.pcolormesh(xbins, ybins, vel_std, cmap='inferno')
    #plt.colorbar(im3, label='std')
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    
    
    
    
    
    
    
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
                       plot=0,
                       colormap='coolwarm'):
                                  
    # Assign x, y, and z values for histogram2d depending on viewing axis
    if viewing_axis == 'x':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[0] for row in data_type[particle_type]['Velocity']*-1.]
    elif viewing_axis == 'y':
        x_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        y_new = [row[2] for row in data_type[particle_type]['Coordinates']]
        vel   = [row[1] for row in data_type[particle_type]['Velocity']*-1.]
    elif viewing_axis == 'z':
        x_new = [row[1] for row in data_type[particle_type]['Coordinates']]
        y_new = [row[0] for row in data_type[particle_type]['Coordinates']*-1.]
        vel   = [row[2] for row in data_type[particle_type]['Velocity']*-1.]
    
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
        plt.savefig('%s/gn_%s_vorbin_output_%s_%s_%s_angle_%s.jpeg' %(str(root_file), str(gn), str(particle_type), str(trim_rad_in), str(viewing_axis), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
    if not plot:
        _, x_gen, y_gen, _, _, bin_count, vel, _, _ = voronoi_2d_binning(points[:,0], points[:,1], points_vel, points_num, target_particles, plot=0, quiet=quiet, pixelsize=(2*boxradius/pixel), sn_func=None)
    
    if not quiet:
        # print number of particles in each bin
        print(bin_count)

    # Create tessalation, append points at infinity to color plot edges
    points     = np.column_stack((x_gen, y_gen))   
    vel_bin    = np.divide(vel, bin_count)     # find mean in each square bin (total velocity / total particles in voronoi bins)
    points_inf = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis=0)    #this simply adds 4 vertices at far distances for visualisation, these points are NOT fed into pafit
    vor        = Voronoi(points_inf)
    #vor        = Voronoi(points)                    this can sometimes fail
    
    return points, vel_bin, vor
    
         
""" 
DESCRIPTION
-----------
- Connects to the SQL database to sample galaxies with stellar mass of 10^9 (about 1000 star particles see Casanueva)
- Finds the total angular momentum of a specific particle type within a given radius, outputs this as a table
- Will mock image each galaxy as just 2dhist or as voronoi, find pa_angles, and then misalignment angles from a viewing axis
   - Can also manually take a manual_GroupNumList which will override the above. This list is kicked out as a print ‘Subhalos plotted’ from subhalo_pa.py


SAMPLE:
------
	Min. galaxy mass 1e9   (galaxy_mass_limit = 1e9)
	Min. sf particle count of 20 within 2 HMR.  (gas_sf_min_particles = 20)
	2D projected angle within 2 HMR (plot_2D_3D = ‘2D’)
	Viewing axis z
	C.o.M max distance of 2.0 pkpc

"""

#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
def velocity_projection(manual_GroupNumList = np.array([]), 
                          SubGroupNum       = 0,
                          galaxy_mass_limit = 10**9,                     # Used in sample
                        kappa_rad_in    = 30,                 # calculate kappa for this radius [pkpc]
                        aperture_rad_in = 30,                 # trim all data to this maximum value before calculations
                        align_rad_in    = False,                             # align galaxy to stellar vector in. this radius [pkpc]
                        orientate_to_axis='z',                                # Keep as 'z'
                        viewing_angle=0,                                      # Keep as 0
                          minangle  = 0, 
                          maxangle  = 0, 
                          stepangle = 30,
                        local_boxradius = True,               # Whether to ignore a fixed boxradius_in, and instead use 1.5x trim_rad_in = spin_rad_in
                          boxradius_in  = None,                 # Graph size of 2dhisto and voronoi projection. If local_boxradius=True, will ignore
                        vel_minmax      = 200,                # Min. max velocity values 200 km/s                                                                                
                                find_uncertainties      = True,                    # whether to find 2D and 3D uncertainties
                                spin_rad_in             = np.array([2.0]),         # np.arange(1.0, 2.5, 0.5), np.array([2.0]) 
                                viewing_axis            = 'z',                     # Which axis to view galaxy from.  DEFAULT 'z'
                                resolution              = 0.7,                     # Bin size for 2dhisto [pkpc].  DEFAULT 2.0
                                target_particles        = 5,                       # Target voronoi bins.  DEFAULT 2.0
                                min_bins                = 10,                       # Minimum number of bins for 2dhisto (and voronoi) to find pafit
                                com_min_distance        = 2.0,                     # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                                gas_sf_min_particles    = 20,                     # Minimum gas sf particles to use galaxy.  DEFAULT 100
                                angle_type_in           = ['stars_gas_sf'],        # PA fits and PA misalignment angles to be found ['stars_gas', 'stars_gas_sf', 'stars_gas_nsf', 'gas_sf_gas_nsf']. Particles making up this data will be automatically found, ei. stars_gas_sf = stars and gas_sf                                                                            
                        root_file = '/Users/c22048063/Documents/EAGLE/plots',      
                        file_format = 'png',
                          print_galaxy       = False,           # Print detailed galaxy stats in chat
                          print_galaxy_short = True,           # Print single-line galaxy stats
                          print_progress     = False,               # Whether to print progress of current task
                          csv_file           = False,              # .csv file will ALL data
                            csv_name = 'data_pa_NEW',
                          showfig        = True,
                          savefig        = True,                 # Whether to save and/or show figures
                            savefigtxt       = '',                # added txt to append to end of savefile
                          debug = False,
                                pa_angle_type_in            = 'voronoi',             # which pa angles to use: '2dhist', 'voronoi', 'both'... compare will use either or both and compare to 3D
                                  plot_2dhist_graph           = False, 
                                  plot_voronoi_graph          = False,
                                  plot_2dhist_pafit_graph     = False,
                                  plot_voronoi_pafit_graph    = False,
                                pa_compare                  = False,               # plot the voronoi and 2dhist data pa_fit comparison for single galaxy
                                    pa_compare_angle_type_in  = 'stars_gas_sf',        # which misangle to compare 
                                    pa_compare_use_rad_in     = 2.0,                   # multiples of halfmass rad
                                pa_radial                   = False,                # plot the voronoi or 2dhist data pa_fit with radius for a single galaxy. Will use spin_rad_in as limits
                                    pa_radial_type_in         = 'both',                 # which pa angles to use: '2dhist, 'voronoi', 'both'
                                    pa_radial_angle_type_in   = 'stars_gas_sf',         # which type of angle to use
                                mis_pa_compare              = True,                # plot misangle - pafit for all selected galaxies
                                mis_angle_histo             = False,                 # plot histogram of pafit misangles USES SAME _type_in, angle_type_in, _use_rad_in as above  
                                    mis_pa_compare_type_in        = 'voronoi',          # what to compare to 3D
                                    mis_pa_compare_angle_type_in  = 'stars_gas_sf',
                                    mis_pa_compare_use_rad_in     = 2.0):               # MAKE SURE THIS IS INCLUDED IN SPIN_RAD_IN
                        
    time_start = time.time()   
    
    # use manual input if values given, else use sample with mstar_limit
    if print_progress:
        print('Creating sample')
        time_start = time.time()
        
    if len(manual_GroupNumList) > 0:
        GroupNumList = manual_GroupNumList
    else:
        sample = Sample(mySims, snapNum, galaxy_mass_limit, 'no')
        print("Sample length: ", len(sample.GroupNum))
        print("  ", sample.GroupNum)
        GroupNumList = sample.GroupNum
        
        
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
    #--------------------------------------------------------------------
    
    
    # Empty dictionaries to collect relevant data
    all_flags         = {}          # has reason why galaxy failed sample
    all_general       = {}          # has total masses, kappa, halfmassrad
    all_coms          = {}
    all_misangles     = {}          # has all 3d angles
    all_particles     = {}          # has all the particle count and mass within rad
    all_data          = {}          # has mass and particle count data
    all_misanglesproj = {}   # has all 2d projected angles from 3d when given a viewing axis and viewing_angle = 0
    all_pafit         = {}          # has all orientation angles, errors, and systematic errors
    all_paangles      = {}          # has all pa angles
    
    
    for GroupNum in tqdm(GroupNumList):
        if debug:
            print('GroupNum: ', GroupNum)
        
        #--------------------------------
        # Create dictionaries
        all_pafit['%s' %str(GroupNum)] = {'2dhist': {}, 'voronoi': {}}
        all_paangles['%s' %str(GroupNum)] = {'2dhist': {}, 'voronoi': {}}
        for viewing_angle_i in np.arange(minangle, maxangle+1, stepangle):
            all_pafit['%s' %str(GroupNum)]['2dhist']['%s' %str(viewing_angle_i)]     = {'rad':[], 'hmr':[], 'stars_angle':[], 'stars_angle_err':[], 'gas_angle':[], 'gas_angle_err':[], 'gas_sf_angle':[], 'gas_sf_angle_err':[], 'gas_nsf_angle':[], 'gas_nsf_angle_err':[]}
            all_pafit['%s' %str(GroupNum)]['voronoi']['%s' %str(viewing_angle_i)]    = {'rad':[], 'hmr':[], 'stars_angle':[], 'stars_angle_err':[], 'gas_angle':[], 'gas_angle_err':[], 'gas_sf_angle':[], 'gas_sf_angle_err':[], 'gas_nsf_angle':[], 'gas_nsf_angle_err':[]}
            all_paangles['%s' %str(GroupNum)]['2dhist']['%s' %str(viewing_angle_i)]  = {'rad':[], 'hmr':[], 'stars_gas_angle':[], 'stars_gas_angle_err':[], 'stars_gas_sf_angle':[], 'stars_gas_sf_angle_err':[], 'stars_gas_nsf_angle':[], 'stars_gas_nsf_angle_err':[], 'gas_sf_gas_nsf_angle':[], 'gas_sf_gas_nsf_angle_err':[]}
            all_paangles['%s' %str(GroupNum)]['voronoi']['%s' %str(viewing_angle_i)] = {'rad':[], 'hmr':[], 'stars_gas_angle':[], 'stars_gas_angle_err':[], 'stars_gas_sf_angle':[], 'stars_gas_sf_angle_err':[], 'stars_gas_nsf_angle':[], 'stars_gas_nsf_angle_err':[], 'gas_sf_gas_nsf_angle':[], 'gas_sf_gas_nsf_angle_err':[]}
        #--------------------------------
        
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Extracting particle data Subhalo_Extract()')
            time_start = time.time()
            
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)

        
        #-------------------------------------------------------------------
        # Automating some later variables to avoid putting them in manually
        spin_rad = spin_rad_in * galaxy.halfmass_rad                #pkpc
        trim_rad = trim_rad_in                                      #rads
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
        if boxradius_in == 'rad':
            boxradius = galaxy.halfmass_rad
        elif boxradius_in == 'tworad':
            boxradius = 2*galaxy.halfmass_rad
        else:
            boxradius = boxradius_in
        #------------------------------------------------------------------
        """ INPUTS:
        angle_selection = [['stars', 'gas'], ['stars', 'gas_sf], [...]]
        spin_rad        HMR array of spins we want
        trim_rad        (same as spin_rad)
        kappa_rad       FALSE or float of pkpc in which to find kappa. Essentially our aperture
        align_rad       FLASE or float of pkpc in which to rotate galaxy to align it to z
        orientate_to_axis='z',                                # Keep as 'z'
        viewing_angle=0,                                      # Keep as 0
        
        NEW:
        aperture_rad         = 30,  sets the maximum analysis radius from centre  
        viewing_axis         = 'z',                     # Which axis to view galaxy from.  DEFAULT 'z'
        com_min_distance     = 2.0,                     # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
        gas_sf_min_particles = 100,                     # Minimum gas sf particles to use galaxy.  DEFAULT 100
        particle_list_in     = ['stars', 'gas', 'gas_sf', ...]
        angle_type_in        = ['stars_gas_sf'],        # PA misalignment angles to be found ['stars_gas', 'stars_gas_sf', 'stars_gas_nsf', 'gas_sf_gas_nsf']
        find_uncertainties
        
        all_general['%s' %str(subhalo.gn)]       = subhalo.general
        all_flags['%s' %str(subhalo.gn)]         = subhalo.flags
        all_particles['%s' %str(subhalo.gn)]     = subhalo.particles
        all_coms['%s' %str(subhalo.gn)]          = subhalo.coms
        all_misangles['%s' %str(subhalo.gn)]     = subhalo.mis_angles
        all_misanglesproj['%s' %str(subhalo.gn)] = subhalo.mis_angles_proj
        
        """
        
        # int count to broadcast print statements of each galaxy
        print_i = 0    
        # Use the data we have called to find a variety of properties
        for viewing_angle in np.arange(minangle, maxangle+1, stepangle):
            
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Running particle data analysis Subhalo()')
                time_start = time.time()
                
            # If we want the original values, enter 0 for viewing angle
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
            if print_i == 0:
                all_general['%s' %str(subhalo.gn)]       = subhalo.general
                all_flags['%s' %str(subhalo.gn)]         = subhalo.flags
                all_particles['%s' %str(subhalo.gn)]     = subhalo.particles
                all_coms['%s' %str(subhalo.gn)]          = subhalo.coms
                all_misangles['%s' %str(subhalo.gn)]     = subhalo.mis_angles
                all_misanglesproj['%s' %str(subhalo.gn)] = subhalo.mis_angles_proj
            
            # Add filter to skip pafitting of galaxy if any basic condition not met
            if len(subhalo.flags) != 0:
                print('Subhalo skipped')
                print(subhalo.flags)
                continue
                
                
            #-------------------------------------------------------------
            # Print galaxy properties for first call
            if print_i == 0:
                if print_galaxy == True:
                    print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
                    print('GALAXY ID:              %i' %subhalo.GalaxyID)
                    print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))
                    print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad)        
                    print('KAPPA:                  %.2f' %subhalo.kappa)
                    print('KAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)
                    print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
                    mask = np.where(np.array(subhalo.coms['hmr'] == 2.0))
                    print('C.O.M 2HMR STARS-SF [pkpc]:  %.2f' %subhalo.coms['stars_gas_sf'][int(mask[0])])
                    #print('VIEWING ANGLES: ', end='')
                elif print_galaxy_short == True:
                    print('GN:\t%s\t|HMR:\t%.2f\t|KAPPA / SF:\t%.2f  %.2f' %(str(subhalo.gn), subhalo.halfmass_rad, subhalo.kappa, subhalo.kappa_gas_sf)) 
                    
            # Print galaxy properties
            if print_galaxy == True:
                if print_i == 0:
                    print('VIEWING ANGLE: %s' %str(viewing_angle), end=' ')
                else:
                    print('%s' %str(viewing_angle), end=' ')
            #-------------------------------------------------------------
        
            # Loop over multiples of halfmass_rad
            print_ii = 0
            for trim_rad_i in trim_rad:
                if debug == True:
                    print('RAD:', trim_rad_i) 
                        
    
                # Function to plot 2dhist-fed data (splinter from _weight_histo)
                def _plot_2dhist(colormap='coolwarm', quiet=1, debug=False):
                    # Initialise figure
                    graphformat(8, 11, 11, 11, 11, 3.75, 3)
                    fig, axs = plt.subplots(nrows=1, ncols=len(particle_list_in), gridspec_kw={'width_ratios': [1, 1.22]}, figsize=(6.5, 3.15), sharex=True, sharey=True)
                    
                    if local_boxradius == True:
                        boxradius = 1.5*trim_rad_i*subhalo.halfmass_rad
                
                    ### 2D HISTO ROUTINE
                    j = 0
                    for particle_list_in_i in particle_list_in:
                        # Check particle count before 2dhisto
                        if subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])] < gas_sf_min_particles:
                            if not quiet:
                                print('\nLOW Particle count %s: %i' %(particle_list_in_i, subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])]))
                            
                            j = j + 1
                            
                            continue
                        else:
                            if not quiet:
                                print('Particle count: %i' %subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])])
                            
                        # Extract points
                        _, _, _, xbins, ybins, vel_weighted = _weight_histo(subhalo.gn, root_file, subhalo.data['%s' %str(trim_rad_i)], particle_list_in_i, viewing_angle, viewing_axis, resolution, boxradius, trim_rad_i*subhalo.halfmass_rad)
                        
                        # Remove 0s and replace with nans
                        vel_weighted[vel_weighted == 0] = math.nan
                        
                        # Labels and plot, and fake plot
                        if particle_list_in_i == 'stars':
                            label = 'Stars'
                        if particle_list_in_i == 'gas':
                            label = 'Total gas'
                        if particle_list_in_i == 'gas_sf':
                            label = 'Star-forming gas'
                        if particle_list_in_i == 'gas_nsf':
                            label = 'Non-star-forming gas'
                        im = axs[j].pcolormesh(xbins, ybins, vel_weighted, cmap=colormap, vmin=-vel_minmax, vmax=vel_minmax)
                        axs[j].plot([-999, -999], [-999, -998], c='k', label=label)
                    
                        # Graph formatting 
                        axs[j].set_xlabel('Position [pkpc]')
                        axs[j].set_xlim(-boxradius, boxradius)
                        axs[j].set_ylim(-boxradius, boxradius)
                        axs[j].set_xticks([-20, -10, 0, 10, 20])
                        axs[j].set_yticks([-20, -10, 0, 10, 20])
                    
                        j = j + 1
                    
                    # Graph formatting
                    axs[0].set_ylabel('Position [pkpc]')
                    for ax in axs:
                        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
                        
                
                    # Annotation
                    plt.suptitle('GalaxyID: %i' %subhalo.GalaxyID)
                    #axs[0].text(-boxradius, boxradius+1, 'resolution: %s pkpc, trim_rad: %s pkpc, hmr: %s pkpc, axis: %s, angle: %s' %(str(resolution), str(trim_rad_i), str(subhalo.halfmass_rad), viewing_axis, str(viewing_angle)), fontsize=8)
                    
                    # Legend
                    for ax in axs:
                        ax.legend(loc='upper right', frameon=False, labelspacing=0.1, fontsize=9, labelcolor='k', handlelength=0)
                        

                    # Colorbar
                    #cax = plt.axes([0.92, 0.11, 0.015, 0.77])
                    plt.colorbar(im, orientation='vertical', label='Mass-weighted mean velocity [km/s]', extend='both')
                    
                    # Other
                    plt.tight_layout()
                    
                    # Savefig
                    if savefig == True:
                        plt.savefig('%s/gn%s_id%s_2dhist_rad%s_ax%s_angle%s%s.%s' %(str(root_file), str(subhalo.gn), str(subhalo.GalaxyID), str(trim_rad_i), str(viewing_axis), str(viewing_angle), savefigtxt, file_format), format=file_format, dpi=300, bbox_inches='tight', pad_inches=0.3)  
                    if showfig == True:
                        plt.show()
                      
                    plt.close()
                
                # Function to plot voronoi-fed data (splinter from _voronoi_tessalate)
                def _plot_voronoi(colormap='coolwarm', quiet=1, debug=False):
                    # Initialise figure
                    graphformat(8, 11, 11, 11, 11, 3.75, 3)
                    fig, axs = plt.subplots(nrows=1, ncols=len(particle_list_in), figsize=(4.5*len(particle_list_in), 4), sharex=True, sharey=True)
                
                    if local_boxradius == True:
                        boxradius = 1.5*trim_rad_i*subhalo.halfmass_rad
                        
                    ### VORONOI TESSALATION ROUTINE
                    j = 0
                    for particle_list_in_i in particle_list_in:
                        # Check particle count before voronoi binning
                        if subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])] < gas_sf_min_particles:
                            if not quiet:
                                print('\nLOW Particle count %s: %i' %(particle_list_in_i, subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])]))
                            
                            j = j + 1 
                            
                            continue
                        else:
                            if not quiet:
                                print('Particle count: %i' %subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])])
                            
                        
                        # Defining break for less than 4 pixels due to fitpa requireing min. of 4. This is run on 2dhisto as voronoi uses it
                        _, _, vel_bin_particle_check, _, _, _ = _weight_histo(subhalo.gn, root_file, subhalo.data['%s' %str(trim_rad_i)], particle_list_in_i, viewing_angle, viewing_axis, resolution, boxradius, trim_rad_i*subhalo.halfmass_rad)
                        # Defining break for less than 4 pixels due to fitpa requireing min. of 4
                        if len(vel_bin_particle_check) < 4:
                            print('Not enough 2dhisto bins to voronoi plot')
                            
                            j = j + 1
                            
                            continue
                            
                        # Extract points
                        points_particle, vel_bin_particle, vor = _voronoi_tessalate(subhalo.gn, root_file, subhalo.data['%s' %str(trim_rad_i)], particle_list_in_i, viewing_angle, viewing_axis, resolution, target_particles, boxradius, trim_rad_i*subhalo.halfmass_rad)
                
                        if debug == True:
                            print('points_particle', points_particle)
                            print('len(points_particle)', len(points_particle))
                            print('len(vel_bin_particle)', len(vel_bin_particle))
                            
                        # normalize chosen colormap
                        norm = mpl.colors.Normalize(vmin=-vel_minmax, vmax=vel_minmax, clip=True)
                        mapper = cm.ScalarMappable(norm=norm, cmap=colormap)         #cmap=cm.coolwarm), cmap='sauron'
                
                        # plot Voronoi diagram, and fill finite regions with color mapped from vel value
                        voronoi_plot_2d(vor, ax=axs[j], show_points=True, show_vertices=False, line_width=0, s=1)
                        for r in range(len(vor.point_region)):
                            region = vor.regions[vor.point_region[r]]
                            if not -1 in region:
                                polygon = [vor.vertices[i] for i in region]
                                axs[j].fill(*zip(*polygon), color=mapper.to_rgba(vel_bin_particle[r]))
                            
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
                    axs[0].text(-boxradius, boxradius+1, 'resolution: %s pkpc, trim_rad: %s, target particles: %s, hmr: %s' %(str(resolution), str(trim_rad_i), str(target_particles), str(subhalo.halfmass_rad)), fontsize=8)
                            
                    # Colorbar
                    cax = plt.axes([0.92, 0.11, 0.015, 0.77])
                    plt.colorbar(mapper, cax=cax, label='mass-weighted mean velocity [km/s]', extend='both')
        
                    if savefig == True:
                        plt.savefig('%s/gn_%s_voronoi_rad%s_ax%s_angle%s%s.jpeg' %(str(root_file), str(subhalo.gn), str(trim_rad_i), str(viewing_axis), str(viewing_angle), savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.3)  
                    if showfig == True:
                        plt.show()
                            
                    plt.close()
                
                #----------------------------------
                # Plot 2dhist of stars and gas
                if plot_2dhist_graph == True:
                    if print_progress:
                        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                        print('Plotting 2dhist graph')
                        time_start = time.time()
                    _plot_2dhist()  
                if plot_voronoi_graph == True:
                    if print_progress:
                        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                        print('Plotting voronoi graph')
                        time_start = time.time()
                    _plot_voronoi()
                #----------------------------------
            
                  
                # Function to plot pa_fit 2dhist-fed data and/or pa angles between components    
                def _pa_fit_2dhist(plot=plot_2dhist_pafit_graph, quiet=1, debug=False):
                    # Append to dictionary
                    all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['rad'].append(trim_rad_i*subhalo.halfmass_rad)
                    all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr'].append(trim_rad_i)
                    
                    if local_boxradius == True:
                        boxradius = 1.5*trim_rad_i*subhalo.halfmass_rad
                        
                    for particle_list_in_i in particle_list_in:
                        # Check particle count before 2dhisto
                        if subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])] < gas_sf_min_particles:
                            if not quiet:
                                print('\nVOID: Particle count %s: %i' %(particle_list_in_i, subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])]))
                            
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            # Flag
                            all_flags['%s' %str(subhalo.gn)].append('\nVOID: Particle count %s: %i' %(particle_list_in_i, subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])]))
                            
                            continue
                        else:
                            if not quiet:
                                print('\nParticle count: %i' %subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])])
                            
                        
                        # Extract points
                        points_particle, _, vel_bin_particle, _, _, _ = _weight_histo(subhalo.gn, root_file, subhalo.data['%s' %str(trim_rad_i)], particle_list_in_i, viewing_angle, viewing_axis, resolution, boxradius, trim_rad_i*subhalo.halfmass_rad)
                        
                        if debug == True:
                            print('points_particle', points_particle)
                            print('len(points_particle)', len(points_particle))
                            print('len(vel_bin_particle)', len(vel_bin_particle))
                            
                        #-------------------------------------------
                        # Defining break for less than 4 (or 10) pixels due to fitpa requireing min. of 4
                        if len(vel_bin_particle) < min_bins:
                            # Append to dictionary                
                            all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            print('VOID: Not enough 2dhisto bins to pafit: %s, %s' %(particle_list_in_i, str(vel_bin_particle)))
                            
                            # Flag
                            all_flags['%s' %str(subhalo.gn)].append('VOID: Not enough 2dhisto bins to pafit: %s, %s' %(particle_list_in_i, str(vel_bin_particle)))
                            continue
                        #---------------------------------------------
                        
                        # Run pa_fit on 2dhist
                        if plot:
                            angle_particle, angle_err_particle, velsyst_particle = fit_kinematic_pa(points_particle[:,0], points_particle[:,1], vel_bin_particle, quiet=1, plot=1)
                            if savefig == True:
                                plt.savefig('%s/gn_%s_PA_2dhist_%s_rad%s_ax%s_angle%s%s.jpeg' %(str(root_file), str(subhalo.gn), str(particle_list_in_i), str(trim_rad_i), str(viewing_axis), str(viewing_angle), savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.3)
                            
                        elif not plot:
                            angle_particle, angle_err_particle, velsyst_particle = fit_kinematic_pa(points_particle[:,0], points_particle[:,1], vel_bin_particle, quiet=1, plot=0)
                            
                            
                        if not quiet:
                            print("PA 2dhist %s: %.1f +/- %.1f" %(particle_list_in_i, angle_particle, angle_err_particle))
                    
                        # Append to dictionary                
                        all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(angle_particle)
                        all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(angle_err_particle)
                        
                        if plot:
                            if showfig == True:
                                plt.show()
                            plt.close()
                        
                    # Append to dictionary
                    all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['rad'].append(trim_rad_i*subhalo.halfmass_rad)
                    all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr'].append(trim_rad_i)
                     
                    # Finding misalignment angles and errors
                    if ('stars_gas' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_gas_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_gas_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA stars-gas: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                    if ('stars_gas_sf' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_sf_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_sf_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_gas_sf_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_gas_sf_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA stars-gas_sf: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                    if ('stars_gas_nsf' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_nsf_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_nsf_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_gas_nsf_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['stars_gas_nsf_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA stars-gas_nsf: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                    if ('gas_sf_gas_nsf' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_sf_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_nsf_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_sf_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_nsf_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_sf_gas_nsf_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['gas_sf_gas_nsf_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA gas_sf-gas_nsf: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                  
                # Function to plot pa_fit voronoi-fed data and/or pa angles between components 
                def _pa_fit_voronoi(plot=plot_voronoi_pafit_graph, quiet=1, debug=False):
                    # Append to dictionary
                    all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['rad'].append(trim_rad_i*subhalo.halfmass_rad)
                    all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['hmr'].append(trim_rad_i)
                    
                    if local_boxradius == True:
                        boxradius = 1.5*trim_rad_i*subhalo.halfmass_rad
                        
                    for particle_list_in_i in particle_list_in:
                        
                        # Check particle count before voronoi binning
                        if subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])] < gas_sf_min_particles:
                            if not quiet:
                                print('\nVOID: Particle count %s: %i' %(particle_list_in_i, subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])]))
                            
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            # Append flag
                            all_flags['%s' %str(subhalo.gn)].append('VOID: Particle count %s: %i' %(particle_list_in_i, subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])]))
                            continue
                        else:
                            if not quiet:
                                print('\nParticle count: %i' %subhalo.particles[particle_list_in_i][int(np.where(subhalo.particles['hmr'] == trim_rad_i)[0])])
                            
                        
                        #--------------------------------
                        # Checks for fail
                        # Defining break for less than 4 pixels due to fitpa requireing min. of 4. This is run on 2dhisto as voronoi uses it
                        _, points_num_check, vel_bin_particle_check, _, _, _ = _weight_histo(subhalo.gn, root_file, subhalo.data['%s' %str(trim_rad_i)], particle_list_in_i, viewing_angle, viewing_axis, resolution, boxradius, trim_rad_i*subhalo.halfmass_rad)
                        
                        if debug == True:
                            print('len(points_num_check)', len(points_num_check))
                            print('len(vel_bin_particle_check)', len(vel_bin_particle_check))
                        
                        # Need at least 4 2dhisto bins for voronoi bin
                        if len(points_num_check) < min_bins:
                            if debug == True:
                                print('Low 2dhisto bins', len(points_num_check))
                            
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            # Flag
                            print('VOID: Not enough 2dhisto bins in voronoi to pafit: %s, %s' %(particle_list_in_i, str(len(points_num_check))))
                            all_flags['%s' %str(subhalo.gn)].append('VOID: Not enough 2dhisto bins in voronoi to pafit: %s, %s' %(particle_list_in_i, str(len(points_num_check))))
                            
                            continue
                            
                        # If all bins already have target_particle count, don't run voronoi as it can't plot it
                        if all(points_num_check_i >= target_particles for points_num_check_i in points_num_check) == True:
                            if debug == True:
                                print('points_num_check S/N', points_num_check)
                            
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            # Flag
                            print('VOID: All points already meet S/N: %s. Voronoi plot fail' %particle_list_in_i)
                            all_flags['%s' %str(subhalo.gn)].append('VOID: All points already meet S/N: %s. Voronoi plot fail' %particle_list_in_i)
                            
                            continue
                        #--------------------------------
                        
                        
                        # Extract points
                        points_particle, vel_bin_particle, vor = _voronoi_tessalate(subhalo.gn, root_file, subhalo.data['%s' %str(trim_rad_i)], particle_list_in_i, viewing_angle, viewing_axis, resolution, target_particles, boxradius, trim_rad_i*subhalo.halfmass_rad)
                        
                        if debug == True:
                            #print('points_particle', points_particle)
                            print('vor len(points_particle)', len(points_particle))
                            print('vor len(vel_bin_particle)', len(vel_bin_particle))
                            
                        #--------------------------------
                        # Check to see that we have minimum of min_bin points to pafit (ideally 10)
                        if len(points_particle) < min_bins:
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            # Flag
                            print('VOID: Not enough voronoi bins to pafit: %s, %s' %(particle_list_in_i, str(len(points_particle))))
                            all_flags['%s' %str(subhalo.gn)].append('VOID: Not enough voronoi bins to pafit: %s, %s' %(particle_list_in_i, str(len(points_particle))))
                            
                            continue
                        #--------------------------------
                            
                        # Run pa_fit on voronoi
                        if plot:
                            angle_particle, angle_err_particle, velsyst_particle = fit_kinematic_pa(points_particle[:,0], points_particle[:,1], vel_bin_particle, quiet=1, plot=1)
                            if savefig == True:
                                plt.savefig('%s/gn_%s_PA_voronoi_%s_rad%s_ax%s_angle%s%s.jpeg' %(str(root_file), str(subhalo.gn), str(particle_list_in_i), str(trim_rad_i), str(viewing_axis), str(viewing_angle), savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.3)
                        
                        elif not plot:
                            angle_particle, angle_err_particle, velsyst_particle = fit_kinematic_pa(points_particle[:,0], points_particle[:,1], vel_bin_particle, quiet=1, plot=0)
                    
                        if not quiet:
                            print("PA voronoi %s: %.1f +/- %.1f" %(particle_list_in_i, angle_particle, angle_err_particle))
                        
                        if plot:
                            if showfig == True:
                                plt.show()
                            plt.close()
                        
                        
                        # Append to dictionary
                        all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(angle_particle)
                        all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(angle_err_particle)

                    # Append rad to dictionary
                    all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['rad'].append(trim_rad_i*subhalo.halfmass_rad)
                    all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['hmr'].append(trim_rad_i)
                    
                    # Finding misalignment angles and errors
                    if ('stars_gas' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_gas_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_gas_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA stars-gas: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                    if ('stars_gas_sf' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_sf_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_sf_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_gas_sf_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_gas_sf_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA stars-gas_sf: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                    if ('stars_gas_nsf' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_nsf_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_nsf_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_gas_nsf_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['stars_gas_nsf_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA stars-gas_nsf: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                    if ('gas_sf_gas_nsf' in angle_type_in):
                        # angle for stars_gas
                        pa_fit_angle = abs(all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_sf_angle'][print_ii] - all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_nsf_angle'][print_ii])
                        if pa_fit_angle >= 180:
                            pa_fit_angle = 360 - pa_fit_angle
                    
                        # error for stars_gas
                        pa_fit_angle_error = all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_sf_angle_err'][print_ii] + all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_nsf_angle_err'][print_ii]
                    
                        # Append to dictionary
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_sf_gas_nsf_angle'].append(pa_fit_angle)
                        all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['gas_sf_gas_nsf_angle_err'].append(pa_fit_angle_error)
                    
                        if not quiet:
                            print("PA gas_sf-gas_nsf: %.1f +/- %.1f" %(pa_fit_angle, pa_fit_angle_error))
                
                #----------------------------------
                # Run PA fit routine once per galaxy to append to galaxy list (viewing_angle = 0)
                #if print_i == 0:
                # PA fit routine - determines what pa angles are recorded
                if pa_angle_type_in == '2dhist':
                    if print_progress:
                        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                        print('Fitting 2dhist pafit')
                        time_start = time.time() 
                    
                    _pa_fit_2dhist()
                elif pa_angle_type_in == 'voronoi':
                    if print_progress:
                        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                        print('Fitting voronoi pafit')
                        time_start = time.time()
                    
                    _pa_fit_voronoi()
                elif pa_angle_type_in == 'both':
                    if print_progress:
                        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                        print('Fitting 2dhist pafit')
                    
                    _pa_fit_2dhist()
                    
                    if print_progress:
                        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                        print('Fitting voronoi pafit')
                        time_start = time.time()
                    
                    _pa_fit_voronoi()
                #----------------------------------
                
                
                print_ii = print_ii + 1
            
            
            #===============================================
            # Start of end of rad iterations
            print_i = print_i + 1
        
        
        #===================================================
        # Start of once per galaxy  (end of viewing angles)
        
        # Add filter to skip pafitting of galaxy if any basic condition not met
        if len(subhalo.flags) != 0:
            #print('    Subhalo skipped')
            #print(subhalo.flags)
            continue       
        
        
        # Plot pa_fit routine difference between voronoi and 2dhist, with 3D for a single galaxy
        def _pa_compare(quiet=1, debug=False):
            # Collect values to plot
            hist_points     = []
            hist_points_err = []
            voro_points     = []
            voro_points_err = []
            
            # Find row for which hmr = pa_compare_use_rad_in (=2.)
            mask = np.where(np.array(all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(maxangle)]['hmr']) == pa_compare_use_rad_in)
            
            for viewing_angle_i in np.arange(minangle, maxangle+1, stepangle):
                hist_points.append(all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s_angle' %pa_compare_angle_type_in][int(mask[0])])
                hist_points_err.append(all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s_angle_err' %pa_compare_angle_type_in][int(mask[0])])
                voro_points.append(all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle_i)]['%s_angle' %pa_compare_angle_type_in][int(mask[0])])
                voro_points_err.append(all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle_i)]['%s_angle_err' %pa_compare_angle_type_in][int(mask[0])])
               
            if debug == True:
                print('\nhist_points', hist_points)
                print('\nhist points err', hist_points_err)
                print('\nvoro_points', voro_points)
                print('\nvoro_points err', voro_points_err) 
            
            # Initialise figure
            plt.figure()
            graphformat(8, 11, 11, 11, 11, 3.75, 3)
                
            # Plot scatter and error bars for both 2dhist and voronoi
            plt.errorbar(np.arange(minangle, maxangle+1, stepangle), hist_points, xerr=None, yerr=hist_points_err, label='2dhist', alpha=0.6, ls='none', ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
            plt.errorbar(np.arange(minangle, maxangle+1, stepangle), voro_points, xerr=None, yerr=voro_points_err, label='voronoi', alpha=0.6, ls='none', ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                   
            # Formatting
            plt.axhline(subhalo.mis_angles['%s_angle' %pa_compare_angle_type_in][int(mask[0])], c='k', ls='--')
            
            plt.xlabel('Viewing angle')
            plt.ylabel('fit_kinematic_pa')
            plt.title('GroupNum %s: %s, %s hmr' %(str(subhalo.gn), pa_compare_angle_type_in, str(pa_compare_use_rad_in)))
            plt.legend()
            plt.tight_layout()
            
            # pa_compare_type_in
            if savefig == True:
                plt.savefig('%s/gn_%s_NEW_PAcompare_%s_rad%s%s.jpeg' %(str(root_file), str(subhalo.gn), pa_compare_angle_type_in, str(pa_compare_use_rad_in), savetxtfig), dpi=300, bbox_inches='tight', pad_inches=0.2)
            if showfig == True:
                plt.show()
            plt.close()      
            
        #------------------------
        if pa_compare == True:
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Running _pa_compare()')
                time_start = time.time()
            _pa_compare()
        #------------------------
        
        
        # Plot for a single galaxy showing how misalignment angle varies with increasing radius
        def _pa_radial(rad_type_plot='hmr', use_projected_plot=True, quiet=0, debug=False):
            # Collect values to plot
            hist_points     = []
            hist_points_err = []
            voro_points     = []
            voro_points_err = []
            
            # Initialise figure
            plt.figure()
            graphformat(8, 11, 11, 11, 11, 3.75, 3)
            plt.xlim(min(spin_rad_in), max(spin_rad_in))
            plt.xticks(np.arange(min(spin_rad_in), max(spin_rad_in)+1, 1))
            
            # formatting 
            if rad_type_plot == 'hmr':
                plt.xlabel('Halfmass rad')
            elif rad_type_plot == 'rad':
                plt.xlabel('Distance from centre [pkpc]')
            plt.ylabel('fit_kinematic_pa')
            plt.ylim(0, 180)
            
            if pa_radial_type_in == '2dhist':
                rad_points      = all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s' %rad_type_plot]
                hist_points     = all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s_angle' %pa_radial_angle_type_in]
                hist_points_err = all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s_angle_err' %pa_radial_angle_type_in]
                proj_points     = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle' %pa_radial_angle_type_in]
                proj_points_lo  = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %pa_radial_angle_type_in][0]
                proj_points_hi  = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %pa_radial_angle_type_in][1]
                
                if debug == True:
                    print(all_misanglesproj.items())
                    print(' ')
                    print('rad ', rad_points)
                    print('proj', proj_points)
                    print('proj err')
                    print(proj_points_lo)
                    print(proj_points_hi)
                    print('\nhist', hist_points)
                    print('hist err', hist_points_err)
                if not quiet:
                    print(' ')
                    print('rad ', rad_points)
                    print('proj', proj_points)
                    print('hist', hist_points)
                
                # Plot 2D projected rad
                if use_projected_plot == True:
                    plt.plot(rad_points, pa_points, label='3D projected', alpha=1.0, ms=2, lw=1)
                    plt.fill_between(rad_points, pa_points_lo, pa_points_hi, alpha=0.3, facecolor='grey')
                    
                # Plot scatter and errorbars
                plt.errorbar(rad_points, hist_points, xerr=None, yerr=hist_points_err, label='2dhist', alpha=0.8, ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                
                # Formatting
                plt.title('GroupNum %s: %s, %s' %(str(subhalo.gn), pa_radial_type_in, pa_radial_angle_type_in))
                plt.legend()
                plt.tight_layout()
                
                # savefig
                if savefig == True:
                    plt.savefig('%s/RadialPA_gn%s_%s_%s%s.jpeg' %(str(root_file), str(subhalo.gn), pa_radial_type_in, pa_radial_angle_type_in, savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
                if showfig == True:
                    plt.show()
                plt.close()
            if pa_radial_type_in == 'voronoi':
                rad_points      = all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle_i)]['%s' %rad_type_plot]
                voro_points     = all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle_i)]['%s_angle' %pa_radial_angle_type_in]
                voro_points_err = all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle_i)]['%s_angle_err' %pa_radial_angle_type_in]
                proj_points     = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle' %pa_radial_angle_type_in]
                proj_points_lo  = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %pa_radial_angle_type_in][0]
                proj_points_hi  = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %pa_radial_angle_type_in][1]
                
                if debug == True:
                    print(all_misanglesproj.items())
                    print(' ')
                    print('rad ', rad_points)
                    print('proj', proj_points)
                    print('proj err')
                    print(proj_points_lo)
                    print(proj_points_hi)
                    print('\nvoro', voro_points)
                    print('voro err', voro_points_err)
                if not quiet:
                    print(' ')
                    print('rad ', rad_points)
                    print('proj', proj_points)
                    print('hist', hist_points)
                
                # Plot 2D projected rad
                if use_projected_plot == True:
                    plt.plot(rad_points, pa_points, label='2D projected', alpha=1.0, ms=2, lw=1)
                    plt.fill_between(rad_points, pa_points_lo, pa_points_hi, alpha=0.3, facecolor='grey')
                    
                # Plot scatter and errorbars
                plt.errorbar(rad_points, voro_points, xerr=None, yerr=voro_points_err, label='voronoi', alpha=0.8, ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                
                # Formatting
                plt.title('GroupNum %s: %s, %s' %(str(subhalo.gn), pa_radial_type_in, pa_radial_angle_type_in, ))
                plt.legend()
                plt.tight_layout()
                
                # savefig
                if savefig == True:
                    plt.savefig('%s/RadialPA_gn%s_%s_%s%s.jpeg' %(str(root_file), str(subhalo.gn), pa_radial_type_in, pa_radial_angle_type_in, savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
                if showfig == True:
                    plt.show()
                plt.close()
            if pa_radial_type_in == 'both':
                rad_points      = all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s' %rad_type_plot]
                hist_points     = all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s_angle' %pa_radial_angle_type_in]
                hist_points_err = all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle_i)]['%s_angle_err' %pa_radial_angle_type_in]
                voro_points     = all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle_i)]['%s_angle' %pa_radial_angle_type_in]
                voro_points_err = all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle_i)]['%s_angle_err' %pa_radial_angle_type_in]
                proj_points     = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle' %pa_radial_angle_type_in]
                proj_points_lo  = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %pa_radial_angle_type_in][0]
                proj_points_hi  = all_misanglesproj['%s' %str(subhalo.gn)][viewing_axis]['%s_angle_err' %pa_radial_angle_type_in][1]
                
                if debug == True:
                    print(all_misanglesproj.items())
                    print(' ')
                    print('rad ', rad_points)
                    print('proj', proj_points)
                    print('proj err')
                    print(proj_points_lo)
                    print(proj_points_hi)
                    print('\nhist', hist_points)
                    print('hist err', hist_points_err)
                    print('\nvoro', voro_points)
                    print('voro err', voro_points_err)
                if not quiet:
                    print(' ')
                    print('rad ', rad_points)
                    print('proj', proj_points)
                    print('hist', hist_points)
                
                # Plot 2D projected rad
                if use_projected_plot == True:
                    plt.plot(rad_points, pa_points, label='2D projected', alpha=1.0, ms=2, lw=1)
                    plt.fill_between(rad_points, pa_points_lo, pa_points_hi, alpha=0.3, facecolor='grey')
                    
                # Plot scatter and errorbars
                plt.errorbar(rad_points, hist_points, xerr=None, yerr=hist_points_err, label='2dhist', alpha=0.8, ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                plt.errorbar(rad_points, voro_points, xerr=None, yerr=voro_points_err, label='voronoi', alpha=0.8, ms=2, capsize=4, elinewidth=1, markeredgewidth=1)
                
                # Formatting
                plt.title('GroupNum %s: %s, %s' %(str(subhalo.gn), pa_radial_type_in, pa_radial_angle_type_in, ))
                plt.legend()
                plt.tight_layout()
                
                # savefig
                if savefig == True:
                    plt.savefig('%s/RadialPA_gn%s_%s_%s%s.jpeg' %(str(root_file), str(subhalo.gn), pa_radial_type_in, pa_radial_angle_type_in, savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
                if showfig == True:
                    plt.show()
                plt.close()
              
        #-----------------------
        if pa_radial == True:
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Running _pa_radial()')
                time_start = time.time()
                
            _pa_radial()
        #-----------------------
        
        
    #=====================================
    # Start of GroupNum loop
    
    """ DATA AVAILABLE
    all_general[GroupNum]
        ['gn']
        ['sgn']
        ['GalaxyID']
        ['stelmass']      - subhalo.stelmass      HAS ALL CORE DATA ON SUBHALO
        ['gasmass']       - subhalo.gasmass       
        ['gasmass_sf']    - subhalo.gasmass_sf
        ['gasmass_nsf']   - subhalo.gasmass_nsf
        ['halfmass_rad']  - subhalo.halfmass_rad
        ['kappa']         - subhalo.kappa               (if not flagged)
        ['kappa_gas']     - subhalo.kappa_gas           (if not flagged)
        ['kappa_gas_sf']  - subhalo.kappa_gas_sf        (if not flagged)
        ['kappa_gas_nsf'] - subhalo.kappa_gas_nsf       (if not flagged)
    
    all_flags[GroupNum]
        [arrays of any flags raised during Subhalo()]       - Will have len(all_flags[GroupNum]) == 0 if all good at particle stage
    
    all_misangles[GroupNum]     -   Has aligned/rotated 3D misalignment angles between stars 
        ['rad']            - [pkpc]
        ['hmr']            - multiples of halfmass_rad
        ['stars_gas_angle']      - [deg]
        ['stars_gas_sf_angle']   - [deg]
        ['stars_gas_nsf_angle']  - [deg]
        ['gas_sf_gas_nsf_angle'] - [deg]
        ['stars_gas_angle_err']      - [lo, hi] [deg]       ^
        ['stars_gas_sf_angle_err']   - [lo, hi] [deg]
        ['stars_gas_nsf_angle_err']  - [lo, hi] [deg]       (assuming it was passed into _main)
        ['gas_sf_gas_nsf_angle_err'] - [lo, hi] [deg]       ^
    
    all_misanglesproj['GroupNum']   -   Has all 2D projected angles from 3D for a given viewing axis
        ['x']                 - Which viewing axis (assumes viewing_angle & minangle = 0 )
        ['y']
        ['z']
            ['rad']             - [pkpc] (trim_rad)
            ['hmr']             - multiples of hmr
            ['stars_gas_angle']             - [deg]
            ['stars_gas_sf_angle']          - [deg]
            ['stars_gas_nsf_angle']         - [deg]
            ['gas_sf_gas_nsf_angle']        - [deg]
            ['stars_gas_angle_err']      - [lo, hi] [deg]       ^
            ['stars_gas_sf_angle_err']   - [lo, hi] [deg]
            ['stars_gas_nsf_angle_err']  - [lo, hi] [deg]       (assuming it was passed into _main)
            ['gas_sf_gas_nsf_angle_err'] - [lo, hi] [deg]       ^
        
    all_data[GroupNum]    -     Has aligned/rotated particle count and mass within spin_rad_in's:
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['stars']        - count
        ['gas']          - count
        ['gas_sf']       - count
        ['gas_nsf']      - count
        ['stars_mass']   - [Msun]
        ['gas_mass']     - [Msun]
        ['gas_sf_mass']  - [Msun]
        ['gas_nsf_mass'] - [Msun]
    
    all_pafit['GroupNum']   -   Has all PA fit angles of various components
        ['voronoi']                 - Whether voronoi or 2dhist was fed
        ['2dhist']
            [viewing_angle]
                ['rad']             - [pkpc] (trim_rad)
                ['hmr']             - multiples of hmr
                ['stars_angle']             - [deg]
                ['stars_angle_err']         - [+/-deg]
                ['gas_angle']               - [deg]
                ['gas_angle_err']           - [+/-deg]
                ['gas_sf_angle']            - [deg]
                ['gas_sf_angle_err']        - [+/-deg]
                ['gas_nsf_angle']           - [deg]
                ['gas_nsf_angle_err']       - [+/-deg]
    
    all_paangles['GroupNum']    -   Has all PA misalignments of various groupings
        ['voronoi']
        ['2dhist']
            [viewing_angle]
                ['rad']             - [pkpc] (trim_rad)
                ['hmr']             - multiples of hmr
                ['stars_gas_angle']             - [deg]
                ['stars_gas_angle_err']         - [+/-deg]
                ['stars_gas_sf_angle']             - [deg]
                ['stars_gas_sf_angle_err']         - [+/-deg]
                ['stars_gas_nsf_angle']             - [deg]
                ['stars_gas_nsf_angle_err']         - [+/-deg]
                ['gas_sf_gas_nsf_angle']             - [deg]
                ['gas_sf_gas_nsf_angle_err']         - [+/-deg]
    """
    
    #=====================================
    if csv_file: 
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Writing CSV')
            time_start = time.time()
            
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
        csv_dict = {'all_general': all_general, 'all_misangles': all_misangles, 'all_misanglesproj': all_misanglesproj, 'all_coms': all_coms, 'all_particles': all_particles, 'all_pafit': all_pafit, 'all_paangles': all_paangles, 'all_flags': all_flags}
        csv_dict.update({'function_input': str(inspect.signature(velocity_projection))})
        
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%s_%s.csv' %(root_file, csv_name, str(datetime.now())), 'w'), cls=NumpyEncoder)
        
        """# Reading JSON file
        dict_new = json.load(open('%s/%s.csv' %(root_file, csv_name), 'r'))
        # example nested dictionaries
        new_general = dict_new['all_general']
        new_misanglesproj = dict_new['all_misanglesproj']
        # example accessing function input
        function_input = dict_new['function_input']"""
    
    
    #=====================================
    # Function to plot 3D angle vs 2D projected angle from pa_fit for a selection of galaxies
    def _mis_pa_compare(useangle='2D', quiet=0, debug=False):
        assert (minangle == 0) & (viewing_angle == 0), "minangle or viewing_angle is not 0"
        
        # Collect values to plot
        mis_points      = []       #3d
        mis_points_err  = []
        mis_points_proj = []
        mis_points_proj_err = []
        pa_points       = []       #2d projected from pafit
        pa_points_err   = []
        GroupNumPlot    = []
        GroupNumNotPlot = []
        
        # Pick first item in list with no flags to make hmr masks
        
        if debug == True:
            print('\n all flags')
            print(all_flags.items())
            print('%s' %GroupNumList)
        
        for GroupNum in GroupNumList:
            # If galaxy not flagged, use galaxy
            if len(all_flags['%s' %str(GroupNum)]) == 0:
                # Mask correct integer (formatting weird but works)
                mask_rad = int(np.where(np.array(all_paangles['%s' %GroupNum][mis_pa_compare_type_in]['0']['hmr']) == mis_pa_compare_use_rad_in)[0])
                
                # Analytical misalignment angles
                mis_points.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %mis_pa_compare_angle_type_in][mask_rad])
                mis_points_proj.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %mis_pa_compare_angle_type_in][mask_rad])
                
                # Analytical 3D and 2D errors [lo, hi]
                mis_points_err.append(all_misangles['%s' %str(GroupNum)]['%s_angle_err' %mis_pa_compare_angle_type_in][mask_rad])
                mis_points_proj_err.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle_err' %mis_pa_compare_angle_type_in][mask_rad])
                
                pa_points.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle' %mis_pa_compare_angle_type_in][mask_rad])
                pa_points_err.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle_err' %mis_pa_compare_angle_type_in][mask_rad])
                
                GroupNumPlot.append(GroupNum)
                
            else:
                GroupNumNotPlot.append(GroupNum)
                
        
        if debug == True:
            print('GroupNumPlot', GroupNumPlot)
            print('GroupNumNotPlot', GroupNumNotPlot)
            print(mis_points)
            print(mis_points_proj)
            print(' ')
            print(pa_points)
            print(pa_points_err)
            
        # Print statements
        print('\nInitial sample:   ', len(GroupNumList))
        if not quiet:
            print(' ', GroupNumList)
        print('\nFinal sample:   ', len(GroupNumPlot))
        if not quiet:
            print(' ', GroupNumPlot)  
        print('\nNot in sample:   ', len(GroupNumNotPlot)) 
        if not quiet:
            print(' ', GroupNumNotPlot)
        print('\n\n==========================================')
        print('Spin radius, axis:   %.1f [pkpc], %s' %(mis_pa_compare_use_rad_in, viewing_axis))
        print('Min. stellar mass:   %.1f [log10 M*]' %np.log10(galaxy_mass_limit))
        print('Min. sf particles:   %i' %gas_sf_min_particles)
        print('Min. c.o.m distance: %.1f [pkpc]' %com_min_distance)
        print('---------------------------------------')
        print('Initial sample:  ', len(GroupNumList))
        print('Final sample:    ', len(GroupNumPlot))
        print('Not in sample:   ', len(GroupNumNotPlot)) 
        print('==========================================')
            
        
        # Initialise figure
        graphformat(8, 11, 11, 11, 11, 3.15, 2.90)
        plt.figure()

        # Labels
        if 'stars_gas' in angle_type_in:
            label = 'Total gas'
            plot_color = 'dodgerblue'
        if 'stars_gas_sf' in angle_type_in:
            label = 'Star-forming gas'
            plot_color = 'darkorange'
        if 'stars_gas_nsf' in angle_type_in:
            label = 'Non-star-forming gas'
            plot_color = 'indigo'
            
        # Converting errors from [[value, value], [value, value]] to [[lo, lo, lo ...], [hi, hi, hi ...]] 
        xerr        = abs(np.array(mis_points_err).T - mis_points)
        xerr_proj   = abs(np.array(mis_points_proj_err).T - mis_points_proj)
        
        # Plot scatter comparing 3D (mis_points) to extracted value from pafit routine (d_points/err)
        if useangle == '2D':
            plt.errorbar(mis_points_proj, pa_points, xerr=xerr_proj, yerr=pa_points_err, ecolor='k', ls='none', alpha=0.5, zorder=4, capsize=2, elinewidth=0.5, markeredgewidth=0.5)
            plt.scatter(mis_points_proj, pa_points, c='k', s=2, zorder=5, label=label)  #M$_{*}$ > 10E9 
        if useangle == '3D':
            plt.errorbar(mis_points, pa_points, xerr=xerr, yerr=pa_points_err, ecolor='k', ls='none', alpha=0.5, zorder=4, capsize=2, elinewidth=0.5, markeredgewidth=5)
            plt.scatter(mis_points, pa_points, c='k', s=2, zorder=5, label=label)  #label='M$_{*}$ > 10E9', 

        
        ### General formatting
        if useangle == '2D':
            plt.xlabel('Stellar-gas PA misalignment') #2D projected $\Psi$$_{gas-star}$
            #plt.title('2D projected angle - PA fit\nmass: %.1f, angle: %s, type: %s, hmr: %s, galaxies: %s/%s, \nparticles: %s, com: %s, bins: %s, ax: %s' %(np.log10(galaxy_mass_limit), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(len(GroupNumPlot)), str(len(GroupNumList)), str(gas_sf_min_particles), str(com_min_distance), str(min_bins), viewing_axis))
        if useangle == '3D':
            plt.xlabel('3D absolute $\Psi$$_{gas-star}$')
            #plt.title('3D absolute angle - PA fit\nmass: %.1f, angle: %s, type: %s, hmr: %s, galaxies: %s/%s, \nparticles: %s, com: %s, bins: %s, ax: %s' %(np.log10(galaxy_mass_limit), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(len(GroupNumPlot)), str(len(GroupNumList)), str(gas_sf_min_particles), str(com_min_distance), str(min_bins), viewing_axis))  
        plt.ylabel('$\Psi$$_{gas-star}$')
        plt.xlim(0, 180)
        plt.ylim(0, 180)
        plt.xticks(np.arange(0, 181, 30))
        plt.yticks(np.arange(0, 181, 30))
        plt.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
        # Annotations
        # Plot straight line (expected)
        plt.plot([0, 180], [0, 180], color='r', ls='--', lw=1, alpha=0.3, zorder=10)
        
        # Legend
        plt.legend(loc='lower right', frameon=False, labelspacing=0.1, fontsize=9, handlelength=0)
        
        # Other
        plt.tight_layout()
        
        # Savefig
        if savefig == True:
            if useangle == '2D':
                plt.savefig('%s/MisPAcompare_2D_mass%s_%s_%s_rad%s_part%s_com%s_bins%s_ax%s%s.%s' %(str(root_file), np.log10(galaxy_mass_limit), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(com_min_distance), str(min_bins), viewing_axis, savefigtxt, file_format), format=file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
            if useangle == '3D':
                plt.savefig('%s/MisPAcompare_3D_mass%s_%s_%s_rad%s_part%s_com%s_bins%s_ax%s%s.%s' %(str(root_file), np.log10(galaxy_mass_limit), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(com_min_distance), str(min_bins), viewing_axis, savefigtxt, file_format), format=file_format, dpi=300, bbox_inches='tight', pad_inches=0.2)
        if showfig == True:
            plt.show()
        plt.close()
        
    #---------------------------
    if mis_pa_compare == True:
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Running _mis_pa_compare()')
            time_start = time.time()
        _mis_pa_compare()
    #---------------------------


    # Plot to create histogram of pafit angles for galaxy selection
    def _mis_angle_histo(quiet=0, debug=False):
        assert (minangle == 0) & (viewing_angle == 0), "minangle or viewing_angle is not 0"
        
        # Collect values to plot
        mis_points      = []       #3d
        mis_points_proj = []
        pa_points       = []       #pafit
        pa_points_err   = []
        GroupNumPlot    = []
        GroupNumNotPlot = []
        
        for GroupNum in GroupNumList:
            # If galaxy not flagged, use galaxy
            if len(all_flags['%s' %str(GroupNum)]) == 0:
                # Mask correct integer (formatting weird but works)
                mask_rad = int(np.where(np.array(all_paangles['%s' %GroupNum][mis_pa_compare_type_in]['0']['hmr']) == mis_pa_compare_use_rad_in)[0])
                
                # Analytical results
                mis_points.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %mis_pa_compare_angle_type_in][mask_rad])
                mis_points_proj.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %mis_pa_compare_angle_type_in][mask_rad])
                
                # Pafit results
                pa_points.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle' %mis_pa_compare_angle_type_in][mask_rad])
                pa_points_err.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle_err' %mis_pa_compare_angle_type_in][mask_rad])
                
                GroupNumPlot.append(GroupNum)
                
            else:
                GroupNumNotPlot.append(GroupNum)
        
        
        
        
        if debug == True:
            print('GroupNumPlot', GroupNumPlot)
            print('GroupNumNotPlot', GroupNumNotPlot)
            print(mis_points)
            print(mis_points_proj)
            print(' ')
            print(pa_points)
            print(pa_points_err)
            
        # Print statements
        print('\nInitial sample:   ', len(GroupNumList))
        if not quiet:
            print(' ', GroupNumList)
        print('\nFinal sample:   ', len(GroupNumPlot))
        if not quiet:
            print(' ', GroupNumPlot)  
        print('\nNot in sample:   ', len(GroupNumNotPlot)) 
        if not quiet:
            print(' ', GroupNumNotPlot)
        print('\n\n==========================================')
        print('Spin radius, axis:   %.1f [pkpc], %s' %(mis_pa_compare_use_rad_in, viewing_axis))
        print('Min. stellar mass:   %.1f [log10 M*]' %np.log10(galaxy_mass_limit))
        print('Min. sf particles:   %i' %gas_sf_min_particles)
        print('Min. c.o.m distance: %.1f [pkpc]' %com_min_distance)
        print('---------------------------------------')
        print('Initial sample:  ', len(GroupNumList))
        print('Final sample:    ', len(GroupNumPlot))
        print('Not in sample:   ', len(GroupNumNotPlot)) 
        print('==========================================')
        
        #-------------------------------------------------
        # Graph initialising and base formatting
        graphformat(8, 11, 11, 11, 11, 5.80, 2.55)
        
        # Labels
        if 'stars_gas' in angle_type_in:
            label = ['Total gas']
            plot_color = 'dodgerblue'
        if 'stars_gas_sf' in angle_type_in:
            label = ['Star-forming gas']
            plot_color = 'darkorange'
        if 'stars_gas_nsf' in angle_type_in:
            label = ['Non-star-forming gas']
            plot_color = 'indigo'
            
        # Plot data as histogram
        plt.hist(pa_points, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', facecolor=plot_color, alpha=0.1)
        plt.hist(pa_points, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor=plot_color, facecolor='none', alpha=1.0)
        
        # Add poisson errors to each bin (sqrt N)
        hist_n, _ = np.histogram(pa_points, bins=np.arange(0, 181, 10), range=(0, 180))
        plt.errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot), ecolor=plot_color, ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
        
        if debug == True:
            print('\npa points: ', pa_points)
            print('pa errors: ', pa_points_err)
        
        
        ### General formatting
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlim(0, 180)
        plt.xticks(np.arange(0, 181, step=30))
        plt.xlabel('$\Psi$$_{gas-star}$')
        plt.ylabel('Percentage of galaxies')
        plt.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)

        # Annotations
        plt.axvline(30, ls='--', lw=0.5, c='k')
        #plt.title("L%s: %s Misalignment\nmass: %s, type: %s, hmr: %s, \nparticles: %s, galaxies: %s/%s, com: %s, ax: %s" %(np.log10(galaxy_mass_limit), str(mySims[0][1]), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(len(GroupNumPlot)), str(len(GroupNumList)), str(com_min_distance), viewing_axis))
        
        # Legend
        legend_elements = [Line2D([0], [0], marker=' ', color='w')]
        plt.legend(handles=legend_elements, labels=label, loc='upper right', frameon=False, labelspacing=0.1, fontsize=8, labelcolor=[plot_color], handlelength=0)
        
        # Other
        plt.tight_layout()
        
        # Savefig
        if savefig == True:
            plt.savefig("%s/MisanglePA_2D_mass%s_%s_%s_rad%s_part%s_com%s_ax%s%s.%s" %(str(root_file), np.log10(galaxy_mass_limit), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt, file_format), format=file_format, bbox_inches='tight', pad_inches=0.2, dpi=300)
        if showfig == True:
            plt.show()
        plt.close()

    #---------------------------
    if mis_angle_histo == True:
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Running _mis_angle_histo()')
            time_start = time.time()
        _mis_angle_histo()
    #---------------------------
    
    
    if csv_file: 
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Writing CSV')
            time_start = time.time()
            
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
        csv_dict = {'all_general': all_general, 'all_misangles': all_misangles, 'all_misanglesproj': all_misanglesproj, 'all_coms': all_coms, 'all_particles': all_particles, 'all_pafit': all_pafit, 'all_paangles': all_paangles, 'all_flags': all_flags}
        csv_dict.update({'function_input': str(inspect.signature(velocity_projection))})
        
        # Writing one massive JSON file
        json.dump(csv_dict, open('%s/%s_%s.csv' %(root_file, csv_name, str(datetime.now())), 'w'), cls=NumpyEncoder)
        
        """# Reading JSON file
        dict_new = json.load(open('%s/%s.csv' %(root_file, csv_name), 'r'))
        # example nested dictionaries
        new_general = dict_new['all_general']
        new_misanglesproj = dict_new['all_misanglesproj']
        # example accessing function input
        function_input = dict_new['function_input']"""
        
#--------------------
velocity_projection()
#--------------------

print('FINISHED')
print('  Time taken: %s s' %(time.time() - time_start_main))


