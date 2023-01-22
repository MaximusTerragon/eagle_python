import h5py
import numpy as np
import random
import math
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
from astropy.constants import G
from scipy.spatial import Voronoi, voronoi_plot_2d
from read_dataset_tools import read_dataset, read_dataset_dm_mass, read_header
from pafit.fit_kinematic_pa import fit_kinematic_pa
from plotbin.sauron_colormap import register_sauron_colormap
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from subhalo_main import Subhalo_Extract, Subhalo
import eagleSqlTools as sql
from graphformat import graphformat


# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28

# Directories of data hdf5 file(s)
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
                  debug=True):
                 
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
    
    if debug:
        # Print mean mass-weighted velocity of each bin
        print(vel_weighted)
    
    if plot:
        # Plot 2d histogram
        plt.figure()
        graphformat(8, 11, 11, 11, 11, 5, 4)
        im = plt.pcolormesh(xbins, ybins, vel_weighted, cmap=colormap, vmin=-150, vmax=150)
        plt.colorbar(im, label='mass-weighted mean velocity', extend='both')
        
        # Formatting
        plt.xlim(-boxradius, boxradius)
        plt.ylim(-boxradius, boxradius)
        
        plt.savefig('%s/gn_%s_2dhist_%s_%s_%s_angle_%s.jpeg' %(str(root_file), str(gn), str(particle_type), str(trim_rad_in), str(viewing_axis), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.5)
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
        print('\nstandard error')
        print(np.divide(points_vel, np.sqrt(points_num)))
        
    
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
    points_inf = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis=0) 
    #vor        = Voronoi(points_inf)
    vor        = Voronoi(points)
    
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
def velocity_projection(manual_GroupNumList = np.array([1, 2, 3, 4, 10]), 
                          SubGroupNum       = 0,
                          galaxy_mass_limit          = 10**9.5,                     # Used in sample
                                spin_rad_in          = np.array([2.0]),         # np.arange(1.0, 2.5, 0.5), np.array([2.0]) 
                                kappa_rad_in         = 30,                      # calculate kappa for this radius [pkpc]
                                angle_selection      = [['stars', 'gas_sf']],   # list of angles to find analytically [[ , ], [ , ] ...]
                          align_rad_in     = False,                             # align galaxy to stellar vector in. this radius [pkpc]
                          orientate_to_axis='z',                                # Keep as 'z'
                          viewing_angle=0,                                      # Keep as 0
                        minangle  = 0, 
                        maxangle  = 0, 
                        stepangle = 30,
                          local_boxradius = True,               # Whether to ignore a fixed boxradius_in, and instead use 1.5x trim_rad_in = spin_rad_in
                            boxradius_in  = 40,                 # Graph size of 2dhisto and voronoi projection. If local_boxradius=True, will ignore
                          vel_minmax      = 200,                # Min. max velocity values
                                viewing_axis         = 'z',                     # Which axis to view galaxy from.  DEFAULT 'z'
                                resolution           = 0.7,                     # Bin size for 2dhisto [pkpc].  DEFAULT 2.0
                                target_particles     = 5,                       # Target voronoi bins.  DEFAULT 2.0
                                com_min_distance     = 2.0,                     # [pkpc] min distance between sfgas and stars.  DEFAULT 2.0 
                                gas_sf_min_particles = 100,                     # Minimum gas sf particles to use galaxy.  DEFAULT 100
                                particle_list_in     = ['stars', 'gas_sf'],     # PA fits to be found
                                angle_type_in        = ['stars_gas_sf'],        # PA misalignment angles to be found ['stars_gas', 'stars_gas_sf', 'stars_gas_nsf', 'gas_sf_gas_nsf']
                        root_file = '/Users/c22048063/Documents/EAGLE/trial_plots',      
                          print_galaxy       = False,           # Print detailed galaxy stats in chat
                          print_galaxy_short = False,           # Print single-line galaxy stats
                          txt_file           = False,             # .txt file for individual galaxy with all results
                          csv_file           = True,              # .csv file will ALL data
                            csv_name = 'data_pa',
                          showfig        = True,
                          savefig        = True,                 # Whether to save and/or show figures
                            savefigtxt       = '_TEST',                # added txt to append to end of savefile
                          debug = False,
                                pa_angle_type_in            = 'voronoi',             # which pa angles to use: '2dhist', 'voronoi', 'both'... compare will use either or both and compare to 3D
                                  plot_2dhist_graph           = True, 
                                  plot_voronoi_graph          = True,
                                  plot_2dhist_pafit_graph     = False,
                                  plot_voronoi_pafit_graph    = False,
                                pa_compare                  = False,               # plot the voronoi and 2dhist data pa_fit comparison for single galaxy
                                    pa_compare_angle_type_in  = 'stars_gas_sf',        # which misangle to compare 
                                    pa_compare_use_rad_in     = 2.0,                   # multiples of halfmass rad
                                pa_radial                   = False,                # plot the voronoi or 2dhist data pa_fit with radius for a single galaxy. Will use spin_rad_in as limits
                                    pa_radial_type_in         = 'both',                 # which pa angles to use: '2dhist, 'voronoi', 'both'
                                    pa_radial_angle_type_in   = 'stars_gas_sf',         # which type of angle to use
                                mis_pa_compare              = True,                # plot misangle - pafit for all selected galaxies
                                mis_angle_histo             = True,                 # plot histogram of pafit misangles USES SAME _type_in, angle_type_in, _use_rad_in as above  
                                    mis_pa_compare_type_in        = 'voronoi',          # what to compare to 3D
                                    mis_pa_compare_angle_type_in  = 'stars_gas_sf',
                                    mis_pa_compare_use_rad_in     = 2.0):               # MAKE SURE THIS IS INCLUDED IN SPIN_RAD_IN
                        
                        
    # Check for common entry errors
    if 'stars_gas' in angle_type_in:
        if ('stars' not in particle_list_in) & ('gas' not in particle_list_in):
            raise Exception("'stars' or 'gas' missing from particle_list_in")
    if 'stars_gas_sf' in angle_type_in:
        if ('stars' not in particle_list_in) & ('gas_sf' not in particle_list_in):
            raise Exception("'stars' or 'gas_sf' missing from particle_list_in")
    if 'stars_gas_nsf' in angle_type_in:
        if ('stars' not in particle_list_in) & ('gas_nsf' not in particle_list_in):
            raise Exception("'stars' or 'gas_nsf' missing from particle_list_in")
    if 'gas_sf_gas_nsf' in angle_type_in:
        if ('gas_sf' not in particle_list_in) & ('gas_nsf' not in particle_list_in):
            raise Exception("'gas_sf' or 'gas_nsf' missing from particle_list_in")
    
    
    # use manual input if values given, else use sample with mstar_limit
    if len(manual_GroupNumList) > 0:
        GroupNumList = manual_GroupNumList
    else:
        sample = Sample(mySims, snapNum, galaxy_mass_limit, 'no')
        print("Sample length: ", len(sample.GroupNum))
        print("  ", sample.GroupNum)
        GroupNumList = sample.GroupNum
        
    # Use spin_rad_in as a way to trim data. This variable swap is from older version but allows future use of trim_rad_in
    trim_rad_in = spin_rad_in
    
    
    # Empty dictionaries to collect relevant data
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
        
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)
        
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
        if boxradius_in == 'rad':
            boxradius = galaxy.halfmass_rad
        elif boxradius_in == 'tworad':
            boxradius = 2*galaxy.halfmass_rad
        else:
            boxradius = boxradius_in
            
        
        # int count to broadcast print statements of each galaxy
        print_i = 0    
        # Use the data we have called to find a variety of properties
        for viewing_angle in np.arange(minangle, maxangle+1, stepangle):
            # If we want the original values, enter 0 for viewing angle
            subhalo = Subhalo(galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas,
                                                angle_selection,
                                                viewing_angle,
                                                spin_rad,
                                                trim_rad, 
                                                kappa_rad, 
                                                align_rad,              #align_rad = False
                                                orientate_to_axis,
                                                quiet=True)
        
                
            # Print galaxy properties for first call
            if print_i == 0:
                if print_galaxy == True:
                    print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
                    print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))
                    print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad)        
                    print('KAPPA:                  %.2f' %subhalo.kappa)
                    print('KAPPA GAS SF:           %.2f' %subhalo.kappa_gas_sf)
                    print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
                    mask = np.where(np.array(subhalo.coms['hmr'] == 2.0))
                    print('C.O.M 2HMR STARS-SF [pkpc]:  %.2f' %subhalo.coms['stars_gas_sf'][int(mask[0])])
                    print(' HALF-\tANGLES (STARS-)\t\t\tPARTICLE COUNT\t\t\tMASS')
                    print(' RAD\tGAS\tSF\tNSF\tSF-NSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
                    for i in np.arange(0, len(spin_rad_in), 1):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            print(' %.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.mis_angles['hmr'][i], subhalo.mis_angles['stars_gas_angle'][i], subhalo.mis_angles['stars_gas_sf_angle'][i], subhalo.mis_angles['stars_gas_nsf_angle'][i], subhalo.mis_angles['gas_sf_gas_nsf_angle'][i], subhalo.particles['stars'][i], subhalo.particles['gas'][i], subhalo.particles['gas_sf'][i], subhalo.particles['gas_nsf'][i], np.log10(subhalo.particles['stars_mass'][i]), np.log10(subhalo.particles['gas_mass'][i]), np.log10(subhalo.particles['gas_sf_mass'][i]), np.log10(subhalo.particles['gas_nsf_mass'][i])))        
                    print('CENTRE [pMpc]:      [%.5f,\t%.5f,\t%.5f]' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))        # [pkpc]
                    print('PERC VEL [pkm/s]:   [%.5f,\t%.5f,\t%.5f]' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))  # [pkm/s]
                    #print('VIEWING ANGLES: ', end='')
                elif print_galaxy_short == True:
                    print('GN:\t%s\t|HMR:\t%.2f\t|KAPPA / SF:\t%.2f  %.2f' %(str(subhalo.gn), subhalo.halfmass_rad, subhalo.kappa, subhalo.kappa_gas_sf)) 
                    
                # Create txt file with output for that galaxy
                if txt_file == True:
                    dash = '-' * 130
                    f = open("%s/gn_%s_velproj.txt" %(str(root_file), str(GroupNum)), 'w+')
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
                    f.write('\nCENTRE:          [%.5f,\t%.5f,\t%.5f]\t[pMpc]' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))                                 # [pMpc]
                    f.write('\nPERC VELOCITY:   [%.5f,\t%.5f,\t%.5f]\t[pkm/s]' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))
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
                    f.write('\nPIXEL RESOLUTION [pkpc]:    %s' %str(resolution))
                    f.write('\nVORONOI TARGET PARTICLES:   %s particles' %str(target_particles))
                    
            # Print galaxy properties
            if print_galaxy == True:
                if print_i == 0:
                    print('VIEWING ANGLE: %s' %str(viewing_angle), end=' ')
                else:
                    print('%s' %str(viewing_angle), end=' ')
            
        
            # Loop over multiples of halfmass_rad
            print_ii = 0
            for trim_rad_i in trim_rad:
                if debug == True:
                    print('RAD:', trim_rad_i) 
                        
    
                # Function to plot 2dhist-fed data (splinter from _weight_histo)
                def _plot_2dhist(colormap='coolwarm', quiet=1, debug=False):
                    # Initialise figure
                    graphformat(8, 11, 11, 11, 11, 3.75, 3)
                    fig, axs = plt.subplots(nrows=1, ncols=len(particle_list_in), figsize=(4.5*len(particle_list_in), 4), sharex=True, sharey=True)
                
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
                        im = axs[j].pcolormesh(xbins, ybins, vel_weighted, cmap=colormap, vmin=-vel_minmax, vmax=vel_minmax)
                    
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
                    axs[0].text(-boxradius, boxradius+1, 'resolution: %s pkpc, trim_rad: %s pkpc, hmr: %s pkpc, axis: %s, angle: %s' %(str(resolution), str(trim_rad_i), str(subhalo.halfmass_rad), viewing_axis, str(viewing_angle)), fontsize=8)

                    # Colorbar
                    cax = plt.axes([0.92, 0.11, 0.015, 0.77])
                    plt.colorbar(im, cax=cax, label='mass-weighted mean velocity [km/s]', extend='both')
            
                    if savefig == True:
                        plt.savefig('%s/gn_%s_2dhist_rad%s_ax%s_angle%s%s.jpeg' %(str(root_file), str(subhalo.gn), str(trim_rad_i), str(viewing_axis), str(viewing_angle), savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.3)  
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
                    _plot_2dhist()  
                if plot_voronoi_graph == True:
                    _plot_voronoi()
                #---------------------------------
            
                  
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
                            
                            
                        # Defining break for less than 4 pixels due to fitpa requireing min. of 4
                        if len(vel_bin_particle) < 4:
                            # Append to dictionary                
                            all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            print('VOID: Not enough 2dhisto bins to pafit: %s, %s' %(particle_list_in_i, str(vel_bin_particle)))
                            
                            continue
                        
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
                            
                        if len(points_num_check) < 4:
                            if debug == True:
                                print('Low 2dhisto bins', len(points_num_check))
                            
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            print('VOID: Not enough 2dhisto bins in voronoi to pafit: %s, %s' %(particle_list_in_i, str(len(points_num_check))))
                            
                            continue
                            
                        # If all bins already have target_particle count, don't run voronoi as it can't plot it
                        if all(points_num_check_i >= target_particles for points_num_check_i in points_num_check) == True:
                            if debug == True:
                                print('points_num_check S/N', points_num_check)
                            
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            print('VOID: All points already meet S/N: %s. Voronoi plot fail' %particle_list_in_i)
                            
                            continue
                        #--------------------------------
                        
                        
                        # Extract points
                        points_particle, vel_bin_particle, vor = _voronoi_tessalate(subhalo.gn, root_file, subhalo.data['%s' %str(trim_rad_i)], particle_list_in_i, viewing_angle, viewing_axis, resolution, target_particles, boxradius, trim_rad_i*subhalo.halfmass_rad)
                        
                        if debug == True:
                            print('points_particle', points_particle)
                            print('len(points_particle)', len(points_particle))
                            print('len(vel_bin_particle)', len(vel_bin_particle))
                            
                        #--------------------------------
                        if len(points_particle) < 4:
                            # Append to dictionary
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i].append(math.nan)
                            all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i].append(math.nan)
                            
                            print('VOID: Not enough voronoi bins to pafit: %s, %s' %(particle_list_in_i, str(len(points_particle))))
                            
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
                
                #---------------------------------
                # Run PA fit routine once per galaxy to append to galaxy list (viewing_angle = 0)
                #if print_i == 0:
                # PA fit routine - determines what pa angles are recorded
                if pa_angle_type_in == '2dhist':
                    _pa_fit_2dhist()
                elif pa_angle_type_in == 'voronoi':
                    _pa_fit_voronoi()
                elif pa_angle_type_in == 'both':
                    _pa_fit_2dhist()
                    _pa_fit_voronoi()
                #--------------------------------
                
                print_ii = print_ii + 1
            
            
            #===============================================
            # Start of end of variable trim_rad_i
            
            # Assign particle data once per galaxy
            if print_i == 0:
                all_particles['%s' %str(subhalo.gn)] = subhalo.particles
                all_coms['%s' %str(subhalo.gn)] = subhalo.coms
                all_misangles['%s' %str(subhalo.gn)] = subhalo.mis_angles
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
                    
            if txt_file:
                # Write into txt file from angles
                if pa_angle_type_in == '2dhist':
                    f.write('\n' + dash)
                    f.write('\nPA FITS 2DHIST [deg]')
                    f.write('\nVIEWING ANGLE: %s' %str(subhalo.viewing_angle))
                    f.write('\nHMR')
                    for particle_list_in_i in particle_list_in:
                        f.write('\t%s\t' %str(particle_list_in_i).upper())
                    i = 0
                    while i < len(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr']):
                        f.write('\n%.1f' %all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr'][i])
                        for particle_list_in_i in particle_list_in:
                            f.write('\t%.1f ± %.1f' %(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i][i], all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i][i]))
                        i = i + 1
                elif pa_angle_type_in == 'voronoi':
                    f.write('\n' + dash)
                    f.write('\nPA FITS VORONOI [deg]')
                    f.write('\nVIEWING ANGLE: %s' %str(subhalo.viewing_angle))
                    f.write('\nHMR')
                    for particle_list_in_i in particle_list_in:
                        f.write('\t%s\t' %str(particle_list_in_i).upper())
                    i = 0
                    while i < len(all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['hmr']):
                        f.write('\n%.1f' %all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['hmr'][i])
                        for particle_list_in_i in particle_list_in:
                            f.write('\t%.1f ± %.1f' %(all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i][i], all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i][i]))
                        i = i + 1
                elif pa_angle_type_in == 'both':
                    f.write('\n' + dash)
                    f.write('\nPA FITS 2DHIST & VORONOI [deg]')
                    f.write('\nVIEWING ANGLE: %s' %str(subhalo.viewing_angle))
                    f.write('\n\t2DHIST\t' + '\t'*2*(len(particle_list_in)-1) + '\tVORONOI')
                    f.write('\nHMR')
                    for particle_list_in_i in particle_list_in:
                        f.write('\t%s\t' %str(particle_list_in_i).upper())
                    for particle_list_in_i in particle_list_in:
                        f.write('\t%s\t' %str(particle_list_in_i).upper())
                    i = 0
                    while i < len(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr']):
                        f.write('\n%.1f' %all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr'][i])
                        for particle_list_in_i in particle_list_in:
                            f.write('\t%.1f ± %.1f' %(all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i][i], all_pafit['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i][i]))
                        for particle_list_in_i in particle_list_in:    
                            f.write('\t%.1f ± %.1f' %(all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %particle_list_in_i][i], all_pafit['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %particle_list_in_i][i]))
                        i = i + 1
                                         
                # Write into txt file from misalignments
                if pa_angle_type_in == '2dhist':
                    f.write('\n' + dash)
                    f.write('\nPA MISALIGNMENT ANGLES 2DHIST [deg]')
                    f.write('\nVIEWING ANGLE: %s' %str(subhalo.viewing_angle))
                    f.write('\nHMR')
                    for angle_type_in_i in angle_type_in:
                        f.write('\t%s' %str(angle_type_in_i).upper())
                    i = 0
                    while i < len(all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr']):
                        f.write('\n%.1f' %all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr'][i])
                        for angle_type_in_i in angle_type_in:
                            f.write('\t%.1f ± %.1f' %(all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %angle_type_in_i][i], all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %angle_type_in_i][i]))
                        i = i + 1
                elif pa_angle_type_in == 'voronoi':
                    f.write('\n' + dash)
                    f.write('\nPA MISALIGNMENT ANGLES VORONOI [deg]')
                    f.write('\nVIEWING ANGLE: %s' %str(subhalo.viewing_angle))
                    f.write('\nHMR')
                    for angle_type_in_i in angle_type_in:
                        f.write('\t%s' %str(angle_type_in_i).upper())
                    i = 0
                    while i < len(all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['hmr']):
                        f.write('\n%.1f' %all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['hmr'][i])
                        for angle_type_in_i in angle_type_in:
                            f.write('\t%.1f ± %.1f' %(all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %angle_type_in_i][i], all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %angle_type_in_i][i]))
                        i = i + 1
                elif pa_angle_type_in == 'both':
                    f.write('\n' + dash)
                    f.write('\nPA MISALIGNMENT ANGLES 2DHIST & VORONOI [deg]')
                    f.write('\nVIEWING ANGLE: %s' %str(subhalo.viewing_angle))
                    f.write('\n\t2DHIST\t' + '\t'*2*(len(angle_type_in)-1) + '\tVORONOI')
                    f.write('\nHMR')
                    for angle_type_in_i in angle_type_in:
                        f.write('\t%s' %str(angle_type_in_i).upper())
                    for angle_type_in_i in angle_type_in:
                        f.write('\t%s' %str(angle_type_in_i).upper())
                    i = 0
                    while i < len(all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr']):
                        f.write('\n%.1f' %all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['hmr'][i])
                        for angle_type_in_i in angle_type_in:
                            f.write('\t%.1f ± %.1f' %(all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle' %angle_type_in_i][i], all_paangles['%s' %str(subhalo.gn)]['2dhist']['%s' %str(viewing_angle)]['%s_angle_err' %angle_type_in_i][i]))
                        for angle_type_in_i in angle_type_in:    
                            f.write('\t%.1f ± %.1f' %(all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle' %angle_type_in_i][i], all_paangles['%s' %str(subhalo.gn)]['voronoi']['%s' %str(viewing_angle)]['%s_angle_err' %angle_type_in_i][i]))
                        i = i + 1

            print_i = print_i + 1
        
        
        #=========================================
        # Start of once per galaxy
        if txt_file:
            f.close()            
        
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
            _pa_radial()
        #-----------------------
        
    #=====================================
    # Start of GroupNum loop
    
    """ DATA AVAILABLE
    all_general[GroupNum]
        ['stelmass']      - subhalo.stelmass      HAS ALL CORE DATA ON SUBHALO
        ['gasmass']       - subhalo.gasmass       
        ['gasmass_sf']    - subhalo.gasmass_sf
        ['gasmass_nsf']   - subhalo.gasmass_nsf
        ['halfmass_rad']  - subhalo.halfmass_rad
        ['kappa']         - subhalo.kappa
        ['kappa_gas']     - subhalo.kappa_gas
        ['kappa_gas_sf']  - subhalo.kappa_gas_sf
        ['kappa_gas_nsf'] - subhalo.kappa_gas_nsf
    
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
        
        # Find row for which hmr = pa_compare_use_rad_in (=2.). Use any galaxy for the mask, here i use first entry [0]
        mask = np.where(np.array(all_paangles['%s' %str(GroupNumList[0])][mis_pa_compare_type_in]['0']['hmr']) == mis_pa_compare_use_rad_in)
        for GroupNum in GroupNumList:
            
            # min. sf particle requirement. THIS SHOUD BE REDUNDANT, BUT WILL DO ANYWAY. Essentially means we miss most of the math.nan appending, though some will remain from failed 4 2dhisto points, or vor fail.
            mask_sf = np.where(all_particles['%s' %str(GroupNumList[0])]['hmr'] == mis_pa_compare_use_rad_in)
            if all_particles['%s' %str(GroupNum)]['gas_sf'][int(mask_sf[0])] >= gas_sf_min_particles:
                
                # min. com distance requirement
                mask_com = np.where(all_coms['%s' %str(GroupNumList[0])]['hmr'] == mis_pa_compare_use_rad_in)
                if all_coms['%s' %str(GroupNum)]['stars_gas_sf'][int(mask_com[0])] <= com_min_distance:
                    
                    # Collect list of GroupNums plotted and not plotted
                    if math.isnan(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])]) == False:
                        # 3D and 2D projected
                        mis_points.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])])
                        mis_points_proj.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])])
                        
                        # 3D and 2D errors [lo, hi]
                        mis_points_err.append(all_misangles['%s' %str(GroupNum)]['%s_angle_err' %mis_pa_compare_angle_type_in][int(mask[0])])
                        mis_points_proj_err.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle_err' %mis_pa_compare_angle_type_in][int(mask[0])])
                        
                        pa_points.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])])
                        pa_points_err.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle_err' %mis_pa_compare_angle_type_in][int(mask[0])])
            
                        GroupNumPlot.append(GroupNum)
                    else:
                        GroupNumNotPlot.append([GroupNum, 'pa NaN'])
                else:
                    GroupNumNotPlot.append([GroupNum, ['com: %.2f' %all_coms['%s' %str(GroupNum)]['stars_gas_sf'][int(mask_com[0])]]])
            else:
                GroupNumNotPlot.append([GroupNum, 'sf part: %i' %all_particles['%s' %str(GroupNum)]['gas_sf'][int(mask_sf[0])]])
           
        
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
        plt.figure()
        graphformat(8, 11, 11, 11, 11, 5, 5)

        # Plot scatter comparing 3D (mis_points) to extracted value from pafit routine (d_points/err)
        if useangle == '2D':
            plt.errorbar(mis_points_proj, pa_points, xerr=None, yerr=pa_points_err, ecolor='k', ls='none', alpha=0.5, zorder=4, capsize=4, elinewidth=1, markeredgewidth=1)
            plt.scatter(mis_points_proj, pa_points, c='k', label='M$_{*}$ > 10E9', s=4, zorder=5)
        if useangle == '3D':
            plt.errorbar(mis_points, pa_points, xerr=None, yerr=pa_points_err, ecolor='k', ls='none', alpha=0.5, zorder=4, capsize=4, elinewidth=1, markeredgewidth=1)
            plt.scatter(mis_points, pa_points, c='k', label='M$_{*}$ > 10E9', s=4, zorder=5)

        # Plot straight line (expected)
        plt.plot([0, 180], [0, 180], c='grey', ls='--')
        
        # Formatting
        if useangle == '2D':
            plt.xlabel('3D projected $\Psi$$_{gas-star}$')
            plt.title('3D projected angle - PA fit\nangle: %s, type: %s, hmr: %s, galaxies: %s/%s, particles: %s, com: %s, ax: %s' %(mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(len(GroupNumPlot)), str(len(GroupNumList)), str(gas_sf_min_particles), str(com_min_distance), viewing_axis))
        if useangle == '3D':
            plt.xlabel('3D absolute $\Psi$$_{gas-star}$')
            plt.title('3D absolute angle - PA fit\nangle: %s, type: %s, hmr: %s, galaxies: %s/%s, particles: %s, com: %s, ax: %s' %(mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(len(GroupNumPlot)), str(len(GroupNumList)), str(gas_sf_min_particles), str(com_min_distance), viewing_axis))  
        plt.ylabel('fit_kinematic_pa')
        plt.xlim(0, 180)
        plt.ylim(0, 180)
        plt.legend()
        plt.tight_layout()
        
        if savefig == True:
            if useangle == '2D':
                plt.savefig('%s/MisPAcompare_2D_%s_%s_rad%s_part%s_com%s_ax%s%s.jpeg' %(str(root_file), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
            if useangle == '3D':
                plt.savefig('%s/MisPAcompare_3D_%s_%s_rad%s_part%s_com%s_ax%s%s.jpeg' %(str(root_file), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt), dpi=300, bbox_inches='tight', pad_inches=0.2)
        if showfig == True:
            plt.show()
        plt.close()
        
    #---------------------------
    if mis_pa_compare == True:
        _mis_pa_compare()
    #---------------------------


    # Plot to create histogram of pafit angles for galaxy selection
    def _mis_angle_histo(quiet=0, debug=False):
        assert (minangle == 0) & (viewing_angle == 0), "minangle or viewing_angle is not 0"
        
        # Collect values to plot
        mis_points      = []       #3d
        mis_points_proj = []
        pa_points       = []       #2d projected from pafit
        pa_points_err   = []
        GroupNumPlot    = []
        GroupNumNotPlot = []
        
        # Find row for which hmr = pa_compare_use_rad_in (=2.). Use any galaxy for the mask, here i use first entry [0]
        mask = np.where(np.array(all_paangles['%s' %str(GroupNumList[0])][mis_pa_compare_type_in]['0']['hmr']) == mis_pa_compare_use_rad_in)
        for GroupNum in GroupNumList:
            
            # min. sf particle requirement. THIS SHOUD BE REDUNDANT, BUT WILL DO ANYWAY. Essentially means we miss most of the math.nan appending, though some will remain from failed 4 2dhisto points, or vor fail.
            mask_sf = np.where(all_particles['%s' %str(GroupNumList[0])]['hmr'] == mis_pa_compare_use_rad_in)
            if all_particles['%s' %str(GroupNum)]['gas_sf'][int(mask_sf[0])] >= gas_sf_min_particles:
                
                # min. com distance requirement
                mask_com = np.where(all_coms['%s' %str(GroupNumList[0])]['hmr'] == mis_pa_compare_use_rad_in)
                if all_coms['%s' %str(GroupNum)]['stars_gas_sf'][int(mask_com[0])] <= com_min_distance:
                    
                    # Collect list of GroupNums plotted and not plotted
                    if math.isnan(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])]) == False:
                        mis_points.append(all_misangles['%s' %str(GroupNum)]['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])])
                        mis_points_proj.append(all_misanglesproj['%s' %str(GroupNum)][viewing_axis]['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])])
                        pa_points.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle' %mis_pa_compare_angle_type_in][int(mask[0])])
                        pa_points_err.append(all_paangles['%s' %str(GroupNum)][mis_pa_compare_type_in]['0']['%s_angle_err' %mis_pa_compare_angle_type_in][int(mask[0])])
                        
                        GroupNumPlot.append(GroupNum)
                    else:
                        GroupNumNotPlot.append([GroupNum, 'pa NaN'])
                else:
                    GroupNumNotPlot.append([GroupNum, ['com: %.2f' %all_coms['%s' %str(GroupNum)]['stars_gas_sf'][int(mask_com[0])]]])
            else:
                GroupNumNotPlot.append([GroupNum, 'sf part: %i' %all_particles['%s' %str(GroupNum)]['gas_sf'][int(mask_sf[0])]])
           
        
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
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig, ax = plt.subplots(1, 1, figsize=[8, 4])

        # Plot data as histogram
        plt.hist(pa_points, weights=np.ones(len(GroupNumPlot))/len(GroupNumPlot), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='b', facecolor='none', alpha=0.8, label=mis_pa_compare_angle_type_in)
        hist_n, _ = np.histogram(pa_points, bins=np.arange(0, 181, 10), range=(0, 180))
    
        # Add poisson errors to each bin (sqrt N)
        plt.errorbar(np.arange(5, 181, 10), hist_n/len(GroupNumPlot), xerr=None, yerr=np.sqrt(hist_n)/len(GroupNumPlot), ecolor='k', ls='none', capsize=4, elinewidth=1, markeredgewidth=1)
        
        if debug == True:
            print('\npa points: ', pa_points)
            print('pa errors: ', pa_points_err)
        
        # General formatting
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 180, step=30))
        ax.set_xlabel('$\Psi$$_{gas-star}$')
        ax.set_ylabel('Percentage of galaxies')
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

        # Annotations
        ax.axvline(30, ls='--', lw=0.5, c='k')
        plt.suptitle("L%s: %s Misalignment\ntype: %s, hmr: %s, particles: %s, galaxies: %s/%s, com: %s, ax: %s" %(str(mySims[0][1]), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(len(GroupNumPlot)), str(len(GroupNumList)), str(com_min_distance), viewing_axis))
        plt.legend()
    
        if savefig == True:
            plt.savefig("%s/MisanglePA_2D_%s_%s_rad%s_part%s_com%s_ax%s%s.jpeg" %(str(root_file), mis_pa_compare_angle_type_in, mis_pa_compare_type_in, str(mis_pa_compare_use_rad_in), str(gas_sf_min_particles), str(com_min_distance), viewing_axis, savefigtxt), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
        if showfig == True:
            plt.show()
        plt.close()

    #---------------------------
    if mis_angle_histo == True:
        _mis_angle_histo()
    #---------------------------
    
    
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
        csv_dict = {'all_general': all_general, 'all_misangles': all_misangles, 'all_misanglesproj': all_misanglesproj, 'all_coms': all_coms, 'all_particles': all_particles, 'all_pafit': all_pafit, 'all_paangles': all_paangles}
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
        


