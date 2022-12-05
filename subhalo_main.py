import h5py
import numpy as np
import astropy.units as u
from astropy.constants import G
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import eagleSqlTools as sql
from mpl_toolkits.mplot3d import Axes3D
from pafit.fit_kinematic_pa import fit_kinematic_pa
from plotbin.sauron_colormap import register_sauron_colormap
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from pyread_eagle import EagleSnapshot
from read_dataset_tools import read_dataset, read_dataset_dm_mass, read_header
from graphformat import graphformat


# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'


"""
Will convert all particle data to Msun, pkpc, s
Boxsize and load_region_length in cMpc/h (I think) 
"""
class Subhalo_Extract:
    
    def __init__(self, sim, data_dir, gn, sgn, centre_galaxy=True, load_region_length=2):       # cMpc/h
        # Assigning subhalo properties
        self.gn           = gn
        self.sgn          = sgn
        
        # Load information from the header
        self.a, self.h, self.boxsize = read_header(data_dir) # units of scale factor, h, and L [cMpc/h]    ASSUMES Z=0 I THINK
        
        # For a given gn and sgn, run sql query on SubFind catalogue
        myData = self.query(sim)
        
        # Assiging subhalo properties
        self.stelmass     = myData['stelmass']                                                                  # [Msun]
        self.gasmass      = myData['gasmass']                                                                   # [Msun]
        self.halfmass_rad = myData['rad']                                                                       # [pkpc]
        self.centre       = np.array([myData['x'], myData['y'], myData['z']]) * u.Mpc.to(u.kpc)                 # [pkpc]
        self.centre_mass  = np.array([myData['x_mass'], myData['y_mass'], myData['z_mass']]) * u.Mpc.to(u.kpc)  # [pkpc]
        self.perc_vel     = np.array([myData['vel_x'], myData['vel_y'], myData['vel_z']])                       # [pkpc/s]
        
        # Load data for stars and gas in non-centred units
        # Msun, pkpc, and pkpc/s
        self.stars        = self.read_galaxy(data_dir, 4, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length) 
        self.gas          = self.read_galaxy(data_dir, 0, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length)
        
        if centre_galaxy == True:
            self.stars['Coordinates'] = self.stars['Coordinates'] - self.centre
            self.gas['Coordinates']   = self.gas['Coordinates'] - self.centre
            self.stars['Velocity'] = self.stars['Velocity'] - self.perc_vel
            self.gas['Velocity']   = self.gas['Velocity'] - self.perc_vel


    def query(self, sim):
        # This uses the eagleSqlTools module to connect to the database with your username and password.
        # If the password is not given, the module will prompt for it.
        con = sql.connect("lms192", password="dhuKAP62")
        
        for sim_name, sim_size in sim:
            #print(sim_name)
    
            # Construct and execute query for each simulation. This query returns properties for a single galaxy
            myQuery = 'SELECT \
                        SH.MassType_Star as stelmass, \
                        SH.MassType_Gas as gasmass, \
                        SH.HalfMassRad_Star as rad, \
                        SH.CentreOfPotential_x as x, \
                        SH.CentreOfPotential_y as y, \
                        SH.CentreOfPotential_z as z, \
                        SH.CentreOfMass_x as x_mass, \
                        SH.CentreOfMass_y as y_mass, \
                        SH.CentreOfMass_z as z_mass, \
                        SH.Velocity_x as vel_x, \
                        SH.Velocity_y as vel_y, \
                        SH.Velocity_z as vel_z \
                       FROM \
        			     %s_Subhalo as SH \
                       WHERE \
        			     SH.SnapNum = 28 \
                         and SH.GroupNumber = %i \
                         and SH.SubGroupNumber = %i \
                      ORDER BY \
        			     SH.MassType_Star desc'%(sim_name, self.gn, self.sgn)
	
            # Execute query.
            myData = sql.execute_query(con, myQuery)
            
            return myData
            
        
    def read_galaxy(self, data_dir, itype, gn, sgn, centre, load_region_length):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates, velocty, and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity."""

        # Where we store all the data
        data = {}
        
        # Initialize read_eagle module.
        eagle_data = EagleSnapshot(data_dir)
        
        # Put centre into pMpc -> cMpc/h units.
        centre_cMpc = centre*self.h

        # Select region to load, a 'load_region_length' cMpc/h cube centred on 'centre'.
        region = np.array([
            (centre_cMpc[0]-0.5*load_region_length), (centre_cMpc[0]+0.5*load_region_length),
            (centre_cMpc[1]-0.5*load_region_length), (centre_cMpc[1]+0.5*load_region_length),
            (centre_cMpc[2]-0.5*load_region_length), (centre_cMpc[2]+0.5*load_region_length)
        ])
        eagle_data.select_region(*region)
                
        # Load data using read_eagle, load conversion factors manually.
        f = h5py.File(dataDir, 'r')
        # If stars, do not load StarFormationRate (as not contained in database)
        if itype == 4:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'Velocity']:
                tmp  = eagle_data.read_dataset(itype, att)
                cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
            f.close()
        # If gas, load StarFormationRate
        elif itype == 0:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'StarFormationRate', 'Velocity']:
                tmp  = eagle_data.read_dataset(itype, att)
                cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
            f.close()
        
        # Mask to selected GroupNumber and SubGroupNumber.
        mask = np.logical_and(data['GroupNumber'] == gn, data['SubGroupNumber'] == sgn)
        for att in data.keys():
            data[att] = data[att][mask]
               
        # Load data, then mask to selected GroupNumber and SubGroupNumber. Automatically converts to pcm from read_dataset, converted to pMpc
        data['Mass'] = data['Mass'] * u.g.to(u.Msun)                   # [Msun]
        data['Coordinates'] = data['Coordinates'] * u.cm.to(u.Mpc)     # [pMpc]
        data['Velocity'] = data['Velocity'] * u.cm.to(u.Mpc)           # [pMpc/s]
        if itype == 0:
            data['StarFormationRate'] = data['StarFormationRate'] * u.g.to(u.Msun)  # [Msun/s]
        
        # Periodic wrap coordinates around centre (in proper units). 
        # boxsize converted from cMpc/h -> pMpc
        boxsize = self.boxsize/self.h       # [pkpc]
        data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
        
        # Converting to pkpc
        data['Coordinates'] = data['Coordinates'] * u.Mpc.to(u.kpc)     # [pkpc]
        data['Velocity'] = data['Velocity'] * u.Mpc.to(u.km)            # [pkm/s]
        
        return data
        
        
        
        
        
        
        
        

        
register_sauron_colormap()

# list of simulations
mySims = np.array([('RefL0012N0188', 12)])  

GroupNum = 4
SubGroupNum = 0

# Initial extraction of galaxy data
galaxy = Subhalo_Extract(mySims, dataDir, GroupNum, SubGroupNum)









