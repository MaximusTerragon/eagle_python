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
        self.perc_vel     = np.array([myData['vel_x'], myData['vel_y'], myData['vel_z']])                       # [pkm/s]
        
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
        

class Subhalo:
    
    def __init__(self, gn, sgn, halfmass_rad, centre, centre_mass, perc_vel, stars, gas, 
                            viewing_angle, 
                            calc_spin_rad, 
                            calc_kappa_rad, 
                            trim_rad,
                            align_rad):
                            
        # Create masks for starforming and non-starforming gas
        self.mask_sf        = np.nonzero(gas['StarFormationRate'])          
        self.mask_nsf       = np.where(gas['StarFormationRate'] == 0)
        
        # Assign bulk galaxy values to this subhalo
        self.gn             = gn
        self.sgn            = sgn
        self.gasmass_sf     = np.sum(gas['Mass'][self.mask_sf])     # [Msun]
        self.gasmass_nsf    = np.sum(gas['Mass'][self.mask_nsf])    # [Msun]
        self.stelmass       = np.sum(stars['Mass'])                 # [Msun]
        self.gasmass        = np.sum(gas['Mass'])                   # [Msun]
        self.halfmass_rad   = halfmass_rad                          # [pkpc]
        self.centre         = centre                                # [pkpc]
        self.centre_mass    = centre_mass                           # [pkpc]
        self.perc_vel       = perc_vel                              # [pkm/s]
        self.viewing_angle  = viewing_angle                         # [deg]
        
        # Create dataset of star-forming and non-star-forming gas
        gas_sf = {}
        gas_nsf = {}
        for arr in gas.keys():
            gas_sf[arr]  = gas[arr][self.mask_sf]
            gas_nsf[arr] = gas[arr][self.mask_nsf]
            
        # Create data of raw (0 degree) data
        data_nil = {}
        parttype_name = ['stars', 'gas', 'gas_sf', 'gas_nsf']
        i = 0
        for parttype in [stars, gas, gas_sf, gas_nsf]:
            data_nil['%s'%parttype_name[i]] = parttype
            i = i + 1
            
        # KAPPA
        # Finding star unit vector within calc_kappa_rad, finding angle between it and z and returning matrix for this
        if calc_kappa_rad:
            stars_spin_kappa, _  = self.find_spin(data_nil['stars'], calc_kappa_rad, 'stars')
            _ , matrix = self.orientate('z', stars_spin_kappa)
            # Orientate entire galaxy according to matrix above, use this to find kappa
            stars_aligned_kappa  = self.rotate_galaxy(matrix, data_nil['stars'])
            self.kappa = self.kappa_co(stars_aligned_kappa, calc_kappa_rad) 
        
            print('kappa', self.kappa)


        # ALIGN GALAXY
        if align_rad:
            stars_spin_align, _  = self.find_spin(data_nil['stars'], align_rad, 'stars')
            _ , matrix = self.orientate('z', stars_spin_align)
            
            # Orientate entire galaxy according to matrix above
            self.data_align = {}
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                if trim_rad:
                    self.data_align['%s'%parttype_name] = self.trim_within_rad(self.rotate_galaxy(matrix, data_nil[parttype_name]), trim_rad)
                elif not trim_rad:
                    self.data_align['%s'%parttype_name] = self.rotate_galaxy(matrix, data_nil[parttype_name])
                
            # Find aligned spin vectors
            self.stars_spin_align, _   = self.find_spin(self.data_align['stars'], calc_spin_rad, 'stars')
            self.gas_spin_align, _     = self.find_spin(self.data_align['gas'], calc_spin_rad, 'gas')      
            self.gas_sf_spin_align, _  = self.find_spin(self.data_align['gas_sf'], calc_spin_rad, 'starforming gas')
            self.gas_nsf_spin_align, _ = self.find_spin(self.data_align['gas_nsf'], calc_spin_rad, 'non-starforming gas')
            
            print('align', self.stars_spin_align)
            
            
            # Compute angle [deg] between star spin and gas angular momentum vector
            self.mis_angle     = self.misalignment_angle(self.stars_spin_align, self.gas_spin_align)           
            self.mis_angle_sf  = self.misalignment_angle(self.stars_spin_align, self.gas_sf_spin_align)    
            self.mis_angle_nsf = self.misalignment_angle(self.stars_spin_align, self.gas_nsf_spin_align) 
            self.mis_angle_sf_nsf = self.misalignment_angle(self.gas_sf_spin_align, self.gas_nsf_spin_align)
            
            print('self.mis_angle ALIGNED', self.mis_angle)
            print(self.mis_angle_sf)
            print(self.mis_angle_nsf)
            print(self.mis_angle_sf_nsf)
        
        
        # SPIN VECTORS AND ROTATE
        if calc_spin_rad:
            # Find original spin vectors
            stars_spin_nil, _   = self.find_spin(data_nil['stars'], calc_spin_rad, 'stars')
            gas_spin_nil, _     = self.find_spin(data_nil['gas'], calc_spin_rad, 'gas')      
            gas_sf_spin_nil, _  = self.find_spin(data_nil['gas_sf'], calc_spin_rad, 'starforming gas')
            gas_nsf_spin_nil, _ = self.find_spin(data_nil['gas_nsf'], calc_spin_rad, 'non-starforming gas')
            
            # Compute angle [deg] between star spin and gas angular momentum vector
            self.mis_angle     = self.misalignment_angle(stars_spin_nil, gas_spin_nil)           
            self.mis_angle_sf  = self.misalignment_angle(stars_spin_nil, gas_sf_spin_nil)    
            self.mis_angle_nsf = self.misalignment_angle(stars_spin_nil, gas_nsf_spin_nil) 
            self.mis_angle_sf_nsf = self.misalignment_angle(gas_sf_spin_nil, gas_nsf_spin_nil)
            
            # Find rotation matrix
            matrix = self.rotate_around_axis('z', 360. - viewing_angle, stars_spin_nil)
        
            self.data_rot = {}
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                if trim_rad:
                    self.data_rot['%s'%parttype_name] = self.trim_within_rad(self.rotate_galaxy(matrix, data_nil[parttype_name]), trim_rad)
                elif not trim_rad:
                    self.data_rot['%s'%parttype_name] = self.rotate_galaxy(matrix, data_nil[parttype_name])
                    
            # Find new spin vectors
            self.stars_spin, _   = self.find_spin(self.data_rot['stars'], calc_spin_rad, 'stars')
            self.gas_spin, _     = self.find_spin(self.data_rot['gas'], calc_spin_rad, 'gas')      
            self.gas_sf_spin, _  = self.find_spin(self.data_rot['gas_sf'], calc_spin_rad, 'starforming gas')
            self.gas_nsf_spin, _ = self.find_spin(self.data_rot['gas_nsf'], calc_spin_rad, 'non-starforming gas')
            
            print('self.mis_angle', self.mis_angle)
            print(self.mis_angle_sf)
            print(self.mis_angle_nsf)
            print(self.mis_angle_sf_nsf)
            
            
            
            # Compute angle [deg] between star spin and gas angular momentum vector
            self.mis_angle     = self.misalignment_angle(self.stars_spin, self.gas_spin)           
            self.mis_angle_sf  = self.misalignment_angle(self.stars_spin, self.gas_sf_spin)    
            self.mis_angle_nsf = self.misalignment_angle(self.stars_spin, self.gas_nsf_spin) 
            self.mis_angle_sf_nsf = self.misalignment_angle(self.gas_sf_spin, self.gas_nsf_spin)
            
            print('self.mis_angle ROTATED', self.mis_angle)
            print(self.mis_angle_sf)
            print(self.mis_angle_nsf)
            print(self.mis_angle_sf_nsf)


    def trim_within_rad(self, arr, radius):
        # Compute distance to centre and mask all within trim_rad
        r  = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.where(r <= radius)

        newData = {}
        for header in arr.keys():
            newData[header] = arr[header][mask]
            
        return newData
        
    def find_spin(self, arr, radius, desc):
        # Compute distance to centre and mask all within stelhalfrad
        r  = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.where(r <= radius)
        
        #print("Total %s particles in subhalo: %i"%(desc, len(r)))
        #r = r[mask]
        #print("Total %s particles in %.5f kpc: %i\n"%(desc, radius*1000, len(r)))
        
        # Finding spin angular momentum vector of each individual particle of gas and stars, where [:, None] is done to allow multiplaction of N3*N1 array. Equation D.25
        L  = np.cross(arr['Coordinates'][mask] * arr['Mass'][:, None][mask], arr['Velocity'][mask])
        
        # Summing for total angular momentum and dividing by mass to get the spin vectors
        spin = np.sum(L, axis=0)/np.sum(arr['Mass'][mask])
        
        # Expressing as unit vector
        spin_unit = spin / (spin[0]**2 + spin[1]**2 + spin[2]**2)**0.5
        
        return spin_unit, L
        
    def misalignment_angle(self, angle1, angle2):
        # Find the misalignment angle
        angle = np.rad2deg(np.arccos(np.clip(np.dot(angle1, angle2), -1.0, 1.0)))     # [deg]
        
        return angle
    
    def rotation_matrix(self, axis, theta):
        '''Return the rotation matrix associated with counterclockwise rotation about
        the user specified axis by theta radians.'''
        
        # Convert theta to radians
        theta = np.deg2rad(theta)
        #print('THETA', theta)
        
        # convert the input to an array
        axis = np.asarray(axis)
        
        # Get unit vector of our axis
        axis = axis / math.sqrt(np.dot(axis, axis))
        
        # take the cosine of out rotation degree in radians
        a = math.cos(theta/2.0)
        
        # get the rest rotation matrix components
        b, c, d = -axis * math.sin(theta/2.0)
        
        # create squared terms
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        # create cross terms
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        
        # return our rotation matrix
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
                        
    def orientate(self, axis, attribute):
        # Finds angle given a defined x, y, z axis:
        if axis == 'z':
            # Compute angle between star spin and z axis
            angle = self.misalignment_angle(np.array([0., 0., 1.]), attribute)
        
            # Find axis of rotation of star spin vector
            x  = np.array([0., 0., 1.])
            x -= x.dot(attribute) * attribute
            x /= np.linalg.norm(x)
            axis_of_rotation = np.cross(attribute, x)
            matrix = self.rotation_matrix(axis_of_rotation, angle)
            
        if axis == 'y':
            # Compute angle between star spin and y axis
            angle = self.misalignment_angle(np.array([0., 1., 0.]), attribute)
        
            # Find axis of rotation of star spin vector
            x  = np.array([0., 1., 0.])
            x -= x.dot(attribute) * attribute
            x /= np.linalg.norm(x)
            axis_of_rotation = np.cross(attribute, x)
            matrix = self.rotation_matrix(axis_of_rotation, angle)
            
        if axis == 'x':
            # Compute angle between star spin and y axis
            angle = self.misalignment_angle(np.array([1., 0., 0.]), attribute)
        
            # Find axis of rotation of star spin vector
            x  = np.array([1., 0., 0.])
            x -= x.dot(attribute) * attribute
            x /= np.linalg.norm(x)
            axis_of_rotation = np.cross(attribute, x)
            matrix = self.rotation_matrix(axis_of_rotation, angle)
            
        return angle, matrix
        
    def rotate_galaxy(self, matrix, data):
        """ For a given set of galaxy data, work out the rotated coordinates 
        and other data centred on [0, 0, 0], accounting for the perculiar 
        velocity of the galaxy"""
        
        # Where we store the new data
        new_data = {}
        
        for header in data.keys():
            if (header == 'Coordinates') or (header == 'Velocity'):
                new_data[header] = self.rotate_coords(matrix, data[header])
            else:
                new_data[header] = data[header]
        
        return new_data
        
    def rotate_coords(self, matrix, coords):
        # Compute new coords after rotation
        rotation = []
        for coord in coords:
            rotation.append(np.dot(matrix, coord))
        
        rotation = np.stack(rotation)
        
        return rotation
        
    def rotate_around_axis(self, axis, angle, attribute):
        # Finds angle given a defined x, y, z axis:
        if axis == 'z':
            # Rotate around z-axis
            axis_of_rotation = np.array([0., 0., 1.])
            matrix = self.rotation_matrix(axis_of_rotation, angle)
            
        if axis == 'y':
            # Rotate around z-axis
            axis_of_rotation = np.array([0., 1., 0.])
            matrix = self.rotation_matrix(axis_of_rotation, angle)

        if axis == 'x':
            # Rotate around z-axis
            axis_of_rotation = np.array([1., 0., 0.])
            matrix = self.rotation_matrix(axis_of_rotation, angle)

        return matrix
        
    def kappa_co(self, arr, radius):
        # Compute distance to centre and mask all within stelhalfrad
        r  = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.where(r <= radius)
        
        # Compute angular momentum within specified radius
        L  = np.cross(arr['Coordinates'][mask] * arr['Mass'][:, None][mask], arr['Velocity'][mask])
        
        # Projected radius along disk within specified radius
        rad_projected = np.linalg.norm(arr['Coordinates'][:,:2][mask], axis=1)
        
        # Total kinetic energy of stars
        K_tot = np.sum(0.5 * arr['Mass'][mask] * np.square(np.linalg.norm(arr['Velocity'][mask], axis=1)))
        
        # Kinetic energy of ordered co-rotation (using only angular momentum in z-axis)
        K_rot = np.sum(0.5 * np.square(L[:,2] / rad_projected) / arr['Mass'][mask])
        
        return K_rot/K_tot
        
        
register_sauron_colormap()

# list of simulations
mySims = np.array([('RefL0012N0188', 12)])  

GroupNum = 4
SubGroupNum = 0

# Initial extraction of galaxy data
galaxy = Subhalo_Extract(mySims, dataDir, GroupNum, SubGroupNum)

calc_spin_rad = 2*galaxy.halfmass_rad   # pkpc
calc_kappa_rad = 30                     # pkpc
trim_rad = 30                           # pkpc
align_rad = 30
viewing_angle = 10

subhalo = Subhalo(galaxy.gn, galaxy.sgn, galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas, viewing_angle,
                                    calc_spin_rad, calc_kappa_rad, trim_rad, align_rad)     #align_rad=False








