import h5py
import numpy as np
import astropy.units as u
import random
import math
import vorbin
import matplotlib as mpl
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from pafit.fit_kinematic_pa import fit_kinematic_pa
from astropy.constants import G
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.colors as colors
import matplotlib.cm as cm
from read_dataset import read_dataset
from read_header import read_header
from read_dataset_dm_mass import read_dataset_dm_mass
from pyread_eagle import EagleSnapshot
import eagleSqlTools as sql
from graphformat import graphformat
from plotbin.sauron_colormap import register_sauron_colormap

# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'

register_sauron_colormap()

class Subhalo_Extract:
    
    def __init__(self, sim, gn, sgn, load_region_length=2):       # cMpc/h
        # Assigning subhalo properties
        self.gn           = gn
        self.sgn          = sgn
        
        # Load information from the header
        self.a, self.h, self.boxsize = read_header() # units of scale factor, h, and L [cMpc/h]        ASSUMES Z=0 I THINK
        
        # For a given gn and sgn, run sql query on SubFind catalogue
        myData = self.query(sim)
        
        # Assiging subhalo properties
        self.stelmass     = myData['stelmass']                                                              # [Msun]
        self.gasmass      = myData['gasmass']                                                               # [Msun]
        self.halfmass_rad = myData['rad']/1000                                                              # [pMpc]
        self.centre       = np.array([myData['x'], myData['y'], myData['z']])                               # [pMpc]
        self.centre_mass  = np.array([myData['x_mass'], myData['y_mass'], myData['z_mass']])                # [pMpc]
        self.perc_vel     = np.array([myData['vel_x'], myData['vel_y'], myData['vel_z']]) * u.km.to(u.Mpc)  # [pMpc/s]
        
        # Load data for stars and gas in non-centred units
        self.stars        = self.read_galaxy(4, gn, sgn, self.centre, load_region_length)
        self.gas          = self.read_galaxy(0, gn, sgn, self.centre, load_region_length)
        

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
            
        
    def read_galaxy(self, itype, gn, sgn, centre, load_region_length):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates, velocty, and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity."""

        # Where we store all the data
        data = {}
        
        # Initialize read_eagle module.
        eagle_data = EagleSnapshot(dataDir)
        
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
        for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'Velocity']:
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
        
        # Periodic wrap coordinates around centre (in proper units). boxsize converted from cMpc/h -> pMpc
        boxsize = self.boxsize/self.h       # [pMpc]
        data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
        
        return data



class Subhalo:
    
    def __init__(self, gn, sgn, stelmass, gasmass, halfmass_rad, centre, centre_mass, perc_vel, stars, gas, viewing_angle):
        # Assign bulk galaxy values to this subhalo
        self.gn             = gn
        self.sgn            = sgn
        self.stelmass       = np.sum(stars['Mass'])     # [Msun]
        self.gasmass        = np.sum(gas['Mass'])       # [Msun]
        self.halfmass_rad   = halfmass_rad              # [pMpc]
        self.centre         = centre                    # [pMpc]
        self.centre_mass    = centre_mass               # [pMpc]
        self.perc_vel       = perc_vel                  # [pMpc/s]
        self.viewing_angle  = viewing_angle             # [deg]
        
        # Find original spin vector and rotation matrix to rotate that vector about z-axis (degrees, clockwise)
        stars_spin_original, stars_L_original = self.spin_vector(stars, self.centre, self.perc_vel, 2*self.halfmass_rad, 'stars')    # unit vector, [Msun pMpc^2 /s]
        gas_spin_original, gas_L_original = self.spin_vector(gas, self.centre, self.perc_vel, 2*self.halfmass_rad, 'gas')    # unit vector, [Msun pMpc^2 /s]
        matrix = self.rotate_around_axis('z', 360. - viewing_angle, stars_spin_original)
        
        # Compute new data of rotated particle data
        stars_rotated = self.rotate_galaxy(matrix, self.centre, self.perc_vel, stars)
        gas_rotated   = self.rotate_galaxy(matrix, self.centre, self.perc_vel, gas)
        
        # Assign particle data
        self.stars_coords = stars_rotated['Coordinates']            # [pMpc] (centred)
        self.gas_coords   = gas_rotated['Coordinates']              # [pMpc] (centred)
        self.stars_vel    = stars_rotated['Velocity']               # [pMpc/s] (centred)
        self.gas_vel      = gas_rotated['Velocity']                 # [pMpc/s] (centred)
        self.stars_mass   = stars_rotated['Mass']
        self.gas_mass     = gas_rotated['Mass']
        
        #print(self.perc_vel*u.Mpc.to(u.km))
        #print(stars['Velocity']*u.Mpc.to(u.km))
        #print(self.stars_vel*u.Mpc.to(u.km))
        
        # Find new spin vectors
        self.stars_spin, self.stars_L = self.spin_vector(stars_rotated, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'stars')  # unit vector, [Msun pMpc^2 /s]
        self.gas_spin, self.gas_L     = self.spin_vector(gas_rotated, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'gas')      # unit vector, [Msun pMpc^2 /s]
        
        # Compute angle between star spin and gas angular momentum vector
        self.mis_angle = self.misalignment_angle(self.stars_spin, self.gas_spin)   # [deg]
    
    
    def spin_vector(self, arr, origin, perculiar_velocity, radius, desc):
        # Compute distance to centre and mask all within stelhalfrad
        r  = np.linalg.norm(arr['Coordinates'] - origin, axis=1)
        mask = np.where(r <= radius)
        
        #print("Total %s particles in subhalo: %i"%(desc, len(r)))
        #r = r[mask]
        #print("Total %s particles in %.5f kpc: %i\n"%(desc, radius*1000, len(r)))
        
        
        # Sanity check plot
        """v = np.linalg.norm((arr['Velocity'] - self.perc_vel)*u.Mpc.to(u.km), axis=1)
        plt.scatter(r, v, s=0.5)
        plt.scatter(r[mask], v[mask], s=0.5)
        plt.show()"""


        # Finding spin angular momentum vector of each individual particle of gas and stars, where [:, None] is done to allow multiplaction of N3*N1 array. Equation D.25
        L  = np.cross((arr['Coordinates'] - origin)[mask] * arr['Mass'][:, None][mask], (arr['Velocity'] - perculiar_velocity)[mask])
        
        # Summing for total angular momentum and dividing by mass to get the spin vectors
        spin = np.sum(L, axis=0)/np.sum(arr['Mass'])
        #print(spin)
        
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
        
        
    def rotate_galaxy(self, matrix, origin, perculiar_velocity, data):
        """ For a given set of galaxy data, work out the rotated coordinates 
        and other data centred on [0, 0, 0], accounting for the perculiar 
        velocity of the galaxy"""
        
        # Where we store the new data
        new_data = {}
        
        new_data['GroupNumber']     = data['GroupNumber']
        new_data['SubGroupNumber']  = data['SubGroupNumber']
        new_data['Mass']            = data['Mass']
        new_data['Coordinates']     = self.rotate_coords(matrix, data['Coordinates'] - origin)
        new_data['Velocity']        = self.rotate_coords(matrix, data['Velocity'] - perculiar_velocity)
        
        return new_data
            
        
    
class Subhalo_Align:
    
    def __init__(self, gn, sgn, stelmass, gasmass, halfmass_rad, centre, centre_mass, perc_vel, stars, gas):
        # Assign bulk galaxy values to this subhalo
        self.gn             = gn
        self.sgn            = sgn
        self.stelmass       = stelmass          # [Msun]
        self.gasmass        = gasmass           # [Msun]
        self.halfmass_rad   = halfmass_rad      # [pMpc]
        self.centre         = centre            # [pMpc]
        self.centre_mass    = centre_mass       # [pMpc]
        self.perc_vel       = perc_vel          # [pMpc/s]
        
        # Compute spin vectors within radius of interest (30 pkpc)        0.03 replace with self.halfmass_rad
        self.stars_spin = self.spin_vector(stars, self.centre, self.perc_vel, 0.03, 'stars') # unit vector centred
        self.gas_spin = self.spin_vector(gas, self.centre, self.perc_vel, 0.03, 'gas')       # unit vector centred
        
        # Find z_angle between stars and a given axis as reference (z)
        self.z_angle, matrix = self.orientate('z', self.stars_spin)
        
        # Compute new data of aligned particle data
        stars_aligned = self.rotate_galaxy(matrix, self.centre, self.perc_vel, stars)
        gas_aligned   = self.rotate_galaxy(matrix, self.centre, self.perc_vel, gas)
        
        # Assign particle data
        self.stars_coords     = stars['Coordinates'] - self.centre
        self.gas_coords       = gas['Coordinates'] - self.centre
        self.stars_coords_new = stars_aligned['Coordinates']
        self.gas_coords_new   = gas_aligned['Coordinates']
        
        # Find new spin vectors within halfmassrad
        self.stars_spin_new = self.spin_vector(stars_aligned, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'stars')
        self.gas_spin_new   = self.spin_vector(gas_aligned, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'gas')
        
        # Compute angle between star spin and gas angular momentum vector
        self.mis_angle = self.misalignment_angle(self.stars_spin_new, self.gas_spin_new)       # [deg]
        
        # Compute stellar cp-rotating kinetic energy fraction within a radius
        self.kappa = self.kappa_co(stars_aligned, np.array([0, 0, 0]), np.array([0, 0, 0]), 0.03)  # [pkpc]
        
    
    def spin_vector(self, arr, origin, perculiar_velocity, radius, desc):
        # Compute distance to centre and mask all within stelhalfrad
        r  = np.linalg.norm(arr['Coordinates'] - origin, axis=1)
        mask = np.where(r <= radius)
        
        #print("Total %s particles in subhalo: %i"%(desc, len(r)))
        #r = r[mask]
        #print("Total %s particles in %.5f kpc: %i\n"%(desc, radius*1000, len(r)))
        
        
        # Sanity check plot
        """v = np.linalg.norm((arr['Velocity'] - self.perc_vel)*u.Mpc.to(u.km), axis=1)
        plt.scatter(r, v, s=0.5)
        plt.scatter(r[mask], v[mask], s=0.5)
        plt.show()"""


        # Finding spin angular momentum vector of each individual particle of gas and stars, where [:, None] is done to allow multiplaction of N3*N1 array. Equation D.25
        L  = np.cross((arr['Coordinates'] - origin)[mask] * arr['Mass'][:, None][mask], (arr['Velocity'] - perculiar_velocity)[mask])
        
        # Summing for total angular momentum and dividing by mass to get the spin vectors
        spin = np.sum(L, axis=0)/np.sum(arr['Mass'])
        #print(spin)
        
        # Expressing as unit vector
        spin_unit = spin / (spin[0]**2 + spin[1]**2 + spin[2]**2)**0.5
        
        return spin_unit
        
    
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
        
    
    def rotate_coords(self, matrix, coords):
        # Compute new coords after rotation
        rotation = []
        for coord in coords:
            rotation.append(np.dot(matrix, coord))
        
        rotation = np.stack(rotation)
        
        return rotation
        
        
    def rotate_galaxy(self, matrix, origin, perculiar_velocity, data):
        """ For a given set of galaxy data, work out the rotated coordinates 
        and other data centred on [0, 0, 0], accounting for the perculiar 
        velocity of the galaxy"""
        
        # Where we store the new data
        new_data = {}
        
        new_data['GroupNumber']     = data['GroupNumber']
        new_data['SubGroupNumber']  = data['SubGroupNumber']
        new_data['Mass']            = data['Mass']
        new_data['Coordinates']     = self.rotate_coords(matrix, data['Coordinates'] - origin)
        new_data['Velocity']        = self.rotate_coords(matrix, data['Velocity'] - perculiar_velocity)
        
        return new_data
        
            
    def kappa_co(self, arr, origin, perculiar_velocity, radius):
        # Compute distance to centre and mask all within stelhalfrad
        r  = np.linalg.norm(arr['Coordinates'] - origin, axis=1)
        mask = np.where(r <= radius)
        
        # Compute angular momentum within specified radius
        L  = np.cross((arr['Coordinates'] - origin)[mask] * arr['Mass'][:, None][mask], (arr['Velocity'] - perculiar_velocity)[mask])
        
        # Projected radius along disk within specified radius
        rad_projected = np.linalg.norm(arr['Coordinates'][:,:2][mask] - origin[:2], axis=1)
        
        # Total kinetic energy of stars
        K_tot = np.sum(0.5 * arr['Mass'][mask] * np.square(np.linalg.norm(arr['Velocity'][mask] - perculiar_velocity, axis=1)))
        
        # Kinetic energy of ordered co-rotation (using only angular momentum in z-axis)
        K_rot = np.sum(0.5 * np.square(L[:,2] / rad_projected) / arr['Mass'][mask])
        
        return K_rot/K_tot
        
        
if __name__ == '__main__':
    # list of simulations
    mySims = np.array([('RefL0012N0188', 12)])   
    
    def galaxy_render(GroupNumList=np.array([4]),
                        SubGroupNum=0, 
                        particles=10000,
                        minangle=0,
                        maxangle=360, 
                        stepangle=30, 
                        boxradius=50,      # [pkpc] 
                        stars=False, 
                        gas=True, 
                        stars_rot=False, 
                        gas_rot=False,
                        stars_total = True, 
                        gas_total = True, 
                        centre_of_pot=True, 
                        centre_of_mass=True,
                        axis=True):
        
        # list of simulations
        mySims = np.array([('RefL0012N0188', 12)])   
        
        for GroupNum in GroupNumList: #np.arange(1, 16, 1):  
            # Initialise subhalo call
            
            # Initial extraction of galaxy data
            galaxy = Subhalo_Extract(mySims, GroupNum, SubGroupNum)
            subhalo_al = Subhalo_Align(galaxy.gn, galaxy.sgn, galaxy.stelmass, galaxy.gasmass, galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas)

            # Print galaxy properties
            print('\nGROUP NUMBER:', subhalo_al.gn) 
            print('KAPPA: %.2f' %subhalo_al.kappa)
            print('STELLAR MASS: %.3f log10 M$_\odot$' %np.log10(subhalo_al.stelmass))            # [pMsun]
            print('HALFMASS RAD: %.3f pkpc' %(1000*subhalo_al.halfmass_rad))        # [pkpc]
            print('MISALIGNMENT ANGLE: %.1f deg' %subhalo_al.mis_angle)     # [deg]
            print('CENTRE [pMpc]:', subhalo_al.centre)                         # [pMpc]
            print('PERCULIAR VELOCITY [pkm/s]:', (subhalo.perc_vel*u.Mpc.to(u.km)))           # [pkm/s]
            
            # Graph initialising and base formatting
            graphformat(8, 11, 11, 11, 11, 5, 5)
            fig = plt.figure() 
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
    
            # Plot n random points 3d scatter so we don't need to use the entire data
            for n in np.arange(0, particles, 1):
                # Mask to random number of particles given by galaxy_render
                mask_stars = random.randint(0, len(subhalo_al.stars_coords[:])-1)
                mask_gas = random.randint(0, len(subhalo_al.gas_coords[:])-1)
                
                if stars == True:
                    # Plot original stars
                    ax.scatter(1000*subhalo_al.stars_coords[mask_stars][0], 1000*subhalo_al.stars_coords[mask_stars][1], 1000*subhalo_al.stars_coords[mask_stars][2], s=0.02, alpha=0.9, c='khaki')
                if gas == True:
                    # Plot original gas
                    ax.scatter(1000*subhalo_al.gas_coords[mask_gas][0], 1000*subhalo_al.gas_coords[mask_gas][1], 1000*subhalo_al.gas_coords[mask_gas][2], s=0.02, alpha=0.9, c='blue')
                if stars_rot == True:
                    # Plot rotated stars
                    ax.scatter(1000*subhalo_al.stars_coords_new[mask_stars][0], 1000*subhalo_al.stars_coords_new[mask_stars][1], 1000*subhalo_al.stars_coords_new[mask_stars][2], s=0.02, alpha=0.9, c='khaki')
                if gas_rot == True:
                    # Plot rotated gas
                    ax.scatter(1000*subhalo_al.gas_coords_new[mask_gas][0], 1000*subhalo_al.gas_coords_new[mask_gas][1], 1000*subhalo_al.gas_coords_new[mask_gas][2], s=0.02, alpha=0.9, c='blue')
    
            if stars == True:
                # Plot original stars spin vector
                ax.quiver(0, 0, 0, subhalo_al.stars_spin[0]*boxradius*0.4, subhalo_al.stars_spin[1]*boxradius*0.4, subhalo_al.stars_spin[2]*boxradius*0.4, color='red', linewidth=1)
            if gas == True:
                # Plot original stars spin vector
                ax.quiver(0, 0, 0, subhalo_al.gas_spin[0]*boxradius*0.4, subhalo_al.gas_spin[1]*boxradius*0.4, subhalo_al.gas_spin[2]*boxradius*0.4, color='navy', linewidth=1)
            if stars_rot == True:
                # Plot rotated stars spin vector
                ax.quiver(0, 0, 0, subhalo_al.stars_spin_new[0]*boxradius*0.4, subhalo_al.stars_spin_new[1]*boxradius*0.4, subhalo_al.stars_spin_new[2]*boxradius*0.4, color='red', linewidth=1)
            if gas_rot == True:
                # Plot rotated stars spin vector
                ax.quiver(0, 0, 0, subhalo_al.gas_spin_new[0]*boxradius*0.4, subhalo_al.gas_spin_new[1]*boxradius*0.4, subhalo_al.gas_spin_new[2]*boxradius*0.4, color='navy', linewidth=1)
                
            if centre_of_pot == True:
                # Plot centre_of_potential
                ax.scatter(0, 0, 0, c='pink', s=3, zorder=10)
            if centre_of_mass == True:
                # Plot centre_of_mass
                ax.scatter(1000*(subhalo_al.centre_mass[0] - subhalo_al.centre[0]), 1000*(subhalo_al.centre_mass[1] - subhalo_al.centre[1]), 1000*(subhalo_al.centre_mass[2] - subhalo_al.centre[2]), c='purple', s=3, zorder=10)
            
            if axis == True:
                # Plot axis
                ax.quiver(0, 0, 0, 10, 0, 0, color='r', linewidth=0.5)
                ax.quiver(0, 0, 0, 0, 10, 0, color='g', linewidth=0.5)
                ax.quiver(0, 0, 0, 0, 0, 10, color='b', linewidth=0.5)
                
            # Plot formatting
            ax.set_facecolor('xkcd:black')
            ax.set_xlim(-boxradius, boxradius)
            ax.set_ylim(-boxradius, boxradius)
            ax.set_zlim(-boxradius, boxradius)   
             
            print('VIEWING ANGLES: ', end='')
            for ii in np.arange(minangle, maxangle+1, stepangle):
                print(ii , end=' ')                 # [deg]
                ax.view_init(0, ii)
                
                plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%i/galaxy_gas/galaxy_%i.jpeg"%(GroupNum,ii), dpi=300)
                
            plt.close()
            print('')
            
    
    def velocity_projection(GroupNumList = np.array([4]),
                            SubGroupNum = 0,
                            minangle = 0,
                            maxangle = 180,
                            stepangle = 10,
                            boxradius = 40,         # [pkpc]
                            resolution = 1,          # [pkpc]
                            target_particles = 10):      
        
        data_catalogue = {}
        
        for GroupNum in GroupNumList:
            # Initial extraction of galaxy data
            galaxy = Subhalo_Extract(mySims, GroupNum, SubGroupNum)
            
            # Empty arrays to collect relevant data
            gn_list            = []
            stelmass_list      = []
            kappa_list         = []
            mis_angle_list     = []
            viewing_angle_list = []
            pa_fit_list        = []
            pa_fit_error_list  = []
            pa_fit_list2        = []
            pa_fit_error_list2  = []
            
            
            # int count to broadcast print statements of each galaxy
            i = 0 
            
            # Use the data we have called to find a variety of properties
            for viewing_angle in np.arange(minangle, maxangle+1, stepangle):
                # If we want the original values, enter 0 for viewing angle
                subhalo = Subhalo(galaxy.gn, galaxy.sgn, galaxy.stelmass, galaxy.gasmass, galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas, viewing_angle)
            
                boxradius = 2*subhalo.halfmass_rad*1000
                
                # Print galaxy properties for first call
                if i == 0:
                    # Call subhalo align for kappa only once
                    subhalo_al = Subhalo_Align(galaxy.gn, galaxy.sgn, galaxy.stelmass, galaxy.gasmass, galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas)
                    
                    print('\nGROUP NUMBER:', subhalo.gn) 
                    print('STELLAR MASS [log10 Msun]: %.3f' %np.log10(subhalo.stelmass))       # [pMsun]
                    print('HALFMASS RAD [pkpc]: %.3f' %(1000*subhalo.halfmass_rad))            # [pkpc]
                    print('KAPPA: %.2f' %subhalo_al.kappa)
                    print('MISALIGNMENT ANGLE [deg]: %.1f' %subhalo.mis_angle)                 # [deg]
                    print('CENTRE [pMpc]:', subhalo.centre)                                    # [pMpc]
                    print('PERCULIAR VELOCITY [pkm/s]:', (subhalo.perc_vel*u.Mpc.to(u.km)))    # [pkpc]
                print('VIEWING ANGLE: ', viewing_angle)
                
                
                """
                #########################
                Purpose
                -------
                Will create 2dhist to mimic flux pixels, mass-weight 
                the velocity and find mean within bins.
                
                Calling function
                ----------------
                centre of square bins, output of bins = weight_histo(coords, velocity, mass)
                
                
                Input Parameters
                ----------------
                
                gn: int
                    Groupnumber for filesave
                viewing_angle: float or int
                    viewing angle for filesave
                coords: [[x, y, z], [x, y, z], ...]
                    Input raw particle coordinates such as 
                    subhalo.stars_coords as array of array. 
                    
                    Take in kpc -> kpc
                velocity: [[x, y, z], [x, y, z], ...]
                    Input raw particle velocity 
                
                    Take in kms ->kms
                mass: [m1, m2, m3, ...]
                    Input raw particle masses, any unit
                bin_res: float
                    Size of bins in pkpc
                boxradius: float
                    2d edges of 2dhistogram in pkpc
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
                
                #######################
                """
                
                def weight_histo(gn, viewing_angle, coords, velocity, mass, 
                                 bin_res=resolution, 
                                 boxradius=boxradius, 
                                 quiet=1, 
                                 plot=1):
                                 
                    # Assign xy, y, and z values for histogram2d
                    x   = [row[1] for row in coords]
                    y   = [row[2] for row in coords]
                    vel = [row[0] for row in velocity*-1.]
                    
                    # Histogram to find counts in each bin
                    pixel = math.ceil(boxradius*2 / bin_res)
                    counts, xbins, ybins = np.histogram2d(y, x, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                    
                    # Histogram to find total mass-weighted velocity in those bins, and account for number of values in each bin
                    vel_weighted, _, _ = np.histogram2d(y, x, weights=vel*mass/np.mean(mass), bins=(xbins, ybins), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                    vel_weighted = np.divide(vel_weighted, counts, out=np.zeros_like(counts), where=counts!=0)
                    
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
                        
                        plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_projection_hist/projection_2r_%s.jpeg' %(str(gn), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.5)
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
                    
                    return points, points_num, points_vel
                    
                
                """
                #########################
                Purpose
                -------
                Will create vornoi plot using the 2dhist, and output mean vel
                values within the bin
                
                Calling function
                ----------------
                centre of square bins, output of bins = voronoi_tessalate(coords, velocity, mass)
                
                
                Input Parameters
                ----------------
                
                gn: int
                    Groupnumber for filesave
                viewing_angle: float or int
                    viewing angle for filesave
                coords: [[x, y, z], [x, y, z], ...]
                    Input raw particle coordinates such as 
                    subhalo.stars_coords as array of array. 
                    
                    Take in kpc -> kpc
                velocity: [[x, y, z], [x, y, z], ...]
                    Input raw particle velocity 
                
                    Take in kms ->kms
                mass: [m1, m2, m3, ...]
                    Input raw particle masses, any unit
                bin_res: float
                    Size of bins in pkpc
                boxradius: float
                    2d edges of 2dhistogram in pkpc
                target_particles: int
                    Min. number of particles to vornoi bin
                quiet: boolean
                    Mutes the prints
                plot: boolean
                    Can plot and savefig
                
                
                
                Output Parameters
                -----------------
                
                points: [[x, y], [x, y], ...]
                    x and y coordinates of CENTRE of bin
                    in units of whatever was put in
                vel_bin: [v1, v2, v3, ...]
                    mass-weighted mean velocity that was
                    grouped by min. particle count
                vor: something
                    voronoi tessalation details to be 
                    fed into the vornoi_plot_2d
                
                #######################
                """
                
                def voronoi_tessalate(gn, viewing_angle, coords, velocity, mass, 
                                      bin_res=1, 
                                      boxradius=boxradius, 
                                      target_particles = target_particles,
                                      quiet=1, 
                                      plot=0):
                                      
                    # Assign xy, y, and z values for histogram2d
                    x   = [row[1] for row in coords]
                    y   = [row[2] for row in coords]
                    vel = [row[0] for row in velocity*-1.]
                    
                    # Histogram to find counts in each bin
                    pixel = math.ceil(boxradius*2 / bin_res)
                    counts, xbins, ybins = np.histogram2d(y, x, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                    
                    # Histogram to find total mass-weighted velocity in those bins, and returns TOTAL velocity (mean in bin will be calculated after voronoi)
                    vel_weighted, _, _ = np.histogram2d(y, x, weights=vel*mass/np.mean(mass), bins=(xbins, ybins), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                    
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
                        plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_projection_voronoi/voronoi_output/vorbin_%s.jpeg' %(str(gn), str(viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.5)
                        plt.close()
                    if not plot:
                        _, x_gen, y_gen, _, _, bin_count, vel, _, _ = voronoi_2d_binning(points[:,0], points[:,1], points_vel, points_num, target_particles, plot=0, quiet=quiet, sn_func=None)
                    
                    if not quiet:
                        # print number of particles in each bin
                        print(bin_count)
                    
                    # Create tessalation, append points at infinity to color plot edges
                    points = np.column_stack((x_gen, y_gen))   
                    vel_bin = np.divide(vel, bin_count)     # find mean in each square bin (total velocity / total particles in voronoi bins)
                    points_inf = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis=0) 
                    vor = Voronoi(points_inf)
                    
                    return points, vel_bin, vor
                
                def plot_2dhist():
                    
                def plot_voronoi():
                    
                # Initialise figure
                graphformat(8, 11, 11, 11, 11, 3.75, 3)
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
                
                
                ### VORONOI TESSALATION ROUTINE
                # Tessalate stars    
                points_stars, vel_bin_stars, vor = voronoi_tessalate(subhalo.gn, subhalo.viewing_angle, subhalo.stars_coords*1000, subhalo.stars_vel*u.Mpc.to(u.km), subhalo.stars_mass)
                
                # normalize chosen colormap
                minima = -abs(max(vel_bin_stars, key=abs))
                minima = -200
                maxima = -minima
                norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
                mapper = cm.ScalarMappable(norm=norm, cmap='sauron')         #cmap=cm.coolwarm)
                    
                # plot Voronoi diagram, and fill finite regions with color mapped from vel value
                voronoi_plot_2d(vor, ax=ax1, show_points=False, show_vertices=False, line_width=0, s=1)
                for r in range(len(vor.point_region)):
                    region = vor.regions[vor.point_region[r]]
                    if not -1 in region:
                        polygon = [vor.vertices[i] for i in region]
                        ax1.fill(*zip(*polygon), color=mapper.to_rgba(vel_bin_stars[r]))
                
                
                # Tessalate gas   
                points_gas, vel_bin_gas, vor = voronoi_tessalate(subhalo.gn, subhalo.viewing_angle, subhalo.gas_coords*1000, subhalo.gas_vel*u.Mpc.to(u.km), subhalo.gas_mass)
                
                # plot Voronoi diagram, and fill finite regions with color mapped from vel value
                voronoi_plot_2d(vor, ax=ax2, show_points=False, show_vertices=False, line_width=0, s=1)
                for r in range(len(vor.point_region)):
                    region = vor.regions[vor.point_region[r]]
                    if not -1 in region:
                        polygon = [vor.vertices[i] for i in region]
                        ax2.fill(*zip(*polygon), color=mapper.to_rgba(vel_bin_gas[r]))
                
                # Graph formatting
                for ax in [ax1, ax2]:
                    ax.set_xlim(-boxradius, boxradius)
                    ax.set_ylim(-boxradius, boxradius)
                    ax.set_xlabel('x-axis [pkpc]')
                ax1.set_ylabel('y-axis [pkpc]')
                ax1.set_title('Stars')
                ax2.set_title('Gas')
                ax1.text(-boxradius, boxradius+1, '1kpc hist, target %i particles' %target_particles, fontsize=8)
                
                # Colorbar
                cax = plt.axes([0.92, 0.11, 0.015, 0.77])
                plt.colorbar(mapper, cax=cax, label='mass-weighted mean velocity [km/s]', extend='both')
                
                """# Overplot original points for comparisson
                plt.scatter([row[1] for row in subhalo.stars_coords*1000], [row[2] for row in subhalo.stars_coords*1000], c=[row[0] for row in subhalo.stars_vel*u.Mpc.to(u.km)*-1.], s=0.5, edgecolor='k')"""
                
                plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_projection_voronoi/voronoi_2r_ß%s.jpeg' %(str(subhalo.gn), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.3)
                plt.close()
                
                """# Plot 2dhisto of velocity
                _, _, _ = weight_histo(subhalo.gn, subhalo.viewing_angle, subhalo.stars_coords*1000, subhalo.stars_vel*u.Mpc.to(u.km), subhalo.stars_mass)"""
                """# Collect voronoi details
                _, _, _ = voronoi_tessalate(subhalo.gn, subhalo.viewing_angle, subhalo.stars_coords*1000, subhalo.stars_vel*u.Mpc.to(u.km), subhalo.stars_mass)"""
        
        
                ### PA FIT ROUTINE
                # Run pa_fit on voronoi       
                angle_stars, angle_err_stars, velsyst_gas = fit_kinematic_pa(points_stars[:,0], points_stars[:,1], vel_bin_stars, quiet=1, plot=1)
                plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_PA/PA_stars_%s.jpeg' %(str(subhalo.gn), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.2)
                plt.close()
                angle_gas, angle_err_gas, velsyst_gas = fit_kinematic_pa(points_gas[:,0], points_gas[:,1], vel_bin_gas, quiet=1, plot=1)
                plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_PA/PA_gas_%s.jpeg' %(str(subhalo.gn), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.2)
                plt.close()
                
                # Find misalignment angle from pa
                pa_fit = abs(angle_stars - angle_gas)
                pa_fit_error = angle_err_stars + angle_err_gas
                print("PA: %.1f +/- %.1f" %(pa_fit, pa_fit_error))
                
                # Append to data
                pa_fit_list       = np.append(pa_fit_list, pa_fit)
                pa_fit_error_list = np.append(pa_fit_error_list, pa_fit_error)
                
                
                # Check for consistency with 2dhist
                points_stars, _, vel_bin_stars = weight_histo(subhalo.gn, subhalo.viewing_angle, subhalo.stars_coords*1000, subhalo.stars_vel*u.Mpc.to(u.km), subhalo.stars_mass)
                points_gas, _, vel_bin_gas = weight_histo(subhalo.gn, subhalo.viewing_angle, subhalo.gas_coords*1000, subhalo.gas_vel*u.Mpc.to(u.km), subhalo.gas_mass)
                
                angle_stars, angle_err_stars, velsyst_gas = fit_kinematic_pa(points_stars[:,0], points_stars[:,1], vel_bin_stars, quiet=1, plot=1)
                plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_PA/PA_stars_hist_%s.jpeg' %(str(subhalo.gn), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.2)
                plt.close()
                angle_gas, angle_err_gas, velsyst_gas = fit_kinematic_pa(points_gas[:,0], points_gas[:,1], vel_bin_gas, quiet=1, plot=1)
                plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_PA/PA_gas_hist_%s.jpeg' %(str(subhalo.gn), str(subhalo.viewing_angle)), dpi=300, bbox_inches='tight', pad_inches=0.2)
                plt.close()
                
                # Find misalignment angle from pa
                pa_fit = abs(angle_stars - angle_gas)
                pa_fit_error = angle_err_stars + angle_err_gas
                print("PA: %.1f +/- %.1f" %(pa_fit, pa_fit_error))
                
                # Append to data
                pa_fit_list2       = np.append(pa_fit_list2, pa_fit)
                pa_fit_error_list2 = np.append(pa_fit_error_list2, pa_fit_error)
                
                
                
                
                i = i + 1
                
            
            ### Start of once per galaxy
            
            print('')
            
            # Collect values for each galaxy
            gn_list        = np.append(gn_list, subhalo.gn)
            stelmass_list  = np.append(stelmass_list, subhalo.stelmass)
            kappa_list     = np.append(kappa_list, subhalo_al.kappa) 
            mis_angle_list = np.append(mis_angle_list, subhalo.mis_angle)
            #pa_fit_list    = np.append(pa_fit_list, pa_fit)  


        ### Start of GroupNum loop
        
        # Append misaslignment angles once per galaxy
        data_catalogue['GroupNum'] = gn_list
        data_catalogue['Stelmass'] = stelmass_list
        data_catalogue['Kappa']    = kappa_list
        data_catalogue['MisAngle'] = mis_angle_list
        data_catalogue['PA']       = pa_fit_list
        data_catalogue['PA_error'] = pa_fit_error_list
        data_catalogue['PA2']       = pa_fit_list2
        data_catalogue['PA_error2'] = pa_fit_error_list2
        
        plt.figure()
        plt.scatter(np.arange(0, 181, 10), data_catalogue['PA'], c='r', label='voronoi')
        plt.scatter(np.arange(0, 181, 10), data_catalogue['PA2'], c='b', label='2dhist')
        plt.errorbar(np.arange(0, 181, 10), data_catalogue['PA'], xerr=None, yerr=data_catalogue['PA_error'], c='r')
        plt.errorbar(np.arange(0, 181, 10), data_catalogue['PA2'], xerr=None, yerr=data_catalogue['PA_error2'], c='k')
        plt.axhline(data_catalogue['MisAngle'], c='k', ls='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/PA_comparison.jpeg' %str(subhalo.gn), dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        
        
        print(data_catalogue)
        
            
        # Here starts total
    
    
    
    def plot_spin_vectors(viewing_angle,
                          GroupNum = 16,
                          SubGroupNum = 0,
                          stars = False,
                          gas = False,
                          stars_rot = True, 
                          gas_rot = True,
                          plot_centre = True, 
                          boxradius = 40,           # [pkpc]
                          particles = 2000):
        
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, GroupNum, SubGroupNum)
        subhalo = Subhalo(galaxy.gn, galaxy.sgn, galaxy.stelmass, galaxy.gasmass, galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas, viewing_angle)
        
        # Print galaxy properties    
        print('\nGROUP NUMBER:', subhalo.gn) 
        print('STELLAR MASS: %.3f [log10 M$_\odot$]' %np.log10(subhalo.stelmass))            # [pMsun]
        print('HALFMASS RAD: %.3f [pkpc]' %(1000*subhalo.halfmass_rad))        # [pkpc]
        print('CENTRE [pkpc]:', subhalo.centre)                         # [pMpc]
        print('PERCULIAR VELOCITY [pkm/s]:', (subhalo.perc_vel*u.Mpc.to(u.km)))           # [pkm/s]
        print('MISALIGNMENT ANGLE [deg]: %.1f' %subhalo.mis_angle)     # [deg]
    
        # Graph initialising and base formatting
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig = plt.figure() 
        ax = Axes3D(fig)
        #ax = fig.add_subplot(projection='3d')
        
        # Plot 100 random points 3d scatter so we don't need to use the entire data
        for n in np.arange(0, particles, 1):
            mask_stars = random.randint(0, len(subhalo.stars_coords[:])-1)
            mask_gas = random.randint(0, len(subhalo.gas_coords[:])-1)
            
            if stars == True:
                # Plot original stars
                ax.scatter(1000*subhalo.stars_coords[mask_stars][0], 1000*subhalo.stars_coords[mask_stars][1], 1000*subhalo.stars_coords[mask_stars][2], s=0.1, alpha=0.8, c='orange')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(1000*subhalo.stars_coords[mask_stars][0], 1000*subhalo.stars_coords[mask_stars][1], 1000*subhalo.stars_coords[mask_stars][2], 
                            subhalo.stars_vel[mask_stars][0]*0.5e14, subhalo.stars_vel[mask_stars][1]*0.5e14, subhalo.stars_vel[mask_stars][2]*0.5e14, linewidth=0.5, color='white')
                
                # Plot original spin vectors
                ax.quiver(0, 0, 0, subhalo.stars_spin[0]/100, subhalo.stars_spin[1]/100, subhalo.stars_spin[2]/100, color='r', linewidth=1)
                
            if gas == True:
                # Plot original gas
                ax.scatter(1000*subhalo.gas_coords[mask_gas][0], 1000*subhalo.gas_coords[mask_gas][1], 1000*subhalo.gas_coords[mask_gas][2], s=0.05, alpha=0.8, c='dodgerblue')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(1000*subhalo.gas_coords[mask_stars][0], 1000*subhalo.gas_coords[mask_stars][1], 1000*subhalo.gas_coords[mask_stars][2], 
                            subhalo.gas_vel[mask_stars][0]*0.5e14, subhalo.gas_vel[mask_stars][1]*0.5e14, subhalo.gas_vel[mask_stars][2]*0.5e14, linewidth=0.5, color='turquoise')
                
                # Plot original spin vectors
                ax.quiver(0, 0, 0, subhalo.gas_spin[0]/100, subhalo.gas_spin[1]/100, subhalo.gas_spin[2]/100, color='blue', linewidth=1)
                
            if stars_rot == True:
                # Plot rotated stars
                ax.scatter(1000*subhalo.stars_coords[mask_stars][0], 1000*subhalo.stars_coords[mask_stars][1], 1000*subhalo.stars_coords[mask_stars][2], s=0.1, alpha=0.8, c='orange')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(1000*subhalo.stars_coords[mask_stars][0], 1000*subhalo.stars_coords[mask_stars][1], 1000*subhalo.stars_coords[mask_stars][2], 
                            subhalo.stars_vel[mask_stars][0]*0.5e14, subhalo.stars_vel[mask_stars][1]*0.5e14, subhalo.stars_vel[mask_stars][2]*0.5e14, linewidth=0.5, color='white')
                
                # Plot rotated spin vectors
                ax.quiver(0, 0, 0, subhalo.stars_spin[0]/100, subhalo.stars_spin[1]/100, subhalo.stars_spin[2]/100, color='r', linewidth=1)
                
            if gas_rot == True:
                # Plot rotated gas
                ax.scatter(1000*subhalo.gas_coords[mask_gas][0], 1000*subhalo.gas_coords[mask_gas][1], 1000*subhalo.gas_coords[mask_gas][2], s=0.05, alpha=0.8, c='dodgerblue')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(1000*subhalo.gas_coords[mask_gas][0], 1000*subhalo.gas_coords[mask_gas][1], 1000*subhalo.gas_coords[mask_gas][2], 
                            subhalo.gas_vel[mask_gas][0]*0.5e14, subhalo.gas_vel[mask_gas][1]*0.5e14, subhalo.gas_vel[mask_gas][2]*0.5e14, linewidth=0.5, color='turquoise')
                
                # Plot rotated spin vectors
                ax.quiver(0, 0, 0, subhalo.gas_spin[0]/100, subhalo.gas_spin[1]/100, subhalo.gas_spin[2]/100, color='blue', linewidth=1)
                
        if plot_centre == True:
            # Plot centre
            ax.scatter(0, 0, 0, c='k', s=3)
        
        # Plot axis
        ax.quiver(0, 0, 0, 5, 0, 0, color='r', linewidth=0.5)
        ax.quiver(0, 0, 0, 0, 5, 0, color='g', linewidth=0.5)
        ax.quiver(0, 0, 0, 0, 0, 5, color='b', linewidth=0.5)
        
        # Plot formatting
        ax.set_facecolor('xkcd:black')
        ax.set_xlim(-boxradius, boxradius)
        ax.set_ylim(-boxradius, boxradius)
        ax.set_zlim(-boxradius, boxradius)
        
        # Viewing angle
        ax.view_init(0, 0)
        
        plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_spinvectors_TEST%s.jpeg" %(str(GroupNum), str(viewing_angle)), dpi=1000)
        plt.close()
        print('')
    
    
    """ Will render a 3D scatter of a galaxy"""
    #galaxy_render()
    
    """ Will plot velocity projection when given any angle (or range)"""
    velocity_projection()

    """ Will plot spin vectors of a rotated galaxy about z axis (anti-clockwise) by any angle (deg)"""
    #plot_spin_vectors(0)
    
    
    # convert plot into function
    # use 2dhist as pa fit
    # plot mis angles of top 20 galaxies
    # compare to previous 
    
    # separate functions
    
    


























    
"""OLD HEXBIN 
    # Plotting 
        # Plotting hexbin of velocity distribution with x pointing toward us
        divnorm = colors.Normalize(vmin=-100, vmax=100)
        
        #print(L.stars_vel_new*u.Mpc.to(u.km))
        
        # Plot y, z, x-axis toward observer
        im = ax1.hexbin([row[1] for row in L.stars_coords_new], 
                        [row[2] for row in L.stars_coords_new], 
                        C=[row[0] for row in L.stars_vel_new*u.Mpc.to(u.km)*-1], 
                        cmap='coolwarm', gridsize=60, norm=divnorm, extent=[-boxradius, boxradius, -boxradius, boxradius], mincnt=1)
        ax2.hexbin([row[1] for row in L.gas_coords_new], 
                    [row[2] for row in L.gas_coords_new], 
                    C=[row[0] for row in L.gas_vel_new*u.Mpc.to(u.km)*-1], 
                    cmap='coolwarm', gridsize=60, norm=divnorm, extent=[-boxradius, boxradius, -boxradius, boxradius], mincnt=1)
    
             
        # General formatting
        ax1.set_xlabel('y-axis [pMpc]')
        ax2.set_xlabel('y-axis [pMpc]')
        ax1.set_ylabel('z-axis [pMpc]')
        ax2.set_ylabel('z-axis [pMpc]')
        
        ax1.set_title('Stars')
        ax2.set_title('Gas')
        
        ## Colorbar
        cax = plt.axes([0.92, 0.11, 0.02, 0.77])
        plt.colorbar(im, cax=cax, label='velocity [km/s]')
        
        plt.savefig("./trial_plots/galaxy_%s/galaxy_PA/galaxy_projection_%s.jpeg" %(str(GroupNum), str(viewing_angle)), dpi=300)
        plt.close()
"""
"""OLD class SUBHALO 
class SubHalo:
    
    def __init__(self, sim, gn, sgn):
        # Allowing these attributes to be called from the object
        self.gn = gn
        self.sgn = sgn
        
        # For a given gn and sgn, returns stellar mass, centre of potential, perculiar velocity, halfmass radius, and catalogue star/gas spin as unit vectors
        myData = self.query(sim)
        
        self.stelmass     = myData['stelmass']
        self.gasmass      = myData['gasmass']
        self.halfmass_rad = myData['rad']
        self.centre       = np.array([myData['x'], myData['y'], myData['z']])
        self.centre_mass  = np.array([myData['x_mass'], myData['y_mass'], myData['z_mass']])
        self.perc_vel     = np.array([myData['vel_x'], myData['vel_y'], myData['vel_z']])
        
        
    def query(self, sim):
        # This uses the eagleSqlTools module to connect to the database with your username and password.
        # If the password is not given, the module will prompt for it.
        con = sql.connect("lms192", password="dhuKAP62")
        
        for sim_name, sim_size in sim:
            #print(sim_name)
    
            # Construct and execute query for each simulation. This query returns properties for a single galaxy
            myQuery = 'SELECT \
                        log10(SH.MassType_Star) as stelmass, \
                        SH.CentreOfPotential_x as x, \
                        SH.CentreOfPotential_y as y, \
                        SH.CentreOfPotential_z as z, \
                        SH.CentreOfMass_x as x_mass, \
                        SH.CentreOfMass_y as y_mass, \
                        SH.CentreOfMass_z as z_mass, \
                        SH.Velocity_x as vel_x, \
                        SH.Velocity_y as vel_y, \
                        SH.Velocity_z as vel_z, \
                        SH.HalfMassRad_Star as rad, \
                        log10(SH.MassType_Gas) as gasmass \
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
    
"""
"""OLD class Spin
    class Spin:

    def __init__(self, gn, sgn, centre, centre_vel, radius_kpc, stelmass, gasmass, viewing_angle, load_region_length=2):  # cMpc/h
                
        # Load information from the header
        self.a, self.h, self.boxsize = read_header() # units of scale factor, h, and L [cMpc/h]
        
        # Allowing these attributes to be called from the object
        self.centre       = centre                              # [pMpc]
        self.perc_vel     = centre_vel * u.km.to(u.Mpc)         # [pMpc/s]
        self.halfmass_rad = radius_kpc/1000                     # [pMpc]
        self.stelmass     = 10**stelmass                        # [log10Msun]
        self.gasmass      = 10**gasmass                         # [log10Msun]
        self.gn           = gn
        self.sgn          = sgn
        
        # Load data.
        stars  = self.read_galaxy(4, gn, sgn, centre, load_region_length)
        gas    = self.read_galaxy(0, gn, sgn, centre, load_region_length)
        stars_coords   = stars['Coordinates'] - self.centre
        gas_coords     = gas['Coordinates'] - self.centre
        stars_vel      = stars['Velocity'] - self.perc_vel
        gas_vel        = gas['Velocity'] - self.perc_vel

        # Compute spin vectors within radius of interest
        spin_stars_unit, spin_stars = self.spin_vector(stars, self.centre, self.perc_vel, self.halfmass_rad, 'stars')    # unit vector
        spin_gas_unit, spin_gas     = self.spin_vector(gas, self.centre, self.perc_vel, self.halfmass_rad, 'gas')        # unit vector
        
        # Compute angle between star spin and gas spin vector
        self.mis_angle = self.misalignment_angle(spin_stars_unit, spin_gas_unit)
        
        # Find rotation matrix about z axis when given a viewing angle (degrees, clockwise)
        matrix = self.rotate_around_axis('z', 360.-viewing_angle, spin_stars_unit)
        
        # Compute new data of rotated particle data
        stars_rotated = self.rotate_galaxy(matrix, stars)
        gas_rotated   = self.rotate_galaxy(matrix, gas)
        self.stars_coords_new = stars_rotated['Coordinates']
        self.gas_coords_new   = gas_rotated['Coordinates']
        self.stars_vel_new    = stars_rotated['Velocity']
        self.gas_vel_new      = gas_rotated['Velocity']
        
        # Find new spin vectors
        self.spin_stars_new, self.spin_stars_abs = self.spin_vector(stars_rotated, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'stars')
        self.spin_gas_new, self.spin_gas_abs     = self.spin_vector(gas_rotated, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'gas')
        
        # Find mass weighted velocity
        self.stars_vel = self.mass_weighted(stars_rotated, self.stelmass)
        self.gas_vel   = self.mass_weighted(gas_rotated, self.gasmass)
        
        print('##################')
        print(self.stars_vel_new)
        print(stars['Mass'])
        print(self.stelmass)
        print(self.stars_vel)
        print('##################')
        
        print(spin_stars)
        print(self.spin_stars_abs)
        print(spin_gas)
        print(self.spin_gas_abs)
        ####################################################
"""
"""OLD pcolormesh

                # Assign x, y, and the weighted z values for gas and stars
                x_stars = [row[1] for row in subhalo.stars_coords*1000]
                y_stars = [row[2] for row in subhalo.stars_coords*1000]
                z_stars = [row[0] for row in subhalo.stars_vel*u.Mpc.to(u.km)*-1.]
                x_gas   = [row[1] for row in subhalo.gas_coords*1000]
                y_gas   = [row[2] for row in subhalo.gas_coords*1000]
                z_gas   = [row[0] for row in subhalo.gas_vel*u.Mpc.to(u.km)*-1.]
                w_stars = subhalo.stars_mass/np.mean(subhalo.stars_mass)
                w_gas   = subhalo.gas_mass/np.mean(subhalo.gas_mass)
        
                # Create histograms to find mean velocity in each bin (dividing weight by mean)
                counts_stars, xbins_stars, ybins_stars = np.histogram2d(y_stars, x_stars, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                counts_gas, xbins_gas, ybins_gas       = np.histogram2d(y_gas, x_gas, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                sums_stars, _, _ = np.histogram2d(y_stars, x_stars, weights=z_stars, bins=(xbins_stars, ybins_stars), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                sums_gas, _, _   = np.histogram2d(y_gas,   x_gas,   weights=z_gas,   bins=(xbins_gas, ybins_gas), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
        
                # Find means 
                with np.errstate(divide='ignore', invalid='ignore'):  # suppress possible divide-by-zero warnings
                    im1 = ax1.pcolormesh(xbins_stars, ybins_stars, sums_stars / counts_stars, cmap='coolwarm', vmin=-250, vmax=250)
                    #im2 = ax2.pcolormesh(xbins_gas, ybins_gas, sums_gas / counts_gas, cmap='coolwarm')
                
                sums_stars, _, _ = np.histogram2d(y_stars, x_stars, weights=z_stars*w_stars, bins=(xbins_stars, ybins_stars), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                with np.errstate(divide='ignore', invalid='ignore'):  # suppress possible divide-by-zero warnings
                    im2 = ax2.pcolormesh(xbins_stars, ybins_stars, sums_stars / counts_stars, cmap='coolwarm', vmin=-250, vmax=250)
                    
                sums_stars, _, _ = np.histogram2d(y_stars, x_stars, weights=subhalo.stars_mass, bins=(xbins_stars, ybins_stars), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                with np.errstate(divide='ignore', invalid='ignore'):  # suppress /0
                    # Plot total mass and mean mass distributions at orientation
                    im3 = ax3.pcolormesh(xbins_stars, ybins_stars, sums_stars, cmap='coolwarm')
                    im4 = ax4.pcolormesh(xbins_stars, ybins_stars, sums_stars / counts_stars, cmap='coolwarm')     
"""
"""OLD voronoi
                def voronoi_tessalate(coords, velocity, mass):
                    # Assign xy, y, and z values for histogram2d
                    x   = [row[1] for row in coords*1000]
                    y   = [row[2] for row in coords*1000]
                    vel = [row[0] for row in velocity*u.Mpc.to(u.km)*-1.]
                    
                    # Histogram to find counts in each bin
                    counts, xbins, ybins = np.histogram2d(y, x, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                    
                    # Histogram to find total mass-weighted velocity in those bins, and account for number of values in each bin
                    vel, _, _ = np.histogram2d(y, x, weights=vel*mass/np.mean(mass), bins=(xbins, ybins), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                    vel = np.divide(vel, counts, out=np.zeros_like(vel), where=counts!=0)
                    
                    # Convert histogram2d data into coords + value
                    i, j = np.nonzero(vel)                                                    # find indicies of non-0 bins
                    a = np.array((xbins[:-1] + 0.5*(xbins[1] - xbins[0]))[i])     # convert bin to centred coords (x, y)
                    b = np.array((ybins[:-1] + 0.5*(ybins[1] - ybins[0]))[j])
                    
                    # Stack points, and add points at infinity (to color Voronoi outer edges)
                    points_vel = vel[np.nonzero(vel)]
                    points = np.column_stack((b, a))
                    points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis=0)
                    
                    # Create tessalation
                    vor = Voronoi(points)
                    
                    # Find min/max values for normalization
                    minmax = abs(max(points_vel, key=abs))                
                    
                    return vor, points_vel, minmax
                
                
                # Initiate points and velocities for voronoi
                vor_stars, vor_stars_vel, minmax_stars = voronoi_tessalate(subhalo.stars_coords, subhalo.stars_vel, subhalo.stars_mass)
                #vor_gas, vor_gas_vel, minmax_gas     = voronoi_tessalate(subhalo.gas_coords, subhalo.gas_vel, subhalo.gas_mass)
                
                minmax_stars = 150
                
                # Create tessalation
                norm_stars   = colors.Normalize(vmin=-minmax_stars, vmax=minmax_stars, clip=True)
                mapper_stars = cm.ScalarMappable(norm=norm_stars, cmap=cm.coolwarm)
                #norm_gas     = mpl.colors.Normalize(vmin=-minmax_gas, vmax=minmax_gas, clip=True)
                #mapper_gas   = cm.ScalarMappable(norm=norm_gas, cmap=cm.coolwarm)
                
                # Graph initialising and base formatting
                graphformat(8, 11, 11, 11, 11, 5, 4)
                #fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[10, 4.2])
        
                # plot Voronoi diagra, and fill finite regions with color mapped from velocity
                voronoi_plot_2d(vor_stars, show_points=False, show_vertices=False, line_width=0, s=1)
                for r in range(len(vor_stars.point_region)):
                    region = vor_stars.regions[vor_stars.point_region[r]]
                    if not -1 in region:
                        polygon = [vor_stars.vertices[i] for i in region]
                        plt.fill(*zip(*polygon), color=mapper_stars.to_rgba(vor_stars_vel[r]))
                        
                # Graph formatting
                plt.xlim(-boxradius, boxradius)
                plt.ylim(-boxradius, boxradius)
                plt.xlabel('y-axis [pkpc]')
                plt.ylabel('x-axis [pkpc]')
                plt.title('Stars')
                
                # Overplot original points for comparisson
                #plt.scatter([row[1] for row in subhalo.starscoords*1000], [row[2] for row in subhalo.stars_coords*1000], c=[row[0] for row in subhalo.stars_vel*u.Mpc.to(u.km)*-1.])

                # Colorbar
                plt.colorbar(mapper_stars, label='mass-weighted mean velocity [km/s]')
        
                plt.tight_layout()
        
                plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_projection_voronoi/voronoi_%s.jpeg' %(str(GroupNum), str(viewing_angle)), dpi=300)
                plt.close()    
"""