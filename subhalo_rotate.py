import h5py
import numpy as np
import astropy.units as u
import random
import math
from astropy.constants import G
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from read_dataset import read_dataset
from read_header import read_header
from read_dataset_dm_mass import read_dataset_dm_mass
from pyread_eagle import EagleSnapshot
import eagleSqlTools as sql
from graphformat import graphformat


# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'


class Spin:

    def __init__(self, gn, sgn, centre, centre_vel, radius_kpc, load_region_length=2):  # cMpc/h
                
        # Load information from the header
        self.a, self.h, self.boxsize = read_header() # units of scale factor, h, and L [cMpc/h]
        
        # Allowing these attributes to be called from the object
        self.centre       = centre                              # [pMpc]
        self.perc_vel     = centre_vel * u.km.to(u.Mpc)         # [pMpc/s]
        self.halfmass_rad = radius_kpc/1000                     # [pMpc]
        self.gn           = gn
        self.sgn          = sgn
        
        # Load data.
        stars  = self.read_galaxy(4, gn, sgn, centre, load_region_length)
        gas    = self.read_galaxy(0, gn, sgn, centre, load_region_length)
        self.stars_coords   = stars['Coordinates'] - self.centre
        self.gas_coords     = gas['Coordinates'] - self.centre

        # Compute spin vectors within radius of interest
        self.spin_stars       = self.spin_vector(stars, self.centre, self.perc_vel, self.halfmass_rad, 'stars')    # unit vector
        self.spin_gas         = self.spin_vector(gas, self.centre, self.perc_vel, self.halfmass_rad, 'gas')        # unit vector
        self.spin_stars_large = self.spin_vector(stars, self.centre, self.perc_vel, self.halfmass_rad*20, 'stars')    # unit vector
        self.spin_gas_large   = self.spin_vector(gas, self.centre, self.perc_vel, self.halfmass_rad*20, 'gas')        # unit vector
        
        # Compute angle between star spin and gas spin vector
        self.mis_angle = self.misalignment_angle(self.spin_stars, self.spin_gas)
            
        
        # Find z_angle when given an axis as reference (z) and an attribute (stars)
        self.z_angle, matrix = self.find_angle('z', self.spin_stars)
        
        # Compute new data of rotated particle data
        stars_rotated = self.rotate_galaxy(matrix, stars)
        gas_rotated   = self.rotate_galaxy(matrix, gas)
        self.stars_coords_new = stars_rotated['Coordinates']
        self.gas_coords_new   = gas_rotated['Coordinates']
        
        # Find new spin vectors
        self.spin_stars_new = self.spin_vector(stars_rotated, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'stars')
        self.spin_gas_new   = self.spin_vector(gas_rotated, np.array([0, 0, 0]), np.array([0, 0, 0]), self.halfmass_rad, 'gas')
     
        
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
        angle = np.arccos(np.clip(np.dot(angle1, angle2), -1.0, 1.0)) * 180/np.pi
        
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
        axis = axis/math.sqrt(np.dot(axis, axis))
        
        # take the cosine of out rotation degree in radians
        a = math.cos(theta/2.0)
        
        # get the rest rotation matrix components
        b, c, d = -axis*math.sin(theta/2.0)
        
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
            
        return rotation


    def find_angle(self, axis, attribute):
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
            x -= x.dot(attribute) * self.spin_stars
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
        
        new_data['GroupNumber']     = data['GroupNumber']
        new_data['SubGroupNumber']  = data['SubGroupNumber']
        new_data['Mass']            = data['Mass']
        new_data['Coordinates']     = self.rotate_coords(matrix, data['Coordinates'] - self.centre)
        new_data['Velocity']        = self.rotate_coords(matrix, data['Velocity'] - self.perc_vel)
        
        return new_data



class SubHalo:
    
    def __init__(self, sim, gn, sgn):
        # Allowing these attributes to be called from the object
        self.gn = gn
        self.sgn = sgn
        
        # For a given gn and sgn, returns stellar mass, centre of potential, perculiar velocity, halfmass radius, and catalogue star/gas spin as unit vectors
        myData = self.query(sim)
        
        self.stelmass     = myData['stelmass']
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
                        SH.HalfMassRad_Star as rad \
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
        
        
        
if __name__ == '__main__':
    
    """Function that will auto-generate a 360 render of the galaxy with spin vectors added"""
    def galaxy_render(GroupNumList=np.array([1]),
                        SubGroupNum=0, 
                        particles=10000,
                        minangle=0,
                        maxangle=360, 
                        stepangle=30, 
                        boxradius=0.1, 
                        stars=True, 
                        gas=False, 
                        stars_rot=False, 
                        gas_rot=False,
                        stars_total = False, 
                        gas_total = False, 
                        centre_of_pot=True, 
                        centre_of_mass=True,
                        axis=True):
        
        # list of simulations
        mySims = np.array([('RefL0012N0188', 12)])   
        
        for GroupNum in GroupNumList: #np.arange(1, 16, 1):  
            # Initialise subhalo call
            subhalo = SubHalo(mySims, GroupNum, SubGroupNum)
            
            # For radius find the spin vectors
            L = Spin(GroupNum, SubGroupNum, subhalo.centre, subhalo.perc_vel, subhalo.halfmass_rad)

            # Assign sql data to variables to extract spin           
            print('CENTRE [pKpc]:', subhalo.centre)                         #pMpc
            print('PERCULIAR VELOCITY [pkm/s]:', subhalo.perc_vel)          #pkm/s
            print('STELLAR MASS [Msun]: %.3f' %subhalo.stelmass)            #Msun
            print('HALFMASS RAD [pkpc]: %.3f' %subhalo.halfmass_rad)        #pkpc
            print('MISALIGNMENT ANGLE [deg]: %.1f'%L.mis_angle)

            # Graph initialising and base formatting
            graphformat(8, 11, 11, 11, 11, 5, 5)
            fig = plt.figure() 
            ax = Axes3D(fig)
    
            # Plot n random points 3d scatter so we don't need to use the entire data
            for n in np.arange(0, particles, 1):
                # Mask to random number of particles given by galaxy_render
                mask_stars = random.randint(0, len(L.stars_coords[:])-1)
                mask_gas = random.randint(0, len(L.gas_coords[:])-1)
                
                if stars == True:
                    # Plot original stars
                    ax.scatter(L.stars_coords[mask_stars][0], L.stars_coords[mask_stars][1], L.stars_coords[mask_stars][2], s=0.02, alpha=0.9, c='khaki')
                if gas == True:
                    # Plot original gas
                    ax.scatter(L.gas_coords[mask_gas][0], L.gas_coords[mask_gas][1], L.gas_coords[mask_gas][2], s=0.02, alpha=0.9, c='blue')
                if stars_rot == True:
                    # Plot rotated stars
                    ax.scatter(L.stars_coords_new[mask_stars][0], L.stars_coords_new[mask_stars][1], L.stars_coords_new[mask_stars][2], s=0.02, alpha=0.9, c='khaki')
                if gas_rot == True:
                    # Plot rotated gas
                    ax.scatter(L.gas_coords_new[mask_gas][0], L.gas_coords_new[mask_gas][1], L.gas_coords_new[mask_gas][2], s=0.02, alpha=0.9, c='blue')
    
            if stars == True:
                # Plot original stars spin vector
                ax.quiver(0, 0, 0, L.spin_stars[0]*boxradius*0.4, L.spin_stars[1]*boxradius*0.4, L.spin_stars[2]*boxradius*0.4, color='red', linewidth=1)
            if gas == True:
                # Plot original stars spin vector
                ax.quiver(0, 0, 0, L.spin_gas[0]*boxradius*0.4, L.spin_gas[1]*boxradius*0.4, L.spin_gas[2]*boxradius*0.4, color='navy', linewidth=1)
            if stars_rot == True:
                # Plot rotated stars spin vector
                ax.quiver(0, 0, 0, L.spin_stars_new[0]*boxradius*0.4, L.spin_stars_new[1]*boxradius*0.4, L.spin_stars_new[2]*boxradius*0.4, color='red', linewidth=1)
            if gas_rot == True:
                # Plot rotated stars spin vector
                ax.quiver(0, 0, 0, L.spin_gas_new[0]*boxradius*0.4, L.spin_gas_new[1]*boxradius*0.4, L.spin_gas_new[2]*boxradius*0.4, color='navy', linewidth=1)
                
            if centre_of_pot == True:
                # Plot centre_of_potential
                ax.scatter(L.centre[0], L.centre[1], L.centre[2], c='k', s=3, zorder=10)
            if centre_of_mass == True:
                # Plot centre_of_mass
                ax.scatter(subhalo.centre_mass[0], subhalo.centre_mass[1], subhalo.centre_mass[2], c='b', s=3, zorder=10)
            
            if axis == True:
                # Plot axis
                ax.quiver(0, 0, 0, 0.01, 0, 0, color='r', linewidth=0.5)
                ax.quiver(0, 0, 0, 0, 0.01, 0, color='g', linewidth=0.5)
                ax.quiver(0, 0, 0, 0, 0, 0.01, color='b', linewidth=0.5)
                
            # Plot formatting
            ax.set_facecolor('xkcd:black')
            ax.set_xlim(-boxradius, boxradius)
            ax.set_ylim(-boxradius, boxradius)
            ax.set_zlim(-boxradius, boxradius)   
            
            if stars_total == True:
                # Plot large-scale (total subhalo) star spin vector
                ax.quiver(0, 0, 0, L.spin_stars_large[0]*boxradius*0.4, L.spin_stars_large[1]*boxradius*0.4, L.spin_stars_large[2]*boxradius*0.4, color='red', linewidth=0.3)
            if gas_total == True:
                # Plot large-scale (total subhalo) gas spin vector
                ax.quiver(0, 0, 0, L.spin_gas_large[0]*boxradius*0.4, L.spin_gas_large[1]*boxradius*0.4, L.spin_gas_large[2]*boxradius*0.4, color='navy', linewidth=0.3)
                
            for ii in np.arange(minangle, maxangle+1, stepangle):
                ax.view_init(0, ii)
                plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%i/galaxy_stars/galaxy_%i.jpeg"%(GroupNum,ii), dpi=300)
                
            plt.close()
            #plt.show()


    # Specify groupnumber, subgroupnumber of galaxy
    galaxy_render()   


"""OLD PRE-DEF CODE 
    SubGroupNum = 0
    #for GroupNum in np.array([galaxy]): #np.arange(1, 16, 1):  

        #subhalo = SubHalo(mySims, GroupNum, SubGroupNum)

        # Assign sql data to variables to extract spin
        #centre = subhalo.centre                          #pMpc
        #centre_vel = subhalo.perc_vel                    #pkm/s
        #radius_of_interest = subhalo.halfmass_rad        #pkpc
        #print('CENTRE:', centre)
        #print('PERCULIAR VELOCITY:', centre_vel)
        #print('STELLAR MASS: %.3f'%subhalo.stelmass)
        #print('HALFMASS RAD: %.3f'%radius_of_interest)
    

        
        # Plot n random points 3d scatter so we don't need to use the entire data
        #for n in np.arange(0, 2000, 1):
            #mask_stars = random.randint(0, len(L.stars_coords[:])-1)
            #mask_gas = random.randint(0, len(L.gas_coords[:])-1)
            
            # Plot original gas and stars
            #ax.scatter(L.stars_coords[mask_stars][0], L.stars_coords[mask_stars][1], L.stars_coords[mask_stars][2], s=0.05, alpha=0.8, c='white')
            #ax.scatter(L.gas_coords[mask_gas][0], L.gas_coords[mask_gas][1], L.gas_coords[mask_gas][2], s=0.05, alpha=0.8, c='lightblue')
            
            # Plot rotated gas and stars
            #ax.scatter(L.stars_coords_new[mask_stars][0], L.stars_coords_new[mask_stars][1], L.stars_coords_new[mask_stars][2], s=0.1, alpha=0.8, c='white')
            #ax.scatter(L.gas_coords_new[mask_gas][0], L.gas_coords_new[mask_gas][1], L.gas_coords_new[mask_gas][2], s=0.1, alpha=0.8, c='blue')
            
        # Plot original spin vectors
        #ax.quiver(0, 0, 0, L.spin_stars[0]/50, L.spin_stars[1]/50, L.spin_stars[2]/50, color='orange', linewidth=1)
        #ax.quiver(0, 0, 0, L.spin_gas[0]/50, L.spin_gas[1]/50, L.spin_gas[2]/50, color='navy', linewidth=1)
        
        # Plot new spin vectors
        #ax.quiver(0, 0, 0, L.spin_stars_new[0]/50, L.spin_stars_new[1]/50, L.spin_stars_new[2]/50, color='red', linewidth=1)
        #ax.quiver(0, 0, 0, L.spin_gas_new[0]/50, L.spin_gas_new[1]/50, L.spin_gas_new[2]/50, color='blue', linewidth=1)

        # Plot centre_of_potential and centre_of_mass
        #ax.scatter(L.centre[0], L.centre[1], L.centre[2], c='k', s=3)
        #ax.scatter(subhalo.centre_mass[0], subhalo.centre_mass[1], subhalo.centre_mass[2], c='b', s=3)
        
        # Plot axis
        #ax.quiver(0, 0, 0, 0.01, 0, 0, color='r', linewidth=0.5)
        #ax.quiver(0, 0, 0, 0, 0.01, 0, color='g', linewidth=0.5)
        #ax.quiver(0, 0, 0, 0, 0, 0.01, color='b', linewidth=0.5)
        
        # Plot formatting
        #ax.set_facecolor('xkcd:black')
        #ax.set_xlim(-0.04, 0.04)
        #ax.set_ylim(-0.04, 0.04)
        #ax.set_zlim(-0.04, 0.04)
        
        #for ii in np.arange(0, 20, 10):
            #ax.view_init(0, ii)
            #plt.savefig("./trial_plots/galaxy_%i/galaxy_render/galaxy_render_%i.jpeg"%(galaxy,ii))
        #ax.view_init(20, 230)
        #plt.savefig("./trial_plots/galaxy_rotate_7.jpeg", dpi=300)
        
        
        #plt.show()
"""
""" OLD SPIN DEF 
        # Compute angle between star spin and z axis
        #self.z_angle = self.misalignment_angle(np.array([0., 0., 1.]), self.spin_stars)
        
        # Find axis of rotation of star spin vector
        #x  = np.array([0., 0., 1.])
        #x -= x.dot(self.spin_stars) * self.spin_stars
        #x /= np.linalg.norm(x)
        #axis_of_rotation = np.cross(self.spin_stars, x)
        #matrix = self.rotation_matrix(axis_of_rotation, self.z_angle)
        
        # Compute new coordinates from rotation
        #self.stars_coords_new = self.rotate_coords(matrix, self.stars_coords)
        #self.gas_coords_new   = self.rotate_coords(matrix, self.gas_coords)
        
        # Compute new spin vectors from rotation (np.array done to be compatible with array or array for coords)
        #self.spin_stars_new = self.rotate_coords(matrix, np.array([self.spin_stars]))[0]
        #self.spin_gas_new   = self.rotate_coords(matrix, np.array([self.spin_gas]))[0]  
"""