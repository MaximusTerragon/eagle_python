import h5py
import numpy as np
import astropy.units as u
import random
import math
from pafit.fit_kinematic_pa import fit_kinematic_pa
from astropy.constants import G
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import matplotlib.colors as colors
from read_dataset import read_dataset
from read_header import read_header
from read_dataset_dm_mass import read_dataset_dm_mass
from pyread_eagle import EagleSnapshot
import eagleSqlTools as sql
from graphformat import graphformat

# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'


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
        
        print("##################")
        print(self.stars_vel_new)
        print(stars['Mass'])
        print(self.stelmass)
        print(self.stars_vel)
        print("##################")
        
        print(spin_stars)
        print(self.spin_stars_abs)
        print(spin_gas)
        print(self.spin_gas_abs)
        
        
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
        
        return spin_unit, spin


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
        
        
    def rotate_to_axis(self, axis, attribute):
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
        
    
    def mass_weighted(self, arr, mass):
        # Function to mass-weight a given input
        velocity = arr['Velocity'] * arr['Mass'][:, None] / mass
        
        return velocity
        
    #def kappa... need to use rotate
        

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
        
        
        
if __name__ == '__main__':
    # list of simulations
    mySims = np.array([('RefL0012N0188', 12)])   
    
    def velocity_projection(GroupNumList = np.array([16]),
                            SubGroupNum = 0,
                            minangle = 0,
                            maxangle = 360,
                            stepangle = 30,
                            boxradius = 0.1,
                            pixel = 30):
        
        for GroupNum in GroupNumList:
            # For radius as multiple of stellar halfmass radius, find the spin vector 
            subhalo = SubHalo(mySims, GroupNum, SubGroupNum)
            mis = Spin(GroupNum, SubGroupNum, subhalo.centre, subhalo.perc_vel, subhalo.halfmass_rad, subhalo.stelmass, subhalo.gasmass, 0)
            
            # Assign sql data to variables to extract spin     
            print('CENTRE:', subhalo.centre)                    #pMpc
            print('PERCULIAR VELOCITY:', subhalo.perc_vel)      #pkm/s
            print('STELLAR MASS: %.3f' %subhalo.stelmass)       #pMsun
            print('HALFMASS RAD: %.3f' %subhalo.halfmass_rad)   #pkpc
            print('MISALIGNMENT ANGLE: %.1f\n'%mis.mis_angle)
            
            boxradius = 2*subhalo.halfmass_rad/1000
            
            for viewing_angle in np.arange(minangle, maxangle+1, stepangle):
                L = Spin(GroupNum, SubGroupNum, subhalo.centre, subhalo.perc_vel, subhalo.halfmass_rad, subhalo.stelmass, subhalo.gasmass, viewing_angle)
                
                # Graph initialising and base formatting
                graphformat(8, 11, 11, 11, 11, 5, 5)
                fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[10, 4.2])
                
                # Custom colormap
        
                # Assign x, y, and the weighted z values for gas and stars
                x_stars = [row[1] for row in L.stars_coords_new]
                y_stars = [row[2] for row in L.stars_coords_new]
                #z_stars = [row[0] for row in L.stars_vel_new*u.Mpc.to(u.km)*-1.]
                z_stars = [row[0] for row in L.stars_vel*u.Mpc.to(u.km)*-1.]
                x_gas   = [row[1] for row in L.gas_coords_new]
                y_gas   = [row[2] for row in L.gas_coords_new]
                #z_gas   = [row[0] for row in L.gas_vel_new*u.Mpc.to(u.km)*-1.]
                z_gas   = [row[0] for row in L.gas_vel*u.Mpc.to(u.km)*-1.]
        
                # Create histograms to find mean velocity in each bin (dividing weight by mean)
                counts_stars, xbins_stars, ybins_stars = np.histogram2d(y_stars, x_stars, bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                counts_gas, xbins_gas, ybins_gas       = np.histogram2d(y_gas,   x_gas,   bins=(pixel, pixel), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                sums_stars, _, _ = np.histogram2d(y_stars, x_stars, weights=z_stars, bins=(xbins_stars, ybins_stars), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
                sums_gas, _, _   = np.histogram2d(y_gas,   x_gas,   weights=z_gas,   bins=(xbins_gas, ybins_gas), range=[[-boxradius, boxradius], [-boxradius, boxradius]])
        
                #with np.errstate(divide='ignore', invalid='ignore'):
                    #print(np.array(np.tile(xbins_stars[:-1], (len(xbins_stars[:-1]), 1))).shape)
                    #print(np.array(ybins_stars).size)
                    #print(np.array(sums_stars/counts_stars).size)
                    #print(np.array(np.tile(xbins_stars[:-1], (len(xbins_stars[:-1]), 1))))
                    #print(np.array(np.tile(ybins_stars[:-1], (len(ybins_stars[:-1]), 1))).T)
                
                #print(sums_gas/counts_gas)
                
                
                
                
                """# Run fit_kinematic_pa
                with np.errstate(divide='ignore', invalid='ignore'):
                    angBest, angErr, vSyst = fit_kinematic_pa(np.array(np.tile(xbins_stars[:-1], (len(xbins_stars[:-1]), 1))), np.array(np.tile(ybins_stars[:-1], (len(ybins_stars[:-1]), 1))).T, sums_stars/counts_stars)
                    print('fit_kinematic_pa best angle = %.1f' %angBest)
                    print(angErr)
                
                    a, b, c, = fit_kinematic_pa(np.array(np.tile(ybins_gas[:-1], (len(ybins_gas[:-1]), 1))), np.array(np.tile(ybins_gas[:-1], (len(ybins_gas[:-1]), 1))).T, sums_gas/counts_gas)
                    print('fit_kinematic_pa best angle = %.1f' %angBest)
                    print(angErr)
                    
                print('Fit delta PA = %.1f' %(angBest-a))"""
                
                
                
                # Find means 
                with np.errstate(divide='ignore', invalid='ignore'):  # suppress possible divide-by-zero warnings
                    im1 = ax1.pcolormesh(xbins_stars, ybins_stars, sums_stars / counts_stars, cmap='coolwarm')
                    im2 = ax2.pcolormesh(xbins_gas, ybins_gas, sums_gas / counts_gas, cmap='coolwarm')
        
                # General formatting
                for ax in [ax1, ax2]:
                    ax.set_xlim(-boxradius, boxradius)
                    ax.set_ylim(-boxradius, boxradius)
                    ax.set_xlabel('y-axis [pMpc]')
                    ax.set_ylabel('z-axis [pMpc]')
                ax1.set_title('Stars')
                ax2.set_title('Gas')

                # Colorbar
                plt.colorbar(im1, ax=ax1, label='mean velocity [km/s]')
                plt.colorbar(im2, ax=ax2, label='mean velocity [km/s]')
        
                plt.tight_layout()
        
                plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_PA/galaxy_NEW%s.jpeg' %(str(GroupNum), str(viewing_angle)), dpi=300)
                plt.close()
    
    
    def plot_spin_vectors(viewing_angle,
                          GroupNum = 16,
                          SubGroupNum = 0,
                          stars = False,
                          gas = False,
                          stars_rot = True, 
                          gas_rot = True,
                          plot_centre = True, 
                          boxradius = 0.005,
                          particles = 2000):
        # For radius and viewing angle (degrees) as multiple of stellar halfmass radius, find the spin vector 
        subhalo = SubHalo(mySims, GroupNum, SubGroupNum)
        L = Spin(GroupNum, SubGroupNum, subhalo.centre, subhalo.perc_vel, subhalo.halfmass_rad, viewing_angle)

        # Assign sql data to variables to extract spin     
        print('CENTRE:', subhalo.centre)                    #pMpc
        print('PERCULIAR VELOCITY:', subhalo.perc_vel)      #pkm/s
        print('STELLAR MASS: %.3f' %subhalo.stelmass)       #pMsun
        print('HALFMASS RAD: %.3f' %subhalo.halfmass_rad)   #pkpc
        print('MISALIGNMENT ANGLE: %.1f'%L.mis_angle)
    
        # Graph initialising and base formatting
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig = plt.figure() 
        ax = Axes3D(fig)
        #ax = fig.add_subplot(projection='3d')
        
        # Plot 100 random points 3d scatter so we don't need to use the entire data
        for n in np.arange(0, particles, 1):
            mask_stars = random.randint(0, len(L.stars_coords_new[:])-1)
            mask_gas = random.randint(0, len(L.gas_coords_new[:])-1)
            
            if stars == True:
                # Plot original stars
                ax.scatter(L.stars_coords_new[mask_stars][0], L.stars_coords_new[mask_stars][1], L.stars_coords_new[mask_stars][2], s=0.1, alpha=0.8, c='orange')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(L.stars_coords_new[mask_stars][0], L.stars_coords_new[mask_stars][1], L.stars_coords_new[mask_stars][2], 
                            L.stars_vel_new[mask_stars][0]*0.5e14, L.stars_vel_new[mask_stars][1]*0.5e14, L.stars_vel_new[mask_stars][2]*0.5e14, linewidth=0.5, color='white')
                
                # Plot original spin vectors
                ax.quiver(0, 0, 0, L.spin_stars_new[0]/100, L.spin_stars_new[1]/100, L.spin_stars_new[2]/100, color='r', linewidth=1)
                
            if gas == True:
                # Plot original gas
                ax.scatter(L.gas_coords_new[mask_gas][0], L.gas_coords_new[mask_gas][1], L.gas_coords_new[mask_gas][2], s=0.05, alpha=0.8, c='dodgerblue')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(L.gas_coords_new[mask_stars][0], L.gas_coords_new[mask_stars][1], L.gas_coords_new[mask_stars][2], 
                            L.gas_vel_new[mask_stars][0]*0.5e14, L.gas_vel_new[mask_stars][1]*0.5e14, L.gas_vel_new[mask_stars][2]*0.5e14, linewidth=0.5, color='turquoise')
                
                # Plot original spin vectors
                ax.quiver(0, 0, 0, L.spin_gas_new[0]/100, L.spin_gas_new[1]/100, L.spin_gas_new[2]/100, color='blue', linewidth=1)
                
            if stars_rot == True:
                # Plot rotated stars
                ax.scatter(L.stars_coords_new[mask_stars][0], L.stars_coords_new[mask_stars][1], L.stars_coords_new[mask_stars][2], s=0.1, alpha=0.8, c='orange')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(L.stars_coords_new[mask_stars][0], L.stars_coords_new[mask_stars][1], L.stars_coords_new[mask_stars][2], 
                            L.stars_vel_new[mask_stars][0]*0.5e14, L.stars_vel_new[mask_stars][1]*0.5e14, L.stars_vel_new[mask_stars][2]*0.5e14, linewidth=0.5, color='white')
                
                # Plot rotated spin vectors
                ax.quiver(0, 0, 0, L.spin_stars_new[0]/100, L.spin_stars_new[1]/100, L.spin_stars_new[2]/100, color='r', linewidth=1)
                
            if gas_rot == True:
                # Plot rotated gas
                ax.scatter(L.gas_coords_new[mask_gas][0], L.gas_coords_new[mask_gas][1], L.gas_coords_new[mask_gas][2], s=0.05, alpha=0.8, c='dodgerblue')
                
                # Plot individual spin vectors (multiplied by a factor to make them visible)
                ax.quiver(L.gas_coords_new[mask_gas][0], L.gas_coords_new[mask_gas][1], L.gas_coords_new[mask_gas][2], 
                            L.gas_vel_new[mask_gas][0]*0.5e14, L.gas_vel_new[mask_gas][1]*0.5e14, L.gas_vel_new[mask_gas][2]*0.5e14, linewidth=0.5, color='turquoise')
                
                # Plot rotated spin vectors
                ax.quiver(0, 0, 0, L.spin_gas_new[0]/100, L.spin_gas_new[1]/100, L.spin_gas_new[2]/100, color='blue', linewidth=1)
                
        if plot_centre == True:
            # Plot centre
            ax.scatter(L.centre[0], L.centre[1], L.centre[2], c='k', s=3)
        
        # Plot axis
        ax.quiver(0, 0, 0, 0.005, 0, 0, color='r', linewidth=0.5)
        ax.quiver(0, 0, 0, 0, 0.005, 0, color='g', linewidth=0.5)
        ax.quiver(0, 0, 0, 0, 0, 0.005, color='b', linewidth=0.5)
        
        # Plot formatting
        ax.set_facecolor('xkcd:black')
        ax.set_xlim(-boxradius, boxradius)
        ax.set_ylim(-boxradius, boxradius)
        ax.set_zlim(-boxradius, boxradius)
        
        # Viewing angle
        ax.view_init(0, 0)
        
        plt.savefig("/Users/c22048063/Documents/EAGLE/trial_plots/galaxy_%s/galaxy_spinvectors_TEST%s.jpeg" %(str(GroupNum), str(viewing_angle)), dpi=1000)
        plt.close()
    
    
    """ Will plot velocity projection when given any angle (or range)"""
    velocity_projection()

    
    """ Will plot spin vectors of a rotated galaxy about z axis (anti-clockwise) by any angle (deg)"""
    #plot_spin_vectors(0)
    
    

    
    
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

        

    
    
