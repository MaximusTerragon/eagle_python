import h5py
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from read_dataset import read_dataset
from read_header import read_header
from read_dataset_dm_mass import read_dataset_dm_mass
from pyread_eagle import EagleSnapshot
import eagleSqlTools as sql
from graphformat import graphformat

# Directories of data hdf5 file(s)
dataDir = './data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'


class Spin:

    def __init__(self, gn, sgn, centre, centre_vel, radius_kpc):
                
        # Load information from the header
        self.a, self.h, self.boxsize = read_header() # units of scale factor, h, and L [cMpc/h]
        
        # Allowing these attributes to be called from the object
        self.centre       = centre                              # [pMpc]
        self.perc_vel     = centre_vel * u.km.to(u.Mpc)         # [pMpc/s]
        self.halfmass_rad = radius_kpc/1000                     # [pMpc]
        self.gn           = gn
        self.sgn          = sgn
        
        # Load data.
        stars  = self.read_galaxy(4, gn, sgn, centre)
        gas    = self.read_galaxy(0, gn, sgn, centre)

        # Compute spin vectors within radius of interest
        self.spin_stars = self.spin_vector(stars, self.halfmass_rad, 'stars')      # unit vector
        self.spin_gas   = self.spin_vector(gas, self.halfmass_rad, 'gas')            # unit vector
        
        # Compute angle between star spin and gas spin vector
        self.mis_angle = self.misalignment_angle()

        
    def read_galaxy(self, itype, gn, sgn, centre):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates, velocty, and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity."""

        # Where we store all the data
        data = {}
        
        # Load data, then mask to selected GroupNumber and SubGroupNumber.
        gns  = read_dataset(itype, 'GroupNumber')
        sgns = read_dataset(itype, 'SubGroupNumber')
        mask = np.logical_and(gns == gn, sgns == sgn)
        
        # Load data, then mask to selected GroupNumber and SubGroupNumber. Automatically converts to pcm from read_dataset, converted to pMpc
        data['Mass'] = read_dataset(itype, 'Mass')[mask] * u.g.to(u.Msun)                   # [Msun]
        data['Coordinates'] = read_dataset(itype, 'Coordinates')[mask] * u.cm.to(u.Mpc)     # [pMpc]
        data['Velocity'] = read_dataset(itype, 'Velocity')[mask] * u.cm.to(u.Mpc)           # [pMpc/s]
            
        # Periodic wrap coordinates around centre (in proper units). boxsize converted from cMpc
        boxsize = self.boxsize/self.h       # [pMpc]
        data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
        
        return data

    
    def spin_vector(self, arr, radius, desc):
        # Compute distance to centre and mask all within stelhalfrad
        r  = np.linalg.norm(arr['Coordinates'] - self.centre, axis=1)
        mask = np.where(r <= radius)
        #print("Total %s particles in subhalo: %i"%(desc, len(r)))
        #r = r[mask]
        #print("Total %s particles in %.5f kpc: %i\n"%(desc, radius*1000, len(r)))
        
        
        """# Sanity check plot
        v = np.linalg.norm((arr['Velocity'] - self.perc_vel)*u.Mpc.to(u.km), axis=1)
        plt.scatter(r, v, s=0.5)
        plt.scatter(r[mask], v[mask], s=0.5)
        plt.show()"""


        # Finding spin angular momentum vector of each individual particle of gas and stars, where [:, None] is done to allow multiplaction of N3*N1 array. Equation D.25
        L  = np.cross((arr['Coordinates'] - self.centre)[mask] * arr['Mass'][:, None][mask], (arr['Velocity'] - self.perc_vel)[mask])
        
        # Summing for total angular momentum and dividing by mass to get the spin vectors
        spin = np.sum(L, axis=0)/np.sum(arr['Mass'])
        #print(spin)
        
        # Expressing as unit vector
        spin_unit = spin / (spin[0]**2 + spin[1]**2 + spin[2]**2)**0.5
        
        return spin_unit

    def misalignment_angle(self):
        # Find the misalignment angle
        angle = np.arccos(np.clip(np.dot(self.spin_stars, self.spin_gas), -1.0, 1.0)) * 180/3.1416
        
        return angle
        


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
        self.perc_vel     = np.array([myData['vel_x'], myData['vel_y'], myData['vel_z']])
        self.spin_stars   = np.array([myData['stars_spin_x'], myData['stars_spin_y'], myData['stars_spin_z']])
        self.spin_gas     = np.array([myData['gas_spin_x'], myData['gas_spin_y'], myData['gas_spin_z']])
        
        
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
                        SH.Velocity_x as vel_x, \
                        SH.Velocity_y as vel_y, \
                        SH.Velocity_z as vel_z, \
                        SH.HalfMassRad_Star as rad, \
                    	(SH.Stars_Spin_x/power(power(SH.Stars_Spin_x,2) + power(SH.Stars_Spin_y,2) + power(SH.Stars_Spin_z,2),0.5)) as stars_spin_x, \
                        (SH.Stars_Spin_y/power(power(SH.Stars_Spin_x,2) + power(SH.Stars_Spin_y,2) + power(SH.Stars_Spin_z,2),0.5)) as stars_spin_y, \
                        (SH.Stars_Spin_z/power(power(SH.Stars_Spin_x,2) + power(SH.Stars_Spin_y,2) + power(SH.Stars_Spin_z,2),0.5)) as stars_spin_z, \
                        (SH.GasSpin_x/power(power(SH.GasSpin_x,2) + power(SH.GasSpin_y,2) + power(SH.GasSpin_z,2),0.5)) as gas_spin_x, \
                        (SH.GasSpin_y/power(power(SH.GasSpin_x,2) + power(SH.GasSpin_y,2) + power(SH.GasSpin_z,2),0.5)) as gas_spin_y, \
                        (SH.GasSpin_z/power(power(SH.GasSpin_x,2) + power(SH.GasSpin_y,2) + power(SH.GasSpin_z,2),0.5)) as gas_spin_z \
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
        


class Sample:
    
    def __init__(self, sim, mstarLimit, satellite):
        # Allowing these attributes to be called from the object
        self.mstar_limit = mstarLimit
        self.sim = sim
        
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
        			         SH.SnapNum = 28 \
                             and SH.MassType_Star >= %f \
                           ORDER BY \
        			         SH.MassType_Star desc'%(sim_name, self.mstar_limit)
            
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



if __name__ == '__main__':
    # list of simulations
    mySims = np.array([('RefL0012N0188', 12)])
    
    galaxy_mass_limit = 1e9
    
    # creates a list of applicable gn (and sgn) to sample. To include satellite galaxies, use 'yes'
    sample = Sample(mySims, galaxy_mass_limit, 'no')
    print('Number of subhalos in sample with M > %e: %i\n'%(galaxy_mass_limit, len(sample.GroupNum)))   
    #print(sample.GroupNum)
    #print(sample.SubGroupNum)     
    
    misalignment_angle = []
    
    SubGroupNum = 0
    for GroupNum in sample.GroupNum: #np.array([1]): 

        subhalo = SubHalo(mySims, GroupNum, SubGroupNum)

        # Assign sql data to variables to extract spin
        centre = subhalo.centre                          #pMpc
        centre_vel = subhalo.perc_vel                    #pkm/s
        radius_of_interest = subhalo.halfmass_rad        #pkpc
        print('CENTRE:', centre)
        print('PERCULIAR VELOCITY:', centre_vel)
        print('HALFMASS RAD: %.5f'%radius_of_interest)
    
        # Figure of individual galaxy rotation vectors projected
        """fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])"""
    
        i = 1
        # For increasing radius as multiple of stellar halfmass radius, find the spin vector (and therefore how it varies)
        for rad in radius_of_interest*np.array([1]):
            L = Spin(GroupNum, SubGroupNum, centre, centre_vel, rad)
            #print(L.spin_stars)
            #print(L.spin_gas)
            print('MISALIGNMENT ANGLE: %.1f\n'%L.mis_angle)
        
            # Plot calculated unit vector of gas/star spin
            """ax.quiver(0, 0, 0, L.spin_stars[0], L.spin_stars[1], L.spin_stars[2], color='orange', linewidth=3/i)
            ax.quiver(0, 0, 0, L.spin_gas[0], L.spin_gas[1], L.spin_gas[2], color='lightblue', linewidth=3/i)
    
            # Plot sql data of gas/star spin (taken at large scale to include all particles that belong to the subhalo)
            ax.quiver(0, 0, 0, subhalo.spin_stars[0], subhalo.spin_stars[1], subhalo.spin_stars[2], color='r')
            ax.quiver(0, 0, 0, subhalo.spin_gas[0], subhalo.spin_gas[1], subhalo.spin_gas[2], color='b')"""
            
            i = i + 0.5
    
        #plt.savefig("./trial_plots/Misalignment_angle%s_subhalo%s.jpeg"%(str(mySims[0][1]), str(GroupNum)), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
        #plt.show()
    
        # Add angle to array
        misalignment_angle.append(L.mis_angle)
    
    # Graph initialising and base formatting
    graphformat(8, 11, 11, 11, 11, 5, 5)
    fig, ax = plt.subplots(1, 1, figsize=[8, 4])
    
    # Plot data as histogram
    plt.hist(misalignment_angle, bins=np.arange(0, 180, 10), histtype='bar', edgecolor='black', facecolor='dodgerblue', alpha=0.8)
    
    # General formatting
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 190, step=30))
    ax.set_ylim(0, 9)
    ax.set_xlabel('$\Psi$$_{gas-star}$')
    ax.set_ylabel('Number')
    ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    
    # Annotations
    ax.text(160, 8, "z = 0", fontsize=10)
    ax.axvline(30, ls='--', lw=0.5, c='k')
    plt.suptitle("L%s: Misalignment angle"%str(mySims[0][1]))
        
    plt.savefig("./trial_plots/Misalignment_angle%s.jpeg"%str(mySims[0][1]), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()
    plt.close()

        
        
        
        