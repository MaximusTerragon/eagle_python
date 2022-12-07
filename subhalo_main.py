import h5py
import numpy as np
import math
import astropy.units as u
from astropy.constants import G
import eagleSqlTools as sql
from pyread_eagle import EagleSnapshot
from read_dataset_tools import read_dataset, read_dataset_dm_mass, read_header


""" 
Purpose
-------
Will connect to sql database to find halfmassrad, perculiar velocity, 
centre coordinates (potential & mass) for a given snapshot, and then
extract data from local files.

Calling function
----------------
galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum, 
                                    centre_galaxy=True, load_region_length=2)

Input Parameters
----------------

mySims:
    mySims = np.array([('RefL0012N0188', 12)])  
    Name and boxsize
dataDir:
    Location of the snapshot data hdf5 file, 
    eg. '/Users/c22048063/Documents/.../snapshot_028_xx/snap_028_xx.0.hdf5'
snapNum: int
    Snapshot number, ei. 28
GroupNum: int
    GroupNumber of the subhalo, starts at 1
SubGroupNum: int
    SubGroupNumber of the subhalo, starts at 0 for each
    subhalo

centre_galaxy: boolean
    Whether to centre the galaxy both in coordinates
    and in perculiar velocity using the sql values
load_region_length: value [cMpc/h] (i think)
    Will use the eagle_data.select_region() routine to 
    select only a boxed region rather than entire box
    for speed. Leave at 2 for now.


Output Parameters
-----------------

.gn: int
    GroupNumber of galaxy
.sgn: int
    SubGroupNumber of galaxy
.a: 
    Scale factor for given snapshot
.h:
    0.6777 for z=0
.boxsize:   [cMpc/h]
    Size of simulation boxsize in [cMpc/h]. Convert
    to [pMpc] by .boxsize / .h
.stelmass:  [Msun]
    SQL value for total stellar mass
.gasmass:   [Msun]
    SQL value for total gas mass
.halfmass_rad:  [pkpc]
    SQL value for halfmass radius
.centre:    [pkpc]
    SQL value of centre of potential for the galaxy
.centre_mass:    [pkpc]
    SQL value of centre of mass for galaxy. This can
    be different to .centre
.perc_vel:  [pkm/s]
    SQL value of perculiar velocity of the galaxy
.stars:     dictionary of particle data:
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 
.gas:     dictionary of particle data
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['StarFormationRate']   - [Msun/s] (i think)
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 

If centre_galaxy == True; 'Coordinates' - .centre, 'Velocity' - .perc_vel
"""
class Subhalo_Extract:
    
    def __init__(self, sim, data_dir, snapNum, gn, sgn, centre_galaxy=True, load_region_length=2):       # cMpc/h
        # Assigning subhalo properties
        self.gn           = gn
        self.sgn          = sgn
        
        # Load information from the header
        self.a, self.h, self.boxsize = read_header(data_dir) # units of scale factor, h, and L [cMpc/h]    ASSUMES Z=0 I THINK
        
        # For a given gn and sgn, run sql query on SubFind catalogue
        myData = self.query(sim, snapNum)
        
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


    def query(self, sim, snapNum):
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
        			     SH.SnapNum = %i \
                         and SH.GroupNumber = %i \
                         and SH.SubGroupNumber = %i \
                      ORDER BY \
        			     SH.MassType_Star desc'%(sim_name, snapNum, self.gn, self.sgn)
	
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
        f = h5py.File(data_dir, 'r')
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
        

""" 
Purpose
-------
Will find useful particle data when given stars, gas

Calling function
----------------
subhalo = Subhalo(galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas, 
                                    viewing_angle,
                                    spin_rad_in,           
                                    kappa_rad_in,           
                                    trim_rad_in,            # False or value [pkpc]
                                    align_rad_in,           # False or value [pkpc]
                                    orientate_to_axis)    

Input Parameters
----------------

galaxy. values all from above function

viewing_angle:
    Will rotate output particle data by this angle
spin_rad_in:    array
    When given a list of values, for example:
    galaxy.halfmass_rad*np.arange(0.5, 10.5, 0.5)
    will calculate spin values within these values
kappa_rad_in:   False or value [pkpc]
    Will calculate kappa for this radius from centre
    of galaxy. Usually 30
trim_rad_in:    False of value [pkpc]
    Will trim the output data to this radius. This is
    used for render and 2dhisto
align_rad_in:   False or value [pkpc]
    Will orientate the galaxy based on the stellar 
    spin vector within this radius. Usually 30
orientate_to_axis:  'x', 'y', 'z'
    When align_rad_in == value, will orientate galaxy
    to this axis. 
quiet:  boolean
    Whether to print grid of:
        RAD, ANGLES-, PARTICLE COUNT, MASS
    

Output Parameters
-----------------

.gn: int
    GroupNumber of galaxy
.sgn: int
    SubGroupNumber of galaxy
.stelmass:  [Msun]
    Total summed mass of subhalo stellar particles
.gasmass:  [Msun]
    Total summed mass of subhalo gas particles
.gasmass_sf:  [Msun]
    Total summed mass of subhalo gas_sf particles
.gasmass_nsf:  [Msun]
    Total summed mass of subhalo gas_nsf particles
.halfmass_rad:  [pkpc]
    SQL value for halfmass radius
.centre:    [pkpc]
    SQL value of centre of potential for the galaxy
.centre_mass:    [pkpc]
    SQL value of centre of mass for galaxy. This can
    be different to .centre
.perc_vel:  [pkm/s]
    SQL value of perculiar velocity of the galaxy
.viewing_angle:     [deg]
    Angle by which we will rotate the galaxy, can be
    0

.mask_sf, mask_nsf:     array
    Used to mask gas for starforming and non-starforming
.kappa:
    Kappa calculated when galaxy orientated from 
    kappa_rad_in

.data, .data_align:    dictionary
    Has aligned/rotated values for 'stars', 'gas', 'gas_sf', 'gas_nsf':
        ['Coordinates']         - [pkpc]
        ['Velocity']            - [pkm/s]
        ['Mass']                - [Msun]
        ['StarFormationRate']   - [Msun/s] (i think)
        ['GroupNumber']         - int array 
        ['SubGroupNumber']      - int array
        
    if trim_rad_in has a value, coordinates lying outside
    these values will be trimmed in final output, but still
    used in calculations.
        
.spins, .spins_align:   dictionary
    Has aligned/rotated spin vectors within spin_rad_in's:
        ['rad']     - [pkpc]
        ['stars']   - [unit vector]
        ['gas']     - [unit vector]
        ['gas_sf']  - [unit vector]
        ['gas_nsf'] - [unit vector]
.particles, .particles_align:   dictionary
    Has aligned/rotated particle count and mass within spin_rad_in's:
        ['rad']          - [pkpc]
        ['stars']        - count
        ['gas']          - count
        ['gas_sf']       - count
        ['gas_nsf']      - count
        ['stars_mass']   - [Msun]
        ['gas_mass']     - [Msun]
        ['gas_sf_mass']  - [Msun]
        ['gas_nsf_mass'] - [Msun]
.mis_angles, .mis_angles_align:     dictionary
    Has aligned/rotated misalignment angles between stars 
    and X within spin_rad_in's:
        ['rad']     - [pkpc]
        ['gas']     - [deg]
        ['gas_sf']  - [deg]
        ['gas_nsf'] - [deg]
        
"""
class Subhalo:
    
    def __init__(self, halfmass_rad, centre, centre_mass, perc_vel, stars, gas, 
                            viewing_angle,
                            spin_rad_in, 
                            trim_rad_in, 
                            kappa_rad_in,
                            align_rad_in, 
                            orientate_to_axis,
                            quiet=False):
        
        # Create masks for starforming and non-starforming gas
        self.mask_sf        = np.nonzero(gas['StarFormationRate'])          
        self.mask_nsf       = np.where(gas['StarFormationRate'] == 0)
        
        # Assign bulk galaxy values to this subhalo
        self.gn             = int(stars['GroupNumber'][0])
        self.sgn            = int(stars['SubGroupNumber'][0])
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
        
        if not quiet:
            print('HALFMASS RAD', self.halfmass_rad)
            
        # KAPPA
        if kappa_rad_in:
            # Finding star unit vector within kappa_rad_in, finding angle between it and z and returning matrix for this
            stars_spin_kappa, _, _ = self.find_spin(data_nil['stars'], kappa_rad_in, 'stars')
            _ , matrix = self.orientate(orientate_to_axis, stars_spin_kappa)
            # Orientate entire galaxy according to matrix above, use this to find kappa
            stars_aligned_kappa  = self.rotate_galaxy(matrix, data_nil['stars'])
            self.kappa = self.kappa_co(stars_aligned_kappa, kappa_rad_in) 
        
            if not quiet:
                print('KAPPA', self.kappa)

        # ALIGN GALAXY
        if align_rad_in:
            # Large-scale stellar spin vector used to align galaxy
            # Finding star unit vector within align_rad_in, finding angle between it and z and returning matrix for this
            stars_spin_align, _, _  = self.find_spin(data_nil['stars'], align_rad_in, 'stars')
            _ , matrix = self.orientate('z', stars_spin_align)
            
            # Orientate entire galaxy according to matrix above
            self.data_align = {}
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                self.data_align['%s'%parttype_name] = self.rotate_galaxy(matrix, data_nil[parttype_name])
                
            # Find aligned spin vectors and particle count within radius
            self.spins_align     = {}
            self.particles_align = {}
            self.spins_align['rad']     = spin_rad_in
            self.spins_align['hmr'] = spin_rad_in/self.halfmass_rad
            self.particles_align['rad'] = spin_rad_in
            self.particles_align['hmr'] = spin_rad_in/self.halfmass_rad
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                tmp_spins = []
                tmp_particles = []
                tmp_mass = []
                for rad in spin_rad_in:
                    spin_x, particle_x, mass_x = self.find_spin(self.data_align[parttype_name], rad, parttype_name)
                    tmp_spins.append(spin_x)
                    tmp_particles.append(particle_x)
                    tmp_mass.append(mass_x)
                    
                self.spins_align[parttype_name]     = tmp_spins 
                self.particles_align[parttype_name] = tmp_particles
                self.particles_align[parttype_name + '_mass'] = tmp_mass
                
            # Find misalignment angles (does not find difference between every component ei. gas_sf and gas_nsf)
            self.mis_angles_align = {}
            self.mis_angles_align['rad'] = spin_rad_in
            self.mis_angles_align['hmr'] = spin_rad_in/self.halfmass_rad
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                for angle_name in ['gas', 'gas_sf', 'gas_nsf']:
                    tmp_angles = []
                    for i in np.arange(0, len(self.spins_align['stars']), 1):
                        tmp_angles.append(self.misalignment_angle(self.spins_align['stars'][i], self.spins_align[angle_name][i]))
                    self.mis_angles_align[angle_name] = tmp_angles
            
            # Trim output data to selected radii (trim_rad)
            if trim_rad_in:
                for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                    self.data_align['%s'%parttype_name] = self.trim_within_rad(self.data_align[parttype_name], trim_rad_in)
                  
            if not quiet:  
                print('MISALIGNMENT ANGLES ALIGN [deg]:')
                print('\t\tANGLES (STARS-)\t\tPARTICLE COUNT\t\t\tMASS')
                print('\tRAD\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
                for i in np.arange(0, len(self.mis_angles_align['rad']), 1):
                    print('\t%.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(self.mis_angles_align['rad'][i]/self.halfmass_rad, self.mis_angles_align['gas'][i], self.mis_angles_align['gas_sf'][i], self.mis_angles_align['gas_nsf'][i], self.particles_align['stars'][i], self.particles_align['gas'][i], self.particles_align['gas_sf'][i], self.particles_align['gas_nsf'][i], np.log10(self.particles_align['stars_mass'][i]), np.log10(self.particles_align['gas_mass'][i]), np.log10(self.particles_align['gas_sf_mass'][i]), np.log10(self.particles_align['gas_nsf_mass'][i])))        
            
        # SPIN VECTORS AND ROTATE
        if len(spin_rad_in) > 0:
            # Find rotation matrix to rotate galaxy
            matrix = self.rotate_around_axis('z', 360. - viewing_angle)
        
            self.data = {}
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                self.data['%s'%parttype_name] = self.rotate_galaxy(matrix, data_nil[parttype_name])

            # Find aligned spin vectors and particle count within radius
            self.spins     = {}
            self.particles = {}
            self.spins['rad']     = spin_rad_in
            self.spins['hmr']     = spin_rad_in/self.halfmass_rad
            self.particles['rad'] = spin_rad_in
            self.particles['hmr'] = spin_rad_in/self.halfmass_rad
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                tmp_spins = []
                tmp_particles = []
                tmp_mass = []
                for rad in spin_rad_in:
                    spin_x, particle_x, mass_x = self.find_spin(self.data[parttype_name], rad, parttype_name)
                    tmp_spins.append(spin_x)
                    tmp_particles.append(particle_x)
                    tmp_mass.append(mass_x)
                
                self.spins[parttype_name]     = tmp_spins 
                self.particles[parttype_name] = tmp_particles
                self.particles[parttype_name + '_mass'] = tmp_mass
                
            # Find misalignment angles (does not find difference between every component ei. gas_sf and gas_nsf)
            self.mis_angles = {}
            self.mis_angles['rad'] = spin_rad_in
            self.mis_angles['hmr'] = spin_rad_in/self.halfmass_rad
            for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                for angle_name in ['gas', 'gas_sf', 'gas_nsf']:
                    tmp_angles = []
                    for i in np.arange(0, len(self.spins['stars']), 1):
                        tmp_angles.append(self.misalignment_angle(self.spins['stars'][i], self.spins[angle_name][i]))
                    self.mis_angles[angle_name] = tmp_angles
        
            # Trim output data to selected radii (trim_rad)
            if trim_rad_in:
                for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                    self.data['%s'%parttype_name] = self.trim_within_rad(self.data[parttype_name], trim_rad_in)
        
            if not quiet:
                print('MISALIGNMENT ANGLES [deg]:')
                print('\t\tANGLES (STARS-)\t\tPARTICLE COUNT\t\t\tMASS')
                print('\tRAD\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
                for i in np.arange(0, len(self.mis_angles['rad']), 1):
                    print('\t%.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(self.mis_angles['rad'][i]/self.halfmass_rad, self.mis_angles['gas'][i], self.mis_angles['gas_sf'][i], self.mis_angles['gas_nsf'][i], self.particles['stars'][i], self.particles['gas'][i], self.particles['gas_sf'][i], self.particles['gas_nsf'][i], np.log10(self.particles['stars_mass'][i]), np.log10(self.particles['gas_mass'][i]), np.log10(self.particles['gas_sf_mass'][i]), np.log10(self.particles['gas_nsf_mass'][i])))        
                print('STELMASS', np.log10(self.stelmass))
                print('GASMASS', np.log10(self.gasmass))
                print('GASMASS_SF', np.log10(self.gasmass_sf))
                print('GASMASS_NSF', np.log10(self.gasmass_nsf))


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
        with np.errstate(divide='ignore', invalid='ignore'):
            spin = np.sum(L, axis=0)/np.sum(arr['Mass'][mask])
        
        # Expressing as unit vector
        spin_unit = spin / (spin[0]**2 + spin[1]**2 + spin[2]**2)**0.5
        
        # OUTPUTS UNIT VECTOR OF SPIN, PARTICLE COUNT WITHIN RAD, MASS WITHIN RAD 
        return spin_unit, len(r[mask]), np.sum(arr['Mass'][mask])
        
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
        
    def rotate_around_axis(self, axis, angle):
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
        
        
"""### MANUAL CALL
# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'

# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
GroupNum = 4
SubGroupNum = 0
snapNum = 28

# Initial extraction of galaxy data
galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)

spin_rad_in = galaxy.halfmass_rad*np.arange(0.5, 10.5, 0.5)   # pkpc
kappa_rad_in = 30                       # [pkpc] False or value
trim_rad_in = 30                        # [pkpc] False or value
align_rad_in = False    #30                       # [pkpc] False or value
viewing_angle = 10                      # [deg] will rotate subhalo.data by this angle about z-axis
orientate_to_axis = 'z'                 # 'z', 'y', 'x', will orientate axis to this angle based on stellar-spin in align_rad_in

spin_rad = spin_rad_in
trim_rad = trim_rad_in
kappa_rad = kappa_rad_in
align_rad = align_rad_in

subhalo = Subhalo(galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas, 
                                    viewing_angle,
                                    spin_rad,
                                    trim_rad, 
                                    kappa_rad, 
                                    align_rad,               #align_rad=False
                                    orientate_to_axis)     
"""







