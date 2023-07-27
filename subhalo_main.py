import h5py
import numpy as np
import math
import random
import astropy.units as u
import time
from tqdm import tqdm
from astropy.constants import G
import eagleSqlTools as sql
from pyread_eagle import EagleSnapshot
from read_dataset_tools import read_dataset, read_dataset_dm_mass, read_header
from astropy.cosmology import FlatLambdaCDM


#================================
""" Create sample. 
This returns a list of IDs, the snap and simulation

Returns:
-------

GroupNum
SubGroupNum
GalaxyID
SnapNum
Redshift

"""
# Creates a list of GN, SGN, GalaxyID, SnapNum, and Redshift and saves to csv file
class Initial_Sample:
    
    def __init__(self, sim, snapNum, mstarMin, mstarMax, satellite):
        # Allowing these attributes to be called from the object
        self.mstar_min   = mstarMin
        self.mstar_max   = mstarMax
        self.sim         = sim
        self.snapNum     = snapNum
        
        if mstarMin >= 1E+9:
            myData = self.samplesize_morph(satellite)
        else:
            myData = self.samplesize_basic(satellite)
            
        
        self.GroupNum     = myData['GroupNumber']
        self.SubGroupNum  = myData['SubGroupNumber']
        self.GalaxyID     = myData['GalaxyID']
        self.DescendantID = myData['DescendantID']
        self.SnapNum      = myData['SnapNum']
        self.Redshift     = myData['Redshift']
        self.halo_mass    = myData['halo_mass']
        self.centre       = np.transpose(np.array([myData['x'], myData['y'], myData['z']]))
        
        if mstarMin >= 1E+9:
            self.MorphoKinem = np.transpose(np.array([myData['ellip'], myData['triax'], myData['kappa_stars'], myData['disp_ani'], myData['disc_to_total'], myData['rot_to_disp_ratio']]))                  
            
    def samplesize_morph(self, satellite = False):
        # This uses the eagleSqlTools module to connect to the database with your username and password.
        # If the password is not given, the module will prompt for it.
        con = sql.connect('lms192', password='dhuKAP62')
        
        if satellite == True:
            for sim_name, sim_size in self.sim:
                #print(sim_name)
            
                # Construct and execute query for each simulation. This query returns properties for a single galaxy
                myQuery = 'SELECT \
                             SH.GroupNumber, \
                             SH.SubGroupNumber, \
                             SH.GalaxyID, \
                             SH.DescendantID, \
                             SH.SnapNum, \
                             SH.Redshift, \
                             SH.CentreOfPotential_x as x, \
                             SH.CentreOfPotential_y as y, \
                             SH.CentreOfPotential_z as z, \
                             FOF.Group_M_Crit200 as halo_mass, \
                             MK.Ellipticity as ellip, \
                             MK.Triaxiality as triax, \
                             MK.KappaCoRot as kappa_stars, \
                             MK.DispAnisotropy as disp_ani, \
                             MK.DiscToTotal as disc_to_total, \
                             MK.RotToDispRatio as rot_to_disp_ratio \
                           FROM \
                             %s_Subhalo as SH, \
                             %s_Aperture as AP, \
                             %s_FOF as FOF, \
                             %s_MorphoKinem as MK \
                           WHERE \
        			         SH.SnapNum = %i \
                             and AP.Mass_Star >= %f \
                             and AP.Mass_Star <= %f \
                             and AP.ApertureSize = 30 \
                             and SH.GalaxyID = AP.GalaxyID \
                             and SH.GalaxyID = MK.GalaxyID \
                             and SH.GroupID = FOF.GroupID \
                           ORDER BY \
        			         AP.Mass_Star desc'%(sim_name, sim_name, sim_name, sim_name, self.snapNum, self.mstar_min, self.mstar_max)
            
            # Execute query.
            myData = sql.execute_query(con, myQuery)
            
        elif satellite == False:
            for sim_name, sim_size in self.sim:
                #print(sim_name)
            
                # Construct and execute query for each simulation. This query returns properties for a single galaxy
                myQuery = 'SELECT \
                             SH.GroupNumber, \
                             SH.SubGroupNumber, \
                             SH.GalaxyID, \
                             SH.DescendantID, \
                             SH.SnapNum, \
                             SH.Redshift, \
                             SH.CentreOfPotential_x as x, \
                             SH.CentreOfPotential_y as y, \
                             SH.CentreOfPotential_z as z, \
                             FOF.Group_M_Crit200 as halo_mass, \
                             MK.Ellipticity as ellip, \
                             MK.Triaxiality as triax, \
                             MK.KappaCoRot as kappa_stars, \
                             MK.DispAnisotropy as disp_ani, \
                             MK.DiscToTotal as disc_to_total, \
                             MK.RotToDispRatio as rot_to_disp_ratio \
                           FROM \
                             %s_Subhalo as SH, \
                             %s_Aperture as AP, \
                             %s_FOF as FOF, \
                             %s_MorphoKinem as MK \
                           WHERE \
        			         SH.SnapNum = %i \
                             and AP.Mass_star >= %f \
                             and AP.Mass_Star <= %f \
                             and SH.SubGroupNumber = 0 \
                             and AP.ApertureSize = 30 \
                             and SH.GalaxyID = AP.GalaxyID \
                             and SH.GalaxyID = MK.GalaxyID \
                             and SH.GroupID = FOF.GroupID \
                           ORDER BY \
        			         AP.Mass_Star desc'%(sim_name, sim_name, sim_name, sim_name, self.snapNum, self.mstar_min, self.mstar_max)
    
            # Execute query.
            myData = sql.execute_query(con, myQuery)
        
        else:
            print('SATELLITE = "no"')

        return myData

    def samplesize_basic(self, satellite = False):
        # This uses the eagleSqlTools module to connect to the database with your username and password.
        # If the password is not given, the module will prompt for it.
        con = sql.connect('lms192', password='dhuKAP62')
        
        if satellite == True:
            for sim_name, sim_size in self.sim:
                #print(sim_name)
            
                # Construct and execute query for each simulation. This query returns properties for a single galaxy
                myQuery = 'SELECT \
                             SH.GroupNumber, \
                             SH.SubGroupNumber, \
                             SH.GalaxyID, \
                             SH.DescendantID, \
                             SH.SnapNum, \
                             SH.Redshift, \
                             SH.CentreOfPotential_x as x, \
                             SH.CentreOfPotential_y as y, \
                             SH.CentreOfPotential_z as z, \
                             FOF.Group_M_Crit200 as halo_mass \
                           FROM \
                             %s_Subhalo as SH, \
                             %s_Aperture as AP, \
                             %s_FOF as FOF \
                           WHERE \
        			         SH.SnapNum = %i \
                             and AP.Mass_Star >= %f \
                             and AP.Mass_Star <= %f \
                             and AP.ApertureSize = 30 \
                             and SH.GalaxyID = AP.GalaxyID \
                             and SH.GroupID = FOF.GroupID \
                           ORDER BY \
        			         AP.Mass_Star desc'%(sim_name, sim_name, sim_name, self.snapNum, self.mstar_min, self.mstar_max)
            
            # Execute query.
            myData = sql.execute_query(con, myQuery)
            
        elif satellite == False:
            for sim_name, sim_size in self.sim:
                #print(sim_name)
            
                # Construct and execute query for each simulation. This query returns properties for a single galaxy
                myQuery = 'SELECT \
                             SH.GroupNumber, \
                             SH.SubGroupNumber, \
                             SH.GalaxyID, \
                             SH.DescendantID, \
                             SH.SnapNum, \
                             SH.Redshift, \
                             SH.CentreOfPotential_x as x, \
                             SH.CentreOfPotential_y as y, \
                             SH.CentreOfPotential_z as z, \
                             FOF.Group_M_Crit200 as halo_mass \
                           FROM \
                             %s_Subhalo as SH, \
                             %s_Aperture as AP, \
                             %s_FOF as FOF \
                           WHERE \
        			         SH.SnapNum = %i \
                             and AP.Mass_star >= %f \
                             and AP.Mass_Star <= %f \
                             and SH.SubGroupNumber = 0 \
                             and AP.ApertureSize = 30 \
                             and SH.GalaxyID = AP.GalaxyID \
                             and SH.GroupID = FOF.GroupID \
                           ORDER BY \
        			         AP.Mass_Star desc'%(sim_name, sim_name, sim_name, self.snapNum, self.mstar_min, self.mstar_max)
    
            # Execute query.
            myData = sql.execute_query(con, myQuery)
        
        else:
            print('SATELLITE = "no"')

        return myData
    

""" Create sample with snips. 
This returns a list of IDs, the snap and simulation

Returns:
-------

GroupNum
SubGroupNum
GalaxyID
SnapNum
Redshift

"""
# Creates a list of GN, SGN, GalaxyID, SnapNum, and Redshift and saves to csv file
class Initial_Sample_Snip:
    
    def __init__(self, tree_dir, sim, snapNum, mstarMin, mstarMax, satellite, debug=False):
        # Allowing these attributes to be called from the object
        self.mstar_min   = mstarMin
        self.mstar_max   = mstarMax
        self.sim         = sim
        self.snapNum     = snapNum
        
        
        #--------------------------------------
        # Open main progenitor trees
        f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
        
        # Find snipNum that the snapnum corresponds to 
        self.snipNum     = np.array(f['Snapnum_Index']['Snapnum_Ref'])[snapNum]         # e.g. snapNum = 160 --> snipNum = 326
        
        
        #--------------------------------------
        # Extract data of snap we want
        data_raw = {}
        
        #for header_name in f['Histories'].keys():
        for header_name in ['GroupNumber', 'SubGroupNumber', 'GalaxyID', 'DescendantID', 'CentreOfPotential_x', 'CentreOfPotential_y', 'CentreOfPotential_z', 'StellarMass', 'HaloMass']:
            data_raw[header_name] = np.array(f['Histories'][header_name][:,snapNum])
        if debug:
            print(data_raw['GalaxyID'])
        
        #--------------------------------------
        # Find all galaxies that meet mstarMin + mstarMax + satellite criteria
        
        if satellite == True:
            satellite_lim = 99999
        elif satellite == False:
            satellite_lim = 0
        else:
            raise Exception('Cannot understand satellite criteria')
        
        mask = np.where((data_raw['StellarMass'] >= mstarMin) & (data_raw['StellarMass'] <= mstarMax) & (data_raw['SubGroupNumber'] <= satellite_lim) & (data_raw['GalaxyID'] != -1))
        
        # Apply mask                               
        myData = {}
        for header_name in data_raw.keys():
            myData[header_name] = data_raw[header_name][mask]
            
        # Fill with SnapNum, Redshift
        myData['SnapNum']  = np.full(len(myData['GroupNumber']), snapNum)
        myData['Redshift'] = np.full(len(myData['GroupNumber']), np.array(f['Snapnum_Index']['Redshift'])[snapNum])
        myData['ellip']             = np.full(len(myData['GroupNumber']), math.nan)
        myData['triax']             = np.full(len(myData['GroupNumber']), math.nan)
        myData['kappa_stars']       = np.full(len(myData['GroupNumber']), math.nan)
        myData['disp_ani']          = np.full(len(myData['GroupNumber']), math.nan)
        myData['disc_to_total']     = np.full(len(myData['GroupNumber']), math.nan)
        myData['rot_to_disp_ratio'] = np.full(len(myData['GroupNumber']), math.nan)
        
        if debug:
            print(len(myData['StellarMass']))
            print(myData['SubGroupNumber'])
        
        
        #--------------------------------------
        # Assign myData
        self.GroupNum     = myData['GroupNumber']
        self.SubGroupNum  = myData['SubGroupNumber']
        self.GalaxyID     = myData['GalaxyID']
        self.DescendantID = myData['DescendantID']
        self.SnapNum      = myData['SnapNum']
        self.Redshift     = myData['Redshift']
        self.halo_mass    = myData['HaloMass']
        self.centre       = np.transpose(np.array([myData['CentreOfPotential_x'], myData['CentreOfPotential_y'], myData['CentreOfPotential_z']]))
        
        self.stelmass     = myData['StellarMass']
        
        
        if mstarMin >= 1E+9:
            self.MorphoKinem = np.transpose(np.array([myData['ellip'], myData['triax'], myData['kappa_stars'], myData['disp_ani'], myData['disc_to_total'], myData['rot_to_disp_ratio']]))                  
        

#================================
""" 
Purpose
-------
Will connect to sql database to find halfmassrad, peculiar velocity, 
centre coordinates (potential & mass) for a given snapshot, and then
extract data from local files.

Calling function
----------------
galaxy = Subhalo_Extract(sample_input['mySims'], dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, aperture_rad, viewing_axis)

Input Parameters
----------------

mySims:
    mySims = np.array([('RefL0012N0188', 12)])  
    Name and boxsize
dataDir:
    Location of the snapshot data hdf5 file, 
    eg. '/Users/c22048063/Documents/.../snapshot_028_xx/snap_028_xx.0.hdf5'
SnapNum: int
    Snapshot number, ei. 28
GroupNum: int
    GroupNumber of the subhalo, starts at 1
SubGroupNum: int
    SubGroupNumber of the subhalo, starts at 0 for each
    subhalo
aperture_rad_in: float, [pkpc]
    Used to trim data, which is used to find peculiar velocity within this sphere
viewing_axis: 'z'
    Used to find projected halfmass radius


Output Parameters
-----------------

.gn: int
    GroupNumber of galaxy
.sgn: int
    SubGroupNumber of galaxy
.a: 
    Scale factor for given snapshot
.aexp: 
    Scale factor exponent for given snapshot
.h:
    0.6777 for z=0
.hexp: 
    h exponent for given snapshot
.boxsize:   [cMpc/h]
    Size of simulation boxsize in [cMpc/h]. Convert
    to [pMpc] by .boxsize / .h
.centre:    [pkpc]
    SQL value of centre of potential for the galaxy
        
.halo_mass:     float
    Value of the halomass within 200 density
.stars_com:     np.array[x, y, z]
    stars C.O.M within aperture, w.r.t [0, 0, 0]
.stars:     dictionary of particle data:
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 
    ['Metallicity']         - see particle paper
.gas:     dictionary of particle data
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['StarFormationRate']   - [Msun/s]
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 
    ['Metallicity']         - see particle paper
.dm:     dictionary of particle data:
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 
.bh:     dictionary of particle data:
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 

If centre_galaxy == True; 'Coordinates' - .centre, 'Velocity' - .perc_vel
"""
# Extracts the particle and SQL data from snips
class Subhalo_Extract:
    
    def __init__(self, sim, data_dir, snapNum, gn, sgn, centre_in, halo_mass_in, aperture_rad_in, viewing_axis,
                            centre_galaxy=True, 
                            load_region_length=1.0,   # cMpc/h 
                            nfiles=1, 
                            debug=False,
                            print_progress=False):       
                            
        # Begining time
        time_start = time.time()
        
        # Assigning subhalo properties
        self.gn           = gn
        self.sgn          = sgn
        
        #----------------------------------------------------
        # Load information from the header for this snapshot to find a, aexp, h, hexp, boxsize
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Reading header')
            time_start = time.time()
        self.a, self.h, self.boxsize = read_header(data_dir) # units of scale factor, h, and L [cMpc/h]  
        
        
        # Distances:    [cMpc/h] * a^1 *h^-1 -> [pMpc]. [pMpc/h] * h^-1 -> [pMpc], [cMpc/h] * h^-1 -> [cMpc]
        # Velocity:     [cx/sh] * a^0.5 * h^0 -> [x/s]
        # Mass:         [Mass/h] * a^0 * h*-1 -> [Mass]
        for i in range(nfiles):
            f = h5py.File(data_dir, 'r')
            tmp = f['PartType4/Coordinates']

            # Get conversion factors.
            self.aexp    = f['PartType4/Coordinates'].attrs.get('aexp-scale-exponent')
            self.hexp    = f['PartType4/Coordinates'].attrs.get('h-scale-exponent')
            
            #print('available headers')
            #print(f.keys())
            #for name_i, header_i in zip(['Gas', 'DM', 'Stars', 'BH'], ['PartType0', 'PartType1', 'PartType4', 'PartType5']):
            #    print('\n %s' %name_i)
            #    for key_i in f[header_i].keys():
            #        print('    %s' %key_i)

            f.close()
        
        if debug:
            print('a        ', self.a)
            print('aexp     ', self.aexp)
            print('h        ', self.h)
            print('hexp     ', self.hexp)
            print('boxsize  [cMpc/h]', self.boxsize)
            print('boxsize  [cMpc]  ', self.boxsize/self.h)
            print('boxsize  [pMpc]  ', self.boxsize*self.a**1)
        
        
        #----------------------------------------------------
        # For a given gn and sgn, run sql query on SubFind catalogue
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Subhalo COP query')
            time_start = time.time()
        
        # Assigning halo mass
        self.halo_mass = halo_mass_in
        
        # These were all originally in cMpc, converted to pMpc through self.a and self.aexp
        self.centre       = centre_in * u.Mpc.to(u.kpc) * self.a**self.aexp                 # [pkpc]
        
        #-------------------------------------------------------------
        # Load data for stars and gas in non-centred units
        # Msun, pkpc, and pkpc/s
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Reading particle data _read_galaxy')
            time_start = time.time()
        stars     = self._read_galaxy(data_dir, 4, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length, snapNum) 
        gas       = self._read_galaxy(data_dir, 0, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length, snapNum)
        dm        = self._read_galaxy(data_dir, 1, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length, snapNum)
        bh        = self._read_galaxy(data_dir, 5, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length, snapNum)
        
        # CENTER COORDS, VELOCITY NOT ADJUSTED
        if centre_galaxy == True:
            stars['Coordinates'] = stars['Coordinates'] - self.centre
            gas['Coordinates']   = gas['Coordinates'] - self.centre
            dm['Coordinates']    = dm['Coordinates'] - self.centre
            bh['Coordinates']    = bh['Coordinates'] - self.centre
            
        
        #---------------------------
        # Finding stars COM within 30pkpc
        stars_com  = self._centre_of_mass(self._trim_within_rad(stars, aperture_rad_in))
        if debug:
            print('stars COM')
            print(stars_com)
        
        # Setting the stars COM in 30pkpc as the new centre, and the external stars_com w.r.t potential (which until now was at [0, 0, 0])
        stars['Coordinates'] = stars['Coordinates'] - stars_com
        gas['Coordinates']   = gas['Coordinates'] - stars_com
        dm['Coordinates']    = dm['Coordinates'] - stars_com
        bh['Coordinates']    = bh['Coordinates'] - stars_com
        self.stars_com       = [0, 0, 0] - stars_com
        
        #---------------------------
        # Creating temporary array for stars centred on stellar COM, trimming this to the aperture
        stars_trimmed = self._trim_within_rad(stars, aperture_rad_in)
        
        # Making the minimum HMR 1 pkpc
        self.halfmass_rad_proj  = max(self._projected_rad(stars_trimmed, viewing_axis), 1)
        self.halfmass_rad       = max(self._half_rad(stars_trimmed), 1)
            
        # Finding peculiar velocity of stars within 30 pkpc of COM
        self.perc_vel = self._peculiar_velocity_part(stars_trimmed)
         
        """# Finding peculiar velocity for all particles
        self.perc_vel = self._peculiar_velocity()
        """
        
        # account for peculiar velocity within aperture rad
        if centre_galaxy == True:            
            stars['Velocity'] = stars['Velocity'] - self.perc_vel
            gas['Velocity']   = gas['Velocity'] - self.perc_vel
            dm['Velocity']    = dm['Velocity'] - self.perc_vel
            bh['Velocity']    = bh['Velocity'] - self.perc_vel
        
        #---------------------------
        # Assigning particle data        
        self.data_nil = {}
        for parttype, parttype_name in zip([stars, gas, dm, bh], ['stars', 'gas', 'dm', 'bh']):
            self.data_nil['%s'%parttype_name] = parttype
        
        
        if debug:
            print('_main DEBUG')
            print(self.halfmass_rad)
            print(self.halfmass_rad_proj)
            print(self.perc_vel)   
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('  EXTRACTION COMPLETE')
        
    def _read_galaxy(self, data_dir, itype, gn, sgn, centre, load_region_length, snapNum, debug=False):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates, velocty, and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity."""
        
        # Where we store all the data
        data = {}
        
        # Initialize read_eagle module.
        eagle_data = EagleSnapshot(data_dir)
        
        # Put centre from pMpc -> cMpc/h units.
        centre_cMpc = centre * self.a**-1 * self.h

        # Select region to load, a 'load_region_length' cMpc/h cube centred on 'centre'.
        region = np.array([
            (centre_cMpc[0]-0.5*load_region_length), (centre_cMpc[0]+0.5*load_region_length),
            (centre_cMpc[1]-0.5*load_region_length), (centre_cMpc[1]+0.5*load_region_length),
            (centre_cMpc[2]-0.5*load_region_length), (centre_cMpc[2]+0.5*load_region_length)
        ])
        eagle_data.select_region(*region)
                
        # Load data using read_eagle, load conversion factors manually.
        f = h5py.File(data_dir, 'r')
        # If gas, load StarFormationRate
        if itype == 0:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'StarFormationRate', 'Velocity', 'ParticleIDs', 'Metallicity']:
                if (int(snapNum) > 28) and att == 'SubGroupNumber':
                    continue
                if att != 'StarFormationRate':
                    tmp  = eagle_data.read_dataset(itype, att)
                    cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                    data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                    
                elif (int(snapNum) <= 28) & (att == 'StarFormationRate'):
                    tmp  = eagle_data.read_dataset(itype, att)
                    cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                    data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                    
                elif (int(snapNum) > 28) & (att == 'StarFormationRate'):                    
                    # Read constants (cgs)
                    m_proton   = f['Constants'].attrs.get('PROTONMASS')
                    cm_per_mpc = f['Constants'].attrs.get('CM_PER_MPC')
                    gravity    = f['Constants'].attrs.get('GRAVITY')
                    msun       = f['Constants'].attrs.get('SOLAR_MASS')
                    
                    
                    # Extract attributes
                    gaspmass   = eagle_data.read_dataset(itype, 'Mass') * f['PartType%i/%s'%(itype, 'Mass')].attrs.get('CGSConversionFactor') * self.a**f['PartType%i/%s'%(itype, 'Mass')].attrs.get('aexp-scale-exponent') * self.h**f['PartType%i/%s'%(itype, 'Mass')].attrs.get('h-scale-exponent')
                    gasu       = eagle_data.read_dataset(itype, 'InternalEnergy') * f['PartType%i/%s'%(itype, 'InternalEnergy')].attrs.get('CGSConversionFactor') * self.a**f['PartType%i/%s'%(itype, 'InternalEnergy')].attrs.get('aexp-scale-exponent') * self.h**f['PartType%i/%s'%(itype, 'InternalEnergy')].attrs.get('h-scale-exponent')
                    gasentropy = eagle_data.read_dataset(itype, 'Entropy') * f['PartType%i/%s'%(itype, 'Entropy')].attrs.get('CGSConversionFactor') * self.a**f['PartType%i/%s'%(itype, 'Entropy')].attrs.get('aexp-scale-exponent') * self.h**f['PartType%i/%s'%(itype, 'Entropy')].attrs.get('h-scale-exponent')
                    gasmetal   = eagle_data.read_dataset(itype, 'Metallicity') * f['PartType%i/%s'%(itype, 'Metallicity')].attrs.get('CGSConversionFactor')  * self.a**f['PartType%i/%s'%(itype, 'Metallicity')].attrs.get('aexp-scale-exponent') * self.h**f['PartType%i/%s'%(itype, 'Metallicity')].attrs.get('h-scale-exponent')
                    gastemp    = eagle_data.read_dataset(itype, 'Temperature') * f['PartType%i/%s'%(itype, 'Temperature')].attrs.get('CGSConversionFactor')  * self.a**f['PartType%i/%s'%(itype, 'Temperature')].attrs.get('aexp-scale-exponent') * self.h**f['PartType%i/%s'%(itype, 'Temperature')].attrs.get('h-scale-exponent')
                    gasdens    = ((2/3)*gasu/gasentropy)**(3/2)
                    gaspres    = (2/3) * gasdens * gasu
                    
                    #print('snip gas dens', gasdens)
                    #gasdens = eagle_data.read_dataset(itype, 'Density') * f['PartType%i/%s'%(itype, 'Density')].attrs.get('CGSConversionFactor')  * self.a**f['PartType%i/%s'%(itype, 'Density')].attrs.get('aexp-scale-exponent') * self.h**f['PartType%i/%s'%(itype, 'Density')].attrs.get('h-scale-exponent')
                    #print('stored gas dens', gasdens)
                    
                    
                    #--------------
                	# Find hydrogen threshold
                    with np.errstate(divide='ignore'):
                        nhthresh  = 0.1 * (gasmetal / 0.002)**(-0.64)           # Threshold for starforming gas
                    
                    # Mask hydrogen threshold
                    mask_wherehigh = nhthresh > 10                          # upper limit from schaye 2015 equation (2)
                    nhigh = np.count_nonzero(mask_wherehigh)                
                    if nhigh > 0:
                        nhthresh[mask_wherehigh] = 10.
                        
                    # Hydrogen density fraction threshold
                    rhothresh = nhthresh * m_proton / 0.752         # X = 0.752
                    nindex  = 1.4                                   
                    nindex2 = 2.                                    # for nH > 1e3 cm^-3
    
                    mask_wheredense = (gasdens/(m_proton/0.752)) > 1e3
                    ndense = np.count_nonzero(mask_wheredense)  
                    sfr = gaspmass * 1.515e-4 * (msun / (3.154e+7 * (cm_per_mpc/1e3)**2)) * (msun / (cm_per_mpc/1e6)**2)**(-nindex) * ((5/3) * 1.0 * gaspres / gravity)**((nindex - 1)/2)
                    
                    if ndense > 0:
                        sfr[mask_wheredense] = gaspmass[mask_wheredense] * 1.515e-4 * (msun / (3.154e+7 * (cm_per_mpc/1e3)**2)) * (msun / (cm_per_mpc/1e6)**2)**(-nindex2) * ((5/3) * 1.0 * gaspres[mask_wheredense] / gravity)**((nindex2 - 1)/2)
                    
                    #--------------
                    # Mask out sfr where rhothres not met or gas too hot
                    mask_wherezero = (gasdens < rhothresh) | (gastemp > 1e6)
                    sfr[mask_wherezero] = 0
                    
                    data[att] = sfr
                    
            f.close()
        # If dm
        elif itype == 1:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'Velocity', 'ParticleIDs']:
                if (int(snapNum) > 28) and att == 'SubGroupNumber':
                    continue
                if att == 'Mass':
                    cgs  = f['PartType0/%s'%(att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType0/%s'%(att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType0/%s'%(att)].attrs.get('h-scale-exponent')
                    
                    # Create special extract dm mass
                    dm_mass     = f['Header'].attrs.get('MassTable')[1]
                    # Create an array of length n_particles each set to dm_mass.
                    m = np.ones(n_particles, dtype='f8') * dm_mass
            
                    data[att] = np.multiply(m, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                else:
                    tmp  = eagle_data.read_dataset(itype, att)
                    cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                    data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                    
                    # Used in next cycle for Mass
                    n_particles = len(tmp)
                
            f.close()
        # If stars, do not load StarFormationRate (as not contained in database)
        elif itype == 4:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'Velocity', 'ParticleIDs', 'Metallicity']:
                if (int(snapNum) > 28) and att == 'SubGroupNumber':
                    continue
                tmp  = eagle_data.read_dataset(itype, att)
                cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                
            f.close()
        # If bhs
        elif itype == 5:
            for att in ['GroupNumber', 'SubGroupNumber', 'BH_Mass', 'BH_Mdot', 'Coordinates', 'Velocity', 'ParticleIDs']:
                if (int(snapNum) > 28) and att == 'SubGroupNumber':
                    continue
                # Ensure we use 'Mass' as name
                if att == 'BH_Mass':
                    tmp  = eagle_data.read_dataset(itype, att)
                    cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                    data['Mass'] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                    
                else:
                    tmp  = eagle_data.read_dataset(itype, att)
                    cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                    data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                    
            f.close()
        
        
        # Mask to selected GroupNumber and SubGroupNumber.
        if (int(snapNum) > 28):
            mask = data['GroupNumber'] == gn
        else:
            mask = np.logical_and(data['GroupNumber'] == gn, data['SubGroupNumber'] == sgn)
        for att in data.keys():
            data[att] = data[att][mask]
               
        # Load data, then mask to selected GroupNumber and SubGroupNumber. Automatically converts to pcm from read_dataset, converted to pMpc
        data['Mass'] = data['Mass'] * u.g.to(u.Msun)                   # [Msun]
        data['Coordinates'] = data['Coordinates'] * u.cm.to(u.Mpc)     # [pMpc]
        data['Velocity'] = data['Velocity'] * u.cm.to(u.Mpc)           # [pMpc/s]
        if itype == 0:
            data['StarFormationRate'] = data['StarFormationRate'] * u.g.to(u.Msun)  # [Msun/s]
        if itype == 5:
            data['BH_Mdot'] = data['BH_Mdot'] * u.g.to(u.Msun)  # [Msun/s]
        
        # Periodic wrap coordinates around centre (in proper units). 
        # boxsize converted from cMpc/h -> pMpc
        boxsize = self.boxsize * self.h**-1 * self.a**1       # [pkpc]
        data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
        
        # Converting to pkpc
        data['Coordinates'] = data['Coordinates'] * u.Mpc.to(u.kpc)     # [pkpc]
        data['Velocity'] = data['Velocity'] * u.Mpc.to(u.km)            # [pkm/s]
        
        return data
        
    def _trim_within_rad(self, arr, radius, debug=False):
        # Compute distance to centre and mask all within Radius in pkpc
        r  = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.where(r <= radius)
        
        newData = {}
        for header in arr.keys():
            newData[header] = arr[header][mask]
            
        return newData
        
    def _centre_of_mass(self, arr, debug=False):
        # Mass-weighted formula from subhalo paper
        mass_weighted  = arr['Coordinates'] * arr['Mass'][:, None]
        centre_of_mass = np.sum(mass_weighted, axis=0)/np.sum(arr['Mass'])
    
        return centre_of_mass
    
    def _peculiar_velocity(self, debug=False):        
        # Mass-weighted formula from subhalo paper
        vel_weighted = 0
        mass_sums = 0
        for arr in [self.stars, self.gas, self.dm, self.bh]:
            if len(arr['Mass']) > 0:
                vel_weighted = vel_weighted + np.sum(arr['Velocity'] * arr['Mass'][:, None], axis=0)
                mass_sums = mass_sums + np.sum(arr['Mass'])
            
        pec_vel = vel_weighted / mass_sums
            
        return pec_vel
        
    def _peculiar_velocity_part(self, arr, debug=False):        
        # Mass-weighted formula from subhalo paper
        vel_weighted = np.sum(arr['Velocity'] * arr['Mass'][:, None], axis=0)
        mass_sums = np.sum(arr['Mass'])
            
        pec_vel = vel_weighted / mass_sums
            
        return pec_vel
    
    def _projected_rad(self, arr, viewing_axis, debug=False):
        # Compute distance to centre (projected)        
        if viewing_axis == 'z':
            r = np.linalg.norm(arr['Coordinates'][:,[0,1]], axis=1)
            mask = np.argsort(r)
            r = r[mask]
        if viewing_axis == 'y':
            r = np.linalg.norm(arr['Coordinates'][:,[0,2]], axis=1)
            mask = np.argsort(r)
            r = r[mask]
        if viewing_axis == 'x':
            r = np.linalg.norm(arr['Coordinates'][:,[1,2]], axis=1)
            mask = np.argsort(r)
            r = r[mask]
            
        stelmass = np.sum(arr['Mass'][mask])
            
        # Compute cumulative mass
        cmass = np.cumsum(arr['Mass'][mask])
        index = np.where(cmass >= stelmass*0.5)[0][0]
        radius = r[index]
        
        return radius
        
    def _half_rad(self, arr, debug=False):
        # Compute distance to centre       
        r = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.argsort(r)
        r = r[mask]
            
        stelmass = np.sum(arr['Mass'][mask])
            
        # Compute cumulative mass
        cmass = np.cumsum(arr['Mass'][mask])
        index = np.where(cmass >= stelmass*0.5)[0][0]
        radius = r[index]
        
        return radius


""" 
Purpose
-------
Will find useful particle data when given stars, gas

Calling function
----------------
Subhalo_Analysis(sample_input['mySims'], GroupNum, SubGroupNum, GalaxyID, SnapNum, galaxy.halfmass_rad, galaxy.halfmass_rad_proj, galaxy.halo_mass, galaxy.stars, galaxy.gas, galaxy.dm, galaxy.bh, 
                                            viewing_axis,
                                            aperture_rad,
                                            kappa_rad, 
                                            trim_hmr, 
                                            align_rad,              #align_rad = False
                                            orientate_to_axis,
                                            viewing_angle,
                                            
                                            angle_selection,        
                                            spin_rad,
                                            spin_hmr,
                                            find_uncertainties,
                                            
                                            com_min_distance,
                                            min_particles,                                            
                                            min_inclination)

If find_uncertainties = False, will append NaNs to uncertainties... so can
        safely plot.

Input Parameters
----------------

galaxy. values all from above function

viewing_axis:   'x', 'y', 'z'
    Defaults to 'z', speeds up process to only find
    uncertainties in this viewing axis
aperture_rad:    value [pkpc]
    Will trim data to this maximum value for quicker processing.
    Usually 30.
kappa_rad:   False or value [pkpc]
    Will calculate kappa for this radius from centre
    of galaxy. Usually 30
trim_hmr:    array [multiples of rad]
    Will trim the output data to this radius. This is
    used for render and 2dhisto
align_rad:   False or value [pkpc]
    Will orientate the galaxy based on the stellar 
    spin vector within this radius. Usually 30
orientate_to_axis:  'x', 'y', 'z'
    When align_rad_in == value, will orientate galaxy
    to this axis. 
viewing_angle:
    Will rotate output particle data by this angle
        
angle_selection:
    Will speed up process to find only specific angles' uncertainties. 
    Automated process.   
        ['stars_gas',            # stars_gas     stars_gas_sf    stars_gas_nsf
         'stars_gas_sf',         # gas_dm        gas_sf_dm       gas_nsf_dm
         'stars_gas_nsf',        # gas_sf_gas_nsf
         'gas_sf_gas_nsf',
         'stars_dm']
spin_rad:    array [pkpc] 
    When given a list of values, for example:
    galaxy.halfmass_rad*np.arange(0.5, 10.5, 0.5)
    will calculate spin values within these values
spin_hmr:    array .
    When given a list of values, for example:
    np.arange(0.5, 10.5, 0.5)
    will calculate spin values within these values
find_uncertainties:     Boolean
    If True, will create rand_spins to find estimate of
    uncertainty for 3D angles
        
com_min_distance: [pkpc]
    Will initiate flag if this not met within spin_rad
    (if c.o.m > 2.0, in 2HMR)
    Will flag but continue to extract all values 
min_particles:
    Minimum gas_sf (and gas_nsf, stars) particle within spin_rad.
    Will flag but continue to extract all analytical values
        uncertainties not calculated if not met
min_inclination:
    Minimum and maximum spin inclination for a given rad.
    Will flag but continue to extract all values 
        
Output Parameters
-----------------

.general:       dictionary
    'GroupNum', 'SubGroupNum', 'GalaxyID', 'SnapNum', 
    'halo_mass', 'stelmass', 'gasmass', 'gasmass_sf', 'gasmass_nsf', 
    'halfmass_rad', 'halfmass_rad_proj'

.flags:     dictionary
    Has list of arrays that will be != if flagged. Contains hmr at failure, or 30pkpc
        ['total_particles']
            will flag if there are missing particles within aperture_rad
            ['stars']       - [hmr]
            ['gas']         - [hmr]
            ['gas_sf']      - [hmr]
            ['gas_nsf']     - [hmr]
            ['dm']          - [hmr]
            ['bh']          - [hmr]
        ['min_particles']
            will flag if min. particles not met within spin_rad (will find spin if particles exist, but no uncertainties)
            ['stars']       - [hmr]
            ['gas']         - [hmr]
            ['gas_sf']      - [hmr]
            ['gas_nsf']     - [hmr]
            ['dm']          - [hmr]
        ['min_inclination']
            will flag if inclination angle not met within spin_rad... all spins and uncertainties still calculated
            ['stars']       - [hmr]
            ['gas']         - [hmr]
            ['gas_sf']      - [hmr]
            ['gas_nsf']     - [hmr]
            ['dm']          - [hmr]
        ['com_min_distance']
            will flag if com distance not met within spin_rad... all spins and uncertainties still calculated
            ['stars_gas']   - [hmr]
            ['stars_gas_sf']- [hmr]
            ... for all angle_selection ...
              
.data:    dictionary
    Has aligned/rotated values for 'stars', 'gas', 'gas_sf', 'gas_nsf':
        [hmr]                       - multiples of hmr, ei. '1.0' that data was trimmed to
            ['Coordinates']         - [pkpc]
            ['Velocity']            - [pkm/s]
            ['Mass']                - [Msun]
            ['StarFormationRate']   - [Msun/s] (i think)
            ['GroupNumber']         - int array 
            ['SubGroupNumber']      - int array
        
    if trim_hmr_in has a value, coordinates lying outside
    these values will be trimmed in final output, but still
    used in calculations.
        
.spins:   dictionary
    Has aligned/rotated spin vectors within spin_rad_in's:
        ['rad']     - [pkpc]
        ['hmr']     - multiples of halfmass_rad
        ['stars']   - [unit vector]
        ['gas']     - [unit vector]
        ['gas_sf']  - [unit vector]
        ['gas_nsf'] - [unit vector]
        ['dm']      - [unit vector]
.counts:   dictionary
    Has aligned/rotated particle count and mass within spin_rad_in's:
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['stars']        - [count]
        ['gas']          - [count]
        ['gas_sf']       - [count]
        ['gas_nsf']      - [count]
        ['dm']           - count at 30pkpc
.masses:   dictionary
    Has aligned/rotated particle count and mass within spin_rad_in's:
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['stars']        - [Msun]
        ['gas']          - [Msun]
        ['gas_sf']       - [Msun]
        ['gas_nsf']      - [Msun]
        ['dm']           - Msun at 30pkpc
.Z:   dictionary        
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['stars']        - mass-weighted metallicity
        ['gas']          - mass-weighted metallicity
        ['gas_sf']       - mass-weighted metallicity
        ['gas_nsf']      - mass-weighted metallicity
.sfr:       dictionary
        ['rad']          - [pkpc]
        ['hmr']          - multiples of halfmass_rad
        ['gas_sf']       - total SFR
.coms:     dictionary
    Has all centres of mass and distances within a spin_rad_in:
        ['rad']          - [pkpc]
        ['hmr']     - multiples of halfmass_rad
        ['stars']          - [x, y, z] [pkpc] distance in 3D
        ['gas']            - [x, y, z] [pkpc]
        ['gas_sf']         - [x, y, z] [pkpc]
        ['gas_nsf']        - [x, y, z] [pkpc]
        ['dm']             - x, y, z [pkpc]  at 30pkpc
        ['adjust']         - [x, y, z] [pkpc] the value added on to each of the above from centering to stellar COM from 30pkpc COM
.mis_angles:     dictionary
    Has aligned/rotated misalignment angles between stars 
    and X within spin_rad_in's. Errors given by iterations,
    which defults to 500.
        ['rad']            - [pkpc]
        ['hmr']            - multiples of halfmass_rad
        ['stars_gas_angle']             - [deg]                -
        ['stars_gas_sf_angle']          - [deg]                .
        ['stars_gas_nsf_angle']         - [deg]                .
        ['gas_sf_gas_nsf_angle']        - [deg]                .
        ['stars_dm_angle']              - [deg]                .
        ['gas_dm_angle']                - [deg]                .
        ['gas_sf_dm_angle']             - [deg]                .
        ['gas_nsf_dm_angle']            - [deg]                .
        ['stars_gas_angle_err']         - [lo, hi] [deg]       .
        ['stars_gas_sf_angle_err']      - [lo, hi] [deg]       .
        ['stars_gas_nsf_angle_err']     - [lo, hi] [deg]       (assuming it was passed into angle_selection and not flagged)
        ['gas_sf_gas_nsf_angle_err']    - [lo, hi] [deg]       .
        ['stars_dm_angle_err']          - [lo, hi] [deg]       . 
        ['gas_dm_angle_err']            - [lo, hi] [deg]       .
        ['gas_sf_dm_angle_err']         - [lo, hi] [deg]       .
        ['gas_nsf_dm_angle_err']        - [lo, hi] [deg]       ^
.mis_angles_proj                    dictionary
    Has projected misalignment angles. Errors given by iterations,
    which defults to 500.
        ['x']
        ['y']
        ['z']
            ['rad']            - [pkpc]
            ['hmr']            - multiples of halfmass_rad
            ['stars_gas_angle']             - [deg]                -
            ['stars_gas_sf_angle']          - [deg]                .
            ['stars_gas_nsf_angle']         - [deg]                .
            ['gas_sf_gas_nsf_angle']        - [deg]                .
            ['stars_dm_angle']              - [deg]                .
            ['gas_dm_angle']                - [deg]                .
            ['gas_sf_dm_angle']             - [deg]                .
            ['gas_nsf_dm_angle']            - [deg]                .
            ['stars_gas_angle_err']         - [lo, hi] [deg]       -
            ['stars_gas_sf_angle_err']      - [lo, hi] [deg]       .
            ['stars_gas_nsf_angle_err']     - [lo, hi] [deg]       (assuming it was passed into angle_selection and not flagged)
            ['gas_sf_gas_nsf_angle_err']    - [lo, hi] [deg]       .
            ['stars_dm_angle_err']          - [lo, hi] [deg]       . 
            ['gas_dm_angle_err']            - [lo, hi] [deg]       .
            ['gas_sf_dm_angle_err']         - [lo, hi] [deg]       .
            ['gas_nsf_dm_angle_err']        - [lo, hi] [deg]       ^
.gas_data:      dictionary
    Has particle data of hmr requested
        ['1.0_hmr']
        ['2.0_hmr']
            ['gas']
            ['gas_sf']
            ['gas_nsf']
                ['ParticleIDs']     - Particle IDs
                ['Mass']            - Particle masses
                ['Metallicity']     - Particle metallicities
.mass_flow:     dictionary
    Has inflow/outflow data if gas_data_old provided. 
    math.nan if not
        ['1.0_hmr']
        ['2.0_hmr']
            ['gas']                     - Inflow/outflow of all gas
            ['gas_sf']                  - Inflow/outflow of only SF gas
            ['gas_nsf']                 - Inflow/outflow of only NSF gas
                ['inflow']              - Accurate inflow
                ['outflow']             - Accurate outflow
                ['massloss']            - Whats left... does not take into account gas particles switching between modes
                ['inflow_Z']            - Inflow metallicity
                ['outflow_Z']           - Outflow metallicity
                ['insitu_Z']            - Whats left... metallicity
"""
# Finds the values we are after and trims particle data
class Subhalo_Analysis:
    
    def __init__(self, sim, GroupNum, SubGroupNum, GalaxyID, SnapNum, MorphoKinem_in, halfmass_rad, halfmass_rad_proj, halo_mass_in, data_nil,
                            viewing_axis, 
                            aperture_rad,
                            kappa_rad,
                            trim_rad, 
                            align_rad, 
                            orientate_to_axis,
                            viewing_angle,
                            
                            angle_selection,        # ['stars_gas', 'stars_gas_sf', ...]
                            spin_rad, 
                            spin_hmr,
                            find_uncertainties,
                            rad_projected,
                            
                            com_min_distance,
                            min_particles,
                            min_inclination,
                            
                            gas_data_old=False,
                            
                            debug=False,
                            print_progress=False):
                            
        
        #======================================================
        # Keep data or align galaxy for all components that have particle counts
        if not align_rad:
            # Find rotation matrix to rotate entire galaxy depending on viewing_angle if viewing_axis is not 0
            if viewing_angle != 0:
                matrix = self._rotate_around_axis('z', 360. - viewing_angle)

                for parttype_name in ['stars', 'gas', 'dm', 'bh']:
                    # If there are no flags (ei. there exists particle data for the given type)... rotate, else: keep as before
                    if len(data_nil[parttype_name]['Mass']) > 0:
                        data_nil[parttype_name] = self._rotate_galaxy(matrix, data_nil[parttype_name])
    
        else:
            # Large-scale stellar spin vector within align_rad used to align galaxy
            # Trim data to particular radius
            trimmed_data = self._trim_data(data_nil, align_rad)
            
            # Finding star unit vector within align_rad_in, finding angle between it and z and returning matrix for this
            stars_spin_align    = self._find_spin(trimmed_data['stars'])
            _ , matrix          = self._orientate('z', stars_spin_align)
            
            # Orientate entire galaxy according to matrix above
            for parttype_name in data_nil.keys():
                # Align if there exists particle data, if not set same as before
                if len(data_nil[parttype_name]['Mass']) > 0:
                    data_nil[parttype_name] = self._rotate_galaxy(matrix, data_nil[parttype_name])
            
        
        #-----------------------------------------------------
        # Create masks for starforming and non-starforming gas
        if print_progress:
            print('Masking gas_sf and gas_nsf')
            time_start = time.time()
        mask_sf        = np.nonzero(data_nil['gas']['StarFormationRate'])          
        mask_nsf       = np.where(data_nil['gas']['StarFormationRate'] == 0)
        
        # Create dataset of star-forming and non-star-forming gas
        gas_sf = {}
        gas_nsf = {}
        for arr in data_nil['gas'].keys():
            gas_sf[arr]  = data_nil['gas'][arr][mask_sf]
            gas_nsf[arr] = data_nil['gas'][arr][mask_nsf]
        
        for parttype, parttype_name in zip([gas_sf, gas_nsf], ['gas_sf', 'gas_nsf']):
            data_nil['%s'%parttype_name] = parttype
            
        
        #=====================================================
        # Extracting MorphoKinem query data 
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Subhalo Morph query')
            time_start = time.time()
          
        MorphoKinem = {}    
        for arr_name, arr_value in zip(['ellip', 'triax', 'kappa_stars', 'disp_ani', 'disc_to_total', 'rot_to_disp_ratio'], MorphoKinem_in):
            MorphoKinem[arr_name] = arr_value
            
        #-----------------------------------------------------
        # Creating empty dictionaries
        self.general            = {}
        self.counts             = {}
        self.masses             = {}
        self.sfr                = {}
        self.Z                  = {}
        self.coms               = {}
        self.coms_proj          = {}
        self.inc_angles         = {}
        self.spins              = {}
        self.mis_angles         = {}
        self.mis_angles_proj    = {}
        self.gas_data           = {}
        self.mass_flow          = {}
        
        # Array to note if galaxy fails any extraction based on filters. Appends math.nan in dictionary, and radius at which it fails in .flags
        self.flags = {'total_particles':  {'stars': [], 'gas': [], 'gas_sf': [], 'gas_nsf': [], 'dm': [], 'bh':[]},      # For finding total particle counts       
                      'min_particles':    {'stars': [], 'gas': [], 'gas_sf': [], 'gas_nsf': [], 'dm': []},               # For finding min. spin particles
                      'min_inclination':  {'stars': [], 'gas': [], 'gas_sf': [], 'gas_nsf': [], 'dm': []}}               # For finding min. inclination
        flag_com_min_distance = {}
        for angle_selection_i in angle_selection:
            flag_com_min_distance.update({'%s' %(angle_selection_i): []})
        self.flags.update({'com_min_distance': flag_com_min_distance})               # For flagging min. com distance
      
        #----------------------------------------------------
        # Creating dictionary of trimmed data to aperture from which we work
        # data_nil has untrimmed data
        # self.data has trimmed to aperture
        self.data = self._trim_data(data_nil, aperture_rad)
        if debug:
            print(max(np.linalg.norm(data_nil['gas']['Coordinates'], axis=1)))
            print(max(np.linalg.norm(self.data['gas']['Coordinates'], axis=1)))
        
        
        #----------------------------------------------------
        # Assigning bulk galaxy values
        self.GroupNum           = GroupNum
        self.SubGroupNum        = SubGroupNum
        self.GalaxyID           = GalaxyID
        self.SnapNum            = SnapNum
        self.halo_mass          = halo_mass_in                          # [Msun] at 200p_crit
        self.stelmass           = np.sum(self.data['stars']['Mass'])     # [Msun] within 30 pkpc (aperture_rad_in)
        self.gasmass            = np.sum(self.data['gas']['Mass'])       # [Msun] within 30 pkpc (aperture_rad_in)
        self.gasmass_sf         = np.sum(self.data['gas_sf']['Mass'])    # [Msun] within 30 pkpc (aperture_rad_in)
        self.gasmass_nsf        = np.sum(self.data['gas_nsf']['Mass'])   # [Msun] within 30 pkpc (aperture_rad_in)
        self.dmmass             = np.sum(self.data['dm']['Mass'])        # [Msun] within 30 pkpc (aperture_rad_in)
        self.halfmass_rad       = halfmass_rad                          # [pkpc]
        self.halfmass_rad_proj  = halfmass_rad_proj                     # [pkpc]
        self.viewing_axis       = viewing_axis                          # 'x', 'y', 'z'
        self.viewing_angle      = viewing_angle                         # [deg]
        
        self.bh_id, self.bh_mass, self.bh_mdot, self.bh_edd = self._bh_accretion(self.data['bh'], halfmass_rad)     # [Msun]/s of largest BH within 0.5 HMR
        
        
        #----------------------------------------------------
        # Filling self.general
        for general_name, general_item in zip(['GroupNum', 'SubGroupNum', 'GalaxyID', 'SnapNum', 'halo_mass', 'stelmass', 'gasmass', 'gasmass_sf', 'gasmass_nsf', 'dmmass', 'bh_id', 'bh_mass', 'bh_mdot', 'bh_edd', 'halfmass_rad', 'halfmass_rad_proj', 'viewing_axis'], 
                                              [self.GroupNum, self.SubGroupNum, self.GalaxyID, self.SnapNum, self.halo_mass, self.stelmass, self.gasmass, self.gasmass_sf, self.gasmass_nsf, self.dmmass, self.bh_id, self.bh_mass, self.bh_mdot, self.bh_edd, self.halfmass_rad, self.halfmass_rad_proj, self.viewing_axis]):
            self.general[general_name] = general_item
        self.general.update(MorphoKinem)
            
        
        
        #====================================================
        # From here we only calculate particle properties that are linked to misalignment angles requested
        
        particle_selection = []         #particle_list_in = []
        compound_selection = []         #angle_selection  = []
        if 'stars_gas' in angle_selection:
            if 'stars' not in particle_selection:
                particle_selection.append('stars')
            if 'gas' not in particle_selection:
                particle_selection.append('gas')
            compound_selection.append(['stars', 'gas'])
        if 'stars_gas_sf' in angle_selection:
            if 'stars' not in particle_selection:
                particle_selection.append('stars')
            if 'gas_sf' not in particle_selection:
                particle_selection.append('gas_sf')
            compound_selection.append(['stars', 'gas_sf'])
        if 'stars_gas_nsf' in angle_selection:
            if 'stars' not in particle_selection:
                particle_selection.append('stars')
            if 'gas_nsf' not in particle_selection:
                particle_selection.append('gas_nsf')
            compound_selection.append(['stars', 'gas_nsf'])
        if 'gas_sf_gas_nsf' in angle_selection:
            if 'gas_sf' not in particle_selection:
                particle_selection.append('gas_sf')
            if 'gas_nsf' not in particle_selection:
                particle_selection.append('gas_nsf')
            compound_selection.append(['gas_sf', 'gas_nsf'])
        if 'stars_dm' in angle_selection:
            if 'stars' not in particle_selection:
                particle_selection.append('stars')
            if 'dm' not in particle_selection:
                particle_selection.append('dm')
            compound_selection.append(['stars', 'dm'])
        if 'gas_dm' in angle_selection:
            if 'gas' not in particle_selection:
                particle_selection.append('gas')
            if 'dm' not in particle_selection:
                particle_selection.append('dm')
            compound_selection.append(['gas', 'dm'])
        if 'gas_sf_dm' in angle_selection:
            if 'gas_sf' not in particle_selection:
                particle_selection.append('gas_sf')
            if 'dm' not in particle_selection:
                particle_selection.append('dm')
            compound_selection.append(['gas_sf', 'dm'])
        if 'gas_nsf_dm' in angle_selection:
            if 'gas_nsf' not in particle_selection:
                particle_selection.append('gas_nsf')
            if 'dm' not in particle_selection:
                particle_selection.append('dm')
            compound_selection.append(['gas_nsf', 'dm'])
        
        
        #-------------------------------
        # Flag galaxy 'total_particles' if it contains no particles of a specific kind
        def total_particles_flag():
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Flagging total particles')
                time_start = time.time()
            
            for particle_type_i in ['stars', 'gas', 'gas_nsf', 'gas_sf', 'dm', 'bh']:
                if len(self.data[particle_type_i]['Mass']) == 0:
                    self.flags['total_particles'][particle_type_i].append(aperture_rad)
        total_particles_flag()
        
        
        #------------------------------
        # Find aligned spin vectors and particle count within radius. Will calculate all properties assuming 1 particle
        # Will also find spins and the likes
        #-------------------------
        if find_uncertainties:
            # 1-sigma
            use_percentiles = 16
            iterations = 500 
            
            spins_rand = {}
            
            ''' structure
            Will be structured as:
            spins_rand['2.0']['stars'] .. from then [[ x, y, z], [x, y, z], ... ]
            '''
        
        #-----------------------------
        # Also extract gas data
        def find_particle_properties(debug=False):
            
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Finding counts, masses, spins, COMs')
                time_start = time.time()
                
            # Create dictionary arrays    
            self.coms['adjust'] = []
            for dict_list in [self.spins, self.counts, self.masses, self.coms]:
                dict_list['rad'] = spin_rad   
                dict_list['hmr'] = spin_hmr
                
                # Create radial distributions of all components, assume dm = 30 pkpc only
                for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf', 'dm']:
                    dict_list[parttype_name] = []
                    
            for dict_list in [self.sfr]:
                dict_list['rad'] = spin_rad   
                dict_list['hmr'] = spin_hmr
                dict_list['gas_sf'] = []
            
            for dict_list in [self.Z]:
                dict_list['rad'] = spin_rad   
                dict_list['hmr'] = spin_hmr
                
                # Create radial distributions of all components
                for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf']:
                    dict_list[parttype_name] = []     
            
            #-----------------------------
            # Baryonic matter that is calculated over several radii
            for rad_i, hmr_i in zip(spin_rad, spin_hmr):
                # Trim data to particular radius
                trimmed_data = self._trim_data(self.data, rad_i)
                
                # Find peculiar velocity of trimmed data
                #pec_vel_rad = self._peculiar_velocity(trimmed_data)
                if len(trimmed_data['stars']['Mass']) > 0:
                    pec_vel_rad = self._peculiar_velocity_part(trimmed_data['stars'])
                    stellar_com = self._centre_of_mass(trimmed_data['stars'])
                    self.coms['adjust'].append(stellar_com)
                else:
                    pec_vel_rad = np.array([math.nan, math.nan, math.nan])
                    stellar_com = np.array([math.nan, math.nan, math.nan])
                    self.coms['adjust'].append(stellar_com)
                
                # Adjust velocity of trimmed_data to account for peculiar velocity
                for parttype_name in trimmed_data.keys():
                    if len(trimmed_data[parttype_name]['Mass']) != 0:
                        trimmed_data[parttype_name]['Velocity']     = trimmed_data[parttype_name]['Velocity'] - pec_vel_rad
                        trimmed_data[parttype_name]['Coordinates']  = trimmed_data[parttype_name]['Coordinates'] - stellar_com 
                        trimmed_data['dm']['Velocity']      = self.data['dm']['Velocity'] - pec_vel_rad
                        trimmed_data['dm']['Coordinates']   = self.data['dm']['Coordinates'] - stellar_com    
                        trimmed_data['dm']['Mass']          = self.data['dm']['Mass']                 
                if debug:
                    print('Stellar peculiar velocity in rad', rad_i)
                    print(pec_vel_rad)
                    print('Stellar COM')
                    print(stellar_com)
                    print('Gas_sf in rad: ', rad_i)
                    print(len(trimmed_data['gas_sf']['Mass']))
                
                
                #===========================
                # Create gas_data
                self.gas_data.update({'%s_hmr' %hmr_i: {'gas': {}, 'gas_sf': {}, 'gas_nsf': {}}}) 
                
                
                #------------------------------
                for parttype_name in ['stars', 'gas', 'gas_sf', 'gas_nsf', 'dm']:
                    
                    #----------------------------------------------
                    # Find particle counts, masses
                    particle_count  = len(trimmed_data[parttype_name]['Mass'])
                    particle_mass   = np.sum(trimmed_data[parttype_name]['Mass'])
                    
                    # For stars and gas, find metallicity
                    if parttype_name != 'dm':
                        if particle_count == 0:
                            particle_Z = math.nan
                        else:
                            particle_Z = np.sum(trimmed_data[parttype_name]['Mass'] * trimmed_data[parttype_name]['Metallicity']) / np.sum(trimmed_data[parttype_name]['Mass'])
                    
                    # For gas, find star-formation rate (ssfr can be calculated with particle_sfr / particle_mass(stars))
                    if parttype_name == 'gas_sf':
                        particle_sfr = np.sum(trimmed_data[parttype_name]['StarFormationRate'])
                    
                    
                    if parttype_name == 'dm':
                        self.counts[parttype_name] = particle_count
                        self.masses[parttype_name] = particle_mass
                    else:
                        self.counts[parttype_name].append(particle_count)
                        self.masses[parttype_name].append(particle_mass)
                        self.Z[parttype_name].append(particle_Z)
                        if parttype_name == 'gas_sf':
                            self.sfr[parttype_name].append(particle_sfr)
                    
                    
                    #===========================
                    # Create dictionary of bare minimum gas_data for inflow/outflow
                    if (parttype_name == 'gas' ) or (parttype_name == 'gas_sf' ) or (parttype_name == 'gas_nsf' ):
                        self.gas_data['%s_hmr' %hmr_i][parttype_name]['ParticleIDs'] = trimmed_data[parttype_name]['ParticleIDs']
                        self.gas_data['%s_hmr' %hmr_i][parttype_name]['Mass']        = trimmed_data[parttype_name]['Mass']
                        self.gas_data['%s_hmr' %hmr_i][parttype_name]['Metallicity'] = trimmed_data[parttype_name]['Metallicity']
                    
                
                    #----------------------------------------------
                    # Find COMs (append math.nan for no particles)
                    if particle_count > 0:
                        particle_com = self._centre_of_mass(trimmed_data[parttype_name])
                        if parttype_name == 'dm':
                            self.coms[parttype_name] = particle_com
                        else:
                            self.coms[parttype_name].append(particle_com)
                    else:
                        particle_com = np.array([math.nan, math.nan, math.nan])
                        if parttype_name == 'dm':
                            
                            self.coms[parttype_name] = particle_com
                        else:
                            self.coms[parttype_name].append(particle_com)
                    
                    #---------------------------------------------
                    # Find unit vector spins (append math.nan for no particles)
                    if particle_count > 1:
                        particle_spin = self._find_spin(trimmed_data[parttype_name])
                        self.spins[parttype_name].append(particle_spin)
                    else:
                        particle_spin = np.array([math.nan, math.nan, math.nan])
                        self.spins[parttype_name].append(particle_spin)
                    
                    #---------------------------------------------
                    # Flag for minimum particles, but do nothing
                    if particle_count < min_particles:
                        if parttype_name == 'dm':
                            self.flags['min_particles'][parttype_name].append(aperture_rad)
                        else:
                            self.flags['min_particles'][parttype_name].append(hmr_i)
                    
                    #---------------------------------------------
                    # Flag for inclination angle, but do nothing
                    if viewing_axis == 'x':  
                        angle_test = self._misalignment_angle([1, 0, 0], particle_spin)
                        if (angle_test < min_inclination) or (angle_test > 180-min_inclination):
                            if parttype_name == 'dm':
                                self.flags['min_inclination'][parttype_name].append(aperture_rad)
                            else:
                                self.flags['min_inclination'][parttype_name].append(hmr_i)
                    if viewing_axis == 'y':
                        angle_test = self._misalignment_angle([0, 1, 0], particle_spin)
                        if (angle_test < min_inclination) or (angle_test > 180-min_inclination):
                            if parttype_name == 'dm':
                                self.flags['min_inclination'][parttype_name].append(aperture_rad)
                            else:
                                self.flags['min_inclination'][parttype_name].append(hmr_i)
                    if viewing_axis == 'z':
                        angle_test = self._misalignment_angle([0, 0, 1], particle_spin)
                        if (angle_test < min_inclination) or (angle_test > 180-min_inclination):
                            if parttype_name == 'dm':
                                self.flags['min_inclination'][parttype_name].append(aperture_rad)
                            else:
                                self.flags['min_inclination'][parttype_name].append(hmr_i)
        
                #------------------------------
                # Find uncertainties
                if find_uncertainties:
                    if print_progress:
                        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                        print('Creating spins_rand')
                        time_start = time.time()
                        
                    spins_rand.update({'%s' %str(hmr_i): {}})
                    
                    # for each particle type...
                    for parttype_name in particle_selection:
                        tmp_spins = [] 
                
                        # Create random array of spins if min_particles met
                        if len(trimmed_data[parttype_name]['Mass']) >= min_particles:
                            for j in range(iterations):
                                spin_i = self._find_spin(trimmed_data[parttype_name], random_sample=True)
                                tmp_spins.append(spin_i)
                        # Else create [[nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]
                        else:
                            tmp_spins = np.full((3, 3), math.nan)
                        spins_rand['%s' %str(hmr_i)][parttype_name] = np.stack(tmp_spins)          
        find_particle_properties()

        #-----------------------------
        def find_inflow_outflow(gas_data_old, debug=False):
            # if gas data not given, set stuff to math.nan
            if gas_data_old == False:
                for hmr_name_i in self.gas_data.keys():
                    self.mass_flow.update({'%s' %hmr_name_i: {}})
                    
                    for parttype_name in self.gas_data[hmr_name_i].keys():
                        self.mass_flow[hmr_name_i][parttype_name] = {'inflow': math.nan,
                                                                     'outflow': math.nan,
                                                                     'massloss': math.nan,
                                                                     'inflow_Z': math.nan,
                                                                     'outflow_Z': math.nan,
                                                                     'insitu_Z': math.nan}
                
            else:
                # gas_data has current galaxy's gas data, gas_data_old has predecessor's gas data
                if print_progress:
                    print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                    print('Finding inflow/outflow')
                    time_start = time.time()
            
                
                for hmr_name_i in self.gas_data.keys():
                    self.mass_flow.update({'%s' %hmr_name_i: {}})
                
                    for parttype_name in self.gas_data[hmr_name_i].keys():
                                        
                        # Cycle through current IDs, to check if they were in _old
                        if hmr_name_i in gas_data_old.keys():
                            previous_mass = np.sum(gas_data_old[hmr_name_i][parttype_name]['Mass'])       # M1
                        else:
                            previous_mass = math.nan
                            continue
                        current_mass  = np.sum(self.gas_data[hmr_name_i][parttype_name]['Mass'])           # M2
                        inflow_mass   = 0
                        outflow_mass  = 0
                        insitu_mass   = 0
                    
                        #------------------
                        # Check for inflow (use current, run check on previous)
                        inflow_mass_metal = 0
                        insitu_mass_metal = 0
                        for ID_i, mass_i, metal_i in zip(self.gas_data[hmr_name_i][parttype_name]['ParticleIDs'], self.gas_data[hmr_name_i][parttype_name]['Mass'], self.gas_data[hmr_name_i][parttype_name]['Metallicity']):
                        
                            # If ID was within 2hmr of _old, gas particle stayed
                            if ID_i in gas_data_old[hmr_name_i][parttype_name]['ParticleIDs']:
                                insitu_mass       = insitu_mass + mass_i
                                insitu_mass_metal = insitu_mass_metal + (mass_i * metal_i)
                                continue
                            # If ID was NOT within 2hmr of _old, gas particle was accreted
                            else:
                                inflow_mass       = inflow_mass + mass_i
                                inflow_mass_metal = inflow_mass_metal + (mass_i * metal_i)
                        
                        #------------------
                        # Find metallicity of inflow
                        if inflow_mass != 0:
                            inflow_Z = inflow_mass_metal / inflow_mass
                        else:
                            inflow_Z = math.nan
                            
                        # Find metallicity of insitu
                        if insitu_mass != 0:
                            insitu_Z = insitu_mass_metal / insitu_mass
                        else:
                            insitu_Z = math.nan
                            
                        #------------------
                        # Check for outflow (use old, run check on current)
                        outflow_mass_metal = 0
                        for ID_i, mass_i, metal_i in zip(gas_data_old[hmr_name_i][parttype_name]['ParticleIDs'], gas_data_old[hmr_name_i][parttype_name]['Mass'], gas_data_old[hmr_name_i][parttype_name]['Metallicity']):
                        
                            # If ID will be within 2hmr of current, gas particle stayed
                            if ID_i in self.gas_data[hmr_name_i][parttype_name]['ParticleIDs']:
                                continue
                            # If ID will NOT be within 2hmr of current, gas particle was outflowed
                            else:
                                outflow_mass       = outflow_mass + mass_i
                                outflow_mass_metal = outflow_mass_metal + (mass_i * metal_i)
                
                        # Find metallicity of outflow
                        if outflow_mass != 0:
                            outflow_Z = outflow_mass_metal / outflow_mass
                        else:
                            outflow_Z = math.nan
                    
                        #------------------
                        # Left with current_mass = previous_mass + inflow_mass - outflow_mass + stellarmassloss
                        stellarmassloss = current_mass - previous_mass - inflow_mass + outflow_mass
                    
                        # Update mass_flow
                        self.mass_flow[hmr_name_i][parttype_name] = {'inflow': inflow_mass,
                                                                     'outflow': outflow_mass,
                                                                     'massloss': stellarmassloss,
                                                                     'inflow_Z': inflow_Z,
                                                                     'outflow_Z': outflow_Z,
                                                                     'insitu_Z': insitu_Z}
        find_inflow_outflow(gas_data_old)
        
        #-----------------------------
        # Finding COM distances and flagging if one is > com_min_distance
        def flag_coms(debug=False):
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Flagging C.o.M in radius')
                time_start = time.time()
            
            # DM handled as individual number self.coms['dm'] = [0, 0, 0]
            com_dm = self.coms['dm']
            for hmr_i, com_star, com_gas, com_gas_sf, com_gas_nsf in zip(self.coms['hmr'], self.coms['stars'], self.coms['gas'], self.coms['gas_sf'], self.coms['gas_nsf']):
                
                if 'stars_gas' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_star - com_gas) > com_min_distance:
                        self.flags['com_min_distance']['stars_gas'].append(hmr_i)
                if 'stars_gas_sf' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_star - com_gas_sf) > com_min_distance:
                        self.flags['com_min_distance']['stars_gas_sf'].append(hmr_i)
                if 'stars_gas_nsf' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_star - com_gas_nsf) > com_min_distance:
                        self.flags['com_min_distance']['stars_gas_nsf'].append(hmr_i)
                if 'gas_sf_gas_nsf' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_gas_sf - com_gas_nsf) > com_min_distance:
                        self.flags['com_min_distance']['gas_sf_gas_nsf'].append(hmr_i)
                if 'stars_dm' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_star - com_dm) > com_min_distance:
                        self.flags['com_min_distance']['stars_dm'].append(hmr_i)
                if 'gas_dm' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_gas - com_dm) > com_min_distance:
                        self.flags['com_min_distance']['gas_dm'].append(hmr_i)
                if 'gas_sf_dm' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_gas_sf - com_dm) > com_min_distance:
                        self.flags['com_min_distance']['gas_sf_dm'].append(hmr_i)
                if 'gas_nsf_dm' in angle_selection:
                    # This will be a math.nan if no particles present
                    if np.linalg.norm(com_gas_nsf - com_dm) > com_min_distance:
                        self.flags['com_min_distance']['gas_nsf_dm'].append(hmr_i)   
        flag_coms()
        
        #-----------------------------
        # Finding 3D misalignment angles and uncertainties
        def find_misalignment_angles(debug=False):
            
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Finding 3D misalignment angles')
                time_start = time.time()
                
            # Create arrays
            for dict_list in [self.mis_angles]:
                dict_list['rad'] = spin_rad   
                dict_list['hmr'] = spin_hmr
                
                # Create radial distributions of all components, assume dm = 30 pkpc only
                for angle_selection_i in angle_selection:
                    dict_list['%s_angle' %angle_selection_i]        = []
                    dict_list['%s_angle_err' %angle_selection_i]    = []
                
            # DM largely all the same but slight differences
            for rad_i, hmr_i, spin_star, spin_gas, spin_gas_sf, spin_gas_nsf, spin_dm in zip(self.spins['rad'], self.spins['hmr'], self.spins['stars'], self.spins['gas'], self.spins['gas_sf'], self.spins['gas_nsf'], self.spins['dm']):
                
                if 'stars_gas' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_star, spin_gas)
                    self.mis_angles['stars_gas_angle'].append(angle)            # Will be NAN if none
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'], spins_rand['%s' %hmr_i]['gas']):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['stars_gas_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['stars_gas_angle_err'].append([math.nan, math.nan])
                if 'stars_gas_sf' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_star, spin_gas_sf)
                    self.mis_angles['stars_gas_sf_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'], spins_rand['%s' %hmr_i]['gas_sf']):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['stars_gas_sf_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['stars_gas_sf_angle_err'].append([math.nan, math.nan])
                if 'stars_gas_nsf' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_star, spin_gas_nsf)
                    self.mis_angles['stars_gas_nsf_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'], spins_rand['%s' %hmr_i]['gas_nsf']):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['stars_gas_nsf_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['stars_gas_nsf_angle_err'].append([math.nan, math.nan])
                if 'gas_sf_gas_nsf' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_gas_sf, spin_gas_nsf)
                    self.mis_angles['gas_sf_gas_nsf_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas_sf'], spins_rand['%s' %hmr_i]['gas_nsf']):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['gas_sf_gas_nsf_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['gas_sf_gas_nsf_angle_err'].append([math.nan, math.nan])
                if 'stars_dm' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_star, spin_dm)
                    self.mis_angles['stars_dm_angle'].append(angle)
                    
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'], spins_rand['%s' %hmr_i]['dm']):                            
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['stars_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['stars_dm_angle_err'].append([math.nan, math.nan])
                if 'gas_dm' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_gas, spin_dm)
                    self.mis_angles['gas_dm_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas'], spins_rand['%s' %hmr_i]['dm']):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['gas_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['gas_dm_angle_err'].append([math.nan, math.nan])       
                if 'gas_sf_dm' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_gas_sf, spin_dm)
                    self.mis_angles['gas_sf_dm_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas_sf'], spins_rand['%s' %hmr_i]['dm']):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['gas_sf_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['gas_sf_dm_angle_err'].append([math.nan, math.nan])
                if 'gas_nsf_dm' in angle_selection:
                    # Find 3D angle
                    angle = self._misalignment_angle(spin_gas_nsf, spin_dm)
                    self.mis_angles['gas_nsf_dm_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas_nsf'], spins_rand['%s' %hmr_i]['dm']):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles['gas_nsf_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles['gas_nsf_dm_angle_err'].append([math.nan, math.nan])
        find_misalignment_angles()
        
        #-----------------------------
        # Find 2D projected misalignment angles + errors 
        def find_2d_misalignment_angles(debug=False):
            
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Finding 2D projected angles')
                time_start = time.time()
        
            self.mis_angles_proj = {'x': {}, 'y': {}, 'z': {}}
            for viewing_axis_i in ['x', 'y', 'z']:
                for dict_list in [self.mis_angles_proj]:
                    dict_list[viewing_axis]['rad'] = spin_rad   
                    dict_list[viewing_axis]['hmr'] = spin_hmr
                
                    # Create radial distributions of all components, assume dm = 30 pkpc only
                    for angle_selection_i in angle_selection:
                        dict_list[viewing_axis]['%s_angle' %angle_selection_i]        = []
                        dict_list[viewing_axis]['%s_angle_err' %angle_selection_i]    = []
                
            # Set indicies
            if viewing_axis == 'x':
                a = 1
                b = 2
            if viewing_axis == 'y':
                a = 0
                b = 2
            if viewing_axis == 'z':
                a = 0
                b = 1
                
            # DM largely the same but minor differences
            for rad_i, hmr_i, spin_star, spin_gas, spin_gas_sf, spin_gas_nsf, spin_dm in zip(self.spins['rad'], self.spins['hmr'], self.spins['stars'], self.spins['gas'], self.spins['gas_sf'], self.spins['gas_nsf'], self.spins['dm']):
                
                if 'stars_gas' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_star[a], spin_star[b]]), np.array([spin_gas[a], spin_gas[b]]))
                    self.mis_angles_proj[viewing_axis]['stars_gas_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'][:,[a, b]], spins_rand['%s' %hmr_i]['gas'][:,[a, b]]):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['stars_gas_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['stars_gas_angle_err'].append([math.nan, math.nan])
                if 'stars_gas_sf' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_star[a], spin_star[b]]), np.array([spin_gas_sf[a], spin_gas_sf[b]]))
                    self.mis_angles_proj[viewing_axis]['stars_gas_sf_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'][:,[a, b]], spins_rand['%s' %hmr_i]['gas_sf'][:,[a, b]]):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['stars_gas_sf_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['stars_gas_sf_angle_err'].append([math.nan, math.nan])
                if 'stars_gas_nsf' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_star[a], spin_star[b]]), np.array([spin_gas_nsf[a], spin_gas_nsf[b]]))
                    self.mis_angles_proj[viewing_axis]['stars_gas_nsf_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'][:,[a, b]], spins_rand['%s' %hmr_i]['gas_nsf'][:,[a, b]]):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['stars_gas_nsf_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['stars_gas_nsf_angle_err'].append([math.nan, math.nan])
                if 'gas_sf_gas_nsf' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_gas_sf[a], spin_gas_sf[b]]), np.array([spin_gas_nsf[a], spin_gas_nsf[b]]))
                    self.mis_angles_proj[viewing_axis]['gas_sf_gas_nsf_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas_sf'][:,[a, b]], spins_rand['%s' %hmr_i]['gas_nsf'][:,[a, b]]):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['gas_sf_gas_nsf_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['gas_sf_gas_nsf_angle_err'].append([math.nan, math.nan])
                if 'stars_dm' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_star[a], spin_star[b]]), np.array([spin_dm[a], spin_dm[b]]))
                    self.mis_angles_proj[viewing_axis]['stars_dm_angle'].append(angle)
                    
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['stars'][:,[a, b]], spins_rand['%s' %hmr_i]['dm'][:,[a, b]]):                            
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['stars_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['stars_dm_angle_err'].append([math.nan, math.nan])
                if 'gas_dm' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_gas[a], spin_gas[b]]), np.array([spin_dm[a], spin_dm[b]]))
                    self.mis_angles_proj[viewing_axis]['gas_dm_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas'][:,[a, b]], spins_rand['%s' %hmr_i]['dm'][:,[a, b]]):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['gas_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['gas_dm_angle_err'].append([math.nan, math.nan])       
                if 'gas_sf_dm' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_gas_sf[a], spin_gas_sf[b]]), np.array([spin_dm[a], spin_dm[b]]))
                    self.mis_angles_proj[viewing_axis]['gas_sf_dm_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas_sf'][:,[a, b]], spins_rand['%s' %hmr_i]['dm'][:,[a, b]]):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['gas_sf_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['gas_sf_dm_angle_err'].append([math.nan, math.nan])
                if 'gas_nsf_dm' in angle_selection:
                    # Find 2D angle
                    angle = self._misalignment_angle(np.array([spin_gas_nsf[a], spin_gas_nsf[b]]), np.array([spin_dm[a], spin_dm[b]]))
                    self.mis_angles_proj[viewing_axis]['gas_nsf_dm_angle'].append(angle)
                
                    if find_uncertainties:
                        # uncertainty
                        tmp_errors_array = []
                        for spin_1, spin_2 in zip(spins_rand['%s' %hmr_i]['gas_nsf'][:,[a, b]], spins_rand['%s' %hmr_i]['dm'][:,[a, b]]):
                            tmp_errors_array.append(self._misalignment_angle(spin_1, spin_2))
                        self.mis_angles_proj[viewing_axis]['gas_nsf_dm_angle_err'].append(np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            
                        if debug:
                            print('\nHMR_i ', hmr_i)
                            print('percentiles: ', np.percentile(tmp_errors_array, [use_percentiles, 100-use_percentiles]))
                            print('angle: ', angle)
                            plt.hist(tmp_errors_array, 100)
                            plt.show()
                            plt.close()
                    else:
                        self.mis_angles_proj[viewing_axis]['gas_nsf_dm_angle_err'].append([math.nan, math.nan])
        find_2d_misalignment_angles()    
        
        #-----------------------------
        # KAPPA
        def find_kappa():
            
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Finding kappa of components')
                time_start = time.time()
            
            # If we already found kappa_star, don't run stars kappa
            if not math.isnan(MorphoKinem['kappa_stars']):
                kappa_parttype = ['gas', 'gas_sf']
            else:
                kappa_parttype = ['stars', 'gas', 'gas_sf']
            
            # Trim data to kappa radius
            trimmed_data = self._trim_data(data_nil, kappa_rad)
            
            # Find peculiar velocity of trimmed data
            pec_vel_rad = self._peculiar_velocity(trimmed_data)

            # Adjust velocity of trimmed_data to account for peculiar velocity
            for parttype_name in trimmed_data.keys():
                if len(trimmed_data[parttype_name]['Mass']) == 0:
                    continue
                else:
                    trimmed_data[parttype_name]['Velocity'] = trimmed_data[parttype_name]['Velocity'] - pec_vel_rad
            if debug:
                print('Peculiar velocity in rad', kappa_rad)
                print(pec_vel_rad)
                print('Gas_sf in rad: ', kappa_rad)
                print(len(trimmed_data['gas_sf']['Mass']))
            
            #-------------------------------    
            for parttype_name in kappa_parttype:
                
                # If particles of type exist within aperture_rad... find kappas
                if len(trimmed_data[parttype_name]['Mass']) > 0:
                    # Finding spin vector within aperture_rad for current kappa
                    spin_kappa = self._find_spin(trimmed_data[parttype_name])
                    _ , matrix = self._orientate(orientate_to_axis, spin_kappa)
                
                    # Orientate entire galaxy according to matrix above, use this to find kappa
                    aligned_data_part = self._rotate_galaxy(matrix, trimmed_data[parttype_name])
                    kappa = self._kappa_co(aligned_data_part, kappa_rad)
                
                    self.general.update({'kappa_%s' %parttype_name: kappa})
                # If no particles of type exist within aperture_rad, assign math.nan
                else:
                    self.general.update({'kappa_%s' %parttype_name: math.nan})   
        find_kappa()
        
        #-----------------------------
        # Trimming data
        def trim_output_data():
            if print_progress:
                print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
                print('Trimming datasets to trim_hmr ', trim_hmr)
            
            tmp_data = {}
            for rad_i in trim_rad:
                tmp_data.update({'%s' %str(rad_i): {}})
            
            # trim_rad = ['1.0_hmr', '2.0_hmr', '30']
            for rad_i in trim_rad:
                
                # if trim_rad is '1HMR_proj':
                if (rad_i == '1.0_hmr') or (rad_i == '2.0_hmr'):
                    # Radius to trim to
                    if (rad_i == '1.0_hmr') and (rad_projected == True):
                        trim_to_rad = 1*halfmass_rad_proj
                    elif (rad_i == '1.0_hmr') and (rad_projected == False):
                        trim_to_rad = 1*halfmass_rad
                    elif (rad_i == '2.0_hmr') and (rad_projected == True):
                        trim_to_rad = 2*halfmass_rad_proj
                    elif (rad_i == '2.0_hmr') and (rad_projected == False):
                        trim_to_rad = 2*halfmass_rad
                    
                    
                    # Trim data to particular radius
                    trimmed_data = self._trim_data(data_nil, trim_to_rad)
                    
                    # Find peculiar velocity of trimmed data
                    pec_vel_rad = self._peculiar_velocity(trimmed_data)
                
                    # Adjust velocity of trimmed_data to account for peculiar velocity
                    for parttype_name in trimmed_data.keys():
                        if len(trimmed_data[parttype_name]['Mass']) == 0:
                            continue
                        else:
                            trimmed_data[parttype_name]['Velocity'] = trimmed_data[parttype_name]['Velocity'] - pec_vel_rad
                    if debug:
                        print('Peculiar velocity in rad', rad_i)
                        print(pec_vel_rad)
                        print('Gas_sf in rad: ', rad_i)
                        print(len(trimmed_data['gas_sf']['Mass']))
                    
                    for parttype_name in trimmed_data.keys():
                        tmp_data['%s' %str(rad_i)][parttype_name] = trimmed_data[parttype_name]
                
                else:
                    # Trim data to particular radius
                    trimmed_data = self._trim_data(data_nil, float(rad_i))
                    
                    # Find peculiar velocity of trimmed data
                    pec_vel_rad = self._peculiar_velocity(trimmed_data)
                
                    # Adjust velocity of trimmed_data to account for peculiar velocity
                    for parttype_name in trimmed_data.keys():
                        if len(trimmed_data[parttype_name]['Mass']) == 0:
                            continue
                        else:
                            trimmed_data[parttype_name]['Velocity'] = trimmed_data[parttype_name]['Velocity'] - pec_vel_rad
                    if debug:
                        print('Peculiar velocity in rad', rad_i)
                        print(pec_vel_rad)
                        print('Gas_sf in rad: ', rad_i)
                        print(len(trimmed_data['gas_sf']['Mass']))
                
                    for parttype_name in trimmed_data.keys():
                        tmp_data['%s' %str(rad_i)][parttype_name] = trimmed_data[parttype_name]   
            self.data = tmp_data
        trim_output_data()

        #-----------------------------
        # Print statements
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('FINISHED EXTRACTION\tGALAXY ID: %s' %GalaxyID)
                    
    def _rotate_galaxy(self, matrix, data, debug=False):
        """ For a given set of galaxy data, work out the rotated coordinates 
        and other data centred on [0, 0, 0], accounting for the peculiar 
        velocity of the galaxy"""
        
        # Where we store the new data
        new_data = {}
        
        for header in data.keys():
            if (header == 'Coordinates') or (header == 'Velocity'):
                new_data[header] = self._rotate_coords(matrix, data[header])
            else:
                new_data[header] = data[header]
        
        return new_data
    
    def _rotate_around_axis(self, axis, angle, debug=False):
        # Finds angle given a defined x, y, z axis:
        if axis == 'z':
            # Rotate around z-axis
            axis_of_rotation = np.array([0., 0., 1.])
            matrix = self._rotation_matrix(axis_of_rotation, angle)
            
        if axis == 'y':
            # Rotate around z-axis
            axis_of_rotation = np.array([0., 1., 0.])
            matrix = self._rotation_matrix(axis_of_rotation, angle)

        if axis == 'x':
            # Rotate around z-axis
            axis_of_rotation = np.array([1., 0., 0.])
            matrix = self._rotation_matrix(axis_of_rotation, angle)

        return matrix
    
    def _rotation_matrix(self, axis, theta, debug=False):
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
    
    def _rotate_coords(self, matrix, coords, debug=False):
        
        # Compute new coords after rotation
        rotation = []
        for coord in coords:
            rotation.append(np.dot(matrix, coord))
        
        if len(rotation) > 0:
            rotation = np.stack(rotation)
        
        return rotation
        
    def _trim_data(self, data_dict, radius, debug=False):
        new_dict = {}
        for parttype_name in data_dict.keys():
            # Compute distance to centre and mask all within 3D radius in pkpc
            r  = np.linalg.norm(data_dict[parttype_name]['Coordinates'], axis=1)
            if debug:
                print('r, radius', r, radius)
            mask = np.where(r <= radius)
            
            newData = {}
            for header in data_dict[parttype_name].keys():
                newData[header] = data_dict[parttype_name][header][mask]
            
            new_dict[parttype_name] = newData
        
        return new_dict
    
    def _trim_data_proj(self, data_dict, radius, viewing_axis, debug=False):
        new_dict = {}
        for parttype_name in data_dict.keys():    
            # Compute distance to centre and mask all within projected radius in pkpc
            if viewing_axis == 'z':
                r = np.linalg.norm(data_dict[parttype_name]['Coordinates'][:,[0,1]], axis=1)
                mask = np.where(r <= radius)
            
            if viewing_axis == 'y':
                r = np.linalg.norm(data_dict[parttype_name]['Coordinates'][:,[0,2]], axis=1)
                mask = np.where(r <= radius)
            
            if viewing_axis == 'x':
                r = np.linalg.norm(data_dict[parttype_name]['Coordinates'][:,[1,2]], axis=1)
                mask = np.where(r <= radius)
            
            newData = {}
            for header in data_dict[parttype_name].keys():
                newData[header] = data_dict[parttype_name][header][mask]
            
            new_dict[parttype_name] = newData
        
        return new_dict
    
    def _peculiar_velocity(self, data_dict, debug=False):   
        # Mass-weighted formula from subhalo paper but accounting for radius
        vel_weighted    = 0
        mass_sums       = 0
        
        # Don't use gas_sf, and gas_nsf
        for parttype_name in ['stars', 'gas', 'dm', 'bh']:
            if len(data_dict[parttype_name]['Mass']) > 0:
                # Both start at 0, will only append if there are particles to use
                vel_weighted = vel_weighted + np.sum(data_dict[parttype_name]['Velocity'] * data_dict[parttype_name]['Mass'][:, None], axis=0)
                mass_sums = mass_sums + np.sum(data_dict[parttype_name]['Mass'])
                
        pec_vel_rad = vel_weighted / mass_sums
            
        return pec_vel_rad        
    
    def _peculiar_velocity_part(self, arr, debug=False):        
        # Mass-weighted formula from subhalo paper
        vel_weighted = np.sum(arr['Velocity'] * arr['Mass'][:, None], axis=0)
        mass_sums = np.sum(arr['Mass'])
            
        pec_vel = vel_weighted / mass_sums
            
        return pec_vel
    
    def _centre_of_mass(self, arr, debug=False):
        # Mass-weighted formula from subhalo paper
        mass_weighted  = arr['Coordinates'] * arr['Mass'][:, None]
        centre_of_mass = np.sum(mass_weighted, axis=0)/np.sum(arr['Mass'])
    
        return centre_of_mass
    
    def _find_spin(self, arr, random_sample=False, debug=False):
        
        # Choose 50% of particles to find spin
        if random_sample:
            # For ease of use
            tmp_coords = arr['Coordinates']
            tmp_mass = arr['Mass'][:, None]
            tmp_velocity = arr['Velocity']
            
            # Making a random mask of data points... 50% = 0.5
            random_mask = np.random.choice(tmp_coords.shape[0], int(np.ceil(tmp_coords.shape[0] * 0.5)), replace=False)
            
            # Applying mask
            tmp_coords = tmp_coords[random_mask]
            tmp_mass   = tmp_mass[random_mask]
            tmp_velocity = tmp_velocity[random_mask]
            
            if debug:
                print('\ntotal particle count', len(arr['Mass']))
                print('particle count in rad masked', len(tmp_mass))

            # Finding spin angular momentum vector of each particle
            L = np.cross(tmp_coords * tmp_mass, tmp_velocity)
            
            # Summing for total angular momentum and dividing by mass to get the spin vectors
            spin = np.sum(L, axis=0)/np.sum(tmp_mass)
        
        
        # Choose all particles to find spin   
        else:
            # Finding spin angular momentum vector of each individual particle of gas and stars, where [:, None] is done to allow multiplaction of N3*N1 array. Equation D.25
            L  = np.cross(arr['Coordinates'] * arr['Mass'][:, None], arr['Velocity'])
        
            if debug:
                print(arr['Mass'][:, None])
                print(np.sum(arr['Mass']))
            
            # Summing for total angular momentum and dividing by mass to get the spin vectors
            spin = np.sum(L, axis=0)/np.sum(arr['Mass'])
                
        # Expressing as unit vector
        spin_unit = spin / np.linalg.norm(spin)
        
        
        return spin_unit
        
    def _misalignment_angle(self, angle1, angle2, debug=False):
        # Find the misalignment angle
        angle = np.rad2deg(np.arccos(np.clip(np.dot(angle1/np.linalg.norm(angle1), angle2/np.linalg.norm(angle2)), -1.0, 1.0)))     # [deg]
        
        return angle

    def _kappa_co(self, arr, debug=False):        
        # Compute angular momentum within specified radius
        L  = np.cross(arr['Coordinates'] * arr['Mass'][:, None], arr['Velocity'])
        # Mask for co-rotating (L_z >= 0)
        L_mask = np.where(L[:,2] >= 0)
        
        # Projected radius along disk within specified radius
        rad_projected = np.linalg.norm(arr['Coordinates'][:,:2][L_mask], axis=1)
        
        # Kinetic energy of ordered co-rotation (using only angular momentum in z-axis)
        K_rot = np.sum(0.5 * np.square(L[:,2][L_mask] / rad_projected) / arr['Mass'][L_mask])
        
        # Total kinetic energy of stars
        K_tot = np.sum(0.5 * arr['Mass'] * np.square(np.linalg.norm(arr['Velocity'], axis=1)))
        
        return K_rot/K_tot    
        
    def _orientate(self, axis, att_spin, debug=False):
        # Finds angle given a defined x, y, z axis:
        if axis == 'z':
            # Compute angle between star spin and z axis
            angle = self._misalignment_angle(np.array([0., 0., 1.]), att_spin)
        
            # Find axis of rotation of star spin vector
            x  = np.array([0., 0., 1.])
            x -= x.dot(att_spin) * att_spin
            x /= np.linalg.norm(x)
            axis_of_rotation = np.cross(att_spin, x)
            matrix = self._rotation_matrix(axis_of_rotation, angle)
            
        if axis == 'y':
            # Compute angle between star spin and y axis
            angle = self._misalignment_angle(np.array([0., 1., 0.]), att_spin)
        
            # Find axis of rotation of star spin vector
            x  = np.array([0., 1., 0.])
            x -= x.dot(att_spin) * att_spin
            x /= np.linalg.norm(x)
            axis_of_rotation = np.cross(att_spin, x)
            matrix = self._rotation_matrix(axis_of_rotation, angle)
            
        if axis == 'x':
            # Compute angle between star spin and y axis
            angle = self._misalignment_angle(np.array([1., 0., 0.]), att_spin)
        
            # Find axis of rotation of star spin vector
            x  = np.array([1., 0., 0.])
            x -= x.dot(att_spin) * att_spin
            x /= np.linalg.norm(x)
            axis_of_rotation = np.cross(att_spin, x)
            matrix = self._rotation_matrix(axis_of_rotation, angle)
            
        return angle, matrix
    
    def _bh_accretion(self, data_dict, hmr, hmr_from_centre=1.0, debug=False):
        # Check if galaxy has BH:
        if len(data_dict['Coordinates']) == 0:
            return math.nan, math.nan, math.nan, math.nan
        
        # Check if galaxy has BH within 0.5 HMR (abs):
        else:
            r = np.linalg.norm(data_dict['Coordinates'], axis=1)
            mask = np.where(r <= hmr*hmr_from_centre)    
            
            if len(data_dict['Mass'][mask]) > 0:
                # Consider the accretion rate of the most massive BH within 0.5 hmr
                #r    = np.linalg.norm(data_dict['Coordinates'], axis=1).min()
                #mask = np.linalg.norm(data_dict['Coordinates'], axis=1).argmin()
                new_mask = data_dict['Mass'][mask].argmax()
        
                if debug:
                    print('distance', r)
                    print('bh list', data_dict['Mass'])
                    print('bh mdot', data_dict['BH_Mdot'])
        
        
                # Consider the accretion rate of the closest to the centre to be the BH of this galaxy
                bh_epsilon = 0.1        # efficiency
                accretion_rate = data_dict['BH_Mdot'][mask][new_mask]
                bh_mass        = data_dict['Mass'][mask][new_mask]             
                bh_edd         = accretion_rate / (bh_mass * 7e-17 / bh_epsilon)
                bh_id          = data_dict['ParticleIDs'][mask][new_mask]
                
                if debug:
                    print('accretion rate', accretion_rate)
            
                return bh_id, bh_mass, accretion_rate, bh_edd
            
            # if no BH within 0.5 HMR, pick next largest within 1 HMR
            elif len(data_dict['Mass'][mask]) == 0:
                r = np.linalg.norm(data_dict['Coordinates'], axis=1)
                mask = np.where(r <= 2* hmr*hmr_from_centre)
            
                if len(data_dict['Mass'][mask]) > 0:
                    # Consider the accretion rate of the most massive BH within 0.5 hmr
                    #r    = np.linalg.norm(data_dict['Coordinates'], axis=1).min()
                    #mask = np.linalg.norm(data_dict['Coordinates'], axis=1).argmin()
                    new_mask = data_dict['Mass'][mask].argmax()
                
        
                    if debug:
                        print('distance', r)
                        print('bh list', data_dict['Mass'])
                        print('bh mdot', data_dict['BH_Mdot'])
        
        
                    # Consider the accretion rate of the closest to the centre to be the BH of this galaxy
                    bh_epsilon = 0.1        # efficiency
                    accretion_rate = data_dict['BH_Mdot'][mask][new_mask]
                    bh_mass        = data_dict['Mass'][mask][new_mask]             
                    bh_edd         = accretion_rate / (bh_mass * 7e-17 / bh_epsilon)
                    bh_id          = data_dict['ParticleIDs'][mask][new_mask]
                
                    if debug:
                        print('accretion rate', accretion_rate)
            
                    return bh_id, bh_mass, accretion_rate, bh_edd
                
                else:
                    return math.nan, math.nan, math.nan, math.nan
        
            else:
                return math.nan, math.nan, math.nan, math.nan
              

""" 
Purpose
-------
Will extract basic properties for small galaxies which don't require full analysis

Calling function
----------------
galaxy = Subhalo_Extract(sample_input['mySims'], dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, GalaxyID, aperture_rad, viewing_axis)

Input Parameters
----------------

mySims:
    mySims = np.array([('RefL0012N0188', 12)])  
    Name and boxsize
dataDir:
    Location of the snapshot data hdf5 file, 
    eg. '/Users/c22048063/Documents/.../snapshot_028_xx/snap_028_xx.0.hdf5'
SnapNum: int
    Snapshot number, ei. 28
GroupNum: int
    GroupNumber of the subhalo, starts at 1
SubGroupNum: int
    SubGroupNumber of the subhalo, starts at 0 for each
    subhalo
GalaxyID:   int
    As before.
aperture_rad_in: float, [pkpc]
    Used to trim data, which is used to find peculiar velocity within this sphere
viewing_axis: 'z'
    Used to find projected halfmass radius


Output Parameters
-----------------

.gn: int
    GroupNumber of galaxy
.sgn: int
    SubGroupNumber of galaxy
.a: 
    Scale factor for given snapshot
.aexp: 
    Scale factor exponent for given snapshot
.h:
    0.6777 for z=0
.hexp: 
    h exponent for given snapshot
.boxsize:   [cMpc/h]
    Size of simulation boxsize in [cMpc/h]. Convert
    to [pMpc] by .boxsize / .h
.centre:    [pkpc]
    SQL value of centre of potential for the galaxy
        
.general:     dictionary of particle data:
    ['GroupNum']
    ['SubGroupNum']
    ['GalaxyID']
    ['SnapNum']
    ['stelmass']        - Masses within 30pkpc
    ['gasmass']
    ['gasmass_sf']
    ['gasmass_nsf']


"""
# Extracts the particle and SQL data
class Subhalo_Extract_Basic:
    
    def __init__(self, sim, data_dir, snapNum, gn, sgn, GalaxyID, centre_in, halo_mass_in, aperture_rad_in,
                            centre_galaxy=True, 
                            load_region_length=1.0,   # cMpc/h 
                            nfiles=16, 
                            debug=False,
                            print_progress=False):       
                            
        # Begining time
        time_start = time.time()
        
        # Assigning subhalo properties
        self.gn           = gn
        self.sgn          = sgn
        
        #----------------------------------------------------
        # Load information from the header for this snapshot to find a, aexp, h, hexp, boxsize
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Reading header')
            time_start = time.time()
        self.a, self.h, self.boxsize = read_header(data_dir) # units of scale factor, h, and L [cMpc/h]  
        
        
        # Distances:    [cMpc/h] * a^1 *h^-1 -> [pMpc]. [pMpc/h] * h^-1 -> [pMpc], [cMpc/h] * h^-1 -> [cMpc]
        # Velocity:     [cx/sh] * a^0.5 * h^0 -> [x/s]
        # Mass:         [Mass/h] * a^0 * h*-1 -> [Mass]
        for i in range(nfiles):
            f = h5py.File(data_dir, 'r')
            tmp = f['PartType4/Coordinates']

            # Get conversion factors.
            self.aexp    = f['PartType4/Coordinates'].attrs.get('aexp-scale-exponent')
            self.hexp    = f['PartType4/Coordinates'].attrs.get('h-scale-exponent')

            f.close()
        
        if debug:
            print('a        ', self.a)
            print('aexp     ', self.aexp)
            print('h        ', self.h)
            print('hexp     ', self.hexp)
            print('boxsize  [cMpc/h]', self.boxsize)
            print('boxsize  [cMpc]  ', self.boxsize/self.h)
            print('boxsize  [pMpc]  ', self.boxsize*self.a**1)
        
        
        #----------------------------------------------------
        # For a given gn and sgn, run sql query on SubFind catalogue
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Subhalo COP query')
            time_start = time.time()
        
        # Assigning halo mass
        self.halo_mass = halo_mass_in
        
        # These were all originally in cMpc, converted to pMpc through self.a and self.aexp
        self.centre    = centre_in * u.Mpc.to(u.kpc) * self.a**self.aexp                 # [pkpc]
        
        #-------------------------------------------------------------
        # Load data for stars and gas in non-centred units
        # Msun, pkpc, and pkpc/s
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Reading particle data _read_galaxy')
            time_start = time.time()
        stars     = self._read_galaxy(data_dir, 4, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length) 
        gas       = self._read_galaxy(data_dir, 0, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length)
        
        
        # CENTER COORDS, VELOCITY NOT ADJUSTED
        if centre_galaxy == True:
            stars['Coordinates'] = stars['Coordinates'] - self.centre
            gas['Coordinates']   = gas['Coordinates'] - self.centre
            
        # Trim data
        stars  = self._trim_within_rad(stars, aperture_rad_in)
        gas    = self._trim_within_rad(gas, aperture_rad_in)
        
        
        
        #====================================================
        # Find basic values we are after
        
        #--------------------------------------
        # Create masks for starforming and non-starforming gas
        if print_progress:
            print('Masking gas_sf and gas_nsf')
            time_start = time.time()
        mask_sf        = np.nonzero(gas['StarFormationRate'])          
        mask_nsf       = np.where(gas['StarFormationRate'] == 0)
        
        # Create dataset of star-forming and non-star-forming gas
        gas_sf = {}
        gas_nsf = {}
        for arr in gas.keys():
            gas_sf[arr]  = gas[arr][mask_sf]
            gas_nsf[arr] = gas[arr][mask_nsf]
            
        #--------------------------------------
        GroupNum           = self.gn
        SubGroupNum        = self.sgn
        self.GalaxyID      = GalaxyID
        self.SnapNum            = snapNum
        self.stelmass           = np.sum(stars['Mass'])     # [Msun] within 30 pkpc (aperture_rad_in)
        self.gasmass            = np.sum(gas['Mass'])       # [Msun] within 30 pkpc (aperture_rad_in)
        self.gasmass_sf         = np.sum(gas_sf['Mass'])    # [Msun] within 30 pkpc (aperture_rad_in)
        self.gasmass_nsf        = np.sum(gas_nsf['Mass'])   # [Msun] within 30 pkpc (aperture_rad_in)  
            
        #----------------------------------------------------
        # Filling self.general
        self.general = {}
        for general_name, general_item in zip(['GroupNum', 'SubGroupNum', 'GalaxyID', 'SnapNum', 'stelmass', 'gasmass', 'gasmass_sf', 'gasmass_nsf'], 
                                              [self.gn, self.sgn, GalaxyID, snapNum, np.sum(stars['Mass']) , np.sum(gas['Mass']), np.sum(gas_sf['Mass']), np.sum(gas_nsf['Mass'])]):
            self.general[general_name] = general_item
            
        # Left with all masses at 30pkpc
              
    def _read_galaxy(self, data_dir, itype, gn, sgn, centre, load_region_length, debug=False):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates, velocty, and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity."""
        
        # Where we store all the data
        data = {}
        
        # Initialize read_eagle module.
        eagle_data = EagleSnapshot(data_dir)
        
        # Put centre from pMpc -> cMpc/h units.
        centre_cMpc = centre * self.a**-1 * self.h

        # Select region to load, a 'load_region_length' cMpc/h cube centred on 'centre'.
        region = np.array([
            (centre_cMpc[0]-0.5*load_region_length), (centre_cMpc[0]+0.5*load_region_length),
            (centre_cMpc[1]-0.5*load_region_length), (centre_cMpc[1]+0.5*load_region_length),
            (centre_cMpc[2]-0.5*load_region_length), (centre_cMpc[2]+0.5*load_region_length)
        ])
        eagle_data.select_region(*region)
                
        # Load data using read_eagle, load conversion factors manually.
        f = h5py.File(data_dir, 'r')
        # If gas, load StarFormationRate
        if itype == 0:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'StarFormationRate']:
                tmp  = eagle_data.read_dataset(itype, att)
                cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
            f.close()
        # If stars, do not load StarFormationRate (as not contained in database)
        elif itype == 4:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates']:
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
        if itype == 0:
            data['StarFormationRate'] = data['StarFormationRate'] * u.g.to(u.Msun)  # [Msun/s]
        
        # Periodic wrap coordinates around centre (in proper units). 
        # boxsize converted from cMpc/h -> pMpc
        boxsize = self.boxsize * self.h**-1 * self.a**1       # [pkpc]
        data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
        
        # Converting to pkpc
        data['Coordinates'] = data['Coordinates'] * u.Mpc.to(u.kpc)     # [pkpc]
        
        return data
        
    def _trim_within_rad(self, arr, radius, debug=False):
        # Compute distance to centre and mask all within Radius in pkpc
        r  = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.where(r <= radius)
        
        newData = {}
        for header in arr.keys():
            newData[header] = arr[header][mask]
            
        return newData
        
   

#================================
""" 
DESCRIPTION
-----------
Will extract the Merger Tree of a given galaxy ID and extimate merger ratio

WILL ONLY WORK FOR Z=0 (need to fix)

- Main branch evolution will have the same 
  TopLeafID
- Main branch will have IDs between target_GalaxyID 
  and TopLeafID
- Entire tree will have IDs between target_GalaxyID 
  and LastProgID
- Galaxies about to merge into main branch will have 
  their DescendentID lie between target_GalaxyID 
  and TopLeafID
- These galaxies can be used to estimate merger 
  ratios by taking their stellar mass before merger 
  (simple), or looking for the largest mass history 
  of that galaxy before merger (complex)

No change is made between SubGroupNumber as some 
mergers may originate from accreted Satellites, while
some may be external.



CALLING FUNCTION
----------------
tree = MergerTree(sim, target_GalaxyID)



INPUT PARAMETERS
----------------
sim:
    Simulation types eg. 
    mySims = np.array([('RefL0012N0188', 12)])   
target_GalaxyID:    int
    Galaxy ID at z=0 that we care about



OUTPUT PARAMETERS
-----------------
.target_GalaxyID:       int
    Galaxy ID at z=0 that we care about
.sim:
    Simulation that we used
.TopLeafID:
    The oldest galaxy ID in the main branch. All galaxies
    on the main branch will have the same TopLeafID
.all_branches:      dict
    Has raw data from all branches, including main branch
    ['redshift']        - redshift list (has duplicates, but ordered largest-> smallest)
    ['snapnum']         - 
    ['GalaxyID']        - Unique ID of specific galaxy
    ['DescendantID']    - DescendantID of the galaxy next up, this has been selected to lie on the main branch
    ['TopLeafID']       - Oldest galaxy ID of branch. Can differ, but all on main have same as .TopLeafID
    ['GroupNumber']     - Will mostly be the same as interested galaxy
    ['SubGroupNumber']  - Will mostly be satellites
    ['stelmass']        - stellar mass in 30pkpc
    ['gasmass']         - gasmass in 30pkpc
    ['totalstelmass']   - total Subhalo stellar mass
.main_branch        dict
    Has raw data from only main branch
    ['redshift']        - redshift list (no duplicates, large -> small)
    ['lookbacktime']        - lookback time
    ['snapnum']         - 
    ['GalaxyID']        - Unique ID of specific galaxy
    ['DescendantID']    - DescendantID of the galaxy next up, this has been selected to lie on the main branch
    ['TopLeafID']       - Oldest galaxy ID of branch. Can differ, but all on main have same as .TopLeafID
    ['GroupNumber']     - Will mostly be the same as interested galaxy
    ['SubGroupNumber']  - Will mostly be satellites
    ['stelmass']        - stellar mass in 30pkpc
    ['gasmass']         - gasmass in 30pkpc
    ['totalstelmass']   - total Subhalo stellar mass
.mergers            dict
    Has complete data on all mergers, ratios, and IDs. Merger data will be entered
    for snapshot immediately AFTER both galaxies do not appear.

    For example: 
        Snap    ID              Mass
        27      35787 264545    10^7 10^4
        28      35786           10^7...
                        ... Here the merger takes place at Snap28
    ['redshift']        - Redshift (unique)
    ['snapnum']         - Snapnums (unique)
    ['ratios']          - Array of mass ratios at this redshift
    ['gasratios']       - Array of gas ratios at this redshift
    ['GalaxyIDs']       - Array of type [Primary GalaxyID, Secondary GalaxyID]


"""
class MergerTree:
    def __init__(self, sim, target_GalaxyID, maxSnap):
        # Assign target_GalaxyID (any snapshot)
        self.target_GalaxyID    = target_GalaxyID
        self.sim                = sim
        
        # SQL query for single galaxy
        myData = self._extract_merger_tree(target_GalaxyID, maxSnap)
         
        redshift        = myData['z']
        snapnum         = myData['SnapNum']
        GalaxyID        = myData['GalaxyID']
        DescendantID    = myData['DescendantID']
        TopLeafID       = myData['TopLeafID']
        GroupNumber     = myData['GroupNumber']
        SubGroupNumber  = myData['SubGroupNumber']
        stelmass        = myData['stelmass']
        gasmass         = myData['gasmass']
        totalstelmass   = myData['totalstelmass']
        totalgasmass    = myData['totalgasmass']
        
        # Extract TopLeafID of main branch (done by masking for target_GalaxyID)
        mask = np.where(GalaxyID == target_GalaxyID)
        self.TopLeafID = TopLeafID[mask]
        
        
        # Create dictionary for all branches (that meet the requirements in the description above)
        self.all_branches = {}
        for name, entry in zip(['redshift', 'snapnum', 'GalaxyID', 'DescendantID', 'TopLeafID', 'GroupNumber', 'SubGroupNumber', 'stelmass', 'gasmass', 'totalstelmass', 'totalgasmass'], [redshift, snapnum, GalaxyID, DescendantID, TopLeafID, GroupNumber, SubGroupNumber, stelmass, gasmass, totalstelmass, totalgasmass]):
            self.all_branches[name] = entry
        
        
        # Extract only main branch data
        self.main_branch = self._main_branch(self.TopLeafID, self.all_branches)
        self.main_branch['lookbacktime'] = ((13.8205298 * u.Gyr) - FlatLambdaCDM(H0=67.77, Om0=0.307, Ob0 = 0.04825).age(self.main_branch['redshift'])).value
        
        
        # Find merger ratios along tree
        self.mergers = self._analyze_tree(target_GalaxyID, self.TopLeafID, self.all_branches)
        
        # gasmass is not sf or nsf
        
        
    def _extract_merger_tree(self, target_GalaxyID, maxSnap):
        # This uses the eagleSqlTools module to connect to the database with your username and password.
        # If the password is not given, the module will prompt for it.
        con = sql.connect('lms192', password='dhuKAP62')
        

        for sim_name, sim_size in self.sim:
            #print(sim_name)
        
            # Construct and execute query to contruct merger tree
            myQuery = 'SELECT \
                         SH.Redshift as z, \
                         SH.SnapNum, \
                         SH.GalaxyID, \
                         SH.DescendantID, \
                         SH.TopLeafID, \
                         SH.GroupNumber, \
                         SH.SubGroupNumber, \
                         AP.Mass_Star as stelmass, \
                         AP.Mass_Gas as gasmass, \
                         SH.MassType_Star as totalstelmass, \
                         SH.MassType_Gas as totalgasmass \
                       FROM \
                         %s_Subhalo as SH, \
                         %s_Aperture as AP, \
                         %s_Subhalo as REF \
                       WHERE \
                         REF.GalaxyID = %s \
                         and SH.SnapNum >= %s \
                         and (((SH.SnapNum > REF.SnapNum and REF.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum <= REF.SnapNum and SH.GalaxyID between REF.GalaxyID and REF.LastProgID))) \
                         and ((SH.DescendantID between SH.GalaxyID and SH.TopLeafID) or (SH.DescendantID between REF.GalaxyID and REF.TopLeafID) or (SH.DescendantID = -1)) \
                         and AP.Mass_Star > 5000000 \
                         and AP.ApertureSize = 30 \
                         and AP.GalaxyID = SH.GalaxyID \
                       ORDER BY \
                         SH.Redshift DESC, \
                         SH.DescendantID DESC' %(sim_name, sim_name, sim_name, target_GalaxyID, maxSnap)
            
            # Execute query.
            myData = sql.execute_query(con, myQuery)
        
        return myData
 
    def _main_branch(self, TopLeafID, arr):
        # arr will have all merger tree data, all those with the same TopLeafID will be on the main branch
        mask = np.where(arr['TopLeafID'] == TopLeafID)
        
        newData = {}
        for header in arr.keys():
            newData[header] = arr[header][mask]
            
        return newData
 
    def _analyze_tree(self, target_GalaxyID, TopLeafID, arr, debug=False):
        # arr is sorted per redshift in descending order (starting at 0)
        redshift_tmp  = []
        snapnum_tmp   = []
        ratios_tmp    = []
        gasratios_tmp = []
        IDs_tmp       = []
        
        
        if debug:
            for z, snapnum, topID, galaxyID, stelmass, gasmass in zip(arr['redshift'], arr['snapnum'], arr['TopLeafID'], arr['GalaxyID'], arr['stelmass'], arr['gasmass']):
                print('%.2f    %s    %s    %s     %.2e     %.2e' %(z, snapnum, topID, galaxyID, stelmass, gasmass))
        
        # Create dictionary for primary components
        primary_dict = {}
        for z, snapnum, topID, galaxyID, stelmass, gasmass in zip(arr['redshift'], arr['snapnum'], arr['TopLeafID'], arr['GalaxyID'], arr['stelmass'], arr['gasmass']):
            if int(topID) == int(TopLeafID):
                primary_dict['%s' %snapnum] = {'GalaxyID': galaxyID,
                                              'Redshift': z,
                                              'SnapNum': snapnum,
                                              'stelmass': stelmass,
                                              'gasmass': gasmass}
                                              
        # Loop over all to find ratios of each
        z_old = 999999
        
        ratios_collect    = []
        gasratios_collect = []
        IDs_collect       = []
        
        for z, snapnum, topID, galaxyID, stelmass, gasmass in zip(arr['redshift'], arr['snapnum'], arr['TopLeafID'], arr['GalaxyID'], arr['stelmass'], arr['gasmass']):
            
            if float(z) != float(z_old):
                # If current z not yet added to list, do so
                redshift_tmp.append(z)
                snapnum_tmp.append(snapnum)
                ratios_tmp.append(ratios_collect)
                gasratios_tmp.append(gasratios_collect)
                IDs_tmp.append(IDs_collect)
                
                # Create tmp arrays at the start of new z
                ratios_collect    = []
                gasratios_collect = []
                IDs_collect       = []
                
            # If mainline, continue
            if int(topID) == int(TopLeafID):
                z_old = z
                continue
            
            # If final snap, continue
            if int(snapnum) == 28:
                z_old = z
                continue
            
            # Assign masses    
            primary_stelmass = float(primary_dict['%s' %snapnum]['stelmass'])
            primary_gasmass = float(primary_dict['%s' %snapnum]['gasmass'])
            secondary_stelmass = float(stelmass)
            secondary_gasmass = float(gasmass)
            
            # find mergers and gas ratio
            merger_ratio = secondary_stelmass / primary_stelmass
            #gas_ratio    = (primary_gasmass + secondary_gasmass) / (primary_stelmass + secondary_stelmass)
            gas_ratio = secondary_gasmass / primary_gasmass
        
            z_old = z
        
            #Grab ID of secondary
            id_secondary = galaxyID
            id_primary = primary_dict['%s' %snapnum]['GalaxyID']
            
            ratios_collect.append(merger_ratio)
            gasratios_collect.append(gas_ratio)
            IDs_collect.append([id_primary, id_secondary])
        
            
        # Create dictionary
        merger_dict = {}
        merger_dict['redshift']  = redshift_tmp
        merger_dict['snapnum']   = snapnum_tmp
        merger_dict['ratios']    = ratios_tmp
        merger_dict['gasratios'] = gasratios_tmp
        merger_dict['GalaxyIDs'] = IDs_tmp
        
        if debug:
            print(snapnum_tmp)
            print(ratios_tmp)
            print(IDs_tmp)
        
        return merger_dict




#================================
# Will find gn, sgn, and snap of galaxy when given ID and sim
def ConvertID(galID, sim):
    # This uses the eagleSqlTools module to connect to the database with your username and password.
    # If the password is not given, the module will prompt for it.
    con = sql.connect("lms192", password="dhuKAP62")
    
    for sim_name, sim_size in sim:
        #print(sim_name)

        # Construct and execute query for each simulation. This query returns properties for a single galaxy
        myQuery = 'SELECT \
                    SH.GroupNumber, \
                    SH.SubGroupNumber, \
                    SH.SnapNum, \
                    SH.Redshift, \
                    SH.CentreOfPotential_x as x, \
                    SH.CentreOfPotential_y as y, \
                    SH.CentreOfPotential_z as z, \
                    FOF.Group_M_Crit200 as halo_mass, \
                    MK.Ellipticity as ellip, \
                    MK.Triaxiality as triax, \
                    MK.KappaCoRot as kappa_stars, \
                    MK.DispAnisotropy as disp_ani, \
                    MK.DiscToTotal as disc_to_total, \
                    MK.RotToDispRatio as rot_to_disp_ratio \
                   FROM \
    			     %s_Subhalo as SH, \
                     %s_FOF as FOF, \
                     %s_MorphoKinem as MK \
                   WHERE \
    			     SH.GalaxyID = %s \
                     and SH.GalaxyID = MK.GalaxyID \
                     and SH.GroupID = FOF.GroupID'%(sim_name, sim_name, sim_name, galID)

        # Execute query.
        myData = sql.execute_query(con, myQuery)
        
        return myData['GroupNumber'], myData['SubGroupNumber'], myData['SnapNum'], myData['Redshift'], myData['halo_mass'], np.array([myData['x'], myData['y'], myData['z']]), np.array([myData['ellip'], myData['triax'], myData['kappa_stars'], myData['disp_ani'], myData['disc_to_total'], myData['rot_to_disp_ratio']])

def ConvertID_noMK(galID, sim):
    # This uses the eagleSqlTools module to connect to the database with your username and password.
    # If the password is not given, the module will prompt for it.
    con = sql.connect("lms192", password="dhuKAP62")
    
    for sim_name, sim_size in sim:
        #print(sim_name)

        # Construct and execute query for each simulation. This query returns properties for a single galaxy
        myQuery = 'SELECT \
                    SH.GroupNumber, \
                    SH.SubGroupNumber, \
                    SH.SnapNum, \
                    SH.Redshift, \
                    SH.CentreOfPotential_x as x, \
                    SH.CentreOfPotential_y as y, \
                    SH.CentreOfPotential_z as z, \
                    FOF.Group_M_Crit200 as halo_mass \
                   FROM \
    			     %s_Subhalo as SH, \
                     %s_FOF as FOF \
                   WHERE \
    			     SH.GalaxyID = %s \
                     and SH.GroupID = FOF.GroupID'%(sim_name, sim_name, galID)

        # Execute query.
        myData = sql.execute_query(con, myQuery)
        
        return myData['GroupNumber'], myData['SubGroupNumber'], myData['SnapNum'], myData['Redshift'], myData['halo_mass'], np.array([myData['x'], myData['y'], myData['z']]), np.array([math.nan, math.nan, math.nan, math.nan, math.nan, math.nan])


#================================
""" 
Purpose
-------
Will connect to sql database to find halfmassrad, peculiar velocity, 
centre coordinates (potential & mass) for a given snapshot, and then
extract data from local files.

Calling function
----------------
galaxy = Subhalo_Extract(sample_input['mySims'], dataDir_dict['%s' %str(SnapNum)], SnapNum, GroupNum, SubGroupNum, aperture_rad, viewing_axis)

Input Parameters
----------------

mySims:
    mySims = np.array([('RefL0012N0188', 12)])  
    Name and boxsize
dataDir:
    Location of the snapshot data hdf5 file, 
    eg. '/Users/c22048063/Documents/.../snapshot_028_xx/snap_028_xx.0.hdf5'
SnapNum: int
    Snapshot number, ei. 28
GroupNum: int
    GroupNumber of the subhalo, starts at 1
SubGroupNum: int
    SubGroupNumber of the subhalo, starts at 0 for each
    subhalo
aperture_rad_in: float, [pkpc]
    Used to trim data, which is used to find peculiar velocity within this sphere
viewing_axis: 'z'
    Used to find projected halfmass radius


Output Parameters
-----------------

.gn: int
    GroupNumber of galaxy
.sgn: int
    SubGroupNumber of galaxy
.a: 
    Scale factor for given snapshot
.aexp: 
    Scale factor exponent for given snapshot
.h:
    0.6777 for z=0
.hexp: 
    h exponent for given snapshot
.boxsize:   [cMpc/h]
    Size of simulation boxsize in [cMpc/h]. Convert
    to [pMpc] by .boxsize / .h
.centre:    [pkpc]
    SQL value of centre of potential for the galaxy
        
.halo_mass:     float
    Value of the halomass within 200 density
.stars_com:     np.array[x, y, z]
    stars C.O.M within aperture, w.r.t [0, 0, 0]
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
.dm:     dictionary of particle data:
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 
.bh:     dictionary of particle data:
    ['Coordinates']         - [pkpc]
    ['Velocity']            - [pkm/s]
    ['Mass']                - [Msun]
    ['GroupNumber']         - int array 
    ['SubGroupNumber']      - int array 

If centre_galaxy == True; 'Coordinates' - .centre, 'Velocity' - .perc_vel
"""
# Extracts the particle and SQL data
class Subhalo_Extract_old:
    
    def __init__(self, sim, data_dir, snapNum, gn, sgn, centre_in, halo_mass_in, aperture_rad_in, viewing_axis,
                            centre_galaxy=True, 
                            load_region_length=1.0,   # cMpc/h 
                            nfiles=16, 
                            debug=False,
                            print_progress=False):       
                            
        # Begining time
        time_start = time.time()
        
        # Assigning subhalo properties
        self.gn           = gn
        self.sgn          = sgn
        
        #----------------------------------------------------
        # Load information from the header for this snapshot to find a, aexp, h, hexp, boxsize
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Reading header')
            time_start = time.time()
        self.a, self.h, self.boxsize = read_header(data_dir) # units of scale factor, h, and L [cMpc/h]  
        
        
        # Distances:    [cMpc/h] * a^1 *h^-1 -> [pMpc]. [pMpc/h] * h^-1 -> [pMpc], [cMpc/h] * h^-1 -> [cMpc]
        # Velocity:     [cx/sh] * a^0.5 * h^0 -> [x/s]
        # Mass:         [Mass/h] * a^0 * h*-1 -> [Mass]
        for i in range(nfiles):
            f = h5py.File(data_dir, 'r')
            tmp = f['PartType4/Coordinates']

            # Get conversion factors.
            self.aexp    = f['PartType4/Coordinates'].attrs.get('aexp-scale-exponent')
            self.hexp    = f['PartType4/Coordinates'].attrs.get('h-scale-exponent')

            f.close()
        
        if debug:
            print('a        ', self.a)
            print('aexp     ', self.aexp)
            print('h        ', self.h)
            print('hexp     ', self.hexp)
            print('boxsize  [cMpc/h]', self.boxsize)
            print('boxsize  [cMpc]  ', self.boxsize/self.h)
            print('boxsize  [pMpc]  ', self.boxsize*self.a**1)
        
        
        #----------------------------------------------------
        # For a given gn and sgn, run sql query on SubFind catalogue
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Subhalo COP query')
            time_start = time.time()
        
        # Assigning halo mass
        self.halo_mass = halo_mass_in
        
        # These were all originally in cMpc, converted to pMpc through self.a and self.aexp
        self.centre       = centre_in * u.Mpc.to(u.kpc) * self.a**self.aexp                 # [pkpc]
        
        #-------------------------------------------------------------
        # Load data for stars and gas in non-centred units
        # Msun, pkpc, and pkpc/s
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('Reading particle data _read_galaxy')
            time_start = time.time()
        stars     = self._read_galaxy(data_dir, 4, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length) 
        gas       = self._read_galaxy(data_dir, 0, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length)
        dm        = self._read_galaxy(data_dir, 1, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length)
        bh        = self._read_galaxy(data_dir, 5, gn, sgn, self.centre*u.kpc.to(u.Mpc), load_region_length)
        
        # CENTER COORDS, VELOCITY NOT ADJUSTED
        if centre_galaxy == True:
            stars['Coordinates'] = stars['Coordinates'] - self.centre
            gas['Coordinates']   = gas['Coordinates'] - self.centre
            dm['Coordinates']    = dm['Coordinates'] - self.centre
            bh['Coordinates']    = bh['Coordinates'] - self.centre
            
        
        #---------------------------
        # Finding stars COM within 30pkpc
        stars_com  = self._centre_of_mass(self._trim_within_rad(stars, aperture_rad_in))
        if debug:
            print('stars COM')
            print(stars_com)
        
        # Setting the stars COM in 30pkpc as the new centre, and the external stars_com w.r.t potential (which until now was at [0, 0, 0])
        stars['Coordinates'] = stars['Coordinates'] - stars_com
        gas['Coordinates']   = gas['Coordinates'] - stars_com
        dm['Coordinates']    = dm['Coordinates'] - stars_com
        bh['Coordinates']    = bh['Coordinates'] - stars_com
        self.stars_com       = [0, 0, 0] - stars_com
        
        #---------------------------
        # Creating temporary array for stars centred on stellar COM, trimming this to the aperture
        stars_trimmed = self._trim_within_rad(stars, aperture_rad_in)
        
        # Making the minimum HMR 1 pkpc
        self.halfmass_rad_proj  = max(self._projected_rad(stars_trimmed, viewing_axis), 1)
        self.halfmass_rad       = max(self._half_rad(stars_trimmed), 1)
            
        # Finding peculiar velocity of stars within 30 pkpc of COM
        self.perc_vel = self._peculiar_velocity_part(stars_trimmed)
         
        """# Finding peculiar velocity for all particles
        self.perc_vel = self._peculiar_velocity()
        """
        
        # account for peculiar velocity within aperture rad
        if centre_galaxy == True:            
            stars['Velocity'] = stars['Velocity'] - self.perc_vel
            gas['Velocity']   = gas['Velocity'] - self.perc_vel
            dm['Velocity']    = dm['Velocity'] - self.perc_vel
            bh['Velocity']    = bh['Velocity'] - self.perc_vel
        
        #---------------------------
        # Assigning particle data        
        self.data_nil = {}
        for parttype, parttype_name in zip([stars, gas, dm, bh], ['stars', 'gas', 'dm', 'bh']):
            self.data_nil['%s'%parttype_name] = parttype
        
        
        if debug:
            print('_main DEBUG')
            print(self.halfmass_rad)
            print(self.halfmass_rad_proj)
            print(self.perc_vel)   
        if print_progress:
            print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
            print('  EXTRACTION COMPLETE')
        
    def _read_galaxy(self, data_dir, itype, gn, sgn, centre, load_region_length, debug=False):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates, velocty, and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity."""
        
        # Where we store all the data
        data = {}
        
        # Initialize read_eagle module.
        eagle_data = EagleSnapshot(data_dir)
        
        # Put centre from pMpc -> cMpc/h units.
        centre_cMpc = centre * self.a**-1 * self.h

        # Select region to load, a 'load_region_length' cMpc/h cube centred on 'centre'.
        region = np.array([
            (centre_cMpc[0]-0.5*load_region_length), (centre_cMpc[0]+0.5*load_region_length),
            (centre_cMpc[1]-0.5*load_region_length), (centre_cMpc[1]+0.5*load_region_length),
            (centre_cMpc[2]-0.5*load_region_length), (centre_cMpc[2]+0.5*load_region_length)
        ])
        eagle_data.select_region(*region)
                
        # Load data using read_eagle, load conversion factors manually.
        f = h5py.File(data_dir, 'r')
        # If gas, load StarFormationRate
        if itype == 0:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'StarFormationRate', 'Velocity', 'ParticleIDs']:
                tmp  = eagle_data.read_dataset(itype, att)
                cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
            f.close()
        # If dm
        elif itype == 1:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'Velocity', 'ParticleIDs']:
                if att == 'Mass':
                    cgs  = f['PartType0/%s'%(att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType0/%s'%(att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType0/%s'%(att)].attrs.get('h-scale-exponent')
                    
                    # Create special extract dm mass
                    dm_mass     = f['Header'].attrs.get('MassTable')[1]
                    # Create an array of length n_particles each set to dm_mass.
                    m = np.ones(n_particles, dtype='f8') * dm_mass
            
                    data[att] = np.multiply(m, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                else:
                    tmp  = eagle_data.read_dataset(itype, att)
                    cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                    data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                    
                    # Used in next cycle for Mass
                    n_particles = len(tmp)
                
            f.close()
        # If stars, do not load StarFormationRate (as not contained in database)
        elif itype == 4:
            for att in ['GroupNumber', 'SubGroupNumber', 'Mass', 'Coordinates', 'Velocity', 'ParticleIDs']:
                tmp  = eagle_data.read_dataset(itype, att)
                cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                
            f.close()
        # If bhs
        elif itype == 5:
            for att in ['GroupNumber', 'SubGroupNumber', 'BH_Mass', 'BH_Mdot', 'Coordinates', 'Velocity', 'ParticleIDs']:
                # Ensure we use 'Mass' as name
                if att == 'BH_Mass':
                    tmp  = eagle_data.read_dataset(itype, att)
                    cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
                    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
                    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
                    data['Mass'] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8')
                    
                else:
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
        if itype == 5:
            data['BH_Mdot'] = data['BH_Mdot'] * u.g.to(u.Msun)  # [Msun/s]
        
        # Periodic wrap coordinates around centre (in proper units). 
        # boxsize converted from cMpc/h -> pMpc
        boxsize = self.boxsize * self.h**-1 * self.a**1       # [pkpc]
        data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
        
        # Converting to pkpc
        data['Coordinates'] = data['Coordinates'] * u.Mpc.to(u.kpc)     # [pkpc]
        data['Velocity'] = data['Velocity'] * u.Mpc.to(u.km)            # [pkm/s]
        
        return data
        
    def _trim_within_rad(self, arr, radius, debug=False):
        # Compute distance to centre and mask all within Radius in pkpc
        r  = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.where(r <= radius)
        
        newData = {}
        for header in arr.keys():
            newData[header] = arr[header][mask]
            
        return newData
        
    def _centre_of_mass(self, arr, debug=False):
        # Mass-weighted formula from subhalo paper
        mass_weighted  = arr['Coordinates'] * arr['Mass'][:, None]
        centre_of_mass = np.sum(mass_weighted, axis=0)/np.sum(arr['Mass'])
    
        return centre_of_mass
    
    def _peculiar_velocity(self, debug=False):        
        # Mass-weighted formula from subhalo paper
        vel_weighted = 0
        mass_sums = 0
        for arr in [self.stars, self.gas, self.dm, self.bh]:
            if len(arr['Mass']) > 0:
                vel_weighted = vel_weighted + np.sum(arr['Velocity'] * arr['Mass'][:, None], axis=0)
                mass_sums = mass_sums + np.sum(arr['Mass'])
            
        pec_vel = vel_weighted / mass_sums
            
        return pec_vel
        
    def _peculiar_velocity_part(self, arr, debug=False):        
        # Mass-weighted formula from subhalo paper
        vel_weighted = np.sum(arr['Velocity'] * arr['Mass'][:, None], axis=0)
        mass_sums = np.sum(arr['Mass'])
            
        pec_vel = vel_weighted / mass_sums
            
        return pec_vel
    
    def _projected_rad(self, arr, viewing_axis, debug=False):
        # Compute distance to centre (projected)        
        if viewing_axis == 'z':
            r = np.linalg.norm(arr['Coordinates'][:,[0,1]], axis=1)
            mask = np.argsort(r)
            r = r[mask]
        if viewing_axis == 'y':
            r = np.linalg.norm(arr['Coordinates'][:,[0,2]], axis=1)
            mask = np.argsort(r)
            r = r[mask]
        if viewing_axis == 'x':
            r = np.linalg.norm(arr['Coordinates'][:,[1,2]], axis=1)
            mask = np.argsort(r)
            r = r[mask]
            
        stelmass = np.sum(arr['Mass'][mask])
            
        # Compute cumulative mass
        cmass = np.cumsum(arr['Mass'][mask])
        index = np.where(cmass >= stelmass*0.5)[0][0]
        radius = r[index]
        
        return radius
        
    def _half_rad(self, arr, debug=False):
        # Compute distance to centre       
        r = np.linalg.norm(arr['Coordinates'], axis=1)
        mask = np.argsort(r)
        r = r[mask]
            
        stelmass = np.sum(arr['Mass'][mask])
            
        # Compute cumulative mass
        cmass = np.cumsum(arr['Mass'][mask])
        index = np.where(cmass >= stelmass*0.5)[0][0]
        radius = r[index]
        
        return radius

