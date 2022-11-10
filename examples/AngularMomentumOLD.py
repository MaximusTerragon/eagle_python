class AngularMomentum:

    # load_region_length is in cMpc/h
    def __init__(self, gn, sgn, centre, load_region_length=2):
        """Initialising 'self' which will contain all the quantities needed throughout, which will then
        be bound to the object called 'self'. Essentially upon execution of the class AngularMomentum, we 
        feed it gn, sgn, centre. init is executed at the start to create the 'space' to access all data from
        within AngularMomentum"""
        
        # Load information from the header
        self.a, self.h, self.boxsize = read_header() # units of scale factor, h, and L [cMpc/h]
        self.centre = centre

        # Put centre into cMpc/h units as load_region_length is in cMpc/h units
        print('centre [cMpc] = ', self.centre)
        centre *= self.h                        
        print('h factor = ', self.h)
        print('new centre [cMpc/h] = ', self.centre)
        
        
        # Load data.
        self.gas    = self.read_galaxy(0, gn, sgn, centre, load_region_length)
        #self.dm     = self.read_galaxy(1, gn, sgn, centre, load_region_length)
        self.stars  = self.read_galaxy(4, gn, sgn, centre, load_region_length)
        #self.bh     = self.read_galaxy(5, gn, sgn, centre, load_region_length)
        
        
        
        
        
        # Plot
        
        
    def read_galaxy(self, itype, gn, sgn, centre, load_region_length):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates, velocty, and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity."""

        # Where we store all the data
        data = {}
        
        # Initialize pyread_eagle module, where dataDir is the loaction of the first hdf5 file
        eagle_data = EagleSnapshot(dataDir)
        
        
        
            # Put centre into...
        
        
        
        # Select region to load, a 'load_region_length' cMpc/h cube centred on 'centre'.
        region = np.array([
            (centre[0]-0.5*load_region_length), (centre[0]+0.5*load_region_length),
            (centre[1]-0.5*load_region_length), (centre[1]+0.5*load_region_length),
            (centre[2]-0.5*load_region_length), (centre[2]+0.5*load_region_length)
        ])
        print('Region to select:', region)
        eagle_data.select_region(*region)
        
        
        # Load data using read_eagle, load conversion factors manually, extract in [proper cgs] units.
        # Extract mass, position, and velocity for gas and stars (which was specified up top in the load region)
        f = h5py.File(dataDir, 'r')
        for att in ['GroupNumber', 'SubGroupNumber', 'Mass','Coordinates', 'Velocity']:
            tmp  = eagle_data.read_dataset(itype, att)
            cgs  = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
            aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
            hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
            
            # convert data into 'proper position' cgs units (see §2.3.8 particle paper)
            data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype='f8') 
            """Coords were in comoving Mpc [h^-1 Mpc], now in proper length [cm]"""
            
        f.close()
        
        
        # Mask to selected GroupNumber and SubGroupNumber
        mask = np.logical_and(data['GroupNumber'] == gn, data['SubGroupNumber'] == sgn)
        # eg, for GroupNumber (and all other fields as above): mask out the ones for the selected galaxy/halo 
        for att in data.keys():
            data[att] = data[att][mask]
            
        # Periodic wrap coordinates around centre (in proper units)
        boxsize = self.boxsize/self.h
        data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
        
        
        return data

    
    def comoving_spin(self):
        # self will have access to everything that was assigned in .self in the init function at the top
        #Calculate CentreOfPotential which is the same as CentreOfPotential in subhalo... just done manually AND different units.
        #----- IS IN cMpc/h UNITS -----
        #See equation D.22 in subfind paper for details"
        
        
        
        
        ##### Tim #### Use hydrogen and helium data only?
        for particle_type in [self.gas, self.stars]:
            np.cumsum(particle_type['Mass'] * particle_type['Coordinates'])/np.cumsum(particle_type['Mass'])
            
            
        
        
        
        # Galaxy comoving spin given by equation D.21 in subfind paper
        # Has dimensions but Spin (see paper) is given as dimensionlessn
        
        #for given gn, sgn, extract star, gas (hydrogen and helium) data for particle mass, particle position, particle velocity
        #extract scale factor a from snap
        
        #for type in (star, gas):
            #sum over mi * a**2 * (xi-cpos_%type) [cross product] (vi-cvel_%type)
            #
            
        #return position vector of spin alignment

    
if __name__ == '__main__':
    # centre extracted from online database
    centre = np.array([12.08809, 4.474372, 1.4133347])

    L = AngularMomentum(1, 0, centre)
    
    #print projection and magnitude of spin for gas and star... define function to take DM too and separately as input
    #print('Spin magnitude: %.5f'%mag(L))
    #print('Spin projection xyz: [%.5f, %.5f, %.5f]', [%L[0], %L[1]], %L[2])
        
        
#Top galaxy:
#coords = 12.08809,4.474372,1.4133347,
#total stellar mass = 7.702664E10
#gas spin: [-3378.3005, 1819.4624, -5320.938]
#halfmass rad = 4.60127 pkpc
#star spin: [-100.83867, -330.70212, 334.77097]