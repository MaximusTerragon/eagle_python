import h5py
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 
from read_dataset import read_dataset
from read_header import read_header
from read_dataset_dm_mass import read_dataset_dm_mass

class RadialMass:

    def __init__(self, gn, sgn, centre, stelmass):

        self.a, self.h, self.boxsize = read_header()
        self.centre = centre
        self.stelmass = stelmass
        
        self.gn = gn
        self.sgn = sgn

        # Load data.
        self.gas    = self.read_galaxy(0, gn, sgn, centre)
        self.dm     = self.read_galaxy(1, gn, sgn, centre)
        self.stars  = self.read_galaxy(4, gn, sgn, centre)
        self.bh     = self.read_galaxy(5, gn, sgn, centre)

        # Plot.
        self.plot()
    

    def read_galaxy(self, itype, gn, sgn, centre):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity. """

        data = {}

        # Load data, then mask to selected GroupNumber and SubGroupNumber. Gets converted into p from read_dataset
        gns  = read_dataset(itype, 'GroupNumber')
        sgns = read_dataset(itype, 'SubGroupNumber')
        mask = np.logical_and(gns == gn, sgns == sgn)
        # Extact the data and convert into pMsun, pMpc (from cMsun, cMpc)
        if itype == 1:
            data['mass'] = read_dataset_dm_mass()[mask] * u.g.to(u.Msun)
        else:
            data['mass'] = read_dataset(itype, 'Mass')[mask] * u.g.to(u.Msun)
        data['coords'] = read_dataset(itype, 'Coordinates')[mask] * u.cm.to(u.Mpc)

        # Periodic wrap coordinates around centre.
        boxsize = self.boxsize/self.h
        data['coords'] = np.mod(data['coords']-centre+0.5*boxsize,boxsize)+centre-0.5*boxsize

        return data

    def compute_radial_mass(self, arr):
        """ Compute the mass distribution of a given galaxy as function of radius from centre of potential. """

        # Compute distance to centre. [pMpc]
        r = np.linalg.norm(arr['coords'] - self.centre, axis=1)
        mask = np.argsort(r)
        r = r[mask]

        # Compute cumulative mass. [pMsun]
        cmass = np.cumsum(arr['mass'][mask])

        # Return r in pMpc and cmass in pMsun.
        return r, cmass

    def compute_halfmass_radius(self, arr, radius):
        """ Compute the mass radius of a given galaxy. """

        # Compute distance to centre. [pMpc]
        r = np.linalg.norm(arr['coords'] - self.centre, axis=1)
        mask = np.argsort(r)
        r = r[mask]

        # Compute cumulative mass. [pMsun]
        cmass = np.cumsum(arr['mass'][mask])

        # Find index where cmass is above halfmass
        index = np.where(cmass >= self.stelmass*0.5)[0][0]
        # Multiply radius by halfmass multiplier
        stel_rad = r[index]*radius
        
        #print(np.where(cmass >= self.stelmass)[0][0])   #index
        #print('\n%sx stellar halfmass = %.5f'%(str(radius),cmass[index]))
        #print('%sx stellar halfmass radius [pkpc]= %.5f'%(str(radius), stel_rad*1000))
        
        # Return r in Mpc and cmass in pMsun.
        return stel_rad
        
        
    def plot(self):
        plt.figure()

        # All parttypes together.
        combined = {}
        combined['mass'] = np.concatenate((self.gas['mass'], self.dm['mass'], self.stars['mass'], self.bh['mass']))
        combined['coords'] = np.vstack((self.gas['coords'], self.dm['coords'], self.stars['coords'], self.bh['coords']))
        
        # halfmass rad of galaxy in question (pMpc)
        halfmassrad = self.compute_halfmass_radius(self.stars, 1)
        
        # Loop over each parttype.
        for x, lab in zip([self.gas, self.dm, self.stars, combined], ['Gas', 'Dark Matter', 'Stars', 'All']):
            r, cmass = self.compute_radial_mass(x)
            plt.plot(r*1000., np.log10(cmass), label=lab)

        # Print various stats
        plt.axhline(np.log10(self.stelmass/2))
        plt.axvline(halfmassrad*1000)
        
        print("Total stellar mass = ", self.stelmass)
        print("Half stellar mass = ", self.stelmass/2)
        print("Half-mass radius [pMpc]= ", halfmassrad)
        
        # Save plot.
        plt.legend(loc='center right')
        plt.minorticks_on()
        plt.ylabel('Mass [Msun]'); plt.xlabel('r [kpc]')
        plt.xlim(1, 50); plt.tight_layout()
        plt.savefig('./trial_plots/RadialMass%s_%s.jpeg'%(str(self.gn),str(self.sgn)))
        plt.show()
        plt.close()
        
        return halfmassrad

if __name__ == '__main__':
    centre = np.array([2.2066073,10.020961,12.453941])
    stelmass = 6.2772183E10
    x = RadialMass(2, 0, centre, stelmass)
    
    #print(x)
    
# Galaxy ID = 3748