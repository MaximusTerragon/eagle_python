import h5py
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 
from read_dataset import read_dataset
from read_header import read_header
from read_dataset_dm_mass import read_dataset_dm_mass

N = 10000
x = np.random.uniform(0, 10, N)
y = np.random.uniform(0, 10, N)
z = x

counts, xbins, ybins = np.histogram2d(y, x, bins=(20, 20))
sums, _, _ = np.histogram2d(y, x, weights=z, bins=(xbins, ybins))

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 4))
m1 = ax1.pcolormesh(ybins, xbins, counts, cmap='coolwarm')
plt.colorbar(m1, ax=ax1)
ax1.set_title('counts')
m2 = ax2.pcolormesh(ybins, xbins, sums, cmap='coolwarm')
plt.colorbar(m2, ax=ax2)
ax2.set_title('sums')
with np.errstate(divide='ignore', invalid='ignore'):  # suppress possible divide-by-zero warnings
    m3 = ax3.pcolormesh(ybins, xbins, sums / counts, cmap='coolwarm')
plt.colorbar(m3, ax=ax3)
ax3.set_title('mean values')
plt.tight_layout()
plt.show()


"""
def AngularMomentum(gn, sgn, centre):
    # Load information from the header
    a, h, boxsize = read_header()
    print(a)
    print(h)
    print(boxsize)
    
    # Loading the gas data
    itype = 0



    # Where we store all the data
    data = {}


    # Load data, then mask to selected GroupNumber and SubGroupNumber.
    gns  = read_dataset(itype, 'GroupNumber')
    sgns = read_dataset(itype, 'SubGroupNumber')
    mask = np.logical_and(gns == gn, sgns == sgn)

    # Load data, then mask to selected GroupNumber and SubGroupNumber.
    # Automatically converts to pcm from read_dataset, converted to pMpc
    data['Mass'] = read_dataset(itype, 'Mass')[mask] * u.g.to(u.Msun)
    data['Coordinates'] = read_dataset(itype, 'Coordinates')[mask] * u.cm.to(u.Mpc) 
    data['Velocity'] = read_dataset(itype, 'Velocity')[mask] * u.cm.to(u.Mpc)
    
    print(u.cm.to(u.Mpc))
    print(data['Coordinates'])
    
    # Periodic wrap coordinates around centre (in proper units)
    boxsize = boxsize/h
    data['Coordinates'] = np.mod(data['Coordinates']-centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
    
    print(data['Coordinates'])
    
    
    
    
# Centre from the online database
centre = np.array([12.08809, 4.474372, 1.4133347])
#centre_vel = np.array([])

AngularMomentum(1, 0, centre)
"""
