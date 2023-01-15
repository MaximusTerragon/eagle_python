import h5py
import numpy as np
import math
import astropy.units as u
import matplotlib.colors as colors
import random
from astropy.constants import G
import matplotlib as mpl
import matplotlib.colors as color
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import pandas as pd
import scipy as sp
import vorbin

from read_dataset_tools import read_dataset, read_dataset_dm_mass, read_header
from pafit.fit_kinematic_pa import fit_kinematic_pa
from plotbin.sauron_colormap import register_sauron_colormap
from vorbin.voronoi_2d_binning import voronoi_2d_binning



x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 1, 1, 1, 1, 1])
err = np.array([1, 4, 4, 2, 2, 2])

    
    
        



#plt.scatter(x, y)
#plt.show()

"""rad = 1
def function1(a = 1, 
              b = 2*rad, 
              c = 3):
    
     rad = 8235
              
     egg = a + b + c
              
     return egg
     
x = function1()

print(x)"""




"""#stars_x = np.array([-3, -2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
#stars_y = np.array([-2, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, -1, 0, 1, 0, 2, 2])
#stars_vel = np.array([-50, -10, -20, -15, 5, -10, 5, 0, 5, 10, -5, 40, 30, 20, 20, 50, 40])
#stars_mass = np.array([10, 5, 10, 10, 10, 10, 5, 10, 10, 10, 5, 10, 5, 10, 10, 10, 10])

N = 100

# Set bins of 2dhist diagram
bins = 20

# Target number of pixels
target_pixels = 10
pixsize = 10/bins

np.random.seed(11111)
stars_x = np.random.normal(0, 1, N)
np.random.seed(22222)
stars_y = np.random.normal(0, 1, N)
np.random.seed(33333)
stars_vel = np.random.normal(0, 10, N)
stars_mass = np.random.rand(N) * 10

# N = 10
#[-0.05, 1.13], mass=6.73, vel= -0.53, weight=0.97, vel_weighted = -0.514
#[-0.72, 4.15], mass=9.35, vel= 29.59, weight=1.35, vel_weighted = 40.04
#mean mass = 6.91

# Scatter for visualisation
im1 = plt.scatter(stars_x, stars_y, c=stars_vel, cmap='coolwarm', s=0.5*stars_mass**2)
plt.colorbar(im1, label='velocity')
#for i, txt in enumerate(np.round(stars_vel,1)):
#    plt.annotate(txt, (stars_x[i], stars_y[i]), fontsize=6)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/trial_scatter.jpeg', dpi=300)
plt.close()



# Histogram to bin by count
counts_stars, xbins, ybins = np.histogram2d(stars_y, stars_x, bins=(bins, bins), range=[[-5,5],[-5,5]])

# Plot histogram count
im2 = plt.pcolormesh(xbins, ybins, counts_stars, cmap='Greens')
plt.colorbar(im2, label='count')
plt.scatter(stars_x, stars_y, c=stars_vel, cmap='coolwarm', s=0.5*stars_mass**2)

#for i, txt in enumerate(np.round(stars_vel,1)):
#    plt.annotate(txt, (stars_x[i], stars_y[i]), fontsize=6)
    
plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/trial_counts.jpeg', dpi=300)
plt.close()



# Histogram that measures total velocity in those bins
vel_weighted, _, _ = np.histogram2d(stars_y, stars_x, weights=stars_vel*stars_mass/np.mean(stars_mass), bins=(xbins, ybins))

# Account for numbers within hist, if divide by 0 then use 0
vel_weighted = np.divide(vel_weighted, counts_stars, out=np.zeros_like(vel_weighted), where=counts_stars!=0)
#print(vel_weighted)

# Plot mass-weighted, 2d histogram mean
im2 = plt.pcolormesh(xbins, ybins, vel_weighted, cmap='coolwarm', vmin=-50, vmax=50)
plt.colorbar(im2, label='mass-weighted mean velocity')

plt.scatter(stars_x, stars_y, c=stars_vel*stars_mass/np.mean(stars_mass), cmap='coolwarm', s=0.5*stars_mass**2)
#for i, txt in enumerate(np.round((stars_vel*stars_mass/np.mean(stars_mass)),1)):
#    plt.annotate(txt, (stars_x[i], stars_y[i]), fontsize=6)

plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/trial_velweighted.jpeg', dpi=300)
plt.close()





# Convert hist data to coords + value
i, j = np.nonzero(counts_stars)                               # find indexes of non-0 bins
a = np.array((xbins[:-1] + 0.5*(xbins[1] - xbins[0]))[i])     # convert bin to centred coords
b = np.array((ybins[:-1] + 0.5*(ybins[1] - ybins[0]))[j])     # convert bin to centred coords

# Stack points (non-0 bin values)
velocity = vel_weighted[np.nonzero(counts_stars)]             # stack non-0 bin values
bin_num  = counts_stars[np.nonzero(counts_stars)]
points = np.column_stack((b, a))                              # stack x-array and y-array into x-y pairs

# Scatter plot the centre of the bins
plt.scatter(points[:,0], points[:,1], c='k', s=0.1)
# Scatter plot the original points
plt.scatter(stars_x, stars_y, c=stars_vel, cmap='coolwarm', edgecolors='k')

plt.xlim(-5, 5)
plt.ylim(-5, 5)


# modified code taken from VorBin, returns central x,y of bin, bin count, and total velocity in each bin
plt.figure()
_, x_gen, y_gen, _, _, bin_count, vel, _, _ = voronoi_2d_binning(points[:,0], points[:,1], velocity, bin_num, target_pixels, plot=1, quiet=1, sn_func=None, pixelsize=pixsize)

# check if bin_count is using noise or actual bin counts

plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/trial_vornoi.jpeg', dpi=300)
plt.close()

#print(bin_count)


# Start new figure
plt.figure

# Create tessalation, append points at infinity to color plot edges
points = np.column_stack((x_gen, y_gen))
points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis=0)    
vel_bin = np.divide(vel, bin_count)            
vor = Voronoi(points)

print(points)
print(vel)
print(bin_count)

# normalize chosen colormap
minima = -abs(max(vel_bin, key=abs))
maxima = -minima
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)

# plot Voronoi diagram, and fill finite regions with color mapped from vel value
voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_width=1, s=1)
for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(vel_bin[r]))

# Graph formatting
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Overplot original points for comparisson
plt.scatter(stars_x, stars_y, c=stars_vel, cmap='coolwarm', s=0.5*stars_mass**2, edgecolor='k')

# colorbar
plt.colorbar(mapper)

plt.savefig('/Users/c22048063/Documents/EAGLE/trial_plots/trial_vornoi_output.jpeg', dpi=300)









# Create tessalation
vor = Voronoi(points)

# Find min/max values for normalization
minima = -abs(max(velocity, key=abs))
maxima = abs(max(velocity, key=abs))

# normalize chosen colormap
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)

# plot Voronoi diagram, and fill finite regions with color mapped from speed value
voronoi_plot_2d(vor, show_points=True, show_vertices=False, line_width=0, s=1)
for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(velocity[r]))

# Graph formatting
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Overplot original points for comparisson
plt.scatter(stars_x, stars_y, c=stars_vel, cmap='Greys')

# colorbar
plt.colorbar(mapper)
"""






""" 
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