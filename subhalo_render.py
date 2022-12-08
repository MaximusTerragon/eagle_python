import h5py
import numpy as np
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from subhalo_main import Subhalo_Extract, Subhalo
from graphformat import graphformat

# Directories of data hdf5 file(s)
dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_028_z000p000/snap_028_z000p000.0.hdf5'

# list of simulations
mySims = np.array([('RefL0012N0188', 12)])   
snapNum = 28
    
#1, 4, 7, 16
#1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21
def galaxy_render(GroupNumList = np.array([1, 2, 3, 4, 6, 5, 7, 9, 14, 16, 11, 8, 13, 12, 15, 18, 10, 20, 22, 24, 21]),
                  SubGroupNum  = 0, 
                  particles    = 10000,    #5000,10000
                  minangle     = 0,
                  maxangle     = 0, 
                  stepangle    = 30, 
                  spin_rad_in  = np.arange(0.5, 10.5, 0.5),     # multiples of rad
                  trim_rad_in  = 100,                      # trim particles <radius, False, 'rad', 'tworad', num [pkpc]
                  kappa_rad_in = 30,                            # calculate kappa for this radius [pkpc]
                  align_rad_in = False, #False                    # align galaxy to stellar vector in. this radius [pkpc]
                  boxradius_in = 40,                # boxradius of render
                  root_file = 'trial_plots',        # 'trial_plots' or 'plots'
                  print_galaxy = True,              # print galaxy stats in chat
                  txt_file     = False,              # create a txt file with print data
                  stars        = True,
                  gas_sf       = True,
                  gas_nsf      = True,
                  orientate_to_axis = 'z',          
                  viewing_angle     = 0,            # Keep as 0
                  plot_spin_vectors = True,
                  spin_vector_rad   = 'tworad',      # radius of spinvector to display
                  centre_of_pot     = True, 
                  centre_of_mass    = False,
                  axis              = True,
                  savefig           = False,
                  plotshow          = True):
        
    for GroupNum in GroupNumList:         
        # Initial extraction of galaxy data
        galaxy = Subhalo_Extract(mySims, dataDir, snapNum, GroupNum, SubGroupNum)
        
        # Detect keywords of rad, tworad. All in [pkpc]
        spin_rad = spin_rad_in * galaxy.halfmass_rad
        
        if trim_rad_in == 'rad':
            trim_rad = galaxy.halfmass_rad
        elif trim_rad_in == 'tworad':
            trim_rad = 2*galaxy.halfmass_rad
        else:
            trim_rad = trim_rad_in
            
        if kappa_rad_in == 'rad':
            kappa_rad = galaxy.halfmass_rad
        elif kappa_rad_in == 'tworad':
            kappa_rad = 2*galaxy.halfmass_rad
        else:
            kappa_rad = kappa_rad_in
            
        if align_rad_in == 'rad':
            align_rad = galaxy.halfmass_rad
        elif align_rad_in == 'tworad':
            align_rad = 2*galaxy.halfmass_rad
        else:
            align_rad = align_rad_in
            
        if boxradius_in == 'rad':
            boxradius = galaxy.halfmass_rad
        elif boxradius_in == 'tworad':
            boxradius = 2*galaxy.halfmass_rad
        else:
            boxradius = boxradius_in
            
        # Galaxy will be rotated to calc_kappa_rad's stellar spin value
        subhalo = Subhalo(galaxy.halfmass_rad, galaxy.centre, galaxy.centre_mass, galaxy.perc_vel, galaxy.stars, galaxy.gas,
                                            viewing_angle,
                                            spin_rad,
                                            trim_rad, 
                                            kappa_rad, 
                                            align_rad,              #align_rad = False
                                            orientate_to_axis,
                                            quiet=True)
                                                                    
        # Print galaxy properties
        if print_galaxy == True:
            print('\nGROUP NUMBER:           %s' %str(subhalo.gn)) 
            print('STELLAR MASS [Msun]:    %.3f' %np.log10(subhalo.stelmass))       # [Msun]
            print('HALFMASS RAD [pkpc]:    %.3f' %subhalo.halfmass_rad)             # [pkpc]
            print('KAPPA:                  %.2f' %subhalo.kappa)
            print('KAPPA RAD CALC [pkpc]:  %s'   %str(kappa_rad_in))
            print(' HALF-\tANGLES (STARS-)\t\tPARTICLE COUNT\t\t\tMASS')
            print(' RAD\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
            for i in [1, 3, len(spin_rad_in)-1]:
                with np.errstate(divide='ignore', invalid='ignore'):
                    print(' %.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.mis_angles['rad'][i]/subhalo.halfmass_rad, subhalo.mis_angles['gas'][i], subhalo.mis_angles['gas_sf'][i], subhalo.mis_angles['gas_nsf'][i], subhalo.particles['stars'][i], subhalo.particles['gas'][i], subhalo.particles['gas_sf'][i], subhalo.particles['gas_nsf'][i], np.log10(subhalo.particles['stars_mass'][i]), np.log10(subhalo.particles['gas_mass'][i]), np.log10(subhalo.particles['gas_sf_mass'][i]), np.log10(subhalo.particles['gas_nsf_mass'][i])))        
            print('CENTRE [pMpc]:      [%.5f,\t%.5f,\t%.5f]' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))        # [pkpc]
            print('PERC VEL [pkm/s]:   [%.5f,\t%.5f,\t%.5f]' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))  # [pkm/s]
            #print('VIEWING ANGLES: ', end='')
            
        # Graph initialising and base formatting
        graphformat(8, 11, 11, 11, 11, 5, 5)
        fig = plt.figure() 
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax, computed_zorder=False)
        
        # Plot formatting
        ax.set_facecolor('xkcd:black')
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        ax.grid(False)
        ax.set_xlim(-boxradius, boxradius)
        ax.set_ylim(-boxradius, boxradius)
        ax.set_zlim(-boxradius, boxradius)
        
        def plot_rand_scatter(dict_name, part_type, color):
            # Keep same seed
            np.random.seed(42)
            
            # Selecting N (particles) sets of coordinates
            if dict_name[part_type]['Coordinates'].shape[0] <= particles:
                coords = dict_name[part_type]['Coordinates'][np.random.choice(dict_name[part_type]['Coordinates'].shape[0], dict_name[part_type]['Coordinates'].shape[0], replace=False), :]
            else:
                coords = dict_name[part_type]['Coordinates'][np.random.choice(dict_name[part_type]['Coordinates'].shape[0], particles, replace=False), :]
            
            # Plot scatter
            ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=0.02, alpha=0.9, c=color, zorder=4)
           
        
        def plot_spin_vector(dict_name, part_type, rad, color):
            if rad == 'rad':
                mask = np.where(dict_name['rad']/subhalo.halfmass_rad == 1.)
            elif rad == 'tworad':
                mask = np.where(dict_name['rad']/subhalo.halfmass_rad == 2.)
            else:
                mask = np.where(dict_name['rad'] > rad)
            
            arrow = dict_name[part_type][int(min(mask))]
            
            # Plot original stars spin vector
            ax.quiver(0, 0, 0, arrow[0]*boxradius*0.6, arrow[1]*boxradius*0.6, arrow[2]*boxradius*0.6, color=color, alpha=1, linewidth=1, zorder=50)

        # Plot scatters and spin vectors   
        if not align_rad:
            if stars:
                plot_rand_scatter(subhalo.data, 'stars', 'lightyellow')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins, 'stars', spin_vector_rad, 'r')
            if gas_sf:
                plot_rand_scatter(subhalo.data, 'gas_sf', 'cyan')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins, 'gas_sf', spin_vector_rad, 'darkturquoise')
            if gas_nsf:
                plot_rand_scatter(subhalo.data, 'gas_nsf', 'royalblue')  
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins, 'gas_nsf', spin_vector_rad, 'blue')   
        elif align_rad:
            if stars:
                plot_rand_scatter(subhalo.data_align, 'stars', 'lightyellow')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins_align, 'stars', spin_vector_rad, 'r')
            if gas_sf:
                plot_rand_scatter(subhalo.data_align, 'gas_sf', 'cyan')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins_align, 'gas_sf', spin_vector_rad, 'teal')
            if gas_nsf:
                plot_rand_scatter(subhalo.data_align, 'gas_nsf', 'royalblue')
                if plot_spin_vectors:
                    plot_spin_vector(subhalo.spins_align, 'gas_nsf', spin_vector_rad, 'blue')
        
        # Plot centre of potential (0,0,0) and mass
        if centre_of_pot == True:
            ax.scatter(0, 0, 0, c='pink', s=3, zorder=10)
        if centre_of_mass == True:
            ax.scatter(subhalo.centre_mass[0] - subhalo.centre[0], subhalo.centre_mass[1] - subhalo.centre[1], subhalo.centre_mass[2] - subhalo.centre[2], c='purple', s=3, zorder=10)
        
        # Plot axis
        if axis == True:
            ax.quiver(0, 0, -boxradius, boxradius/3, 0, 0, color='r', linewidth=0.5)
            ax.quiver(0, 0, -boxradius, 0, boxradius/3, 0, color='g', linewidth=0.5)
            ax.quiver(0, 0, -boxradius, 0, 0, boxradius/3, color='b', linewidth=0.5)
                          
        
        for ii in np.arange(minangle, maxangle+1, stepangle):
            #print(ii , end=' ')                 # [deg]
            ax.view_init(0, ii)

            if savefig:
                if (stars == True) & (gas_sf == True) & (gas_nsf == True) & (align_rad_in == False):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_all_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == True) & (gas_sf == False) & (gas_nsf == False) & (align_rad_in == False):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_s_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == False) & (gas_sf == True) & (gas_nsf == True) & (align_rad_in == False):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_sf_nsf_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == False) & (gas_sf == True) & (gas_nsf == False) & (align_rad_in == False):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_sf_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == False) & (gas_sf == False) & (gas_nsf == True) & (align_rad_in == False):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_nsf_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == True) & (gas_sf == True) & (gas_nsf == True):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_rot_all_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == True) & (gas_sf == False) & (gas_nsf == False):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_rot_s_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == False) & (gas_sf == True) & (gas_nsf == False):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_rot_sf_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)
                elif (stars == False) & (gas_sf == False) & (gas_nsf == True):
                    plt.savefig("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render_rot_nsf_%s_%s_%s.jpeg" %(str(root_file), str(GroupNum), str(trim_rad_in), str(spin_vector_rad), str(ii)), dpi=300)

        if plotshow:
            plt.show()
            
        plt.close()
        
        # Create txt file with output for that galaxy
        if txt_file == True:
            dash = '-' * 100
            f = open("/Users/c22048063/Documents/EAGLE/%s/galaxy_%s/render.txt" %(str(root_file), str(GroupNum)), 'w+')
            f.write(dash)
            f.write('\nGROUP NUMBER:           %s' %str(subhalo.gn))
            f.write('\nSUBGROUP NUMBER:        %s' %str(subhalo.sgn))
            f.write('\n' + dash)
            f.write('\nTOTAL STELLAR MASS:     %.3f   \t[Msun]' %np.log10(subhalo.stelmass))     # [pMsun]
            f.write('\nTOTAL GAS MASS:         %.3f   \t[Msun]' %np.log10(subhalo.gasmass))      # [pMsun]
            f.write('\nTOTAL SF GAS MASS:      %.3f   \t[Msun]' %np.log10(subhalo.gasmass_sf))   # [pMsun]
            f.write('\nTOTAL NON-SF GAS MASS:  %.3f   \t[Msun]' %np.log10(subhalo.gasmass_nsf))  # [pMsun]
            f.write('\n' + dash)
            f.write('\nHALFMASS RAD:           %.3f   \t[pkpc]' %subhalo.halfmass_rad)   # [pkpc]
            f.write('\n' + dash)
            f.write('\nKAPPA:                  %.2f' %subhalo.kappa)  
            f.write('\nKAPPA RADIUS CALC:      %s     \t\t[pkpc]' %str(kappa_rad_in))
            f.write('\n' + dash)
            f.write('\nMISALIGNMENT ANGLES [deg]:')
            f.write('\nHALF-\tANGLES (STARS-)\t\tPARTICLE COUNT\t\t\tMASS')
            f.write('\nRAD\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF\tSTARS\tGAS\tSF\tNSF')
            i = 0
            while i < len(subhalo.mis_angles['rad']):
                with np.errstate(divide="ignore"):
                    f.write('\n%.1f\t%.1f\t%.1f\t%.1f\t%i\t%i\t%i\t%i\t%.1f\t%.1f\t%.1f\t%.1f' %(subhalo.mis_angles['rad'][i]/subhalo.halfmass_rad, subhalo.mis_angles['gas'][i], subhalo.mis_angles['gas_sf'][i], subhalo.mis_angles['gas_nsf'][i], subhalo.particles['stars'][i], subhalo.particles['gas'][i], subhalo.particles['gas_sf'][i], subhalo.particles['gas_nsf'][i], np.log10(subhalo.particles['stars_mass'][i]), np.log10(subhalo.particles['gas_mass'][i]), np.log10(subhalo.particles['gas_sf_mass'][i]), np.log10(subhalo.particles['gas_nsf_mass'][i])))        
                i += 1
            f.write('\n' + dash)
            f.write('\nCENTRE:          [%.5f,\t%.5f,\t%.5f]\t[pMpc]\n' %(subhalo.centre[0]/1000, subhalo.centre[1]/1000, subhalo.centre[2]/1000))                                 # [pMpc]
            f.write('PERC VELOCITY:   [%.5f,\t%.5f,\t%.5f]\t[pkm/s]\n' %(subhalo.perc_vel[0], subhalo.perc_vel[1], subhalo.perc_vel[2]))
            f.write('\n' + dash)
            
            f.close()
        
        print('')
        
        
galaxy_render()
