import eagleSqlTools as sql
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from graphformat import graphformat


# Array of chosen simulations. Entries refer to the simulation name and comoving box length.
mySims = np.array([('RefL0012N0188', 12)]) #, ('RefL0050N0752', 50), ('RefL0025N0376', 25)])
#mySnaps = 

# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con = sql.connect("lms192", password="dhuKAP62")


### Stellar mass vs BH mass

for sim_name, sim_size in mySims:
    print(sim_name)
    
    # Construct and execute query for each simulation. This query returns the number of galaxies 
    # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width). 
    myQuery = "SELECT \
    			log10(SH.MassType_Star) as stelmass, \
    			log10(SH.BlackHoleMass) as bhmass \
               FROM \
			%s_Subhalo as SH \
               WHERE \
			SH.MassType_Star > 1e8 and \
			SH.MassType_BH > 0 and \
			SH.SnapNum = 28 \
               ORDER BY \
			SH.MassType_Star desc"%(sim_name)
	
        
    # Execute query.
    myData = sql.execute_query(con, myQuery)

    print("Number of subhalos over 1e8", len(myData['stelmass']))
    
    ## Graph initialising
    graphformat(8, 11, 11, 11, 11, 5, 5)
    fig, ax = plt.subplots(1, 1, figsize=[5, 5])

    
    ## Plotting the data as scatter
    divnorm = colors.Normalize(vmin=0, vmax=50)
    im = ax.hexbin(myData['stelmass'], myData['bhmass'], cmap='Spectral_r', gridsize=100, norm=divnorm, extent=[8, 12.5, 6, 10.5], mincnt=1)
    #plt.scatter(myData['stelmass'], myData['bhmass'], s=1)

    ## General formatting
    ax.set_xlim(8, 12.5)
    ax.set_ylim(6, 10.5)
    ax.set_xlabel('log$_{10}$ M$_*$ (M$_\odot$)')
    ax.set_ylabel('log$_{10}$ M$_{BH}$ (M$_\odot$)')
    ax.tick_params(axis='both', direction='in')

    ## Annotations
    ax.text(8.1, 10, "Number of galaxies\n(M$_*$ > 1e8 M$_\odot$) = %s"%str(len(myData['stelmass'])), rotation='horizontal', fontsize=10)
    ax.text(12, 6.2, "z = 0", fontsize=10)

    ## Colorbar
    cax = plt.axes([0.92, 0.11, 0.02, 0.77])
    plt.colorbar(im, cax=cax, label='number of galaxies')

    plt.suptitle("L%s: Stelmass - BH mass"%str(sim_size))
    plt.savefig("./trial_plots/MstarMbh_Z0_L%s.jpeg"%str(sim_size), format='jpeg', bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.show()