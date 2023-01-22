import h5py
import numpy as np
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import csv
import json
from datetime import datetime
from tqdm import tqdm
import eagleSqlTools as sql
from graphformat import graphformat



#not sure why i needed extra aexp in _main, but this didn't used it for boxsize

# list of simulations
mySims = np.array([('RefL0012N0188', 12), ('RefL0100N1504', 100.)])  
snapNum = 28


""" 
PURPOSE
-------

Create a stellar mass function plot of a given simulation,
and can optionally import an existing sample group from
csv_load to compare this to.
"""
def _stellar_mass_func(galaxy_mass_limit = 10**8,
                        root_file = '/Users/c22048063/Documents/EAGLE/trial_plots',
                        csv_load = 'data_misalignment_2023-01-20 14/30/51.894731',
                        debug = False):
    
    
    
    for sim_name, sim_size in mySims:
        con = sql.connect('lms192', password='dhuKAP62')
        
    	# Construct and execute query for each simulation. This query returns the number of galaxies 
    	# for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width). 
        myQuery = 'SELECT \
                     0.1+floor(log10(SH.MassType_Star)/0.2)*0.2 as mass, \
                     count(*) as num \
                   FROM \
                     %s_SubHalo as SH \
                   WHERE \
			         SH.MassType_Star > %f and \
                     SH.SnapNum = %i \
                   GROUP BY \
			         0.1+floor(log10(SH.MassType_Star)/0.2)*0.2 \
                   ORDER BY \
			         mass'%(sim_name, galaxy_mass_limit, snapNum)
                    
	
        # Execute query.
        myData 	= sql.execute_query(con, myQuery)
        
        # Normalize by volume and bin width.
        hist = myData['num'][:] / (float(sim_size))**3.
        hist = hist / 0.2
        
        plt.plot(myData['mass'], np.log10(hist), label=sim_name, linewidth=2)

    # Load existing json dictionary
    #dict_new = json.load(open('%s/%s.csv' %(root_file, csv_load), 'r'))
    #new_general = dict_new['all_general']
    
    
    
    #add sample requirements
    
    
            
    # Label plot.
    plt.xlim(8.9, 12.5)
    plt.xlabel(r'log$_{10}$ M$_{*}$ [M$_{\odot}$]', fontsize=20)
    plt.ylabel(r'log$_{10}$ dn/dlog$_{10}$(M$_{*}$) [cMpc$^{-3}$]', fontsize=20)
    plt.tight_layout()
    plt.legend()
    
    plt.show()

    #plt.savefig('GSMF.png')
    plt.close()
    
#----------------------
_stellar_mass_func()
#----------------------