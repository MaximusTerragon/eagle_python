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
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     l local\n     a serpens_snap\n     i snip\n")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================


f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
snip_list = f['Snapnum_Index']['Snapnum_Ref']
snap_list = f['Snapnum_Index']['Snapnum']


#np.array(f['Histories']['StellarMass']) # <- needs index


# Starts at snip 306, corresponding to a 'snap' of 151
def _sample_misalignment_snips(mySims = [('RefL0012N0188', 12)],
                              #--------------------------
                              galaxy_mass_min    = 10**9,            # Lower mass limit within 30pkpc
                              galaxy_mass_max    = 10**15,           # Lower mass limit within 30pkpc
                              snapNum            = 151,               # Target snapshot
                              use_satellites     = False,             # Whether to include SubGroupNum =/ 0
                              print_sample       = False,             # Print list of IDs
                              #--------------------------   
                              csv_file = True,                       # Will write sample to csv file in sapmle_dir
                                csv_name = '',
                              #--------------------------     
                              print_progress = False,
                              debug = True):
                         
                         
    #=====================================  
    # Create sample
    if print_progress:
        print('Creating sample')
        time_start = time.time()
    
    sample = Initial_Sample_Snip(mySims, snapNum, galaxy_mass_min, galaxy_mass_max, use_satellites)
    if debug:
        print(sample.GroupNum)
        print(sample.SubGroupNum)
        print(sample.GalaxyID)
        print(sample.SnapNum)
        print(sample.Redshift)
    if print_progress:
        print('  TIME ELAPSED: %.3f s' %(time.time() - time_start))
    
    if print_sample:
        print("  ", sample.GroupNum)
    
    print('\n===================')
    print('SAMPLE CREATED:\n  SnapNum: %s\n  Redshift: %s\n  Min Mass: %.2E M*\n  Max mass: %.2E M*\n  Satellites: %s' %(snapNum, sample.Redshift[0], galaxy_mass_min, galaxy_mass_max, use_satellites))
    print("  SAMPLE LENGTH: ", len(sample.GalaxyID))
    print('===================')
    
        
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
        
        mask = np.where((data_raw['StellarMass'] >= mstarMin) & (data_raw['StellarMass'] <= mstarMax) & (data_raw['SubGroupNumber'] <= satellite_lim) & (data_raw['SubGroupNumber'] != -1))
        
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
        
        if mstarMin >= 1E+9:
            self.MorphoKinem = np.transpose(np.array([myData['ellip'], myData['triax'], myData['kappa_stars'], myData['disp_ani'], myData['disc_to_total'], myData['rot_to_disp_ratio']]))                  
        
        
#===========================    
_sample_misalignment_snips()


#===========================    