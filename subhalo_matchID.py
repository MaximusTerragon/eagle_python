import h5py
import numpy as np
import math
import random
import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import astropy.units as u
from astropy.cosmology import z_at_value, FlatLambdaCDM
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID, ConvertID_noMK, ConvertID_snip, MergerTree
import eagleSqlTools as sql
from graphformat import set_rc_params
from read_dataset_directories import _assign_directories


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local\n")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#====================================


ID_list = [15851557, 3327115, 10438463, 10670173, 13866051, 17480553, 8077031, 8494196, 8763330, 9777730, 10009377, 10145405, 14216102, 16049479, 17374402, 17718284, 18447769]
ID_output = []
ID_list = [16761384, 9982142]

# When given ID and snip/snap, will try to find closest match ID in snap/snip
def match_galaxyID(mySims = [('RefL0100N1504', 100)],
                    #--------------------------
                    GalaxyID_List = ID_list,
                    snap_snip     = 'snap',             # run of above ID
                    #--------------------------
                    # Extract galaxies within range
                    CoP_range  = 0.02,                   # Mpc
                    mass_range = 50,                    # percent   20 = 20%
                    sub_range = 20,                      # plus/minus SubGroupNumber                 
                    #--------------------------
                    print_progress = False,
                    print_galaxy   = True,
                    debug = False):
                    
    if print_progress:
        print('Extracting GroupNum, SubGroupNum, SnapNum lists')
        time_start = time.time()
    
    assert answer == '4', 'need to use: 4 snip local'
    
    
    #-----------------------
    # Run through IDs
    for GalaxyID in tqdm(GalaxyID_List):
        
        #-----------------------
        # Extract snap/snipshot, stellar mass, gn, sgn, and coordinates
        if snap_snip == 'snap':
            # SQL details
            con = sql.connect("lms192", password="dhuKAP62")
    
            # Construct and execute query for each simulation. This query returns properties for a single galaxy
            myQuery = 'SELECT \
                        SH.GroupNumber, \
                        SH.SubGroupNumber, \
                        SH.SnapNum, \
                        SH.Redshift, \
                        SH.CentreOfPotential_x as x, \
                        SH.CentreOfPotential_y as y, \
                        SH.CentreOfPotential_z as z, \
                        SH.MassType_Star as mass \
                       FROM \
        			     %s_Subhalo as SH \
                       WHERE \
        			     SH.GalaxyID = %s'%(mySims[0][0], GalaxyID)

            # Execute query.
            myData = sql.execute_query(con, myQuery)
            
            GroupNum    = myData['GroupNumber']
            SubGroupNum = myData['SubGroupNumber']
            SnapNum     = myData['SnapNum']
            Redshift    = myData['Redshift']
            CoP_x       = myData['x']
            CoP_y       = myData['y']
            CoP_z       = myData['z']
            stelmass    = myData['mass']
        elif snap_snip == 'snip':
            # This will navigate the merger tree to find the snap and other properties of a given galaxy 
            f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
    
            # Find snapNum
            row_mask, snapNum = np.where(np.array(f['Histories']['GalaxyID']) == GalaxyID)
            row_mask = row_mask[0]
            SnapNum  = snapNum[0]
            Redshift = f['Snapnum_Index']['Redshift'][SnapNum]
            
            myData = {}
            for header_name in ['GroupNumber', 'SubGroupNumber', 'GalaxyID', 'CentreOfPotential_x', 'CentreOfPotential_y', 'CentreOfPotential_z', 'StellarMass']:
                myData[header_name] = f['Histories'][header_name][row_mask,SnapNum]
            
            f.close()
            
            GroupNum    = myData['GroupNumber']
            SubGroupNum = myData['SubGroupNumber']
            CoP_x       = myData['CentreOfPotential_x']
            CoP_y       = myData['CentreOfPotential_y']
            CoP_z       = myData['CentreOfPotential_z']
            stelmass    = myData['StellarMass'] 
        else:
            raise Exception('Could not read snap_snip')
        
        if print_galaxy:
            print('INPUT:')
            print('|%s| %.2f |ID: %s\t|M*: %.2e |CoP: %.2f %.2f %.2f | ' %(SnapNum, Redshift, GalaxyID, stelmass, CoP_x, CoP_y, CoP_z)) 
            
            
        
        #-----------------------
        # Left with: 
        # GroupNum, SubGroupNum, SnapNum, Redshift, stelmass, CoP_x, CoP_y, CoP_z, GalaxyID
        
        if debug:
            print('GalaxyID', GalaxyID)
            print('GroupNum', GroupNum)
            print('SubGroupNum', SubGroupNum)
            print('SnapNum', SnapNum)
            print('Redshift', Redshift)
            print('Stelmass', stelmass)
            print('Center  [%.3f  %.3f  %.3f]' %(CoP_x, CoP_y, CoP_z))
            
            
        #-----------------------
        # Find closest snap/snip and extract closest matches snap/snipshot, stellar mass, gn, sgn, and coordinates
        if snap_snip == 'snap':
            # This will navigate the merger tree to find the snap and other properties of a given galaxy 
            f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
            
            mask_snip     = np.argmin(np.absolute(np.array(f['Snapnum_Index']['Redshift']) - Redshift))
            mask_criteria = (np.array(f['Histories']['StellarMass'][:,mask_snip]) > stelmass*(1-(mass_range/100))) & (np.array(f['Histories']['StellarMass'][:,mask_snip]) < stelmass*(1+(mass_range/100))) & (np.array(f['Histories']['CentreOfPotential_x'][:,mask_snip]) > (CoP_x-CoP_range)) & (np.array(f['Histories']['CentreOfPotential_x'][:,mask_snip]) < (CoP_x+CoP_range)) & (np.array(f['Histories']['CentreOfPotential_y'][:,mask_snip]) > (CoP_y-CoP_range)) & (np.array(f['Histories']['CentreOfPotential_y'][:,mask_snip]) < (CoP_y+CoP_range)) & (np.array(f['Histories']['CentreOfPotential_z'][:,mask_snip]) > (CoP_z-CoP_range)) & (np.array(f['Histories']['CentreOfPotential_z'][:,mask_snip]) < (CoP_z+CoP_range)) & (np.array(f['Histories']['SubGroupNumber'][:,mask_snip]) <= SubGroupNum + sub_range)
            
            # Assign variables
            GalaxyID_match    = np.array(f['Histories']['GalaxyID'][mask_criteria,mask_snip])
            Redshift_match    = np.full(len(GalaxyID_match), f['Snapnum_Index']['Redshift'][mask_snip])
            SnapNum_match     = np.full(len(GalaxyID_match), f['Snapnum_Index']['Snapnum'][mask_snip])
            GroupNum_match    = np.array(f['Histories']['GroupNumber'][mask_criteria,mask_snip])
            SubGroupNum_match = np.array(f['Histories']['SubGroupNumber'][mask_criteria,mask_snip])
            CoP_x_match       = np.array(f['Histories']['CentreOfPotential_x'][mask_criteria,mask_snip])
            CoP_y_match       = np.array(f['Histories']['CentreOfPotential_y'][mask_criteria,mask_snip])
            CoP_z_match       = np.array(f['Histories']['CentreOfPotential_z'][mask_criteria,mask_snip])
            stelmass_match    = np.array(f['Histories']['StellarMass'][mask_criteria,mask_snip])    
            
            f.close()
            
            # Find distance to CoP
            distance      = np.sqrt((CoP_x_match - CoP_x)**2 + (CoP_y_match - CoP_y)**2 + (CoP_z_match - CoP_z)**2)
            mask_sort     = np.argsort(distance)
            
            # Sort by distance
            GalaxyID_match    = GalaxyID_match[mask_sort]
            Redshift_match    = Redshift_match[mask_sort]
            SnapNum_match     = SnapNum_match[mask_sort]
            GroupNum_match    = GroupNum_match[mask_sort]
            SubGroupNum_match = SubGroupNum_match[mask_sort]
            CoP_x_match       = CoP_x_match[mask_sort]
            CoP_y_match       = CoP_y_match[mask_sort]
            CoP_z_match       = CoP_z_match[mask_sort]
            stelmass_match    = stelmass_match[mask_sort]
        elif snap_snip == 'snip':
            z_snaps = np.array([20.00, 15.13, 9.99, 8.99, 8.07, 7.05, 5.97, 5.49, 5.04, 4.49, 3.98, 3.53, 3.02, 2.48, 2.24, 2.01, 1.74, 1.49, 1.26, 1.00, 0.87, 0.74, 0.62, 0.50, 0.37, 0.27, 0.18, 0.10, 0.00])
            snapnum_ref = np.argmin(np.absolute(z_snaps - Redshift))
        
            # SQL details
            con = sql.connect("lms192", password="dhuKAP62")
    
            # Construct and execute query for each simulation. This query returns properties for a single galaxy
            myQuery = 'SELECT \
                        SH.GroupNumber, \
                        SH.SubGroupNumber, \
                        SH.GalaxyID, \
                        SH.SnapNum, \
                        SH.Redshift, \
                        SH.CentreOfPotential_x as x, \
                        SH.CentreOfPotential_y as y, \
                        SH.CentreOfPotential_z as z, \
                        SH.MassType_Star as mass \
                       FROM \
        			     %s_Subhalo as SH \
                       WHERE \
        			     SH.SnapNum = %s \
                         and ((SH.CentreOfPotential_x > %s) and (SH.CentreOfPotential_x < %s)) \
                         and ((SH.CentreOfPotential_y > %s) and (SH.CentreOfPotential_y < %s)) \
                         and ((SH.CentreOfPotential_z > %s) and (SH.CentreOfPotential_z < %s)) \
                         and ((SH.MassType_Star > %s) and (SH.MassType_Star < %s)) \
                         and SH.SubGroupNumber <= %s \
                       ORDER BY \
        			     mass desc'%(mySims[0][0], snapnum_ref, (CoP_x-CoP_range), (CoP_x+CoP_range), (CoP_y-CoP_range), (CoP_y+CoP_range), (CoP_z-CoP_range), (CoP_z+CoP_range), (stelmass*(1-(mass_range/100))), (stelmass*(1+(mass_range/100))), (SubGroupNum + sub_range))
            # Execute query.
            myData = sql.execute_query(con, myQuery)
            
            if myData.size > 1:
                GalaxyID_match    = np.array(myData['GalaxyID'])
                Redshift_match    = np.array(myData['Redshift'])
                SnapNum_match     = np.array(myData['SnapNum'])
                GroupNum_match    = np.array(myData['GroupNumber'])
                SubGroupNum_match = np.array(myData['SubGroupNumber'])
                CoP_x_match       = np.array(myData['x'])
                CoP_y_match       = np.array(myData['y'])
                CoP_z_match       = np.array(myData['z'])
                stelmass_match    = np.array(myData['mass'])
            else:
                GalaxyID_match    = np.array([np.array([myData])[0][2]])
                Redshift_match    = np.array([np.array([myData])[0][4]])
                SnapNum_match     = np.array([np.array([myData])[0][3]])
                GroupNum_match    = np.array([np.array([myData])[0][0]])
                SubGroupNum_match = np.array([np.array([myData])[0][1]])
                CoP_x_match       = np.array([np.array([myData])[0][5]])
                CoP_y_match       = np.array([np.array([myData])[0][6]])
                CoP_z_match       = np.array([np.array([myData])[0][7]])
                stelmass_match    = np.array([np.array([myData])[0][8]])
                    
            # Find distance to CoP
            distance      = np.sqrt((CoP_x_match - CoP_x)**2 + (CoP_y_match - CoP_y)**2 + (CoP_z_match - CoP_z)**2)
            mask_sort     = np.argsort(distance)
            
            # Sort by distance
            GalaxyID_match    = GalaxyID_match[mask_sort]
            Redshift_match    = Redshift_match[mask_sort]
            SnapNum_match     = SnapNum_match[mask_sort]
            GroupNum_match    = GroupNum_match[mask_sort]
            SubGroupNum_match = SubGroupNum_match[mask_sort]
            CoP_x_match       = CoP_x_match[mask_sort]
            CoP_y_match       = CoP_y_match[mask_sort]
            CoP_z_match       = CoP_z_match[mask_sort]
            stelmass_match    = stelmass_match[mask_sort]
        
        if print_galaxy:
            print('MATCHES:    %s' %len(GalaxyID_match))
            for id_i, sn_i, red_i, gn_i, sgn_i, x_i, y_i, z_i, mass_i, d_i in zip(GalaxyID_match, SnapNum_match, Redshift_match, GroupNum_match, SubGroupNum_match, CoP_x_match, CoP_y_match, CoP_z_match, stelmass_match, distance[mask_sort]):
                print('|%s| %.2f |ID: %s\t|M*: %.2e |CoP:  %.2f %.2f %.2f  | %.3f kpc' %(sn_i, red_i, id_i, mass_i, x_i, y_i, z_i, d_i)) 
            
        
        # want 13866056
        ID_output.append(GalaxyID_match[0])


#=========================
match_galaxyID()

print('\nID_output:')
print(ID_output)
#=========================
  
            

            
            
            
            
            
            
        
    