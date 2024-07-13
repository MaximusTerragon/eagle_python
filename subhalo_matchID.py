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
# casanueva paper
ID_list = [16761384, 9982142]

#ID_list = [19250448, 19615775, 18996097, 19845188, 15018060, 17559548, 14227200, 14308492, 15386755, 17611276, 63927060, 15274006, 15233503, 14471018, 15507373, 15214259, 15859420, 15569345, 14195766, 1745906, 17189709, 18281743, 17966783, 14402768, 17552187, 12659357, 18477103, 18445256, 18063592, 17791089, 8396743, 15708288, 15490693, 8283638, 8745475, 18058633, 9430664, 8546955, 9463308, 18153798, 12707838, 8787255, 10655576, 17183688, 18428672, 18359999, 15505664, 9845893, 8894369, 13154469, 9284260, 10751314, 9051081, 8797511, 13747962, 9808391, 14107961, 9788483, 8978301, 17518214, 9599139, 8429144, 13741791, 9766117, 56190924, 9044113, 8640506, 12152678, 13854599, 9014149, 8550656, 247998, 17635406, 9806843, 9805593, 9335719, 9759527, 9194557, 9946987, 9970077, 10375596, 9627417, 9762290, 10195407, 18120450, 8612765, 10108400, 10005162, 9654901, 10145406, 10446679, 17275742, 9396018, 10148850, 11513068, 10422406, 8493458, 8605415, 10304880, 13806265, 9142108, 10791968, 10007497, 10452290, 8578291, 10839221, 10078536, 10189421, 8477086, 10297941, 9409132, 13250441, 10965881, 2412731, 2823428, 2506302, 10805323, 8384309, 2958110, 8649269, 10250787, 8561075, 11334241, 8451343, 8617278, 10332220, 11164348, 10771328, 11349964, 8330576, 4481174, 8902166, 11668259, 8701932, 10246654, 10799409, 2578654, 11759492, 10521504, 11313469, 2821522, 11619654, 2816573, 10043717, 2743609, 11590970, 58282646, 11927781, 8447688, 12017675, 10789799, 12018990, 6681735]

# Auriga galaxies
#ID_list = [8677914, 8712653, 8799477, 8840875, 8894368, 8905407, 8937439, 9020523, 9114530, 9225417, 9256845, 9352515, 9355439, 9366625, 9380244, 9384896, 9397255, 9620150, 9652693, 9678375, 9744960, 9781462, 9835509, 9915689, 18131526, 18308116, 18356558, 18399452, 18452163, 18475201]
# Auriga galaxy equivolents in snips
#ID_list = [349869588, 65479759, 200582686, 14435116, 141671870, 280826716, 95589210, 14482957, 33850819, 147906152, 433737821, 467324083, 7582389, 42704061, 377471504, 37808721, 251939360, 277697191, 306562705, 306583243, 437608510, 411956841, 271490834, 353097548, 423252305, 251840829, 475874776, 75941645, 297036450, 58667562]

ID_list = [251899973]

# When given ID and snip/snap, will try to find closest match ID in snap/snip
def match_galaxyID(mySims = [('RefL0100N1504', 100)],
                    #--------------------------
                    GalaxyID_List = ID_list,
                    snap_snip     = 'snip',             # run of above ID
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
                        SH.MassType_Star as mass, \
                        FOF.Group_M_Crit200 as Rcrit \
                       FROM \
        			     %s_Subhalo as SH, \
        			     %s_FOF as FOF \
                       WHERE \
        			     SH.SnapNum = %s \
                         and ((SH.CentreOfPotential_x > %s) and (SH.CentreOfPotential_x < %s)) \
                         and ((SH.CentreOfPotential_y > %s) and (SH.CentreOfPotential_y < %s)) \
                         and ((SH.CentreOfPotential_z > %s) and (SH.CentreOfPotential_z < %s)) \
                         and ((SH.MassType_Star > %s) and (SH.MassType_Star < %s)) \
                         and SH.SubGroupNumber <= %s \
                         and SH.GroupID = FOF.GroupID \
                       ORDER BY \
        			     mass desc'%(mySims[0][0], mySims[0][0], snapnum_ref, (CoP_x-CoP_range), (CoP_x+CoP_range), (CoP_y-CoP_range), (CoP_y+CoP_range), (CoP_z-CoP_range), (CoP_z+CoP_range), (stelmass*(1-(mass_range/100))), (stelmass*(1+(mass_range/100))), (SubGroupNum + sub_range))
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
                FOF_match         = np.array(myData['Rcrit'])
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
                FOF_match         = np.array([np.array([myData])[0][9]])
                    
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
            FOF_match         = FOF_match[mask_sort]
        
        if print_galaxy:
            print('MATCHES:    %s' %len(GalaxyID_match))
            for id_i, sn_i, red_i, gn_i, sgn_i, x_i, y_i, z_i, mass_i, fof_i, d_i in zip(GalaxyID_match, SnapNum_match, Redshift_match, GroupNum_match, SubGroupNum_match, CoP_x_match, CoP_y_match, CoP_z_match, stelmass_match, FOF_match, distance[mask_sort]):
                print('|%s| %.2f |ID: %s\t|M*: %.2e |CoP:  %.2f %.2f %.2f  | FOF %.2e | %.3f kpc' %(sn_i, red_i, id_i, mass_i, x_i, y_i, z_i, fof_i, d_i)) 
            
        
        # want 13866056
        if len(GalaxyID_match) > 0:
            ID_output.append(GalaxyID_match[0])
        


#=========================
match_galaxyID(snap_snip='snip')

print('\nID_output:')
print(ID_output)
#=========================
  
            

            
            
            
            
            
            
        
    