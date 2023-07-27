

# COPY SAMPLE
#scp -r c22048063@physxlogin.astro.cf.ac.uk:/home/user/c22048063/Documents/EAGLE/samples /Users/c22048063/Documents/EAGLE/
# COPY OUTPUT
#scp -r c22048063@physxlogin.astro.cf.ac.uk:/home/cosmos_c22048063/outputs /Users/c22048063/Documents/EAGLE/

# COPY SAMPLE
#scp -r c22048063@physxlogin.astro.cf.ac.uk:/home/user/c22048063/Documents/EAGLE/samples_snips /Users/c22048063/Documents/EAGLE/
# COPY OUTPUT
#scp -r c22048063@physxlogin.astro.cf.ac.uk:/home/cosmos_c22048063/outputs_snips /Users/c22048063/Documents/EAGLE/

# COPY CODE
#scp -r /Users/c22048063/Documents/EAGLE/code  c22048063@physxlogin.astro.cf.ac.uk:/home/user/c22048063/Documents/EAGLE/


# Assigns correct output files and the likes
def _assign_directories(file_type):
    
    # L12 local on Mac
    if file_type == '1':
        # Directories
        EAGLE_dir       = '/Users/c22048063/Documents/EAGLE'
        dataDir_main    = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/'
        treeDir_main    = '/Users/c22048063/Desktop/'
        
        # Other directories
        sample_dir      = EAGLE_dir + '/samples'
        output_dir      = EAGLE_dir + '/outputs'
        fig_dir         = EAGLE_dir + '/plots'
        
        # Directories of data hdf5 file(s)
        dataDir_dict = {}
        dataDir_dict['10'] = dataDir_main + 'snapshot_010_z003p984/snap_010_z003p984.0.hdf5'
        dataDir_dict['11'] = dataDir_main + 'snapshot_011_z003p528/snap_011_z003p528.0.hdf5'
        dataDir_dict['12'] = dataDir_main + 'snapshot_012_z003p017/snap_012_z003p017.0.hdf5'
        dataDir_dict['13'] = dataDir_main + 'snapshot_013_z002p478/snap_013_z002p478.0.hdf5'
        dataDir_dict['14'] = dataDir_main + 'snapshot_014_z002p237/snap_014_z002p237.0.hdf5'
        dataDir_dict['15'] = dataDir_main + 'snapshot_015_z002p012/snap_015_z002p012.0.hdf5'
        dataDir_dict['16'] = dataDir_main + 'snapshot_016_z001p737/snap_016_z001p737.0.hdf5'
        dataDir_dict['17'] = dataDir_main + 'snapshot_017_z001p487/snap_017_z001p487.0.hdf5'
        dataDir_dict['18'] = dataDir_main + 'snapshot_018_z001p259/snap_018_z001p259.0.hdf5'
        dataDir_dict['19'] = dataDir_main + 'snapshot_019_z001p004/snap_019_z001p004.0.hdf5'
        dataDir_dict['20'] = dataDir_main + 'snapshot_020_z000p865/snap_020_z000p865.0.hdf5'
        dataDir_dict['21'] = dataDir_main + 'snapshot_021_z000p736/snap_021_z000p736.0.hdf5'
        dataDir_dict['22'] = dataDir_main + 'snapshot_022_z000p615/snap_022_z000p615.0.hdf5'
        dataDir_dict['23'] = dataDir_main + 'snapshot_023_z000p503/snap_023_z000p503.0.hdf5'
        dataDir_dict['24'] = dataDir_main + 'snapshot_024_z000p366/snap_024_z000p366.0.hdf5'
        dataDir_dict['25'] = dataDir_main + 'snapshot_025_z000p271/snap_025_z000p271.0.hdf5'
        dataDir_dict['26'] = dataDir_main + 'snapshot_026_z000p183/snap_026_z000p183.0.hdf5'
        dataDir_dict['27'] = dataDir_main + 'snapshot_027_z000p101/snap_027_z000p101.0.hdf5'
        dataDir_dict['28'] = dataDir_main + 'snapshot_028_z000p000/snap_028_z000p000.0.hdf5'
        #dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
        #dataDir = '/home/universe/spxtd1-shared/RefL0100N1504/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
        
        return EAGLE_dir, sample_dir, treeDir_main, output_dir, fig_dir, dataDir_dict
    # L100 snaps on serpens 
    elif file_type == '2':
        # Directories serpens
        EAGLE_dir       = '/home/user/c22048063/Documents/EAGLE'
        dataDir_main   = '/home/universe/spxtd1-shared/RefL0100N1504/'
        dataDir_alt    = '/home/cosmos/c22048063/EAGLE_snapshots/RefL0100N1504/'
        treeDir_main    = ''
        
        # Other directories
        sample_dir      = EAGLE_dir + '/samples'
        output_dir      = EAGLE_dir + '/outputs'
        fig_dir         = EAGLE_dir + '/plots'
        
        # Directories of data hdf5 file(s)
        dataDir_dict = {}
        dataDir_dict['10'] = dataDir_alt + 'snapshot_010_z003p984/snap_010_z003p984.0.hdf5'
        dataDir_dict['11'] = dataDir_alt + 'snapshot_011_z003p528/snap_011_z003p528.0.hdf5'
        dataDir_dict['12'] = dataDir_alt + 'snapshot_012_z003p017/snap_012_z003p017.0.hdf5'
        dataDir_dict['13'] = dataDir_alt + 'snapshot_013_z002p478/snap_013_z002p478.0.hdf5'
        dataDir_dict['14'] = dataDir_alt + 'snapshot_014_z002p237/snap_014_z002p237.0.hdf5'
        dataDir_dict['15'] = dataDir_alt + 'snapshot_015_z002p012/snap_015_z002p012.0.hdf5'
        dataDir_dict['16'] = dataDir_alt + 'snapshot_016_z001p737/snap_016_z001p737.0.hdf5'
        dataDir_dict['17'] = dataDir_alt + 'snapshot_017_z001p487/snap_017_z001p487.0.hdf5'
        dataDir_dict['18'] = dataDir_alt + 'snapshot_018_z001p259/snap_018_z001p259.0.hdf5'
        dataDir_dict['19'] = dataDir_alt + 'snapshot_019_z001p004/snap_019_z001p004.0.hdf5'
        dataDir_dict['20'] = dataDir_alt + 'snapshot_020_z000p865/snap_020_z000p865.0.hdf5'
        dataDir_dict['21'] = dataDir_alt + 'snapshot_021_z000p736/snap_021_z000p736.0.hdf5'
        dataDir_dict['22'] = dataDir_alt + 'snapshot_022_z000p615/snap_022_z000p615.0.hdf5'
        dataDir_dict['23'] = dataDir_alt + 'snapshot_023_z000p503/snap_023_z000p503.0.hdf5'
        dataDir_dict['24'] = dataDir_alt + 'snapshot_024_z000p366/snap_024_z000p366.0.hdf5'
        dataDir_dict['25'] = dataDir_main + 'snapshot_025_z000p271/snap_025_z000p271.0.hdf5'
        dataDir_dict['26'] = dataDir_main + 'snapshot_026_z000p183/snap_026_z000p183.0.hdf5'
        dataDir_dict['27'] = dataDir_main + 'snapshot_027_z000p101/snap_027_z000p101.0.hdf5'
        dataDir_dict['28'] = dataDir_main + 'snapshot_028_z000p000/snap_028_z000p000.0.hdf5'
        #dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
        #dataDir = '/home/universe/spxtd1-shared/RefL0100N1504/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
        
        return EAGLE_dir, sample_dir, treeDir_main, output_dir, fig_dir, dataDir_dict
    # L100 snips on serpens   
    elif file_type == '3':
        # Directories snipshots
        EAGLE_dir       = '/home/user/c22048063/Documents/EAGLE'
        dataDir_main    = '/home/cosmos/c22048063/EAGLE_snipshots/RefL0100N1504/'
        treeDir_main    = '/home/cosmos/c22048063/EAGLE_mergertree/'
        
        # Other directories
        sample_dir      = EAGLE_dir + '/samples_snips'
        output_dir      = EAGLE_dir + '/outputs_snips'
        fig_dir         = EAGLE_dir + '/plots_snips'
        
        # Directories of data hdf5 file(s)
        dataDir_dict = {}
        dataDir_dict['151'] = dataDir_main + 'snipshot_306_z000p628/snip_306_z000p628.0.hdf5'
        dataDir_dict['152'] = dataDir_main + 'snipshot_308_z000p609/snip_308_z000p609.0.hdf5'
        dataDir_dict['153'] = dataDir_main + 'snipshot_310_z000p590/snip_310_z000p590.0.hdf5'
        dataDir_dict['154'] = dataDir_main + 'snipshot_312_z000p572/snip_312_z000p572.0.hdf5'
        dataDir_dict['155'] = dataDir_main + 'snipshot_316_z000p544/snip_316_z000p544.0.hdf5'
        dataDir_dict['156'] = dataDir_main + 'snipshot_318_z000p527/snip_318_z000p527.0.hdf5'
        dataDir_dict['157'] = dataDir_main + 'snipshot_320_z000p509/snip_320_z000p509.0.hdf5'
        dataDir_dict['158'] = dataDir_main + 'snipshot_322_z000p492/snip_322_z000p492.0.hdf5'
        dataDir_dict['159'] = dataDir_main + 'snipshot_324_z000p475/snip_324_z000p475.0.hdf5'
        dataDir_dict['160'] = dataDir_main + 'snipshot_326_z000p459/snip_326_z000p459.0.hdf5'
        dataDir_dict['161'] = dataDir_main + 'snipshot_328_z000p443/snip_328_z000p443.0.hdf5'
        dataDir_dict['162'] = dataDir_main + 'snipshot_330_z000p427/snip_330_z000p427.0.hdf5'
        dataDir_dict['163'] = dataDir_main + 'snipshot_332_z000p411/snip_332_z000p411.0.hdf5'
        dataDir_dict['164'] = dataDir_main + 'snipshot_334_z000p396/snip_334_z000p396.0.hdf5'
        dataDir_dict['165'] = dataDir_main + 'snipshot_336_z000p381/snip_336_z000p381.0.hdf5'
        dataDir_dict['166'] = dataDir_main + 'snipshot_338_z000p366/snip_338_z000p366.0.hdf5'      
        
        
        #dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
        #dataDir = '/home/universe/spxtd1-shared/RefL0100N1504/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
        
        return EAGLE_dir, sample_dir, treeDir_main, output_dir, fig_dir, dataDir_dict
        
    else:
        raise Exception('nuh-uh')
    
    
    

    


    
    
    
    