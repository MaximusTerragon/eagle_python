

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
        output_dir      = '/home/cosmos_c22048063/outputs'
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
        output_dir      = '/home/cosmos_c22048063/outputs_snips'
        fig_dir         = EAGLE_dir + '/plots_snips'
        
        # Directories of data hdf5 file(s)
        dataDir_dict = {}
        dataDir_dict['147'] = dataDir_main + 'particledata_snip_298_z000p707/eagle_subfind_snip_particles_298_z000p707.0.hdf5'
        dataDir_dict['148'] = dataDir_main + 'particledata_snip_300_z000p687/eagle_subfind_snip_particles_300_z000p687.0.hdf5'
        dataDir_dict['149'] = dataDir_main + 'particledata_snip_302_z000p667/eagle_subfind_snip_particles_302_z000p667.0.hdf5'
        dataDir_dict['150'] = dataDir_main + 'particledata_snip_304_z000p647/eagle_subfind_snip_particles_304_z000p647.0.hdf5'
        dataDir_dict['151'] = dataDir_main + 'particledata_snip_306_z000p628/eagle_subfind_snip_particles_306_z000p628.0.hdf5'
        dataDir_dict['152'] = dataDir_main + 'particledata_snip_308_z000p609/eagle_subfind_snip_particles_308_z000p609.0.hdf5'
        dataDir_dict['153'] = dataDir_main + 'particledata_snip_310_z000p590/eagle_subfind_snip_particles_310_z000p590.0.hdf5'
        dataDir_dict['154'] = dataDir_main + 'particledata_snip_312_z000p572/eagle_subfind_snip_particles_312_z000p572.0.hdf5'
        dataDir_dict['155'] = dataDir_main + 'particledata_snip_316_z000p544/eagle_subfind_snip_particles_316_z000p544.0.hdf5'
        dataDir_dict['156'] = dataDir_main + 'particledata_snip_318_z000p527/eagle_subfind_snip_particles_318_z000p527.0.hdf5'
        dataDir_dict['157'] = dataDir_main + 'particledata_snip_320_z000p509/eagle_subfind_snip_particles_320_z000p509.0.hdf5'
        dataDir_dict['158'] = dataDir_main + 'particledata_snip_322_z000p492/eagle_subfind_snip_particles_322_z000p492.0.hdf5'
        dataDir_dict['159'] = dataDir_main + 'particledata_snip_324_z000p475/eagle_subfind_snip_particles_324_z000p475.0.hdf5'
        dataDir_dict['160'] = dataDir_main + 'particledata_snip_326_z000p459/eagle_subfind_snip_particles_326_z000p459.0.hdf5'
        dataDir_dict['161'] = dataDir_main + 'particledata_snip_328_z000p443/eagle_subfind_snip_particles_328_z000p443.0.hdf5'
        dataDir_dict['162'] = dataDir_main + 'particledata_snip_330_z000p427/eagle_subfind_snip_particles_330_z000p427.0.hdf5'
        dataDir_dict['163'] = dataDir_main + 'particledata_snip_332_z000p411/eagle_subfind_snip_particles_332_z000p411.0.hdf5'
        dataDir_dict['164'] = dataDir_main + 'particledata_snip_334_z000p396/eagle_subfind_snip_particles_334_z000p396.0.hdf5'
        dataDir_dict['165'] = dataDir_main + 'particledata_snip_336_z000p381/eagle_subfind_snip_particles_336_z000p381.0.hdf5'
        dataDir_dict['166'] = dataDir_main + 'particledata_snip_338_z000p366/eagle_subfind_snip_particles_338_z000p366.0.hdf5'
        dataDir_dict['167'] = dataDir_main + 'particledata_snip_340_z000p351/eagle_subfind_snip_particles_340_z000p351.0.hdf5'
        dataDir_dict['168'] = dataDir_main + 'particledata_snip_342_z000p337/eagle_subfind_snip_particles_342_z000p337.0.hdf5'
        dataDir_dict['169'] = dataDir_main + 'particledata_snip_344_z000p323/eagle_subfind_snip_particles_344_z000p323.0.hdf5'
        dataDir_dict['170'] = dataDir_main + 'particledata_snip_346_z000p309/eagle_subfind_snip_particles_346_z000p309.0.hdf5'
        dataDir_dict['171'] = dataDir_main + 'particledata_snip_348_z000p296/eagle_subfind_snip_particles_348_z000p296.0.hdf5'
        dataDir_dict['172'] = dataDir_main + 'particledata_snip_350_z000p282/eagle_subfind_snip_particles_350_z000p282.0.hdf5'
        dataDir_dict['173'] = dataDir_main + 'particledata_snip_352_z000p269/eagle_subfind_snip_particles_352_z000p269.0.hdf5'
        dataDir_dict['174'] = dataDir_main + 'particledata_snip_354_z000p256/eagle_subfind_snip_particles_354_z000p256.0.hdf5'
        dataDir_dict['175'] = dataDir_main + 'particledata_snip_356_z000p244/eagle_subfind_snip_particles_356_z000p244.0.hdf5'
        dataDir_dict['176'] = dataDir_main + 'particledata_snip_358_z000p231/eagle_subfind_snip_particles_358_z000p231.0.hdf5'
        dataDir_dict['177'] = dataDir_main + 'particledata_snip_360_z000p219/eagle_subfind_snip_particles_360_z000p219.0.hdf5'
        dataDir_dict['178'] = dataDir_main + 'particledata_snip_362_z000p207/eagle_subfind_snip_particles_362_z000p207.0.hdf5'
        dataDir_dict['179'] = dataDir_main + 'particledata_snip_364_z000p196/eagle_subfind_snip_particles_364_z000p196.0.hdf5'
        dataDir_dict['180'] = dataDir_main + 'particledata_snip_366_z000p184/eagle_subfind_snip_particles_366_z000p184.0.hdf5'
        dataDir_dict['181'] = dataDir_main + 'particledata_snip_368_z000p173/eagle_subfind_snip_particles_368_z000p173.0.hdf5'
        dataDir_dict['182'] = dataDir_main + 'particledata_snip_370_z000p162/eagle_subfind_snip_particles_370_z000p162.0.hdf5'
        dataDir_dict['183'] = dataDir_main + 'particledata_snip_372_z000p151/eagle_subfind_snip_particles_372_z000p151.0.hdf5'
        dataDir_dict['184'] = dataDir_main + 'particledata_snip_374_z000p140/eagle_subfind_snip_particles_374_z000p140.0.hdf5' 
        dataDir_dict['185'] = dataDir_main + 'particledata_snip_376_z000p130/eagle_subfind_snip_particles_376_z000p130.0.hdf5'
        dataDir_dict['186'] = dataDir_main + 'particledata_snip_378_z000p119/eagle_subfind_snip_particles_378_z000p119.0.hdf5'
        dataDir_dict['187'] = dataDir_main + 'particledata_snip_380_z000p109/eagle_subfind_snip_particles_380_z000p109.0.hdf5' 
        dataDir_dict['188'] = dataDir_main + 'particledata_snip_382_z000p099/eagle_subfind_snip_particles_382_z000p099.0.hdf5' 
        dataDir_dict['189'] = dataDir_main + 'particledata_snip_384_z000p089/eagle_subfind_snip_particles_384_z000p089.0.hdf5' 
        dataDir_dict['190'] = dataDir_main + 'particledata_snip_386_z000p080/eagle_subfind_snip_particles_386_z000p080.0.hdf5' 
        dataDir_dict['191'] = dataDir_main + 'particledata_snip_388_z000p070/eagle_subfind_snip_particles_388_z000p070.0.hdf5'
        dataDir_dict['192'] = dataDir_main + 'particledata_snip_390_z000p061/eagle_subfind_snip_particles_390_z000p061.0.hdf5'
        dataDir_dict['193'] = dataDir_main + 'particledata_snip_392_z000p052/eagle_subfind_snip_particles_392_z000p052.0.hdf5'
        dataDir_dict['194'] = dataDir_main + 'particledata_snip_394_z000p043/eagle_subfind_snip_particles_394_z000p043.0.hdf5'    
        dataDir_dict['195'] = dataDir_main + 'particledata_snip_396_z000p038/eagle_subfind_snip_particles_396_z000p038.0.hdf5'    
        dataDir_dict['196'] = dataDir_main + 'particledata_snip_398_z000p030/eagle_subfind_snip_particles_398_z000p030.0.hdf5'  
        dataDir_dict['197'] = dataDir_main + 'particledata_snip_400_z000p021/eagle_subfind_snip_particles_400_z000p021.0.hdf5'   
        dataDir_dict['198'] = dataDir_main + 'particledata_snip_402_z000p012/eagle_subfind_snip_particles_402_z000p012.0.hdf5'    
        dataDir_dict['199'] = dataDir_main + 'particledata_snip_404_z000p004/eagle_subfind_snip_particles_404_z000p004.0.hdf5' 
        dataDir_dict['200'] = dataDir_main + 'particledata_snip_405_z000p000/eagle_subfind_snip_particles_405_z000p000.0.hdf5'  
        
        
        #dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
        #dataDir = '/home/universe/spxtd1-shared/RefL0100N1504/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
        
        return EAGLE_dir, sample_dir, treeDir_main, output_dir, fig_dir, dataDir_dict
    # L100 snips on local   
    elif file_type == '4':
        # Directories
        EAGLE_dir       = '/Users/c22048063/Documents/EAGLE'
        dataDir_main    = ''
        treeDir_main    = '/Users/c22048063/Desktop/'
        
        # Other directories
        sample_dir      = EAGLE_dir + '/samples_snips'
        output_dir      = EAGLE_dir + '/outputs_snips'
        fig_dir         = EAGLE_dir + '/plots_snips'
        
        # Directories of data hdf5 file(s)
        dataDir_dict = {}
        dataDir_dict['147'] = dataDir_main + 'particledata_snip_298_z000p707/eagle_subfind_snip_particles_298_z000p707.0.hdf5'
        dataDir_dict['148'] = dataDir_main + 'particledata_snip_300_z000p687/eagle_subfind_snip_particles_300_z000p687.0.hdf5'
        dataDir_dict['149'] = dataDir_main + 'particledata_snip_302_z000p667/eagle_subfind_snip_particles_302_z000p667.0.hdf5'
        dataDir_dict['150'] = dataDir_main + 'particledata_snip_304_z000p647/eagle_subfind_snip_particles_304_z000p647.0.hdf5'
        dataDir_dict['151'] = dataDir_main + 'particledata_snip_306_z000p628/eagle_subfind_snip_particles_306_z000p628.0.hdf5'
        dataDir_dict['152'] = dataDir_main + 'particledata_snip_308_z000p609/eagle_subfind_snip_particles_308_z000p609.0.hdf5'
        dataDir_dict['153'] = dataDir_main + 'particledata_snip_310_z000p590/eagle_subfind_snip_particles_310_z000p590.0.hdf5'
        dataDir_dict['154'] = dataDir_main + 'particledata_snip_312_z000p572/eagle_subfind_snip_particles_312_z000p572.0.hdf5'
        dataDir_dict['155'] = dataDir_main + 'particledata_snip_316_z000p544/eagle_subfind_snip_particles_316_z000p544.0.hdf5'
        dataDir_dict['156'] = dataDir_main + 'particledata_snip_318_z000p527/eagle_subfind_snip_particles_318_z000p527.0.hdf5'
        dataDir_dict['157'] = dataDir_main + 'particledata_snip_320_z000p509/eagle_subfind_snip_particles_320_z000p509.0.hdf5'
        dataDir_dict['158'] = dataDir_main + 'particledata_snip_322_z000p492/eagle_subfind_snip_particles_322_z000p492.0.hdf5'
        dataDir_dict['159'] = dataDir_main + 'particledata_snip_324_z000p475/eagle_subfind_snip_particles_324_z000p475.0.hdf5'
        dataDir_dict['160'] = dataDir_main + 'particledata_snip_326_z000p459/eagle_subfind_snip_particles_326_z000p459.0.hdf5'
        dataDir_dict['161'] = dataDir_main + 'particledata_snip_328_z000p443/eagle_subfind_snip_particles_328_z000p443.0.hdf5'
        dataDir_dict['162'] = dataDir_main + 'particledata_snip_330_z000p427/eagle_subfind_snip_particles_330_z000p427.0.hdf5'
        dataDir_dict['163'] = dataDir_main + 'particledata_snip_332_z000p411/eagle_subfind_snip_particles_332_z000p411.0.hdf5'
        dataDir_dict['164'] = dataDir_main + 'particledata_snip_334_z000p396/eagle_subfind_snip_particles_334_z000p396.0.hdf5'
        dataDir_dict['165'] = dataDir_main + 'particledata_snip_336_z000p381/eagle_subfind_snip_particles_336_z000p381.0.hdf5'
        dataDir_dict['166'] = dataDir_main + 'particledata_snip_338_z000p366/eagle_subfind_snip_particles_338_z000p366.0.hdf5'
        dataDir_dict['167'] = dataDir_main + 'particledata_snip_340_z000p351/eagle_subfind_snip_particles_340_z000p351.0.hdf5'
        dataDir_dict['168'] = dataDir_main + 'particledata_snip_342_z000p337/eagle_subfind_snip_particles_342_z000p337.0.hdf5'
        dataDir_dict['169'] = dataDir_main + 'particledata_snip_344_z000p323/eagle_subfind_snip_particles_344_z000p323.0.hdf5'
        dataDir_dict['170'] = dataDir_main + 'particledata_snip_346_z000p309/eagle_subfind_snip_particles_346_z000p309.0.hdf5'
        dataDir_dict['171'] = dataDir_main + 'particledata_snip_348_z000p296/eagle_subfind_snip_particles_348_z000p296.0.hdf5'
        dataDir_dict['172'] = dataDir_main + 'particledata_snip_350_z000p282/eagle_subfind_snip_particles_350_z000p282.0.hdf5'
        dataDir_dict['173'] = dataDir_main + 'particledata_snip_352_z000p269/eagle_subfind_snip_particles_352_z000p269.0.hdf5'
        dataDir_dict['174'] = dataDir_main + 'particledata_snip_354_z000p256/eagle_subfind_snip_particles_354_z000p256.0.hdf5'
        dataDir_dict['175'] = dataDir_main + 'particledata_snip_356_z000p244/eagle_subfind_snip_particles_356_z000p244.0.hdf5'
        dataDir_dict['176'] = dataDir_main + 'particledata_snip_358_z000p231/eagle_subfind_snip_particles_358_z000p231.0.hdf5'
        dataDir_dict['177'] = dataDir_main + 'particledata_snip_360_z000p219/eagle_subfind_snip_particles_360_z000p219.0.hdf5'
        dataDir_dict['178'] = dataDir_main + 'particledata_snip_362_z000p207/eagle_subfind_snip_particles_362_z000p207.0.hdf5'
        dataDir_dict['179'] = dataDir_main + 'particledata_snip_364_z000p196/eagle_subfind_snip_particles_364_z000p196.0.hdf5'
        dataDir_dict['180'] = dataDir_main + 'particledata_snip_366_z000p184/eagle_subfind_snip_particles_366_z000p184.0.hdf5'
        dataDir_dict['181'] = dataDir_main + 'particledata_snip_368_z000p173/eagle_subfind_snip_particles_368_z000p173.0.hdf5'
        dataDir_dict['182'] = dataDir_main + 'particledata_snip_370_z000p162/eagle_subfind_snip_particles_370_z000p162.0.hdf5'
        dataDir_dict['183'] = dataDir_main + 'particledata_snip_372_z000p151/eagle_subfind_snip_particles_372_z000p151.0.hdf5'
        dataDir_dict['184'] = dataDir_main + 'particledata_snip_374_z000p140/eagle_subfind_snip_particles_374_z000p140.0.hdf5' 
        dataDir_dict['185'] = dataDir_main + 'particledata_snip_376_z000p130/eagle_subfind_snip_particles_376_z000p130.0.hdf5'
        dataDir_dict['186'] = dataDir_main + 'particledata_snip_378_z000p119/eagle_subfind_snip_particles_378_z000p119.0.hdf5'
        dataDir_dict['187'] = dataDir_main + 'particledata_snip_380_z000p109/eagle_subfind_snip_particles_380_z000p109.0.hdf5' 
        dataDir_dict['188'] = dataDir_main + 'particledata_snip_382_z000p099/eagle_subfind_snip_particles_382_z000p099.0.hdf5' 
        dataDir_dict['189'] = dataDir_main + 'particledata_snip_384_z000p089/eagle_subfind_snip_particles_384_z000p089.0.hdf5' 
        dataDir_dict['190'] = dataDir_main + 'particledata_snip_386_z000p080/eagle_subfind_snip_particles_386_z000p080.0.hdf5' 
        dataDir_dict['191'] = dataDir_main + 'particledata_snip_388_z000p070/eagle_subfind_snip_particles_388_z000p070.0.hdf5'
        dataDir_dict['192'] = dataDir_main + 'particledata_snip_390_z000p061/eagle_subfind_snip_particles_390_z000p061.0.hdf5'
        dataDir_dict['193'] = dataDir_main + 'particledata_snip_392_z000p052/eagle_subfind_snip_particles_392_z000p052.0.hdf5'
        dataDir_dict['194'] = dataDir_main + 'particledata_snip_394_z000p043/eagle_subfind_snip_particles_394_z000p043.0.hdf5'    
        dataDir_dict['195'] = dataDir_main + 'particledata_snip_396_z000p038/eagle_subfind_snip_particles_396_z000p038.0.hdf5'    
        dataDir_dict['196'] = dataDir_main + 'particledata_snip_398_z000p030/eagle_subfind_snip_particles_398_z000p030.0.hdf5'  
        dataDir_dict['197'] = dataDir_main + 'particledata_snip_400_z000p021/eagle_subfind_snip_particles_400_z000p021.0.hdf5'   
        dataDir_dict['198'] = dataDir_main + 'particledata_snip_402_z000p012/eagle_subfind_snip_particles_402_z000p012.0.hdf5'    
        dataDir_dict['199'] = dataDir_main + 'particledata_snip_404_z000p004/eagle_subfind_snip_particles_404_z000p004.0.hdf5' 
        dataDir_dict['200'] = dataDir_main + 'particledata_snip_405_z000p000/eagle_subfind_snip_particles_405_z000p000.0.hdf5'  
        #dataDir = '/Users/c22048063/Documents/EAGLE/data/RefL0012N0188/snapshot_0%s_z000p101/snap_0%s_z000p101.0.hdf5' %(snapNum, snapNum)
        #dataDir = '/home/universe/spxtd1-shared/RefL0100N1504/snapshot_0%s_z000p000/snap_0%s_z000p000.0.hdf5' %(snapNum, snapNum)
        
        return EAGLE_dir, sample_dir, treeDir_main, output_dir, fig_dir, dataDir_dict
        
    else:
        raise Exception('nuh-uh')
    
    
    

    


    
    
    
    