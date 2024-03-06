from snip_timescales import _analyse_tree






#================================================================  
# _50particles
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 50,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _100particles
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 100,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
                  
#================================================================             
# _noLatency
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = None, 
                  time_extra       = 0, 
                  time_no_misangle = 0,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _highLatency
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.2, 
                  time_extra       = 0.2, 
                  time_no_misangle = 0.2,  
                use_merger_criteria = False,          
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _veryhighLatency
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.5, 
                  time_extra       = 0.5, 
                  time_no_misangle = 0.5,  
                use_merger_criteria = False,          
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  

#================================================================       
# _normalLatency_anyMergers_anyMorph
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_hardMorph
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = False,          
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.35, 0.45])
                  
                  
#--------------------------------------------------------------    
# _normalLatency_anyMergers_LTG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_ETG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,     
                use_merger_criteria = False,       
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_hardLTG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG'],
                  morph_limits     = [0.35, 0.45])
                  
# _normalLatency_anyMergers_hardETG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['ETG'],
                  morph_limits     = [0.35, 0.45])
                  
                  
#--------------------------------------------------------------    
# _normalLatency_anyMergers_LTG-LTG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_ETG-ETG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,      
                use_merger_criteria = False,      
                relaxation_morph   = ['ETG-ETG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_hardLTG-LTG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.35, 0.45])
                  
# _normalLatency_anyMergers_hardETG-ETG
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = None, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,      
                use_merger_criteria = False,   
                relaxation_morph   = ['ETG-ETG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.35, 0.45])
                  
                  
#================================================================      
# _normalLatency_anyMergers_anyMorph_1010
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = 10**10, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = False,
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
                  

# _normalLatency_normalMergers_anyMorph_1010
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = 10**10, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = True,
                  min_stellar_ratio   = 0.1,       max_stellar_ratio   = 1/0.1,     # [ value ] -> set to 0 if we dont care, set to 999 if we dont care
                  min_gas_ratio       = None,      max_gas_ratio       = None,      # [ None / value ]
                  max_merger_pre      = 0.5,       max_merger_post     = 0.5,       # [0.2 + 0.5 / Gyr] -/+ max time to closest merger from point of misalignment
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_normalMergers_Major_anyMorph_1010
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = 10**10, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = True,
                  min_stellar_ratio   = 0.3,       max_stellar_ratio   = 1/0.3,     # [ value ] -> set to 0 if we dont care, set to 999 if we dont care
                  min_gas_ratio       = None,      max_gas_ratio       = None,      # [ None / value ]
                  max_merger_pre      = 0.5,       max_merger_post     = 0.5,       # [0.2 + 0.5 / Gyr] -/+ max time to closest merger from point of misalignment
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_strictMergers_anyMorph_1010
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = 10**10, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = True,
                  min_stellar_ratio   = 0.1,       max_stellar_ratio   = 1/0.1,     # [ value ] -> set to 0 if we dont care, set to 999 if we dont care
                  min_gas_ratio       = None,      max_gas_ratio       = None,      # [ None / value ]
                  max_merger_pre      = 0.2,       max_merger_post     = 0.2,       # [0.2 + 0.5 / Gyr] -/+ max time to closest merger from point of misalignment
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_strictMergers_Major_anyMorph_1010
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = 10**10, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = True,
                  min_stellar_ratio   = 0.3,       max_stellar_ratio   = 1/0.3,     # [ value ] -> set to 0 if we dont care, set to 999 if we dont care
                  min_gas_ratio       = None,      max_gas_ratio       = None,      # [ None / value ]
                  max_merger_pre      = 0.2,       max_merger_post     = 0.2,       # [0.2 + 0.5 / Gyr] -/+ max time to closest merger from point of misalignment
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_verystrictMergers_anyMorph_1010
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = 10**10, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = True,
                  min_stellar_ratio   = 0.1,       max_stellar_ratio   = 1/0.1,     # [ value ] -> set to 0 if we dont care, set to 999 if we dont care
                  min_gas_ratio       = None,      max_gas_ratio       = None,      # [ None / value ]
                  max_merger_pre      = 0.1,       max_merger_post     = 0.1,       # [0.2 + 0.5 / Gyr] -/+ max time to closest merger from point of misalignment
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_verystrictMergers_Major_anyMorph_1010
_analyse_tree(load_csv_file = False, csv_file = False,
                min_particles      = 20,
                min_stelmass       = 10**10, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = True,
                  min_stellar_ratio   = 0.3,       max_stellar_ratio   = 1/0.3,     # [ value ] -> set to 0 if we dont care, set to 999 if we dont care
                  min_gas_ratio       = None,      max_gas_ratio       = None,      # [ None / value ]
                  max_merger_pre      = 0.1,       max_merger_post     = 0.1,       # [0.2 + 0.5 / Gyr] -/+ max time to closest merger from point of misalignment
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  

                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  