from snip_timescales import _analyse_tree


#================================================================       
# _normalLatency_anyMergers_anyMorph
print('\t_20Thresh_30Peak_normalLatency_anyMergers_anyMorph')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_anyMorph',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_hardMorph
print('\t_20Thresh_30Peak_normalLatency_anyMergers_hardMorph')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardMorph',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = False,          
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.35, 0.45])
                  
                  
#--------------------------------------------------------------    
# _normalLatency_anyMergers_LTG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_LTG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_LTG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_ETG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_ETG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_ETG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,     
                use_merger_criteria = False,       
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_hardLTG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_hardLTG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardLTG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG'],
                  morph_limits     = [0.35, 0.45])
                  
# _normalLatency_anyMergers_hardETG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_hardETG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardETG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['ETG'],
                  morph_limits     = [0.35, 0.45])
                  
                  
#--------------------------------------------------------------    
# _normalLatency_anyMergers_LTG-LTG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_LTG-LTG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_LTG-LTG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_ETG-ETG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_ETG-ETG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_ETG-ETG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,      
                use_merger_criteria = False,      
                relaxation_morph   = ['ETG-ETG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _normalLatency_anyMergers_hardLTG-LTG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_hardLTG-LTG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardLTG-LTG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.35, 0.45])
                  
# _normalLatency_anyMergers_hardETG-ETG
print('\t_20Thresh_30Peak_normalLatency_anyMergers_hardETG-ETG')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardETG-ETG',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,      
                use_merger_criteria = False,   
                relaxation_morph   = ['ETG-ETG'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.35, 0.45])
                  

#================================================================  
# _50particles
print('\t_20Thresh_30Peak_50particles')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_50particles',
                min_particles      = 50,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _100particles
print('\t_20Thresh_30Peak_100particles')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_100particles',
                min_particles      = 100,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,    
                use_merger_criteria = False,        
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
                  
#================================================================             
# _noLatency
print('\t_20Thresh_30Peak_noLatency')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_noLatency',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = None, 
                  time_extra       = 0, 
                  time_no_misangle = 0,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _highLatency
print('\t_20Thresh_30Peak_highLatency')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_highLatency',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.2, 
                  time_extra       = 0.2, 
                  time_no_misangle = 0.2,  
                use_merger_criteria = False,          
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
# _veryhighLatency
print('\t_20Thresh_30Peak_veryhighLatency')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_veryhighLatency',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.5, 
                  time_extra       = 0.5, 
                  time_no_misangle = 0.5,  
                use_merger_criteria = False,          
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
                  
#================================================================      
# _normalLatency_anyMergers_anyMorph_1010
print('\t_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010')
_analyse_tree(load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010',
                min_particles      = 20,
                min_stelmass       = 10**10, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,  
                use_merger_criteria = False,
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
                  

                  
                  
                  
                  
                  
                  
                  