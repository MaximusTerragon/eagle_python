from snip_timescales_tree import _analyse_tree



#================================================================       
# _normalLatency_anyMergers_anyMorph (default)
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_anyMorph')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_anyMorph_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_hardMorph')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardMorph_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_LTG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_LTG_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_ETG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_ETG_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_hardLTG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardLTG_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_hardETG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardETG_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_LTG-LTG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_LTG-LTG_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_ETG-ETG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_ETG-ETG_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_hardLTG-LTG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardLTG-LTG_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_hardETG-ETG')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_hardETG-ETG_NEW',
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
print('\n\n\t_20Thresh_30Peak_50particles')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_50particles_NEW',
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
print('\n\n\t_20Thresh_30Peak_100particles')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_100particles_NEW',
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
# _20error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph
print('\n\n\t_20error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_NEW',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                  max_uncertainty  = 20,
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])

#_10error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph
print('\n\n\t_10error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '10error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_NEW',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 30, 
                  max_uncertainty  = 10,
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])
                  
                  
#================================================================             
# _noLatency
print('\n\n\t_20Thresh_30Peak_noLatency')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_noLatency_NEW',
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
print('\n\n\t_20Thresh_30Peak_highLatency')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_highLatency_NEW',
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
print('\n\n\t_20Thresh_30Peak_veryhighLatency')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_veryhighLatency_NEW',
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
print('\n\n\t_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010_NEW',
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
                  
                  

                  
                  
                  
#====================================================================
"""print('\n\n\t_normalLatency_anyMergers_anyMorph')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '_normalLatency_anyMergers_anyMorph_NEW',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 30,
                  peak_misangle    = 30, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])"""

"""print('\n\n\t_lowmisangle_thresh')
_analyse_tree(csv_tree = 'L100_galaxy_tree__NEW_NEW_BH', load_csv_file = False, csv_file = True, csv_name = '_lowmisangle_thresh_NEW',
                min_particles      = 20,
                min_stelmass       = None, 
                misangle_threshold = 20,
                  peak_misangle    = 20, 
                latency_time       = 0.1, 
                  time_extra       = 0.1, 
                  time_no_misangle = 0.1,   
                use_merger_criteria = False,         
                relaxation_morph   = ['LTG-LTG', 'ETG-ETG', 'LTG-ETG', 'ETG-LTG', 'other'],
                misalignment_morph = ['LTG', 'ETG'],
                  morph_limits     = [0.4, 0.4])"""
                  
                  
                  