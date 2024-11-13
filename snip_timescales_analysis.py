import h5py
import numpy as np
import scipy
from scipy import stats
import math
import random
import uuid
import hashlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, NullFormatter, ScalarFormatter, FuncFormatter)
import seaborn as sns
import pandas as pd
from plotbin.sauron_colormap import register_sauron_colormap
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import astropy.units as u
from astropy.cosmology import z_at_value, FlatLambdaCDM
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID, ConvertID_noMK, MergerTree
import eagleSqlTools as sql
from graphformat import set_rc_params, lighten_color
from read_dataset_directories import _assign_directories
from extract_misalignment_trees import _extract_tree


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
register_sauron_colormap()
#====================================


#-------------------------
# Given ID at z=0, will look for misalignments in tree            
def _find_misalignment_evolution(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Find IDs:
                        GalaxyID_list_z0 = [453139689, 251899973],    # GalaxyID at z=0
                        ID_search_range   = 80,                                 # GalaxyID + value to look in misalignment_tree
                      #-----------------------------
                      debug = False):
    
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax']
    #-------------------------
    
    
    #==========================================================================
    # Gather data, average stelmass over misalignment
    for GalaxyID_i in GalaxyID_list_z0:
        print('\nID: %s' %GalaxyID_i)
        
        for ID_i in misalignment_tree.keys():
        
            if int(ID_i) in np.arange(GalaxyID_i, GalaxyID_i+ID_search_range+1, 1):
                print('\n  Found misalignment:')
                print('     lookback: %.2f Gyr' %np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s'] + 1])
                print('     initial misangle: %.2f deg' %np.array(misalignment_tree['%s' %ID_i]['stars_gas_sf'])[misalignment_tree['%s' %ID_i]['index_s'] + 1])
                print('     trelax        : \t%.2f Gyr' %misalignment_tree['%s' %ID_i]['relaxation_time'])
                print('     trelax/tdyn   : \t%.2f tdyn' %misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
                print('     trelax/ttorque: \t%.2f ttorque' %misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
    
                       
#-------------------------
# Plot sample histogram of misalignments extracted              
def _plot_sample_hist(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                        set_bin_width_mass                  = 0.01,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
    
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax']
    #-------------------------
    
    
    #==========================================================================
    # Gather data, average stelmass in first misaligned snipshot
    stelmass_plot = []
    for ID_i in misalignment_tree.keys():
        stelmass_plot.append(np.log10(np.array(misalignment_tree['%s' %ID_i]['stelmass'])[misalignment_tree['%s' %ID_i]['index_s']+1]))
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    axs.hist(stelmass_plot, bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), histtype='bar', edgecolor='none', facecolor='b', alpha=0.1)
    axs.hist(stelmass_plot, bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), histtype='bar', edgecolor='b', facecolor='none', alpha=1.0)
    
    print('Median stellar mass: \t%.2e.' %(10**np.median(stelmass_plot)))
    
    #-------------
    ### Formatting
    axs.set_xlabel(r'log$_{10}$ M$_{*}$ ($2r_{50}$) [M$_{\odot}$]')
    axs.set_ylabel('Galaxies in sample')
    axs.set_xticks(np.arange(9, 12.1, 0.5))
    axs.set_xlim(9, 12.1)
    
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': 'MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%ssample_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%ssample_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
#-------------------------
# Returns values of average particle count of relaxations
def _average_particle_count(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    particle_count_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        # Average particle count during relaxation
        particle_count_array.append(np.mean(misalignment_tree['%s' %ID_i]['sfparticlecount_1hmr'][misalignment_tree['%s' %ID_i]['index_s']:misalignment_tree['%s' %ID_i]['index_r']]))
        ID_plot.append(ID_i)
    
            
    print('  Using sample: ', len(ID_plot))
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10, 3], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    axs.hist(particle_count_array, bins=np.arange(0, 501, 1), histtype='bar', edgecolor='none', facecolor='b', alpha=0.1)
    axs.hist(particle_count_array, bins=np.arange(0, 501, 1), histtype='bar', edgecolor='b', facecolor='none', alpha=1.0)
    
    print('Median sf particle coun: \t%.2e.' %(np.median(particle_count_array)))
    
    #-------------
    ### Formatting
    axs.set_xlabel('mean gas_sf particle count while relaxing ($r_{50}$) [count]')
    axs.set_ylabel('Galaxies in sample')
    axs.set_xticks(np.arange(0, 501, 20))
    axs.set_xlim(0, 500)
    axs.set_ylim(0, 35)
    
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%ssample_gassf_particle_count_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%ssample_gassf_particle_count_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
 
#-------------------------
# Plot sample histogram of misalignments extracted              
def _plot_sample_vs_dist_hist(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                              #==============================================
                              # Graph settings
                              compare_to_dist          = True,
                                compare_stelmass       = True,
                                compare_gassf          = True,
                              #----------------------
                              # General formatting
                              use_PDF                  = False,        # uses probability density function
                                set_bin_width_mass     = 0.25,
                              #==============================================
                              showfig       = True,
                              savefig       = False,    
                                file_format = 'pdf',
                                savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                              #-----------------------------
                              debug = False):
    
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax']
    #-------------------------
    

    from misalignment_distributions import _plot_misalignment
    
    # Extract sample masses at z=0.1
    masses_dict = _plot_misalignment(csv_sample = 'L100_188_all_sample_misalignment_9.5', csv_output = '_Rad_Err__stars_gas_stars_gas_sf_gas_sf_gas_nsf_stars_dm_gas_dm_gas_sf_dm_', use_angle = 'stars_gas_sf', ETG_or_LTG = 'both', cluster_or_field   = 'both', use_proj_angle = False, add_observational  = False, showfig = False, savefig = False, output_masses_in_sample = True, 
                                     print_summary = False)
    # masses_dict['stelmass': [...], 'gassf': [...]] in absolute quantities, not log
    
    
    #==========================================================================
    # Gather data, take stelmass at first misalignment
    stelmass_plot = []
    gassf_plot    = []
    for ID_i in misalignment_tree.keys():
        stelmass_plot.append(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_m']])
        gassf_plot.append(misalignment_tree['%s' %ID_i]['sfmass'][misalignment_tree['%s' %ID_i]['index_m']])
        
    print('\tLenth of relaxation sample:                  ', len(gassf_plot))
    
    print('----------------\nProperties in 2 HMR:')
    print('Median stellar mass z=0.1 sample:               \t%.2e.' %(np.median(masses_dict['stelmass'])))
    print('Median stellar mass relaxation sample:          \t%.2e.' %(np.median(stelmass_plot)))
    print('Median gassf mass z=0.1 sample:                 \t%.2e.' %(np.median(masses_dict['gassf'])))
    print('Median gassf mass relaxation sample:            \t%.2e.' %(np.median(gassf_plot)))
    print('Median gassf fraction (2r_50) z=0 sample:       \t%.4f' %(np.median(np.divide(masses_dict['gassf'], np.array(masses_dict['gassf']) + np.array(masses_dict['stelmass'])))))
    print('Median gassf fraction (2r_50) relaxation sample: \t%.4f' %(np.median(np.divide(gassf_plot, np.array(stelmass_plot) + np.array(gassf_plot)))))
    
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.2], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Add z=0.1 sample
    if compare_stelmass:
        axs.hist(np.log10(masses_dict['stelmass']), bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), density=use_PDF, histtype='bar', edgecolor='none', facecolor='darkorange', alpha=0.3, label='$M_*$, $z\sim0.1$ sample', zorder=2)
        axs.hist(np.log10(stelmass_plot), bins=np.arange(9.0, 12+set_bin_width_mass, set_bin_width_mass), density=use_PDF, histtype='step', edgecolor='r', facecolor='none', alpha=1.0, label='$M_*$, relaxation sample', zorder=5)

        
    if compare_gassf:
        axs.hist(np.log10(masses_dict['gassf']), bins=np.arange(7.0, 12+set_bin_width_mass, set_bin_width_mass), density=use_PDF, histtype='bar', edgecolor='none', facecolor='dodgerblue', alpha=0.3, label='$M_{\mathrm{SF}}$, $z\sim0.1$ sample', zorder=2)
        axs.hist(np.log10(gassf_plot), bins=np.arange(7.0, 12+set_bin_width_mass, set_bin_width_mass), density=use_PDF, histtype='step', edgecolor='b', facecolor='none', alpha=1.0, label='$M_{\mathrm{SF}}$, relaxation sample', zorder=5)
        
        

    #-------------
    ### Formatting
    axs.set_xlabel(r'log$_{10}$ $M$ ($2r_{50}$) [M$_{\odot}$]')
    axs.set_xticks(np.arange(7.5, 12.5, 0.5))
    axs.set_xlim(7.5, 12.1)
    if use_PDF:
        axs.set_ylabel('PDF')
        axs.set_yscale('log')
        axs.set_ylim(0.0005, 100)
    else:
        axs.set_ylabel('Number of galaxies')
        axs.set_yscale('log')
        axs.set_ylim(0.8, 30000)
        axs.set_yticks([1, 10, 100, 1000, 10000])
        axs.set_yticklabels(['1', '10', '100', '1000', '10000'])
        
    
    
    #------------
    ### Legend
    ncol=1
    axs.legend(loc='upper right', frameon=False, labelspacing=0.1, labelcolor=['darkorange', 'r', 'dodgerblue', 'b'], ncol=ncol)
    
    
    #-----------
    ### other
    plt.tight_layout()

    #-----------
    # savefig
    metadata_plot = {'Author': 'MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                 
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        if use_PDF:
            savefig_txt = savefig_txt + 'PDF'
    
        plt.savefig("%s/relax_samples/%ssample_vs_dist_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/relax_samples/%ssample_vs_dist_hist_%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
    

#-------------------------
# Plot sample histogram of misalignments extracted       
def _plot_timescale_histogram(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.2,     # [ 0.25 / Gyr ]
                      set_thist_ymax_trelax               = 0.35,             # 0.45 / 500  yaxis max
                      set_min_trelax                      = 0,
                      #-----------------------------
                      # Plot options
                      set_plot_percentage               = True,
                      set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_inset                       = True,     # whether to have smaller second plot
                        add_inset_bestfit               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    #-------------------------
                      
                      
    #========================================================         
    # Gather data
    relaxationtime_plot = []
    co_co_array           = []
    co_counter_array      = []
    counter_co_array      = []
    counter_counter_array = []
    collect_array         = []
    collect_array_2         = []
    collect_array_3         = []
    for ID_i in misalignment_tree.keys():

        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_min_trelax:
            continue
        else:
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
    
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                if misalignment_tree['%s' %ID_i]['angle_peak'] > 135:
                    collect_array.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_time'] > 2:
                collect_array_2.append(ID_i)
        
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                collect_array_3.append(ID_i)
            
        
    
    #print('-------------------------------------------------------------')
    #print('Number of counter-co misalignments:  ', len(collect_array_3))
    #print(collect_array_3)
    #print('Number of >135 co-co misalignments: ', len(collect_array))
    #print(collect_array)
    #print('\nNumber of >2 Gyr misalignments: ', len(collect_array_2))
    #print(collect_array_2)
    print('  Using sample: ', len(relaxationtime_plot))
    print('\nMax trelax:  %.2f Gyr' %max(relaxationtime_plot))  
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if set_plot_relaxation_type:
        if set_plot_percentage:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(co_co_array))/len(relaxationtime_plot), np.ones(len(counter_counter_array))/len(relaxationtime_plot), np.ones(len(co_counter_array))/len(relaxationtime_plot), np.ones(len(counter_co_array))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(co_co_array))/len(relaxationtime_plot), np.ones(len(counter_counter_array))/len(relaxationtime_plot), np.ones(len(co_counter_array))/len(relaxationtime_plot), np.ones(len(counter_co_array))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        else:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
    else:
        if set_plot_percentage:
            axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        else:
            axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax+0.5*set_bin_width_trelax, set_bin_width_trelax), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
    
    #-------------
    ### Inset second axes
    if add_inset:
        axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
        
        if set_plot_percentage:
            axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        else: 
            axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
            
            
        #----------
        # Formatting
        axins.set_yscale('log')
        axins.set_xlim(0, set_bin_limit_trelax)
        axins.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axins.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]', fontsize = 5)
        if set_plot_percentage:
            axins.set_ylim(0.0002, set_thist_ymax_trelax)
            axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
            axins.set_ylabel('Percentage of\nmisalignments', fontsize=5)
        else:
            axins.set_ylim(0.5, set_thist_ymax_trelax)
            axins.set_ylabel('Number of misalignments', fontsize=5)
            
            
        #----------
        # Number of values in bin
        hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        print('bin_count of hist:', bin_count)
        
        #-----------
        # Add uncertainty
        #print('bin_count of hist:', bin_count)
        #print('hist_n of hist:', hist_n)
        #print('poisson uncertainty:', np.sqrt(hist_n))
        
        
        # Add poisson errors to each bin (sqrt N)
        axins.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor='k', ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.5)
        
        # Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        xdata = np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax)[mask]
        ydata = (hist_n/np.sum(hist_n))[mask]
        yerr  = (np.sqrt(hist_n)/np.sum(hist_n))[mask]
        
        ydata_log = np.log10(ydata)
        yerr_log  = np.log10(ydata + yerr) - np.log10(ydata)
        
        
        # Define linear bestfit
        def cal_func(x,c,m):
            return m*x + c
            
        popt, pcov = scipy.optimize.curve_fit(cal_func, xdata[1:], ydata_log[1:], sigma=yerr_log[1:], absolute_sigma=True)
        intercept = popt[0]
        slope = popt[1]
            
        """# Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
        """
        
        print('Best fit line:     frac = %.2f x 10^(%.2f t)' %((10**intercept), slope))
        print('             log10 frac = %.2f t + %.2f ' %(slope, intercept))
            
        if add_inset_bestfit:
            axins.plot([xdata[1], 8], [(10**intercept) * (10**(slope*xdata[1])), (10**intercept) * (10**(slope*8))], lw=0.7, ls='--', alpha=1, c='purple', label='best-fit')
            #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
    
    
    #-----------
    ### General formatting
    # Axis labels
    if set_plot_histogram_log:
        axs.set_yscale('log')
    if set_plot_percentage:
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    if set_plot_percentage:
        axs.set_ylabel('Percentage of misalignments')
    else:
        axs.set_ylabel('Number of misalignments')
    axs.set_ylim(0, set_thist_ymax_trelax)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    # add z
    #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('k')
    
    if set_plot_relaxation_type:
        if 'co-co' in relaxation_type:
            legend_labels.append('     co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
    if add_inset:
        ncol=2
    else:
        ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
    metadata_plot = {'Author': 'MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr\nmax: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale, max(relaxationtime_plot)),
                     'Producer': str(hist_n)}
                     
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_inset' if add_inset else '') + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%stime_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%stime_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn histogram    
def _plot_tdyn_histogram(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_tdyn                  = 60,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_thist_ymax_tdyn                 = 0.45,             # 0.35 / 400 yaxis max
                      set_min_trelax                      = 0,
                      #-----------------------------
                      # Plot options
                      set_plot_percentage               = True,
                      set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_inset                       = True,     # whether to have smaller second plot
                        add_inset_bestfit               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax']    
    use_angle          = misalignment_input['use_angle']
    #-------------------------       
            
    # Gather data
    relaxationtime_plot = []
    co_co_array           = []
    co_counter_array      = []
    counter_co_array      = []
    counter_counter_array = []
    ID_collect_1          = []
    ID_collect_2          = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_min_trelax:
            continue
        else:
            # append average tdyn over misalignment
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
    
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 10:
                ID_collect_1.append(ID_i)
            if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 20:   
                ID_collect_2.append(ID_i)
            
            
        #if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 20:
        #    print('\tFound >20 tdyn:   ID: %s\tMass: %.2e Msun' %(ID_i, np.mean(misalignment_tree['%s' %ID_i]['stelmass'])))
        #    print('\t  type: %s, morph: %s, time: %.2f, tdyn: %.2f, ttorque: %.2f' %(misalignment_tree['%s' %ID_i]['relaxation_type'], misalignment_tree['%s' %ID_i]['relaxation_morph'], misalignment_tree['%s' %ID_i]['relaxation_time'], misalignment_tree['%s' %ID_i]['relaxation_tdyn'], misalignment_tree['%s' %ID_i]['relaxation_ttorque']))
    
    #print('Number of >10 tdyn misalignments:   ', len(ID_collect_1))
    #print(ID_collect_1)
    #print('\nNumber of >20 tdyn misalignments:   ', len(ID_collect_2))
    #print(ID_collect_2)
    print('\n  Using sample: ', len(relaxationtime_plot))    
    print('\nMax tdyn/trelax:  %.2f' %max(relaxationtime_plot)) 
            
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if set_plot_relaxation_type:
        if set_plot_percentage:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(co_co_array))/len(relaxationtime_plot), np.ones(len(counter_counter_array))/len(relaxationtime_plot), np.ones(len(co_counter_array))/len(relaxationtime_plot), np.ones(len(counter_co_array))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(co_co_array))/len(relaxationtime_plot), np.ones(len(counter_counter_array))/len(relaxationtime_plot), np.ones(len(co_counter_array))/len(relaxationtime_plot), np.ones(len(counter_co_array))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        else:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
    else:
        if set_plot_percentage:
            axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn+0.5*set_bin_width_tdyn, set_bin_width_tdyn), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        else:
            axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn+0.5*set_bin_width_tdyn, set_bin_width_tdyn), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
    
    #-------------
    ### Inset second axes
    if add_inset:
        axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
        
        if set_plot_percentage:
            axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        else:
            axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        
                
        #-----------
        # Formatting
        axins.set_yscale('log')
        axins.set_xlim(0, set_bin_limit_tdyn)
        axins.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=8))
        axins.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$', fontsize=5)
        
        if set_plot_percentage:
            axins.set_ylim(0.0002, set_thist_ymax_tdyn)
            axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
            axins.set_ylabel('Percentage of\nmisalignments', fontsize=5)
        else:
            axins.set_ylim(0.5, set_thist_ymax_tdyn)
            axins.set_ylabel('Number of misalignments', fontsize=5)
        
        #----------
        # Number of values in bin
        hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        print('bin_count of hist:', bin_count)
        
        
        #-----------
        # Add uncertainty
        #print('bin_count of hist:', bin_count)
        #print('hist_n of hist:', hist_n)
        #print('poisson uncertainty:', np.sqrt(hist_n))
        
        # Add poisson errors to each bin (sqrt N)
        axins.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor='k', ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.5)
        
        
        #-----------
        # Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        xdata = np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn)[mask]
        ydata = (hist_n/np.sum(hist_n))[mask]
        yerr  = (np.sqrt(hist_n)/np.sum(hist_n))[mask]
        
        ydata_log = np.log10(ydata)
        yerr_log  = np.log10(ydata + yerr) - np.log10(ydata)
        
        
        # Define linear bestfit
        def cal_func(x,c,m):
            return m*x + c
            
        popt, pcov = scipy.optimize.curve_fit(cal_func, xdata[1:], ydata_log[1:], sigma=yerr_log[1:], absolute_sigma=True)
        intercept = popt[0]
        slope = popt[1]
            
        """# Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
        """
        
        print('Best fit line:     frac = %.2f x 10^(%.2f t)' %((10**intercept), slope))
        print('             log10 frac = %.2f t + %.2f ' %(slope, intercept))
            
        if add_inset_bestfit:
            axins.plot([xdata[1], 100], [(10**intercept) * (10**(slope*xdata[1])), (10**intercept) * (10**(slope*100))], lw=0.7, ls='--', alpha=1, c='purple', label='best-fit')
            #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor')
        
        
        
    #-----------
    ### General formatting
    # Axis labels
    if set_plot_histogram_log:
        axs.set_yscale('log')
    if set_plot_percentage:
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    if set_plot_percentage:
        axs.set_ylabel('Percentage of misalignments')
    else:
        axs.set_ylabel('Number of misalignments')
    axs.set_ylim(0, set_thist_ymax_tdyn)

    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    # add z
    #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('k')
            
    if set_plot_relaxation_type:
        if 'co-co' in relaxation_type:
            legend_labels.append('     co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
    if add_inset:
        ncol=2
    else:
        ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
        
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f\nmax: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn, max(relaxationtime_plot)),
                     'Producer': str(hist_n)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_inset' if add_inset else '') + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%stdyn_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%stdyn_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque histogram    
def _plot_ttorque_histogram(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      set_bin_limit_ttorque               = 30,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1.0,      # [ multiples ]
                      set_thist_ymax_ttorque              = 0.4,             # 0.35 / 400 yaxis max
                      set_min_trelax                      = 0,
                      #-----------------------------
                      # Plot options
                      set_plot_percentage               = True,
                      set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_inset                       = True,     # whether to have smaller second plot
                        add_inset_bestfit               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    #-------------------------   


    #==================================================================
    # Gather data
    relaxationtime_plot = []
    co_co_array           = []
    co_counter_array      = []
    counter_co_array      = []
    counter_counter_array = []
    ID_collect_1          = []
    ID_collect_2          = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_min_trelax:
            continue
        else:
            # append average ttorque over misalignment
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
    
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                co_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                co_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                counter_co_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                counter_counter_array.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 10:
                ID_collect_1.append(ID_i)
            
            if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 23:
                ID_collect_2.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        
        
        
    #print('Number of >10 ttorque misalignments:   ', len(ID_collect_1))
    #print(ID_collect_1)
    print(ID_collect_2)
    print('\n  Using sample: ', len(relaxationtime_plot))
    print('\nMax tdyn/ttorque:  %.2f' %max(relaxationtime_plot)) 
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if set_plot_relaxation_type:
        if set_plot_percentage:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(co_co_array))/len(relaxationtime_plot), np.ones(len(counter_counter_array))/len(relaxationtime_plot), np.ones(len(co_counter_array))/len(relaxationtime_plot), np.ones(len(counter_co_array))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], weights=(np.ones(len(co_co_array))/len(relaxationtime_plot), np.ones(len(counter_counter_array))/len(relaxationtime_plot), np.ones(len(co_counter_array))/len(relaxationtime_plot), np.ones(len(counter_co_array))/len(relaxationtime_plot)), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        else:
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', alpha=0.5, stacked=True)
            axs.hist([co_co_array, counter_counter_array, co_counter_array, counter_co_array], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, stacked=True)
        
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
    else:
        if set_plot_percentage:
            axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque+0.5*set_bin_width_ttorque, set_bin_width_ttorque), hist_n/len(relaxationtime_plot), xerr=None, yerr=np.sqrt(hist_n)/len(relaxationtime_plot), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        else:
            axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axs.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', alpha=1.0)
    
            # Add poisson errors to each bin (sqrt N)
            hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque+0.5*set_bin_width_ttorque, set_bin_width_ttorque), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor='k', ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
    
    
    #-------------
    ### Inset second axes
    if add_inset:
        axins = axs.inset_axes([0.45, 0.2, 0.5, 0.6])
        
        if set_plot_percentage:
            axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, weights=np.ones(len(relaxationtime_plot))/len(relaxationtime_plot), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        else:
            axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='none', facecolor='k', alpha=0.1)
            bin_count, _, _ = axins.hist(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='bar', edgecolor='k', facecolor='none', lw=0.7, alpha=0.7)
        
        
        #-----------
        # Formatting
        axins.set_yscale('log')
        axins.set_xlim(0, set_bin_limit_ttorque)
        axins.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=4))
        axins.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$', fontsize=5)
        if set_plot_percentage:
            axins.set_ylim(0.0002, set_thist_ymax_ttorque)
            axins.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
            axins.set_ylabel('Percentage of\nmisalignments', fontsize=5)
        else:
            axins.set_ylim(0.5, set_thist_ymax_ttorque)
            axins.set_ylabel('Percentage of\nmisalignments', fontsize=5)
        
        
        #----------
        # Number of values in bin
        hist_n, _ = np.histogram(relaxationtime_plot, bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        print('bin_count of hist:', bin_count)
            
        #-----------
        # Add uncertainty
        #print('bin_count of hist:', bin_count)
        #print('hist_n of hist:', hist_n)
        #print('poisson uncertainty:', np.sqrt(hist_n))
    
        # Add poisson errors to each bin (sqrt N)
        axins.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor='k', ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.5)
    
    
        #-----------
        # Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        xdata = np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque)[mask]
        ydata = (hist_n/np.sum(hist_n))[mask]
        yerr  = (np.sqrt(hist_n)/np.sum(hist_n))[mask]
    
        ydata_log = np.log10(ydata)
        yerr_log  = np.log10(ydata + yerr) - np.log10(ydata)
    
    
        # Define linear bestfit
        def cal_func(x,c,m):
            return m*x + c
        
        popt, pcov = scipy.optimize.curve_fit(cal_func, xdata[1:], ydata_log[1:], sigma=yerr_log[1:], absolute_sigma=True)
        intercept = popt[0]
        slope = popt[1]
        
        """# Best-fit data
        mask = np.where(np.array(bin_count) > 0)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque)[mask][1:], np.log10(np.array(bin_count)[mask])[1:])
        """
    
        print('Best fit line:     frac = %.2f x 10^(%.2f t)' %((10**intercept), slope))
        print('             log10 frac = %.2f t + %.2f ' %(slope, intercept))
        
        if add_inset_bestfit:
            axins.plot([xdata[1], 100], [(10**intercept) * (10**(slope*xdata[1])), (10**intercept) * (10**(slope*100))], lw=0.7, ls='--', alpha=1, c='purple', label='best-fit')
            #axins.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0, labelcolor='linecolor') 
            
        
    #-----------
    ### General formatting
    # Axis labels
    if set_plot_histogram_log:
        axs.set_yscale('log')
    if set_plot_percentage:
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    if set_plot_percentage:
        axs.set_ylabel('Percentage of misalignments')
    else:
        axs.set_ylabel('Number of misalignments')
    axs.set_ylim(0, set_thist_ymax_ttorque)

    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    # add z
    #legend_labels.append('${%.1f<z<%.1f}$' %((0 if min_z == None else min_z), (1.0 if max_z == None else max_z)))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('k')
            
    if set_plot_relaxation_type:
        if 'co-co' in relaxation_type:
            legend_labels.append('     co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
    if add_inset:
        ncol=2
    else:
        ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
        
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f\nmax: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque, max(relaxationtime_plot)),
                     'Producer': str(hist_n)}
                     
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_inset' if add_inset else '') + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/time_spent_misaligned/%sttorque_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned/%sttorque_spent_misaligned_%s_stacked%s_percentage%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), set_plot_relaxation_type, set_plot_percentage, savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Stacked single 1x1 graphs
def _plot_stacked_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.25,     # [ 0.25 / Gyr ]
                      set_thist_ymax_trelax               = 0.35,             # 0.45 / 500  yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    time_collect = []
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
            
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        if set_plot_type == 'time':
            timeaxis_plot = -1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1])
        elif set_plot_type == 'raw_time':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - (0)
        elif set_plot_type == 'snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - misalignment_tree['%s' %ID_i]['SnapNum'][misalignment_tree['%s' %ID_i]['index_s']+1]
        elif set_plot_type == 'raw_snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - (0)
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_trelax:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1]+0.120)
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_time'] > 2:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.02)
            c     = lighten_color(line_color, (misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time'])**(-0.5)) 
            
            
            
        #axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
        axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
        
        ### Annotate
        if set_add_GalaxyIDs:
            axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    #print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    #print('  List of >2 Gyr trelax ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 5.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 5.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 5.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 5.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                 
    #-----------
    # Add threshold
    axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #ßfor ID_i in ID_plot:
    #    print(' ')
    #    print(' ID: ', ID_i)
    #    for time_i, angle_i, merger_i in zip(misalignment_tree['%s' %ID_i]['Lookbacktime'], misalignment_tree['%s' %ID_i][use_angle], misalignment_tree['%s' %ID_i]['merger_ratio_stars']):
    #        print('%.2f\t%.1f\t' %(time_i, angle_i), merger_i)
    print(' remaining sample aaaaaa: ', len(ID_plot))
    
    
    
    #-----------
    ### Formatting
    axs.set_ylim(0, 180)
    axs.set_yticks(np.arange(0, 181, 30))
    axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    if set_plot_type == 'time':
        axs.set_xlim(0-3*time_extra, set_bin_limit_trelax)
        axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, 1))
        axs.set_xlabel('Relaxation time [Gyr]')
    elif set_plot_type == 'raw_time':
        axs.set_xlim(8, 0)
        axs.set_xticks(np.arange(8, -0.1, -1))
        axs.set_xlabel('Lookbacktime [Gyr]')
    elif set_plot_type == 'snap':
        axs.set_xlim(-10, 70)
        axs.set_xticks(np.arange(-10, 71, 10))
        axs.set_xlabel('Snapshots since misalignment')
    elif set_plot_type == 'raw_snap':
        axs.set_xlim(140, 200)
        axs.set_xticks(np.arange(140, 201, 5))
        axs.set_xlabel('Snapshots')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    if (set_plot_type == 'time') or (set_plot_type == 'snap'):
        axs.axvline(0, ls='-', lw=1, c='grey')
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        legend_elements = []
        legend_labels = []
        legend_colors = []
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels.append('co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
            
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
        legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        axs.add_artist(legend2)
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['trelax']['co-co']))
    median_co_co = np.median(np.array(summary_dict['trelax']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['trelax']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['trelax']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['trelax']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['trelax']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['trelax']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['trelax']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['trelax']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['trelax']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['trelax']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['trelax']['counter-co']))
    print('Relaxation timescales:')
    print('   [Gyr]     all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/trelax_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/trelax_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_stacked_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_tdyn                  = 60,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_thist_ymax_tdyn                 = 0.45,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================        
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    time_collect = []
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
                    
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_tdyn:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 20:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.02)
            c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_tdyn']/6))**(-0.5))
        
            
            
        axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
        
        ### Annotate
        if set_add_GalaxyIDs:
            axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    print('  Using sample: ', len(ID_plot))
    print('  List of >20 trelax/tdyn ', len(ID_collect))
    print(ID_collect)
    print('asugaigfaiuf')
    print('median: ', np.median(time_collect))
    
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 32.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                        
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 32.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 32.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 32.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                 
    #-----------
    # Add threshold
    axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    axs.set_ylim(0, 180)
    axs.set_yticks(np.arange(0, 181, 30))
    axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    axs.set_xlim(-1, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, 4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    
    #-----------
    ### Annotations
    if (set_plot_type == 'time') or (set_plot_type == 'snap'):
        axs.axvline(0, ls='-', lw=1, c='k')
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        legend_elements = []
        legend_labels = []
        legend_colors = []
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels.append('co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
            
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
        legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        axs.add_artist(legend2)
    
    
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['tdyn']['co-co']))
    median_co_co = np.median(np.array(summary_dict['tdyn']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['tdyn']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['tdyn']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['tdyn']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['tdyn']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['tdyn']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['tdyn']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['tdyn']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['tdyn']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['tdyn']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['tdyn']['counter-co']))
    print('trelax/tdyn multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/tdyn_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/tdyn_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_stacked_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_ttorque               = 30,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_thist_ymax_ttorque              = 0.4,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #-------------------------        
        
    
    #===============================================================
    # Graph initialising and base formatting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    time_collect = []
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        time_collect.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_ttorque:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 10:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.02)
            c     = lighten_color(line_color, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, ((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*((misalignment_tree['%s' %ID_i]['relaxation_ttorque']/2))**(-0.5))
        
            
            
        axs.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)
        
        ### Annotate
        if set_add_GalaxyIDs:
            axs.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    #print('-------------------------------------------------------------')
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    print('  Using sample: ', len(ID_plot))
    print('  List of >10 trelax/ttorque ', len(ID_collect))
    print(ID_collect)
    print('asugaigfaiuf')
    print('median: ', np.median(time_collect))
    
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_co))+0.01, 12.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
                        
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-1*np.median(diff_co_counter))+0.01, 12.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_counter))+0.01, 12.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-1*np.median(diff_counter_co))+0.01, 12.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            
            #----------
            # plot
            axs.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            axs.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            axs.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
                 
    #-----------
    # Add threshold
    axs.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
    axs.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    axs.set_ylim(0, 180)
    axs.set_yticks(np.arange(0, 181, 30))
    axs.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    axs.set_xlim(-0.5, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, 1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        
    #-----------
    ### Annotations
    if (set_plot_type == 'time') or (set_plot_type == 'snap'):
        axs.axvline(0, ls='-', lw=1, c='k')
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        legend_elements = []
        legend_labels = []
        legend_colors = []
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels.append('co → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C0')
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels.append('counter → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C1')
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels.append('     co → counter')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C2')
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels.append('counter → co')
            legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
            legend_colors.append('C3')
            
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
        legend2 = axs.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        axs.add_artist(legend2)
    
    
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['ttorque']['co-co']))
    median_co_co = np.median(np.array(summary_dict['ttorque']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['ttorque']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['ttorque']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['ttorque']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['ttorque']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['ttorque']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['ttorque']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['ttorque']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['ttorque']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['ttorque']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['ttorque']['counter-co']))
    print('trelax/ttorque multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/ttorque_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/ttorque_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()

    
#-------------------------
# Stacked 2x2 graphs
def _plot_stacked_trelax_2x2(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.25,     # [ 0.25 / Gyr ]
                      set_thist_ymax_trelax               = 0.35,             # 0.45 / 500  yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================
    # Graph initialising and base formatting
    fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    diff_co_co           = []   # average time spacing
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        if set_plot_type == 'time':
            timeaxis_plot = -1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1])
        elif set_plot_type == 'raw_time':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - (0)
        elif set_plot_type == 'snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - misalignment_tree['%s' %ID_i]['SnapNum'][misalignment_tree['%s' %ID_i]['index_s']+1]
        elif set_plot_type == 'raw_snap':
            timeaxis_plot = np.array(misalignment_tree['%s' %ID_i]['SnapNum']) - (0)
        
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                ax = ax_co_co
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                ax = ax_co_counter
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                ax = ax_counter_co
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                ax = ax_counter_counter
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_trelax:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1]+0.120)
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_time'] > 2:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.02)
            c     = lighten_color(line_color, (misalignment_tree['%s' %ID_i]['relaxation_time']+0.01)**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time']+0.01)**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time']+0.01)**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.6, (misalignment_tree['%s' %ID_i]['relaxation_time']**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*(misalignment_tree['%s' %ID_i]['relaxation_time']+0.01)**(-0.5)) 
            
            
        ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
        ### Annotate
        if set_add_GalaxyIDs:
            ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of >2 Gyr trelax ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-2*np.median(diff_co_co))+0.01, 5.09, np.median(diff_co_co))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess' below 10 degrees
            #mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            mask_median = np.where(np.array(median_array) > 15)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > 15)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > 15)[0][-1] + 1
            
            #----------
            # plot
            ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-2*np.median(diff_co_counter))+0.01, 5.09, np.median(diff_co_counter))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            mask_median = np.where(np.array(median_array) < (180-15))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-15))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-15))[0][-1] + 1
            
            #----------
            # plot
            ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-2*np.median(diff_counter_counter))+0.01, 5.09, np.median(diff_counter_counter))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            mask_median = np.where(np.array(median_array) < (180-15))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-15))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-15))[0][-1] + 1
            
            #----------
            # plot
            ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-2*np.median(diff_counter_co))+0.01, 5.09, np.median(diff_counter_co))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            mask_median = np.where(np.array(median_array) > 15)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > 15)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > 15)[0][-1] + 1
            
            #----------
            # plot
            ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(bins+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
    
             
    #-----------
    # Add threshold
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    ax_co_co.set_ylim(0, 180)
    ax_co_co.set_yticks(np.arange(0, 181, 30))
    ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
    if set_plot_type == 'time':
        ax_counter_counter.set_xlim(0-4*time_extra, set_bin_limit_trelax)
        ax_counter_counter.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, 1))
        ax_counter_co.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
        ax_counter_counter.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
        #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    elif set_plot_type == 'raw_time':
        ax_counter_counter.set_xlim(8, 0)
        ax_counter_counter.set_xticks(np.arange(8, -0.1, -1))
        ax_counter_counter.set_xlabel('Lookbacktime [Gyr]')
        ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    elif set_plot_type == 'snap':
        ax_counter_counter.set_xlim(-10, 70)
        ax_counter_counter.set_xticks(np.arange(-10, 71, 10))
        ax_counter_counter.set_xlabel('Snapshots since misalignment')
        ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    elif set_plot_type == 'raw_snap':
        ax_counter_counter.set_xlim(140, 200)
        ax_counter_counter.set_xticks(np.arange(140, 201, 5))
        ax_counter_counter.set_xlabel('Snapshots')
        ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
            
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels = ['co → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C0']
            legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_co.add_artist(legend2)
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels = ['counter → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C1']
            legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_counter.add_artist(legend2)
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels = ['     co → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C2']
            legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_counter.add_artist(legend2)
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels = ['counter → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C3']
            legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_co.add_artist(legend2)
            
    #-----------
    ### title
    if plot_annotate:
        ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['trelax']['co-co']))
    median_co_co = np.median(np.array(summary_dict['trelax']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['trelax']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['trelax']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['trelax']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['trelax']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['trelax']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['trelax']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['trelax']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['trelax']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['trelax']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['trelax']['counter-co']))
    print('Relaxation timescales:')
    print('   [Gyr]     all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_timescale, (math.nan if len(summary_dict['trelax']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['trelax']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['trelax']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['trelax']['counter-co']) == 0 else std_counter_co)))
    
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/trelax_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/trelax_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_stacked_tdyn_2x2(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_tdyn                  = 60,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_thist_ymax_tdyn                 = 0.45,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 

    
    #===========================================================================
    # Graph initialising and base formatting
    fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['tdyn'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                ax = ax_co_co
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)     
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                ax = ax_co_counter
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                ax = ax_counter_co
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                ax = ax_counter_counter
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_tdyn:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] > 10:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6)**1.5)/10), 0.02)
            c     = lighten_color(line_color, (((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*(((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*(((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*(((misalignment_tree['%s' %ID_i]['relaxation_tdyn']+0.01)/6))**(-0.5))
            
            
            
            
        ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
        ### Annotate
        if set_add_GalaxyIDs:
            ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    #print('  List of >10 trelax/tdyn ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-2*np.median(diff_co_co))+0.01, 32.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            mask_median = np.where(np.array(median_array) > 10)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > 10)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > 10)[0][-1] + 1
            
            #----------
            # plot
            ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-2*np.median(diff_co_counter))+0.01, 32.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            mask_median = np.where(np.array(median_array) < (180-10))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-10))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-10))[0][-1] + 1
            
            #----------
            # plot
            ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-2*np.median(diff_counter_counter))+0.01, 32.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            mask_median = np.where(np.array(median_array) < (180-10))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-10))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-10))[0][-1] + 1
            
            #----------
            # plot
            ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-2*np.median(diff_counter_co))+0.01, 32.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            mask_median = np.where(np.array(median_array) > 10)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > 10)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > 10)[0][-1] + 1
            
            #----------
            # plot
            ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            
    #-----------
    # Add threshold
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    ax_co_co.set_ylim(0, 180)
    ax_co_co.set_yticks(np.arange(0, 181, 30))
    ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
    ax_counter_counter.set_xlim(-4, set_bin_limit_tdyn)
    ax_counter_counter.set_xticks(np.arange(-4, set_bin_limit_tdyn+0.1, 4))
    ax_counter_counter.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    ax_counter_co.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
    
    #-----------
    ### title
    if plot_annotate:
        ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
            
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels = ['co → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C0']
            legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_co.add_artist(legend2)
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels = ['counter → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C1']
            legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_counter.add_artist(legend2)
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels = ['     co → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C2']
            legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_counter.add_artist(legend2)
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels = ['counter → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C3']
            legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_co.add_artist(legend2)
            
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['tdyn']['co-co']))
    median_co_co = np.median(np.array(summary_dict['tdyn']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['tdyn']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['tdyn']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['tdyn']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['tdyn']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['tdyn']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['tdyn']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['tdyn']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['tdyn']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['tdyn']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['tdyn']['counter-co']))
    print('trelax/tdyn multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_tdyn, (math.nan if len(summary_dict['tdyn']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['tdyn']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['tdyn']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['tdyn']['counter-co']) == 0 else std_counter_co)))
    
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/tdyn_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/tdyn_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_stacked_ttorque_2x2(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # General formatting
                      set_bin_limit_ttorque               = 30,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_thist_ymax_ttorque              = 0.4,             # 0.35 / 400 yaxis max
                      #-----------------------------
                      # Plot options
                      set_plot_type                       = 'time',            # 'time', 'snap', 'raw_time', 'raw_snap'
                      set_stacked_plot_type               = 'misangle',        # 'misangle', 'merger' where to lineup stacks to
                      set_plot_extra_time                 = False,              # Plot extra time after relaxation
                      set_plot_merger_limit               = None,               # [ None / merger ratio ] None will not plot legend or squares
                      set_add_GalaxyIDs                   = False,             # Add GalaxyID of entry
                      set_plot_relaxation_type_stacked    = True,             # Stack histogram types
                        add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                        set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #-------------------------         


    #=================================================================================================
    # Graph initialising and base formatting
    fig, ((ax_co_co, ax_co_counter), (ax_counter_counter, ax_counter_co)) = plt.subplots(2, 2, figsize=[2*10/3, 2*1.8], sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    
    # Creating colormaps to mark mergers
    merger_colormap = plt.get_cmap('Blues', 5)
    merger_normalize = colors.Normalize(vmin=0, vmax=1)
    timescale_colormap = plt.get_cmap('inferno')
    timescale_normalize = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cm.ScalarMappable(norm=timescale_normalize, cmap=timescale_colormap)
    
    #-----------
    ### Loop over all windows and plot them
    ID_plot     = []
    ID_collect  = []        # flexible array used to extract weird relaxations
    diff_co_co           = []
    diff_co_counter      = []
    diff_counter_counter = []
    diff_counter_co      = []
    scatter_x = []
    scatter_y = []
    scatter_c = []
    scatter_s = []
    
    stats_lines = {'co-co': {'time': [],
                             'angles': []},
                   'co-counter': {'time': [],
                                  'angles': []},
                   'counter-counter': {'time': [],
                                       'angles': []},
                   'counter-co': {'time': [],
                                  'angles': []}}
    
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_stacked_relaxation_type:
            continue
        
        ID_plot.append(misalignment_tree['%s' %ID_i]['GalaxyID'][0])
        timeaxis_plot = (-1*np.array(np.array(misalignment_tree['%s' %ID_i]['Lookbacktime']) - misalignment_tree['%s' %ID_i]['Lookbacktime'][misalignment_tree['%s' %ID_i]['index_s']+1]))/np.mean(np.array(misalignment_tree['%s' %ID_i]['ttorque'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1])
        
        #-------------
        # Plot stacked
        line_color = 'k'
        alpha = 0.2
        if set_plot_relaxation_type_stacked:
            alpha = 0.1
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                line_color='C0'
                append_angle = 10
                ax = ax_co_co
                diff_co_co.append(timeaxis_plot[1]-timeaxis_plot[0])
                
                # collect IDs of weird relaxations
                #if max(misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]) > 135:
                #    ID_collect.append(ID_i)
                    
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                line_color='C2'
                append_angle = 170
                ax = ax_co_counter
                diff_co_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                line_color='C3'
                append_angle = 10
                ax = ax_counter_co
                diff_counter_co.append(timeaxis_plot[1]-timeaxis_plot[0])
            elif misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                line_color='C1'
                append_angle = 170
                ax = ax_counter_counter
                diff_counter_counter.append(timeaxis_plot[1]-timeaxis_plot[0])
        
        
        #-------------
        # append to stats and add extra values of relaxed state until end of set_bin_limit_trelax
        timeaxis_stats = timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]
        angles_stats   = misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]                   
        # Add missing 120Mya time slots until end reached
        while timeaxis_stats[-1] < set_bin_limit_ttorque:
            timeaxis_stats = np.append(timeaxis_stats, timeaxis_stats[-1] + (timeaxis_stats[1]-timeaxis_stats[0]))
            angles_stats   = np.append(angles_stats, append_angle)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['time'].extend(timeaxis_stats)
        stats_lines['%s' %misalignment_tree['%s' %ID_i]['relaxation_type']]['angles'].extend(angles_stats)
        
        
        #-------------
        # pick out long relaxersß
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] > 5:
            ID_collect.append(ID_i)
                        
        
        #-------------
        # format and plot line
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2)**1.5)/10), 0.02)
            c     = lighten_color(line_color, (((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2)**1.5)/10), 0.03)
            c     = lighten_color(line_color, 0.7*(((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2)**1.5)/2), 0.02)
            c     = lighten_color(line_color, 0.7*(((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2))**(-0.5)) 
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            alpha = max(min(0.5, (((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2)**1.5)/3), 0.02)
            c     = lighten_color(line_color, 0.7*(((misalignment_tree['%s' %ID_i]['relaxation_ttorque']+0.01)/2))**(-0.5))
            
            
            
            
        ax.plot(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], lw=0.3, c=c, alpha=alpha)       # c=scalarMap.to_rgba(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
        ### Annotate
        if set_add_GalaxyIDs:
            ax.text(timeaxis_plot[0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+0.1, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1]+5, '%s' %misalignment_tree['%s' %ID_i]['GalaxyID'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)][-1], fontsize=7)
        
        # Plot mergers (some may be missing if they are out of window)
        if set_plot_merger_limit != None:
            for time_i, angle_i, ratio_i, ratio_gas_i in zip(timeaxis_plot, misalignment_tree['%s' %ID_i][use_angle][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_stars'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)], misalignment_tree['%s' %ID_i]['merger_ratio_gas'][0:(len(misalignment_tree['%s' %ID_i]['SnapNum'])+1 if set_plot_extra_time == True else misalignment_tree['%s' %ID_i]['index_r']+1)]):
                if len(ratio_i) > 0:
                    if max(ratio_i) >= set_plot_merger_limit:
                        scatter_x.append(time_i)
                        scatter_y.append(angle_i)
                        scatter_c.append(ratio_gas_i[np.argmax(np.array(ratio_i))])
                        scatter_s.append(50*(ratio_i[np.argmax(np.array(ratio_i))])**0.5)
    
    print('-------------------------------------------------------------')
    print('  Using sample: ', len(ID_plot))
    #print('  List of IDs with co-co relax but max(angle) > 135:  ', len(ID_collect))
    #print('  List of >5 trelax/ttorque ', len(ID_collect))
    #print(ID_collect)
    
    #--------------
    # Find mean/median and 1 sigma behaviour for each relaxation type
    if add_stacked_median:
        if len(stats_lines['co-co']['time']) != 0:
            line_color='C0'
            line_color='k'
            bins = np.arange((-2*np.median(diff_co_co))+0.01, 12.09, np.median(diff_co_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-co']['time']), bins=bins)
            
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-co']['time'])[mask]
                current_angles = np.array(stats_lines['co-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
            
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            mask_median = np.where(np.array(median_array) > 10)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > 10)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > 10)[0][-1] + 1
            
            #----------
            # plot
            ax_co_co.plot(bins[0:mask_median+1]+np.median(diff_co_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_upper+1]+np.median(diff_co_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_co.plot(bins[0:mask_lower+1]+np.median(diff_co_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['co-counter']['time']) != 0:
            line_color='C2'
            line_color='k'
            bins = np.arange((-2*np.median(diff_co_counter))+0.01, 12.09, np.median(diff_co_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['co-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['co-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['co-counter']['time'])[mask]
                current_angles = np.array(stats_lines['co-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            mask_median = np.where(np.array(median_array) < (180-10))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-10))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-10))[0][-1] + 1
            
            #----------
            # plot
            ax_co_counter.plot(bins[0:mask_median+1]+np.median(diff_co_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_upper+1]+np.median(diff_co_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_co_counter.plot(bins[0:mask_lower+1]+np.median(diff_co_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-counter']['time']) != 0:
            line_color='C1'
            line_color='k'
            bins = np.arange((-2*np.median(diff_counter_counter))+0.01, 12.09, np.median(diff_counter_counter))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-counter']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-counter']['time'])[mask]
                current_angles = np.array(stats_lines['counter-counter']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) < (180-misangle_threshold))[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) < (180-misangle_threshold))[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) < (180-misangle_threshold))[0][-1] + 1
            mask_median = np.where(np.array(median_array) < (180-10))[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) < (180-10))[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) < (180-10))[0][-1] + 1
            
            #----------
            # plot
            ax_counter_counter.plot(bins[0:mask_median+1]+np.median(diff_counter_counter), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_upper+1]+np.median(diff_counter_counter), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_counter.plot(bins[0:mask_lower+1]+np.median(diff_counter_counter), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
        if len(stats_lines['counter-co']['time']) != 0:
            line_color='C3'
            line_color='k'
            bins = np.arange((-2*np.median(diff_counter_co))+0.01, 12.09, np.median(diff_counter_co))
        
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter-co']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter-co']['time']), bins=bins)
        
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            use_percentiles = 16        # 1 sigma
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter-co']['time'])[mask]
                current_angles = np.array(stats_lines['counter-co']['angles'])[mask]
                            
                median_array.append(np.percentile(current_angles, 50))
                median_upper.append(np.percentile(current_angles, 100-use_percentiles))
                median_lower.append(np.percentile(current_angles, use_percentiles))
        
            #----------
            # remove 'excess'
            #mask_median = np.where(np.array(median_array) > misangle_threshold)[0][-1] + 1
            #mask_upper  = np.where(np.array(median_upper) > misangle_threshold)[0][-1] + 1
            #mask_lower  = np.where(np.array(median_lower) > misangle_threshold)[0][-1] + 1
            mask_median = np.where(np.array(median_array) > 10)[0][-1] + 1
            mask_upper  = np.where(np.array(median_upper) > 10)[0][-1] + 1
            mask_lower  = np.where(np.array(median_lower) > 10)[0][-1] + 1
            
            #----------
            # plot
            ax_counter_co.plot(bins[0:mask_median+1]+np.median(diff_counter_co), np.array(median_array)[0:mask_median+1], color=line_color, ls='-', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_upper+1]+np.median(diff_counter_co), np.array(median_upper)[0:mask_upper+1], color=line_color, ls='--', lw=1, zorder=100)
            ax_counter_co.plot(bins[0:mask_lower+1]+np.median(diff_counter_co), np.array(median_lower)[0:mask_lower+1], color=line_color, ls='--', lw=1, zorder=100)
            #axs.fill_between(np.arange(-0.249, 5.09, 0.125)+(0.5*0.125), median_lower, median_upper, color=line_color, lw=0, alpha=0.15, zorder=5)
            
    #-----------
    # Add threshold
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.axhspan(0, misangle_threshold, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-misangle_threshold, 180, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(0, peak_misangle, alpha=0.15, ec=None, fc='grey')
        ax.axhspan(180-peak_misangle, 180, alpha=0.15, ec=None, fc='grey')
    
    
    #-----------
    ### Formatting
    ax_co_co.set_ylim(0, 180)
    ax_co_co.set_yticks(np.arange(0, 181, 30))
    ax_co_co.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    ax_counter_counter.set_ylabel('Misalignment angle, $\psi_{\mathrm{3D}}$')
    #ax_counter_counter.get_yaxis().set_label_coords(-0.12,1)
    ax_counter_counter.set_xlim(-2, set_bin_limit_ttorque)
    ax_counter_counter.set_xticks(np.arange(-2, set_bin_limit_ttorque+0.1, 2))
    ax_counter_counter.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    ax_counter_co.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    #ax_counter_counter.get_xaxis().set_label_coords(1,-0.12)
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### Annotations
    for ax in [ax_co_co, ax_co_counter, ax_counter_counter, ax_counter_co]:
        if (set_plot_type == 'time') or (set_plot_type == 'snap'):
            ax.axvline(0, ls='-', lw=1, c='grey', alpha=1)
    
    #-----------
    ### title
    if plot_annotate:
        ax_co_co.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
      
    #-----------
    ### Customise legend labels
    if set_plot_merger_limit != None:
        axs.scatter(scatter_x, scatter_y, c=scatter_c, cmap=merger_colormap, norm=merger_normalize, s=scatter_s, marker='s', edgecolors='grey', zorder=99)
        
        plt.scatter(-20, -160, c=0.1, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.1')
        plt.scatter(-20, -150, c=0.3, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=0.3')
        plt.scatter(-20, -140, c=1.0, s=50*(0.5**0.5), cmap=merger_colormap, norm=merger_normalize, marker='s', edgecolors='grey', label='$\mu_{\mathrm{gas}}$=1.0')
        plt.scatter(-20, -160, c='w', s=50*(0.1**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.1')
        plt.scatter(-20, -150, c='w', s=50*(0.3**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=0.3')
        plt.scatter(-20, -140, c='w', s=50*(1.0**0.5), marker='s', edgecolors='grey', label='$\mu_{\mathrm{*}}$=1.0')
        
        legend1 = axs.legend(loc='upper right', frameon=False, labelspacing=0.1, handlelength=0)
        axs.add_artist(legend1)
    if set_plot_relaxation_type_stacked:
        if set_plot_merger_limit != None:
            loc = [0.62, 0.35]
        else:
            loc = 'upper right'
            
        if 'co-co' in set_stacked_relaxation_type:
            legend_labels = ['co → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C0']
            legend2 = ax_co_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_co.add_artist(legend2)
        if 'counter-counter' in set_stacked_relaxation_type:
            legend_labels = ['counter → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C1']
            legend2 = ax_counter_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_counter.add_artist(legend2)
        if 'co-counter' in set_stacked_relaxation_type:
            legend_labels = ['     co → counter']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C2']
            legend2 = ax_co_counter.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_co_counter.add_artist(legend2)
        if 'counter-co' in set_stacked_relaxation_type:
            legend_labels = ['counter → co']
            legend_elements = [Line2D([0], [0], marker=' ', color='w')]
            legend_colors = ['C3']
            legend2 = ax_counter_co.legend(handles=legend_elements, labels=legend_labels, loc=loc, handletextpad=5, frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
            ax_counter_co.add_artist(legend2)
            
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    #### Savefig
    # Average timescales
    mean_co_co   = np.mean(np.array(summary_dict['ttorque']['co-co']))
    median_co_co = np.median(np.array(summary_dict['ttorque']['co-co']))
    std_co_co    = np.std(np.array(summary_dict['ttorque']['co-co']))
    mean_co_counter   = np.mean(np.array(summary_dict['ttorque']['co-counter']))
    median_co_counter = np.median(np.array(summary_dict['ttorque']['co-counter']))
    std_co_counter    = np.std(np.array(summary_dict['ttorque']['co-counter']))
    mean_counter_counter   = np.mean(np.array(summary_dict['ttorque']['counter-counter']))
    median_counter_counter = np.median(np.array(summary_dict['ttorque']['counter-counter']))
    std_counter_counter    = np.std(np.array(summary_dict['ttorque']['counter-counter']))
    mean_counter_co   = np.mean(np.array(summary_dict['ttorque']['counter-co']))
    median_counter_co = np.median(np.array(summary_dict['ttorque']['counter-co']))
    std_counter_co    = np.std(np.array(summary_dict['ttorque']['counter-co']))
    print('trelax/ttorque multiples:')
    print('             all   co-co   co-counter   counter-counter   counter-co')
    print('   Mean:    %.2f    %.2f      %.2f           %.2f            %.2f' %(mean_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else mean_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else mean_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else mean_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else mean_counter_co)))   
    print('   Median:  %.2f    %.2f      %.2f           %.2f            %.2f' %(median_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else median_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else median_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else median_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else median_counter_co)))   
    print('   std:     %.2f    %.2f      %.2f           %.2f            %.2f' %(std_ttorque, (math.nan if len(summary_dict['ttorque']['co-co']) == 0 else std_co_co), (math.nan if len(summary_dict['ttorque']['co-counter']) == 0 else std_co_counter), (math.nan if len(summary_dict['ttorque']['counter-counter']) == 0 else std_counter_counter), (math.nan if len(summary_dict['ttorque']['counter-co']) == 0 else std_counter_co)))
    
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/stacked_misalignments/ttorque_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/stacked_misalignments/ttorque_2x2_stacked_misalignments_%s_%s_%s.%s" %(fig_dir, set_plot_type, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plot box and whisker of relaxation distributions
def _plot_box_and_whisker_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_whisker_morphs                  = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    for ID_i in misalignment_tree.keys():
        
        # only plot morphs we care about (default ETG-ETG, LTG-LTG)
        if 'ETG' in set_whisker_morphs:
            if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                continue
        elif 'ETG-ETG' in set_whisker_morphs: 
            if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                continue
        
        # Gather data
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            relaxationtype_plot.append('co\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            relaxationtype_plot.append('co\n ↓ \ncounter')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            relaxationtype_plot.append('counter\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            relaxationtype_plot.append('counter\n ↓ \ncounter')
    
        
        if 'ETG' in set_whisker_morphs:
            relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
        elif 'ETG-ETG' in set_whisker_morphs: 
            #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
    
    #----------------------
    # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
    print('\n--------------------------------------')
    print('trelax')
    print('  Using sample: ', len(relaxationtype_plot))
    for relaxation_type_i in relaxation_type:
        
        if relaxation_type_i == 'co-co':
            relaxation_type_i = 'co\n ↓ \nco'
        if relaxation_type_i == 'co-counter':
            relaxation_type_i = 'co\n ↓ \ncounter'
        if relaxation_type_i == 'counter-co':
            relaxation_type_i = 'counter\n ↓ \nco'
        if relaxation_type_i == 'counter-counter':
            relaxation_type_i = 'counter\n ↓ \ncounter'
        
        # Select only relaxation morphs
        if 'ETG' in set_whisker_morphs:
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
        elif 'ETG-ETG' in set_whisker_morphs:    
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
        
        # KS test relaxation types between ETG and LTG
        if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

            res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
            
            if relaxation_type_i == 'co\n ↓ \nco':
                relaxation_type_i = 'co-co'
            if relaxation_type_i == 'co\n ↓ \ncounter':
                relaxation_type_i = 'co-counter'
            if relaxation_type_i == 'counter\n ↓ \nco':
                relaxation_type_i = 'counter-co'
            if relaxation_type_i == 'counter\n ↓ \ncounter':
                relaxation_type_i = 'counter-counter'
        
            if 'ETG' in set_whisker_morphs:
                print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            elif 'ETG-ETG' in set_whisker_morphs:  
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
            print('   p-value: %s' %res.pvalue)
        else:
            print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
    print('--------------------------------------')
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    

    if 'ETG' in set_whisker_morphs:
        order = ['ETG', 'LTG']
    elif 'ETG-ETG' in set_whisker_morphs:   
        order = ['ETG → ETG', 'LTG → LTG']
    #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
    #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
    my_pal = {order[0]: "r", order[1]: "b"}
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', fill=False, linewidth=1, alpha=1)
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', linewidth=0.01, alpha=0.1, legend=False)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-------------
    ### Formatting
    axs.set_xlim(left=0)
    #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_ylabel('Relaxation path')
    
    #print(max(relaxationtime_plot))
    
    
    #------------
    # Legend
    axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f Gyr\nMedian: %.2f Gyr\nstd: %.2f Gyr' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_timescale, median_timescale, std_timescale)}
    
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/violinplot_relaxation_morph/trelax_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/violinplot_relaxation_morph/trelax_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_box_and_whisker_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_whisker_morphs                  = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================   
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    for ID_i in misalignment_tree.keys():
        
        # only plot morphs we care about (default ETG-ETG, LTG-LTG)
        if 'ETG' in set_whisker_morphs:
            if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                continue
        elif 'ETG-ETG' in set_whisker_morphs: 
            if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                continue
        
        # Gather data
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            relaxationtype_plot.append('co\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            relaxationtype_plot.append('co\n ↓ \ncounter')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            relaxationtype_plot.append('counter\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            relaxationtype_plot.append('counter\n ↓ \ncounter')
        
        if 'ETG' in set_whisker_morphs:
            relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
        elif 'ETG-ETG' in set_whisker_morphs: 
            #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
    
    #----------------------
    # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
    print('\n--------------------------------------')
    print('tdyn')
    print('  Using sample: ', len(relaxationtype_plot))
    for relaxation_type_i in relaxation_type:
        
        if relaxation_type_i == 'co-co':
            relaxation_type_i = 'co\n ↓ \nco'
        if relaxation_type_i == 'co-counter':
            relaxation_type_i = 'co\n ↓ \ncounter'
        if relaxation_type_i == 'counter-co':
            relaxation_type_i = 'counter\n ↓ \nco'
        if relaxation_type_i == 'counter-counter':
            relaxation_type_i = 'counter\n ↓ \ncounter'
        
        # Select only relaxation morphs
        if 'ETG' in set_whisker_morphs:
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
        elif 'ETG-ETG' in set_whisker_morphs:    
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
        
        # KS test relaxation types between ETG and LTG
        if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

            res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
            
            if relaxation_type_i == 'co\n ↓ \nco':
                relaxation_type_i = 'co-co'
            if relaxation_type_i == 'co\n ↓ \ncounter':
                relaxation_type_i = 'co-counter'
            if relaxation_type_i == 'counter\n ↓ \nco':
                relaxation_type_i = 'counter-co'
            if relaxation_type_i == 'counter\n ↓ \ncounter':
                relaxation_type_i = 'counter-counter'
        
            if 'ETG' in set_whisker_morphs:
                print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            elif 'ETG-ETG' in set_whisker_morphs:  
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
            print('   p-value: %s' %res.pvalue)
        else:
            print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
    print('--------------------------------------')
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    

    if 'ETG' in set_whisker_morphs:
        order = ['ETG', 'LTG']
    elif 'ETG-ETG' in set_whisker_morphs:   
        order = ['ETG → ETG', 'LTG → LTG']
    #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
    #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
    my_pal = {order[0]: "r", order[1]: "b"}
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', fill=False, linewidth=1, alpha=1)
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', linewidth=0.01, alpha=0.1, legend=False)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-------------
    ### Formatting
    axs.set_xlim(left=0)
    #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Relaxation path')
    
    #------------
    # Legend
    axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_tdyn, median_tdyn, std_tdyn)}
    
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/violinplot_relaxation_morph/tdyn_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/violinplot_relaxation_morph/tdyn_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_box_and_whisker_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_whisker_morphs                  = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================  
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    for ID_i in misalignment_tree.keys():
        
        # only plot morphs we care about (default ETG-ETG, LTG-LTG)
        if 'ETG' in set_whisker_morphs:
            if (misalignment_tree['%s' %ID_i]['misalignment_morph'] not in set_whisker_morphs):
                continue
        elif 'ETG-ETG' in set_whisker_morphs: 
            if (misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_whisker_morphs):
                continue
        
        # Gather data
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        #relaxationtype_plot.append(misalignment_tree['%s' %ID_i]['relaxation_type'])
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
            relaxationtype_plot.append('co\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
            relaxationtype_plot.append('co\n ↓ \ncounter')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
            relaxationtype_plot.append('counter\n ↓ \nco')
        if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
            relaxationtype_plot.append('counter\n ↓ \ncounter')
        
        if 'ETG' in set_whisker_morphs:
            relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['misalignment_morph'])
        elif 'ETG-ETG' in set_whisker_morphs: 
            #relaxationmorph_plot.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation type': relaxationtype_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot})
    
    #----------------------
    # Run k-s test on co-co, counter-counter, counter-co, co-counter between morphology types
    print('\n--------------------------------------')
    print('ttorque')
    print('  Using sample: ', len(relaxationtype_plot))
    for relaxation_type_i in relaxation_type:
        
        if relaxation_type_i == 'co-co':
            relaxation_type_i = 'co\n ↓ \nco'
        if relaxation_type_i == 'co-counter':
            relaxation_type_i = 'co\n ↓ \ncounter'
        if relaxation_type_i == 'counter-co':
            relaxation_type_i = 'counter\n ↓ \nco'
        if relaxation_type_i == 'counter-counter':
            relaxation_type_i = 'counter\n ↓ \ncounter'
        
        # Select only relaxation morphs
        if 'ETG' in set_whisker_morphs:
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG')]
        elif 'ETG-ETG' in set_whisker_morphs:    
            df_ETG_ETG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'ETG → ETG')]
            df_LTG_LTG = df.loc[(df['Relaxation type'] == relaxation_type_i) & (df['Morphology'] == 'LTG → LTG')]
        
        # KS test relaxation types between ETG and LTG
        if (df_ETG_ETG.shape[0] > 0) and (df_LTG_LTG.shape[0] > 0):

            res = stats.ks_2samp(df_ETG_ETG['Relaxation time'], df_LTG_LTG['Relaxation time'])
            
            if relaxation_type_i == 'co\n ↓ \nco':
                relaxation_type_i = 'co-co'
            if relaxation_type_i == 'co\n ↓ \ncounter':
                relaxation_type_i = 'co-counter'
            if relaxation_type_i == 'counter\n ↓ \nco':
                relaxation_type_i = 'counter-co'
            if relaxation_type_i == 'counter\n ↓ \ncounter':
                relaxation_type_i = 'counter-counter'
        
            if 'ETG' in set_whisker_morphs:
                print('K-S TEST FOR ETG and LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            elif 'ETG-ETG' in set_whisker_morphs:  
                print('K-S TEST FOR ETG-ETG and LTG-LTG %s:    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
            print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((df_ETG_ETG.shape[0] + df_LTG_LTG.shape[0])/(df_ETG_ETG.shape[0]*df_LTG_LTG.shape[0])))))
            print('   p-value: %s' %res.pvalue)
        else:
            print('K-S TEST FOR ETG-ETG and LTG-LTG %s:\tSKIPPED    %s %s' %(relaxation_type_i, df_ETG_ETG.shape[0], df_LTG_LTG.shape[0]))
    print('--------------------------------------')
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    

    if 'ETG' in set_whisker_morphs:
        order = ['ETG', 'LTG']
    elif 'ETG-ETG' in set_whisker_morphs:   
        order = ['ETG → ETG', 'LTG → LTG']
    #sns.violinplot(data=df, y='Relaxation time', x='Morphology', hue='Relaxation type', scale='width', order=order, hue_order=['co-co', 'counter-counter', 'co-counter', 'counter-co'])
    #sns.violinplot(data=df, y='Relaxation time', x='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], inner='quart', linewidth=1)
    my_pal = {order[0]: "r", order[1]: "b"}
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', fill=False, linewidth=1, alpha=1)
    sns.violinplot(data=df, x='Relaxation time', y='Relaxation type', hue='Morphology', split=True, density_norm='width', gap=0.2, order=['co\n ↓ \nco', 'counter\n ↓ \ncounter', 'co\n ↓ \ncounter', 'counter\n ↓ \nco'], hue_order=['LTG → LTG', 'ETG → ETG'], palette=my_pal, inner='quart', linewidth=0.01, alpha=0.1, legend=False)
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #-------------
    ### Formatting
    axs.set_xlim(left=0)
    #axs.set_yticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Relaxation path')
    
    #------------
    # Legend
    axs.legend(loc='center right', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    metadata_plot = {'Author': '# MISALIGNMENTS: %s\nco-co: %s\ncnt-cnt: %s\nco-cnt: %s\ncnt-co: %s\nETG-ETG: %s\nLTG-LTG: %s\nETG-LTG: %s\nLTG-ETG: %s\nMean: %.2f\nMedian: %.2f\nstd: %.2f' %(len(misalignment_tree.keys()), len(summary_dict['ID']['co-co']), len(summary_dict['ID']['counter-counter']), len(summary_dict['ID']['co-counter']), len(summary_dict['ID']['counter-co']), len(summary_dict['ID']['ETG-ETG']), len(summary_dict['ID']['LTG-LTG']), len(summary_dict['ID']['ETG-LTG']), len(summary_dict['ID']['LTG-ETG']), mean_ttorque, median_ttorque, std_ttorque)}
    
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/violinplot_relaxation_morph/ttorque_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/violinplot_relaxation_morph/ttorque_relaxation_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plot delta angle, trelax. Looks at peak angle from 180
def _plot_offset_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_plot_offset_type                = ['co-co', 'counter-counter'],                     # [ 'co-co', 'co-counter' ]  or False
                      use_offset_morphs                   = True,
                        set_offset_morphs                 = ['LTG-LTG', 'ETG-ETG'],           # [ None / ['LTG-LTG', 'ETG-ETG'] ] Can be either relaxation_morph or misalignment_morph
                      #-----------------------
                      # General formatting
                      set_plot_offset_range               = [30, 150],                         # [min angle , max angle] of peak misangle
                        set_plot_offset_log               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    angles_plot = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # check offset angle within range            
        if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
            continue
        if set_plot_offset_type:
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                continue
                            
        # Add angles
        angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
            
        ID_plot.append(ID_i)
    
    # Collect data into dataframe
    print('  Using sample: ', len(ID_plot))
    df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
            
            
    #-------------
    # Plotting
    fig = plt.figure(figsize=(10/3, 10/3))
    gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    bin_width = (set_plot_offset_range[1] - set_plot_offset_range[0])/7
    bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
    c = 'k'
    
    # Bin hist data, find sigma percentiles
    binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    #print(bin_medians)
    
    #-------------------
    ### Plot scatter
    #ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=0.2, c='k', edgecolor='grey', marker='.', alpha=1, zorder=20)
    ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=2, c='k', edgecolor='k', marker='.', linewidths=0, alpha=0.5, zorder=-2)
    
    ### Plot upper, median, and lower sigma
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
    ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.2, zorder=-1)
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
    ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=105, label='sample')
    #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
    
    
    ### Plot histograms
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10, 0.125), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 10, 0.125), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    #-------------
    ### Plot by morphology
    if use_offset_morphs:
        for offset_morph_i in set_offset_morphs:
            if offset_morph_i == 'ETG-ETG':
                offset_morph_i = 'ETG → ETG'
                c = 'r'
                ls = '--'
            elif offset_morph_i == 'LTG-LTG':
                offset_morph_i = 'LTG → LTG'
                c = 'b'
                ls = 'dashdot'
        
            # Dataframe of morphs matching this
            df_morph = df.loc[df['Morphology'] == offset_morph_i]
        
            # Bin hist data, find sigma percentiles
            binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
            bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
            #print(bin_medians)
        
            #-------------------
            ### Plot scatter
            #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
        
            ### Plot upper, median, and lower sigma
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
            #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
            ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.9)
            
            
            yerr_top    = bin_medians[:,3] - bin_medians[:,2]
            yerr_bottom = bin_medians[:,2] - bin_medians[:,1]
            ax.errorbar(bins[:-1]+(bin_width/2), bin_medians[:,2], yerr=(yerr_bottom, yerr_top), lw=0.7, c=c, ls='none', zorder=99, alpha=0.9, ecolor=c, elinewidth=0.5, capsize=1.5)
            
            #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
            ### Plot histograms
            ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
            ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 10, 0.125), log=True, orientation='horizontal', facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
            #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 3.1, 0.125), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    
    #-----------
    ### General formatting
    # Axis labels
    ax.set_ylabel('$t_{\mathrm{relax}}$ [Gyr]')
    ax.set_xlabel('Peak angle from coplanarity')
    ax_histy.set_xlabel('Count')
    ax_histx.set_ylabel('Count')
    if set_plot_offset_log:
        ax.set_yscale('log')
        ax.set_ylim(0.1, 10)
        ax.set_yticks([0.1, 1, 10])
        ax.set_yticklabels(['0.1', '1', '10'])
    else:
        ax.set_ylim(0, 3)
    ax.set_xticks(np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, 15))
    ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
    
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([1, 10, 100, 1000])
    ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
    
    #------------
    ### Add title
    if set_plot_offset_type:
        
        set_title = ''
        
        if 'co-co' in set_plot_offset_type:
            set_title = set_title + r'co → co'
        if 'counter-counter' in set_plot_offset_type:
            set_title = set_title + r', counter → counter'
        if 'co-counter' in set_plot_offset_type:
            set_title = set_title + r', co → counter'
        if 'counter-co' in set_plot_offset_type:
            set_title = set_title + r', counter → co'
            
        ax_histx.set_title(r'%s' %set_title, size=7, loc='left', pad=3)
            
    
    #------------
    # Legend
    ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #------------
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/offset_angle/trelax_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/offset_angle/trelax_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_offset_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_plot_offset_type                = ['co-co', 'counter-counter'],                     # [ 'co-co', 'co-counter' ]  or False
                      use_offset_morphs                   = True,
                        set_offset_morphs                 = ['LTG-LTG', 'ETG-ETG'],           # [ None / ['LTG-LTG', 'ETG-ETG'] ] Can be either relaxation_morph or misalignment_morph
                      #-----------------------
                      # General formatting
                      set_plot_offset_range               = [30, 150],                         # [min angle , max angle] of peak misangle
                        set_plot_offset_log               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    angles_plot = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # check offset angle within range            
        if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
            continue
        if set_plot_offset_type:
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                continue
                            
        # Add angles
        angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
            
        ID_plot.append(ID_i)
        
    # Collect data into dataframe
    print('  Using sample: ', len(ID_plot))
    df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
            
    #-------------
    # Plotting
    fig = plt.figure(figsize=(10/3, 10/3))
    gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    bin_width = (set_plot_offset_range[1] - set_plot_offset_range[0])/7
    bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
    c = 'k'
    
    #-------------
    # Bin hist data, find sigma percentiles
    binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    #print(bin_medians)
    
    #-------------------
    ### Plot scatter
    ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=2, c='k', edgecolor='k', marker='.', linewidths=0, alpha=0.5, zorder=-2)
    
    ### Plot upper, median, and lower sigma
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
    ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.2, zorder=-1)
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
    ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=105, label='sample')
    #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
    
    
    ### Plot histograms
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 50, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 50, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
    
    
    
    #-------------
    ### Plot by morphology
    for offset_morph_i in set_offset_morphs:

        if offset_morph_i == 'ETG-ETG':
            offset_morph_i = 'ETG → ETG'
            c = 'r'
            ls = '--'
        elif offset_morph_i == 'LTG-LTG':
            offset_morph_i = 'LTG → LTG'
            c = 'b'
            ls = 'dashdot'
        
        # Dataframe of morphs matching this
        df_morph = df.loc[df['Morphology'] == offset_morph_i]
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.9)
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        yerr_top    = bin_medians[:,3] - bin_medians[:,2]
        yerr_bottom = bin_medians[:,2] - bin_medians[:,1]
        ax.errorbar(bins[:-1]+(bin_width/2), bin_medians[:,2], yerr=(yerr_bottom, yerr_top), lw=0.7, c=c, ls='none', zorder=99, alpha=0.9, ecolor=c, elinewidth=0.5, capsize=1.5)
        
        
        ### Plot histograms
        ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 50, 1), log=True, orientation='horizontal', facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 15.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    
    #-----------
    ### General formatting
    # Axis labels
    ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    ax.set_xlabel('Peak angle from coplanarity')
    ax_histy.set_xlabel('Count')
    ax_histx.set_ylabel('Count')
    if set_plot_offset_log:
        ax.set_yscale('log')
        ax.set_ylim(0.5, 50)
        ax.set_yticks([1, 10])
        ax.set_yticklabels(['1', '10'])
    else:
        ax.set_ylim(0, 15)
    ax.set_xticks(np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, 15))
    ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
    
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([1, 10, 100, 1000])
    ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
    
    #------------
    ### Add title
    if set_plot_offset_type:
        
        set_title = ''
        
        if 'co-co' in set_plot_offset_type:
            set_title = set_title + r'co → co'
        if 'counter-counter' in set_plot_offset_type:
            set_title = set_title + r', counter → counter'
        if 'co-counter' in set_plot_offset_type:
            set_title = set_title + r', co → counter'
        if 'counter-co' in set_plot_offset_type:
            set_title = set_title + r', counter → co'
            
        ax_histx.set_title(r'%s' %set_title, size=7, loc='left', pad=3)
            
    
    #------------
    # Legend
    ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #------------
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/offset_angle/tdyn_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/offset_angle/tdyn_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_offset_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_plot_offset_type                = ['co-co', 'counter-counter'],                     # [ 'co-co', 'co-counter' ]  or False
                      use_offset_morphs                   = True,
                        set_offset_morphs                 = ['LTG-LTG', 'ETG-ETG'],           # [ None / ['LTG-LTG', 'ETG-ETG'] ] Can be either relaxation_morph or misalignment_morph
                      #-----------------------
                      # General formatting
                      set_plot_offset_range               = [30, 150],                         # [min angle , max angle] of peak misangle
                        set_plot_offset_log               = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #========================================================================== 
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    angles_plot = []
    ID_plot     = []
    number_of_offset_40 = 0
    number_of_ttorque_1 = 0
    for ID_i in misalignment_tree.keys():
        # check offset angle within range            
        if not set_plot_offset_range[0] <=  misalignment_tree['%s' %ID_i]['angle_peak'] < set_plot_offset_range[1]:
            continue
        if set_plot_offset_type:
            if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_plot_offset_type:
                continue
                            
        # Add angles
        angles_plot.append(misalignment_tree['%s' %ID_i]['angle_peak'])
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
            
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] < 1:
            number_of_ttorque_1 += 1
            if misalignment_tree['%s' %ID_i]['angle_peak'] < 40:
                number_of_offset_40 += 1
        
            
    
    # Collect data into dataframe
    print('  Using sample: ', len(ID_plot))
    print('  number of sub ttorque 1 relaxations: \t%s' %number_of_ttorque_1)
    print('            ...of which sub 40 offset: \t%s' %number_of_offset_40)
    df = pd.DataFrame(data={'Peak misalignment angle': angles_plot, 'Morphology': relaxationmorph_plot, 'Relaxation time': relaxationtime_plot, 'GalaxyIDs': ID_plot})
            
    #-------------
    # Plotting
    fig = plt.figure(figsize=(10/3, 10/3))
    gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.1, hspace=0.1)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    bin_width = (set_plot_offset_range[1] - set_plot_offset_range[0])/7
    bins = np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, bin_width)
    c = 'k'
    
    #-------------
    # Bin hist data, find sigma percentiles
    binned_data_arg = np.digitize(df['Peak misalignment angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    #print(bin_medians)
        
    #-------------------
    ### Plot scatter
    ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=4, c='k', edgecolor='k', marker='.', linewidths=0, alpha=0.5, zorder=-2)
    #ax.scatter(df['Peak misalignment angle'], df['Relaxation time'], s=5, c='grey', edgecolor='k', marker='.', linewidths=0, alpha=0.6, zorder=-2)
        
    ### Plot upper, median, and lower sigma
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
    ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.2, zorder=-1)
    #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
    ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls='-', zorder=105, label='sample')
    #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        
    ### Plot histograms
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=1)
    ax_histx.hist(df['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 30, 1), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=c, histtype='step', alpha=0.8)
    ax_histy.hist(df['Relaxation time'], bins=np.arange(0, 30, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    #-------------
    ### Plot by morphology
    for offset_morph_i in set_offset_morphs:

        if offset_morph_i == 'ETG-ETG':
            offset_morph_i = 'ETG → ETG'
            c = 'r'
            ls = '--'
        elif offset_morph_i == 'LTG-LTG':
            offset_morph_i = 'LTG → LTG'
            c = 'b'
            ls = 'dashdot'
        
        # Dataframe of morphs matching this
        df_morph = df.loc[df['Morphology'] == offset_morph_i]
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df_morph['Peak misalignment angle'], bins=bins)
        bin_medians     = np.stack([np.percentile(df_morph['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
        
        #-------------------
        ### Plot scatter
        #ax.scatter(df_morph['Peak misalignment angle'], df_morph['Relaxation time'], s=0.05, c=c, marker='.', alpha=1, zorder=1)
        
        ### Plot upper, median, and lower sigma
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor=c, alpha=0.15, zorder=5)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor=c, alpha=0.35, zorder=6)
        #ax.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,2], bin_medians[:,3], facecolor=c, alpha=0.9, zorder=7, label=offset_morph_i)
        ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=ls, label=offset_morph_i, zorder=99, alpha=0.9)
        #ax.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.7, c=c, ls=':')
        
        yerr_top    = bin_medians[:,3] - bin_medians[:,2]
        yerr_bottom = bin_medians[:,2] - bin_medians[:,1]
        ax.errorbar(bins[:-1]+(bin_width/2), bin_medians[:,2], yerr=(yerr_bottom, yerr_top), lw=0.7, c=c, ls='none', zorder=99, alpha=0.9, ecolor=c, elinewidth=0.5, capsize=1.5)
        
        
        ### Plot histograms
        ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor='none', linewidth=0.7, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histx.hist(df_morph['Peak misalignment angle'], bins=bins, log=True, facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
        ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 30, 1), log=True, orientation='horizontal', facecolor='none', linewidth=0.8, edgecolor=c, histtype='step', alpha=0.8)
        #ax_histy.hist(df_morph['Relaxation time'], bins=np.arange(0, 10.1, 1), log=True, orientation='horizontal', facecolor=c, linewidth=1, edgecolor='none', alpha=0.1)
        
    
    #-----------
    ### General formatting
    # Axis labels
    ax.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    ax.set_xlabel('Peak angle displaced from coplanarity')
    ax_histy.set_xlabel('Count')
    ax_histx.set_ylabel('Count')
    if set_plot_offset_log:
        ax.set_yscale('log')
        ax.set_ylim(0.2, 25)
        ax.set_yticks([0.2, 1, 10])
        ax.set_yticklabels(['0.2', '1', '10'])
    else:
        ax.set_ylim(0, 10)
    ax.set_xticks(np.arange(set_plot_offset_range[0], set_plot_offset_range[1]+1, 15))
    ax.set_xlim(set_plot_offset_range[0], set_plot_offset_range[-1])
    
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xticks([1, 10, 100, 1000])
    ax_histy.set_xticklabels(['1', '$10^1$', '$10^2$', '$10^3$'])
    
    #------------
    ### Add title
    if set_plot_offset_type:
        
        set_title = ''
        
        if 'co-co' in set_plot_offset_type:
            set_title = set_title + r'co → co'
        if 'counter-counter' in set_plot_offset_type:
            set_title = set_title + r', counter → counter'
        if 'co-counter' in set_plot_offset_type:
            set_title = set_title + r', co → counter'
        if 'counter-co' in set_plot_offset_type:
            set_title = set_title + r', counter → co'
            
        ax_histx.set_title(r'%s' %set_title, size=7, loc='left', pad=3)       
    
    #------------
    # Legend
    ax.legend(loc='upper left', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #------------
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/offset_angle/ttorque_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/offset_angle/ttorque_offset_angle_morph_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()   


#-------------------------
# Number of mergers within window with relaxation time (suited for mass of 1010 and above)
def _plot_merger_count_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_min_merger_trelax               = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                      set_plot_merger_count_lim           = 0.1,          # stellar ratio (will also pick reciprocal)
                      add_plot_merger_count_gas           = True,         # will colour by gas ratio
                        set_plot_merger_count_log         = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    mergercount_plot     = []
    mergerstarratio_plot = []
    mergergasratio_plot  = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_merger_trelax:
            continue
        
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        
        # Find mergers within window
        merger_count = 0
        ratio_star_list = []
        ratio_gas_list  = [] 
        for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
            if len(merger_i) > 0:
                for merger_ii, gas_ii in zip(merger_i, gas_i):
                    if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                        merger_count += 1
                        ratio_star_list.append(merger_ii)
                        ratio_gas_list.append(gas_ii)
        
        # Append number of mergers, and average stellar and gas ratio of these mergers
        mergercount_plot.append(merger_count)
        mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
        mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
        

    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
    print('\n--------------------------------------')
    print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
    print('NUMBER OF MERGERS > 0.1 - RELAXATION TIME SPEARMAN:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    
    #-------------
    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
    
    df_0       = df.loc[(df['Number of mergers'] == 0)]
    df_mergers = df.loc[(df['Number of mergers'] > 0)]
    
    im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
    axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
    
    plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')
    
    
    #-------------
    ### Formatting
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
    if set_plot_merger_count_log:
        axs.set_xscale('log')
        axs.set_xlim(0.1, 6)
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(['0.1', '1', '10'])
    else:
        axs.set_xlim(0.5, 6)
        axs.set_xticks(np.arange(1, 6.1))
    axs.set_ylim(-0.3, 3.3)
    axs.set_yticks(np.arange(0, 3.1, 1))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    axs.set_title(r'$\rho$ = %.2f p-value = %.1e' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #------------
    # Legend
    axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/number_mergers_relaxtime/trelax_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/number_mergers_relaxtime/trelax_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_merger_count_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_min_merger_tdyn                 = 0,          # [ trelax/dyn ] min relaxation time, as we dont care about short relaxers
                      set_plot_merger_count_lim           = 0.1,          # stellar ratio (will also pick reciprocal)
                      add_plot_merger_count_gas           = True,         # will colour by gas ratio
                        set_plot_merger_count_log         = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    #===================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    mergercount_plot     = []
    mergerstarratio_plot = []
    mergergasratio_plot  = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] <= set_min_merger_tdyn:
            continue
        
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        
        # Find mergers within window
        merger_count = 0
        ratio_star_list = []
        ratio_gas_list  = [] 
        for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
            if len(merger_i) > 0:
                for merger_ii, gas_ii in zip(merger_i, gas_i):
                    if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                        merger_count += 1
                        ratio_star_list.append(merger_ii)
                        ratio_gas_list.append(gas_ii)
        
        # Append number of mergers, and average stellar and gas ratio of these mergers
        mergercount_plot.append(merger_count)
        mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
        mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
        

    print('  Using sample: ', len(ID_plot))    
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
    print('\n--------------------------------------')
    print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
    print('NUMBER OF MERGERS > 0.1 - RELAXATION TDYN SPEARMAN:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    
    #-------------
    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
    
    df_0       = df.loc[(df['Number of mergers'] == 0)]
    df_mergers = df.loc[(df['Number of mergers'] > 0)]
    
    im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
    axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
    
    plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')
    
    
    #-------------
    ### Formatting
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
    if set_plot_merger_count_log:
        axs.set_xscale('log')
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(['0.1', '1', '10'])
        axs.set_xlim(0.6, 60)
    else:
        axs.set_xlim(0, 20)
        axs.set_xticks(np.arange(0, 60, 5))
    axs.set_ylim(-0.3, 3.3)
    axs.set_yticks(np.arange(0, 3.1, 1))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    axs.set_title(r'$\rho$ = %.2f p-value = %.2f' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #------------
    # Legend
    axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/number_mergers_relaxtime/tdyn_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/number_mergers_relaxtime/tdyn_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_merger_count_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_min_merger_ttorque              = 0,          # [ trelax/torque ] min relaxation time, as we dont care about short relaxers
                      set_plot_merger_count_lim           = 0.1,          # stellar ratio (will also pick reciprocal)
                      add_plot_merger_count_gas           = True,         # will colour by gas ratio
                        set_plot_merger_count_log         = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    #===================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    mergercount_plot     = []
    mergerstarratio_plot = []
    mergergasratio_plot  = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] <= set_min_merger_ttorque:
            continue
        
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        
        # Find mergers within window
        merger_count = 0
        ratio_star_list = []
        ratio_gas_list  = [] 
        for merger_i, gas_i in zip(misalignment_tree['%s' %ID_i]['merger_ratio_stars'], misalignment_tree['%s' %ID_i]['merger_ratio_gas']):
            if len(merger_i) > 0:
                for merger_ii, gas_ii in zip(merger_i, gas_i):
                    if set_plot_merger_count_lim < merger_ii < (1/set_plot_merger_count_lim):
                        merger_count += 1
                        ratio_star_list.append(merger_ii)
                        ratio_gas_list.append(gas_ii)
        
        # Append number of mergers, and average stellar and gas ratio of these mergers
        mergercount_plot.append(merger_count)
        mergerstarratio_plot.append((0 if merger_count == 0 else np.mean(ratio_star_list)))
        mergergasratio_plot.append((0 if merger_count == 0 else np.mean(ratio_gas_list)))
        

    print('  Using sample: ', len(ID_plot))
              
    # Collect data into dataframe
    df = pd.DataFrame(data={'Number of mergers': mergercount_plot, 'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Mean stellar ratio': mergerstarratio_plot, 'Mean gas ratio': mergergasratio_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, mergercount_plot)
    print('\n--------------------------------------')
    print('Size of merger count sample: ', len([i for i in mergercount_plot if i > 0]))
    print('NUMBER OF MERGERS > 0.1 - RELAXATION TTORQUE SPEARMAN:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    
    #-------------
    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0, vmax=1.05, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')         #cmap=cm.coolwarm), cmap='sauron'
    
    df_0       = df.loc[(df['Number of mergers'] == 0)]
    df_mergers = df.loc[(df['Number of mergers'] > 0)]
    
    im1 = axs.scatter(df_mergers['Relaxation time'], df_mergers['Number of mergers'], c=df_mergers['Mean gas ratio'], s=10, norm=norm, cmap='viridis', zorder=99, edgecolors='k', linewidths=0.3, alpha=0.95)
    axs.scatter(df_0['Relaxation time'], df_0['Number of mergers'], c='lightgrey', s=10, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
    
    plt.colorbar(im1, ax=axs, label=r'$\bar{\mu}_{\mathrm{gas}}$', extend='max')

    
    #-------------
    ### Formatting
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Number of mergers' +'\n' + r'($\bar{\mu}_{\mathrm{*}}>0.1$)')
    if set_plot_merger_count_log:
        axs.set_xscale('log')
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(['0.1', '1', '10'])
        axs.set_xlim(0.6, 40)
    else:
        axs.set_xlim(0.5, 40)
        axs.set_xticks(np.arange(2, 40, 4))
    axs.set_ylim(-0.3, 3.3)
    axs.set_yticks(np.arange(0, 3.1, 1))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    axs.set_title(r'$\rho$ = %.2f p-value = %.2f' %(res.correlation, res.pvalue), size=7, loc='left', pad=3)
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #------------
    # Legend
    axs.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    
    #------------
    ### other
    plt.tight_layout()
    
    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/number_mergers_relaxtime/ttorque_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/number_mergers_relaxtime/ttorque_Nmergers_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots hist of average DM-stars misangle co-co preceeding misalignment with fraction
def _plot_halo_misangle_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution          = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_trelax                 = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      use_only_centrals              = True,        # Use only centrals
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_halo_trelax:
            continue
            
        # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
        if use_only_centrals: 
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            
            if not np.all(sgn == 0):
                continue
            
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        
        
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
                        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle trelax plot: ', len(ID_plot))
    print('Stars-DM misangle vs trelax:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
    plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line for total sample
    ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter.set_ylabel('$t_{\mathrm{relax}}$ [Gyr]')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
    
    if set_plot_halo_misangle_log:
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylim(0.1, 10)
        ax_scatter.set_yticks([0.1, 1, 10])
        ax_scatter.set_yticklabels(['0.1', '1', '10'])
    else:
        ax_scatter.set_ylim(0, 6)
        ax_scatter.set_yticks(np.arange(0, 6.1, 1))
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter.set_xlim(0, 180)
    ax_scatter.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if plot_annotate:
        ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/trelax_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/trelax_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_halo_misangle_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution        = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_tdyn                 = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      use_only_centrals              = True,        # Use only centrals
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_tdyn'] <= set_min_halo_tdyn:
            continue
            
        # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
        if use_only_centrals: 
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            
            if not np.all(sgn == 0):
                continue
            
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
                        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle tdyn plot: ', len(ID_plot))
    print('Stars-DM misangle vs tdyn:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, alpha=0.8)
    plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #--------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line
    ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
    if set_plot_halo_misangle_log:
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylim(0.3, 35)
        ax_scatter.set_yticks([1, 10])
        ax_scatter.set_yticklabels(['1', '10'])
    else:
        ax_scatter.set_ylim(0, 30)
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter.set_xlim(0, 180)
    ax_scatter.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if plot_annotate:
        ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/tdyn_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/tdyn_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_halo_misangle_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution          = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_ttorque                = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      use_only_centrals              = True,        # Use only centrals
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_ttorque'] <= set_min_halo_ttorque:
            continue
            
        # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
        if use_only_centrals: 
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            
            if not np.all(sgn == 0):
                continue
        
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
                        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle ttorque plot: ', len(ID_plot))
    print('Stars-DM misangle vs ttorque:')
    print('   ρ:       %.2f' %res.correlation)
    print('   p-value: %s' %res.pvalue)
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter, ax_line) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2.5, 1]}, figsize=[10/3, 2.5], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    #im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    im1 = ax_scatter.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    plt.colorbar(im1, ax=ax_scatter, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    #--------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line
    ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101)
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
        
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ during relaxation')
    if set_plot_halo_misangle_log:
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylim(0.1, 25)
        ax_scatter.set_yticks([0.1, 1, 10])
        ax_scatter.set_yticklabels(['0.1', '1', '10'])
    else:
        ax_scatter.set_ylim(0, 10)
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter.set_xlim(0, 180)
    ax_scatter.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if plot_annotate:
        ax_scatter.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/ttorque_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/ttorque_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# manual    
def _plot_halo_misangle_manual(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_halo_trelax_resolution          = 0,        # [ Gyr ] min filter applied to ALL, to avoid resolution limits 
                      set_min_halo_trelax                 = 0,          # [ Gyr ] min relaxation time, as we dont care about short relaxers
                        add_plot_halo_morph_median        = True,
                        set_plot_halo_misangle_log        = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    relaxationtime_plot  = []
    relaxationtorque_plot  = []
    relaxationmorph_plot = []
    relaxationkappa_plot = []
    halomisangle_plot    = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # remove misalignments that are too below resolution
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_halo_trelax_resolution:
            continue
        
        # remove misalignments that are too short
        if misalignment_tree['%s' %ID_i]['relaxation_time'] <= set_min_halo_trelax:
            continue
            
        # Collect relaxation time
        relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
        relaxationtorque_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        else:
            relaxationmorph_plot.append('other')
        
        # Collect average kappa during misalignment
        relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        
        # Collect average stellar-DM misalignment angle
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']+1]))
        

    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation torque': relaxationtorque_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation morph': relaxationmorph_plot, 'DM-stars angle': halomisangle_plot, 'GalaxyIDs': ID_plot})
    
    #-------------
    # Stats
    res = stats.spearmanr(relaxationtime_plot, halomisangle_plot)
    print('\n--------------------------------------')
    print('Size of halo-misangle trelax plot: ', len(ID_plot))
    print('--------------------------------------')
    
    #-------------
    # Plotting scatter
    fig, (ax_scatter1, ax_scatter2, ax_line) = plt.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [2.5, 2.5, 1]}, figsize=[10/3, 4.0], sharex=True, sharey=False, layout='constrained')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Normalise colormap
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = ax_scatter1.scatter(df['DM-stars angle'], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=0.9)
    plt.colorbar(im1, ax=ax_scatter1, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    im2 = ax_scatter2.scatter(df['DM-stars angle'], df['Relaxation torque'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=0.9)
    plt.colorbar(im2, ax=ax_scatter2, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line for total sample
    #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
    #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='--', zorder=101)
    ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=101, label='sample')
    #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=1, c='k', ls='--', zorder=101)

    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation torque'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    print(bin_medians)
    
    # Plot average line for total sample
    #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
    ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
    #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='--', zorder=101)
    ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=101, label='sample')
    #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=1, c='k', ls='--', zorder=101)
    
    if add_plot_halo_morph_median:
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='b', ls='--', zorder=101)
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='b', ls='--', zorder=101)
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter1.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$')
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='r', ls='--', zorder=101)
        #ax_scatter1.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='r', ls='--', zorder=101)
    
        #-------------
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation torque'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='b', alpha=0.2, zorder=6)
        ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='b', ls='--', zorder=101)
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='b', ls='--', zorder=101)
    
        # Bin hist data, find sigma percentiles
        bin_width = 15
        bins = np.arange(0, 181, bin_width)
        binned_data_arg = np.digitize(df['DM-stars angle'][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation torque'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
        #ax_scatter2.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='r', alpha=0.2, zorder=6)
        ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$')
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.8, c='r', ls='--', zorder=101)
        #ax_scatter2.plot(bins[:-1]+(bin_width/2), bin_medians[:,3], lw=0.8, c='r', ls='--', zorder=101)
        
    
    #-------------
    # Plotting hist
    #bins = np.arange(0, 181, 30)
    #ax_hist.hist(df['DM-stars angle'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='k', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'LTG → LTG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C0', histtype='step', alpha=1)
    #ax_hist.hist(df['DM-stars angle'][df['Relaxation morph'] == 'ETG → ETG'], bins=bins, log=False, facecolor='none', linewidth=1, edgecolor='C1', histtype='step', alpha=1)
    
    
    #-------------
    # Plotting average kappa
    
    # Bin hist data, find sigma percentiles
    bin_width = 15
    bins = np.arange(0, 181, bin_width)
    binned_data_arg = np.digitize(df['DM-stars angle'], bins=bins)
    bin_medians     = np.stack([np.percentile(df['Relaxation kappa'][binned_data_arg == i], q=[5, 16, 50, 84, 95]) for i in range(1, len(bins))])
    
    ### Plot upper, median, and lower sigma
    #ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,4], facecolor='k', alpha=0.2, zorder=6)
    ax_line.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,1], bin_medians[:,3], facecolor='k', alpha=0.2, zorder=6)
    ax_line.plot(bins[:-1]+(bin_width/2), bin_medians[:,2], lw=0.8, c='k', ls='-', zorder=7)
    
    
    #-------------
    ### Formatting
    ax_scatter1.set_ylabel('$t_{\mathrm{relax}}$ [Gyr]')
    ax_scatter2.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    #ax_hist.set_ylabel('Count')
    ax_line.set_ylabel(r'$\bar{\kappa}_{\mathrm{co}}^{*}$')
    ax_line.set_xlabel('Average stellar-DM misalignment angle')
    if set_plot_halo_misangle_log:
        ax_scatter1.set_yscale('log')
        ax_scatter1.set_ylim(0.1, 10)
        ax_scatter1.set_yticks([0.1, 1, 10])
        ax_scatter1.set_yticklabels(['0.1', '1', '10'])
        ax_scatter2.set_yscale('log')
        ax_scatter2.set_ylim(0.1, 25)
        ax_scatter2.set_yticks([0.1, 1, 10])
        ax_scatter2.set_yticklabels(['0.1', '1', '10'])
    else:
        ax_scatter1.set_ylim(0, 6)
        ax_scatter1.set_yticks(np.arange(0, 6.1, 1))
        ax_scatter2.set_ylim(0, 10)
    #ax_hist.set_yscale('log')
    #ax_hist.set_ylim(bottom=0)
    ax_line.set_ylim(0.1, 0.6)
    ax_line.set_yticks(np.arange(0.2, 0.61, 0.2))
    ax_scatter1.set_xlim(0, 180)
    ax_scatter1.set_xticks(np.arange(0, 180.1, 30))
    for ax in [ax_scatter1, ax_scatter2, ax_line]:
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
        
    #-----------
    ### title
    if plot_annotate:
        ax_scatter1.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    ax_scatter1.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)
    ax_scatter2.legend(loc='best', frameon=False, labelspacing=0.1, handlelength=1)

    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_misangle_relaxtime/both_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_misangle_relaxtime/both_halomisangle_trelax_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots hist of average DM-stars misangle co-co preceeding misalignment with fraction
def _plot_halo_misangle_pre_frac(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_misanglepre_morph               = True,         # will use a stacked histogram for ETG-ETG, LTG-LTG, ETG-LTG, etc.
                      set_misanglepre_type                = ['co-co', 'co-counter'],           # [ 'co-co', 'co-counter' ]  or False
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    halomisangle_plot    = []
    relaxationmorph_plot = []
    ID_plot     = []
    for ID_i in misalignment_tree.keys():
        
        # If not a co-co or co-counter, skip
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_misanglepre_type:
            continue
        
        # Collect average stellar-DM misangle before misalignment.
        halomisangle_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stars_dm'][0:misalignment_tree['%s' %ID_i]['index_s']+1]))
        ID_plot.append(ID_i)
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
            relaxationmorph_plot.append('ETG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
            relaxationmorph_plot.append('LTG → LTG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
            relaxationmorph_plot.append('LTG → ETG')
        elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
            relaxationmorph_plot.append('ETG → LTG')
        else:
            relaxationmorph_plot.append('other')
        

    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'DM-stars angle': halomisangle_plot, 'Relaxation morph': relaxationmorph_plot, 'GalaxyIDs': ID_plot})
    ETG_ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    LTG_LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    ETG_LTG_df = df.loc[(df['Relaxation morph'] == 'ETG → LTG')]
    LTG_ETG_df = df.loc[(df['Relaxation morph'] == 'LTG → ETG')]
    
    #-------------
    # Plotting ongoing fraction histogram
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    colors = ['orangered', 'orange', 'cornflowerblue', 'mediumblue']
    
    if set_misanglepre_morph:
        axs.hist([ETG_ETG_df['DM-stars angle'], ETG_LTG_df['DM-stars angle'], LTG_ETG_df['DM-stars angle'], LTG_LTG_df['DM-stars angle']], weights=(np.ones(len(ETG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(ETG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs'])), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', color = colors, alpha=0.5, stacked=True)
        hist_n, _, _ = axs.hist([ETG_ETG_df['DM-stars angle'], ETG_LTG_df['DM-stars angle'], LTG_ETG_df['DM-stars angle'], LTG_LTG_df['DM-stars angle']], weights=(np.ones(len(ETG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(ETG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_ETG_df['GalaxyIDs']))/len(df['GalaxyIDs']), np.ones(len(LTG_LTG_df['GalaxyIDs']))/len(df['GalaxyIDs'])), bins=np.arange(0, 181, 10), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7, stacked=True)
    else:
        axs.hist(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), histtype='bar', edgecolor='none', alpha=0.5)
        hist_n, _, _ = axs.hist(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), histtype='bar', facecolor='none', edgecolor='k', alpha=0.9, lw=0.7)
        
    print('Hist bins morphology: ', hist_n)
    hist_n, _ = np.histogram(df['DM-stars angle'], weights=np.ones(len(df['GalaxyIDs']))/len(df['GalaxyIDs']), bins=np.arange(0, 181, 10), range=(0, 181))
    print('Hist bins total: ', hist_n)
        
    #-------------
    ### Formatting
    axs.set_xlabel(r'$\bar{\psi}_{\mathrm{DM-stars}}$ pre-instability')
    axs.set_xlim(0, 180)
    axs.set_xticks(np.arange(0, 181, step=30))
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(0, 0.2)
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate == None:
        plot_annotate = ''
    if set_misanglepre_type:
        plot_annotate = plot_annotate + set_misanglepre_type[0] + ', ' + set_misanglepre_type[1]
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    #------------
    # Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    
    if set_misanglepre_morph:
        legend_labels.append('ETG → ETG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orangered')
            
        legend_labels.append('ETG → LTG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orange')
            
        legend_labels.append('LTG → ETG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('cornflowerblue')
            
        legend_labels.append('LTG → LTG')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('mediumblue')
        
        axs.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0)
        
    #------------
    ### other
    #plt.tight_layout()

    #-----------
    # savefig
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/halo_premisangle_fraction/halopremisangle_fraction_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/halo_premisangle_fraction/halopremisangle_fraction_%s_%s.%s" %(fig_dir, len(misalignment_tree.keys()), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots stacked histograms for different morphologies, and fraction caused by major, minor, or other origins
def _plot_origins(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      use_only_start_morph     = True,          # use only ETG - and LTG - 
                      set_origins_morph        = ['ETG-ETG', 'ETG-LTG', 'LTG-ETG', 'LTG-LTG'],
                      add_total                = True,          # Add total in sample
                      #-----------------------
                      # Mergers
                      use_alt_merger_criteria = True,
                        half_window         = 0.3,      # [ 0.2 / +/-Gyr ] window centred on first misaligned snap to look for mergers
                        min_ratio           = 0.1,   
                        merger_lookback_time = 2,       # Gyr, number of years to check for peak stellar mass
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Loading mergertree file to establish windows
    f = h5py.File(tree_dir + 'Snip100_MainProgenitorTrees.hdf5', 'r')
    GalaxyID_tree             = np.array(f['Histories']['GalaxyID'])
    DescendantID_tree         = np.array(f['Histories']['DescendantID'])
    Lookbacktime_tree         = np.array(f['Snapnum_Index']['LookbackTime'])
    StellarMass_tree          = np.array(f['Histories']['StellarMass'])
    GasMass_tree              = np.array(f['Histories']['GasMass'])
    f.close()
    
    tally_minor = []
    tally_major = []
    tally_other = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_morph'] not in set_origins_morph:
            continue
            
        
        #---------------------------------------------------------
        # Find location of begin of misalignment in merger tree
        (row_i, snap_i) = np.where(GalaxyID_tree == int(misalignment_tree['%s' %ID_i]['GalaxyID'][misalignment_tree['%s' %ID_i]['index_s']+1]))
        row_mask  = row_i[0]
        snap_mask = snap_i[0]
        
        # Find window limits of mergers [SnapNum_merger_min:SnapNum_merger_max]
        SnapNum_merger_min = 1 + np.where(Lookbacktime_tree >= (Lookbacktime_tree[snap_i] + half_window))[0][-1]
        if len(np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0]) > 0:
            SnapNum_merger_max = np.where(Lookbacktime_tree <= (Lookbacktime_tree[snap_i] - half_window))[0][0]
        else:
            SnapNum_merger_max = snap_mask
        
        # List of all elligible descendants
        GalaxyID_list       = np.array(GalaxyID_tree)[row_mask, SnapNum_merger_min:SnapNum_merger_max]
        

        merger_ID_array_array    = []
        merger_ratio_array_array = []
        merger_gas_array_array   = []
        for SnapNum_i, GalaxyID_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), GalaxyID_list):
            if int(GalaxyID_i) == -1:
                continue
            
            merger_mask = [i for i in np.where(np.array(DescendantID_tree)[:,int(SnapNum_i-1)] == GalaxyID_i)[0] if i != row_mask]
            
            # If misalignment found, its position is given by i in merger_mask, SnapNum_i
            merger_ID_array    = []
            merger_ratio_array = []
            merger_gas_array   = []
            if len(merger_mask) > 0:
                # find peak stelmass of those galaxies
                for mask_i in merger_mask:
                    # Find last snap up to 2 Gyr ago
                    SnapNum_merger = np.where(Lookbacktime_tree >= (Lookbacktime_tree[SnapNum_i] + merger_lookback_time))[0][-1]
            
                    # Find largest stellar mass of this satellite, per method of Rodriguez-Gomez et al. 2015, Qu et al. 2017 (see crain2017)
                    mass_mask = np.argmax(StellarMass_tree[mask_i][int(SnapNum_merger-100):int(SnapNum_i)]) + (SnapNum_merger-100)
            
                    # Extract secondary properties
                    primary_stelmass   = StellarMass_tree[row_mask][mass_mask]
                    primary_gasmass    = GasMass_tree[row_mask][mass_mask]
                    component_stelmass = StellarMass_tree[mask_i][mass_mask]
                    component_gasmass  = GasMass_tree[mask_i][mass_mask]
            
                    if primary_stelmass <= 0.0:
                        # Adjust stelmass
                        primary_stelmass   = math.nan
                        primary_gasmass    = math.nan
                

                    # Find ratios
                    merger_ratio = component_stelmass / primary_stelmass 
                    if merger_ratio > 1:
                        merger_ratio = 1/merger_ratio
                    gas_ratio    = (primary_gasmass + component_gasmass) / (primary_stelmass + component_stelmass)

                    # Append
                    merger_ID_array.append(GalaxyID_tree[mask_i][int(SnapNum_i-1)])
                    merger_ratio_array.append(merger_ratio)
                    merger_gas_array.append(gas_ratio)
                    
            merger_ID_array_array.append(merger_ID_array)
            merger_ratio_array_array.append(merger_ratio_array)
            merger_gas_array_array.append(merger_gas_array)      
        if debug:
            print(misalignment_tree['%s' %ID_i]['SnapNum'])
            print(misalignment_tree['%s' %ID_i]['merger_ratio_stars'])
            for snap_i, star_i in zip(np.arange(SnapNum_merger_min, SnapNum_merger_max+1), merger_ratio_array_array):
                print(snap_i, star_i)
                
            
        merger_count = 0
        for merger_i in merger_ratio_array_array:
            if len(merger_i) > 0:
                if (max(merger_i) > min_ratio):
                    if merger_count == 0:
                        if 0.3 > max(merger_i) > 0.1:
                            tally_minor.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                        if max(merger_i) > 0.3:
                            tally_major.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
                    merger_count += 1
        if merger_count == 0:
            tally_other.append(misalignment_tree['%s' %ID_i]['relaxation_morph'])
    

    print('  Using sample: ', len(merger_ID_array_array))     
    tally_total_1 = len(tally_major)+len(tally_minor)+len(tally_other)
    
    print('\n======================================')
    print('Using merger criteria, half_window = %.1f Gyr, min_ratio = %.1f' %(half_window, min_ratio))
    print('\tmajor: %s    %.3f  ' %(len(tally_major), (len(tally_major)/tally_total_1)))
    print('\tminor: %s    %.3f  ' %(len(tally_minor), (len(tally_minor)/tally_total_1)))
    print('\tother: %s    %.3f  ' %(len(tally_other), (len(tally_other)/tally_total_1)))
    print('\ttotal: %s   ' %tally_total_1)
    
    
    # left with tallo_minor = ['ETG-ETG', 'LTG-ETG', 'ETG-ETG', etc..]
    plot_dict = {'ETG-ETG': {}, 'LTG-ETG': {}, 'ETG-LTG': {}, 'LTG-LTG': {}}
    for dict_i in plot_dict.keys():
        plot_dict[dict_i] = {'array': [],
                             'major': [],
                             'minor': [],
                             'other': []}
    
    for morph_i in set_origins_morph:
        plot_dict['%s' %morph_i]['major'] = len([i for i in tally_major if i == morph_i])
        plot_dict['%s' %morph_i]['minor'] = len([i for i in tally_minor if i == morph_i])
        plot_dict['%s' %morph_i]['other'] = len([i for i in tally_other if i == morph_i])
        
        temp_array = []
        temp_array.extend([i for i in tally_major if i == morph_i])
        temp_array.extend([i for i in tally_minor if i == morph_i])
        temp_array.extend([i for i in tally_other if i == morph_i])
        plot_dict['%s' %morph_i]['array'] = temp_array
        
        
        
    #---------------------
    ### Plotting
    if not use_only_start_morph:
        #-----------
        # Plotting ongoing fraction histogram
        fig, ((ax_ETG_ETG, ax_ETG_LTG), (ax_LTG_ETG, ax_LTG_LTG), (ax_spare_1, ax_spare_2)) = plt.subplots(3, 2, figsize=[10/3, 10/3], gridspec_kw={'height_ratios': [1, 1, 0.1]}, sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0, hspace=0)
        
        #-----------
        # Add pie plots
        colors = ['royalblue', 'orange', 'orangered']
        for morph_i in set_origins_morph:
            y = np.array([plot_dict['%s' %morph_i]['other'], plot_dict['%s' %morph_i]['minor'], plot_dict['%s' %morph_i]['major']])
            mylabels = ['%s' %(' ' if plot_dict['%s' %morph_i]['other'] == 0 else plot_dict['%s' %morph_i]['other']), '%s' %(' ' if plot_dict['%s' %morph_i]['minor'] == 0 else plot_dict['%s' %morph_i]['minor']), '%s' %(' ' if plot_dict['%s' %morph_i]['major'] == 0 else plot_dict['%s' %morph_i]['major'])]
        
            if morph_i == 'ETG-ETG':
                ax_ETG_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_ETG_ETG.set_title('ETG → ETG', loc='center', pad=0, fontsize=8)
            if morph_i == 'ETG-LTG':
                ax_ETG_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_ETG_LTG.set_title('ETG → LTG', loc='center', pad=0, fontsize=8)
            if morph_i == 'LTG-ETG':
                ax_LTG_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_spare_1.set_title('\nLTG → ETG', loc='center', y=-1, fontsize=8)
            if morph_i == 'LTG-LTG':
                ax_LTG_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_spare_2.set_title('\nLTG → LTG', loc='center', y=-1, fontsize=8)
        ax_spare_1.axis('off')
        ax_spare_2.axis('off')
    
            
        #------------
        # Add legend
        legend_labels   = []
        legend_elements = []
        legend_colors   = []

        legend_labels.append('Major\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C2')
        legend_labels.append('Minor\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C1')
        legend_labels.append('Other')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('C0')
    
        ax_LTG_LTG.legend(handles=legend_elements, labels=legend_labels, loc='lower left', bbox_to_anchor=(0.9, 0.7), labelspacing=1, frameon=False, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        #-----------
        # savefig
        if savefig:
            savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/misangle_origins/origins_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misangle_origins/origins_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()  
    elif add_total:
        #-----------
        # Plotting ongoing fraction histogram
        #fig, ((ax_tot, _),( ax_ETG, ax_LTG)) = plt.subplots(2, 2, figsize=[10/3, 2.5], sharex=True, sharey=False)
        #plt.subplots_adjust(wspace=0, hspace=0)
        fig = plt.figure(layout='constrained', figsize=[10/3, 2.5])
        gs = GridSpec(2, 2, figure=fig)
        ax_tot = fig.add_subplot(gs[0, :])
        ax_ETG = fig.add_subplot(gs[1, 0])
        ax_LTG = fig.add_subplot(gs[1, 1])
        plt.subplots_adjust(wspace=0, hspace=0)
        
        
        #-----------
        # Add pie plots
        #colors = ['orange', 'cornflowerblue', 'blue']
        colors = ['royalblue', 'orange', 'orangered']
        for morph_i in ['ETG', 'LTG']:
            
            y = np.array([plot_dict['%s-ETG' %morph_i]['other']+plot_dict['%s-LTG' %morph_i]['other'], plot_dict['%s-ETG' %morph_i]['minor']+plot_dict['%s-LTG' %morph_i]['minor'], plot_dict['%s-ETG' %morph_i]['major']+plot_dict['%s-LTG' %morph_i]['major']])
            mylabels = ['%s' %(' ' if y[0] == 0 else y[0]), '%s' %(' ' if y[1] == 0 else y[1]), '%s' %(' ' if y[2] == 0 else y[2])]
            
            if morph_i == 'ETG':
                ax_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_ETG.set_title('ETG', loc='center', pad=0, fontsize=8)
                ax_ETG.pie([1], startangle = 90, colors = ['w'], radius=0.5, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
            if morph_i == 'LTG':
                ax_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_LTG.set_title('LTG', loc='center', pad=0, fontsize=8)
                ax_LTG.pie([1], startangle = 90, colors = ['w'], radius=0.5, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                
                
        #-----------
        # Add total pie plot
        y = np.array([plot_dict['ETG-ETG']['other']+plot_dict['ETG-LTG']['other']+plot_dict['LTG-ETG']['other']+plot_dict['LTG-LTG']['other'], plot_dict['ETG-ETG']['minor']+plot_dict['ETG-LTG']['minor']+plot_dict['LTG-ETG']['minor']+plot_dict['LTG-LTG']['minor'], plot_dict['ETG-ETG']['major']+plot_dict['ETG-LTG']['major']+plot_dict['LTG-ETG']['major']+plot_dict['LTG-LTG']['major']])
        mylabels = ['%s' %(' ' if y[0] == 0 else y[0]), '%s' %(' ' if y[1] == 0 else y[1]), '%s' %(' ' if y[2] == 0 else y[2])]
        
        ax_tot.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
        ax_tot.set_title('Total', loc='center', pad=0, fontsize=8)
        ax_tot.pie([1], startangle = 90, colors = ['w'], radius=0.5, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
        
            
        #------------
        # Add legend
        legend_labels   = []
        legend_elements = []
        legend_colors   = []

        legend_labels.append('Major\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orangered')
        legend_labels.append('Minor\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orange')
        legend_labels.append('Other')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('royalblue')
    
        ax_tot.legend(handles=legend_elements, labels=legend_labels, loc='center right', bbox_to_anchor=(1.6, 0.6), labelspacing=1, frameon=False, labelcolor=legend_colors, handlelength=0, ncol=1, handletextpad=0)
        
        #-----------
        # savefig
        if savefig:
            savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/misangle_origins/origins_total_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misangle_origins/origins_total_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()    
    else:
        #-----------
        # Plotting ongoing fraction histogram
        fig, (ax_ETG, ax_LTG) = plt.subplots(1, 2, figsize=[10/3, 2.5], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0, hspace=0)
        
        #-----------
        # Add pie plots
        #colors = ['orange', 'cornflowerblue', 'blue']
        colors = ['royalblue', 'orange', 'orangered']
        for morph_i in ['ETG', 'LTG']:
            
            y = np.array([plot_dict['%s-ETG' %morph_i]['other']+plot_dict['%s-LTG' %morph_i]['other'], plot_dict['%s-ETG' %morph_i]['minor']+plot_dict['%s-LTG' %morph_i]['minor'], plot_dict['%s-ETG' %morph_i]['major']+plot_dict['%s-LTG' %morph_i]['major']])
            mylabels = ['%s' %(' ' if y[0] == 0 else y[0]), '%s' %(' ' if y[1] == 0 else y[1]), '%s' %(' ' if y[2] == 0 else y[2])]
            
            if morph_i == 'ETG':
                ax_ETG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_ETG.set_title('ETG', loc='center', pad=0, fontsize=8)
                ax_ETG.pie([1], startangle = 90, colors = ['w'], radius=0.5, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
            if morph_i == 'LTG':
                ax_LTG.pie(y, labels = mylabels, startangle = 90, colors = colors, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
                ax_LTG.set_title('LTG', loc='center', pad=0, fontsize=8)
                ax_LTG.pie([1], startangle = 90, colors = ['w'], radius=0.5, wedgeprops = {'linewidth': 1, 'edgecolor': 'w'})
        
            
        #------------
        # Add legend
        legend_labels   = []
        legend_elements = []
        legend_colors   = []

        legend_labels.append('Major\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orangered')
        legend_labels.append('Minor\nmerger')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orange')
        legend_labels.append('Other')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('royalblue')
    
        ax_LTG.legend(handles=legend_elements, labels=legend_labels, loc='center left', bbox_to_anchor=(0.9, 0.5), labelspacing=1, frameon=False, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        #-----------
        # savefig
        if savefig:
            savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/misangle_origins/origins_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/misangle_origins/origins_altmorph_%s_%s.%s" %(fig_dir, len(tally_major)+len(tally_minor), savefig_txt, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    """  
    # Plotting ongoing fraction histogram
    fig, axs = plt.subplots(1, 1, figsize=[2.5, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #------------
    # plotting bar
    bottom = np.zeros(4)
    weight_counts = {
        'Major': np.array([plot_dict['ETG-ETG']['major']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['major']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['major']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['major']/len(plot_dict['LTG-LTG']['array'])]),
        'Minor': np.array([plot_dict['ETG-ETG']['minor']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['minor']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['minor']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['minor']/len(plot_dict['LTG-LTG']['array'])]),
        'Other': np.array([plot_dict['ETG-ETG']['other']/len(plot_dict['ETG-ETG']['array']), plot_dict['LTG-ETG']['other']/len(plot_dict['LTG-ETG']['array']), plot_dict['ETG-LTG']['other']/len(plot_dict['ETG-LTG']['array']), plot_dict['LTG-LTG']['other']/len(plot_dict['LTG-LTG']['array'])])}
    bar_x = []
    for morph_i in set_origins_morph:
        if morph_i == 'ETG-ETG':
            bar_x.append('ETG →\nETG')
        if morph_i == 'LTG-ETG':
            bar_x.append('LTG →\nETG')
        if morph_i == 'ETG-LTG':
            bar_x.append('ETG →\nLTG')
        if morph_i == 'LTG-LTG':
            bar_x.append('LTG →\nLTG')
    
    for boolean, weight_count in weight_counts.items():
        axs.bar(bar_x, weight_count, width=0.5, bottom=bottom, edgecolor='none', alpha=0.5, label=boolean)
        axs.bar(bar_x, weight_count, width=0.5, bottom=bottom, facecolor='none', edgecolor='k', alpha=0.9, lw=0.7)
        bottom += weight_count
    
    
    #-------------
    ### Formatting
    axs.set_xlabel('Relaxation type')
    axs.set_ylabel('Percentage of misalignments')
    #axs.set_ylim(0, 1)
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=0))
    #axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    #-----------
    ### title
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    
    #------------
    # Legend
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, labelspacing=0.1, handlelength=0.5)
    """
    

#-------------------------
# Plots scatter of gas fraction vs relax time
def _plot_timescale_gas_scatter_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['co-co'],          # which paths to use
                      set_gashist_min_trelax              = 0.2,                # removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
    print('  Using sample: ', len(ID_plot))
        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #-------------
    ### Plotting scatter
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-------------
    # Colourbar for kappa
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = axs.scatter(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    plt.colorbar(im1, ax=axs, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    if add_plot_gas_morph_median:

        # Bin hist data, find sigma percentiles
        bin_width = 0.15
        bins = np.arange(0, 0.7, bin_width)
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    #-------------
    ### Formatting
    axs.set_ylabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_xlabel('$f_{\mathrm{%s}}(<r_{50})$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    axs.set_ylim(0, 4.5)
    axs.set_yticks(np.arange(0, 4.1, 1))
    axs.set_xlim(0, 0.6)
    axs.set_xticks(np.arange(0, 0.61, 0.1))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### title
    if len(set_gashist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_gashist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        if len(set_gashist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_gas/scatter_trelax_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/scatter_trelax_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()        
# tdyn
def _plot_timescale_gas_scatter_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['co-co'],          # which paths to use
                      set_gashist_min_trelax              = 0.2,                # removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
    print('  Using sample: ', len(ID_plot))
        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #-------------
    ### Plotting scatter
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-------------
    # Colourbar for kappa
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = axs.scatter(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    plt.colorbar(im1, ax=axs, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    if add_plot_gas_morph_median:

        # Bin hist data, find sigma percentiles
        bin_width = 0.15
        bins = np.arange(0, 0.7, bin_width)
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    #-------------
    ### Formatting
    axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_xlabel('$f_{\mathrm{%s}}(<r_{50})$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    axs.set_yscale('log')
    axs.set_ylim(0.5, 35)
    axs.set_yticks([1, 10])
    axs.set_yticklabels(['1', '10'])
    axs.set_xlim(0, 0.6)
    axs.set_xticks(np.arange(0, 0.61, 0.1))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### title
    if len(set_gashist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_gashist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        if len(set_gashist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_gas/scatter_tdyn_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/scatter_tdyn_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()        
# ttorque
def _plot_timescale_gas_scatter_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['co-co'],          # which paths to use
                      set_gashist_min_trelax              = 0.2,                # removing low resolution
                        add_plot_gas_morph_median         = False,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
    print('  Using sample: ', len(ID_plot))
        
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #-------------
    ### Plotting scatter
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    #-------------
    # Colourbar for kappa
    norm = mpl.colors.Normalize(vmin=0.15, vmax=0.65, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
    im1 = axs.scatter(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], df['Relaxation time'], c=df['Relaxation kappa'], s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
    plt.colorbar(im1, ax=axs, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}$', extend='both', pad=0.025)
    
    
    #---------------
    # Bin hist data, find sigma percentiles
    if add_plot_gas_morph_median:

        # Bin hist data, find sigma percentiles
        bin_width = 0.15
        bins = np.arange(0, 0.7, bin_width)
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        print(bin_medians)
    
        # Plot average line for total sample
        axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='k', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='k', ls='-', zorder=101, label='sample')
        
        
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] > 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] > 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='b', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=1, c='b', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}>0.4$')
    
        # Bin hist data, find sigma percentiles
        binned_data_arg = np.digitize(df['%s fraction' %('Gas' if gas_fraction_type == 'gas' else 'SF')][df['Relaxation kappa'] < 0.4], bins=bins)
        bin_medians     = np.stack([np.percentile(df['Relaxation time'][df['Relaxation kappa'] < 0.4][binned_data_arg == i], q=[16, 50, 84]) for i in range(1, len(bins))])
        #print(bin_medians)
    
        # Plot average line for total sample
        #axs.fill_between(bins[:-1]+(bin_width/2), bin_medians[:,0], bin_medians[:,2], facecolor='r', alpha=0.2, zorder=6)
        axs.plot(bins[:-1]+(bin_width/2), bin_medians[:,1], lw=0.7, c='r', ls='-', zorder=101, label=r'$\bar{\kappa}_{\mathrm{co}}^{*}<0.4$', alpha=0.9)
    
    #-------------
    ### Formatting
    axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_xlabel('$f_{\mathrm{%s}}(<r_{50})$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    axs.set_yscale('log')
    axs.set_ylim(0.2, 25)
    axs.set_yticks([0.2, 1, 10])
    axs.set_yticklabels(['0.2', '1', '10'])
    axs.set_xlim(0, 0.6)
    axs.set_xticks(np.arange(0, 0.61, 0.1))
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### title
    if len(set_gashist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_gashist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        if len(set_gashist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_gas/scatter_ttorque_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/scatter_ttorque_%s_%s_%s_subsample%s_%s.%s" %(fig_dir, gas_fraction_type, len(misalignment_tree.keys()), gas_fraction_type, len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()        
         
# Plots histogram of different gas fraction vs relax time
def _plot_timescale_gas_histogram_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['enter below'],          # which paths to use
                      set_gashist_min_trelax              = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_errorbars                = True,
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                        gas_fraction_limits               = [0.1, 0.3],         # [ < lower - upper < ] e.g. [0.2, 0.4] means <0.2, 0.2-0.4, >0.4
                      #--------------------
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.2,     # [ 0.25 / Gyr ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            #gassf_fraction_array.append(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1]))
            
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    if gas_fraction_type == 'gas':
        gas_1_df = df.loc[(df['Gas fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[0]) & (df['Gas fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[1])]
    elif gas_fraction_type == 'gas_sf':
        gas_1_df = df.loc[(df['SF fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['SF fraction'] > gas_fraction_limits[0]) & (df['SF fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['SF fraction'] > gas_fraction_limits[1])]
    
    
    
    #print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tGas %s fractions:' %('' if gas_fraction_type == 'gas' else 'SF'))
    print('\t.  0 - %.2f:    ' %(gas_fraction_limits[0]), len(gas_1_df))
    print('\t%.2f - %.2f:    ' %(gas_fraction_limits[0], gas_fraction_limits[1]), len(gas_2_df))
    print('\t%.2f +    :     ' %(gas_fraction_limits[1]), len(gas_3_df))
    
    print('-------')
    print('Medians:       [ Gyr ]')
    print('   0 - %.2f:    %.2f' %(gas_fraction_limits[0], np.median(gas_1_df['Relaxation time'])))
    print('%.2f - %.2f:    %.2f' %(gas_fraction_limits[0], gas_fraction_limits[1], np.median(gas_2_df['Relaxation time'])))
    print('%.2f +    :     %.2f' %(gas_fraction_limits[1], np.median(gas_3_df['Relaxation time'])))
    
    
    #---------------
    # KS test
    print('-------------')
    res = stats.ks_2samp(gas_1_df['Relaxation time'], gas_2_df['Relaxation time'])
    print('KS-test:     lowest gas range - middle gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_1_df.index) + len(gas_2_df.index))/(len(gas_1_df.index)*len(gas_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(gas_2_df['Relaxation time'], gas_3_df['Relaxation time'])
    print('KS-test:     middle gas range - highest gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_2_df.index) + len(gas_3_df.index))/(len(gas_2_df.index)*len(gas_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(gas_1_df['Relaxation time'], gas_3_df['Relaxation time'])
    print('KS-test:     lowest gas range - highest gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_1_df.index) + len(gas_3_df.index))/(len(gas_1_df.index)*len(gas_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for gas_df, plot_color in zip([gas_1_df, gas_2_df, gas_3_df], ['turquoise', 'teal', 'mediumblue']):
        # Add hist
        axs.hist(gas_df['Relaxation time'], weights=np.ones(len(gas_df['Relaxation time']))/len(gas_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(gas_df['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$0.0<f_{\mathrm{%s}}<%.1f$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('turquoise')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<%.1f$' %(gas_fraction_limits[0], 'gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('teal')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<1.0$' %(gas_fraction_limits[1], 'gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('mediumblue')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if len(set_gashist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_gashist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        if len(set_gashist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
        
        plt.savefig("%s/time_spent_misaligned_gas/%strelax_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/%strelax_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_timescale_gas_histogram_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['enter below'],          # which paths to use
                      set_gashist_min_trelax              = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                        gas_fraction_limits               = [0.1, 0.3],         # [ < lower - upper < ] e.g. [0.2, 0.4] means <0.2, 0.2-0.4, >0.4
                        add_plot_errorbars                = True,
                      #--------------------
                      # General formatting
                      set_bin_limit_tdyn                  = 50,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):


    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            #gassf_fraction_array.append(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1]))
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    if gas_fraction_type == 'gas':
        gas_1_df = df.loc[(df['Gas fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[0]) & (df['Gas fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[1])]
    elif gas_fraction_type == 'gas_sf':
        gas_1_df = df.loc[(df['SF fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['SF fraction'] > gas_fraction_limits[0]) & (df['SF fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['SF fraction'] > gas_fraction_limits[1])]
    
    
    
    #print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tGas %s fractions:' %('' if gas_fraction_type == 'gas' else 'SF'))
    print('\t.  0 - %.2f:    ' %(gas_fraction_limits[0]), len(gas_1_df))
    print('\t%.2f - %.2f:    ' %(gas_fraction_limits[0], gas_fraction_limits[1]), len(gas_2_df))
    print('\t%.2f +    :     ' %(gas_fraction_limits[1]), len(gas_3_df))
    
    print('-------')
    print('Medians:       [ tdyn ]')
    print('   0 - %.2f:    %.2f' %(gas_fraction_limits[0], np.median(gas_1_df['Relaxation time'])))
    print('%.2f - %.2f:    %.2f' %(gas_fraction_limits[0], gas_fraction_limits[1], np.median(gas_2_df['Relaxation time'])))
    print('%.2f +    :     %.2f' %(gas_fraction_limits[1], np.median(gas_3_df['Relaxation time'])))

    
    #---------------
    # KS test
    print('-------------')
    res = stats.ks_2samp(gas_1_df['Relaxation time'], gas_2_df['Relaxation time'])
    print('KS-test:     lowest gas range - middle gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_1_df.index) + len(gas_2_df.index))/(len(gas_1_df.index)*len(gas_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(gas_2_df['Relaxation time'], gas_3_df['Relaxation time'])
    print('KS-test:     middle gas range - highest gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_2_df.index) + len(gas_3_df.index))/(len(gas_2_df.index)*len(gas_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(gas_1_df['Relaxation time'], gas_3_df['Relaxation time'])
    print('KS-test:     lowest gas range - highest gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_1_df.index) + len(gas_3_df.index))/(len(gas_1_df.index)*len(gas_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for gas_df, plot_color in zip([gas_1_df, gas_2_df, gas_3_df], ['turquoise', 'teal', 'mediumblue']):
        # Add hist
        axs.hist(gas_df['Relaxation time'], weights=np.ones(len(gas_df['Relaxation time']))/len(gas_df['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(gas_df['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$0.0<f_{\mathrm{%s}}<%.1f$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('turquoise')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<%.1f$' %(gas_fraction_limits[0], 'gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('teal')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<1.0$' %(gas_fraction_limits[1], 'gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('mediumblue')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if len(set_gashist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_gashist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        if len(set_gashist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_gas/%stdyn_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/%stdyn_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_timescale_gas_histogram_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_gashist_type                    = ['enter below'],          # which paths to use
                      set_gashist_min_trelax              = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      gas_fraction_type                   = 'gas_sf',              # [ 'gas' / 'gas_sf' ]
                        gas_fraction_limits               = [0.1, 0.3],         # [ < lower - upper < ] e.g. [0.2, 0.4] means <0.2, 0.2-0.4, >0.4
                        add_plot_errorbars                = True,
                      #--------------------
                      # General formatting
                      set_bin_limit_ttorque               = 20,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):


    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationkappa_plot = []
    gas_fraction_array   = []
    gassf_fraction_array = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_gashist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_gashist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # take average gas fraction while unstable
            gas_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            gassf_fraction_array.append(np.mean(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])))
            #gassf_fraction_array.append(np.divide(np.array(misalignment_tree['%s' %ID_i]['sfmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1], np.array(misalignment_tree['%s' %ID_i]['gasmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1] + np.array(misalignment_tree['%s' %ID_i]['stelmass_1hmr'])[misalignment_tree['%s' %ID_i]['index_s']+1]))
            
            
            # Collect average kappa during misalignment
            relaxationkappa_plot.append(np.mean(misalignment_tree['%s' %ID_i]['kappa_stars'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation kappa': relaxationkappa_plot, 'Relaxation type': relaxationtype_plot, 'Gas fraction': gas_fraction_array, 'SF fraction': gassf_fraction_array, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    if gas_fraction_type == 'gas':
        gas_1_df = df.loc[(df['Gas fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[0]) & (df['Gas fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['Gas fraction'] > gas_fraction_limits[1])]
    elif gas_fraction_type == 'gas_sf':
        gas_1_df = df.loc[(df['SF fraction'] < gas_fraction_limits[0])]
        gas_2_df = df.loc[(df['SF fraction'] > gas_fraction_limits[0]) & (df['SF fraction'] < gas_fraction_limits[1])]
        gas_3_df = df.loc[(df['SF fraction'] > gas_fraction_limits[1])]
    
    
    
    #print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tGas %s fractions:' %('' if gas_fraction_type == 'gas' else 'SF'))
    print('\t.  0 - %.2f:    ' %(gas_fraction_limits[0]), len(gas_1_df))
    print('\t%.2f - %.2f:    ' %(gas_fraction_limits[0], gas_fraction_limits[1]), len(gas_2_df))
    print('\t%.2f +    :     ' %(gas_fraction_limits[1]), len(gas_3_df))
    
    print('-------')
    print('Medians:       [ ttorque ]')
    print('   0 - %.2f:    %.2f' %(gas_fraction_limits[0], np.median(gas_1_df['Relaxation time'])))
    print('%.2f - %.2f:    %.2f' %(gas_fraction_limits[0], gas_fraction_limits[1], np.median(gas_2_df['Relaxation time'])))
    print('%.2f +    :     %.2f' %(gas_fraction_limits[1], np.median(gas_3_df['Relaxation time'])))
    
    
    #---------------
    # KS test
    print('-------------')
    res = stats.ks_2samp(gas_1_df['Relaxation time'], gas_2_df['Relaxation time'])
    print('KS-test:     lowest gas range - middle gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_1_df.index) + len(gas_2_df.index))/(len(gas_1_df.index)*len(gas_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(gas_2_df['Relaxation time'], gas_3_df['Relaxation time'])
    print('KS-test:     middle gas range - highest gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_2_df.index) + len(gas_3_df.index))/(len(gas_2_df.index)*len(gas_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(gas_1_df['Relaxation time'], gas_3_df['Relaxation time'])
    print('KS-test:     lowest gas range - highest gas range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(gas_1_df.index) + len(gas_3_df.index))/(len(gas_1_df.index)*len(gas_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for gas_df, plot_color in zip([gas_1_df, gas_2_df, gas_3_df], ['turquoise', 'teal', 'mediumblue']):
        # Add hist
        axs.hist(gas_df['Relaxation time'], weights=np.ones(len(gas_df['Relaxation time']))/len(gas_df['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(gas_df['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$0.0<f_{\mathrm{%s}}<%.1f$' %('gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('turquoise')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<%.1f$' %(gas_fraction_limits[0], 'gas' if gas_fraction_type == 'gas' else 'gas,SF', gas_fraction_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('teal')
    
    legend_labels.append(r'$%.1f<f_{\mathrm{%s}}<1.0$' %(gas_fraction_limits[1], 'gas' if gas_fraction_type == 'gas' else 'gas,SF'))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('mediumblue')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if len(set_gashist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_gashist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_gashist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_gashist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        if len(set_gashist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_gas/%sttorque_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_gas/%sttorque_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', gas_fraction_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()      


#-------------------------
# Plots histogram of satellite vs central and relax time. Central == sgn=0, satellite == sgn>1, mixed == sgn switchs
def _plot_timescale_occupation_histogram_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_errorbars                = True,
                      use_occ_morph                       = True,               # Differentiates between ETG centrals, and LTG centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.2,     # [ 0.25 / Gyr ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
                
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    cen_df = df.loc[(df['Occupation'] == 'central')]
    sat_df = df.loc[(df['Occupation'] == 'satellite')]
    mix_df = df.loc[(df['Occupation'] == 'mixed')]
    cen_ETG_df = df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'ETG → ETG')]
    sat_ETG_df = df.loc[(df['Occupation'] == 'satellite') & (df['Relaxation morph'] == 'ETG → ETG')]
    cen_LTG_df = df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'LTG → LTG')]
    sat_LTG_df = df.loc[(df['Occupation'] == 'satellite') & (df['Relaxation morph'] == 'LTG → LTG')]
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tHalo occupations:')
    print('\tcentrals:    %i \tETG-ETG: %i \tLTG-LTG: %i' %(len(cen_df), len(cen_ETG_df), len(cen_LTG_df)))
    print('\tsatellites:  %i \tETG-ETG: %i \tLTG-LTG: %i' %(len(sat_df), len(sat_ETG_df), len(sat_LTG_df)))
    print('\tmixed:       %i' %(len(mix_df)))
    
    print('-------')
    print('Medians:       [ Gyr ]')
    print('centrals:    %.2f \tETG-ETG: %.2f \tLTG-LTG: %.2f' %(np.median(cen_df['Relaxation time']), np.median(cen_ETG_df['Relaxation time']), np.median(cen_LTG_df['Relaxation time'])))
    print('satellites:  %.2f \tETG-ETG: %.2f \tLTG-LTG: %.2f' %(np.median(sat_df['Relaxation time']), np.median(sat_ETG_df['Relaxation time']), np.median(sat_LTG_df['Relaxation time'])))
    

    
    #---------------
    # KS test
    res = stats.ks_2samp(cen_df['Relaxation time'], sat_df['Relaxation time'])
    print('KS-test:     centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_df.index) + len(sat_df.index))/(len(cen_df.index)*len(sat_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(cen_ETG_df['Relaxation time'], sat_ETG_df['Relaxation time'])
    print('KS-test:     ETG-ETG centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_ETG_df.index) + len(sat_ETG_df.index))/(len(cen_ETG_df.index)*len(sat_ETG_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(cen_LTG_df['Relaxation time'], sat_LTG_df['Relaxation time'])
    print('KS-test:     LTG-LTG centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_LTG_df.index) + len(sat_LTG_df.index))/(len(cen_LTG_df.index)*len(sat_LTG_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    if use_occ_morph:
        fig, (axs_ETG, axs_LTG) = plt.subplots(2, 1, figsize=[10/3, 3], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if use_occ_morph:
        for df_i, plot_color in zip([cen_ETG_df, sat_ETG_df], ['maroon', 'coral']):
            # Add hist
            axs_ETG.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
            # Add error bars
            if add_plot_errorbars:
                axs_ETG.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)   
        for df_i, plot_color in zip([cen_LTG_df, sat_LTG_df], ['mediumblue', 'cornflowerblue']):
            # Add hist
            axs_LTG.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
            # Add error bars
            if add_plot_errorbars:
                axs_LTG.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)     
    else:
        for df_i, plot_color in zip([cen_df, sat_df], ['k', 'grey']):
            # Add hist
            axs.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
            # Add error bars
            if add_plot_errorbars:
                axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    #-----------
    ### General formatting
    # Axis labels
    if use_occ_morph:
        axs_ETG.set_yscale('log')
        axs_ETG.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs_ETG.set_xlim(0, set_bin_limit_trelax)
        axs_ETG.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        #axs_ETG.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
        axs_ETG.set_ylim(bottom=0.00025)
        
        axs_LTG.set_yscale('log')
        axs_LTG.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs_LTG.set_xlim(0, set_bin_limit_trelax)
        axs_LTG.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        #axs_LTG.set_ylabel('Percentage of misalignments')
        axs_LTG.set_ylim(bottom=0.00025)
        

        fig.supylabel('Percentage of misalignments', fontsize=9, x=0)
        axs_LTG.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]') 
    else:
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs.set_xlim(0, set_bin_limit_trelax)
        axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
        axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
        axs.set_ylabel('Percentage of\nmisalignments')
        axs.set_ylim(bottom=0.00025)
    
        
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    if use_occ_morph:
        legend_labels.append(' ETG → ETG centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('maroon')
    
        legend_labels.append('ETG → ETG satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('coral')
        
        ncol=1
        axs_ETG.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
        
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('LTG → LTG centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('mediumblue')
    
        legend_labels.append('LTG → LTG satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('cornflowerblue')
        
        ncol=1
        axs_LTG.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)   
    else:
        legend_labels.append('centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')
    
        legend_labels.append('satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('grey')
        
        ncol=1
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if len(set_hist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        if use_occ_morph:
            axs_ETG.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        else:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
        
    #-----------
    ### other
    if not use_occ_morph:
        plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_occ_morph:
            savefig_txt = savefig_txt + 'morphs'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_occupation/%strelax_occupation_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_occupation/%strelax_occupation_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_timescale_occupation_histogram_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_errorbars                = True,
                      use_occ_morph                       = True,               # Differentiates between ETG centrals, and LTG centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_tdyn                  = 50,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
                
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'GalaxyIDs': ID_plot})


    #===================================================================================
    cen_df = df.loc[(df['Occupation'] == 'central')]
    sat_df = df.loc[(df['Occupation'] == 'satellite')]
    mix_df = df.loc[(df['Occupation'] == 'mixed')]
    cen_ETG_df = df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'ETG → ETG')]
    sat_ETG_df = df.loc[(df['Occupation'] == 'satellite') & (df['Relaxation morph'] == 'ETG → ETG')]
    cen_LTG_df = df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'LTG → LTG')]
    sat_LTG_df = df.loc[(df['Occupation'] == 'satellite') & (df['Relaxation morph'] == 'LTG → LTG')]
    
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tHalo occupations:')
    print('\tcentrals:    %i \tETG-ETG: %i \tLTG-LTG: %i' %(len(cen_df), len(cen_ETG_df), len(cen_LTG_df)))
    print('\tsatellites:  %i \tETG-ETG: %i \tLTG-LTG: %i' %(len(sat_df), len(sat_ETG_df), len(sat_LTG_df)))
    print('\tmixed:       %i' %(len(mix_df)))
    
    print('-------')
    print('Medians:       [ tdyn ]')
    print('centrals:    %.2f \tETG-ETG: %.2f \tLTG-LTG: %.2f' %(np.median(cen_df['Relaxation time']), np.median(cen_ETG_df['Relaxation time']), np.median(cen_LTG_df['Relaxation time'])))
    print('satellites:  %.2f \tETG-ETG: %.2f \tLTG-LTG: %.2f' %(np.median(sat_df['Relaxation time']), np.median(sat_ETG_df['Relaxation time']), np.median(sat_LTG_df['Relaxation time'])))

    
    #---------------
    # KS test
    res = stats.ks_2samp(cen_df['Relaxation time'], sat_df['Relaxation time'])
    print('KS-test:     centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_df.index) + len(sat_df.index))/(len(cen_df.index)*len(sat_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(cen_ETG_df['Relaxation time'], sat_ETG_df['Relaxation time'])
    print('KS-test:     ETG-ETG centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_ETG_df.index) + len(sat_ETG_df.index))/(len(cen_ETG_df.index)*len(sat_ETG_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(cen_LTG_df['Relaxation time'], sat_LTG_df['Relaxation time'])
    print('KS-test:     LTG-LTG centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_LTG_df.index) + len(sat_LTG_df.index))/(len(cen_LTG_df.index)*len(sat_LTG_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    if use_occ_morph:
        fig, (axs_ETG, axs_LTG) = plt.subplots(2, 1, figsize=[10/3, 3], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if use_occ_morph:
        for df_i, plot_color in zip([cen_ETG_df, sat_ETG_df], ['maroon', 'coral']):
            # Add hist
            axs_ETG.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
            # Add error bars
            if add_plot_errorbars:
                axs_ETG.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)        
        for df_i, plot_color in zip([cen_LTG_df, sat_LTG_df], ['mediumblue', 'cornflowerblue']):
            # Add hist
            axs_LTG.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
            # Add error bars
            if add_plot_errorbars:
                axs_LTG.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)               
    else:
        for df_i, plot_color in zip([cen_df, sat_df], ['k', 'grey']):
            # Add hist
            axs.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
            # Add error bars
            if add_plot_errorbars:
                axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
        
    #-----------
    ### General formatting
    # Axis labels
    if use_occ_morph:
        axs_ETG.set_yscale('log')
        axs_ETG.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs_ETG.set_xlim(0, set_bin_limit_tdyn)
        axs_ETG.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
        #axs_ETG.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
        axs_ETG.set_ylim(bottom=0.00025)
        
        axs_LTG.set_yscale('log')
        axs_LTG.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs_LTG.set_xlim(0, set_bin_limit_tdyn)
        axs_LTG.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
        #axs_LTG.set_ylabel('Percentage of misalignments')
        axs_LTG.set_ylim(bottom=0.00025)
        

        fig.supylabel('Percentage of misalignments', fontsize=9, x=0)
        axs_LTG.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    else:
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs.set_xlim(0, set_bin_limit_tdyn)
        axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        axs.set_ylabel('Percentage of\nmisalignments')
        axs.set_ylim(bottom=0.00025)
        
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    if use_occ_morph:
        legend_labels.append(' ETG → ETG centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('maroon')
    
        legend_labels.append('ETG → ETG satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('coral')
        
        ncol=1
        axs_ETG.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
        
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('LTG → LTG centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('mediumblue')
    
        legend_labels.append('LTG → LTG satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('cornflowerblue')
        
        ncol=1
        axs_LTG.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)    
    else:
        legend_labels.append('centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')
    
        legend_labels.append('satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('grey')
        
        ncol=1
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if len(set_hist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        if use_occ_morph:
            axs_ETG.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        else:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
            
    #-----------
    ### other
    if not use_occ_morph:
        plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_occ_morph:
            savefig_txt = savefig_txt + 'morphs'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_occupation/%stdyn_occupation_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_occupation/%stdyn_occupation_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_timescale_occupation_histogram_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_errorbars                = True,
                      use_occ_morph                       = True,               # Differentiates between ETG centrals, and LTG centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_ttorque               = 20,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
                
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
                
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'GalaxyIDs': ID_plot})

    
    #===================================================================================
    cen_df = df.loc[(df['Occupation'] == 'central')]
    sat_df = df.loc[(df['Occupation'] == 'satellite')]
    mix_df = df.loc[(df['Occupation'] == 'mixed')]
    cen_ETG_df = df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'ETG → ETG')]
    sat_ETG_df = df.loc[(df['Occupation'] == 'satellite') & (df['Relaxation morph'] == 'ETG → ETG')]
    cen_LTG_df = df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'LTG → LTG')]
    sat_LTG_df = df.loc[(df['Occupation'] == 'satellite') & (df['Relaxation morph'] == 'LTG → LTG')]
    
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    print('\tHalo occupations:')
    print('\tcentrals:    %i \tETG-ETG: %i \tLTG-LTG: %i' %(len(cen_df), len(cen_ETG_df), len(cen_LTG_df)))
    print('\tsatellites:  %i \tETG-ETG: %i \tLTG-LTG: %i' %(len(sat_df), len(sat_ETG_df), len(sat_LTG_df)))
    print('\tmixed:       %i' %(len(mix_df)))
    
    print('-------')
    print('Medians:       [ ttorque ]')
    print('centrals:    %.2f \tETG-ETG: %.2f \tLTG-LTG: %.2f' %(np.median(cen_df['Relaxation time']), np.median(cen_ETG_df['Relaxation time']), np.median(cen_LTG_df['Relaxation time'])))
    print('satellites:  %.2f \tETG-ETG: %.2f \tLTG-LTG: %.2f' %(np.median(sat_df['Relaxation time']), np.median(sat_ETG_df['Relaxation time']), np.median(sat_LTG_df['Relaxation time'])))

    
    #---------------
    # KS test
    res = stats.ks_2samp(cen_df['Relaxation time'], sat_df['Relaxation time'])
    print('KS-test:     centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_df.index) + len(sat_df.index))/(len(cen_df.index)*len(sat_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(cen_ETG_df['Relaxation time'], sat_ETG_df['Relaxation time'])
    print('KS-test:     ETG-ETG centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_ETG_df.index) + len(sat_ETG_df.index))/(len(cen_ETG_df.index)*len(sat_ETG_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(cen_LTG_df['Relaxation time'], sat_LTG_df['Relaxation time'])
    print('KS-test:     LTG-LTG centrals - satellites')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(cen_LTG_df.index) + len(sat_LTG_df.index))/(len(cen_LTG_df.index)*len(sat_LTG_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    if use_occ_morph:
        fig, (axs_ETG, axs_LTG) = plt.subplots(2, 1, figsize=[10/3, 3], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 1.8], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    if use_occ_morph:
        for df_i, plot_color in zip([cen_ETG_df, sat_ETG_df], ['maroon', 'coral']):
            # Add hist
            axs_ETG.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
            # Add error bars
            if add_plot_errorbars:
                axs_ETG.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)    
        for df_i, plot_color in zip([cen_LTG_df, sat_LTG_df], ['mediumblue', 'cornflowerblue']):
            # Add hist
            axs_LTG.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
            # Add error bars
            if add_plot_errorbars:
                axs_LTG.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)    
    else:
        for df_i, plot_color in zip([cen_df, sat_df], ['k', 'grey']):
            # Add hist
            axs.hist(df_i['Relaxation time'], weights=np.ones(len(df_i['Relaxation time']))/len(df_i['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.7, lw=1, edgecolor=plot_color)
            hist_n, _ = np.histogram(df_i['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
            # Add error bars
            if add_plot_errorbars:
                axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    #-----------
    ### General formatting
    # Axis labels
    if use_occ_morph:
        axs_ETG.set_yscale('log')
        axs_ETG.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs_ETG.set_xlim(0, set_bin_limit_ttorque)
        axs_ETG.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
        axs_ETG.set_ylim(bottom=0.00025)
        
        axs_LTG.set_yscale('log')
        axs_LTG.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs_LTG.set_xlim(0, set_bin_limit_ttorque)
        axs_LTG.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
        #axs_LTG.set_ylabel('Percentage of misalignments')
        axs_LTG.set_ylim(bottom=0.00025)
        

        fig.supylabel('Percentage of misalignments', fontsize=9, x=0)
        axs_LTG.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    else:
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
        axs.set_xlim(0, set_bin_limit_ttorque)
        axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
        axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        axs.set_ylabel('Percentage of\nmisalignments')
        axs.set_ylim(bottom=0.00025)
        
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    if use_occ_morph:
        legend_labels.append(' ETG → ETG centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('maroon')
    
        legend_labels.append('ETG → ETG satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('coral')
        
        ncol=1
        axs_ETG.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
        
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('LTG → LTG centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('mediumblue')
    
        legend_labels.append('LTG → LTG satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('cornflowerblue')
        
        ncol=1
        axs_LTG.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)  
    else:
        legend_labels.append('centrals')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')
    
        legend_labels.append('satellites')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('grey')
        
        ncol=1
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if len(set_hist_type) < 4:
        plot_annotate = ''
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        if use_occ_morph:
            axs_ETG.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        else:
            axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
            
    
    #-----------
    ### other
    if not use_occ_morph:
        plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_occ_morph:
            savefig_txt = savefig_txt + 'morphs'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_occupation/%sttorque_occupation_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_occupation/%sttorque_occupation_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots histogram of halo mass (average) vs relax time. cluster: > 1014, group/field: < 1014
def _plot_timescale_environment_histogram_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_errorbars                = True,
                      use_occ_morph                       = True,               # Differentiates between ETG halo masses, and LTG halo masses
                        halomass_limits                   = [12.5, 13.5],         # [ < lower - upper < ] e.g. [12.5, 14] means <10**12.5, 10**12.5-10**14, >10**14
                      use_only_centrals                   = False,                   # Whether to only use centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.2,     # [ 0.25 / Gyr ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    halomass_plot        = []
    ID_plot              = []
    cluster_ID           = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
            
            # Find halo mass
            halomass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['halomass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))      
            
            if np.mean(misalignment_tree['%s' %ID_i]['halomass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]) > 10**14:
                cluster_ID.append(ID_i)
                            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  number of clusters: ', len(cluster_ID))
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'Halo mass': halomass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    if use_only_centrals:
        halo_1_df = df.loc[(df['Halo mass'] < 10**halomass_limits[0]) & (df['Occupation'] == 'central')]
        halo_2_df = df.loc[(df['Halo mass'] > 10**halomass_limits[0]) & (df['Occupation'] == 'central')]
    else:
        halo_1_df = df.loc[(df['Halo mass'] < 10**halomass_limits[0])]
        halo_2_df = df.loc[(df['Halo mass'] > 10**halomass_limits[0])]
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tHalo mass thresholds:')
    print('\t10e11.0 - 10e%s:    %i' %(halomass_limits[0], len(halo_1_df)))
    print('\t10e%s +        :    %i' %(halomass_limits[0], len(halo_2_df)))
    
    print('-------')
    print('Medians:       [ Gyr ]')
    print('10e11.0 - 10e%s:    %.2f' %(halomass_limits[0], np.median(halo_1_df['Relaxation time'])))
    print('10e%s +        :    %.2f' %(halomass_limits[0], np.median(halo_2_df['Relaxation time'])))
    

    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(halo_1_df['Relaxation time'], halo_2_df['Relaxation time'])
    print('KS-test:     lowest halo range - middle halo range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_1_df.index) + len(halo_2_df.index))/(len(halo_1_df.index)*len(halo_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(halo_2_df['Relaxation time'], halo_3_df['Relaxation time'])
    #print('KS-test:     middle halo range - highest halo range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_2_df.index) + len(halo_3_df.index))/(len(halo_2_df.index)*len(halo_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(halo_1_df['Relaxation time'], halo_3_df['Relaxation time'])
    #print('KS-test:     lowest halo range - highest halo range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_1_df.index) + len(halo_3_df.index))/(len(halo_1_df.index)*len(halo_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for halo_df, plot_color in zip([halo_1_df, halo_2_df], ['violet', 'indigo']):
        # Add hist
        axs.hist(halo_df['Relaxation time'], weights=np.ones(len(halo_df['Relaxation time']))/len(halo_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(halo_df['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$M_{\mathrm{200}}$' + '/M$_{\odot}$' + r'$<10^{%s}$' %(halomass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('violet')
    
    #legend_labels.append(r'$10^{%s}<M_{\mathrm{200}}$'%(halomass_limits[0]) + '/M$_{\odot}$' + r'$<10^{%s}$' %(halomass_limits[1]))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('mediumorchid')

    legend_labels.append(r'$M_{\mathrm{200}}$/M$_{\odot}$$>10^{%s}$'%(halomass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('indigo')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_occ_morph:
            savefig_txt = savefig_txt + 'morphs'
        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_environment/%strelax_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_environment/%strelax_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_timescale_environment_histogram_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_errorbars                = True,
                      use_occ_morph                       = True,               # Differentiates between ETG halo masses, and LTG halo masses
                        halomass_limits                   = [12.5, 13.5],         # [ < lower - upper < ] e.g. [12.5, 14] means <10**12.5, 10**12.5-10**14, >10**14
                      use_only_centrals                   = False,                   # Whether to only use centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_tdyn                  = 50,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    halomass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
                
            # Find halo mass
            halomass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['halomass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))           
                            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'Halo mass': halomass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    if use_only_centrals:
        halo_1_df = df.loc[(df['Halo mass'] < 10**halomass_limits[0]) & (df['Occupation'] == 'central')]
        halo_2_df = df.loc[(df['Halo mass'] > 10**halomass_limits[0]) & (df['Occupation'] == 'central')]
    else:
        halo_1_df = df.loc[(df['Halo mass'] < 10**halomass_limits[0])]
        halo_2_df = df.loc[(df['Halo mass'] > 10**halomass_limits[0])]
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tHalo mass thresholds:')
    print('\t10e11.0 - 10e%s:    %i' %(halomass_limits[0], len(halo_1_df)))
    print('\t10e%s +        :    %i' %(halomass_limits[0], len(halo_2_df)))
    
    print('-------')
    print('Medians:       [ tdyn ]')
    print('10e11.0 - 10e%s:    %.2f' %(halomass_limits[0], np.median(halo_1_df['Relaxation time'])))
    print('10e%s +        :    %.2f' %(halomass_limits[0], np.median(halo_2_df['Relaxation time'])))
    

    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(halo_1_df['Relaxation time'], halo_2_df['Relaxation time'])
    print('KS-test:     lowest halo range - middle halo range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_1_df.index) + len(halo_2_df.index))/(len(halo_1_df.index)*len(halo_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(halo_2_df['Relaxation time'], halo_3_df['Relaxation time'])
    #print('KS-test:     middle halo range - highest halo range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_2_df.index) + len(halo_3_df.index))/(len(halo_2_df.index)*len(halo_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(halo_1_df['Relaxation time'], halo_3_df['Relaxation time'])
    #print('KS-test:     lowest halo range - highest halo range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_1_df.index) + len(halo_3_df.index))/(len(halo_1_df.index)*len(halo_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for halo_df, plot_color in zip([halo_1_df, halo_2_df], ['violet', 'indigo']):
        # Add hist
        axs.hist(halo_df['Relaxation time'], weights=np.ones(len(halo_df['Relaxation time']))/len(halo_df['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(halo_df['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$M_{\mathrm{200}}$' + '/M$_{\odot}$' + r'$<10^{%s}$' %(halomass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('violet')
    
    #legend_labels.append(r'$10^{%s}<M_{\mathrm{group}}$'%(halomass_limits[0]) + '/M$_{\odot}$' + r'$<10^{%s}$' %(halomass_limits[1]))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('mediumorchid')

    legend_labels.append(r'$M_{\mathrm{200}}$/M$_{\odot}$$>10^{%s}$'%(halomass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('indigo')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_occ_morph:
            savefig_txt = savefig_txt + 'morphs'
        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_environment/%stdyn_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_environment/%stdyn_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_timescale_environment_histogram_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        add_plot_errorbars                = True,
                      use_occ_morph                       = True,               # Differentiates between ETG halo masses, and LTG halo masses
                        halomass_limits                   = [12.5, 13.5],         # [ < lower - upper < ] e.g. [12.5, 14] means <10**12.5, 10**12.5-10**14, >10**14
                      use_only_centrals                   = False,                   # Whether to only use centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_ttorque               = 20,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    halomass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
                
            # Find halo mass
            halomass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['halomass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))           
                            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'Halo mass': halomass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    if use_only_centrals:
        halo_1_df = df.loc[(df['Halo mass'] < 10**halomass_limits[0]) & (df['Occupation'] == 'central')]
        halo_2_df = df.loc[(df['Halo mass'] > 10**halomass_limits[0]) & (df['Occupation'] == 'central')]
    else:
        halo_1_df = df.loc[(df['Halo mass'] < 10**halomass_limits[0])]
        halo_2_df = df.loc[(df['Halo mass'] > 10**halomass_limits[0])]
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tHalo mass thresholds:')
    print('\t10e11.0 - 10e%s:    %i' %(halomass_limits[0], len(halo_1_df)))
    print('\t10e%s +        :    %i' %(halomass_limits[0], len(halo_2_df)))
    
    print('-------')
    print('Medians:       [ ttorque ]')
    print('10e11.0 - 10e%s:    %.2f' %(halomass_limits[0], np.median(halo_1_df['Relaxation time'])))
    print('10e%s +        :    %.2f' %(halomass_limits[0], np.median(halo_2_df['Relaxation time'])))
    

    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(halo_1_df['Relaxation time'], halo_2_df['Relaxation time'])
    print('KS-test:     lowest halo range - middle halo range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_1_df.index) + len(halo_2_df.index))/(len(halo_1_df.index)*len(halo_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(halo_2_df['Relaxation time'], halo_3_df['Relaxation time'])
    #print('KS-test:     middle halo range - highest halo range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_2_df.index) + len(halo_3_df.index))/(len(halo_2_df.index)*len(halo_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(halo_1_df['Relaxation time'], halo_3_df['Relaxation time'])
    #print('KS-test:     lowest halo range - highest halo range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(halo_1_df.index) + len(halo_3_df.index))/(len(halo_1_df.index)*len(halo_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    for halo_df, plot_color in zip([halo_1_df, halo_2_df], ['violet', 'indigo']):
        # Add hist
        axs.hist(halo_df['Relaxation time'], weights=np.ones(len(halo_df['Relaxation time']))/len(halo_df['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(halo_df['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$M_{\mathrm{200}}$' + '/M$_{\odot}$' + r'$<10^{%s}$' %(halomass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('violet')
    
    #legend_labels.append(r'$10^{%s}<M_{\mathrm{group}}$'%(halomass_limits[0]) + '/M$_{\odot}$' + r'$<10^{%s}$' %(halomass_limits[1]))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append('mediumorchid')

    legend_labels.append(r'$M_{\mathrm{200}}$/M$_{\odot}$$>10^{%s}$'%(halomass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('indigo')
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_occ_morph:
            savefig_txt = savefig_txt + 'morphs'
        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_environment/%sttorque_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_environment/%sttorque_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots histogram (with option of scatter) of accretion rates onto system
def _plot_timescale_accretion_histogram_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot scatter to check
                      plot_scatter_test                   = True,
                      #--------------------
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      #use_morph                           = False,               # Differentiates between ETG halo masses, and LTG halo masses
                      ignore_trelax_inflow                = 0.3,                 # [ 0.3 / 0 ] do not consider inflow within this region as this counts toward initial formation
                        inflow_radius                     = 1.0,                # HMR to use
                        inflow_limits                     = [5, 10],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+
                        use_gas_type                      = 'gas',                  # [ 'gas' / 'gas_sf' ]
                        use_only_centrals                 = True,           # Use only centrals      
                      #--------------------
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.2,     # [ 0.25 / Gyr ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    inflow_rate_plot     = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
            
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_hist_type:
            continue
        
        # Ensure we actually have inflow to consider
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < ignore_trelax_inflow:
            continue
            
        # If we only want centrals, select
        sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
        if use_only_centrals:
            if not np.all(sgn == 0):
                continue
            
        # Ensure we have at least 1 snapshot that we can evaluate
        mask_inflow = (np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1] - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] > ignore_trelax_inflow)    
        if mask_inflow.any() == True:        
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # Mask inflows we are using
            if use_gas_type == 'gas':
                inflow_array = np.array(misalignment_tree['%s' %ID_i]['inflow_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(inflow_array[mask_inflow]))
            if use_gas_type == 'gas_sf':
                inflow_array = np.array(misalignment_tree['%s' %ID_i]['inflow_sf_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(inflow_array[mask_inflow]))
            
            # Find stellar mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
                      
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Inflow rate': inflow_rate_plot, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    inflow_1_df = df.loc[(df['Inflow rate'] < inflow_limits[0])]
    #inflow_2_df = df.loc[(df['Inflow rate'] > inflow_limits[0]) & (df['Inflow rate'] < inflow_limits[1])]
    inflow_3_df = df.loc[(df['Inflow rate'] > inflow_limits[1])]
    #ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    #LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tInflow rates %s [ Msun/yr ]:'%use_gas_type)
    print('\t0  - %i:    %i' %(inflow_limits[0], len(inflow_1_df)))
    #print('\t%i - %i:    %i' %(inflow_limits[0], inflow_limits[1], len(inflow_2_df)))
    print('\t%i +   :    %i' %(inflow_limits[1], len(inflow_3_df)))
    print(' ')
    print('-------')
    print('Medians:       [ Gyr ]')
    print('range 1:    %.2f' %(np.median(inflow_1_df['Relaxation time'])))
    #print('range 2:    %.2f' %(np.median(inflow_2_df['Relaxation time'])))
    print('range 3:    %.2f' %(np.median(inflow_3_df['Relaxation time'])))
    

    # plot scatter to see if anything 
    if plot_scatter_test:
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Colourbar for kappa
        norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
        im1 = axs.scatter(df['Inflow rate'], df['Relaxation time'], c=np.log10(df['Stellar mass']), s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
        axs.set_xlabel('%s inflow [Msun/yr]'%use_gas_type)
        axs.set_ylabel(r'$t_{\mathrm{relax}}$')
        
        plt.show()
    
    
    #---------------
    # KS test
    #print('-------------')
    #res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_2_df['Relaxation time'])
    #print('KS-test:     lowest inflow range - middle inflow range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_2_df.index))/(len(inflow_1_df.index)*len(inflow_2_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(inflow_2_df['Relaxation time'], inflow_3_df['Relaxation time'])
    #print('KS-test:     middle inflow range - highest inflow range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_2_df.index) + len(inflow_3_df.index))/(len(inflow_2_df.index)*len(inflow_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     lowest inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_3_df.index))/(len(inflow_1_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Greens')
    c1 = cmap(0.5)
    #c2 = cmap(0.7)
    c3 = cmap(1.0)
    
    #for inflow_df, plot_color in zip([inflow_1_df, inflow_2_df, inflow_3_df], [c1, c2, c3]):
    for inflow_df, plot_color in zip([inflow_1_df, inflow_3_df], [c1, c3]):
        # Add hist
        axs.hist(inflow_df['Relaxation time'], weights=np.ones(len(inflow_df['Relaxation time']))/len(inflow_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(inflow_df['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$\dot{M}_{\mathrm{gas}}$' + r'$<%s$' %(inflow_limits[0]) + ' M$_{\odot}$ yr$^{-1}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    #legend_labels.append(r'$%s<\dot{M}_{\mathrm{gas}}$'%(inflow_limits[0]) + '/M$_{\odot}$ yr$^{-1}$' + r'$<%s$' %(inflow_limits[1]))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append(c2)

    legend_labels.append(r'$\dot{M}_{\mathrm{gas}}$' + r'$>%s$'%(inflow_limits[1]) + ' M$_{\odot}$ yr$^{-1}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate == None:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        savefig_txt = savefig_txt + '_%i'%inflow_radius
        
        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_inflow/%strelax_inflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_inflow/%strelax_inflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_timescale_accretion_histogram_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot scatter to check
                      plot_scatter_test                   = True,
                      #--------------------
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      #use_morph                           = False,               # Differentiates between ETG halo masses, and LTG halo masses
                      ignore_trelax_inflow                = 0.3,                 # [ 0.3 / 0 ] do not consider inflow within this region as this counts toward initial formation
                        inflow_radius                     = 1.0,                # HMR to use
                        inflow_limits                     = [5, 10],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+
                        use_gas_type                      = 'gas',                  # [ 'gas' / 'gas_sf' ]
                        use_only_centrals                 = True,           # Use only centrals      
                      #--------------------
                      # General formatting
                      set_bin_limit_tdyn                  = 50,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    inflow_rate_plot     = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
            
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_hist_type:
            continue
        
        # Ensure we actually have inflow to consider
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < ignore_trelax_inflow:
            continue
            
        # If we only want centrals, select
        sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
        if use_only_centrals:
            if not np.all(sgn == 0):
                continue
            
        # Ensure we have at least 1 snapshot that we can evaluate
        mask_inflow = (np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1] - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] > ignore_trelax_inflow)
        if mask_inflow.any() == True:        
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # Mask inflows we are using
            if use_gas_type == 'gas':
                inflow_array = np.array(misalignment_tree['%s' %ID_i]['inflow_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(inflow_array[mask_inflow]))
            if use_gas_type == 'gas_sf':
                inflow_array = np.array(misalignment_tree['%s' %ID_i]['inflow_sf_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(inflow_array[mask_inflow]))
            
            # Find stellar mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
                      
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Inflow rate': inflow_rate_plot, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    inflow_1_df = df.loc[(df['Inflow rate'] < inflow_limits[0])]
    #inflow_2_df = df.loc[(df['Inflow rate'] > inflow_limits[0]) & (df['Inflow rate'] < inflow_limits[1])]
    inflow_3_df = df.loc[(df['Inflow rate'] > inflow_limits[1])]
    #ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    #LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tInflow rates %s [ Msun/yr ]:'%use_gas_type)
    print('\t0  - %i:    %i' %(inflow_limits[0], len(inflow_1_df)))
    #print('\t%i - %i:    %i' %(inflow_limits[0], inflow_limits[1], len(inflow_2_df)))
    print('\t%i +   :    %i' %(inflow_limits[1], len(inflow_3_df)))
    print(' ')
    print('-------')
    print('Medians:       [ tdyn ]')
    print('range 1:    %.2f' %(np.median(inflow_1_df['Relaxation time'])))
    #print('range 2:    %.2f' %(np.median(inflow_2_df['Relaxation time'])))
    print('range 3:    %.2f' %(np.median(inflow_3_df['Relaxation time'])))
    

    # plot scatter to see if anything 
    if plot_scatter_test:
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Colourbar for kappa
        norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
        im1 = axs.scatter(df['Inflow rate'], df['Relaxation time'], c=np.log10(df['Stellar mass']), s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
        axs.set_xlabel('%s inflow [Msun/yr]'%use_gas_type)
        axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        
        plt.show()
    
    
    #---------------
    # KS test
    #print('-------------')
    #res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_2_df['Relaxation time'])
    #print('KS-test:     lowest inflow range - middle inflow range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_2_df.index))/(len(inflow_1_df.index)*len(inflow_2_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(inflow_2_df['Relaxation time'], inflow_3_df['Relaxation time'])
    #print('KS-test:     middle inflow range - highest inflow range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_2_df.index) + len(inflow_3_df.index))/(len(inflow_2_df.index)*len(inflow_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     lowest inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_3_df.index))/(len(inflow_1_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Greens')
    c1 = cmap(0.5)
    #c2 = cmap(0.7)
    c3 = cmap(1.0)
    
    #for inflow_df, plot_color in zip([inflow_1_df, inflow_2_df, inflow_3_df], [c1, c2, c3]):
    for inflow_df, plot_color in zip([inflow_1_df, inflow_3_df], [c1, c3]):
        # Add hist
        axs.hist(inflow_df['Relaxation time'], weights=np.ones(len(inflow_df['Relaxation time']))/len(inflow_df['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(inflow_df['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$\dot{M}_{\mathrm{gas}}$' + r'$<%s$' %(inflow_limits[0]) + ' M$_{\odot}$ yr$^{-1}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    #legend_labels.append(r'$%s<\dot{M}_{\mathrm{gas}}$'%(inflow_limits[0]) + '/M$_{\odot}$ yr$^{-1}$' + r'$<%s$' %(inflow_limits[1]))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append(c2)

    legend_labels.append(r'$\dot{M}_{\mathrm{gas}}$' + r'$>%s$'%(inflow_limits[1]) + ' M$_{\odot}$ yr$^{-1}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate == None:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = plot_annotate + ' centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        savefig_txt = savefig_txt + '_%i'%inflow_radius

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_inflow/%stdyn_inflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_inflow/%stdyn_inflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_timescale_accretion_histogram_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot scatter to check
                      plot_scatter_test                   = True,
                      #--------------------
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      #use_morph                           = False,               # Differentiates between ETG halo masses, and LTG halo masses
                      ignore_trelax_inflow                = 0.3,                 # [ 0.3 / 0 ] do not consider inflow within this region as this counts toward initial formation
                        inflow_radius                     = 1.0,                # HMR to use
                        inflow_limits                     = [5, 10],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+
                        use_gas_type                      = 'gas',                  # [ 'gas' / 'gas_sf' ]
                        use_only_centrals                 = True,           # Use only centrals      
                      #--------------------
                      # General formatting
                      set_bin_limit_ttorque               = 20,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    inflow_rate_plot     = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
            
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_hist_type:
            continue
        
        # Ensure we actually have inflow to consider
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < ignore_trelax_inflow:
            continue
            
        # If we only want centrals, select
        sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
        if use_only_centrals:
            if not np.all(sgn == 0):
                continue
            
        # Ensure we have at least 1 snapshot that we can evaluate
        mask_inflow = (np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1] - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] > ignore_trelax_inflow)
        if mask_inflow.any() == True:        
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # Mask inflows we are using
            if use_gas_type == 'gas':
                inflow_array = np.array(misalignment_tree['%s' %ID_i]['inflow_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(inflow_array[mask_inflow]))
            if use_gas_type == 'gas_sf':
                inflow_array = np.array(misalignment_tree['%s' %ID_i]['inflow_sf_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(inflow_array[mask_inflow]))
            
            # Find stellar mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
                      
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Inflow rate': inflow_rate_plot, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    inflow_1_df = df.loc[(df['Inflow rate'] < inflow_limits[0])]
    #inflow_2_df = df.loc[(df['Inflow rate'] > inflow_limits[0]) & (df['Inflow rate'] < inflow_limits[1])]
    inflow_3_df = df.loc[(df['Inflow rate'] > inflow_limits[1])]
    #ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    #LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tInflow rates %s [ Msun/yr ]:'%use_gas_type)
    print('\t0  - %i:    %i' %(inflow_limits[0], len(inflow_1_df)))
    #print('\t%i - %i:    %i' %(inflow_limits[0], inflow_limits[1], len(inflow_2_df)))
    print('\t%i +   :    %i' %(inflow_limits[1], len(inflow_3_df)))
    print(' ')
    print('-------')
    print('Medians:       [ ttorque ]')
    print('range 1:    %.2f' %(np.median(inflow_1_df['Relaxation time'])))
    #print('range 2:    %.2f' %(np.median(inflow_2_df['Relaxation time'])))
    print('range 3:    %.2f' %(np.median(inflow_3_df['Relaxation time'])))
    

    # plot scatter to see if anything 
    if plot_scatter_test:
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Colourbar for kappa
        norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
        im1 = axs.scatter(df['Inflow rate'], df['Relaxation time'], c=np.log10(df['Stellar mass']), s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
        axs.set_xlabel('%s inflow [Msun/yr]'%use_gas_type)
        axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        
        plt.show()
    
    
    #---------------
    # KS test
    #print('-------------')
    #res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_2_df['Relaxation time'])
    #print('KS-test:     lowest inflow range - middle inflow range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_2_df.index))/(len(inflow_1_df.index)*len(inflow_2_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    #res = stats.ks_2samp(inflow_2_df['Relaxation time'], inflow_3_df['Relaxation time'])
    #print('KS-test:     middle inflow range - highest inflow range')
    #print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_2_df.index) + len(inflow_3_df.index))/(len(inflow_2_df.index)*len(inflow_3_df.index))))))
    #print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     lowest inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_3_df.index))/(len(inflow_1_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Greens')
    c1 = cmap(0.5)
    #c2 = cmap(0.7)
    c3 = cmap(1.0)
    
    #for inflow_df, plot_color in zip([inflow_1_df, inflow_2_df, inflow_3_df], [c1, c2, c3]):
    for inflow_df, plot_color in zip([inflow_1_df, inflow_3_df], [c1, c3]):
        # Add hist
        axs.hist(inflow_df['Relaxation time'], weights=np.ones(len(inflow_df['Relaxation time']))/len(inflow_df['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(inflow_df['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$\dot{M}_{\mathrm{gas}}$' + r'$<%s$' %(inflow_limits[0]) + ' M$_{\odot}$ yr$^{-1}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    #legend_labels.append(r'$%s<\dot{M}_{\mathrm{gas}}$'%(inflow_limits[0]) + '/M$_{\odot}$ yr$^{-1}$' + r'$<%s$' %(inflow_limits[1]))
    #legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    #legend_colors.append(c2)

    legend_labels.append(r'$\dot{M}_{\mathrm{gas}}$' + r'$>%s$'%(inflow_limits[1]) + ' M$_{\odot}$ yr$^{-1}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate == None:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        savefig_txt = savefig_txt + '_%i'%inflow_radius

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_inflow/%sttorque_inflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_inflow/%sttorque_inflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots histogram (with option of scatter) of specific accretion rates onto system
def _plot_timescale_specaccretion_histogram_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot scatter to check
                      plot_scatter_test                   = True,
                      #--------------------
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      #use_morph                           = False,               # Differentiates between ETG halo masses, and LTG halo masses
                      ignore_trelax_inflow                = 0.3,                 # [ 0.3 / 0 ] do not consider inflow within this region as this counts toward initial formation
                        inflow_radius                     = 1.0,                # HMR to use
                        inflow_limits                     = [1e-9, 2e-9],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+
                        use_gas_type                      = 'gas',                  # [ 'gas' / 'gas_sf' ]
                        use_only_centrals                 = True,           # Use only centrals   
                      #--------------------
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.2,     # [ 0.25 / Gyr ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    inflow_rate_plot     = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
            
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_hist_type:
            continue
        
        # Ensure we actually have inflow to consider
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < ignore_trelax_inflow:
            continue
        
        # If we only want centrals, select
        sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
        if use_only_centrals:
            if not np.all(sgn == 0):
                continue
                
        # Ensure we have at least 1 snapshot that we can evaluate
        mask_inflow = (np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1] - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] > ignore_trelax_inflow)
        if mask_inflow.any() == True:        
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # Mask inflows we are using
            if use_gas_type == 'gas':
                spec_inflow_array = np.array(misalignment_tree['%s' %ID_i]['s_inflow_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(spec_inflow_array[mask_inflow]))
            if use_gas_type == 'gas_sf':
                spec_inflow_array = np.array(misalignment_tree['%s' %ID_i]['s_inflow_sf_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(spec_inflow_array[mask_inflow]))
                
            
            # Find stellar mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
                        
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
                      
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Inflow rate': inflow_rate_plot, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    inflow_1_df = df.loc[(df['Inflow rate'] < inflow_limits[0])]
    inflow_2_df = df.loc[(df['Inflow rate'] > inflow_limits[0]) & (df['Inflow rate'] < inflow_limits[1])]
    inflow_3_df = df.loc[(df['Inflow rate'] > inflow_limits[1])]
    #ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    #LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tSpecific inflow rates %s [ /yr ]:'%use_gas_type)
    print('\t0  - %s   :    %i' %(inflow_limits[0], len(inflow_1_df)))
    print('\t%s - %s:    %i' %(inflow_limits[0], inflow_limits[1], len(inflow_2_df)))
    print('\t%s +      :    %i' %(inflow_limits[1], len(inflow_3_df)))
    print(' ')
    print('-------')
    print('Medians:       [ Gyr ]')
    print('range 1:    %.2f' %(np.median(inflow_1_df['Relaxation time'])))
    print('range 2:    %.2f' %(np.median(inflow_2_df['Relaxation time'])))
    print('range 3:    %.2f' %(np.median(inflow_3_df['Relaxation time'])))
    

    # plot scatter to see if anything 
    if plot_scatter_test:
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Colourbar for kappa
        norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
        im1 = axs.scatter(df['Inflow rate'], df['Relaxation time'], c=np.log10(df['Stellar mass']), s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
        axs.set_xlabel('Specific %s inflow [/yr]'%use_gas_type)
        axs.set_ylabel(r'$t_{\mathrm{relax}}$')
        
        plt.show()
    
    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_2_df['Relaxation time'])
    print('KS-test:     lowest inflow range - middle inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_2_df.index))/(len(inflow_1_df.index)*len(inflow_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_2_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     middle inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_2_df.index) + len(inflow_3_df.index))/(len(inflow_2_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     lowest inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_3_df.index))/(len(inflow_1_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Reds')
    c1 = cmap(0.4)
    c2 = cmap(0.7)
    c3 = cmap(1.0)
    
    for inflow_df, plot_color in zip([inflow_1_df, inflow_2_df, inflow_3_df], [c1, c2, c3]):
        # Add hist
        axs.hist(inflow_df['Relaxation time'], weights=np.ones(len(inflow_df['Relaxation time']))/len(inflow_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(inflow_df['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$\dot{m}_{\mathrm{%s}}$'%('gas' if use_gas_type == 'gas' else 'SF') + '/yr$^{-1}$' + r'$<10^{-9}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    legend_labels.append(r'$10^{-9}<\dot{m}_{\mathrm{%s}}$'%('gas' if use_gas_type == 'gas' else 'SF')  + '/yr$^{-1}$' + r'$<2\times10^{-9}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c2)

    legend_labels.append(r'$\dot{m}_{\mathrm{%s}}$/yr$^{-1}$$>2\times10^{-9}$'%('gas' if use_gas_type == 'gas' else 'SF') )
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate == None:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        savefig_txt = savefig_txt + '_%i'%inflow_radius

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
        
            
        plt.savefig("%s/time_spent_misaligned_specinflow/%strelax_specinflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_specinflow/%strelax_specinflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format))
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_timescale_specaccretion_histogram_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot scatter to check
                      plot_scatter_test                   = True,
                      #--------------------
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      #use_morph                           = False,               # Differentiates between ETG halo masses, and LTG halo masses
                      ignore_trelax_inflow                = 0.3,                 # [ 0.3 / 0 ] do not consider inflow within this region as this counts toward initial formation
                        inflow_radius                     = 1.0,                # HMR to use
                        inflow_limits                     = [1e-9, 2e-9],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+
                        use_gas_type                      = 'gas',                  # [ 'gas' / 'gas_sf' ]
                        use_only_centrals                 = True,           # Use only centrals   
                      #--------------------
                      # General formatting
                      set_bin_limit_tdyn                  = 50,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    inflow_rate_plot     = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
            
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_hist_type:
            continue
        
        # Ensure we actually have inflow to consider
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < ignore_trelax_inflow:
            continue
        
        # If we only want centrals, select
        sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
        if use_only_centrals:
            if not np.all(sgn == 0):
                continue
                
        # Ensure we have at least 1 snapshot that we can evaluate
        mask_inflow = (np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1] - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] > ignore_trelax_inflow)
        if mask_inflow.any() == True:        
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # Mask inflows we are using
            if use_gas_type == 'gas':
                spec_inflow_array = np.array(misalignment_tree['%s' %ID_i]['s_inflow_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(spec_inflow_array[mask_inflow]))
            if use_gas_type == 'gas_sf':
                spec_inflow_array = np.array(misalignment_tree['%s' %ID_i]['s_inflow_sf_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(spec_inflow_array[mask_inflow]))
            
            # Find stellar mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
                      
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Inflow rate': inflow_rate_plot, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    inflow_1_df = df.loc[(df['Inflow rate'] < inflow_limits[0])]
    inflow_2_df = df.loc[(df['Inflow rate'] > inflow_limits[0]) & (df['Inflow rate'] < inflow_limits[1])]
    inflow_3_df = df.loc[(df['Inflow rate'] > inflow_limits[1])]
    #ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    #LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tSpecific inflow rates %s [ /yr ]:'%use_gas_type)
    print('\t0  - %s   :    %i' %(inflow_limits[0], len(inflow_1_df)))
    print('\t%s - %s:    %i' %(inflow_limits[0], inflow_limits[1], len(inflow_2_df)))
    print('\t%s +      :    %i' %(inflow_limits[1], len(inflow_3_df)))
    print(' ')
    print('-------')
    print('Medians:       [ tdyn ]')
    print('range 1:    %.2f' %(np.median(inflow_1_df['Relaxation time'])))
    print('range 2:    %.2f' %(np.median(inflow_2_df['Relaxation time'])))
    print('range 3:    %.2f' %(np.median(inflow_3_df['Relaxation time'])))
    

    # plot scatter to see if anything 
    if plot_scatter_test:
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Colourbar for kappa
        norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
        im1 = axs.scatter(df['Inflow rate'], df['Relaxation time'], c=np.log10(df['Stellar mass']), s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
        axs.set_xlabel('Specific %s inflow [/yr]'%use_gas_type)
        axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
        
        plt.show()
    
    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_2_df['Relaxation time'])
    print('KS-test:     lowest inflow range - middle inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_2_df.index))/(len(inflow_1_df.index)*len(inflow_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_2_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     middle inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_2_df.index) + len(inflow_3_df.index))/(len(inflow_2_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     lowest inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_3_df.index))/(len(inflow_1_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Reds')
    c1 = cmap(0.4)
    c2 = cmap(0.7)
    c3 = cmap(1.0)
    
    for inflow_df, plot_color in zip([inflow_1_df, inflow_2_df, inflow_3_df], [c1, c2, c3]):
        # Add hist
        axs.hist(inflow_df['Relaxation time'], weights=np.ones(len(inflow_df['Relaxation time']))/len(inflow_df['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(inflow_df['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$\dot{m}_{\mathrm{%s}}$'%('gas' if use_gas_type == 'gas' else 'SF') + '/yr$^{-1}$' + r'$<10^{-9}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    legend_labels.append(r'$10^{-9}<\dot{m}_{\mathrm{%s}}$'%('gas' if use_gas_type == 'gas' else 'SF')  + '/yr$^{-1}$' + r'$<2\times10^{-9}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c2)

    legend_labels.append(r'$\dot{m}_{\mathrm{%s}}$/yr$^{-1}$$>2\times10^{-9}$'%('gas' if use_gas_type == 'gas' else 'SF') )
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate == None:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        savefig_txt = savefig_txt + '_%i'%inflow_radius

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'

        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_specinflow/%stdyn_specinflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_specinflow/%stdyn_specinflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_timescale_specaccretion_histogram_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot scatter to check
                      plot_scatter_test                   = True,
                      #--------------------
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                      #use_morph                           = False,               # Differentiates between ETG halo masses, and LTG halo masses
                      ignore_trelax_inflow                = 0.3,                 # [ 0.3 / 0 ] do not consider inflow within this region as this counts toward initial formation
                        inflow_radius                     = 1.0,                # HMR to use
                        inflow_limits                     = [1e-9, 2e-9],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+
                        use_gas_type                      = 'gas',                  # [ 'gas' / 'gas_sf' ]
                        use_only_centrals                 = True,           # Use only centrals   
                      #--------------------
                      # General formatting
                      set_bin_limit_ttorque               = 20,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    inflow_rate_plot     = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
            
        if misalignment_tree['%s' %ID_i]['relaxation_type'] not in set_hist_type:
            continue
        
        # Ensure we actually have inflow to consider
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < ignore_trelax_inflow:
            continue
        
        # If we only want centrals, select
        sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
        if use_only_centrals:
            if not np.all(sgn == 0):
                continue
                
        # Ensure we have at least 1 snapshot that we can evaluate
        mask_inflow = (np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1] - np.array(misalignment_tree['%s' %ID_i]['Lookbacktime'])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']] > ignore_trelax_inflow)
        if mask_inflow.any() == True:        
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # Mask inflows we are using
            if use_gas_type == 'gas':
                spec_inflow_array = np.array(misalignment_tree['%s' %ID_i]['s_inflow_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(spec_inflow_array[mask_inflow]))
            if use_gas_type == 'gas_sf':
                spec_inflow_array = np.array(misalignment_tree['%s' %ID_i]['s_inflow_sf_rate_%ihmr' %int(inflow_radius)])[misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]
                inflow_rate_plot.append(np.mean(spec_inflow_array[mask_inflow]))
            
            # Find stellar mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
                      
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Inflow rate': inflow_rate_plot, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    inflow_1_df = df.loc[(df['Inflow rate'] < inflow_limits[0])]
    inflow_2_df = df.loc[(df['Inflow rate'] > inflow_limits[0]) & (df['Inflow rate'] < inflow_limits[1])]
    inflow_3_df = df.loc[(df['Inflow rate'] > inflow_limits[1])]
    #ETG_df = df.loc[(df['Relaxation morph'] == 'ETG → ETG')]
    #LTG_df = df.loc[(df['Relaxation morph'] == 'LTG → LTG')]
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tSpecific inflow rates %s [ /yr ]:'%use_gas_type)
    print('\t0  - %s   :    %i' %(inflow_limits[0], len(inflow_1_df)))
    print('\t%s - %s:    %i' %(inflow_limits[0], inflow_limits[1], len(inflow_2_df)))
    print('\t%s +      :    %i' %(inflow_limits[1], len(inflow_3_df)))
    print(' ')
    print('-------')
    print('Medians:       [ ttorque ]')
    print('range 1:    %.2f' %(np.median(inflow_1_df['Relaxation time'])))
    print('range 2:    %.2f' %(np.median(inflow_2_df['Relaxation time'])))
    print('range 3:    %.2f' %(np.median(inflow_3_df['Relaxation time'])))
    

    # plot scatter to see if anything 
    if plot_scatter_test:
        ### Plotting
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Colourbar for kappa
        norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
        im1 = axs.scatter(df['Inflow rate'], df['Relaxation time'], c=np.log10(df['Stellar mass']), s=1.5, norm=norm, cmap='Spectral', zorder=99, edgecolors='k', linewidths=0.1, alpha=1)
        plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
        axs.set_xlabel('Specific %s inflow [/yr]'%use_gas_type)
        axs.set_ylabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
        
        plt.show()
    
    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_2_df['Relaxation time'])
    print('KS-test:     lowest inflow range - middle inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_2_df.index))/(len(inflow_1_df.index)*len(inflow_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_2_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     middle inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_2_df.index) + len(inflow_3_df.index))/(len(inflow_2_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(inflow_1_df['Relaxation time'], inflow_3_df['Relaxation time'])
    print('KS-test:     lowest inflow range - highest inflow range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(inflow_1_df.index) + len(inflow_3_df.index))/(len(inflow_1_df.index)*len(inflow_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Reds')
    c1 = cmap(0.4)
    c2 = cmap(0.7)
    c3 = cmap(1.0)
    
    for inflow_df, plot_color in zip([inflow_1_df, inflow_2_df, inflow_3_df], [c1, c2, c3]):
        # Add hist
        axs.hist(inflow_df['Relaxation time'], weights=np.ones(len(inflow_df['Relaxation time']))/len(inflow_df['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(inflow_df['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$\dot{m}_{\mathrm{%s}}$'%('gas' if use_gas_type == 'gas' else 'SF') + '/yr$^{-1}$' + r'$<10^{-9}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    legend_labels.append(r'$10^{-9}<\dot{m}_{\mathrm{%s}}$'%('gas' if use_gas_type == 'gas' else 'SF')  + '/yr$^{-1}$' + r'$<2\times10^{-9}$')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c2)

    legend_labels.append(r'$\dot{m}_{\mathrm{%s}}$/yr$^{-1}$$>2\times10^{-9}$'%('gas' if use_gas_type == 'gas' else 'SF') )
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if plot_annotate == None:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        savefig_txt = savefig_txt + '_%i'%inflow_radius

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'

        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_specinflow/%sttorque_specinflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_specinflow/%sttorque_specinflow_%s_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', use_gas_type, len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format))
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots histogram of stel mass (average) vs relax time.
def _plot_timescale_stelmass_histogram_trelax(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        stelmass_limits                   = [10, 11],         # [ < lower - upper < ] e.g. [12.5, 14] means <10**12.5, 10**12.5-10**14, >10**14
                      use_only_centrals                   = False,                   # Whether to only use centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_trelax                = 6,        # [ None / Gyr ]
                      set_bin_width_trelax                = 0.2,     # [ 0.25 / Gyr ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
            
            # Find halo mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))           
            #stelmass_plot.append(np.array(misalignment_tree['%s' %ID_i]['stelmass'])[misalignment_tree['%s' %ID_i]['index_s']+1])       
            
                           
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    if use_only_centrals:
        stelmass_1_df = df.loc[(df['Stellar mass'] < 10**stelmass_limits[0]) & (df['Occupation'] == 'central')]
        stelmass_2_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[0]) & (df['Stellar mass'] < 10**stelmass_limits[1]) & (df['Occupation'] == 'central')]
        stelmass_3_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[1]) & (df['Occupation'] == 'central')]
    else:
        stelmass_1_df = df.loc[(df['Stellar mass'] < 10**stelmass_limits[0])]
        stelmass_2_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[0]) & (df['Stellar mass'] < 10**stelmass_limits[1])]
        stelmass_3_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[1])]
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tStellar mass thresholds:')
    print('\t10e9.0 - 10e%s:    %i' %(stelmass_limits[0], len(stelmass_1_df)))
    print('\t10e%.1f - 10e%s:    %i' %(stelmass_limits[0], stelmass_limits[1], len(stelmass_2_df)))
    print('\t10e%s +        :    %i' %(stelmass_limits[1], len(stelmass_3_df)))
    
    print('-------')
    print('Medians:       [ Gyr ]')
    print('10e11.0 - 10e%s:    %.2f' %(stelmass_limits[0], np.median(stelmass_1_df['Relaxation time'])))
    print('10e%.1f - 10e%s:    %.2f' %(stelmass_limits[0], stelmass_limits[1], np.median(stelmass_2_df['Relaxation time'])))
    print('10e%s +        :    %.2f' %(stelmass_limits[1], np.median(stelmass_3_df['Relaxation time'])))
    

    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(stelmass_1_df['Relaxation time'], stelmass_2_df['Relaxation time'])
    print('KS-test:     lowest stelmass range - middle stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_1_df.index) + len(stelmass_2_df.index))/(len(stelmass_1_df.index)*len(stelmass_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(stelmass_2_df['Relaxation time'], stelmass_3_df['Relaxation time'])
    print('KS-test:     middle stelmass range - highest stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_2_df.index) + len(stelmass_3_df.index))/(len(stelmass_2_df.index)*len(stelmass_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(stelmass_1_df['Relaxation time'], stelmass_3_df['Relaxation time'])
    print('KS-test:     lowest stelmass range - highest stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_1_df.index) + len(stelmass_3_df.index))/(len(stelmass_1_df.index)*len(stelmass_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_trelax == None:
        set_bin_limit_trelax = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Oranges')
    c1 = cmap(0.4)
    c2 = cmap(0.7)
    c3 = cmap(1.0)
    for stelmass_df, plot_color in zip([stelmass_1_df, stelmass_2_df, stelmass_3_df], [c1, c2, c3]):
        # Add hist
        axs.hist(stelmass_df['Relaxation time'], weights=np.ones(len(stelmass_df['Relaxation time']))/len(stelmass_df['Relaxation time']), bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(stelmass_df['Relaxation time'], bins=np.arange(0, set_bin_limit_trelax+set_bin_width_trelax, set_bin_width_trelax), range=(0, set_bin_limit_trelax))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_trelax/2, set_bin_limit_trelax, set_bin_width_trelax), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_trelax)
    axs.set_xticks(np.arange(0, set_bin_limit_trelax+0.1, step=1))
    axs.set_xlabel('$t_{\mathrm{relax}}$ [Gyr]')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$M_{\mathrm{*}}$' + '/M$_{\odot}$' + r'$<10^{%s}$' %(stelmass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    legend_labels.append(r'$10^{%s}<M_{\mathrm{*}}$'%(stelmass_limits[0]) + '/M$_{\odot}$' + r'$<10^{%s}$' %(stelmass_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c2)

    legend_labels.append(r'$M_{\mathrm{*}}$/M$_{\odot}$$>10^{%s}$'%(stelmass_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if (len(set_hist_type) < 4) or use_only_centrals:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_stelmass/%strelax_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_stelmass/%strelax_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# tdyn
def _plot_timescale_stelmass_histogram_tdyn(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        stelmass_limits                   = [10, 11],         # [ < lower - upper < ] e.g. [12.5, 14] means <10**12.5, 10**12.5-10**14, >10**14
                      use_only_centrals                   = False,                   # Whether to only use centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_tdyn                  = 50,       # [ None / multiples ]
                      set_bin_width_tdyn                  = 2,        # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_tdyn'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
            
            # Find halo mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))           
            #stelmass_plot.append(np.array(misalignment_tree['%s' %ID_i]['stelmass'])[misalignment_tree['%s' %ID_i]['index_s']+1])           
                            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    if use_only_centrals:
        stelmass_1_df = df.loc[(df['Stellar mass'] < 10**stelmass_limits[0]) & (df['Occupation'] == 'central')]
        stelmass_2_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[0]) & (df['Stellar mass'] < 10**stelmass_limits[1]) & (df['Occupation'] == 'central')]
        stelmass_3_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[1]) & (df['Occupation'] == 'central')]
    else:
        stelmass_1_df = df.loc[(df['Stellar mass'] < 10**stelmass_limits[0])]
        stelmass_2_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[0]) & (df['Stellar mass'] < 10**stelmass_limits[1])]
        stelmass_3_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[1])]
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tStellar mass thresholds:')
    print('\t10e9.0 - 10e%s:    %i' %(stelmass_limits[0], len(stelmass_1_df)))
    print('\t10e%.1f - 10e%s:    %i' %(stelmass_limits[0], stelmass_limits[1], len(stelmass_2_df)))
    print('\t10e%s +        :    %i' %(stelmass_limits[1], len(stelmass_3_df)))
    
    print('-------')
    print('Medians:       [ tdyn ]')
    print('10e11.0 - 10e%s:    %.2f' %(stelmass_limits[0], np.median(stelmass_1_df['Relaxation time'])))
    print('10e%.1f - 10e%s:    %.2f' %(stelmass_limits[0], stelmass_limits[1], np.median(stelmass_2_df['Relaxation time'])))
    print('10e%s +        :    %.2f' %(stelmass_limits[1], np.median(stelmass_3_df['Relaxation time'])))
    

    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(stelmass_1_df['Relaxation time'], stelmass_2_df['Relaxation time'])
    print('KS-test:     lowest stelmass range - middle stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_1_df.index) + len(stelmass_2_df.index))/(len(stelmass_1_df.index)*len(stelmass_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(stelmass_2_df['Relaxation time'], stelmass_3_df['Relaxation time'])
    print('KS-test:     middle stelmass range - highest stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_2_df.index) + len(stelmass_3_df.index))/(len(stelmass_2_df.index)*len(stelmass_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(stelmass_1_df['Relaxation time'], stelmass_3_df['Relaxation time'])
    print('KS-test:     lowest stelmass range - highest stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_1_df.index) + len(stelmass_3_df.index))/(len(stelmass_1_df.index)*len(stelmass_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_tdyn == None:
        set_bin_limit_tdyn = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Oranges')
    c1 = cmap(0.4)
    c2 = cmap(0.7)
    c3 = cmap(1.0)
    for stelmass_df, plot_color in zip([stelmass_1_df, stelmass_2_df, stelmass_3_df], [c1, c2, c3]):
        # Add hist
        axs.hist(stelmass_df['Relaxation time'], weights=np.ones(len(stelmass_df['Relaxation time']))/len(stelmass_df['Relaxation time']), bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(stelmass_df['Relaxation time'], bins=np.arange(0, set_bin_limit_tdyn+set_bin_width_tdyn, set_bin_width_tdyn), range=(0, set_bin_limit_tdyn))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_tdyn/2, set_bin_limit_tdyn, set_bin_width_tdyn), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_tdyn)
    axs.set_xticks(np.arange(0, set_bin_limit_tdyn+0.1, step=4))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{dyn}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$M_{\mathrm{*}}$' + '/M$_{\odot}$' + r'$<10^{%s}$' %(stelmass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    legend_labels.append(r'$10^{%s}<M_{\mathrm{*}}$'%(stelmass_limits[0]) + '/M$_{\odot}$' + r'$<10^{%s}$' %(stelmass_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c2)

    legend_labels.append(r'$M_{\mathrm{*}}$/M$_{\odot}$$>10^{%s}$'%(stelmass_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if (len(set_hist_type) < 4) or use_only_centrals:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_stelmass/%stdyn_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_stelmass/%stdyn_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()
# ttorque
def _plot_timescale_stelmass_histogram_ttorque(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0.2,                # [ 0.25 / 0 ] removing low resolution
                        stelmass_limits                   = [10, 11],         # [ < lower - upper < ] e.g. [12.5, 14] means <10**12.5, 10**12.5-10**14, >10**14
                      use_only_centrals                   = False,                   # Whether to only use centrals
                      #--------------------
                      # General formatting
                      set_bin_limit_ttorque               = 20,       # [ None / multiples ]
                      set_bin_width_ttorque               = 1,      # [ multiples ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    subgroupnum_class    = []
    stelmass_plot        = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_ttorque'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            # find subgroupnum, and classify as following:  central: all==0, satellite: all>0, mixed: ones that change
            sgn = np.array(misalignment_tree['%s' %ID_i]['SubGroupNum'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']])
            if np.all(sgn == 0):
                subgroupnum_class.append('central')
            elif np.all(sgn > 0):
                subgroupnum_class.append('satellite')
            else:
                subgroupnum_class.append('mixed')
            
            # Find halo mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))           
            #stelmass_plot.append(np.array(misalignment_tree['%s' %ID_i]['stelmass'])[misalignment_tree['%s' %ID_i]['index_s']+1])   
                            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Occupation': subgroupnum_class, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    if use_only_centrals:
        stelmass_1_df = df.loc[(df['Stellar mass'] < 10**stelmass_limits[0]) & (df['Occupation'] == 'central')]
        stelmass_2_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[0]) & (df['Stellar mass'] < 10**stelmass_limits[1]) & (df['Occupation'] == 'central')]
        stelmass_3_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[1]) & (df['Occupation'] == 'central')]
    else:
        stelmass_1_df = df.loc[(df['Stellar mass'] < 10**stelmass_limits[0])]
        stelmass_2_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[0]) & (df['Stellar mass'] < 10**stelmass_limits[1])]
        stelmass_3_df = df.loc[(df['Stellar mass'] > 10**stelmass_limits[1])]
    
    
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    if use_only_centrals:
        print('  Using only - CENTRALS -:')
    print('\tStellar mass thresholds:')
    print('\t10e9.0 - 10e%s:    %i' %(stelmass_limits[0], len(stelmass_1_df)))
    print('\t10e%.1f - 10e%s:    %i' %(stelmass_limits[0], stelmass_limits[1], len(stelmass_2_df)))
    print('\t10e%s +        :    %i' %(stelmass_limits[1], len(stelmass_3_df)))
    
    print('-------')
    print('Medians:       [ ttorque ]')
    print('10e11.0 - 10e%s:    %.2f' %(stelmass_limits[0], np.median(stelmass_1_df['Relaxation time'])))
    print('10e%.1f - 10e%s:    %.2f' %(stelmass_limits[0], stelmass_limits[1], np.median(stelmass_2_df['Relaxation time'])))
    print('10e%s +        :    %.2f' %(stelmass_limits[1], np.median(stelmass_3_df['Relaxation time'])))
    

    
    #---------------
    # KS test
    #print('-------------')
    res = stats.ks_2samp(stelmass_1_df['Relaxation time'], stelmass_2_df['Relaxation time'])
    print('KS-test:     lowest stelmass range - middle stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_1_df.index) + len(stelmass_2_df.index))/(len(stelmass_1_df.index)*len(stelmass_2_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(stelmass_2_df['Relaxation time'], stelmass_3_df['Relaxation time'])
    print('KS-test:     middle stelmass range - highest stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_2_df.index) + len(stelmass_3_df.index))/(len(stelmass_2_df.index)*len(stelmass_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    res = stats.ks_2samp(stelmass_1_df['Relaxation time'], stelmass_3_df['Relaxation time'])
    print('KS-test:     lowest stelmass range - highest stelmass range')
    print('   D:       %.2f       D$_{crit}$ (0.05):       %.2f' %(res.statistic, (1.358*np.sqrt((len(stelmass_1_df.index) + len(stelmass_3_df.index))/(len(stelmass_1_df.index)*len(stelmass_3_df.index))))))
    print('   p-value: %s' %res.pvalue)
    
    
    #-------------
    ### Plotting
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
    if set_bin_limit_ttorque == None:
        set_bin_limit_ttorque = math.ceil(max(relaxationtime_plot))
    if set_plot_histogram_log:
        set_plot_relaxation_type = False 
    
    #-------------
    ### Plot histogram
    cmap = cm.get_cmap('Oranges')
    c1 = cmap(0.4)
    c2 = cmap(0.7)
    c3 = cmap(1.0)
    for stelmass_df, plot_color in zip([stelmass_1_df, stelmass_2_df, stelmass_3_df], [c1, c2, c3]):
        # Add hist
        axs.hist(stelmass_df['Relaxation time'], weights=np.ones(len(stelmass_df['Relaxation time']))/len(stelmass_df['Relaxation time']), bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), histtype='step', facecolor='none', alpha=0.9, lw=1, edgecolor=plot_color)
        hist_n, _ = np.histogram(stelmass_df['Relaxation time'], bins=np.arange(0, set_bin_limit_ttorque+set_bin_width_ttorque, set_bin_width_ttorque), range=(0, set_bin_limit_ttorque))
        
        # Add error bars
        if add_plot_errorbars:
            axs.errorbar(np.arange(set_bin_width_ttorque/2, set_bin_limit_ttorque, set_bin_width_ttorque), hist_n/np.sum(hist_n), xerr=None, yerr=np.sqrt(hist_n)/np.sum(hist_n), ecolor=plot_color, ls='none', capsize=1, elinewidth=0.7, markeredgewidth=0.7, alpha=0.3)
        
    
    
    #-----------
    ### General formatting
    # Axis labels
    axs.set_yscale('log')
    axs.yaxis.set_major_formatter(PercentFormatter(1, symbol='', decimals=1))
    axs.set_xlim(0, set_bin_limit_ttorque)
    axs.set_xticks(np.arange(0, set_bin_limit_ttorque+0.1, step=2))
    axs.set_xlabel(r'$t_{\mathrm{relax}}/\bar{t}_{\rm{torque}}$')
    axs.set_ylabel('Percentage of\nmisalignments')
    axs.set_ylim(bottom=0.00025)
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    
    legend_labels.append(r'$M_{\mathrm{*}}$' + '/M$_{\odot}$' + r'$<10^{%s}$' %(stelmass_limits[0]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c1)
    
    legend_labels.append(r'$10^{%s}<M_{\mathrm{*}}$'%(stelmass_limits[0]) + '/M$_{\odot}$' + r'$<10^{%s}$' %(stelmass_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c2)

    legend_labels.append(r'$M_{\mathrm{*}}$/M$_{\odot}$$>10^{%s}$'%(stelmass_limits[1]))
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append(c3)
    
    ncol=1
    axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=ncol)
    
    
    #-----------
    ### title
    if (len(set_hist_type) < 4) or use_only_centrals:
        plot_annotate = ''
    if use_only_centrals:
        plot_annotate = ''
        plot_annotate = plot_annotate + 'centrals'
    if len(set_hist_type) < 4:
        if 'co-co' in set_hist_type:
            plot_annotate = plot_annotate + 'co → co'
        if 'counter-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → counter'
        if 'co-counter' in set_hist_type:
            plot_annotate = plot_annotate + ', co → counter'
        if 'counter-co' in set_hist_type:
            plot_annotate = plot_annotate + ', counter → co'
    if plot_annotate:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    #-----------
    ### other
    plt.tight_layout()
    
    
    #-----------
    ### Savefig  
                     
    if savefig:
        savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

        if use_only_centrals:
            savefig_txt = savefig_txt + '_centrals'
        if len(set_hist_type) == 4:
            savefig_txt = savefig_txt + '_allpaths'
            
        plt.savefig("%s/time_spent_misaligned_stelmass/%sttorque_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/time_spent_misaligned_stelmass/%sttorque_halomass_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
    if showfig:
        plt.show()
    plt.close()


#-------------------------
# Plots scatter and histogram of change in theoretical ttorque and ellip during misalignmnet
def _plot_morphology_change_ttorque_ellip(misalignment_tree, misalignment_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options
                      set_hist_type                       = ['enter below'],          # which paths to use
                      set_hist_min_trelax                 = 0,                # [ 0.25 / 0 ] removing low resolution
                      plot_scatter                        = True,
                      plot_histogram                      = True,
                        plot_ellip                        = True,       # change in ellip vs trelax
                        plot_ttorque                      = True,       # change in ttorque theoretical vs trelax
                        plot_ellip_ttorque                = True,       # change in ellip vs change in ttorque theoretical
                      #--------------------
                      # General formatting
                      set_bin_limit                = 5,        # [ None / fraction ]
                      set_bin_width                = 0.25,     # [ 0.25 / fraction ]
                      set_plot_histogram_log              = False,    # set yaxis as log
                        add_plot_errorbars                = True,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-------------------------
    # Average timescales from input (for use in metadata)
    mean_timescale   = np.mean(np.array(summary_dict['trelax']['array']))
    median_timescale = np.median(np.array(summary_dict['trelax']['array']))
    std_timescale    = np.std(np.array(summary_dict['trelax']['array']))
    # Average tdyn
    mean_tdyn   = np.mean(np.array(summary_dict['tdyn']['array']))
    median_tdyn = np.median(np.array(summary_dict['tdyn']['array']))
    std_tdyn    = np.std(np.array(summary_dict['tdyn']['array']))
    # Average ttorque
    mean_ttorque   = np.mean(np.array(summary_dict['ttorque']['array']))
    median_ttorque = np.median(np.array(summary_dict['ttorque']['array']))
    std_ttorque    = np.std(np.array(summary_dict['ttorque']['array']))
    
    #-------------------------
    relaxation_type    = misalignment_input['relaxation_type']
    relaxation_morph   = misalignment_input['relaxation_morph']
    misalignment_morph = misalignment_input['misalignment_morph']
    morph_limits       = misalignment_input['morph_limits']
    peak_misangle      = misalignment_input['peak_misangle']
    min_trelax         = misalignment_input['min_trelax']        
    max_trelax         = misalignment_input['max_trelax'] 
    use_angle          = misalignment_input['use_angle']
    misangle_threshold = misalignment_input['misangle_threshold']
    time_extra         = misalignment_input['time_extra']
    #------------------------- 
    
    
    #==========================================================================
    # Gather data
    relaxationtime_plot  = []
    relaxationtype_plot  = []
    relaxationmorph_plot = []
    stelmass_plot        = []
    delta_torque_plot    = []
    delta_ellip_plot     = []
    ID_plot              = []
    for ID_i in misalignment_tree.keys():
        
        if misalignment_tree['%s' %ID_i]['relaxation_time'] < set_hist_min_trelax:
            continue
    
        if misalignment_tree['%s' %ID_i]['relaxation_type'] in set_hist_type:
            
            ID_plot.append(ID_i)
            relaxationtime_plot.append(misalignment_tree['%s' %ID_i]['relaxation_time'])
            
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-co':
                relaxationtype_plot.append('C0')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'co-counter':
                relaxationtype_plot.append('C1')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-counter':
                relaxationtype_plot.append('C2')
            if misalignment_tree['%s' %ID_i]['relaxation_type'] == 'counter-co':
                relaxationtype_plot.append('C3')
            
            
            # Extract average theretical ttorque and ellip pre-misalignment, and average ttorque and ellip during misalignment
            ellip_pre = np.mean(misalignment_tree['%s' %ID_i]['ellip'][0:misalignment_tree['%s' %ID_i]['index_s']+1]) 
            ellip_mid = np.mean(misalignment_tree['%s' %ID_i]['ellip'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]) 
            delta_ellip_plot.append(ellip_mid/ellip_pre)        # decrease = more spherical
            
            theottorque_pre = np.mean(misalignment_tree['%s' %ID_i]['ttorque'][0:misalignment_tree['%s' %ID_i]['index_s']+1]) 
            theottorque_mid = np.mean(misalignment_tree['%s' %ID_i]['ttorque'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]) 
            delta_torque_plot.append(theottorque_mid/theottorque_pre)       # increase = more spherical possibly
            
            """
            if int(ID_i) in range(453139689, 453139689+100):
                print(ellip_pre)
                print(ellip_mid)
                print(ellip_mid/ellip_pre)
                print(' ')
                print(theottorque_pre)
                print(theottorque_mid)
                print(theottorque_mid/theottorque_pre)
            """
            
            # Find stellar mass
            stelmass_plot.append(np.mean(misalignment_tree['%s' %ID_i]['stelmass'][misalignment_tree['%s' %ID_i]['index_s']+1:misalignment_tree['%s' %ID_i]['index_r']]))
            
                            
            # Collect morphology
            if misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-ETG':
                relaxationmorph_plot.append('ETG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-LTG':
                relaxationmorph_plot.append('LTG → LTG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'LTG-ETG':
                relaxationmorph_plot.append('LTG → ETG')
            elif misalignment_tree['%s' %ID_i]['relaxation_morph'] == 'ETG-LTG':
                relaxationmorph_plot.append('ETG → LTG')
            else:
                relaxationmorph_plot.append('other')
            
    print('  Using sample: ', len(ID_plot))
    
    # Collect data into dataframe
    df = pd.DataFrame(data={'Relaxation time': relaxationtime_plot, 'Relaxation morph': relaxationmorph_plot, 'Relaxation type': relaxationtype_plot, 'Delta ttorque': delta_torque_plot, 'Delta ellip': delta_ellip_plot, 'Stellar mass': stelmass_plot, 'GalaxyIDs': ID_plot})
    
    #===================================================================================
    print('-------------------------------------------------------------')
    print('Number of relaxations: ', len(ID_plot))
    
    if plot_scatter:
        if plot_ellip:
            ## Print statements
            df_temp1 = df.loc[(df['Relaxation time'] > 1)]
            df_temp2 = df.loc[(df['Relaxation time'] > 1) & (df['Delta ellip'] < 0.80)]
            print('  Number of relaxations >1 Gyr:                                      ', len(df_temp1))
            print('  Number of relaxations >1 Gyr with a delta ellip decrease of >20%:  ', len(df_temp2))
            df_temp1 = df.loc[(df['Relaxation time'] > 2)]
            df_temp2 = df.loc[(df['Relaxation time'] > 2) & (df['Delta ellip'] < 0.80)]
            print('  Number of relaxations >2 Gyr:                                      ', len(df_temp1))
            print('  Number of relaxations >2 Gyr with a delta ellip decrease of >20%:  ', len(df_temp2))
            #print(df_temp2['GalaxyIDs'])
            
            
            #-------------
            ### Plotting ellip
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
            #-----------
            # Colourbar for kappa
            #norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
            #mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
            #-----------
            im1 = axs.scatter(df['Delta ellip'], df['Relaxation time'], c='k', s=0.3, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
            #plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
            #-----------
            ### Formatting
            axs.set_xlabel(r'$\epsilon_{\mathrm{mis}}/\epsilon_{\mathrm{pre}}$')
            axs.set_ylabel(r'$t_{\mathrm{relax}}$')
            #axs.set_xscale('log')
            #axs.set_xlim(0.4, 3)
            #axs.set_xticks(     [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3])
            #axs.set_xticklabels(['0.4', '', '0.6', '', '', '', '1', '2', '3'])
            axs.set_xlim(0.35, 2.55)
            axs.set_xticks(np.arange(0.5, 2.55, 0.5))
            axs.set_yscale('log')
            axs.set_ylim(0.1, 10)
            axs.set_yticks([0.1, 1, 10])
            axs.set_yticklabels(['0.1', '1', '10'])
        
            #-----------
            ### Text
            axs.text(0.4, 6, '$\leftarrow$ elliptical', color='grey', )
            axs.text(1.2, 6, r'disky $\rightarrow$', color='grey')
            axs.axvline(1.0, ls='-', lw=0.7, c='r', alpha=0.3)
        
            #-----------
            ### title
            if (len(set_hist_type) < 4):
                plot_annotate = ''
            if len(set_hist_type) < 4:
                if 'co-co' in set_hist_type:
                    plot_annotate = plot_annotate + 'co → co'
                if 'counter-counter' in set_hist_type:
                    plot_annotate = plot_annotate + ', counter → counter'
                if 'co-counter' in set_hist_type:
                    plot_annotate = plot_annotate + ', co → counter'
                if 'counter-co' in set_hist_type:
                    plot_annotate = plot_annotate + ', counter → co'
            if plot_annotate:
                axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
            #-----------
            ### other
            plt.tight_layout()
    
    
            #-----------
            ### Savefig  
                     
            if savefig:
                savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

                if len(set_hist_type) == 4:
                    savefig_txt = savefig_txt + '_allpaths'
            
                plt.savefig("%s/relaxation_time_morph_change/%s_ellipchange_trelax_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
                print("\n  SAVED: %s/relaxation_time_morph_change/%s_ellipchange_trelax_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
            if showfig:
                plt.show()
            plt.close()
        if plot_ttorque:
            ## Print statements
            df_temp1 = df.loc[(df['Relaxation time'] > 1)]
            df_temp2 = df.loc[(df['Relaxation time'] > 1) & (df['Delta ttorque'] > 1.20)]
            print('  Number of relaxations >1 Gyr:                                       ', len(df_temp1))
            print('  Number of relaxations >1 Gyr with a delta torque decrease of >20%:  ', len(df_temp2))
            df_temp1 = df.loc[(df['Relaxation time'] > 2)]
            df_temp2 = df.loc[(df['Relaxation time'] > 2) & (df['Delta ttorque'] > 1.20)]
            print('  Number of relaxations >2 Gyr:                                       ', len(df_temp1))
            print('  Number of relaxations >2 Gyr with a delta torque decrease of >20%:  ', len(df_temp2))
            
            
            #-------------
            ### Plotting ellip
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.1], sharex=True, sharey=False)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
            #-----------
            # Colourbar for kappa
            #norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
            #mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
            #-----------
            im1 = axs.scatter(df['Delta ttorque'], df['Relaxation time'], c='k', s=0.3, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
            #plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
            #-----------
            ### Formatting
            axs.set_xlabel(r'$\bar{t}_{\mathrm{torque,mis}}/\bar{t}_{\mathrm{torque,pre}}$')
            axs.set_ylabel(r'$t_{\mathrm{relax}}$')
            #axs.set_xscale('log')
            #axs.set_xlim(0.3, 3)
            #axs.set_xticks(     [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3])
            #axs.set_xticklabels(['0.3', '', '', '0.6', '', '', '', '1', '2', '3'])
            axs.set_xlim(0.2, 3.5)
            axs.set_xticks(np.arange(0.5, 3.51, 0.5))
            axs.set_yscale('log')
            axs.set_ylim(0.1, 10)
            axs.set_yticks([0.1, 1, 10])
            axs.set_yticklabels(['0.1', '1', '10'])
        
            #-----------
            ### Text
            axs.text(0.37, 6, '$\leftarrow$ reduced', color='grey')
            axs.text(1.5, 6, r'extended $\rightarrow$', color='grey')
            axs.axvline(1.0, ls='-', lw=0.7, c='r', alpha=0.3)
        
            #-----------
            ### title
            if (len(set_hist_type) < 4):
                plot_annotate = ''
            if len(set_hist_type) < 4:
                if 'co-co' in set_hist_type:
                    plot_annotate = plot_annotate + 'co → co'
                if 'counter-counter' in set_hist_type:
                    plot_annotate = plot_annotate + ', counter → counter'
                if 'co-counter' in set_hist_type:
                    plot_annotate = plot_annotate + ', co → counter'
                if 'counter-co' in set_hist_type:
                    plot_annotate = plot_annotate + ', counter → co'
            if plot_annotate:
                axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
            #-----------
            ### other
            plt.tight_layout()
    
    
            #-----------
            ### Savefig  
                     
            if savefig:
                savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

                if len(set_hist_type) == 4:
                    savefig_txt = savefig_txt + '_allpaths'
            
                plt.savefig("%s/relaxation_time_morph_change/%s_ttorquechange_trelax_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
                print("\n  SAVED: %s/relaxation_time_morph_change/%s_ttorquechange_trelax_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
            if showfig:
                plt.show()
            plt.close()
        if plot_ellip_ttorque:
            #-------------
            ### PRINT STATEMENT AND STATS
            res = stats.spearmanr(df['Delta ellip'], 1/df['Delta ttorque'], nan_policy='omit')
            print('   Spearman correlation:    %.5f'%res.correlation, '     p-value:    ', res.pvalue)
            print('   Standard deviation x:    %.5f'%np.std(df['Delta ellip']))
            print('   Standard deviation y:    %.5f'%np.std(df['Delta ttorque']))
            
            #-----------
            ### Plotting ellip
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=True, sharey=False)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
            #-----------
            # Colourbar for kappa
            #norm = mpl.colors.Normalize(vmin=9, vmax=11, clip=True)
            #mapper = cm.ScalarMappable(norm=norm, cmap='Spectral')         #cmap=cm.coolwarm), cmap='sauron'
    
            #-----------
            im1 = axs.scatter(df['Delta ellip'], 1/df['Delta ttorque'], c='k', s=0.3, zorder=99, edgecolors='k', linewidths=0.3, alpha=0.8)
            #plt.colorbar(im1, ax=axs, label=r'$M_{*}$ M$_{\odot}$', extend='both', pad=0.025)
        
            #-----------
            ### Formatting
            axs.set_xlabel(r'$\epsilon_{\mathrm{mis}}/\epsilon_{\mathrm{pre}}$')
            axs.set_ylabel(r'$1/(\bar{t}_{\mathrm{torque,mis}}/\bar{t}_{\mathrm{torque,pre}})$')
            #axs.set_xscale('log')
            #axs.set_xlim(0.3, 3)
            #axs.set_xticks(     [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3])
            #axs.set_xticklabels(['0.3', '', '', '0.6', '', '', '', '1', '2', '3'])
            axs.set_xlim(0.2, 2.4)
            axs.set_xticks(np.arange(0.5, 2.5, 0.5))
            axs.set_ylim(0.2, 2.4)
            axs.set_yticks(np.arange(0.5, 2.5, 0.5))
        
            #-----------
            ### Text
            #axs.text(0.37, 6, '$\leftarrow$ reduced', color='grey')
            #axs.text(1.5, 6, r'extended $\rightarrow$', color='grey')
            #axs.axvline(1.0, ls='-', lw=0.7, c='r', alpha=0.3)
            axs.text(1.48, 2.15, r'$\rho_{\mathrm{spearman}}$ = $%.3f$'%(res.correlation) + '\np-value = $%.1e$' %(res.pvalue))
            axs.grid(alpha=0.3)
        
            #-----------
            ### title
            if (len(set_hist_type) < 4):
                plot_annotate = ''
            if len(set_hist_type) < 4:
                if 'co-co' in set_hist_type:
                    plot_annotate = plot_annotate + 'co → co'
                if 'counter-counter' in set_hist_type:
                    plot_annotate = plot_annotate + ', counter → counter'
                if 'co-counter' in set_hist_type:
                    plot_annotate = plot_annotate + ', co → counter'
                if 'counter-co' in set_hist_type:
                    plot_annotate = plot_annotate + ', counter → co'
            if plot_annotate:
                axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
            #-----------
            ### other
            plt.tight_layout()
    
    
            #-----------
            ### Savefig  
                     
            if savefig:
                savefig_txt = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)

                if len(set_hist_type) == 4:
                    savefig_txt = savefig_txt + '_allpaths'
            
                plt.savefig("%s/relaxation_time_morph_change/%s_ellipttorquechange_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format), format=file_format, bbox_inches='tight', dpi=600)    
                print("\n  SAVED: %s/relaxation_time_morph_change/%s_ellipttorquechange_%s_subsample%s_%s.%s" %(fig_dir, 'L100_', len(misalignment_tree.keys()), len(ID_plot), savefig_txt, file_format)) 
            if showfig:
                plt.show()
            plt.close()
              

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Set starting parameters
load_csv_file_in = '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_NEW'  
plot_annotate_in                                           = None
savefig_txt_in   = load_csv_file_in               # [ 'manual' / load_csv_file_in ] 'manual' will prompt txt before saving
       
#'_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_NEW'                                                                                                                                               
#'_20Thresh_30Peak_normalLatency_anyMergers_hardMorph_NEW'  
#'_20Thresh_30Peak_normalLatency_anyMergers_ETG_NEW' 
#'_20Thresh_30Peak_normalLatency_anyMergers_LTG_NEW'  
#'_20Thresh_30Peak_normalLatency_anyMergers_hardETG_NEW'  
#'_20Thresh_30Peak_normalLatency_anyMergers_hardLTG_NEW'                                                                                                                          
#'_20Thresh_30Peak_normalLatency_anyMergers_ETG-ETG_NEW'                                                                                                                          
#'_20Thresh_30Peak_normalLatency_anyMergers_LTG-LTG_NEW'                                                                                                                         
#'_20Thresh_30Peak_normalLatency_anyMergers_hardETG-ETG_NEW'                                                                                                                         
#'_20Thresh_30Peak_normalLatency_anyMergers_hardLTG-LTG_NEW'                                                                                                                                          
#'_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010_NEW'
#'_20Thresh_30Peak_noLatency_NEW'
#'_20Thresh_30Peak_highLatency_NEW'
#'_20Thresh_30Peak_veryhighLatency_NEW'
#'_20Thresh_30Peak_50particles_NEW'
#'_20Thresh_30Peak_100particles_NEW'
#'_20error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_NEW'
#'_10error_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_NEW'
#'_normalLatency_anyMergers_anyMorph_NEW'   # no unstable
#'_lowmisangle_thresh_NEW.csv'              # no unstable, limit at 20

#  False 
# 'ETG → ETG' , 'LTG → LTG'
#  r'ETG ($\bar{\kappa}_{\mathrm{co}}^{\mathrm{*}} < 0.35$)'
# '$t_{\mathrm{relax}}>3\bar{t}_{\mathrm{torque}}$'    



#==================================================================================================================================
misalignment_tree, misalignment_input, summary_dict = _extract_tree(load_csv_file=load_csv_file_in, plot_annotate=plot_annotate_in, print_summary=True, EAGLE_dir=EAGLE_dir, sample_dir=sample_dir, tree_dir=tree_dir, output_dir=output_dir, fig_dir=fig_dir, dataDir_dict=dataDir_dict)
#==================================================================================================================================
if load_csv_file_in == '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010' or load_csv_file_in == '_20Thresh_30Peak_normalLatency_anyMergers_anyMorph_1010_NEW':
    print('\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n\t\tUSING 1010 SAMPLE\nTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')




# FIND MISALIGNMENT OF AN EVOLUTION
"""_find_misalignment_evolution(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in)"""


# SAMPLE MASS
"""_plot_sample_hist(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = False)"""
"""_average_particle_count(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = False)"""

"""_plot_sample_vs_dist_hist(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                              use_PDF   = False,        # uses probability density function
                            showfig = True,
                            savefig = False)"""


# TIMESCALE HISTOGRAMS
"""_plot_timescale_histogram(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                            set_plot_histogram_log            = False,    # set yaxis as log
                              add_inset                       = True,
                            showfig = True,
                            savefig = False)
_plot_tdyn_histogram(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                            set_plot_histogram_log            = False,    # set yaxis as log
                              add_inset                       = True, 
                            showfig = True,
                            savefig = False)
_plot_ttorque_histogram(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_plot_relaxation_type          = True,     # SET TO FALSE IF BELOW TRUE. Stack histogram types   
                            set_plot_histogram_log            = False,    # set yaxis as log
                              add_inset                       = True, 
                            showfig = True,
                            savefig = False)"""


# STACKED SINGLE
"""_plot_stacked_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""
"""_plot_stacked_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""
"""_plot_stacked_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = False)"""


# STACKED 2x2
"""_plot_stacked_trelax_2x2(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = True)
_plot_stacked_tdyn_2x2(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = True)
_plot_stacked_ttorque_2x2(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            add_stacked_median                = True,             # Whether to add a median line (will lower transparency of other lines)
                            set_stacked_relaxation_type       = ['co-co', 'co-counter', 'counter-co', 'counter-counter'],          # ['co-co', 'co-counter', 'counter-co', 'counter-counter']
                            showfig = True,
                            savefig = True)"""
                            

# PLOT BOX AND WHISKER OF RELAXATION DISTRIBUTIONS
"""_plot_box_and_whisker_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_whisker_morphs = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = True)
_plot_box_and_whisker_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_whisker_morphs = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = True)
_plot_box_and_whisker_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_whisker_morphs = ['ETG-ETG', 'LTG-LTG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = True)"""


# PLOT DELTA ANGLE, LOOKS AT PEAK ANGLE FROM 0, co-co and counter-counter
"""_plot_offset_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            use_offset_morphs    = True,
                              set_offset_morphs  = ['LTG-LTG', 'ETG-ETG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = True)
_plot_offset_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            use_offset_morphs    = True,
                              set_offset_morphs  = ['LTG-LTG', 'ETG-ETG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = True)
_plot_offset_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            use_offset_morphs    = True,
                              set_offset_morphs  = ['LTG-LTG', 'ETG-ETG'],           # Can be either relaxation_morph or misalignment_morph
                            showfig = True,
                            savefig = True)"""


# AVERAGE HALO MISALIGNMENT WITH RELAXTIME
"""_plot_halo_misangle_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_trelax = 0,
                              use_only_centrals              = True,        # Use only centrals
                            showfig = True,
                            savefig = True)
_plot_halo_misangle_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_tdyn = 0,
                              use_only_centrals              = True,        # Use only centrals
                            showfig = True,
                            savefig = True)
_plot_halo_misangle_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_ttorque = 0,
                              use_only_centrals              = True,        # Use only centrals
                            showfig = True,
                            savefig = True)"""
"""_plot_halo_misangle_manual(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_min_halo_trelax = 0,
                            showfig = True,
                            savefig = False)"""


# PLOTS RAW SCATTER OF GAS FRACTION WITH RELAXATION TIME
"""_plot_timescale_gas_scatter_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_gashist_min_trelax              = 0.2,
                              add_plot_gas_morph_median         = True,
                            showfig = True,
                            savefig = False)     # will auto-rename to _allpath if all 4 set_gashist_type used
_plot_timescale_gas_scatter_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_gashist_min_trelax              = 0.2,
                              add_plot_gas_morph_median         = True,
                            showfig = True,
                            savefig = False)
_plot_timescale_gas_scatter_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_gashist_min_trelax              = 0.2,
                              add_plot_gas_morph_median         = True,
                            showfig = True,
                            savefig = False)"""
# PLOTS HISTOGRAM OF GAS FRACTION WITH RELAXATION TIME
"""_plot_timescale_gas_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_gashist_min_trelax              = 0.2,
                            showfig = True,
                            savefig = True)     # will auto-rename to _allpath if all 4 set_gashist_type used
_plot_timescale_gas_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_gashist_min_trelax              = 0.2,
                            showfig = True,
                            savefig = True)
_plot_timescale_gas_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_gashist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_gashist_min_trelax              = 0.2,
                            showfig = True,
                            savefig = True)"""
                            


# PLOTS HISTOGRAM OF CENTRALS VS SATELLITES WITH RELAXATION TIME
"""_plot_timescale_occupation_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = False,        # Differentiates between ETG centrals, and LTG centrals. Will autorename
                            showfig = True,
                            savefig = True)      # will auto-rename to _allpath if all 4 set_gashist_type used
_plot_timescale_occupation_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = False,               
                            showfig = True,
                            savefig = True)
_plot_timescale_occupation_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = False,        
                            showfig = True,
                            savefig = True)"""
                            
"""_plot_timescale_occupation_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = True,        # Differentiates between ETG centrals, and LTG centrals. Will autorename
                            showfig = True,
                            savefig = True)      # will auto-rename to _allpath if all 4 set_gashist_type used
_plot_timescale_occupation_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = True,               
                            showfig = True,
                            savefig = True)   
_plot_timescale_occupation_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = True,        
                            showfig = True,
                            savefig = True)"""


# PLOTS HISTOGRAM OF STELLAR MASS WITH RELAXATION TIME
"""_plot_timescale_stelmass_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_only_centrals              = False,        # Use only centrals
                            showfig = True,
                            savefig = True)      # will auto-rename to _allpath if all 4 set_gashist_type used
_plot_timescale_stelmass_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_only_centrals              = False,        # Use only centrals
                            showfig = True,
                            savefig = True)      # will auto-rename to _allpath if all 4 set_gashist_type used
_plot_timescale_stelmass_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_only_centrals              = False,        # Use only centrals
                            showfig = True,
                            savefig = True)      # will auto-rename to _allpath if all 4 set_gashist_type used"""


# PLOTS HISTOGRAM OF HALO MASS WITH RELAXATION TIME
"""_plot_timescale_environment_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = False,        # Differentiates between ETG centrals, and LTG centrals. Will autorename
                              use_only_centrals              = True,        # Use only centrals
                            showfig = True,
                            savefig = True)      # will auto-rename to _allpath if all 4 set_gashist_type used
_plot_timescale_environment_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = False,      
                              use_only_centrals              = True,        # Use only centrals  
                            showfig = True,
                            savefig = True)
_plot_timescale_environment_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              use_occ_morph                  = False,       
                              use_only_centrals              = True,        # Use only centrals                               
                            showfig = True,
                            savefig = True)"""


# PLOTS HISTOGRAM OF GAS INFLOW WITH RELAXATION TIME - simplified    (un-comment df_2 lines)             
"""_plot_timescale_accretion_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              inflow_radius                  = 2.0,                # HMR to use
                              inflow_limits                  = [10, 10],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+      
                              use_gas_type                   = 'gas',                  # [ 'gas' / 'gas_sf' ]        
                              use_only_centrals              = True,        # Use only centrals                                                        
                            showfig = True,
                            savefig = False,
                            savefig_txt = 'simple')
_plot_timescale_accretion_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              inflow_radius                  = 2.0,                # HMR to use
                              inflow_limits                  = [10, 10],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+   
                              use_gas_type                   = 'gas',                  # [ 'gas' / 'gas_sf' ]     
                              use_only_centrals              = True,        # Use only centrals                                                              
                            showfig = True,
                            savefig = False,
                            savefig_txt = 'simple')
_plot_timescale_accretion_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              inflow_radius                  = 2.0,                # HMR to use
                              inflow_limits                  = [10, 10],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+         
                              use_gas_type                   = 'gas',                  # [ 'gas' / 'gas_sf' ]   
                              use_only_centrals              = True,        # Use only centrals                                                          
                            showfig = True,
                            savefig = False,
                            savefig_txt = 'simple')"""
                            
                            
# PLOTS HISTOGRAM OF SPECIFIC GAS INFLOW WITH RELAXATION TIME
"""_plot_timescale_specaccretion_histogram_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              inflow_radius                  = 2.0,                # HMR to use
                              inflow_limits                  = [1e-9, 2e-9],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+   
                              use_gas_type                   = 'gas',                  # [ 'gas' / 'gas_sf' ]       
                              use_only_centrals              = True,        # Use only centrals                                           
                            showfig = True,
                            savefig = True)
_plot_timescale_specaccretion_histogram_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              inflow_radius                  = 2.0,                # HMR to use
                              inflow_limits                  = [1e-9, 2e-9],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+     
                              use_gas_type                   = 'gas',                  # [ 'gas' / 'gas_sf' ]     
                              use_only_centrals              = True,        # Use only centrals                                                                          
                            showfig = True,
                            savefig = True)
_plot_timescale_specaccretion_histogram_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0.2,
                              inflow_radius                  = 2.0,                # HMR to use
                              inflow_limits                  = [1e-9, 2e-9],            # [ < lower - upper < Msun/yr ] e.g. [5, 10] means <5, 5-10, 10+   
                              use_gas_type                   = 'gas',                  # [ 'gas' / 'gas_sf' ]     
                              use_only_centrals              = True,        # Use only centrals                                                                            
                            showfig = True,
                            savefig = True)"""


# PLOTS HISTOGRAM OF FRACTIONAL CHANGE IN TTORQUE, ELLIP, AND BOTH
"""_plot_morphology_change_ttorque_ellip(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_hist_type                    = ['co-co', 'counter-counter', 'co-counter', 'counter-co'],
                            set_hist_min_trelax              = 0,
                            plot_scatter                        = True,
                            plot_histogram                      = True,
                              plot_ellip                        = True,       # change in ellip vs trelax
                              plot_ttorque                      = True,       # change in ttorque theoretical vs trelax
                              plot_ellip_ttorque                = True,
                            showfig = True,
                            savefig = True)      # will auto-rename to _allpath if all 4 set_gashist_type used"""


       
#--------------------------------
# SUITED FOR 1010 SAMPLE AND ABOVE   

# PLOT NUMBER OF MERGERS WITH RELAXATION TIME, SUITED FOR 1010 SAMPLE AND ABOVE
"""_plot_merger_count_trelax(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = True)
_plot_merger_count_tdyn(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = True)
_plot_merger_count_ttorque(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            showfig = True,
                            savefig = True)"""


# READ IN STATS OF TREE, BUT ONLY WITH MERGERS
"""misalignment_tree, misalignment_input, summary_dict = _extract_tree(load_csv_file=load_csv_file_in, plot_annotate=plot_annotate_in, print_summary=True, EAGLE_dir=EAGLE_dir, sample_dir=sample_dir, tree_dir=tree_dir, output_dir=output_dir, fig_dir=fig_dir, dataDir_dict=dataDir_dict,
                  use_alt_merger_criteria = True,
                    half_window         = 0.5,      # [ 0.2 / +/-Gyr ] window centred on first misaligned snap to look for mergers
                    min_ratio           = 0.1,   
                    merger_lookback_time = 2)       # Gyr, number of years to check for peak stellar mass"""

# AVERAGE HALO MISALIGNMENT BEFORE MISALIGNMENT WITH FRACTIONAL OCCURENCE
"""_plot_halo_misangle_pre_frac(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            set_misanglepre_type = ['co-co', 'co-counter'],           # [ 'co-co', 'co-counter' ]  or False
                            showfig = True,
                            savefig = True)"""

  
# PLOTS STACKED BAR CHART. WILL USE use_alt_merger_criteria FROM EARLIER TO FIND MERGERS,          
"""_plot_origins(misalignment_tree=misalignment_tree, misalignment_input=misalignment_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                            # Mergers
                            use_alt_merger_criteria = True,
                            add_total               = True,     # add total (+1 PIE CHART)
                              half_window         = 0.3,      # [ 0.2 / 0.3 / 0.5 +/-Gyr ] window centred on first misaligned snap to look for mergers
                              min_ratio           = 0.1,   
                              merger_lookback_time = 2,       # Gyr, number of years to check for peak stellar mass
                            showfig = True,
                            savefig = True)    """          
#====================================













    