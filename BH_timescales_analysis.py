import h5py
import numpy as np
import scipy
from scipy import stats
import math
import random
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
import csv
import json
import time
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter
from subhalo_main import Initial_Sample, Subhalo_Extract, Subhalo_Analysis, ConvertID, ConvertID_noMK, MergerTree
import eagleSqlTools as sql
from graphformat import set_rc_params, lighten_color
from read_dataset_directories import _assign_directories
from extract_misalignment_trees import _extract_BHmis_tree, _refine_BHmis_sample


#====================================
# finding directories
answer = input("-----------------\nDirectories?:\n     1 local\n     2 serpens_snap\n     3 snip\n     4 snip local           ->  ")
EAGLE_dir, sample_dir, tree_dir, output_dir, fig_dir, dataDir_dict = _assign_directories(answer)
#register_sauron_colormap()
#====================================



#--------------------------------
# Create plots of SFR, sSFR, BH mass within our sample using BH_tree for misaligned
def _BH_sample(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Plot options:
                        plot_yaxis = ['sfr'],     # ['sfr', 'ssfr', 'bhmass']
                      # Sample options
                        apply_at_start = True,      # True = first snip, False = takes mean over window
                        #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                        min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                        min_bhmass   = None,      max_bhmass   = None,
                        min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                        min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                        # Mergers, looked for within range considered +/- halfwindow
                        use_merger_criteria = False,
                      #==============================================
                      showfig       = True,
                      savefig       = True,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    
    # Establish sub-sample we wish to focus on
    BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                              apply_at_start = apply_at_start,  
                                                              min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                              min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                              min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                              min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                              use_merger_criteria = use_merger_criteria)
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    bhmass_plot    = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    state_plot     = []
    ID_plot        = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            if apply_at_start:
                # Append first BH mass
                if use_CoP_BH:
                    bhmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[0])
                else:
                    bhmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[0])
                    
                stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[0])
                sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[0])
                ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[0])
                kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[0])
                
                state_plot.append(galaxy_state)
                ID_plot.append(ID_i)
            
            else:
                # Append means
                if use_CoP_BH:
                    bhmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])))
                else:
                    bhmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])))
                    
                stelmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])))
                sfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])))
                ssfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])))
                kappa_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])))
                
                state_plot.append(galaxy_state)
                ID_plot.append(ID_i)
                
                    
            
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'BH mass': bhmass_plot, 'Morphology': kappa_plot, 'State': state_plot, 'GalaxyIDs': ID_plot})        
    #print('\nSample after processing:\n   aligned: %s\n   misaligned: %s\n   counter: %s' %(len(df.loc[(df['State'] == 'aligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'misaligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'counter')]['GalaxyIDs'])))
    
    
    #df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'ETG → ETG')]
    
    
    if 'bhmass' in plot_yaxis:
        #------------------------
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
        color_dict = {'aligned':'grey', 'misaligned':'r', 'counter':'b'}
        plt.scatter(np.log10(df['stelmass']), df['BH mass'], s=0.1, c=[color_dict[i] for i in df['State']], edgecolor=[color_dict[i] for i in df['State']], marker='.', alpha=0.8)
    
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9, 11.5)
        axs.set_ylim(1e5, 1e10)
        axs.set_yscale('log')
        axs.set_xlabel('Stellar mass 2r50')
        axs.set_ylabel('BH mass [Msun]')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')
        legend_labels.append('misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('r')
        legend_labels.append('counter')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('b')
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        
        axs.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == False else plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        # other
        plt.tight_layout()
    
        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/BH_sample_analysis/%sstelmass_bhmass_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_bhmass_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if 'sfr' in plot_yaxis:
        #------------------------
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
        color_dict = {'aligned':'grey', 'misaligned':'r', 'counter':'b'}
        plt.scatter(np.log10(df['stelmass']), df['SFR'], s=0.1, c=[color_dict[i] for i in df['State']], edgecolor=[color_dict[i] for i in df['State']], marker='.', alpha=0.8)
        
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9, 11.5)
        axs.set_ylim(0.01, 100)
        axs.set_yscale('log')
        axs.set_xlabel('Stellar mass 2r50')
        axs.set_ylabel('SFR [Msun/yr]')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')
        legend_labels.append('misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('r')
        legend_labels.append('counter')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('b')
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        
        axs.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == False else plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        # other
        plt.tight_layout()
    
        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/BH_sample_analysis/%sstelmass_sfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_sfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    if 'ssfr' in plot_yaxis:
        #------------------------
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
        color_dict = {'aligned':'grey', 'misaligned':'r', 'counter':'b'}
        plt.scatter(np.log10(df['stelmass']), df['sSFR'], s=0.1, c=[color_dict[i] for i in df['State']], edgecolor=[color_dict[i] for i in df['State']], marker='.', alpha=0.8)
    
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9, 11.5)
        axs.set_ylim(3*10**-13, 3*10**-9)
        axs.set_yscale('log')
        axs.set_xlabel('Stellar mass 2r50')
        axs.set_ylabel('sSFR [/yr]')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='both', width=0.8, length=2)
        
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('k')
        legend_labels.append('misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('r')
        legend_labels.append('counter')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('b')
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        axs.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == False else plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        # other
        plt.tight_layout()
    
        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/BH_sample_analysis/%sstelmass_ssfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_ssfr_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
    
    
    



#--------------------------------
# Create stacked plots of BH growth over time
def _BH_stacked_evolution(BH_tree, BH_input, summary_dict, plot_annotate = None, savefig_txt_in = None,
                      #==============================================
                      # Sample options
                        min_window_size   = 1,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                        #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                        min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                        min_bhmass   = None,      max_bhmass   = None,
                        min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                        min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                        # Mergers, looked for within range considered +/- halfwindow
                        use_merger_criteria = False,
                      # Plot options
                        use_random_sample  = 50,        # Number of aligned galaxies to randomly select (within limits set above)
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-----------------------------------
    # Establish sub-sample we wish to focus on
    BH_subsample, summary_dict_subsample = _refine_sample(BH_tree = BH_tree, BH_input = BH_input, summary_dict = summary_dict,
                                                              min_window_size   = min_window_size,     
                                                              min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                              min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                              min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                              min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                              use_merger_criteria = use_merger_criteria)
    
    
    #===================================================================================================
    # Go through subsample
    stelmass_plot  = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    state_plot      = []
    bhmass_plot    = []
    timeaxis_plot   = []
    ID_plot     = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            bhmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass']))            
            timeaxis_plot.append(-1*np.array(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'][0]))
            
            # Append means
            stelmass_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])))
            sfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])))
            ssfr_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])))
            kappa_plot.append(np.mean(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])))
            state_plot.append(galaxy_state)
            ID_plot.append(ID_i)
            
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'bhmass': bhmass_plot, 'timeaxis': timeaxis_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Morphology': kappa_plot, 'State': state_plot, 'GalaxyIDs': ID_plot})   
    print('\nSub-sample after processing:')
    print('  aligned:     ', len(df.loc[(df['State'] == 'aligned')]['GalaxyIDs']))
    print('  misaligned:  ', len(df.loc[(df['State'] == 'misaligned')]['GalaxyIDs']))
    print('  counter:     ', len(df.loc[(df['State'] == 'counter')]['GalaxyIDs']))
            
            
    
    #------------------------
    # Figure initialising
    fig, axs = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    color_dict = {'aligned':'grey', 'misaligned':'r', 'counter':'b'}
        
    test_df = df.loc[(df['State'] == 'misaligned')]
    for index, row in test_df.iterrows():        
        axs.plot(row['timeaxis'], row['bhmass'], lw=0.3, alpha=0.5)
        
    axs.set_xlim(-1, 8)
    axs.set_xticks(np.arange(0, 8+0.1, 1))
    axs.set_xlabel('Time (Gyr)')
    axs.set_yscale('log')
    axs.minorticks_on()
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    
    


    fig, axs2 = plt.subplots(1, 1, figsize=[10/3, 2.5], sharex=False, sharey=False)
    
    test_df = df.loc[(df['State'] == 'counter')]
    for index, row in test_df.iterrows():        
        axs2.plot(row['timeaxis'], row['bhmass'], lw=0.3, alpha=0.5)
    
    axs2.set_xlim(-1, 8)
    axs2.set_xticks(np.arange(0, 8+0.1, 1))
    axs2.set_xlabel('Time (Gyr)')
    axs2.set_yscale('log')
    axs2.minorticks_on()
    axs2.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    axs2.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    plt.show()
    
            
            


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# Set starting parameters
load_BHmis_tree_in = '_CoPFalse_window0.5_trelax0.3___05Gyr__no_seed'
plot_annotate_in   = False
savefig_txt_in     = load_BHmis_tree_in               # [ 'manual' / load_csv_file_in ] 'manual' will prompt txt before saving

# '_CoPTrue_window0.5_trelax0.3___05Gyr__no_seed'
# '_CoPTrue_window0.5_trelax0.3___05Gyr___'
#  False 
# 'ETG → ETG' , 'LTG → LTG'
#  r'ETG ($\bar{\kappa}_{\mathrm{co}}^{\mathrm{*}} < 0.35$)'
# '$t_{\mathrm{relax}}>3\bar{t}_{\mathrm{torque}}$'    

#==================================================================================================================================
# Specify what BH to use in 'refine sample' function
BHmis_tree, BHmis_input, BH_input, BHmis_summary = _extract_BHmis_tree(csv_BHmis_tree=load_BHmis_tree_in, plot_annotate=plot_annotate_in, print_summary=True, EAGLE_dir=EAGLE_dir, sample_dir=sample_dir, tree_dir=tree_dir, output_dir=output_dir, fig_dir=fig_dir, dataDir_dict=dataDir_dict)
#==================================================================================================================================





# SAMPLE BH mass
"""_BH_sample(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                plot_yaxis = ['bhmass', 'sfr', 'ssfr'],
                                  apply_at_start = True,
                                min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                                min_bhmass   = None,      max_bhmass   = None,        # seed = 1.48*10**5
                                min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                                min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                                use_merger_criteria = False,
                              showfig = True,
                              savefig = False)"""
                              
                              

                              


   # adjust overlay plot                    





                            
# STACKED BH growth
"""_BH_stacked_evolution(BH_tree=BH_tree, BH_input=BH_input, summary_dict=summary_dict, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                                min_bhmass   = None,      max_bhmass   = None,
                                min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                                min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                                use_merger_criteria = False,
                                  use_random_sample  = 50,                      # Number of aligned galaxies to randomly select (within limits set above)       
                              showfig = True,
                              savefig = False) """   
                            
                            
                            
      
                            
                            
                            
                            
#We want a way to overlay the BH growths of misaligned galaxies, over that of non-misaligned and counter-rotators
## ANALYSE TREE:
#- Should now have a sample of misaligned, aligned, and counter-rotating galaxies with all BH growth values
#   - PLOT 0: Establish sample
#	- PLOT 1: Overlay all BH growths, within similar window lengths, plot medians
#   - PLOT 2: x-y of BH mass at start vs delta mass, coloured for counter/misaligned/aligned -> a few of these for 500 mya (trim longer aligend/counter ones), 1 gya, 2 gyr
#   - PLOT 3: x-y of BH mass at start vs delta mass/time, coloured for counter/misaligned/aligned
#	- may also want to define a sf_gas_kappa, and look at *when* BH growth is boosted   
#	- Trying to answer: for similar galaxies, does a misalignment induce more BH growth? Should have highest in counter -> misaligned -> aligned


                
                            
                            
                            
                            
                            
                            