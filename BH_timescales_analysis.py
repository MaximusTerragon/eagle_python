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
                        plot_yaxis = ['bhmass', 'window'],     # ['sfr', 'ssfr', 'bhmass', 'window']
                          add_seed_limit    = 1*10**6,       # [ False / value ] seed mass = 1.48*10**5, quality cut made at 5x
                          add_observational = False,                # not working
                          add_bulge_ratios  = True,
                          add_histograms    = True, 
                      # Sample options
                        apply_at_start = True,      # True = first snip, False = takes mean over window
                      # Sample refinement
                        run_refinement = False,
                          #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                          # basic properties
                          min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                          min_bhmass   = None,      max_bhmass   = None,
                          min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                          min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                          # Mergers, looked for within range considered +/- halfwindow
                          use_merger_criteria = False,
                      #==============================================
                      showfig       = True,
                      savefig       = False,    
                        file_format   = 'pdf',
                        savefig_txt = 'better_format',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                      #-----------------------------
                      debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
        
    
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    bhmass_plot    = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            if apply_at_start:
                # establish starting index (0 for aligned/counter, )
                if galaxy_state == 'misaligned':
                    starting_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']
                else:
                    starting_index = 0
                
                # Append first BH mass
                if use_CoP_BH:
                    bhmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[starting_index])
                else:
                    bhmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[starting_index])
                    
                stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[starting_index])
                sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[starting_index])
                ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[starting_index])
                kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[starting_index])
                
                duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                if galaxy_state == 'misaligned':
                    trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                
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
                
                duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                if galaxy_state == 'misaligned':
                    trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                
                state_plot.append(galaxy_state)
                ID_plot.append(ID_i)
                
                    
            
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'BH mass': bhmass_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    #print('\nSample after processing:\n   aligned: %s\n   misaligned: %s\n   counter: %s' %(len(df.loc[(df['State'] == 'aligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'misaligned')]['GalaxyIDs']), len(df.loc[(df['State'] == 'counter')]['GalaxyIDs'])))
    
    
    #df.loc[(df['Occupation'] == 'central') & (df['Relaxation morph'] == 'ETG → ETG')]
    
    color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
    
    if 'bhmass' in plot_yaxis:
        #---------------------------  
        # Figure initialising
        if add_histograms:
            
            fig = plt.figure(figsize=(10/3, 10/3))
            gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.1, hspace=0.1)
            # Create the Axes.
            axs = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=axs)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=axs)
    
            #---------------------
            ### Plot histograms
            for state_i in color_dict.keys():

                df_state = df.loc[df['State'] == state_i]
                
                ax_histx.hist(np.log10(df_state['stelmass']), bins=np.arange(9.25, 11.5+0.001, 0.25), log=True, facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=1)
                ax_histx.hist(np.log10(df_state['stelmass']), bins=np.arange(9.25, 11.5+0.001, 0.25), log=True, facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
                ax_histy.hist(np.log10(df_state['BH mass']), bins=np.arange(5, 10.1, 0.25), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=0.8)
                ax_histy.hist(np.log10(df_state['BH mass']), bins=np.arange(5, 10.1, 0.25), log=True, orientation='horizontal', facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
                
                
                #-------------
                # Formatting
                ax_histy.set_xlabel('Count')
                ax_histx.set_ylabel('Count')
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)
                ax_histx.set_yticks([1, 10, 100, 1000])
                ax_histx.set_yticklabels(['', '$10^1$', '', '$10^3$'])
                ax_histy.set_xticks([1, 10, 100, 1000])
                ax_histy.set_xticklabels(['', '$10^1$', '', '$10^3$'])
        else:
            fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
        

        #--------------
        # scatter
        axs.scatter(np.log10(df['stelmass']), np.log10(df['BH mass']), s=6, c=[color_dict[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)

    
        #-----------
        # Annotations
        if add_seed_limit:
            #axs.axhline(add_seed_limit, c='k', lw=0.5, ls='-', zorder=999, alpha=0.7)
            axs.axhspan(ymin=2,ymax=np.log10(add_seed_limit), facecolor='grey', zorder=999, alpha=0.3)
            axs.text(10.8, 6.05, 'sample limit', color='k', alpha=0.7, fontsize=6)
        if add_bulge_ratios:
            axs.text(9.31, np.log10(2.7*10**7), '$M_{\mathrm{BH}}/M_{*}=0.01$', color='k', fontsize=6, rotation=22, rotation_mode='anchor')
            axs.plot([8, 12], [6, 10], ls='--', lw=0.5, color='k')

            axs.text(9.31, np.log10(2.7*10**6), '$M_{\mathrm{BH}}/M_{*}=0.001$', color='k', fontsize=6, rotation=22, rotation_mode='anchor')
            axs.plot([8, 12], [5, 9], ls='--', lw=0.5, color='k')

            #axs.text(9.3, np.log10(2.5*10**5), '$M_{\mathrm{BH}}/M_{*}=10^{-4}$', color='k', alpha=0.7, fontsize=6, rotation=22, rotation_mode='anchor')
            #axs.plot([8, 12], [4, 8], ls='--', lw=0.5, color='k')
        
    
    
        #-----------
        ### General formatting
        # Axis labels
        axs.set_xlim(9.25, 11.5)
        axs.set_ylim(5, 10)
        #axs.set_yscale('log')
        axs.set_xlabel(r'log$_{10}$ M$_{*}(2r_{50})$ [M$_{\odot}$]')
        #axs.set_ylabel('BH mass [Msun]')
        axs.set_ylabel(r'log$_{10}$ most massive $M_{\mathrm{BH}}(<r_{50}$) [M$_{\odot}]$')
        #axs.minorticks_on()
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
        #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('aligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('grey')
        legend_labels.append('misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('orangered')
        legend_labels.append('counter-rotating')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append('dodgerblue')
        axs.legend(handles=legend_elements, labels=legend_labels, loc='upper left', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    
        #axs.set_title(r'CoP BH: %s %s' %(use_CoP_BH, '' if plot_annotate == False else plot_annotate), size=7, loc='left', pad=3)
    
        #-----------
        # other
        plt.tight_layout()

        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt) + ('hist' if add_histograms else '')
        
            plt.savefig("%s/BH_sample_analysis/%sstelmass_bhmass_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%sstelmass_bhmass_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()   
    if 'sfr' in plot_yaxis:
        #------------------------
        # Figure initialising
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
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
        legend_colors.append(color_dict['aligned'])
        legend_labels.append('misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
        legend_labels.append('counter')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
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
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
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
        legend_colors.append(color_dict['aligned'])
        legend_labels.append('misaligned')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['misaligned'])
        legend_labels.append('counter')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        legend_colors.append(color_dict['counter'])
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
    if 'window' in plot_yaxis:
        
        #-------------
        ### Plotting
        fig, (ax_trelax, ax_mis, ax_cnt) = plt.subplots(nrows=3, ncols=1, figsize=[10/3, 10/3], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        xaxis_max = 8       #[Gyr]
        bin_width = 0.05
        
        #------------
        # Histograms
        # misaligned
        df_mis = df.loc[df['State'] == 'misaligned']
        ax_mis.hist(df_mis['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='none', facecolor=color_dict['misaligned'], alpha=0.1)
        bin_count, _, _ = ax_mis.hist(df_mis['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor=color_dict['misaligned'], lw=0.4, facecolor='none', alpha=1.0)

        # Add poisson errors to each bin (sqrt N)
        #hist_n, _ = np.histogram(df_mis['window'], bins=np.arange(0, xaxis_max+0.0001, 0.bin_width), range=(0, xaxis_max+0.01))
        #ax_mis.errorbar(np.arange(0.1/2, xaxis_max+0.5*bin_width, bin_width), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor=color_dict['misaligned'], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        
        # counter
        df_cnt = df.loc[df['State'] == 'counter']
        ax_cnt.hist(df_cnt['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='none', facecolor=color_dict['counter'], alpha=0.1)
        bin_count, _, _ = ax_cnt.hist(df_cnt['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor=color_dict['counter'], lw=0.4, facecolor='none', alpha=1.0)

        # Add poisson errors to each bin (sqrt N)
        #hist_n, _ = np.histogram(df_cnt['window'], bins=np.arange(0, xaxis_max+0.0001, bin_width), range=(0, xaxis_max+bin_width))
        #ax_cnt.errorbar(np.arange(0.1/2, xaxis_max+0.5*bin_width, bin_width), hist_n, xerr=None, yerr=np.sqrt(hist_n), ecolor=color_dict['misaligned'], ls='none', capsize=2, elinewidth=1.0, markeredgewidth=1, alpha=0.9)
        
        
        # trelax (misaligned)
        ax_trelax.hist(trelax_plot, bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='none', facecolor='r', alpha=0.1)
        bin_count, _, _ = ax_trelax.hist(trelax_plot, bins=np.arange(0, xaxis_max+0.0001, bin_width), histtype='bar', edgecolor='r', lw=0.4, facecolor='none', alpha=1.0)
        
        print('For guide for later analysis:')
        print('Total misaligned: %s' %len(trelax_plot))
        print('Number of trelax >0.3  Gyr: %s' %len([i for i in trelax_plot if i > 0.3]))
        print('Number of trelax >0.4  Gyr: %s' %len([i for i in trelax_plot if i > 0.4]))
        print('Number of trelax >0.5  Gyr: %s' %len([i for i in trelax_plot if i > 0.5]))
        print('Number of trelax >0.6  Gyr: %s' %len([i for i in trelax_plot if i > 0.6]))
        print('Number of trelax >0.7  Gyr: %s' %len([i for i in trelax_plot if i > 0.7]))
        print('Number of trelax >0.8  Gyr: %s' %len([i for i in trelax_plot if i > 0.8]))
        print('Number of trelax >0.9  Gyr: %s' %len([i for i in trelax_plot if i > 0.9]))
        print('Number of trelax >1.0  Gyr: %s' %len([i for i in trelax_plot if i > 1.0]))
        
        #-----------
        ### General formatting
        # Axis labels
        ax_mis.set_xlim(0, xaxis_max)
        ax_mis.set_xticks(np.arange(0, xaxis_max+0.01, step=1))
        ax_cnt.set_xlabel('Duration [Gyr]')
        ax_mis.set_ylim(0.7, 500)
        ax_mis.set_yscale('log')
        ax_mis.set_ylabel('Number in sample')
    
    
        #-----------
        ### Legend
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('misaligned (trelax)')
        legend_colors.append('r')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        ax_trelax.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)

        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('misaligned (window)')
        legend_colors.append('orangered')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        ax_mis.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
        
        legend_elements = []
        legend_labels = []
        legend_colors = []
        legend_labels.append('counter-rotating (window)')
        legend_colors.append('dodgerblue')
        legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
        ax_cnt.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    
        #-----------
        ### title
        if plot_annotate:
            axs_mis.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
        #-----------
        ### other
        #plt.tight_layout()
        
        #-----------
        ### Savefig                   
        if savefig:
            savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
            
            plt.savefig("%s/BH_sample_analysis/%swindow_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), format=file_format, bbox_inches='tight', dpi=600)    
            print("\n  SAVED: %s/BH_sample_analysis/%swindow_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
        if showfig:
            plt.show()
        plt.close()
        
        
#--------------------------------
# Create stacked plots of BH growth over time, misaligned systems are not necessarily relaxed yet
def _BH_stacked_evolution(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            min_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                            add_stacked_median = True,
                            #use_random_sample  = 50,        # Number of aligned galaxies to randomly select (within limits set above)
                              target_stelmass       = False,          # [ 10** Msun / False ]
                                target_stelmass_err = 0.1,      # [ Dex ]
                              target_bhmass         = 6.5,              # [ 10**[] Msun / False ]
                                target_bhmass_err   = 0.1,        # [ Dex ] e.g. log(10**7) = 7 ± 0.1
                              
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    bhmass_plot    = []
    bhmass_start_plot = []
    lookbacktime_plot = []
    diff_co        = []     # arrays to collect an estimate of temporal separation
    diff_mis       = []
    diff_cnt       = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    
    stats_lines = {'aligned': {'time': [],
                               'mass': []},
                   'misaligned': {'time': [],
                                  'mass': []},
                   'counter': {'time': [],
                               'mass': []}}
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < min_window_size:
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']
                if use_CoP_BH:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[check_index]
                else:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[check_index]
                stel_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[check_index]
                time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (BH_check >= (10**(target_bhmass - target_bhmass_err) if target_bhmass else 0)) & (BH_check <= (10**(target_bhmass + target_bhmass_err) if target_bhmass else 10*15)) & (stel_check >= (10**(target_stelmass - target_stelmass_err) if target_stelmass else 0)) & (stel_check <= (10**(target_stelmass + target_stelmass_err) if target_stelmass else 10**15)) & (time_check >= min_window_size):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    #index_stop  = np.where(duration_array > min_window_size)[0][0] + 1
                    
                    
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    lookbacktime_plot.append(time_axis)
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:]
                        bhmass_plot.append(mass_axis)
                        bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start])
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:]
                        bhmass_plot.append(mass_axis)
                        bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start])
                    
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
            
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    
                    trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
            
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                    
                    # Stats
                    diff_mis.append(time_axis[1])
                    stats_lines['misaligned']['time'].extend(time_axis)
                    stats_lines['misaligned']['mass'].extend(mass_axis)
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                
                if use_CoP_BH:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])
                else:
                    BH_check   = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])
                stel_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])
                
                mask_check = np.where((BH_check >= (10**(target_bhmass - target_bhmass_err) if target_bhmass else 0)) & (BH_check <= (10**(target_bhmass + target_bhmass_err) if target_bhmass else 10*15)) & (stel_check >= (10**(target_stelmass - target_stelmass_err) if target_stelmass else 0)) & (stel_check <= (10**(target_stelmass + target_stelmass_err) if target_stelmass else 10**15)))[0]
                if len(mask_check) > 0:
                    # array of indexes that will meet the bhmass, stelmass, and min_window_size criteria:
                    check_index_array = []
                    for index_i in mask_check:
                        time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                        if time_check > min_window_size:
                            check_index_array.append(index_i)
                    
                    # If there exists at least one valid entry, pick random to append min_window_max entries to
                    if len(check_index_array) > 0:
                        # Pick random starting point
                        index_start = random.choice(check_index_array)
                        duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        #index_stop  = np.where(duration_array > min_window_size)[0][0] + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        lookbacktime_plot.append(time_axis)
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:]
                            bhmass_plot.append(mass_axis)
                            bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start])
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:]
                            bhmass_plot.append(mass_axis)
                            bhmass_start_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start])
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
            
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
            
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                        
                        # Stats
                        if galaxy_state == 'aligned':
                            diff_co.append(time_axis[1])
                            stats_lines['aligned']['time'].extend(time_axis)
                            stats_lines['aligned']['mass'].extend(mass_axis)
                        if galaxy_state == 'counter':
                            diff_cnt.append(time_axis[1])
                            stats_lines['counter']['time'].extend(time_axis)
                            stats_lines['counter']['mass'].extend(mass_axis)
                    

            
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'BH mass': bhmass_plot, 'Time axis': lookbacktime_plot, 'BH mass start': bhmass_start_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target BH_mass median of 10**%s:' %target_bhmass)
    print('  aligned:     %s\t\t%.4f (first snipshot, use below instead)' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start']))))
    print('  misaligned:  %s\t\t%.4f' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start']))))
    print('  counter:     %s\t\t%.4f' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start']))))
    print(' ')
    
    #------------------------
    # Figure initialising
    fig, (ax_combi, ax_co, ax_mis, ax_cnt) = plt.subplots(nrows=4, ncols=1, figsize=[10/3, 5.5], sharex=True, sharey=True) 
    plt.subplots_adjust(wspace=0.2, hspace=0, bottom=0.13)
    
    color_dict = {'aligned':'k', 'misaligned':'orangered', 'counter':'dodgerblue'}
    
    #------------
    # Plot lines
    for index, row in df_co.iterrows():        
        ax_co.plot(row['Time axis'], np.log10(row['BH mass']), c='lightgrey', lw=0.3, alpha=0.5, zorder=-1)
        #ax_co.plot(row['Time axis'], np.log10(row['BH mass']), 'k^', alpha=0.5, zorder=-1)
        
    for index, row in df_mis.iterrows():        
        ax_mis.plot(row['Time axis'], np.log10(row['BH mass']), c='lightsalmon', lw=0.3, alpha=0.3, zorder=-1)
        #ax_mis.plot(row['Time axis'], np.log10(row['BH mass']), 'k^', alpha=0.3, zorder=-1)
        
    for index, row in df_cnt.iterrows():        
        ax_cnt.plot(row['Time axis'], np.log10(row['BH mass']), c='lightsteelblue', lw=0.3, alpha=0.5, zorder=-1)
        #ax_cnt.plot(row['Time axis'], np.log10(row['BH mass']), 'k^', alpha=0.5, zorder=-1)
    
    #------------
    # Plot median and percentiles
    if add_stacked_median:
        
        stats_growth = {'aligned': {'median': [],
                                    'upper': [],
                                    'lower': []},
                        'misaligned': {'median': [],
                                       'upper': [],
                                       'lower': []},
                        'counter': {'median': [],
                                    'upper': [],
                                    'lower': []}}
        
        use_percentiles = 16        # 1 sigma
        bins = np.arange(-(0.125/2), min_window_size+(0.125/2)+0.01, 0.125)
        plot_bins = np.arange(0, min_window_size+0.01, 0.125)
        
        if len(diff_co) != 0:
            line_color = color_dict['aligned']
            #bins = np.arange((-1*np.median(diff_co))+0.00001, min_window_size+0.05, np.median(diff_co))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['aligned']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['aligned']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['aligned']['time'])[mask]
                current_mass   = np.array(stats_lines['aligned']['mass'])[mask]
                            
                median_array.append(np.percentile(current_mass, 50))
                median_upper.append(np.percentile(current_mass, 100-use_percentiles))
                median_lower.append(np.percentile(current_mass, use_percentiles))
            
            stats_growth['aligned']['median'] = np.log10(median_array)
            stats_growth['aligned']['upper'] = np.log10(median_upper)
            stats_growth['aligned']['lower'] = np.log10(median_lower)
                   
            #----------
            # plot
            ax_co.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100)
            ax_co.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100)
            ax_co.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100)
            
            ax_combi.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
        if len(diff_mis) != 0:
            line_color = color_dict['misaligned']
            #bins = np.arange((-1*np.median(diff_co))+0.00001, min_window_size+0.05, np.median(diff_mis))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['misaligned']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['misaligned']['time']), bins=bins)
            
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['misaligned']['time'])[mask]
                current_mass   = np.array(stats_lines['misaligned']['mass'])[mask]
                            
                median_array.append(np.percentile(current_mass, 50))
                median_upper.append(np.percentile(current_mass, 100-use_percentiles))
                median_lower.append(np.percentile(current_mass, use_percentiles))
            
            stats_growth['misaligned']['median'] = np.log10(median_array)
            stats_growth['misaligned']['upper'] = np.log10(median_upper)
            stats_growth['misaligned']['lower'] = np.log10(median_lower)
                 
            #----------
            # plot
            ax_mis.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100)
            ax_mis.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100)
            ax_mis.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100)
            
            ax_combi.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
        if len(diff_cnt) != 0:
            line_color = color_dict['counter']
            #bins = np.arange((-1*np.median(diff_co))+0.00001, min_window_size+0.05, np.median(diff_cnt))
            
            # Create mask that is essentially the bins
            digitized = np.digitize(np.array(stats_lines['counter']['time']), bins=bins)
            bins_count, _ = np.histogram(np.array(stats_lines['counter']['time']), bins=bins)
                
            # Collect median (50%), and 1sigma around that (16% and 84%)
            median_array = []
            median_upper = []
            median_lower = []
            for bin_i in np.arange(1, len(bins_count)+1, 1):
                # extract values in current bin
                mask = digitized == bin_i
                current_time   = np.array(stats_lines['counter']['time'])[mask]
                current_mass   = np.array(stats_lines['counter']['mass'])[mask]
                            
                median_array.append(np.percentile(current_mass, 50))
                median_upper.append(np.percentile(current_mass, 100-use_percentiles))
                median_lower.append(np.percentile(current_mass, use_percentiles))
            
            stats_growth['counter']['median'] = np.log10(median_array)
            stats_growth['counter']['upper'] = np.log10(median_upper)
            stats_growth['counter']['lower'] = np.log10(median_lower)
                       
            #----------
            # plot
            ax_cnt.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100)
            ax_cnt.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100)
            ax_cnt.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100)
            
            ax_combi.plot(plot_bins, np.log10(np.array(median_array)), color=line_color, ls='-', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_upper)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
            ax_combi.plot(plot_bins, np.log10(np.array(median_lower)), color=line_color, ls='--', lw=1, zorder=100, alpha=0.8)
    
        print('Medians with 0.125 bins before/after/delta with min %s Gyr:' %min_window_size)
        print('  aligned:     %.4f\t\t%.4f\t\tdelta %.4f dex' %(stats_growth['aligned']['median'][0], stats_growth['aligned']['median'][-1], (stats_growth['aligned']['median'][-1]-stats_growth['aligned']['median'][0])))
        print('  misaligned:  %.4f\t\t%.4f\t\tdelta %.4f dex' %(stats_growth['misaligned']['median'][0], stats_growth['misaligned']['median'][-1], (stats_growth['misaligned']['median'][-1]-stats_growth['misaligned']['median'][0])))
        print('  counter:     %.4f\t\t%.4f\t\tdelta %.4f dex' %(stats_growth['counter']['median'][0], stats_growth['counter']['median'][-1], (stats_growth['counter']['median'][-1]-stats_growth['counter']['median'][0])))
    
    #-----------
    ### other
    #plt.tight_layout()
        
    #-----------
    ### General formatting
    # Axis labels
    ax_co.set_xlim(-0.05, min_window_size+0.05)
    ax_co.set_xticks(np.arange(0, min_window_size+0.05, 0.1))
    ax_cnt.set_xlabel('Time [Gyr]')
    ax_co.set_ylim(target_bhmass-(1.5*target_bhmass_err), np.log10(median_upper)[-1]+0.2)
    fig.supylabel(r'log$_{10}$ most massive $M_{\mathrm{BH}}(<r_{50}$) [M$_{\odot}]$', fontsize=9)
    ax_co.minorticks_on()
    ax_co.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    ax_co.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')
    
    #-----------
    ### Annotation
    ax_mis.axvline(x=0, ymin=0, ymax=1, color='grey', zorder=999, lw=0.7, ls='--', alpha=0.7)
    ax_mis.text(-0.012, target_bhmass+0.1, 'last stable', color='grey', alpha=0.7, fontsize=6, rotation=90, rotation_mode='anchor')
    
    
    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_colors.append('k')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_labels.append('misaligned')
    legend_colors.append('orangered')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_labels.append('counter-rotating')
    legend_colors.append('dodgerblue')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_combi.legend(handles=legend_elements, labels=legend_labels, loc='best', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_colors.append('k')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_co.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)

    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('misaligned')
    legend_colors.append('orangered')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_mis.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)
    
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('counter-rotating')
    legend_colors.append('dodgerblue')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    ax_cnt.legend(handles=legend_elements, labels=legend_labels, loc='upper right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    ### title
    plot_annotate = '$>%.2f$ Gyr window, target BHmass = %.2f±%.2f'%(min_window_size, target_bhmass, target_bhmass_err) + (plot_annotate if plot_annotate else '')
    ax_combi.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    
    
    #-----------
    ### Savefig       
    
    
    metadata_plot = {'Title': 'medians start/end/delta\nali: %.4f %.4f %.4f\nmis: %.4f %.4f %.4f\ncnt: %.4f %.4f %.4f' %(stats_growth['aligned']['median'][0], stats_growth['aligned']['median'][-1], (stats_growth['aligned']['median'][-1]-stats_growth['aligned']['median'][0]), stats_growth['misaligned']['median'][0], stats_growth['misaligned']['median'][-1], (stats_growth['misaligned']['median'][-1]-stats_growth['misaligned']['median'][0]), stats_growth['counter']['median'][0], stats_growth['counter']['median'][-1], (stats_growth['counter']['median'][-1]-stats_growth['counter']['median'][0]))}
    
                
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt)
        
        plt.savefig("%s/BH_stacked_evolution/%sstacked_BH%s_M%s_t%s_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', target_bhmass, target_stelmass, min_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_stacked_evolution/%swstacked_BH%s_M%s_t%s_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', target_bhmass, target_stelmass, min_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
    

#--------------------------------
# x-y of BH mass at start vs delta mass, coloured for counter/misaligned/aligned -> a few of these for 500 mya (trim longer aligend/counter ones), 0.75 gya, 1 gyr. Will pick random min_window_segment per aligned/counter entry
def _BH_deltamass_in_window(BHmis_tree = None, BHmis_input = None, BHmis_summary = None, plot_annotate = None, savefig_txt_in = None,
                          #==============================================
                          # Plot options
                            target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                              window_err      = 0.05,           # [ +/- Gyr ] trim
                            add_histograms    = False,   
                            add_growth_ratios = True,
                          # Sample refinement
                            run_refinement = False,
                              #use_hmr_general_sample = '2.0',   # [ 1.0 / 2.0 / aperture]
                              # basic properties
                              min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                              min_bhmass   = None,      max_bhmass   = None,
                              min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                              min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                              # Mergers, looked for within range considered +/- halfwindow
                              use_merger_criteria = False,
                          #==============================================
                          showfig       = True,
                          savefig       = False,    
                            file_format   = 'pdf',
                            savefig_txt = '',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
                          #-----------------------------
                          debug = False):
                      
    #-----------------------------------
    use_CoP_BH                  = BHmis_input['use_CoP_BH']
    apply_at_start = True
    target_bhmass = False
    target_stelmass = False
    
    # Establish sub-sample we wish to focus on
    if run_refinement:
        BH_subsample, BH_subsample_summary = _refine_BHmis_sample(BHmis_tree = BHmis_tree, BHmis_input = BHmis_input, BHmis_summary = BHmis_summary,
                                                                  apply_at_start = apply_at_start,  
                                                                  min_stelmass = min_stelmass,      max_stelmass = max_stelmass,       
                                                                  min_bhmass   = min_bhmass,        max_bhmass   = max_bhmass,
                                                                  min_sfr      = min_sfr,           max_sfr      = max_sfr,        
                                                                  min_ssfr     = min_ssfr,          max_ssfr     = max_ssfr,    
                                                                  use_merger_criteria = use_merger_criteria)
    else:
        print('==================================================================')
        print('No refinement -> BH_subsample = clean BH_sample from above\n')
        BH_subsample = BHmis_tree
        BH_subsample_summary = BHmis_summary
        BH_subsample_summary.update({'total_sub': BH_subsample_summary['clean_sample']})
    
    #===================================================================================================
    # Go through sample
    stelmass_plot  = []
    time_delta_plot = []
    bhmass_start_plot = []
    bhmass_delta_plot = []
    bhmdot_plot    = []
    sfr_plot       = []
    ssfr_plot      = []
    kappa_plot     = []
    trelax_plot    = []
    trelax_ID      = []
    duration_plot  = []
    state_plot     = []
    ID_plot        = []
    for galaxy_state in ['aligned', 'misaligned', 'counter']:
        for ID_i in BH_subsample['%s' %galaxy_state].keys():
            
            # check if duration is shorter than window size
            if BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'] < (target_window_size-window_err):
                continue
            
            #-----------------
            # Misaligned sample
            if galaxy_state == 'misaligned':
                
                check_index = BH_subsample['%s' %galaxy_state]['%s' %ID_i]['index_s']
                time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[check_index] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                
                # check if last snapshot in stable regime (index_s) is within target limits + window_size
                if (time_check >= (target_window_size - window_err)):
                    
                    index_start = check_index
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where((duration_array > (target_window_size-window_err)) & (duration_array < (target_window_size+window_err)))[0]
                    
                    # check if the next snapshot is not too far away
                    if len(index_stop_array) > 0:
                        index_stop = random.choice(index_stop_array) + 1
                        
                        #---------------------------
                        # Add evolution entry and append
                        time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                        time_delta_plot.append(time_axis[-1])
                        if use_CoP_BH:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_plot.append((mass_axis[-1] - mass_axis[0])/time_axis[-1])
                        else:
                            mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                            bhmass_start_plot.append(mass_axis[0])
                            bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                            bhmdot_plot.append((mass_axis[-1] - mass_axis[0])/time_axis[-1])
                    
                        #--------------------------
                        # Singular values
                        stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                        sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                        ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                        kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
            
                        duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                        trelax_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['relaxation_time'])
                        trelax_ID.append(ID_i)
                        state_plot.append(galaxy_state)
                        ID_plot.append(ID_i)
                    
                    
                else:
                    continue
                
            #-----------------
            # Aligned and counter sample
            else:
                # Consider all indexes
                stelmass_index_length = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])

                check_index_array = []
                for index_i in np.arange(0, len(stelmass_index_length), 1).astype(int):
                    time_check = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[-1]
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_i])
                    
                    # check if within window limits
                    if (time_check >= (target_window_size - window_err)) & (duration_array[-1] <= (target_window_size + window_err)):
                        check_index_array.append(index_i)
                
                # If there exists at least one valid entry, pick random to append min_window_max entries to
                if len(check_index_array) > 0:
                    # Pick random starting point
                    index_start = random.choice(check_index_array)
                    duration_array = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime']) - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    index_stop_array  = np.where(duration_array > (target_window_size-window_err))[0]
                    index_stop = random.choice(index_stop_array) + 1
                                        
                    #---------------------------
                    # Add evolution entry and append
                    time_axis = -1*(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start:index_stop] - np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['Lookbacktime'])[index_start])
                    time_delta_plot.append(time_axis[-1])
                    if use_CoP_BH:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_plot.append((mass_axis[-1] - mass_axis[0])/time_axis[-1])
                    else:
                        mass_axis = np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['bh_mass_alt'])[index_start:index_stop]
                        bhmass_start_plot.append(mass_axis[0])
                        bhmass_delta_plot.append(mass_axis[-1] - mass_axis[0])
                        bhmdot_plot.append((mass_axis[-1] - mass_axis[0])/time_axis[-1])
                
                    #--------------------------
                    # Singular values
                    stelmass_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['stelmass'])[index_start])
                    sfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['sfr'])[index_start])
                    ssfr_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['ssfr'])[index_start])
                    kappa_plot.append(np.array(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['kappa_stars'])[index_start])
        
                    duration_plot.append(BH_subsample['%s' %galaxy_state]['%s' %ID_i]['entry_duration'])
                    state_plot.append(galaxy_state)
                    ID_plot.append(ID_i)
                
                
                          
    # Collect data into dataframe
    df = pd.DataFrame(data={'stelmass': stelmass_plot, 'SFR': sfr_plot, 'sSFR': ssfr_plot, 'Time delta': time_delta_plot, 'BH mass start': bhmass_start_plot, 'BH mass delta': bhmass_delta_plot, 'BH mdot': bhmdot_plot, 'Morphology': kappa_plot, 'State': state_plot, 'window': duration_plot, 'GalaxyIDs': ID_plot})        
    df_co  = df.loc[(df['State'] == 'aligned')]
    df_mis = df.loc[(df['State'] == 'misaligned')]
    df_cnt = df.loc[(df['State'] == 'counter')]
    
    print('Sub-sample in range, with target_window_size median of %.2f±%.2f:' %(target_window_size, window_err))
    print('  aligned:     %s\t%.4f M_bh (first snip)\t%.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta'])))
    print('  misaligned:  %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta'])))
    print('  counter:     %s\t%.4f M_bh             \t%.3f Gyr' %(len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta'])))
    print(' ')
    print('For guide for later analysis:')
    print('Total misaligned: %s' %len(trelax_plot))
    print('Number of trelax >0.3  Gyr: %s' %len([i for i in trelax_plot if i > 0.3]))
    print('Number of trelax >0.4  Gyr: %s' %len([i for i in trelax_plot if i > 0.4]))
    print('Number of trelax >0.5  Gyr: %s' %len([i for i in trelax_plot if i > 0.5]))
    print('Number of trelax >0.6  Gyr: %s' %len([i for i in trelax_plot if i > 0.6]))
    print('Number of trelax >0.7  Gyr: %s' %len([i for i in trelax_plot if i > 0.7]))
    print('Number of trelax >0.8  Gyr: %s' %len([i for i in trelax_plot if i > 0.8]))
    print('Number of trelax >0.9  Gyr: %s' %len([i for i in trelax_plot if i > 0.9]))
    print('Number of trelax >1.0  Gyr: %s' %len([i for i in trelax_plot if i > 1.0]))
    
    
    
    color_dict = {'aligned':'darkgrey', 'misaligned':'orangered', 'counter':'dodgerblue'}
    
    #---------------------------  
    # Figure initialising
    if add_histograms:
        
        fig = plt.figure(figsize=(10/3, 10/3))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)
        # Create the Axes.
        axs = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=axs)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=axs)

        #---------------------
        ### Plot histograms
        for state_i in color_dict.keys():

            df_state = df.loc[df['State'] == state_i]
            
            ax_histx.hist(np.log10(df_state['BH mass start']), bins=np.arange(6, 11, 0.25), log=True, facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=1)
            ax_histx.hist(np.log10(df_state['BH mass start']), bins=np.arange(6, 11, 0.25), log=True, facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
            ax_histy.hist(np.log10(df_state['BH mass delta']), bins=np.arange(0, 10.1, 0.25), log=True, orientation='horizontal', facecolor='none', linewidth=1, edgecolor=color_dict[state_i], histtype='step', alpha=0.8)
            ax_histy.hist(np.log10(df_state['BH mass delta']), bins=np.arange(0, 10.1, 0.25), log=True, orientation='horizontal', facecolor=color_dict[state_i], linewidth=1, edgecolor='none', alpha=0.1)
            
            
            #-------------
            # Formatting
            ax_histy.set_xlabel('Count')
            ax_histx.set_ylabel('Count')
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histx.set_yticks([1, 10, 100, 1000])
            ax_histx.set_yticklabels(['', '$10^1$', '', '$10^3$'])
            ax_histy.set_xticks([1, 10, 100, 1000])
            ax_histy.set_xticklabels(['', '$10^1$', '', '$10^3$'])    
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10/3, 3], sharex=False, sharey=False) 
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
    
    #--------------
    # scatter
    axs.scatter(np.log10(df['BH mass start']), np.log10(df['BH mass delta']), s=6, c=[color_dict[i] for i in df['State']], edgecolor='k', marker='.', linewidths=0, alpha=0.8, zorder=-2)

    #--------------
    # annotation
    if add_growth_ratios:
        axs.text(9.3, 9, '$1$', color='k', fontsize=6, rotation=22, rotation_mode='anchor')
        axs.plot([5, 10], [5, 10], ls='--', lw=0.5, color='k')

        axs.text(9.3, 8, '$0.1$', color='grey', fontsize=6, rotation=22, rotation_mode='anchor')
        axs.plot([5, 10], [4, 9], ls='--', lw=0.5, color='grey')
        
        axs.text(9.3, 7, '$0.01$', color='silver', alpha=0.7, fontsize=6, rotation=22, rotation_mode='anchor')
        axs.plot([5, 10], [3, 8], ls='--', lw=0.5, color='silver')
        
        axs.text(9.3, 6, '$0.001$', color='lightgrey', alpha=0.7, fontsize=6, rotation=22, rotation_mode='anchor')
        axs.plot([5, 10], [2, 7], ls='--', lw=0.5, color='lightgrey')


    #-----------
    ### title
    plot_annotate = 'target window %.2f±%.2f Gyr window'%(target_window_size, window_err) + (plot_annotate if plot_annotate else '')
    if add_histograms:
        ax_histx.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
    else:
        axs.set_title(r'%s' %(plot_annotate), size=7, loc='left', pad=3)
        
    

    #-----------
    ### General formatting
    # Axis labels
    axs.set_xlim(5.8, 10)
    axs.set_ylim(0, 10)
    axs.set_xlabel(r'log$_{10}$ $M_{\mathrm{BH,initial}}$ [M$_{\odot}]$')
    #axs.set_ylabel('BH mass [Msun]')
    axs.set_ylabel(r'log$_{10}$ $\Delta M_{\mathrm{BH}}$ [M$_{\odot}]$')
    #axs.minorticks_on()
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='major')
    #axs.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True, which='minor')

    #-----------
    ### Legend
    legend_elements = []
    legend_labels = []
    legend_colors = []
    legend_labels.append('aligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('grey')
    legend_labels.append('misaligned')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('orangered')
    legend_labels.append('counter-rotating')
    legend_elements.append(Line2D([0], [0], marker=' ', color='w'))
    legend_colors.append('dodgerblue')
    axs.legend(handles=legend_elements, labels=legend_labels, loc='lower right', frameon=False, labelspacing=0.1, labelcolor=legend_colors, handlelength=0, ncol=1)


    #-----------
    # other
    plt.tight_layout()

    #-----------
    ### Savefig        
    
    metadata_plot = {'Title': 'sample/median M_bh/median window\nali: %s %.4f %.3f Gyr\nmis: %s %.4f %.3f Gyr\ncnt: %s %.4f %.3f Gyr' %(len(df_co['GalaxyIDs']), np.median(np.log10(df_co['BH mass start'])), np.median(df_co['Time delta']), len(df_mis['GalaxyIDs']), np.median(np.log10(df_mis['BH mass start'])), np.median(df_mis['Time delta']), len(df_cnt['GalaxyIDs']), np.median(np.log10(df_cnt['BH mass start'])), np.median(df_cnt['Time delta']))}
                   
    if savefig:
        savefig_txt_save = savefig_txt_in + ('_' + input('\n  -> Enter savefig_txt:   ') if savefig_txt == 'manual' else savefig_txt) + ('hist' if add_histograms else '')
    
        plt.savefig("%s/BH_deltamass/%sbhmass_delta_window%s_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format), metadata=metadata_plot, format=file_format, bbox_inches='tight', dpi=600)    
        print("\n  SAVED: %s/BH_deltamass/%sbhmass_delta_window%s_clean%s_subsample%s_%s.%s" %(fig_dir, 'L100_', target_window_size, BHmis_summary['clean_sample'], BH_subsample_summary['total_sub'], savefig_txt_save, file_format)) 
    if showfig:
        plt.show()
    plt.close()
    
    

    #   - PLOT 3: x-y of BH mass at start vs delta mass, coloured for counter/misaligned/aligned -> a few of these for 500 mya (trim longer aligend/counter ones), 0.75 gya, 1 gyr








# def mass_change - row:3, col:5, for 106, 106.5, 107, 107.5, 108, 108.5 +/-0.25 mass ranges and 3 relaxation times, plot for different mass ranges

# def accretion rate - row:3, col:5, for 106, 106.5, 107, 107.5, 108, 108.5 +/-0.1dex, divide by relaxation times such that it is average over [window]





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
                                plot_yaxis = ['bhmass'],
                                  apply_at_start = True,
                                run_refinement = False,
                                showfig = True,
                                savefig = False)"""
"""_BH_sample(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                plot_yaxis = ['window'],
                                  apply_at_start = True,
                                run_refinement = False,
                                showfig = True,
                                savefig = False)"""
                                
# STACKED BH growth
"""for target_bh_mass_i in [6.1, 6.3, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8]:
    _BH_stacked_evolution(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    min_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                      target_bhmass         = target_bh_mass_i,              # [ 10**[] Msun / False ]
                                      target_stelmass       = False,                        # [ 10** Msun / False ]
                                    run_refinement = False,
                                      showfig = True,
                                      savefig = False)"""
"""for target_bh_mass_i in [6.1, 6.3, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8]:
    _BH_stacked_evolution(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    min_window_size   = 0.75,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                      target_bhmass         = target_bh_mass_i,              # [ 10**[] Msun / False ]
                                      target_stelmass       = False,                        # [ 10** Msun / False ]
                                    run_refinement = False,
                                      showfig = True,
                                      savefig = False)"""

# SCATTER x-y mass at start and mass at end after X Gyr
"""_BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    target_window_size   = 0.5,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                      window_err      = 0.05,           # [ +/- Gyr ] trim
                                    run_refinement = False,
                                      showfig = True,
                                      savefig = False)"""
"""_BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    target_window_size   = 0.75,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                      window_err      = 0.075,           # [ +/- Gyr ] trim
                                    run_refinement = False,
                                      showfig = True,
                                      savefig = False)"""
"""_BH_deltamass_in_window(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                    target_window_size   = 1.0,         # [ Gyr ] trim to at least 1 Gyr to allow overlay
                                      window_err      = 0.1,           # [ +/- Gyr ] trim
                                    run_refinement = False,
                                      showfig = True,
                                      savefig = False)"""
                            
      
                            
                            
                            
# delta/MBH histogram
# delta/MBH vs fgas
# delta/MBH vs gas density relation
                
                            
                            
                            
                            
                            
                            