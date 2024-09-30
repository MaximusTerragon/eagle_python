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
                          add_seed_limit    = 5*(1.48*10**5),       # [ False / value ] seed mass = 1.48*10**5, quality cut made at 5x
                          add_observational = False,                # not working
                          add_bulge_ratios  = True,
                          add_histograms    = True, 
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
                        savefig_txt = 'better_format',            # [ '' / 'any text' / 'manual' ] 'manual' will prompt txt before saving
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
    trelax_plot    = []
    duration_plot  = []
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
            axs.text(10.75, 5.9, 'sample limit', color='k', alpha=0.7, fontsize=6)
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
        legend_labels.append('counter')
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
        legend_labels.append('counter (window)')
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
        
        
    


# Create sample of durations (misaligned/counter) to see what we have to work with

# update stacked - within certain mass ranges, look at ~0.5 Gyr, ~0.7 Gyr, ~1 Gyr mass increases, plot all and plot medians for each state

# def mass_change - row:3, col:5, for 106, 106.5, 107, 107.5, 108, 108.5 +/-0.25 mass ranges and 3 relaxation times, plot for different mass ranges

# def accretion rate - row:3, col:5, for 106, 106.5, 107, 107.5, 108, 108.5 +/-0.1dex, divide by relaxation times


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
load_BHmis_tree_in = '_CoPFalse_window0.5_trelax0.3___05Gyr___'
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
                                min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                                min_bhmass   = None,      max_bhmass   = None,        # seed = 1.48*10**5
                                min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                                min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                                use_merger_criteria = False,
                              showfig = True,
                              savefig = False)"""
_BH_sample(BHmis_tree=BHmis_tree, BHmis_input=BHmis_input, BHmis_summary=BHmis_summary, plot_annotate=plot_annotate_in, savefig_txt_in=savefig_txt_in,
                                plot_yaxis = ['window'],
                                  apply_at_start = True,
                                min_stelmass = None,      max_stelmass = None,        # [ 10**9 / Msun ]
                                min_bhmass   = None,      max_bhmass   = None,        # seed = 1.48*10**5
                                min_sfr      = None,      max_sfr      = None,        # [ Msun/yr ] SF limit of ~ 0.1
                                min_ssfr     = None,      max_ssfr     = None,        # [ /yr ] SF limit of ~ 1e-10-11
                                use_merger_criteria = False,
                              showfig = True,
                              savefig = True)
                              
                              

                              

      





                            
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


                
                            
                            
                            
                            
                            
                            