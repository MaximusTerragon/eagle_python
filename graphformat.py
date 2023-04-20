import matplotlib as plt

def set_rc_params(mult=1):
    
    # Add font
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    plt.rcParams['font.family'] = 'Latin Modern Roman'
    #plt.rcParams["font.family"] = "DeJavu Serif"
    #plt.rcParams['font.serif'] = 'Times New Roman'
    #plt.rcParams['font.serif'] = ['Times New Roman']
    
    plt.rcParams.update({'font.size': 20*mult})
    plt.rcParams['legend.fontsize'] = 17.5*mult
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.labelsize'] = 20*mult
    plt.rcParams['ytick.labelsize'] = 20*mult
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    #plt.rcParams['ytick.right'] = True
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    #params = {'mathtext.default': 'regular'}
    #plt.rcParams.update(params)
    plt.rcParams['axes.labelsize'] = 20*mult
    
set_rc_params(0.9) 




"""def graphformat(size1, size2, size3, size4, size5, width, height):
    
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # General graph font size formatting
    plt.rc('font', size=size1)          # controls default text sizes
    
    plt.rc('figure', titlesize=size2)   # fontsize of the figure title
    
    plt.rc('axes', titlesize=size3)     # fontsize of the axes title
    plt.rc('axes', labelsize=size3)     # fontsize of the x and y labels
    
    plt.rc('xtick', labelsize=size4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size4)    # fontsize of the tick labels

    plt.rc('legend', fontsize=size5)    # legend fontsize

    plt.rc('figure', figsize=(width, height))  # figure size [inches]
"""