import matplotlib.pyplot as plt

def graphformat(size1, size2, size3, size4, size5, width, height):
    # General graph font size formatting
    plt.rc('font', size=size1)          # controls default text sizes
    
    plt.rc('figure', titlesize=size2)   # fontsize of the figure title
    
    plt.rc('axes', titlesize=size3)     # fontsize of the axes title
    plt.rc('axes', labelsize=size3)     # fontsize of the x and y labels
    
    plt.rc('xtick', labelsize=size4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size4)    # fontsize of the tick labels

    plt.rc('legend', fontsize=size5)    # legend fontsize

    plt.rc('figure', figsize=(width, height))  # figure size [inches]