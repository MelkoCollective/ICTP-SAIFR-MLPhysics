import numpy as np
import matplotlib.pyplot as plt
import argparse
import math as m
plt.style.use('classic')

def plot_ising2d_observables(L):

    fig = plt.figure(figsize=(12,9), facecolor='w', edgecolor='k')

    # Plot properties
    colors = ["#43FF7C","#3425FF","#FF5858","#68C3FF","#AA68FF","#FFC368"]
    mark  = '^'
    lineW = 2.0
    lineS = '-'
    markerSize = 12
   
    # RBMs used in the plot (sorted with number of hidden nodes) 
    hidden = [4,16,64]
    
    # Open MC data file
    nameMC  = 'data_ising2d/observables/MC_ising2d_L'
    nameMC += str(L)
    nameMC += '_Observables.txt'
    fileMC = open(nameMC,'r')

    # Get observables names
    header = fileMC.readline().lstrip('#').split()
    dataMC = np.loadtxt(fileMC)

    x = [i for i in range(len(dataMC))]
    observables = ['E','M','C','S']

    sb = 221

    # Plot different thermodynamics observables
    for obs in observables:
        plt.subplot(sb)
        sb += 1

        error = 'd' + obs
        data = dataMC[:,header.index(obs)]
        plt.plot(x,data,color="k", 
            marker='o',
            linewidth=lineW, 
            linestyle=lineS,
            markersize = markerSize)
        
        # Plot from RBMs with different number of hidden nodes
        for i, n_h in enumerate(hidden):
        
            nameRBM = 'data_ising2d/observables/RBM_nH'
            nameRBM += str(n_h) + '_ising2d_L'
            nameRBM += str(L) + '_Observables.txt'
            fileRBM = open(nameRBM,'r')
            header  = fileRBM.readline().lstrip('#').split()
            dataRBM = np.loadtxt(fileRBM)
            data = dataRBM[:,header.index(obs)]
            err  = dataRBM[:,header.index(error)]
            lab = '$n_h$=' 
            lab += str(n_h)
            
            plt.errorbar(x,data,yerr=err,
                    color = colors[i],
                    linewidth = lineW,
                    linestyle = lineS,
                    marker = mark,
                    markersize = markerSize,
                    label =lab)
    
        # Set Label and Tickes
        plt.xlabel('$T$', fontsize=20)
        plt.xticks([0,5,10],[1.0,2.269,3.54],fontsize=15)
        plt.xlim([-0.3,10.3])
        ylab = '$' + obs + '$'
        plt.ylabel(ylab,fontsize=20)

    plt.legend(loc='upper left')    
    plt.tight_layout()
    plt.show()
    #savefig('observables.pdf', format='pdf', dpi=1000)


if __name__ == "__main__":
    
    """ Read command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument('-L',type=int)
    parser.add_argument('-T',type=float)
    parser.add_argument('-B',type=float)
     
    par = parser.parse_args()

    plot_ising2d_observables(par.L)

