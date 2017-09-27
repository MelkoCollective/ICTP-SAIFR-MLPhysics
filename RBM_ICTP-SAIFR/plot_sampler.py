import numpy as np
import matplotlib.pyplot as plt
import argparse
plt.style.use('classic')

# Plot the average observables for the 2d ising model
def observe_ising2d():
    plt.ion()
    L = 4

    # Plot properties
    lw = 2
    lw_exact = 1
    mS = 9
    mS_exact = 3
    plt.figure(figsize=(12,9), facecolor='w', edgecolor='k')
    
    # Load the MC averages
    nameMC  = 'data_ising2d/observables/MC_ising2d_L'
    nameMC += str(L)
    nameMC += '_Observables.txt'
    fileMC = open(nameMC,'r')
    header = fileMC.readline().lstrip('#').split()
    dataMC = np.loadtxt(fileMC)
    xMC = [i for i in range(len(dataMC))] 
 
    while(True):
         
        plt.clf()
                
        # Load the rbm averages
        observables = np.loadtxt('data_ising2d/observables/sampler_observer.txt')
        
        # Plot the energy
        plt.subplot(221)
        x = [i for i in range(observables.shape[0])]
        print observables
        print x
        plt.plot(x,observables[:,0],color='red',marker='o',markersize=mS,linewidth=lw)
        data = dataMC[:,header.index('E')]
        plt.plot(xMC,data,color='k',marker='o',linewidth=lw_exact,markersize=mS_exact)
        plt.ylim(-2.05,-0.6)
        plt.xlim(-0.1,10.1)
        plt.ylabel('$<E>$',fontsize=25)
        plt.xlabel('$T$',fontsize=25)
 
        # Plot the magnetization
        plt.subplot(222)
        data = dataMC[:,header.index('M')]
        plt.plot(x,observables[:,1],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.plot(xMC,data,color='k',marker='o',linewidth=lw_exact,markersize=mS_exact)
        plt.ylim(0.4,1.0)
        plt.xlim(-0.1,10.1)
        plt.ylabel('$<|M|>$',fontsize=25)
        plt.xlabel('$T$',fontsize=25)
 
        # Plot the specific heat
        plt.subplot(223)
        data = dataMC[:,header.index('C')]
        plt.plot(x,observables[:,2],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.plot(xMC,data,color='k',marker='o',linewidth=lw_exact,markersize=mS_exact)
        plt.ylim(-0.05,1.2)
        plt.xlim(-0.1,10.1) 
        plt.ylabel('$<C_V>$',fontsize=25)
        plt.xlabel('$T$',fontsize=25)
 
        # Plot the susceptibility
        plt.subplot(224)
        data = dataMC[:,header.index('S')]
        plt.plot(x,observables[:,3],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.plot(xMC,data,color='k',marker='o',linewidth=lw_exact,markersize=mS_exact)
        plt.ylim(-0.05,0.55)
        plt.xlim(-0.1,10.1)
        plt.ylabel('$<\chi>$',fontsize=25)
        plt.xlabel('$T$',fontsize=25)

        plt.tight_layout() 
        plt.pause(0.2)

if __name__ == "__main__":
  observe_ising2d()


