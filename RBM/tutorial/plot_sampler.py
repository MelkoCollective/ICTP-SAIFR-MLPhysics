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
    nameMC  = '../data/ising2d/observables/MC_ising2d_L'
    nameMC += str(L)
    nameMC += '_Observables.txt'
    fileMC = open(nameMC,'r')
    header = fileMC.readline().lstrip('#').split()
    dataMC = np.loadtxt(fileMC)
    xMC = [i for i in range(len(dataMC))] 
 
    while(True):
         
        plt.clf()
                
        # Load the rbm averages
        observables = np.loadtxt('../data/ising2d/observables/sampler_observer.txt')
        
        # Plot the energy
        plt.subplot(221)
        x = [i for i in range(observables.shape[0])]
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
    
# Plot the average observables for the 1d transverse-field ising model
def observe_tfim1d():
    plt.ion()
    
    # Plot properties
    L=10
    plt.figure(figsize=(12,9), facecolor='w', edgecolor='k')
    lw = 2
    lw_exact = 1
    mS = 9
    mS_exact = 3
    levels = [0.000,0.0005,0.001,0.005,0.01,0.02,0.03,0.04,0.06,0.08,0.1,0.2]
    lineCol =['#1a1a1a','#1a1a1a','#1a1a1a','#1a1a1a','#1a1a1a','#1a1a1a',
              '#1a1a1a','#1a1a1a','#262626','#333333','#404040','#4d4d4d','#595959']
   
    # Load the ed data
    nameMC  = '../data/tfim1d/observables/ed_tfim1d_L'
    nameMC += str(L)
    nameMC += '_observables.txt'
    fileMC = open(nameMC,'r')
    header = fileMC.readline().lstrip('#').split()
    dataMC = np.loadtxt(fileMC)
    nameCorr = '../data/tfim1d/observables/ed_tfim1d_L'+str(L)
    nameCorr += '_correlations.txt'
    fileCorr = open(nameCorr,'r')
    ed_corr_full = np.loadtxt(nameCorr)
    ed_corr_crit = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            ed_corr_crit[i][j] = ed_corr_full[4*L+i][j]
 
    while(True):
         
        plt.clf()
                
        # Load the rbm data
        observables = np.loadtxt('../data/tfim1d/observables/sampler_observer.txt')
        correlations= np.loadtxt('../data/tfim1d/observables/sampler_observer_corr.txt') 
        plt.subplot(221)
        
        # Plot magnetization Z
        data = dataMC[:,header.index('<|Sz|>')]
        xMC = [i for i in range(len(data))] 
        x = [i for i in range(observables.shape[0])]
        plt.plot(xMC,data,color='k',marker='o',linewidth=lw_exact,markersize=mS_exact)
        plt.plot(x,observables[:,0],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.ylabel('$<|S_z|>$',fontsize=25)
        plt.xlabel('$B$',fontsize=25)
        plt.ylim([-0.05,1.05]) 
        # Plot magnetization X
        plt.subplot(222)
        data = dataMC[:,header.index('<Sx>')]
        plt.plot(xMC,data,color='k',marker='o',linewidth=lw_exact,markersize=mS_exact)
        plt.plot(x,observables[:,1],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.ylabel('$<S_x>$',fontsize=25)
        plt.xlabel('$B$',fontsize=25)
        plt.ylim([-0.05,1.05])

        # Plot ed critical ZZ correlations
        plt.subplot(223)
        xg = np.asarray(range(10))
        yg = np.asarray(range(10))
        X,Y = np.meshgrid(xg,yg)
        
        plt.contourf(X,Y,ed_corr_crit,50, cmap=plt.cm.rainbow)
        plt.contour(X,Y,ed_corr_crit,levels,colors=lineCol,linewidths=1.5)
        
        # Plot rbm critical ZZ correlations
        plt.subplot(224)
        plt.contourf(X,Y,correlations,50, cmap=plt.cm.rainbow)
        plt.contour(X,Y,correlations,levels,colors=lineCol,linewidths=1.5)
 
        plt.tight_layout()
        plt.pause(0.1)
    

if __name__ == "__main__":
    
    """ Read command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('model',type=str)
    args = parser.parse_args()

    if args.model == 'ising2d':
        observe_ising2d()

    if args.model == 'tfim1d':
        observe_tfim1d()


