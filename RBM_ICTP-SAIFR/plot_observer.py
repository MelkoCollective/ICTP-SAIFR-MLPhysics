import numpy as np
import matplotlib.pyplot as plt
import argparse
plt.style.use('classic')

def observe_ising2d(args):
    temps=[1.0,1.254,1.508,1.762,2.016,
           2.269,2.524,2.778,3.032,3.286,3.540]

    plt.ion()
    
    L=4
    T=args.T
    nH = args.nH
    
    for i in range(len(temps)):
        if temps[i] == T:
            t_index=i
            break
    
    plt.figure(figsize=(12,9), facecolor='w', edgecolor='k')
    lw = 1
    lw_exact = 2
    mS = 6
 
    nameMC  = 'data_ising2d/observables/MC_ising2d_L'
    nameMC += str(L)
    nameMC += '_Observables.txt'
    fileMC = open(nameMC,'r')
    header = fileMC.readline().lstrip('#').split()
    dataMC = np.loadtxt(fileMC)
 
    while(True):
         
        plt.clf()
        # Open MC data file
        exact_energy = dataMC[t_index,header.index('E')]
        exact_magnetization = dataMC[t_index,header.index('M')]
        
        observer = np.loadtxt('data_ising2d/observables/training_observer.txt')
        x = [i for i in range(observer.shape[0])]
     
        plt.subplot(221)
        plt.ylabel('$KL$',fontsize=25)
        plt.plot(x,observer[:,0],color='red',marker='o',markersize=mS,linewidth=lw)
        
        plt.subplot(222)
        plt.ylabel('$<NLL>$',fontsize=25)
        plt.plot(x,observer[:,1],color='red',marker='o',markersize=mS,linewidth=lw)
     
        plt.subplot(223)
        plt.ylabel('$<E>$',fontsize=25)
        plt.plot(x,observer[:,2],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.axhline(y=exact_energy, xmin=0, xmax=x[-1], linewidth=2, color = 'k',label='Exact') 
        
        plt.subplot(224)
        plt.ylabel('$<M>$',fontsize=25)
        plt.plot(x,observer[:,3],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.axhline(y=exact_magnetization, xmin=0, xmax=x[-1], linewidth=2, color = 'k',label='Exact') 
        
        plt.tight_layout()
        plt.pause(0.05)
    
    #plt.show()


if __name__ == "__main__":
    
    """ Read command line arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument('-nH',type=int,default=4)
    parser.add_argument('-L',type=int,default=4)
    parser.add_argument('-T',type=float)
    parser.add_argument('-B',type=float)
     
    args = parser.parse_args()

    observe_ising2d(args)



