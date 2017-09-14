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
 
    nameMC  = '../data/ising2d/observables/MC_ising2d_L'
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
        
        observer = np.loadtxt('../data/ising2d/observables/training_observer.txt')
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

def observe_tfim1d(args):
    
    fields = [0.2,0.4,0.6,0.8,1.0,
              1.2,1.4,1.6,1.8,2.0]

    plt.ion()
    
    L=10
    B=args.B
    nH = 10
    
    for i in range(len(fields)):
        if fields[i] == B:
            b_index=i
            break
    # Open MC data file
    nameMC  = '../data/tfim1d/observables/ed_tfim1d_L'
    nameMC += str(L)
    nameMC += '_Observables.txt'
    fileMC = open(nameMC,'r')
    header = fileMC.readline().lstrip('#').split()
    dataMC = np.loadtxt(fileMC)
    exact_Sz = dataMC[b_index,header.index('<|Sz|>')]
    exact_Sx = dataMC[b_index,header.index('<Sx>')]
    exact_E  = dataMC[b_index,header.index('E')] 
    #nameCorr = '../data/tfim1d/observables/ed_tfim1d_L'+str(L)
    #nameCorr += '_correlations.txt'
    #fileCorr = open(nameCorr,'r')
    #
    #ed_corr_full = np.loadtxt(nameCorr)
    #ed_corr_crit = np.zeros((L,L))
    #for i in range(L):
    #    for j in range(L):
    #        ed_corr_crit[i][j] = ed_corr_full[b_index*L+i][j]
 
    plt.figure(figsize=(12,9), facecolor='w', edgecolor='k')
    lw = 1
    lw_exact = 2
    mS = 6



    while(True):
         
        plt.clf()
        observer = np.loadtxt('../data/tfim1d/observables/training_observer.txt')
        #correlations= np.loadtxt('../data/tfim1d/observables/training_observer_corr.txt') 
 
        x = [i for i in range(observer.shape[0])]
    
        #plt.subplot(221)
        #xg = np.asarray(range(10))
        #yg = np.asarray(range(10))
        #X,Y = np.meshgrid(xg,yg)
        #plt.contourf(X,Y,ed_corr_crit,20, cmap=plt.cm.rainbow)
        #levels = [0.000,0.0005,0.001,0.005,0.01,0.02,0.03,0.04,0.06,0.08,0.1,0.2]
        #lineCol =['#1a1a1a','#1a1a1a','#1a1a1a','#1a1a1a','#1a1a1a','#1a1a1a',
        #          '#1a1a1a','#1a1a1a','#262626','#333333','#404040','#4d4d4d','#595959']
        #plt.contour(X,Y,ed_corr_crit,levels,colors=lineCol,linewidths=1.5)
        #
        #plt.subplot(222)
        #plt.contourf(X,Y,correlations,20, cmap=plt.cm.rainbow)
        #plt.contour(X,Y,correlations,levels,colors=lineCol,linewidths=1.5)
 
        plt.subplot(221)
        plt.ylabel('O',fontsize=25)
        plt.plot(x,observer[:,0],color='red', marker='o',markersize=mS,linewidth=lw)
        plt.ylim([0,1.05])

        plt.subplot(222)
        plt.ylabel('$<H>$',fontsize=25)
        plt.plot(x,observer[:,2],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.axhline(y=exact_E, xmin=0, xmax=x[-1], linewidth=2, color = 'k',label='Exact') 
 
        plt.subplot(223)
        plt.ylabel('$<|S_z|>$',fontsize=25)
        plt.plot(x,observer[:,3],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.axhline(y=exact_Sz, xmin=0, xmax=x[-1], linewidth=2, color = 'k',label='Exact') 
        plt.yticks([0,0.25,0.5,0.75,1.0]) 
        plt.subplot(224)
        plt.ylabel('$<S_x>$',fontsize=25)
        plt.plot(x,observer[:,4],color='red',marker='o',markersize=mS,linewidth=lw)
        plt.axhline(y=exact_Sx, xmin=0, xmax=x[-1], linewidth=2, color = 'k',label='Exact') 
        plt.yticks([0,0.25,0.5,0.75,1.0]) 
 
        plt.tight_layout()
        plt.pause(0.1)
    
    plt.show()


if __name__ == "__main__":
    
    """ Read command line arguments """
    parser = argparse.ArgumentParser()


    parser.add_argument('model',type=str)
    parser.add_argument('-nH',type=int,default=4)
    parser.add_argument('-L',type=int,default=4)
    parser.add_argument('-T',type=float)
    parser.add_argument('-B',type=float)
     
    args = parser.parse_args()

    if args.model == 'ising2d':
        observe_ising2d(args)

    if args.model == 'tfim1d':
        observe_tfim1d(args)


