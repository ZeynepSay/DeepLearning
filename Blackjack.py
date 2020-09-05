# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import random # np.random is slow

def step(s, a):
    if a ==  1 : #"hit":
        card, color = draw()
        player = s["player"] + color*card
        dealer = s["dealer"]
        terminal = 0
        reward = 0
        if player > 21 or player < 1:
            terminal = 1
            reward = -1
    if a == 0: #"stick":
        player = s["player"]
        dealer = s["dealer"]
        while dealer < 17 and dealer > 0:
            card, color = draw()
            dealer = dealer + card*color
        if dealer > 21 or dealer < player:
            reward = 1
        elif dealer > player:
            reward = -1
        elif dealer == player:
            reward = 0
        else:
            raise ValueError('Error in step.')
        terminal = 1
    return {"player": player, "dealer": dealer, "terminal": terminal}, reward

def draw():
    #np.random is way too slow (10x slower)
#    card = np.random.randint(1,11)
#    color = np.random.choice([-1, 1], p=[1/3, 2/3])
    card = int(random.random()*10+1)
    color = 2*int(random.random()<(2/3))-1
    return card, color

def policy(Qs, epsilon):
#    if np.random.random() > epsilon: #greedy action
#        return np.argmax(Qs)
#    return np.random.choice([0,1]) #epsilon action
    if random.random() > epsilon: #greedy action
        return np.argmax(Qs)
    return int(random.random()<0.5) #epsilon action

def MonteCarlo(Nepisodes):
    Q = np.zeros((22,11,2)) # Q(player, dealer, 0(stick)/1(hit))
    N = np.zeros((22,11,2)) # number of times state/action visited
    gamma = 1
    N0 = 100
    
    for i in range(Nepisodes):
        # Starting state
        s={}
        s["player"], _  =  draw()
        s["dealer"], _  =  draw()
        s["terminal"] = 0
        
        # Generate episode
        episode = []
        reward = []
        while s["terminal"] == 0:
            Nst = np.sum(N[s["player"],s["dealer"],:])
            epsilon = N0/( N0 + Nst )
            a = policy(Q[s["player"],s["dealer"],:], epsilon)
            sp, r = step(s,a)
            episode.append((s["player"],s["dealer"], a))
            reward.append(r) # reward episode[t+1]
            s = sp
            
        # for each step in episode t = T-1, T-2, ..., 0
        while episode != []: 
            state = episode.pop()
            r = reward.pop()
            G = 0
            if state not in episode: # first visit MC
                N[state] += 1
                alpha = 1/N[state]
                G = gamma*G + r
                Q[state] += alpha*(G-Q[state])
    return Q


def Sarsa(Nepisodes, lambd, returnMSEhistoryQ=None):
    MSEhistory=[] # can be ignored along with  ^
    Q = np.zeros((22,11,2)) # Q(player, dealer, 0(stick)/1(hit))
    N = np.zeros((22,11,2)) # number of times state/action visited
    gamma = 1
    N0 = 100
    
    for i in range(Nepisodes):
        # Starting state
        s={}
        s["player"], _  =  draw()
        s["dealer"], _  =  draw()
        s["terminal"] = 0
        
        # eligibilty trace
        E = np.zeros((22,11,2))
        
        # play episode
        while s["terminal"] == 0:
            Nst = np.sum(N[s["player"],s["dealer"],:])
            epsilon = N0/( N0 + Nst )

            a = policy(Q[s["player"],s["dealer"],:], epsilon)
            sp, r = step(s,a)
            
            N[s["player"],s["dealer"],a] += 1
            
            if  sp["terminal"] == 0:
                ap = policy(Q[sp["player"],sp["dealer"],:], epsilon)
                delta = r + gamma*Q[sp["player"],sp["dealer"],ap] - Q[s["player"],s["dealer"],a]
            else:
                delta = r - Q[s["player"],s["dealer"],a]
                
            E[s["player"],s["dealer"],a] +=1
            
            #alpha = 1/N(s,a)
            Q[E!=0] += delta*E[E!=0]/N[E!=0]
            E = gamma*lambd*E
            
            s = sp
            
        ## can be ignored
        if returnMSEhistoryQ is not None:
            se = (Q-returnMSEhistoryQ)**2
            MSEhistory.append(se.sum())
    if returnMSEhistoryQ is not None:
        return Q, np.array(MSEhistory)
    ##
        
    return Q

def wirePlot(V, title=None, savef=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(1, 11), np.arange(1, 22))
    ax.set_xlim(1, 10)
    ax.set_ylim(1, 21)
    ax.set_xticks(np.arange(1,11,3))
    ax.set_yticks(np.arange(1,22,4))
    ax.set_zticks(np.arange(-0.2,1.4,0.4))
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('V*')
    ax.plot_wireframe(x, y, V)
    if title is not None:
        plt.title(title)
    if savef is not None:
        plt.savefig(savef,transparent=True)

def main(MC_numEpisodes = int(1e6), TD_numEpisodes = int(1e6), 
         TD_lambda = 0.5, TD_MSE_num_episodes = int(1e3) ):
    import time
    import os
    
    # For consistant reproduction
    np.random.seed(42)
    random.seed(42)

    # Save to save time
    MCsavepath = "MC_Q_%.0e.npy"%MC_numEpisodes
    TDsavepath = "Sarsa(%1.1f)_Q_%.0e.npy"%(TD_lambda, TD_numEpisodes)
    if os.path.exists(MCsavepath):
        MC_Q = np.load(MCsavepath)
        print(MCsavepath+" loaded from last run!")
    else:
        tic = time.time()
        MC_Q = MonteCarlo(MC_numEpisodes)
        toc = time.time()
        np.save(MCsavepath, MC_Q)
        print("Monty Carlo over %.0e episodes, runtime: %ds"%(
                MC_numEpisodes, toc-tic))
    if os.path.exists(TDsavepath):
        TD_Q = np.load(TDsavepath)
        print(TDsavepath+" loaded from last run!")
    else:
        tic = time.time()
        TD_Q = Sarsa(TD_numEpisodes, 0)
        toc = time.time()
        np.save(TDsavepath, TD_Q)
        print("Sarsa(%1.1f) over %.0e episodes, runtime: %ds"%(
                TD_lambda, TD_numEpisodes, toc-tic))
    
    # plot value function for MC
    MC_Vstar = np.max(MC_Q[1:,1:,:],axis=2)
    wirePlot(MC_Vstar, "MC %.0e episodes"%MC_numEpisodes, MCsavepath+".pdf")
    
    # plot value function for Sarsa
    TD_Vstar = np.max(TD_Q[1:,1:,:],axis=2)
    wirePlot(TD_Vstar, "Sarsa(%1.1f) %.0e episodes"%(
            TD_lambda, TD_numEpisodes), TDsavepath+".pdf")
    
    # calculate MSE of Sarsa(λ) vs MC for λ = [0, .1, .2, ..., 1]
    # For λ = 0 and λ = 1 plot learning curve of MSE against episode number
    lambds = np.arange(0,1,.1)
    MSE = []
    Q0, MSE0 = Sarsa(TD_MSE_num_episodes, 0, returnMSEhistoryQ=MC_Q)  #λ = 0 
    MSE.append(MSE0[-1])
    for lambd in lambds[1:-1]: # λ = [.1, .2, ..., .9]
        TD_Q = Sarsa(TD_MSE_num_episodes, lambd)
        se = (TD_Q-MC_Q)**2
        MSE.append(se.sum())
    Q1, MSE1 = Sarsa(TD_MSE_num_episodes, 1, returnMSEhistoryQ=MC_Q)  #λ = 1
    MSE.append(MSE1[-1])

    # plot MSE results
    plt.figure()
    plt.plot(lambds,MSE,'o-')
    plt.title(r"Sarsa($\lambda$) error")
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Squared-error')
    plt.show()
    plt.savefig('MSEvsLAMBDA.pdf',transparent=True)
    
    # For λ = 0 and λ = 1 plot learning curve of MSE against episode number
    episode = np.arange(1,MSE0.shape[0]+1)
    #wirePlot(np.max(Q0[1:,1:,:],axis=2), r"MSE Sarsa(0)")
    #wirePlot(np.max(Q0[1:,1:,:],axis=2), r"MSE Sarsa(1)")
    plt.figure()
    plt.plot(episode,MSE0)
    plt.plot(episode,MSE1)
    plt.legend(('Sarsa(0)', 'Sarsa(1)'), loc='upper right')
    
    plt.title(r"Sarsa($\lambda$) error during training")
    plt.xlabel("Episode")
    plt.ylabel('Squared-error')
    plt.show()
    plt.savefig('MSEvsEpi.pdf',transparent=True)

# only run if this file was not imported
if __name__ == "__main__":
    main()