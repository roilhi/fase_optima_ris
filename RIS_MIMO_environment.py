import gym
from gym import spaces
import numpy as np
from numpy.random import standard_normal
from scipy.linalg import svd

class RIS_MIMO_env(gym.Env):
    def __init__(self,ntx_antennas,nrx_antennas,n_users,
                 nris_surfaces,nris_elements):
        super(RIS_MIMO_env,self).__init__()
        self.Nt = ntx_antennas
        self.Nr = nrx_antennas
        self.K = n_users
        self.L = nris_surfaces
        self.Ns = nris_elements
        #self.n_episodes = n_episodes
        self.action_dim = (self.Ns,2,self.L) # Nris rows dim=0, real(phi)+im(phi) (2 cols) dim=1, number of elements L dim=2
        self.obs_dim = (self.Ns,(4*self.Nt)+2,self.L)# Nris rows dim=0, 4*Nt columns from Gn and Hnk,+ 2 cols from Phi aciton dim=1, number ob elements L dim=2 
        assert self.Nt == self.Nr*self.K # Nt BS antennas number must be equal to Nr*K users
        self.Gn = None
        self.H_nk = None
        self.phi = None
        self.state = None 
        self.done = None
        self.info = {'episode': None, 'reward': None}
        self.action_space = spaces.Box(low=-1,high=1, shape=self.action_dim, dtype=float)
        self.observation_space = spaces.Box(low=-1, high=1, shape=self.obs_dim, dtype=float)
    def reset(self):
        #self.n_episode = 0
        self.info["reward"] = 0
        self.Gn = (1/np.sqrt(2))*(standard_normal((self.Ns,self.Nt,self.L))+1j*standard_normal((self.Ns,self.Nt,self.L))) #H_SR
        self.H_nk = (1/np.sqrt(2))*(standard_normal((self.Nr*self.K,self.Ns,self.L))+1j*standard_normal((self.Nr*self.K,self.Ns,self.L))) # HNK -H11,H21,...
        Hnk_t = np.transpose(self.H_nk,(1,0,2))
        real_Gn, imag_Gn = np.real(self.Gn),np.imag(self.Gn)
        real_Hnk, imag_Hnk = np.real(Hnk_t), np.imag(Hnk_t)
        state_Channels = np.hstack((real_Gn, imag_Gn,real_Hnk, imag_Hnk))
        init_phi = (1/np.sqrt(2))*(standard_normal((self.Ns,1,self.L))+1j*standard_normal(size=(self.Ns,1,self.L)))
        real_phi_init, im_phi_init = np.real(init_phi), np.imag(init_phi)
        init_action = np.hstack((real_phi_init,im_phi_init))
        self.state = np.concatenate((init_action,state_Channels), axis=1)
        return self.state
    def step(self,action):
        #self.n_episode += 1
        phi_real = action[:,:-1,:]   
        phi_imag = action[:,-1:,:]
        self.phi = phi_real + 1j*phi_imag
        reward = self._compute_reward(self.phi)
        self.info["reward"] = reward
        Hnk_t = np.transpose(self.H_nk,(1,0,2))
        real_Gn, imag_Gn = np.real(self.Gn),np.imag(self.Gn)
        real_Hnk, imag_Hnk = np.real(Hnk_t), np.imag(Hnk_t)
        state_Channels = np.hstack((real_Gn, imag_Gn,real_Hnk, imag_Hnk)) 
        self.state = np.concatenate((action,state_Channels), axis=1)
        #done = self.n_episode >= self.n_episodes
        done = self.done
        return self.state, reward, done, self.info
    def _compute_reward(self,phi):
        #reward = 0
        SINR = np.empty([self.K,], dtype=float)
        Heq = np.empty([self.Nr*self.K,self.Nt,self.L], dtype="complex")
        alpha =  np.empty_like(Heq, dtype="complex")
        for ris in range(self.L):
            H_RD = self.H_nk[:,:,ris]
            H_SR = self.Gn[:,:,ris]
            Phi = np.diagflat(phi[:,:,ris])
            Heq[:,:,ris] = H_RD @ Phi @ H_SR
            alpha[:,:,ris] = H_RD @ H_SR
        Heq = np.sum(Heq, axis=2)
        alpha = np.sum(alpha, axis=2)
        H = [Heq[r:r+self.Nr,:] for r in range(0, Heq.shape[0],self.Nr)]
        alpha_norm = [alpha[r:r+self.Nr,:] for r in range(0, alpha.shape[0],self.Nr)]
        for k in range(self.K):
            Hc = np.vstack([H[i] for i in range(len(H)) if i != k]) #bar(Heq), where kth element is omitted 
            [_,_,V] = svd(Hc)
            W = V.T.conj()[:,-self.Nr:]
            num = np.sum(np.square(np.abs(H[k] @ W)))
            denom = np.sum(np.square(np.abs(alpha_norm[k] @ W)))
            SINR[k] = num/denom
            #reward_all += num/denom
        reward = SINR[0] #taking reward just from 1st user to demodulate
        return reward
