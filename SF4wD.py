import numpy as np
import pandas as pd
from data import Data
from initialize import Initialize
from PMCMC import PMCMC
from TK import TK
from DA import DA
import timeit

# model
### A: without determinants
### B: determinants on mean
### C: determinants on variance
### D: determinants on mean and variance
model = 'D'
#method
### PMCMC: Particle Metropolis Within Gibbs sampler (choose gpu parallel computation)
### TK: Two_parametrization method (Tsiona and Kunmbhaker(2014))
### DA: data_augmentation
method = 'PMCMC'

H = 10000 #number of particles
S = 10 # simulation length
N = 100 # number of individual
T = 10 # time period
class SF4wD():
    """4SFwD model
    """
    def __init__(self,model,method,data_name ='',S=11000, H=10000, gpu=False):
        self.model = model
        self.method = method
        self.data_name = data_name
        self.S = S
        self.H = H
        self.gpu = gpu
    def set_particle_number(self,H):
        self.H = H
    def set_method(self,method):
        self.method = method
    def set_model(self,model):
        self.model=model
    def set_simu_length(self, S):
        self.S = S
    def set_gpu(self,gpu):
        self.gpu=gpu
    def set_data_name(self, data_name):
        self.data_name = data_name
    def set_simu_data(self,N,T):
        self.N = N
        self.T = T
    def run(self):
        start = timeit.default_timer()
        #data
        if data_name =="":
            if self.N == None:
                N = 100
            if self.T == None:
                T = 10
            y,x,w,pi = Data().simulate(N,T)
        else:
            transient_determinants = ['ROA']
            persistent_determinants = ['E/A']
            y,x,w,pi = Data().preprcessing(data_name, persistent_determinants, transient_determinants)
            NT = y.shape[0]
            N = w.shape[0]
            T = int(NT/N)
        #run
        if method == 'PMCMC':
            estimator = PMCMC(y,x,w,pi,self.H,gpu=self.gpu)
        elif method == 'TK':
            estimator = TK(y,x,w,pi)
        elif method == 'DA':
            estimator = DA(y,x,w,pi)
        s_beta, s_xi, s_delta, s_z, s_gamma, s_sigma_alpha_sqr, s_sigma_v_sqr, s_alpha,s_eta, s_u = estimator.run(S,N,T,model)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        #store results
        name = 'sf2_with_panel' +'_'+ data_name +'_'+ 'model' + model + '_' + 'gpuH' + str(H) + '_12y_4_EAineta_ROAinu'
        #pd.DataFrame(all_beta[1000:,:]).to_csv(name +'_'+'beta')
        #pd.DataFrame(all_xi[1000:,:]).to_csv(name +'_'+'xi')
        #pd.DataFrame(all_delta[1000:,:]).to_csv(name +'_'+'delta')
        #pd.DataFrame(all_gamma[1000:,:]).to_csv(name +'_'+'gamma')
        #pd.DataFrame(all_z[1000:,:]).to_csv(name +'_'+'z')
        #pd.DataFrame(all_sigma_v_sqr[1000:]).to_csv(name +'_'+'sigma_v_sqr')
        #pd.DataFrame(all_sigma_alpha_sqr[1000:]).to_csv(name +'_'+'sigma_alpha_sqr')
        #pd.DataFrame(all_alpha[1000:,:]).to_csv(name +'_'+'alpha')
        #pd.DataFrame(all_eta[1000:,:]).to_csv(name +'_'+'eta')
        #pd.DataFrame(all_u[1000:,:]).to_csv(name +'_'+'u')
if __name__ == '__main__':


