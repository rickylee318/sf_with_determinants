import numpy as np
import pandas as pd
from data import Data
from initialize import Initialize
from Method import PMCMC, TK, DA
import timeit
import arviz as az
import xarray as xr
# model
### A: without determinants
### B: determinants on mean
### C: determinants on variance
### D: determinants on mean and variance
# model = 'D'
#method
### PMCMC: Particle Metropolis Within Gibbs sampler (choose gpu parallel computation)
### TK: Two_parametrization method (Tsiona and Kunmbhaker(2014))
### DA: data_augmentation
# method = 'DA'

# H = 10000 #number of particles
# S = 10 # simulation length
# N = 100 # number of individual
# T = 10 # time period
class SF4wD(object):
    """4SFwD model
    """
    def __init__(self,model,method,data_name ='',S=11000, H=10000, gpu=False, save=False):
        self.model = model
        self.method = method
        self.data_name = data_name
        self.S = S
        self.H = H
        self.gpu = gpu
        self.N = None
        self.T = None
        self.transient_determinants = ['ROA']
        self.persistent_determinants = ['E/A']
        self.save = save
        
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
        self.N = int(N)
        self.T = int(T)
    
    def set_deter(self,tran_det,per_det):
        self.transient_determinants = tran_det
        self.persistent_determinants = per_det
        
    def _error_msg(self, method):
        return (method + ' not in defaults%s' %
                self.__class__.__name__)
        
    def _check(self):
        if self.model not in ['A','B','C','D']:
            raise NotImplementedError(self.model + 'not in model list')
            
        if self.method not in ['DA','TK','PMCMC']:
            raise NotImplementedError(self.method + 'not in method list')
            
            
    def run(self):
        self._check()
        #data
        if self.data_name =="":
            if self.N == None:
                N = 100
            if self.T == None:
                T = 10
            Y,X,W,PI = Data().simulate(N,T)
        else:
            Y,X,W,PI = Data().preprcessing(data_name, self.persistent_determinants, self.transient_determinants)
            NT = y.shape[0]
            N = w.shape[0]
            T = int(NT/N)
        #run
        if self.method == 'PMCMC':
            estimator = PMCMC(y=Y,x=X,w=W,pi=PI,H=self.H,gpu=self.gpu)
        elif self.method == 'TK':
            estimator = TK(y=Y,x=X,w=W,pi=PI)
        elif self.method == 'DA':
            estimator = DA(y=Y,x=X,w=W,pi=PI)
        start = timeit.default_timer()
        s_beta, s_xi, s_delta, s_z, s_gamma, s_sigma_alpha_sqr, s_sigma_v_sqr, s_alpha,s_eta, s_u = estimator.run(self.S,N,T,self.model)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        
        ###Report
        d = dict()
        for i in range(s_beta.shape[1]):
            name = 'beta' + str(i)
            d[name] = (['chain','draw'], s_beta[:,i].reshape(-1,1).T)
        for i in range(s_xi.shape[1]):
            name = 'xi' + str(i)
            d[name] = (['chain','draw'], s_xi[:,i].reshape(-1,1).T)
        for i in range(s_delta.shape[1]):
            name = 'delta' + str(i)
            d[name] = (['chain','draw'], s_delta[:,i].reshape(-1,1).T)  
        for i in range(s_z.shape[1]):
            name = 'z' + str(i)
            d[name] = (['chain','draw'], s_z[:,i].reshape(-1,1).T)
        for i in range(s_gamma.shape[1]):
            name = 'gamma' + str(i)
            d[name] = (['chain','draw'], s_gamma[:,i].reshape(-1,1).T)
        d['sigma_alpha_sqr'] = (['chain','draw'], s_sigma_alpha_sqr.reshape(-1,1).T)
        d['sigma_v_sqr'] = (['chain','draw'], s_sigma_v_sqr.reshape(-1,1).T)
        ds = xr.Dataset(
            data_vars=d,
            coords=dict(
                chain=(["chain"], [1]),
                draw=(["draw"], np.arange(self.S).tolist()),
            ),
        )
        pd.set_option('display.max_rows', None)
        print(az.summary(ds))
        
        #save results
        if self.save:
            name = 'sf4' +'_'+ self.data_name +'_'+ self.model +'model' + '_'+ self.method  + 'method' +'gpuH' + str(self.H) + '_12y_4_EAineta_ROAinu'
            pd.DataFrame(all_beta).to_csv(name +'_'+'beta')
            pd.DataFrame(all_xi).to_csv(name +'_'+'xi')
            pd.DataFrame(all_delta).to_csv(name +'_'+'delta')
            pd.DataFrame(all_gamma).to_csv(name +'_'+'gamma')
            pd.DataFrame(all_z).to_csv(name +'_'+'z')
            pd.DataFrame(all_sigma_v_sqr).to_csv(name +'_'+'sigma_v_sqr')
            pd.DataFrame(all_sigma_alpha_sqr).to_csv(name +'_'+'sigma_alpha_sqr')
            pd.DataFrame(all_alpha).to_csv(name +'_'+'alpha')
            pd.DataFrame(all_eta).to_csv(name +'_'+'eta')
            pd.DataFrame(all_u).to_csv(name +'_'+'u')
if __name__ == '__main__':
    model = SF4wD('D','PMCMC',"",10,100)
    model.run()


