import numpy as np
import pandas as pd
import cupy as cp
from numpy.lib import index_tricks
from cupyx.scipy.special import ndtr
from cupy.core import internal
from cupy.linalg import inv
from numpy.linalg import inv
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import invgamma
from scipy.stats import gamma
from numpy.random import random
from scipy.stats import multivariate_normal
import copy
from data import Data
from initialize import Initialize
from PMCMC import PMCMC
import timeit

start = timeit.default_timer()

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
S = 11000 # simulation length
N = 100 # number of individual
T = 10 # time period
data_name ='group6'
transient_determinants = ['ROA']
persistent_determinants = ['E/A']

#data
if data_name =="":
    y,x,w,pi = Data().simulate(N=100,T=10)
else:
    y,x,w,pi = Data().preprcessing(data_name, persistent_determinants, transient_determinants)
#run
if metohd == 'PMCMC':
    estimator = PMCMC(y,x,w,pi,H,gpu=True)
elif method == 'TK':
    estimator = TK(y,x,w,pi)
elif mehtod == 'DA':
    estimator = DA(y,x,w,pi)
s_beta, s_xi, s_delta, s_z, s_gamma, s_sigma_alpha_sqr, s_sigma_v_sqr, s_alpha,s_eta, s_u = estimator.run(S,N,T,model)

#store results
name = 'sf2_with_panel' +'_'+ data_name +'_'+ 'model' + model + '_' + 'gpuH' + str(H) + '_12y_4_EAineta_ROAinu'
pd.DataFrame(all_beta[1000:,:]).to_csv(name +'_'+'beta')
pd.DataFrame(all_xi[1000:,:]).to_csv(name +'_'+'xi')
pd.DataFrame(all_delta[1000:,:]).to_csv(name +'_'+'delta')
pd.DataFrame(all_gamma[1000:,:]).to_csv(name +'_'+'gamma')
pd.DataFrame(all_z[1000:,:]).to_csv(name +'_'+'z')
pd.DataFrame(all_sigma_v_sqr[1000:]).to_csv(name +'_'+'sigma_v_sqr')
pd.DataFrame(all_sigma_alpha_sqr[1000:]).to_csv(name +'_'+'sigma_alpha_sqr')
pd.DataFrame(all_alpha[1000:,:]).to_csv(name +'_'+'alpha')
pd.DataFrame(all_eta[1000:,:]).to_csv(name +'_'+'eta')
pd.DataFrame(all_u[1000:,:]).to_csv(name +'_'+'u')


