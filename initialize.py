import numpy as np
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import multivariate_normal

class Initialize():
    """empirical US bank data or simulation data in 4SFwD
    """
    def __init__(self):
        pass
    def generate(self,model,x,pi,w):
        # initialize
        sigma_v_sqr = 1.
        sigma_alpha_sqr = 1.
        beta = np.ones([x.shape[1]]) * 0.5

        delta = np.ones([pi.shape[1]]) * 0.5
        xi = np.ones([pi.shape[1]]) * 0.5
        
        if model == 'A':
            xi = np.zeros([pi.shape[1]])
            delta = np.zeros([pi.shape[1]])
        elif model=='B':
            xi = np.zeros([pi.shape[1]])
        elif model == 'C':
            delta = np.zeros([pi.shape[1]])


        z = np.ones([w.shape[1]]) * 0.5
        gamma = np.ones([w.shape[1]]) * 0.5
        
        if model == 'A':
            gamma = np.zeros([w.shape[1]])
            z = np.zeros([w.shape[1]])
        if model == 'B':
            gamma = np.zeros([w.shape[1]])
        elif model == 'C':
            z = np.zeros([w.shape[1]])

        myclip_a = 0
        my_mean = np.dot(pi,delta)
        my_std = np.exp(np.dot(pi,xi)/2)
        a, b = (myclip_a - my_mean) / my_std, np.inf
        u = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)

        myclip_a = 0
        my_mean = np.dot(w,z)
        my_std = np.exp(np.dot(w,gamma)/2)
        a, b = (myclip_a - my_mean) / my_std, np.inf
        eta = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)
        alpha = np.random.normal(0, sigma_alpha_sqr ** 0.5, [w.shape[0],])
        return beta, delta, xi, gamma, z, u, eta, alpha, sigma_alpha_sqr, sigma_v_sqr
        
if __name__ == '__main__':
    main()
