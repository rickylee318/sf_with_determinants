import numpy as np
from numpy.linalg import inv
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import multivariate_normal
from initialize import Initialize
from HMC import HMC


class DA():
    """Data augmentation with Gibbs sampler in 4SFwD
    """
    def __init__(self,y=None,x=None,w=None,pi=None):
        self.y = y
        self.x = x
        self.w = w
        self.pi = pi
    
    def _prior_set(self):
        # prior
        self.sigma_beta_sqr = 10
        self.sigma_delta_sqr = 10
        self.sigma_xi_sqr = 10
        self.sigma_z_sqr = 10
        self.sigma_gamma_sqr = 10

    def run(self, S, N, T, model):
        NT = N*T
        #data
        y = self.y
        x = self.x
        pi = self.pi
        w = self.w

        #prior
        self._prior_set()
#        sigma_beta_sqr = self.sigma_beta_sqr
#        sigma_xi_sqr = self.sigma_xi_sqr
#        sigma_z_sqr = self.sigma_z_sqr
#        sigma_gamma_sqr = self.sigma_gamma_sqr

        #initialize
        beta, delta, xi, gamma, z, u, eta, alpha, sigma_alpha_sqr, sigma_v_sqr = Initialize().generate(model,x,pi,w)

        all_beta = np.zeros([S, x.shape[1]])
        all_xi = np.zeros([S,pi.shape[1]])
        all_delta = np.zeros([S,pi.shape[1]])
        all_z = np.zeros([S,w.shape[1]])
        all_gamma = np.zeros([S,w.shape[1]])
        all_sigma_v_sqr = np.zeros([S,])
        all_sigma_alpha_sqr = np.zeros([S,])
        all_alpha = np.zeros([S,N])
        all_eta = np.zeros([S,N])
        all_u = np.zeros([S,NT])

        #beginning 
        for i in range(S):
            print(i)
            ### Posterior
            # beta
            V_beta = inv((np.dot(x.T,x) * self.sigma_beta_sqr + sigma_v_sqr)/(sigma_v_sqr * self.sigma_beta_sqr))
            mu_beta = np.dot(V_beta, np.dot(x.T, y-u-np.kron(alpha, np.ones([T,])) - np.kron(eta, np.ones(T,)))/sigma_v_sqr)
            beta = multivariate_normal.rvs(mu_beta, V_beta)
            # sigma_v_sqr
            y_tilda = y-np.dot(x,beta)-u-np.kron(alpha,np.ones(T,))-np.kron(eta,np.ones(T,))
            shape = (NT+1)/2
            scale = 2 / (0.0001 + np.dot(y_tilda.T, y_tilda))
            sigma_v_sqr = 1/np.random.gamma(shape,scale)
            # sigma_alpha_sqr
            shape = (N+1)/2
            scale = 2/ (0.0001 + np.dot(alpha.T,alpha))
            sigma_alpha_sqr = 1/np.random.gamma(shape,scale)
            #eta
            V_eta = 1/(np.exp(-np.dot(w, gamma))+T/sigma_v_sqr)
            mu_eta = V_eta * ((y-np.dot(x, beta)- u -np.kron(alpha,np.ones(T,))).reshape([N,T]).sum(axis=1)/sigma_v_sqr + np.exp(-np.dot(w, gamma))*np.dot(w,z))
            myclip_a = 0
            my_mean = mu_eta
            my_std = V_eta** 0.5
            a, b = (myclip_a - my_mean) / my_std, np.inf*np.ones([N,])
            eta = truncnorm.rvs(a,b,loc=my_mean,scale=my_std)
            #u
            V_u = 1/(np.exp(-np.dot(pi, xi))+1/sigma_v_sqr)
            mu_u = V_u * ((y-np.dot(x, beta)-np.kron(eta,np.ones(T,))-np.kron(alpha,np.ones(T,)))/sigma_v_sqr + np.exp(-np.dot(pi, xi))*np.dot(pi,delta))
            myclip_a = 0
            my_mean = mu_u
            my_std = V_u** 0.5
            a, b = (myclip_a - my_mean) / my_std, np.inf*np.ones([NT,])
            u = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)    
            #alpha
            scale = sigma_alpha_sqr * sigma_v_sqr / (T * sigma_alpha_sqr + sigma_v_sqr)
            y_bar = y-np.dot(x, beta)-np.kron(eta, np.ones(T,))-u
            loc = scale / sigma_v_sqr * y_bar.reshape([N,T]).sum(axis=1)
            alpha = norm.rvs(loc = loc, scale = scale)
            #determinants
            kwargs = {'delta':delta, 'sigma_delta_sqr':self.sigma_delta_sqr,'pi':pi,'u':u,'xi':xi,'sigma_xi_sqr':self.sigma_xi_sqr,
            'z':z, 'sigma_z_sqr':self.sigma_z_sqr,'w':w,'gamma':gamma,'eta':eta}
            if model == 'A':
                #sigma_eta
                shape = (N+1)/2
                scale = 2 / (0.0001 + np.dot(eta.T, eta))
                sigma_eta_sqr = 1/np.random.gamma(shape,scale)    
                #sigma_u
                shape = (NT+1)/2
                scale = 2 / (0.0001 + np.dot(u.T, u))
                sigma_u_sqr = 1/np.random.gamma(shape,scale)
                #decompose for PMCMC
                gamma = np.array([np.log(sigma_eta_sqr),0])
                xi = np.array([np.log(sigma_u_sqr),0])
            elif model == 'B':
                delta, z = HMC().sampler(model,**kwargs)
            elif model == 'C':
                xi, gamma = HMC().sampler(model,**kwargs)
            elif model == 'D':
                delta,z,xi,gamma = HMC().sampler(model,**kwargs)
                            
            print('beta')
            print(beta)
            print('delta')
            print(delta)
            print('xi')
            print(xi)
            print('z')
            print(z)
            print('gamma')
            print(gamma)
            print('sigma_alpha_sqr')
            print(sigma_alpha_sqr)
            print('sigma_v_sqr')
            print(sigma_v_sqr)
            print('eta_mean_std')
            print(eta.mean())
            print(eta.std())
            print('u_mean_std')
            print(u.mean())
            print(u.std())
            all_beta[i,:] = beta
            all_xi[i,:] = xi
            all_delta[i,:] = delta
            all_z[i,:] = z
            all_gamma[i,:] = gamma
            all_sigma_alpha_sqr[i] = sigma_alpha_sqr
            all_sigma_v_sqr[i] = sigma_v_sqr
            all_alpha[i,:] = alpha
            all_eta[i,:] = eta
            all_u[i,:] = u
        return all_beta, all_xi, all_delta, all_z, all_gamma, all_sigma_alpha_sqr, all_sigma_v_sqr, all_alpha,all_eta,all_u
        
if __name__ == '__main__':
    main()
