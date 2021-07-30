import numpy as np
from scipy.stats import truncnorm
from HMC import HMC
from initialize import Initialize
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from scipy.stats import norm

class PMCMC():
    """Particle Metropolis within Gibbs sampler for latent variables in 4SFwD
    """
    def __init__(self,y=None,x=None,w=None,pi=None,H = None,gpu=False):
        self.y = y
        self.x = x
        self.w = w
        self.pi = pi
        self.H = H
        self.gpu = gpu
        if gpu:
            import cupy as cp
            from cupyx.scipy.special import ndtr
            from cupy.core import internal
            from truncnorm import TruncNormal
    #multinomial resampling
    def _vectorized(self,prob_matrix):
        s = prob_matrix.cumsum(axis=0)
        r = cp.random.rand(prob_matrix.shape[1])
        k = (s < r).sum(axis=0)
        return k

    def _vectorized2(self,prob_matrix):
        s = prob_matrix.cumsum(axis=0)
        r = np.random.rand(prob_matrix.shape[1])
        k = (s < r).sum(axis=0)
        return k

    def _gpu_normal_pdf(self,X):
        inv_sqrt_2pi = 0.3989422804014327
        pdf = inv_sqrt_2pi * cp.exp(-cp.square(X)/2)
        return pdf
    
    def sampler(self,w,gamma,pi,xi,z,alpha,u,N,T,y,x,beta,sigma_v_sqr,sigma_alpha_sqr,eta,H,delta):
        NT = N*T
        # sample u from N(mu_u, V_u)
        my_mean = cp.asarray(np.dot(w, z))
        my_std = cp.asarray(np.exp(np.dot(w, gamma))** 0.5)
        a, b =  - my_mean / my_std, cp.inf * cp.ones([N,])
    #    eta_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,N))
        eta_particles = TruncNormal.rvs(a,b,my_mean, my_std, (H,N))
        eta_particles = cp.concatenate([eta_particles, cp.asarray(eta).reshape(-1,1).T], axis=0)
        
        my_mean = cp.asarray(np.dot(pi, delta))
        my_std = cp.asarray(np.exp(np.dot(pi, xi))** 0.5)
        a, b =  - my_mean / my_std, cp.inf * cp.ones([NT,])
        u_particles = TruncNormal.rvs(a,b,my_mean, my_std, (H,NT))
        u_particles = cp.concatenate([u_particles, cp.asarray(u).reshape(-1,1).T], axis=0)
        
    #   alpha_particles = norm.rvs(0, sigma_alpha_sqr ** 0.5, size=(H,N))
    #    alpha_particles = np.asarray(alpha_particles)
    #    alpha_particles = np.concatenate([alpha_particles, alpha.reshape(-1,1).T], axis=0)
        alpha_particles = cp.random.normal(0,sigma_alpha_sqr ** 0.5, size = (H,N))
        alpha_particles = cp.concatenate([alpha_particles, cp.asarray(alpha).reshape(-1,1).T], axis=0)


        inv_sqrt_2pi = 0.3989422804014327
        W = inv_sqrt_2pi * cp.exp(-cp.square(((cp.asarray(y - np.dot(x,beta))-cp.kron(alpha_particles+eta_particles, cp.ones([T,]))-u_particles)/(sigma_v_sqr**0.5)))/2) / (sigma_v_sqr**0.5)
        #x_ = ((cp.asarray(y - np.dot(x,beta))-alpha_particles_)/(sigma_v_sqr**0.5))
        #del alpha_particles_
        
        #w = gpu_normal_pdf(x_)
        w_ = W.reshape([H+1,N,T]).prod(axis=2)
        w_ = w_/w_.sum(axis=0)
        
        index = self._vectorized(w_)
        new_alpha = alpha_particles[index,cp.arange(N)].get()
        new_eta = eta_particles[index, cp.arange(N)].get()
        new_u = u_particles[cp.kron(index, cp.ones([T,])).astype(int),cp.arange(N*T)].get()
        return new_eta, new_alpha, new_u

    def sampler2(self,w,gamma,pi,xi,z,alpha,u,N,T,y,x,beta,sigma_v_sqr,sigma_alpha_sqr,eta,H,delta):
        NT = N*T
        # sample eta from N(mu_u, V_u)
        my_mean = np.dot(w, z)
        my_std = np.exp(np.dot(w, gamma))** 0.5
        a, b =  - my_mean / my_std, np.inf * np.ones([N,])
        eta_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,N))
        eta_particles = np.concatenate([eta_particles, eta.reshape(-1,1).T], axis=0)
        # sample u 
        my_mean = np.dot(pi, delta)
        my_std = np.exp(np.dot(pi, xi))** 0.5
        a, b =  - my_mean / my_std, np.inf * np.ones([NT,])
        u_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,NT))
        u_particles = np.concatenate([u_particles, u.reshape(-1,1).T], axis=0)
        # sample alpha
        alpha_particles = norm.rvs(0, sigma_alpha_sqr ** 0.5, size=(H,N))
        alpha_particles = np.concatenate([alpha_particles, alpha.reshape(-1,1).T], axis=0)
   
        inv_sqrt_2pi = 0.3989422804014327
        w = inv_sqrt_2pi * np.exp(-np.square(((y - np.dot(x,beta)-np.kron(alpha_particles+eta_particles, np.ones([T,]))-u_particles)/(sigma_v_sqr**0.5)))/2) / (sigma_v_sqr**0.5)
        w_ = w.reshape([H+1,N,T]).prod(axis=2)
        w_ = w_/w_.sum(axis=0)
        
        index = self._vectorized2(w_)
        new_alpha = alpha_particles[index,np.arange(N)]
        new_eta = eta_particles[index, np.arange(N)]
        new_u = u_particles[np.kron(index, np.ones([T,])).astype(int),np.arange(N*T)]
        return new_eta, new_alpha, new_u

    def _prior_set(self):
        # prior
        self.sigma_beta_sqr = 10
        self.sigma_delta_sqr = 10
        self.sigma_xi_sqr = 10
        self.sigma_z_sqr = 10
        self.sigma_gamma_sqr = 10

    def run(self,S,N,T,model):

        NT = N*T
        #data
        y = self.y
        x = self.x
        pi = self.pi
        w = self.w

        #prior
        self._prior_set()
        #sigma_beta_sqr = self.sigma_beta_sqr
        #sigma_xi_sqr = self.sigma_xi_sqr
        #sigma_z_sqr = self.sigma_z_sqr
        #sigma_gamma_sqr = self.sigma_gamma_sqr
        #sigma_delta_sqr = self.sigma_sqr
        #initialize
        beta, delta, xi, gamma, z, u, eta, alpha, sigma_alpha_sqr, sigma_v_sqr = Initialize().generate(model,x,pi,w)

        #all_MCMC
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
        #adjust particles number
        H = self.H
        H = H - 1
        gpu = self.gpu

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
            
            if gpu:
                eta, alpha, u =self.sampler(w,gamma,pi,xi,z,alpha,u,N,T,y,x,beta,sigma_v_sqr,sigma_alpha_sqr,eta,H,delta)
            else:
                eta, alpha, u =self.sampler2(w,gamma,pi,xi,z,alpha,u,N,T,y,x,beta,sigma_v_sqr,sigma_alpha_sqr,eta,H,delta)

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

