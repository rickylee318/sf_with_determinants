import numpy as np
from numpy.linalg import inv
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import multivariate_normal
from initialize import Initialize
from HMC import HMC

class Method(object):
    """Base method in 4SFwD
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.sigma_beta_sqr = 10
        self.sigma_delta_sqr = 10
        self.sigma_xi_sqr = 10
        self.sigma_z_sqr = 10
        self.sigma_gamma_sqr = 10
        
    def _error_msg(self, method):
        return ('method ' + method + ' not implemented in class%s' %
                self.__class__.__name__)
        
    def _initalize(self, S,N,T,model):
        try:
            NT = N*T

            #initialize
            self.beta, self.delta, self.xi, self.gamma, self.z, self.u, self.eta, self.alpha, self.sigma_alpha_sqr, self.sigma_v_sqr = Initialize().generate(model,self.x,self.pi,self.w)

            all_beta = np.zeros([S, self.x.shape[1]])
            all_xi = np.zeros([S,self.pi.shape[1]])
            all_delta = np.zeros([S,self.pi.shape[1]])
            all_z = np.zeros([S,self.w.shape[1]])
            all_gamma = np.zeros([S,self.w.shape[1]])
            all_sigma_v_sqr = np.zeros([S,])
            all_sigma_alpha_sqr = np.zeros([S,])
            all_alpha = np.zeros([S,N])
            all_eta = np.zeros([S,N])
            all_u = np.zeros([S,NT])

            return all_beta, all_xi, all_delta, all_z, all_gamma, all_sigma_v_sqr, all_sigma_alpha_sqr, all_alpha, all_eta, all_u
        except:
            raise NotImplementedError(self._error_msg('initalize'))
    def run(self):
        raise NotImplementedError(self._error_msg('run'))

    
class DA(Method):
    """Data augmentation with Gibbs sampler in 4SFwD
    """


    def run(self, S, N, T, model):
        NT = N*T
        #data
        y = self.y
        x = self.x
        pi = self.pi
        w = self.w

        #initalize
        all_beta, all_xi, all_delta, all_z, all_gamma, all_sigma_v_sqr, all_sigma_alpha_sqr, all_alpha, all_eta, all_u = self._initalize(S,N,T,model)
        beta = self.beta
        delta = self.delta
        xi = self.xi
        gamma = self.gamma
        z = self.z
        u = self.u
        eta = self.eta
        alpha = self.alpha
        sigma_alpha_sqr = self.sigma_alpha_sqr
        sigma_v_sqr = self.sigma_v_sqr

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

    
    

class TK(Method):
    """Two-parametrization MCMC sampler for 4SFwD
    """

    def _F(self, delta_,mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T,sigma_v_sqr):
        R = (y-np.dot(x,beta)-u - np.kron(delta_, np.ones([T,])))
        tmp = ratio * (delta_ - mu_eta) /(sigma_) + mu_eta * sigma_ / ((sigma_alpha_sqr * sigma_eta_sqr)**0.5)
        F = norm.cdf(tmp) * np.exp(-0.5*(delta_-mu_eta)**2/(sigma_**2)) * np.exp(-0.5 * (R.reshape([N,T])**2).sum(axis=1) / sigma_v_sqr)
        return F

    def _G(self, delta_,mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T,sigma_v_sqr):
        R = (y-np.dot(x,beta)-u - np.kron(delta_, np.ones([T,])))
        tmp = ratio * (delta_ - mu_eta) /(sigma_) + mu_eta * sigma_ / ((sigma_alpha_sqr * sigma_eta_sqr)**0.5)
        ratio_pdf = (norm.pdf(tmp) / norm.cdf(tmp))
        G = -(delta_ - mu_eta)/(sigma_**2) + ratio_pdf * ratio /sigma_ + (R.reshape([N,T]).sum(axis=1)/sigma_v_sqr)
        return G

    def _H(self, delta_,mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T,sigma_v_sqr):
        tmp = ratio * (delta_ - mu_eta) /(sigma_) + mu_eta * sigma_ / ((sigma_alpha_sqr * sigma_eta_sqr)**0.5)
        ratio_pdf = (norm.pdf(tmp) / norm.cdf(tmp))
        H = (-1/sigma_**2) + ((ratio_pdf * ratio / sigma_)**2 + ratio_pdf * ratio**2 / sigma_**2) - T/sigma_v_sqr
        return H
    
    def _newton(self,x0,args):
        i = 0
        diff = 1.
        max_iterations = 1000
        tol = 0.0001
        while diff > tol and i < max_iterations:
            i+=1
            # first evaluate fval
            fval = np.asarray(self._F(x0, *args))
            # If all fval are 0, all roots have been found, then terminate
            fder = np.asarray(self._G(x0, *args))
            # Newton step
            dp = fval / fder
            fder2 = np.asarray(self._H(x0, *args))
    #         dp = fder/fder2
            dp = dp / (1.0 - 0.5 * dp * fder2 / fder)
            # only update nonzero derivatives
            x0 = x0 + dp * 10 
            diff = abs(np.log(fval).sum() - np.log(np.asarray(self._F(x0, *args))).sum())
        H = np.asarray(self._H(x0, *args))
        print(np.log(np.asarray(self._F(x0, *args))).sum())
        return x0,H
    

    def run(self, S, N, T, model):
        NT = N*T
        #data
        y = self.y
        x = self.x
        pi = self.pi
        w = self.w

        #initalize
        all_beta, all_xi, all_delta, all_z, all_gamma, all_sigma_v_sqr, all_sigma_alpha_sqr, all_alpha, all_eta, all_u = self._initalize(S,N,T,model)
        beta = self.beta
        delta = self.delta
        xi = self.xi
        gamma = self.gamma
        z = self.z
        u = self.u
        eta = self.eta
        alpha = self.alpha
        sigma_alpha_sqr = self.sigma_alpha_sqr
        sigma_v_sqr = self.sigma_v_sqr

        #beginning 
        for i in range(S):
            print(i)
            ### Posterior
            # delta_ 
            delta_ = alpha + eta
            #delta_
            mu_eta = np.dot(w,z)
            sigma_eta_sqr = np.exp(np.dot(w,gamma))
            sigma_ = (sigma_alpha_sqr + sigma_eta_sqr)**0.5
            ratio = (sigma_eta_sqr/sigma_alpha_sqr)**0.5
            args = (mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T,sigma_v_sqr)
            mu_delta_, H_ = self._newton(delta_,args)

            delta__star = np.random.normal(mu_delta_,(-1/H_)**0.5)
            A = (self._F(delta__star,*args)/self._F(mu_delta_,*args))* np.exp(-0.5 * (delta__star - mu_delta_)**2 / (-1/H_))
            U = np.random.uniform(0,1,[N,])
            select = U < A
            if select.any():
            	delta_[select] = delta__star[select] #change value if over u
            # beta
            V_beta = inv((np.dot(x.T,x) * self.sigma_beta_sqr + sigma_v_sqr)/(sigma_v_sqr * self.sigma_beta_sqr))
            mu_beta = np.dot(V_beta, np.dot(x.T, y-u-np.kron(delta_, np.ones([T,])))/sigma_v_sqr)
            beta = multivariate_normal.rvs(mu_beta, V_beta)
            # sigma_v_sqr
            y_tilda = y-np.dot(x,beta)-u-np.kron(delta_,np.ones(T,))
            shape = (NT+1)/2
            scale = 2 / (0.0001 + np.dot(y_tilda.T, y_tilda))
            sigma_v_sqr = 1/np.random.gamma(shape,scale)
            #eta
            A = inv(np.ones([T,T]) * sigma_alpha_sqr + np.diag(np.ones([T])) * sigma_v_sqr)
            Sigma_eta = 1/(A.sum() + 1/sigma_eta_sqr)
            eta_tilda = y - np.dot(x,beta)-u
            Mu_eta = Sigma_eta * (np.dot(eta_tilda.reshape([N,T]),A).sum(axis=1) + mu_eta/sigma_eta_sqr)
            myclip_a = 0
            my_mean = Mu_eta
            my_std = Sigma_eta** 0.5
            a, b = (myclip_a - my_mean) / my_std, np.inf*np.ones([N,])
            eta = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)
            # alpha
            alpha = delta_ - eta
            # sigma_alpha_sqr
            shape = (N+1)/2
            scale = 2/ (0.0001 + np.dot(alpha.T,alpha))
            sigma_alpha_sqr = 1/np.random.gamma(shape,scale)
            # u
            V_u = 1/(np.exp(-np.dot(pi, xi))+1/sigma_v_sqr)
            mu_u = V_u * ((y-np.dot(x, beta)-np.kron(delta_,np.ones(T,)))/sigma_v_sqr + np.exp(-np.dot(pi, xi))*np.dot(pi,delta))
            myclip_a = 0
            my_mean = mu_u
            my_std = V_u** 0.5
            a, b = (myclip_a - my_mean) / my_std, np.inf*np.ones([NT,])
            u = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)

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

    
class PMCMC(Method):
    """Particle Metropolis within Gibbs sampler for latent variables in 4SFwD
    """
#     def __init__(self,y=None,x=None,w=None,pi=None,H = None,gpu=False):
#         self.y = y
#         self.x = x
#         self.w = w
#         self.pi = pi
#         self.H = H
#         self.gpu = gpu
    def _check_gpu(self):
        if self.gpu:
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

    def run(self,S,N,T,model):
        self._check_gpu()
        NT = N*T
        #data
        y = self.y
        x = self.x
        pi = self.pi
        w = self.w

        #initalize
        all_beta, all_xi, all_delta, all_z, all_gamma, all_sigma_v_sqr, all_sigma_alpha_sqr, all_alpha, all_eta, all_u = self._initalize(S,N,T,model)
        beta = self.beta
        delta = self.delta
        xi = self.xi
        gamma = self.gamma
        z = self.z
        u = self.u
        eta = self.eta
        alpha = self.alpha
        sigma_alpha_sqr = self.sigma_alpha_sqr
        sigma_v_sqr = self.sigma_v_sqr
        
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