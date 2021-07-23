import numpy as np
from numpy.linalg import inv
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import multivariate_normal
from HMC import HMC
from initialize import Initialize

class TK():
    """Two-parametrization MCMC sampler for 4SFwD
    """
    def __init__(self,y=None,x=None,w=None,pi=None):
        self.y = y
        self.x = x
        self.w = w
        self.pi = pi

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
        
if __name__ == '__main__':
    main()

