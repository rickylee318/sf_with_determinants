import numpy as np
from numpy.linalg import inv
from scipy.stats import norm

        
class HMC(object):
    """HMC sampler for determinants variables in 4SFwD
    """
    def __init__(self):
        pass
    def _loggrad_xi(self,xi):
        """
        theta = [delta, sigma_xi_sqr, pi, u, xi]
        """
        # pi = thetas['pi']
        # xi = thetas['xi']
        # delta = thetas['delta']
        # u = thetas['u']
        # sigma_xi_sqr = thetas['sigma_xi_sqr']
        K = xi.shape[0]
        # Precision matrix with covariance [1, 1.98; 1.98, 4].
        V_u = np.exp(np.dot(self.pi, xi))
        mu_u = np.dot(self.pi,self.delta)
        logp = -0.5 * ((self.u - mu_u)**2 * (1/V_u)).sum() + np.dot(xi.T, xi) * -0.5 / self.sigma_xi_sqr - (np.dot(self.pi, xi)/2).sum() - (np.log(norm.cdf(mu_u/(V_u**0.5)))).sum()
        grad = np.dot(self.pi.T,(self.u - mu_u)**2 * (1/V_u)) * 0.5 -xi/self.sigma_xi_sqr - 0.5 * self.pi.sum(axis=0) + 0.5 * np.dot(self.pi.T,norm.pdf(mu_u/(V_u**0.5))*(mu_u/(V_u**0.5))/norm.cdf(mu_u/(V_u**0.5)))
        return -logp, -grad

    def _loggrad_gamma(self,gamma):
        """
        theta = [z, sigma_z_sqr, w, gamma, eta]
        """
        # z = thetas['z']
        # w = thetas['w']
        # gamma = thetas['gamma']
        # eta = thetas['eta']
        # sigma_z_sqr = thetas['sigma_z_sqr']
        K = self.z.shape[0]
        # Precision matrix with covariance [1, 1.98; 1.98, 4].
        V_eta = np.exp(np.dot(self.w, gamma))
        mu_eta = np.dot(self.w,self.z)
        logp = -0.5 * ((self.eta - mu_eta)**2 * (1/V_eta)).sum() + np.dot(gamma.T, gamma) * -0.5 / self.sigma_z_sqr - (np.dot(self.w, gamma)/2).sum() - (np.log(norm.cdf(mu_eta/(V_eta**0.5)))).sum()
        grad = np.dot(self.w.T,(self.eta - mu_eta)**2 * (1/V_eta)) * 0.5 -gamma/self.sigma_z_sqr - 0.5 * self.w.sum(axis=0) + 0.5 * np.dot(self.w.T,norm.pdf(mu_eta/(V_eta**0.5))*(mu_eta/(V_eta**0.5))/norm.cdf(mu_eta/(V_eta**0.5)))
        return -logp, -grad

    def _loggrad_delta(self,delta):
        """
        theta = [delta, sigma_delta_sqr, pi, u, xi]
        """
        # pi = thetas['pi']
        # xi = thetas['xi']
        # delta = thetas['delta']
        # u = thetas['u']
        # sigma_delta_sqr = thetas['sigma_delta_sqr']
        K = delta.shape[0]
        # Precision matrix with covariance [1, 1.98; 1.98, 4].
        # A = np.linalg.inv( cov )
        V_u = np.exp(np.dot(self.pi, self.xi))
        mu_u = np.dot(self.pi,delta)
        V_delta = inv(np.dot(self.pi.T,np.dot(np.diag(1/V_u), self.pi)) + 1/self.sigma_delta_sqr * np.diag(np.ones(K)))
        mu_delta = np.dot(V_delta, np.dot(self.pi.T,np.dot(np.diag(1/V_u), self.u)))

        logp = -0.5 * np.dot((delta - mu_delta).T, np.dot(inv(V_delta), delta-mu_delta))-np.log(norm.cdf(mu_u/(V_u**0.5))).sum()
        grad = - np.dot(inv(V_delta), delta) + np.dot(inv(V_delta), mu_delta) - np.dot(self.pi.T,norm.pdf(mu_u/(V_u**0.5))/(norm.cdf(mu_u/(V_u**0.5)) * V_u ** 0.5))
        return -logp, -grad

    def _loggrad_z(self,z):
        """
        theta = [z, sigma_z_sqr, w, gamma, eta]
        """
        # z = thetas['z']
        # w = thetas['w']
        # gamma = thetas['gamma']
        # eta = thetas['eta']
        # sigma_z_sqr = thetas['sigma_z_sqr']
        K = z.shape[0]
        # Precision matrix with covariance [1, 1.98; 1.98, 4].
        # A = np.linalg.inv( cov )
        V_eta = np.exp(np.dot(self.w, self.gamma))
        mu_eta = np.dot(self.w,z)
        V_z = inv(np.dot(self.w.T,np.dot(np.diag(1/V_eta), self.w)) + 1/self.sigma_z_sqr * np.diag(np.ones(K)))
        mu_z = np.dot(V_z, np.dot(self.w.T,np.dot(np.diag(1/V_eta), self.eta)))

        logp = -0.5 * np.dot((z - mu_z).T, np.dot(inv(V_z), z-mu_z))-np.log(norm.cdf(mu_eta/(V_eta**0.5))).sum()
        grad = - np.dot(inv(V_z), z) + np.dot(inv(V_z), mu_z) - np.dot(self.w.T,norm.pdf(mu_eta/(V_eta**0.5))/(norm.cdf(mu_eta/(V_eta**0.5)) * V_eta ** 0.5))
        return -logp, -grad
        
 
    def _sampler_scalar(self, theta, target_name, epsilon, L, f):
        current_q = theta.copy()
        q = theta
        k = q.shape[0]
        p = np.random.normal(0,1,k)

        current_U,_ = f(q) # logp
        current_K = (p**2).sum() / 2 

        # Make a half step for momentum at the beginning
        _ , grad = f(q) 
        p-= epsilon * grad / 2

        # Alternate full steps for position and momentum

        for i in range(L):
            # Make a full step for the position
            q = q + epsilon * p

            # Make a full step for the momentum, except at the end of trajectory
            if i!=(L-1):
                _ , grad = f(q) 
                p = p - epsilon * grad
        # Make a half step for momentum at the end
        proposed_U , grad = f(q) 
        p = p - epsilon * grad / 2

        # Negate momentum at end trajectory to make the proposal symmetric
        p = -p

        # Evaluate potential and kinetic energies at start and end of trajectory

        # proposed_U = f(q) #logp
        proposed_K = (p**2).sum() / 2
    #     print(np.exp(current_U - proposed_U + current_K - proposed_K))
    #     print(q)
        if np.log(np.random.rand()) < current_U - proposed_U + current_K - proposed_K:
            return q
        return current_q

    def sampler(self, model_name, **kwargs):
        self.delta = kwargs['delta']
        self.__dict__.update(kwargs)
        if model_name =='B':
            #model B
            delta = self._sampler_scalar(self.delta, 'delta', 0.01, 50, self._loggrad_delta)
            z = self._sampler_scalar(self.z, 'z', 0.01, 100, self._loggrad_z)
            return delta, z
        elif model_name=='C':
            #model C
            xi = self._sampler_scalar(self.xi, 'xi', 0.001, 50, self._loggrad_xi)
            gamma = self._sampler_scalar(self.gamma, 'gamma', 0.001, 50, self._loggrad_gamma)
            return xi, gamma
        elif model_name == 'D':
            #model D
            delta = self._sampler_scalar(self.delta, 'delta', 0.01, 50, self._loggrad_delta)
            z = self._sampler_scalar(self.z, 'z', 0.01, 100, self._loggrad_z)
            self.delta = delta
            self.z = z
            xi = self._sampler_scalar(self.xi, 'xi', 0.001, 50, self._loggrad_xi)
            gamma = self._sampler_scalar(self.gamma, 'gamma', 0.001, 50, self._loggrad_gamma)
            return delta, z, xi, gamma

if __name__ == '__main__':
    main()


