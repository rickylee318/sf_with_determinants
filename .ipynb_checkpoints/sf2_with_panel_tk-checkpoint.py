import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.stats import invgamma
from scipy.stats import gamma
from numpy.random import random
from scipy.stats import multivariate_normal
import copy
import timeit

start = timeit.default_timer()

def HMC(theta, target_name, epsilon, L, accpt_num, f):
    thetas = copy.deepcopy(theta)
    current_q = thetas[target_name]
    q = current_q.copy()
    k = delta.shape[0]
    current_p = np.random.normal(0,1,k)
    p = current_p.copy()

    current_U,_ = f(thetas) # logp
    current_K = (current_p**2).sum() / 2 

    # Make a half step for momentum at the beginning
    thetas[target_name] = q
    _ , grad = f(thetas) 
    p-= epsilon * grad / 2

    # Alternate full steps for position and momentum

    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        # Make a full step for the momentum, except at the end of trajectory
        if i!=(L-1):
            thetas[target_name] = q
            _ , grad = f(thetas) 
            p = p - epsilon * grad
    # Make a half step for momentum at the end
    thetas[target_name] = q
    proposed_U , grad = f(thetas) 
    p = p - epsilon * grad / 2

    # Negate momentum at end trajectory to make the proposal symmetric
    p = -p

    # Evaluate potential and kinetic energies at start and end of trajectory

    # proposed_U = f(q) #logp
    proposed_K = (p**2).sum() / 2
#     print(np.exp(current_U - proposed_U + current_K - proposed_K))
#     print(q)
    if np.log(np.random.rand()) < current_U - proposed_U + current_K - proposed_K:
        accpt_num += 1
        return q, accpt_num
    return current_q, accpt_num

def loggrad_xi(thetas):
    """
    theta = [delta, sigma_xi_sqr, pi, u, xi]
    """
    pi = thetas['pi']
    xi = thetas['xi']
    delta = thetas['delta']
    u = thetas['u']
    sigma_xi_sqr = thetas['sigma_xi_sqr']
    K = xi.shape[0]
    # Precision matrix with covariance [1, 1.98; 1.98, 4].
    V_u = np.exp(np.dot(pi, xi))
    mu_u = np.dot(pi,delta)
    logp = -0.5 * ((u - mu_u)**2 * (1/V_u)).sum() + np.dot(xi.T, xi) * -0.5 / sigma_xi_sqr - (np.dot(pi, xi)/2).sum() - (np.log(norm.cdf(mu_u/(V_u**0.5)))).sum()
    grad = np.dot(pi.T,(u - mu_u)**2 * (1/V_u)) * 0.5 -xi/sigma_xi_sqr - 0.5 * pi.sum(axis=0) + 0.5 * np.dot(pi.T,norm.pdf(mu_u/(V_u**0.5))*(mu_u/(V_u**0.5))/norm.cdf(mu_u/(V_u**0.5)))
    return -logp, -grad

def loggrad_gamma(thetas):
    """
    theta = [z, sigma_z_sqr, w, gamma, eta]
    """
    z = thetas['z']
    w = thetas['w']
    gamma = thetas['gamma']
    eta = thetas['eta']
    sigma_z_sqr = thetas['sigma_z_sqr']
    K = z.shape[0]
    # Precision matrix with covariance [1, 1.98; 1.98, 4].
    V_eta = np.exp(np.dot(w, gamma))
    mu_eta = np.dot(w,z)
    logp = -0.5 * ((eta - mu_eta)**2 * (1/V_eta)).sum() + np.dot(gamma.T, gamma) * -0.5 / sigma_z_sqr - (np.dot(w, gamma)/2).sum() - (np.log(norm.cdf(mu_eta/(V_eta**0.5)))).sum()
    grad = np.dot(w.T,(eta - mu_eta)**2 * (1/V_eta)) * 0.5 -gamma/sigma_z_sqr - 0.5 * w.sum(axis=0) + 0.5 * np.dot(w.T,norm.pdf(mu_eta/(V_eta**0.5))*(mu_eta/(V_eta**0.5))/norm.cdf(mu_eta/(V_eta**0.5)))
    return -logp, -grad

def loggrad_delta(thetas):
    """
    theta = [delta, sigma_delta_sqr, pi, u, xi]
    """
    pi = thetas['pi']
    xi = thetas['xi']
    delta = thetas['delta']
    u = thetas['u']
    sigma_delta_sqr = thetas['sigma_delta_sqr']
    K = delta.shape[0]
    # Precision matrix with covariance [1, 1.98; 1.98, 4].
    # A = np.linalg.inv( cov )
    V_u = np.exp(np.dot(pi, xi))
    mu_u = np.dot(pi,delta)
    V_delta = inv(np.dot(pi.T,np.dot(np.diag(1/V_u), pi)) + 1/sigma_delta_sqr * np.diag(np.ones(K)))
    mu_delta = np.dot(V_delta, np.dot(pi.T,np.dot(np.diag(1/V_u), u)))

    logp = -0.5 * np.dot((delta - mu_delta).T, np.dot(inv(V_delta), delta-mu_delta))-np.log(norm.cdf(mu_u/(V_u**0.5))).sum()
    grad = - np.dot(inv(V_delta), delta) + np.dot(inv(V_delta), mu_delta) - np.dot(pi.T,norm.pdf(mu_u/(V_u**0.5))/(norm.cdf(mu_u/(V_u**0.5)) * V_u ** 0.5))
    return -logp, -grad

def loggrad_z(thetas):
    """
    theta = [z, sigma_z_sqr, w, gamma, eta]
    """
    z = thetas['z']
    w = thetas['w']
    gamma = thetas['gamma']
    eta = thetas['eta']
    sigma_z_sqr = thetas['sigma_z_sqr']
    K = z.shape[0]
    # Precision matrix with covariance [1, 1.98; 1.98, 4].
    # A = np.linalg.inv( cov )
    V_eta = np.exp(np.dot(w, gamma))
    mu_eta = np.dot(w,z)
    V_z = inv(np.dot(w.T,np.dot(np.diag(1/V_eta), w)) + 1/sigma_z_sqr * np.diag(np.ones(K)))
    mu_z = np.dot(V_z, np.dot(w.T,np.dot(np.diag(1/V_eta), eta)))

    logp = -0.5 * np.dot((z - mu_z).T, np.dot(inv(V_z), z-mu_z))-np.log(norm.cdf(mu_eta/(V_eta**0.5))).sum()
    grad = - np.dot(inv(V_z), z) + np.dot(inv(V_z), mu_z) - np.dot(w.T,norm.pdf(mu_eta/(V_eta**0.5))/(norm.cdf(mu_eta/(V_eta**0.5)) * V_eta ** 0.5))
    return -logp, -grad

def F(delta_,mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T):
    R = (y-np.dot(x,beta)-u - np.kron(delta_, np.ones([T,])))
    tmp = ratio * (delta_ - mu_eta) /(sigma_) + mu_eta * sigma_ / ((sigma_alpha_sqr * sigma_eta_sqr)**0.5)
    F = norm.cdf(tmp) * np.exp(-0.5*(delta_-mu_eta)**2/(sigma_**2)) * np.exp(-0.5 * (R.reshape([N,T])**2).sum(axis=1) / sigma_v_sqr)
    return F

def G(delta_,mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T):
    R = (y-np.dot(x,beta)-u - np.kron(delta_, np.ones([T,])))
    tmp = ratio * (delta_ - mu_eta) /(sigma_) + mu_eta * sigma_ / ((sigma_alpha_sqr * sigma_eta_sqr)**0.5)
    ratio_pdf = (norm.pdf(tmp) / norm.cdf(tmp))
    G = -(delta_ - mu_eta)/(sigma_**2) + ratio_pdf * ratio /sigma_ + (R.reshape([N,T]).sum(axis=1)/sigma_v_sqr)
    return G

def H(delta_,mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T):
    tmp = ratio * (delta_ - mu_eta) /(sigma_) + mu_eta * sigma_ / ((sigma_alpha_sqr * sigma_eta_sqr)**0.5)
    ratio_pdf = (norm.pdf(tmp) / norm.cdf(tmp))
    H = (-1/sigma_**2) + ((ratio_pdf * ratio / sigma_)**2 + ratio_pdf * ratio**2 / sigma_**2) - T/sigma_v_sqr
    return H

def newton(x0,func,fprime,fprime2,args):
    i = 0
    diff = 1.
    max_iterations = 1000
    tol = 0.0001
    while diff > tol and i < max_iterations:
        i+=1
        # first evaluate fval
        fval = np.asarray(func(x0, *args))
        # If all fval are 0, all roots have been found, then terminate
        fder = np.asarray(fprime(x0, *args))
        # Newton step
        dp = fval / fder
        fder2 = np.asarray(fprime2(x0, *args))
#         dp = fder/fder2
        dp = dp / (1.0 - 0.5 * dp * fder2 / fder)
        # only update nonzero derivatives
        x0 = x0 + dp * 10 
        diff = abs(np.log(fval).sum() - np.log(np.asarray(func(x0, *args))).sum())
    H = np.asarray(fprime2(x0, *args))
    print(np.log(np.asarray(func(x0, *args))).sum())
    return x0,H

N = 1000
T=10
NT = N * T
K = 2
x = np.concatenate([np.ones([NT,1]),np.random.normal(0,1,[NT,1])],axis=1)
pi = np.concatenate([np.ones([NT,1]),np.random.normal(0,1,[NT,1])],axis=1)
w = np.concatenate([np.ones([N,1]),np.random.normal(0,1,[N,1])],axis=1)
true_beta = np.array([1,1])
true_delta = np.array([-0.3,0.5])
true_xi = np.array([0.5,-0.3])
true_z = np.array([-1,1])
true_gamma = np.array([1,-1])
true_delta = np.array([1,1])
true_xi = np.array([1,1])
true_z = np.array([1,1])
true_gamma = np.array([1,1])
true_sigma_v=0.5
true_sigma_alpha = 0.5

# data simulation
myclip_a = 0
my_mean = np.dot(pi,true_delta)
my_std = np.exp(np.dot(pi,true_xi)/2)
a, b = (myclip_a - my_mean) / my_std, np.inf
u = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)

myclip_a = 0
my_mean = np.dot(w,true_z)
my_std = np.exp(np.dot(w,true_gamma)/2)
a, b = (myclip_a - my_mean) / my_std, np.inf
eta = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)

v = np.random.normal(0, true_sigma_v,[NT,])
alpha = np.random.normal(0, true_sigma_alpha, [N,])

y = np.dot(x,true_beta) + v + u + np.kron(eta, np.ones([T,])) + np.kron(alpha,np.ones([T,]))

# prior
sigma_beta_sqr = 10
sigma_delta_sqr = 10
sigma_xi_sqr = 10
sigma_z_sqr = 10
sigma_gamma_sqr = 10

# initialize
sigma_v_sqr = 0.5
sigma_alpha_sqr = 1.
beta = np.array([0.5,1.5])

delta = np.array([0.5,1.5])
xi = np.array([0.5,1.5])
delta = true_delta
xi = true_xi

z = np.array([1.,1.])
gamma = np.array([1.,1])
gamma = true_gamma

delta_ = alpha + eta

accpt_num_delta = 0
accpt_num_xi = 0
accpt_num_z = 0
accpt_num_gamma = 0
S = 11000
all_beta = np.zeros([S, K])
all_xi = np.zeros([S,K])
all_delta = np.zeros([S,K])
all_z = np.zeros([S,K])
all_gamma = np.zeros([S,K])
all_sigma_v_sqr = np.zeros([S,])
all_sigma_alpha_sqr = np.zeros([S,])


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
    args = (mu_eta,sigma_eta_sqr,sigma_alpha_sqr,ratio,sigma_,y,x,beta,u,N,T)
    mu_delta_, H_ = newton(delta_,F,G,H,args)
    delta__star = np.random.normal(mu_delta_,(-1/H_)**0.5)
    A = (F(delta__star,*args)/F(mu_delta_,*args))* np.exp(-0.5 * (delta__star - mu_delta_)**2 / (-1/H_))
    U = np.random.uniform(0,1,[N,])
    select = U < A
    delta_[select,:] = delta__star[select,:] #change value if over u
    # beta
    V_beta = inv((np.dot(x.T,x) * sigma_beta_sqr + sigma_v_sqr)/(sigma_v_sqr * sigma_beta_sqr))
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

    # delta
#     delta, accpt_num_delta = draw_delta(pi,xi,delta,u,sigma_delta_sqr,K,accpt_num_delta)
    delta, accpt_num_delta = HMC({'delta':delta, 'sigma_delta_sqr':sigma_delta_sqr,'pi':pi,'u':u,'xi':xi,'sigma_xi_sqr':sigma_xi_sqr}, 'delta', 0.01, 50, accpt_num_delta, loggrad_delta)

    # z
    z, accpt_num_z = HMC({'z':z, 'sigma_z_sqr':sigma_z_sqr,'w':w,'gamma':gamma,'eta':eta}, 'z', 0.01, 100, accpt_num_z, loggrad_z)
    
    # xi
##    xi, accpt_num_xi = draw_xi(pi, xi, delta, sigma_delta_sqr, u, sigma_xi_sqr, K, accpt_num_xi)
    xi, accpt_num_xi = HMC({'delta':delta, 'sigma_delta_sqr':sigma_delta_sqr,'pi':pi,'u':u,'xi':xi,'sigma_xi_sqr':sigma_xi_sqr}, 'xi', 0.001, 50, accpt_num_xi, loggrad_xi)
    
    # gamma
    gamma, accpt_num_gamma = HMC({'z':z, 'sigma_z_sqr':sigma_z_sqr,'w':w,'gamma':gamma,'eta':eta}, 'gamma', 0.001, 100, accpt_num_gamma, loggrad_gamma)
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
    
    
print('delta')
print(accpt_num_delta)
print('z')
print(accpt_num_z)
print('xi')
print(accpt_num_xi)
print('gamma')
print(accpt_num_gamma)
stop = timeit.default_timer()

print('Time: ', stop - start)
pd.DataFrame(all_beta).to_csv('sf2_with_panel_beta3_tkN1000')
pd.DataFrame(all_xi).to_csv('sf2_with_panel_xi3_tkN1000')
pd.DataFrame(all_delta).to_csv('sf2_with_panel_delta3_tkN1000')
pd.DataFrame(all_z).to_csv('sf2_with_panel_z3_tkN1000')
pd.DataFrame(all_gamma).to_csv('sf2_with_panel_gamma3_tkN1000')
pd.DataFrame(all_sigma_alpha_sqr).to_csv('sf2_with_panel_sigma_alpha_sqr3_tkN1000')
pd.DataFrame(all_sigma_v_sqr).to_csv('sf2_with_panel_sigma_v_sqr3_tkN1000')


