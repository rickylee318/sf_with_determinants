import numpy as np
import pandas as pd
import cupy as cp
from numpy.lib import index_tricks
import cupy
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
import timeit

start = timeit.default_timer()

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function to 1-D slices along the given axis.
    Args:
        func1d (function (M,) -> (Nj...)): This function should accept 1-D
            arrays. It is applied to 1-D slices of ``arr`` along the specified
            axis. It must return a 1-D ``cupy.ndarray``.
        axis (integer): Axis along which ``arr`` is sliced.
        arr (cupy.ndarray (Ni..., M, Nk...)): Input array.
        args: Additional arguments for ``func1d``.
        kwargs: Additional keyword arguments for ``func1d``.
    Returns:
        cupy.ndarray: The output array. The shape of ``out`` is identical to
            the shape of ``arr``, except along the ``axis`` dimension. This
            axis is removed, and replaced with new dimensions equal to the
            shape of the return value of ``func1d``. So if ``func1d`` returns a
            scalar ``out`` will have one fewer dimensions than ``arr``.
    .. seealso:: :func:`numpy.apply_over_axes`
    """
    ndim = arr.ndim
    axis = internal._normalize_axis_index(axis, ndim)
    inarr_view = cupy.moveaxis(arr, axis, -1)

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars
    inds = index_tricks.ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError(
            'Cannot apply_along_axis when any iteration dimensions are 0'
        )
    res = func1d(inarr_view[ind0], *args, **kwargs)
    if cupy.isscalar(res):
        # scalar outputs need to be transfered to a device ndarray
        res = cupy.asarray(res)

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = cupy.empty(inarr_view.shape[:-1] + res.shape, res.dtype)

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = func1d(inarr_view[ind], *args, **kwargs)

    # restore the inserted axes back to where they belong
    for i in range(res.ndim):
        buff = cupy.moveaxis(buff, -1, axis)

    return buff
   
def PMCMC_alpha(sigma_alpha_sqr,eta,u,N,T,y,x,beta,sigma_v_sqr):
    H = 5000-1 # particle numbers
    # sample u from N(mu_u, V_u)
    alpha_particles = norm.rvs(0, sigma_alpha_sqr**0.5,size=(H,N))
    alpha_particles = np.concatenate([alpha_particles, alpha.reshape(-1,1).T], axis=0)
    alpha_particles_ = np.kron(alpha_particles, np.ones([T,]))
#    y_til = (y-np.dot(x, beta)-np.kron(eta,np.ones(T,))).reshape([N,T]).mean(axis=1)

    # calculate weight
    w = norm.pdf((y-np.dot(x, beta)-np.kron(eta,np.ones([T,]))-alpha_particles_)/(sigma_v_sqr**0.5))/(sigma_v_sqr**0.5)
    w = w.reshape([H+1,N,T]).prod(axis=2)
    w = w/w.sum(axis=0)
    
#    new_alpha = np.zeros([N,])
    index = np.apply_along_axis(func1d=choice, axis=0, arr=w,h=H)
    new_alpha = alpha_particles[index,np.arange(N)]
#    for i in range(N):
#        new_alpha[i,] = np.random.choice(alpha_particles[:,i],p=w[:,i])
#    print('new_alpha_size')
#    print(new_alpha.shape())

    return new_alpha

def choice(weight,h):
    ind = cp.arange(h+1)
    return cp.random.choice(ind, size=1, replace=True, p=weight)

def PMCMC_eta_(w,gamma,pi,xi,z,alpha,u,N,T,y,x,beta,sigma_v_sqr,sigma_alpha_sqr,eta):
    H = 7000-1 # particle numbers
    # sample u from N(mu_u, V_u)
    V_eta = np.exp(np.dot(w, gamma))
    mu_eta = np.dot(w, z)
    myclip_a = 0
    my_mean = mu_eta
    my_std = V_eta** 0.5
    a, b = (myclip_a - my_mean) / my_std, np.inf * np.ones([N,])
    eta_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,N))
#    eta_particles = cp.asarray(eta_particles)
    eta_particles = np.concatenate([eta_particles,eta.reshape(-1,1).T], axis=0)
    eta_particles_ = np.kron(eta_particles, np.ones([T,]))
    
    alpha_particles = norm.rvs(0, sigma_alpha_sqr ** 0.5, size=(H,N))
#    alpha_particles = cp.asarray(alpha_particles)
    alpha_particles = np.concatenate([alpha_particles,alpha.reshape(-1,1).T], axis=0)
    alpha_particles_ = np.kron(alpha_particles, np.ones([T,]))   
    
    V_u = np.exp(np.dot(pi, xi))
    mu_u = np.dot(pi, delta)
    myclip_a = 0
    my_mean = mu_u
    my_std = V_u** 0.5
    a, b = (myclip_a - my_mean) / my_std, np.inf * np.ones([NT,])
    u_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,NT))
#    u_particles = cp.asarray(u_particles)
    u_particles = np.concatenate([u_particles, u.reshape(-1,1).T], axis=0)
    
    x_ = (y-np.dot(x, beta)-alpha_particles_-eta_particles_-u_particles)/(sigma_v_sqr**0.5)
    w = norm.pdf(x_)
    w_ = w.reshape([H+1,N,T]).prod(axis=2)
    w_ = w_/w_.sum(axis=0)

    w_ = cp.asarray(w_)
    index = apply_along_axis(func1d=choice, axis=0, arr=w_,h=H)
    index = index.get()
    new_alpha = alpha_particles[index,np.arange(N)]
    new_eta = eta_particles[index, np.arange(N)]
    new_u = u_particles[np.kron(index, np.ones([T,])).astype(int),np.arange(N*T)]
  

    return new_eta.flatten(), new_alpha.flatten(), new_u.flatten()
    
def gpu_normal_pdf(X):
    inv_sqrt_2pi = 0.3989422804014327
    pdf = inv_sqrt_2pi * cp.exp(-cp.square(X)/2)
    return pdf


def PMCMC_eta(w,gamma,pi,xi,z,alpha,u,N,T,y,x,beta,sigma_v_sqr,sigma_alpha_sqr,eta):
    H = 10000-1 # particle numbers
    # sample u from N(mu_u, V_u)
    V_eta = np.exp(np.dot(w, gamma))
    mu_eta = np.dot(w, z)
    myclip_a = 0
    my_mean = mu_eta
    my_std = V_eta** 0.5
    a, b = (myclip_a - my_mean) / my_std, np.inf * np.ones([N,])
    eta_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,N))
    eta_particles = cp.asarray(eta_particles)
    eta_particles = cp.concatenate([eta_particles,cp.asarray(eta).reshape(-1,1).T], axis=0)
    eta_particles_ = cp.kron(eta_particles, cp.ones([T,]))
    
    alpha_particles = cp.random.normal(0, sigma_alpha_sqr ** 0.5, size=(H,N))
    alpha_particles = cp.concatenate([alpha_particles,cp.asarray(alpha).reshape(-1,1).T], axis=0)
    alpha_particles_ = cp.kron(alpha_particles, cp.ones([T,]))   
    
    V_u = np.exp(np.dot(pi, xi))
    mu_u = np.dot(pi, delta)
    myclip_a = 0
    my_mean = mu_u
    my_std = V_u** 0.5
    a, b = (myclip_a - my_mean) / my_std, np.inf * np.ones([NT,])
    u_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,NT))
    u_particles = cp.asarray(u_particles)
    u_particles = cp.concatenate([u_particles, cp.asarray(u).reshape(-1,1).T], axis=0)
    
    x_ = (cp.asarray(y)-cp.dot(cp.asarray(x), cp.asarray(beta))-alpha_particles_-eta_particles_-u_particles)/(sigma_v_sqr**0.5)
    w = gpu_normal_pdf(x_)
    w_ = w.reshape([H+1,N,T]).prod(axis=2)
    w_ = w_/w_.sum(axis=0)
    
    index = apply_along_axis(func1d=choice, axis=0, arr=w_,h=H)
    
    new_alpha = alpha_particles[index,cp.arange(N)].get()
    new_eta = eta_particles[index, cp.arange(N)].get()
    new_u = u_particles[cp.kron(index, cp.ones([T,])).astype(int),cp.arange(N*T)].get()
  

    return new_eta.flatten(), new_alpha.flatten(), new_u.flatten()

    
    
    
    
# PMCMC
def PMCMC_u(pi,xi,delta,alpha, eta, NT, T, y, x, beta,sigma_v_sqr,u):
    H = 5000-1 # particle numbers
    # sample u from N(mu_u, V_u)
    V_u = np.exp(np.dot(pi, xi))
    mu_u = np.dot(pi, delta)
    myclip_a = 0
    my_mean = mu_u
    my_std = V_u** 0.5
    a, b = (myclip_a - my_mean) / my_std, np.inf * np.ones([NT,])
    u_particles = truncnorm.rvs(a,b,loc = my_mean, scale = my_std, size = (H,NT))
    u_particles = np.concatenate([u_particles, u.reshape(-1,1).T], axis=0)

    # calculate weight
    w = norm.pdf((y-np.dot(x, beta)-u_particles-np.kron(alpha,np.ones([T,]))-np.kron(eta,np.ones(T,)))/(sigma_v_sqr**0.5))
    w = w/w.sum(axis=0)
    
    new_u = np.zeros([NT,])
    for i in range(NT):
        new_u[i,] = np.random.choice(u_particles[:,i],p=w[:,i])
    return new_u

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
for i in range(S):
    print(i)
    ### Posterior
    # beta
    V_beta = inv((np.dot(x.T,x) * sigma_beta_sqr + sigma_v_sqr)/(sigma_v_sqr * sigma_beta_sqr))
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
    
    # alpha
#    scale = sigma_alpha_sqr * sigma_v_sqr / (T * sigma_alpha_sqr + sigma_v_sqr)
#    y_bar = y-np.dot(x, beta)-np.kron(eta, np.ones(T,))-u
#    loc = scale / sigma_v_sqr * y_bar.reshape([N,T]).sum(axis=1)
#    alpha = norm.rvs(loc = loc, scale = scale)
#    alpha = PMCMC_alpha(sigma_alpha_sqr,eta,u,N,T,y,x,beta,sigma_v_sqr)
    
    # eta
#    V_eta = 1/(np.exp(-np.dot(w, gamma))+T/sigma_v_sqr)
#    mu_eta = V_eta * ((y-np.dot(x, beta)- u -np.kron(alpha,np.ones(T,))).reshape([N,T]).sum(axis=1)/sigma_v_sqr + np.exp(-np.dot(w, gamma))*np.dot(w,z))
#    myclip_a = 0
#    my_mean = mu_eta
#    my_std = V_eta** 0.5
#    a, b = (myclip_a - my_mean) / my_std, np.inf*np.ones([N,])
#    eta = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)    
    eta, alpha, u =PMCMC_eta(w,gamma,pi,xi,z,alpha,u,N,T,y,x,beta,sigma_v_sqr,sigma_alpha_sqr,eta)
    
    # u
#    V_u = 1/(np.exp(-np.dot(pi, xi))+1/sigma_v_sqr)
#    mu_u = V_u * ((y-np.dot(x, beta)-np.kron(eta,np.ones(T,))-np.kron(alpha,np.ones(T,)))/sigma_v_sqr + np.exp(-np.dot(pi, xi))*np.dot(pi,delta))
#    myclip_a = 0
#    my_mean = mu_u
#    my_std = V_u** 0.5
#    a, b = (myclip_a - my_mean) / my_std, np.inf*np.ones([NT,])
#    u = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)
#    u = PMCMC_u(pi,xi,delta,alpha, eta, NT, T, y, x, beta,sigma_v_sqr,u)

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
pd.DataFrame(all_beta).to_csv('sf2_with_panel_beta3_gpuN1000H7000')
pd.DataFrame(all_xi).to_csv('sf2_with_panel_xi3_gpuN1000H7000')
pd.DataFrame(all_delta).to_csv('sf2_with_panel_delta3_gpuN1000H7000')
pd.DataFrame(all_z).to_csv('sf2_with_panel_z3_gpuN1000H7000')
pd.DataFrame(all_gamma).to_csv('sf2_with_panel_gamma3_gpuN1000H7000')
pd.DataFrame(all_sigma_alpha_sqr).to_csv('sf2_with_panel_sigma_alpha_sqr3_gpuN1000H7000')
pd.DataFrame(all_sigma_v_sqr).to_csv('sf2_with_panel_sigma_v_sqr3_gpuN1000H7000')



