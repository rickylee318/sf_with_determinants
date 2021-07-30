# SF4wD #

four-component stochastic frontier model with determinants

## Motivation ##

This package was developed to complement four-component stochastic frontier that consider 

determinants in mean and variance parameters of inefficiency distributions 

by Ruei-Chi Lee.

## Installation ##

Install via `$ pip install 4SFwD`

## Features ##

* **SF4wD**: main.py - set method and model to run simulation or real data

* **HMC**: Hamilton Monte Carlo designed for determinants parameters. 

* **DA**: Data augmentation for the model

* **TK**: Two-parametrization method originally proposed by Tsiona and Kunmbhaker (2014) for four-component model without determinants. 

* **PMCMC**: Particle MCMC for the model (perferred approach) - speed up by GPU parallel computation


## Example ##

Here is how you run a simulation estimation for a four-component stochastic frontier model via PMCMC: 

- Parameter setting guideline in the SF4wD.py

- Simulation data only offers stochastic frontier model that consider determinants in both mean and variance parameter of inefficiencies.

```python
import SF4wD
#model:str - different way to consider determinants
#method:str - different Bayesian method to estimate the model
#data_name : str - simulation data or data in data/.
#S : int - MCMC length
#H : int - number of particles in PMCMC
#gpu: boolean - use parallel computation to run PMCMC
#save: boolean - save MCMC data
my_model = SF4wD(model = 'D', method = 'PMCMC', data_name ='',S=10, H=100, gpu=False, save=False)
my_model.run()
```

output: 
```python
                  mean     sd  hpd_3%  hpd_97%  mcse_mean  mcse_sd  ess_mean  ess_sd  ess_bulk  ess_tail  r_hat
beta0            2.412  0.093   2.318    2.555      0.046    0.035       4.0     4.0       7.0      10.0    NaN
beta1            1.078  0.074   0.977    1.242      0.023    0.017      10.0    10.0      10.0      10.0    NaN
xi0              0.580  0.043   0.531    0.652      0.014    0.011       9.0     9.0       8.0      10.0    NaN
xi1              0.694  0.127   0.479    0.867      0.073    0.058       3.0     3.0       3.0      10.0    NaN
delta0           0.141  0.072   0.013    0.273      0.023    0.019      10.0     8.0      10.0      10.0    NaN
delta1           0.774  0.137   0.620    0.984      0.079    0.063       3.0     3.0       3.0      10.0    NaN
z0              -0.461  0.716  -1.844    0.609      0.376    0.291       4.0     4.0       4.0      10.0    NaN
z1               2.728  0.889   1.268    3.941      0.459    0.354       4.0     4.0       4.0      10.0    NaN
gamma0           0.662  0.092   0.500    0.773      0.052    0.041       3.0     3.0       3.0      10.0    NaN
gamma1           0.412  0.061   0.349    0.519      0.021    0.015       9.0     9.0       9.0      10.0    NaN
sigma_alpha_sqr  1.377  0.178   1.095    1.693      0.075    0.057       6.0     6.0       6.0      10.0    NaN
sigma_v_sqr      2.575  2.523   1.290    9.515      1.062    0.793       6.0     6.0       3.0      10.0    NaN
```

## License ##

Ruei-Chi Lee is the main author and contributor.

Bug reports, feature requests, questions, rants, etc are welcome, preferably 
on the github page. 
