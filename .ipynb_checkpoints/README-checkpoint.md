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

my_model = SF4wD(model='D',method='PMCMC')
my_model.run()




## License ##

Ruei-Chi Lee is the main author and contributor.

Bug reports, feature requests, questions, rants, etc are welcome, preferably 
on the github page. 