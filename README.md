### 4 component stochastic frontier with determinants of inefficiencies
===========================================================

#### Intro: The model includes long-term, short-term inefficiency and firm heterognenity. Moreover, we incorporate factors to explain both persistent and transient technical ineﬃciencies

---------------------------------------

#### Method:
- data augmentation: sf_with_panel.py (set estimator='data augmentation')
- PMCMC (particle Markov chain Monte Carlo): sf_with_panel.py (set estimator = 'PMCMC')
- PMCMC (particle Markov chain Monte Carlo) spped up with GPU: sf_with_panel_gpu.py
- two-step estimator: sf_with_panel_tk.py



