import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import beta

# Set random seed 
SEED = 42
np.random.seed(SEED)
az.style.use("arviz-darkgrid")

# Load the dataset
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv")

# Separate data by version 
ret_1_A = data[data['version'] == 'gate_30']['retention_1']
ret_1_B = data[data['version'] == 'gate_40']['retention_1']
ret_7_A = data[data['version'] == 'gate_30']['retention_7']
ret_7_B = data[data['version'] == 'gate_40']['retention_7']


# Define priors
a_prior, b_prior = 1, 1

# Calculate posterior parameters for gate_30 (control group)
a_30 = a_prior + ret_1_A.sum()
b_30 = b_prior + len(ret_1_A) - ret_1_A.sum()

# Calculate posterior parameters for gate_40 (treatment group)
a_40 = a_prior + ret_1_B.sum()
b_40 = b_prior + len(ret_1_B) - ret_1_B.sum()

# Plot 
x = np.linspace(0, 1, 1000)
plt.figure(figsize=(12, 6))
plt.plot(x, beta.pdf(x, a_30, b_30), label='Gate 30 (Control)', color='blue')
plt.plot(x, beta.pdf(x, a_40, b_40), label='Gate 40 (Treatment)', color='orange')
plt.title('1-Day Retention Rates: Beta Posterior Distributions')
plt.xlabel('Retention Rate')
plt.ylabel('Density')
plt.legend()
plt.show()


samp_30 = np.random.beta(a_30, b_30, 10000)
samp_40 = np.random.beta(a_40, b_40, 10000)
prob_40_1 = (samp_40 > samp_30).mean()
print(f"Probability that Gate 40 has a higher 1-day retention rate: {prob_40_1:.4f}")



with pm.Model() as model_1:
    # Priors
    p_30 = pm.Uniform('p_30', 0, 1)
    p_40 = pm.Uniform('p_40', 0, 1)
    
    # Likelihoods
    obs_30 = pm.Bernoulli('obs_30', p_30, observed=ret_1_A)
    obs_40 = pm.Bernoulli('obs_40', p_40, observed=ret_1_B)
    
    # Sampling
    trace_1 = pm.sample(2000, step=pm.Metropolis(), chains=2, random_seed=SEED)

# Plot  distributions for 1-day retention
az.plot_posterior(trace_1, var_names=['p_30', 'p_40'], hdi_prob=0.95)
plt.suptitle("Posterior Distributions for 1-Day Retention")
plt.show()

# Extract posterior samples
post_30 = trace_1.posterior.p_30.values.flatten()
post_40 = trace_1.posterior.p_40.values.flatten()

# Compare PyMC posterior samples
prob_40_1 = (post_40 > post_30).mean()
print(f"Probability that Gate 40 has a higher 1-day retention rate: {prob_40_1:.4f}")

# Analytical (Beta Posterior)
a_30_7 = a_prior + ret_7_A.sum()
b_30_7 = b_prior + len(ret_7_A) - ret_7_A.sum()

a_40_7 = a_prior + ret_7_B.sum()
b_40_7 = b_prior + len(ret_7_B) - ret_7_B.sum()

x = np.linspace(0, 1, 1000)
plt.figure(figsize=(12, 6))
plt.plot(x, beta.pdf(x, a_30_7, b_30_7), label='Gate 30 (Control)', color='blue')
plt.plot(x, beta.pdf(x, a_40_7, b_40_7), label='Gate 40 (Treatment)', color='orange')
plt.title('7-Day Retention Rates: Beta Posterior Distributions')
plt.xlabel('Retention Rate')
plt.ylabel('Density')
plt.legend()
plt.show()

samp_30_7 = np.random.beta(a_30_7, b_30_7, 10000)
samp_40_7 = np.random.beta(a_40_7, b_40_7, 10000)
prob_40_7 = (samp_40_7 > samp_30_7).mean()
print(f"Probability that Gate 40 has a higher 7-day retention rate: {prob_40_7:.4f}")

# 7-Day Retention
with pm.Model() as model_7:
    # Priors
    p_30_7 = pm.Uniform('p_30_7', 0, 1)
    p_40_7 = pm.Uniform('p_40_7', 0, 1)
    
    # Likelihoods
    obs_30_7 = pm.Bernoulli('obs_30_7', p_30_7, observed=ret_7_A)
    obs_40_7 = pm.Bernoulli('obs_40_7', p_40_7, observed=ret_7_B)
    
    # Sampling
    trace_7 = pm.sample(2000, step=pm.Metropolis(), chains=2, random_seed=SEED)

# Plot posterior distributions for 7-day retention
az.plot_posterior(trace_7, var_names=['p_30_7', 'p_40_7'], hdi_prob=0.95)
plt.suptitle("Posterior Distributions for 7-Day Retention")
plt.show()

# Extract posterior samples
post_30_7 = trace_7.posterior.p_30_7.values.flatten()
post_40_7 = trace_7.posterior.p_40_7.values.flatten()

# Compare posterior samples
prob_40_7 = (post_40_7 > post_30_7).mean()
print(f"Probability that Gate 40 has a higher 7-day retention rate: {prob_40_7:.4f}")
