import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.set_page_config(page_title="Bayes Explorer", layout="centered")

st.title("🎯 A simple visulization of Bayesian updates")
st.write("The effect of Likelihood and Beta prior on posterior distribution")

# User inputs
# Sidebar inputs
st.sidebar.header("Simulation Parameters")
alpha_prior = st.sidebar.slider("Prior α (Beta)", 0.1, 10.0, 2.0)
beta_prior = st.sidebar.slider("Prior β (Beta)", 0.1, 10.0, 2.0)
heads = st.sidebar.number_input("Number of Heads (Successes)", min_value=0, value=6)
trials = st.sidebar.number_input("Total Flips (Trials)", min_value=1, value=10)

tails = trials - heads
if tails < 0:
    st.error("Number of heads cannot exceed total flips.")
    st.stop()

# Posterior parameters
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

# x-axis for plotting
x = np.linspace(0, 1, 500)

# Calculate distributions
prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(x, alpha_post, beta_post)
likelihood = x**heads * (1 - x)**tails
likelihood_scaled = likelihood / np.max(likelihood)  # Rescale to max=1

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, prior_pdf, label="Prior (Beta)", lw=2)
ax.plot(x, likelihood_scaled, label="Likelihood (rescaled)", lw=2)
ax.plot(x, posterior_pdf, label="Posterior (Beta)", lw=2)
ax.set_xlabel("Probability of Heads (p)")
ax.set_ylabel("Density / Relative Likelihood")
ax.set_title("Bayesian Updating with Coin Flips")
ax.legend()

st.pyplot(fig)
