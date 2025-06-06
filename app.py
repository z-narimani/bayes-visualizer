import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.set_page_config(page_title="Bayes Explorer", layout="centered")

st.title("🎯 A simple visulization of Bayesian updates")
st.write("The effect of Likelihood and Beta prior on posterior distribution")

# Inputs
st.sidebar.header("🔧 Setting:")
alpha = st.sidebar.slider("α (prior alpha)", 0.1, 10.0, 2.0)
beta_param = st.sidebar.slider("β (prior beta)", 0.1, 10.0, 2.0)
n = st.sidebar.slider("Number of coin toss (n)", 1, 100, 10)
x = st.sidebar.slider("Number of heads (success)", 0, n, 5)

# Distributions
theta = np.linspace(0, 1, 500)
prior = beta.pdf(theta, alpha, beta_param)
posterior = beta.pdf(theta, alpha + x, beta_param + n - x)

# Plotting
fig, ax = plt.subplots()
ax.plot(theta, prior, label="Prior", color="blue")
ax.plot(theta, posterior, label="Posterior", color="green")
ax.set_xlabel("θ")
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)

# Summary
st.markdown(f"**Prior:** Beta({alpha}, {beta_param})")
st.markdown(f"**Posterior:** Beta({alpha + x}, {beta_param + n - x})")
