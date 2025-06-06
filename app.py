import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.set_page_config(page_title="Bayes Explorer", layout="centered")

st.title("🎯 A simple visulization of Bayesian updates")
st.write("The effect of Likelihood and Beta prior on posterior distribution")

# User inputs
st.sidebar.header("Setting:")
a = st.sidebar.slider("Prior α (alpha)", 0.1, 10.0, 2.0, 0.1)
b = st.sidebar.slider("Prior β (beta)", 0.1, 10.0, 2.0, 0.1)
n_heads = st.sidebar.slider("Number of heads (successes)", 0, 100, 6)
n_total = st.sidebar.slider("Total coin tosses (trials)", n_heads, 100, 10)

# Compute posterior
posterior_a = a + n_heads
posterior_b = b + (n_total - n_heads)

# x range for plotting
x = np.linspace(0, 1, 500)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot prior
prior_pdf = beta.pdf(x, a, b)
ax.plot(x, prior_pdf, label=f"Prior: Beta({a}, {b})", color="blue", linestyle="--")

# Plot likelihood (relative, not normalized)
likelihood = x**n_heads * (1 - x)**(n_total - n_heads)
likelihood /= np.max(likelihood)
ax.plot(x, likelihood, label=f"Likelihood (relative): Heads={n_heads}, Total={n_total}", color="orange", linestyle=":")

# Plot posterior
posterior_pdf = beta.pdf(x, posterior_a, posterior_b)
ax.plot(x, posterior_pdf, label=f"Posterior: Beta({posterior_a}, {posterior_b})", color="green")

# Final plot settings
ax.set_xlabel("θ (probability of heads)")
ax.set_ylabel("Density")
ax.set_title("Bayesian Update for Coin Tossing")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown("""
**Interpretation**
- The **prior** shows your belief about the coin's fairness before observing any data.
- The **likelihood** shows how likely the observed data is for different values of θ.
- The **posterior** combines both to update your belief about θ after seeing the data.
""")
