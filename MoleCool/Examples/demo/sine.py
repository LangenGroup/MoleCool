"""
only documenting how to plot a sine wave
========================================

This tutorial shows how to create and plot a simple sine wave.

The top-level docstring becomes the intro text.
"""

# %%
# Step 1 — imports
# ----------------
import numpy as np
import matplotlib.pyplot as plt

# %%
# Step 2 — generate data
# ----------------------
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

# %%
# Step 3 — plot
# -------------
plt.plot(x, y, label="sin(x)")
plt.title("Simple sine wave")
plt.legend()
plt.show()