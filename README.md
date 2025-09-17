# NetBayesABM

**Agent-based models of plant–pollinator networks with Bayesian inference (ABC).**

NetBayesABM is a Python library for simulating agent-based models (ABMs) of ecological networks, focusing on plant–pollinator interactions. It provides tools to:

- Initialize agents (plants and pollinators) with different spatial configurations.  
- Construct and evolve bipartite networks dynamically.  
- Define and sample prior distributions (Gamma, Exponential) for interaction parameters.  
- Visualize abundances, priors, networks, and degree distributions.  
- Evaluate simulated networks against empirical data using multiple metrics.

---

## 🚀 Installation

Once published on PyPI, you can install with:

```bash
pip install NetBayesABM
```

For development (local clone):

```bash
git clone https://github.com/galeanojav/NetBayesABM.git
cd NetBayesABM
pip install -e .
```

📖 Quick Example

```bash
import numpy as np
import pandas as pd
from netbayesabm.classes import Environment_plant, Environment_pol
from netbayesabm.modelling import initial_network, update_totalinks, remove_zero
from netbayesabm.visualization import plot_agents, plot_priors

# --- Define plant environment (random positions) ---
df_plants = pd.DataFrame({
    "Plant_id": [1, 2, 3],
    "Plant_sp": ["rose", "daisy", "sunflower"],
    "X": [0, 0, 0],
    "Y": [0, 0, 0],
    "Plant_sp_complete": ["Rosa sp.", "Bellis perennis", "Helianthus annuus"]
})
envp = Environment_plant(df_plants, random_position=True, xmin=0, xmax=10, ymin=0, ymax=10)

# --- Define pollinators ---
df_pols = pd.DataFrame({
    "Pol_id": [1, 2],
    "Specie": ["bee", "butterfly"],
    "x": [2.0, 8.0],
    "y": [3.0, 7.0],
    "Radius": [3.0, 3.0]
})
envpol = Environment_pol(df_pols)

# --- Build bipartite network ---
B = initial_network([p.Pol_id for p in envpol.pol_list],
                    [pl.Plant_id for pl in envp.plant_list])

# --- Run short simulation ---
update_totalinks(50, envpol, envp, B, xmin=0, xmax=10, ymin=0, ymax=10)
remove_zero(B)

# --- Priors ---
prior_specialist = pd.Series(np.random.gamma(2, 2, size=1000))
prior_generalist = pd.Series(np.random.gamma(2, 2, size=1000))
plot_priors(prior_specialist, prior_generalist, "example_priors")
```
📊 Features
	•	Agent and environment classes (Environment_plant, Environment_pol).
	•	Network initialization and evolution functions.
	•	Visualization utilities for abundances, priors, and degree distributions.
	•	Evaluation metrics (Hellinger, Jensen–Shannon, KL, Wasserstein, etc.).
	•	Example notebooks for a quick start.

 👩‍💻 Authors
	•	Javier Galeano — [javier.galeano@upm.es]
	•	Blanca Arroyo-Correa — [blanca.arroyo@ebd.csic.es]
	•	Mario Castro — [marioc@iit.comillas.edu]


📜 License

This project is licensed under the MIT License — see the LICENSE file for details.
