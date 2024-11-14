.. supplyseer documentation master file, created by
   sphinx-quickstart on Thu Nov 14 09:06:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SupplySeer's documentation!
====================================

SupplySeer is a Python library for computational supply chain analytics, combining stochastic processes, 
optimization, and machine learning approaches to model and analyze supply chain dynamics.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials/index
   api/index
   examples/index

Features
--------
* Stochastic demand modeling using Geometric Brownian Motion
* Inventory optimization and control
* Supply chain network analysis
* Risk assessment and simulation
* Advanced visualization tools

Quick Install
------------
.. code-block:: bash

   pip install supplyseer

Quick Example
------------
.. code-block:: python

   from supplyseer.models import StochasticDemandProcess, DemandSimConfig

   # Configure demand simulation
   config = DemandSimConfig(
       initial_demand=600000,
       drift=0.05,
       volatility=0.1,
       decay_rate=0.05,
       time_horizon=2.0,
       n_paths=1000,
       n_steps=50,
       random_seed=42
   )

   # Create and run simulation
   sim = StochasticDemandProcess(config)
   results = sim.simulate()

   # Visualize results
   sim.plot_results()

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Demand Modeling
=============

StochasticDemandProcess
----------------------

.. autoclass:: supplyseer.models.StochasticDemandProcess
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
------------

.. autoclass:: supplyseer.models.DemandSimConfig
   :members:
   :undoc-members:
   :show-inheritance:

Results
-------

.. autoclass:: supplyseer.models.DemandPathResult
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from supplyseer.models import StochasticDemandProcess, DemandSimConfig
   
   # Create configuration
   config = DemandSimConfig(
       initial_demand=600000,
       drift=0.05,
       volatility=0.1,
       decay_rate=0.05,
       time_horizon=2.0,
       n_paths=1000,
       n_steps=50
   )
   
   # Initialize and run simulation
   process = StochasticDemandProcess(config)
   results = process.simulate()
   
   # Access results
   mean_demand = results.mean_path
   quantiles = process.get_quantile_paths([0.05, 0.95])
   
   # Visualize
   process.plot_results()

Advanced Usage
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Configure with specific parameters
   config = DemandSimConfig(
       initial_demand=600000,
       drift=0.05,
       volatility=0.1,
       decay_rate=0.05,
       time_horizon=2.0,
       n_paths=1000,
       n_steps=50,
       random_seed=42
   )
   
   process = StochasticDemandProcess(config)
   results = process.simulate()
   
   # Calculate probability of demand falling below threshold
   threshold = 500000
   prob_below = process.get_demand_threshold_probability(threshold)
   
   # Access final distribution statistics
   stats = results.final_distribution
   print(f"Mean final demand: {stats['mean']:.2f}")
   print(f"Standard deviation: {stats['std']:.2f}")
   print(f"Median: {stats['median']:.2f}")