  [![PyPI](https://img.shields.io/pypi/v/supplyseer.svg)](https://pypi.org/project/supplyseer/)
  [![Python Version](https://img.shields.io/pypi/pyversions/supplyseer.svg)](https://pypi.org/project/supplyseer/)
  [![License](https://img.shields.io/github/license/supplyseer-ai/supplyseer.svg)](https://github.com/supplyseer-ai/supplyseer/blob/main/LICENSE)
  [![Tests](https://github.com/supplyseer-ai/supplyseer/actions/workflows/run-tests.yml/badge.svg)](https://github.com/supplyseer-ai/supplyseer/actions)
  
<div align="center>
<p>
    <a target="_blank">
      <img width="100%" src="https://github.com/jakorostami/supplyseer/blob/develop/assets/supplyseerfront.png" alt="SupplySeer Vision banner"></a>
  </p>

# <div align="center"> SupplySeer </div>
Welcome to version 0.2 (pre-release alpha)!

⚠️ **Pre-release Software Notice**: This library is currently in pre-release alpha (v0.2). The repo may undergo significant changes before the 1.0.0 release. While the statistical implementations are sound, we recommend testing thoroughly before using in production environments.

`supplyseer` is a Python library focused on providing the tools and methods for real-world Supply Chain & Logistics challenges. <br>
<br>
You'll find Bayesian Economic Order Quantity (dynamical stochastic EOQ), Probabilistic Bayesian Networks, Neural Networks, <br>
Principal Component Analaysis, time series models like ARIMA, and evaluation metrics for models and for information content. <br>
<br>
Supplyseer provides Permutation Complexity as a metric for time series analysis but also Manipulability Index and Hurst Exponent and many more.
<br>
<br>
**Check Tutorials section for guides and examples!**

## <div align="center"> Installation </div>

You can install `supplyseer` directly from PyPI:

```bash
pip install supplyseer==0.2.2
```

For development installation, see our [Contributing Guide](CONTRIBUTING.md).

## <div align="center"> Features </div>
🚀 Features

* Advanced Forecasting Models: ARIMA, Neural Networks, and Mixture Density Networks <br>
* Uncertainty Modeling: Bayesian Networks and Probabilistic Models <br>
* Inventory Optimization: Dynamic Bayesian EOQ and Traditional EOQ <br>
* Time Series Analysis: Complex metrics and tools for deep analysis <br>
* Supply Chain Optimization: Scheduling and routing solutions <br>

## <div align="center"> Models </div>
Below are some models listed

| Model | Use case |
| --- | --- |
| [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) | Time Series |
| [Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network) | Uncertainty Modeling, Prediction, and Causal Inference |
| [Bayesian EOQ](https://en.wikipedia.org/wiki/Economic_order_quantity) | Economic Order Quantity with distributions instead of fixed values |
| [Neural Network](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) | Machine Learning Modeling | 
| [Mixture Density Network](https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/) | Probabilistic Machine Learning Modeling with multi-modal data |
| [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) | Embeddings in Machine Learning (dimensionality reduction) |
| [Hawkes Process](https://en.m.wikipedia.org/wiki/Hawkes_process) | Multivariate Hawkes process in supply chains models how disruptions in one area can trigger related issues across the network, predicting ripple effects from initial events. |
| [Supply Chain Digital Twin Network](https://towardsdatascience.com/what-is-a-supply-chain-digital-twin-e7a8cd9aeb75?gi=120a86059486) | A Supply Chain Digital Twin of the real Supply Chain is a computer model that represents the processes and components of the real one |
| [Game Theory Module](https://pubsonline.informs.org/doi/pdf/10.1287/educ.1063.0023) | Cooperative Supply Chain game with Coalition based gaming among players (suppliers, manufacturers, retailers) and you. |
<br>

## <div align="center"> Tools & Metrics </div>

| Name | Use case | 
| --- | --- |
| [Time Upsampling](https://en.wikipedia.org/wiki/Upsampling) | Good when you have missing dates between samples |
| [Taken's Embeddings](https://en.wikipedia.org/wiki/Takens%27s_theorem) | Extract the dynamics of a time series/signal |
| Economic Order Quantity | This is the basic function of EOQ that returns a value while the Bayesian EOQ is a dynamic model |
| [Manipulability Index](https://iaeme.com/MasterAdmin/Journal_uploads/IJMET/VOLUME_6_ISSUE_6/IJMET_06_06_002.pdf) | Another way of measuring volatility and stability of a time series. Also a measure of "responsiveness", e.g. promotional campaigns as interventions |
| [Hurst Exponent R/S](https://en.wikipedia.org/wiki/Hurst_exponent) | Measure long-term memory or autocorrelation in a time series |
| [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) | Measures the unpredictability or randomnesss |
| [Permutation Entropy](https://materias.df.uba.ar/dnla2019c1/files/2019/03/permutation_entropy.pdf) | Quantifies the diversity of patterns in the ordinal structure of a time series. It is the first output of `permutation_complexity()`|
| [Statistical Complexity](https://arxiv.org/pdf/1009.1498) | Measures the structural complexity of a system. It combines entropy with disequilibrium (a measure of structure). It is the second output of `permutation_complexity()`|

<br>

## <div align="center"> Optimization </div>
This library also supports basic optimization with Google's `ortools`. See below example for a Truck Driver scheduling problem.

<p>
    <a target="_blank">
      <img width="100%" src="https://github.com/jakorostami/supplyseer/blob/develop/assets/truckdriver.png" alt="Truck routing banner"></a>
  </p>

  [Truck Driver Scheduling problem](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/truck-driver-routing.ipynb) - You have some truck drivers that you need to schedule for over a time window of 3 days with 3 shifts. Morning, afternoon, and evening. If they had the evening shift they cannot have the morning shift the day after because they need to rest. Also, they have to deliver at least 2 shifts during the 3 day window. <br>
  <br>
  Problem: Schedule truck drivers over a 3-day window with multiple constraints: <br>
    
    * Three shifts per day (Morning, Afternoon, Evening)
    * Rest period required between evening and next morning shift
    * Minimum 2 shifts per driver over 3 days
    
  <br>
  
 [Demand & Inventory Control](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/demand-inventory-control.ipynb) - A Supply Chain department for a retail company needs to balance their inventory and demand such that there is also enough inventory to match the demand but the inventory is not allowed to go below a certain level nor above a certain level. <br>
 <br>
 Problem: Optimize inventory levels while,

    * Meeting demand requirements
    * Maintaining minimum safety stock
    * Respecting maximum storage capacity

## <div align="center"> Tutorials & Examples </div>
In this section you'll find Tutorials and Examples, they exist in respective subfolder. Their differences is that tutorials are comprehensive and examples are just quick demonstrations of the modules.

| Tutorial | Description |
| --- | --- |
| [Supply Chain Digital Twin](https://github.com/supplyseer-ai/supplyseer/blob/develop/tutorials/supplychain-digitaltwin-network/digitaltwin.ipynb) | Simulating Digital Twins with classical policy, diffusion based reorders, and hybrid mode. Kinetic energy represents how well the system is "flowing" in balancing inventory. |
| [Stochastic Demand Simulation with decaying Geometric Brownian Motion](https://github.com/supplyseer-ai/supplyseer/blob/develop/tutorials/demand-simulation/demand.ipynb) | Simulate stochastic demand processes using a Geometric Brownian Motion (GBM) model with decay, produce mean path trajectories, simulate thousands of paths, and find the final distribution of your demand at end time T|
| [Game Theoretic Supply Chain](https://github.com/supplyseer-ai/supplyseer/blob/develop/tutorials/game-theory/coopgame.ipynb) | Create a Cooperative Game in your Supply Chain with your suppliers and manufacturers to find partnerships, coalitions, and Nash equilibrium |
| [Geopolitical Risk API & GDELT Monitor API](https://github.com/supplyseer-ai/supplyseer/blob/develop/tutorials/geopolitical-risk/geopolitical.ipynb) | Explore the supplyseer API for Geopolitical Risk assessments and do Sentiment Analysis with the GDELT Monitor API and HuggingFace | 
<br>

| Example | Description |
| --- | --- |
| [Bayesian Economic Order Quantity Modeling](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/order-quantity.ipynb) | Simulate Bayesian EOQ with Approximate Bayesian Computation with Normal distributions. Dynamic and Stochastic approach with credible intervals. |
| [Multivariate Hawkes Demand and Inventory](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/hawkes-supplychain.ipynb) | Creates a self-exciting Supply Chain simulation of demand and inventory process. |
| [Probabilistic Bayesian Network](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/supplychain-example.ipynb) | Model your expertise, knowledge, or data as a Probabilistic Network to do causal analysis, counterfactual analysis, or probabilistic modeling |
| [Vector Field Dynamics Analysis of Demand and Inventory](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/vector-field.ipynb) | Use Physics based approaches to your demand and inventory analysis by using Vector Fields to find equilibrium states, convergence, or divergence paths in your Supply Chain system. | 
| [Truck Driver Scheduling Optimization](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/truck-driver-routing.ipynb) | Schedule the most optimal way for your truck drivers with realistic constraints. |
| [Demand and Inventory Control](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/demand-inventory-control.ipynb) | Find the most optimal way for your demand and inventory that holds your costs. |
| [Topological Time Series](https://github.com/supplyseer-ai/supplyseer/blob/develop/examples/time-series.ipynb) | Use financial stock tickers, Tesla and Apple, and Takens Embeddings with PCA to do phase space reconstruction of the signal. |


## <div align="center"> Contributing 🤝 </div>



We love contributions! Whether you're fixing bugs, adding features, or improving documentation, your help makes `supplyseer` better for everyone.

Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and join our friendly community. No contribution is too small, and all contributors are valued!

Want to help but not sure how? See our [Issues](https://github.com/supplyseer-ai/supplyseer/issues) or start a [Discussion](https://github.com/supplyseer-ai/supplyseer/discussions). We're happy to guide you! 🎲✨



    
</div>
