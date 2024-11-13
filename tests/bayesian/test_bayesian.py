import numpy as np
from src.supplyseer.bayesian_eoq import (
    EOQParameters,
    ProbabilityDistribution,
    BayesianEOQ,
    bayesian_eoq_full
)

def test_that_bayesian_eoq_and_computation_works():
    """
    Test the complete Bayesian EOQ computation workflow including probability calculations,
    posterior computations, and final EOQ calculations.
    """
    # Test parameters
    d = 100
    a = 10
    h = 1
    d_range = np.linspace(50, 250, 100)
    a_range = np.linspace(5, 25, 100)
    h_range = np.linspace(0.5, 2, 100)
    initial_d = 100
    initial_a = 10
    initial_h = 1
    n_param_values = 100
    parameter_space = "full"
    n_simulations = 1000

    # Test normal PDF calculation
    normal_probability_density = ProbabilityDistribution.normal_pdf(d_range, d, d*.1)
    assert len(normal_probability_density) == 100, "Length of normal_probability_density should be 100"
    assert np.isnan(normal_probability_density).any() == False, "normal_probability_density should not contain NaN values"

    # Test posterior calculations
    params = EOQParameters(
        demand=d,
        order_cost=a,
        holding_cost=h,
        min_demand=min(d_range),
        max_demand=max(d_range),
        min_order_cost=min(a_range),
        max_order_cost=max(a_range),
        min_holding_cost=min(h_range),
        max_holding_cost=max(h_range),
        initial_demand=initial_d,
        initial_order_cost=initial_a,
        initial_holding_cost=initial_h
    )
    
    bayesian_calc = BayesianEOQ(params, n_param_values)
    
    # Check posterior distributions
    assert len(bayesian_calc.posterior_d) == 100, "Length of posterior_d should be 100"
    assert len(bayesian_calc.posterior_a) == 100, "Length of posterior_a should be 100"
    assert len(bayesian_calc.posterior_h) == 100, "Length of posterior_h should be 100"
    assert np.isnan(bayesian_calc.posterior_d).any() == False, "posterior_d should not contain NaN values"
    assert np.isnan(bayesian_calc.posterior_a).any() == False, "posterior_a should not contain NaN values"
    assert np.isnan(bayesian_calc.posterior_h).any() == False, "posterior_h should not contain NaN values"

    # Test full EOQ calculation
    eoq = bayesian_eoq_full(
        d=d, 
        a=a, 
        h=h, 
        min_d=min(d_range), 
        max_d=max(d_range),
        min_a=min(a_range), 
        max_a=max(a_range), 
        min_h=min(h_range), 
        max_h=max(h_range),
        initial_d=initial_d, 
        initial_a=initial_a, 
        initial_h=initial_h,
        n_param_values=n_param_values, 
        parameter_space=parameter_space, 
        n_simulations=n_simulations
    )
    
    # Check EOQ result structure
    assert eoq is not None, "eoq should not be None"
    assert isinstance(eoq, dict), "eoq should be a dictionary"
    
    # Check required keys
    required_keys = [
        'bayesian_eoq_most_probable',
        'bayesian_eoq_min_least_probable',
        'bayesian_eoq_max_least_probable',
        'eoq_distribution',
        'eoq_credible_interval'
    ]
    for key in required_keys:
        assert key in eoq, f"eoq should contain '{key}'"
    
    # Check types of result components
    assert isinstance(eoq['bayesian_eoq_most_probable'], dict), \
        "eoq['bayesian_eoq_most_probable'] should be a dictionary"
    assert isinstance(eoq['bayesian_eoq_min_least_probable'], dict), \
        "eoq['bayesian_eoq_min_least_probable'] should be a dictionary"
    assert isinstance(eoq['bayesian_eoq_max_least_probable'], dict), \
        "eoq['bayesian_eoq_max_least_probable'] should be a dictionary"
    assert isinstance(eoq['eoq_distribution'], list), \
        "eoq['eoq_distribution'] should be a list"
    assert isinstance(eoq['eoq_credible_interval'], list), \
        "eoq['eoq_credible_interval'] should be a list"
    
    # Check distribution is not empty
    assert len(eoq['eoq_distribution']) > 0, \
        "eoq['eoq_distribution'] should not be empty"
    
    # Check values are reasonable
    assert all(v > 0 for v in eoq['eoq_distribution']), \
        "All EOQ values should be positive"
    assert all(v > 0 for v in eoq['eoq_credible_interval']), \
        "All credible interval values should be positive"
    assert eoq['bayesian_eoq_most_probable']['eoq'] > 0, \
        "Most probable EOQ should be positive"
