import numpy as np
from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Tuple
from src.supplyseer.eoq import eoq

@dataclass
class EOQParameters:
    """Data class to hold EOQ parameters and their ranges."""
    demand: float
    order_cost: float
    holding_cost: float
    min_demand: float
    max_demand: float
    min_order_cost: float
    max_order_cost: float
    min_holding_cost: float
    max_holding_cost: float
    initial_demand: float
    initial_order_cost: float
    initial_holding_cost: float

class ProbabilityDistribution:
    """Class for handling probability distributions and calculations."""
    
    @staticmethod
    def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Calculate normal probability density function."""
        return (1/np.sqrt(2 * np.pi * std**2)) * np.exp(-.5 * ((x - mean)**2) / std**2)

class BayesianEOQ:
    """Main class for Bayesian EOQ calculations."""
    
    def __init__(self, parameters: EOQParameters, n_param_values: int):
        self.params = parameters
        self.n_param_values = n_param_values
        self._initialize_parameter_ranges()
        self._calculate_posteriors()
        
    def _initialize_parameter_ranges(self):
        """Initialize parameter ranges for calculations."""
        self.d_range = np.linspace(self.params.min_demand, self.params.max_demand, self.n_param_values)
        self.a_range = np.linspace(self.params.min_order_cost, self.params.max_order_cost, self.n_param_values)
        self.h_range = np.linspace(self.params.min_holding_cost, self.params.max_holding_cost, self.n_param_values)

    def _bayesian_posterior(self, x: float, parameter_range: np.ndarray, initial_guess: float) -> np.ndarray:
        """Calculate posterior distribution for a parameter."""
        prior = ProbabilityDistribution.normal_pdf(parameter_range, x, x * 0.1)
        likelihood = ProbabilityDistribution.normal_pdf(initial_guess, parameter_range, x * 0.1)
        unnormalized_posterior = prior * likelihood
        marginal_likelihood = np.trapz(unnormalized_posterior, parameter_range)
        return unnormalized_posterior / marginal_likelihood

    def _calculate_posteriors(self):
        """Calculate posterior distributions for all parameters."""
        self.posterior_d = self._bayesian_posterior(
            self.params.demand, self.d_range, self.params.initial_demand)
        self.posterior_a = self._bayesian_posterior(
            self.params.order_cost, self.a_range, self.params.initial_order_cost)
        self.posterior_h = self._bayesian_posterior(
            self.params.holding_cost, self.h_range, self.params.initial_holding_cost)

    def calculate_credible_interval(self) -> List[float]:
        """Calculate credible interval for EOQ."""
        quantiles = [0.025, 0.975]
        
        def get_interval_values(posterior: np.ndarray, param_range: np.ndarray) -> List[float]:
            indices = [np.argmin(np.abs(posterior - np.quantile(posterior, q))) for q in quantiles]
            return [param_range[i] for i in indices]
        
        interval_d = get_interval_values(self.posterior_d, self.d_range)
        interval_a = get_interval_values(self.posterior_a, self.a_range)
        interval_h = get_interval_values(self.posterior_h, self.h_range)
        
        return [eoq(d, a, h) 
                for d in interval_d 
                for a in interval_a 
                for h in interval_h]

    def calculate_distribution(self, parameter_space: str = 'full', n_simulations: int = 1) -> List[float]:
        """Calculate EOQ distribution based on parameter space."""
        if parameter_space == 'full':
            return [eoq(d, a, h) 
                    for d in self.d_range 
                    for a in self.a_range 
                    for h in self.h_range]
        elif parameter_space == 'montecarlo':
            eoq_montecarlo = eoq(self.d_range, self.a_range, self.h_range)
            return np.random.choice(eoq_montecarlo, size=n_simulations, replace=True).tolist()
        return []

    def get_most_probable_values(self) -> Dict[str, float]:
        """Get most probable values for EOQ parameters."""
        return {
            'eoq': eoq(
                self.d_range[np.argmax(self.posterior_d)],
                self.a_range[np.argmax(self.posterior_a)],
                self.h_range[np.argmax(self.posterior_h)]
            ),
            'd': self.d_range[np.argmax(self.posterior_d)],
            'a': self.a_range[np.argmax(self.posterior_a)],
            'h': self.h_range[np.argmax(self.posterior_h)]
        }

    def get_least_probable_values(self) -> Dict[str, Dict[str, float]]:
        """Get least probable values for EOQ parameters."""
        return {
            'min': {
                'eoq': eoq(self.params.min_demand, self.params.min_order_cost, self.params.min_holding_cost),
                'd': self.params.min_demand,
                'a': self.params.min_order_cost,
                'h': self.params.min_holding_cost
            },
            'max': {
                'eoq': eoq(self.params.max_demand, self.params.max_order_cost, self.params.max_holding_cost),
                'd': self.params.max_demand,
                'a': self.params.max_order_cost,
                'h': self.params.max_holding_cost
            }
        }

    def compute_full_analysis(self, parameter_space: str = 'full', n_simulations: int = 1) -> Dict:
        """Compute complete Bayesian EOQ analysis."""
        return {
            'bayesian_eoq_most_probable': self.get_most_probable_values(),
            'bayesian_eoq_min_least_probable': self.get_least_probable_values()['min'],
            'bayesian_eoq_max_least_probable': self.get_least_probable_values()['max'],
            'eoq_distribution': self.calculate_distribution(parameter_space, n_simulations),
            'eoq_credible_interval': self.calculate_credible_interval()
        }

# Example usage:
def bayesian_eoq_full(d: float, a: float, h: float,
                      min_d: float, max_d: float, 
                      min_a: float, max_a: float,
                      min_h: float, max_h: float,
                      initial_d: float, initial_a: float, initial_h: float,
                      n_param_values: int, 
                      parameter_space: str = 'full',
                      n_simulations: int = 1) -> Dict:
    """Wrapper function for backward compatibility."""
    parameters = EOQParameters(
        demand=d, order_cost=a, holding_cost=h,
        min_demand=min_d, max_demand=max_d,
        min_order_cost=min_a, max_order_cost=max_a,
        min_holding_cost=min_h, max_holding_cost=max_h,
        initial_demand=initial_d, initial_order_cost=initial_a,
        initial_holding_cost=initial_h
    )
    
    bayesian_eoq = BayesianEOQ(parameters, n_param_values)
    return bayesian_eoq.compute_full_analysis(parameter_space, n_simulations)
