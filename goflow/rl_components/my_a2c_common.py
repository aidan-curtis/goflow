from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from rl_games.common import vecenv

from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from goflow.rl_components.my_experience import ExperienceBuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics
from rl_games.algos_torch import  model_builder
from rl_games.interfaces.base_algorithm import  BaseAlgorithm
import numpy as np
import time
import gym
import zuko
from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn
import torch.distributed as dist
from typing import List, Dict, Tuple
from time import sleep
from rl_games.common import common_losses
import math
import matplotlib
matplotlib.use('Agg')
import random
import matplotlib.pyplot as plt
import io
from PIL import Image
from scipy.optimize import minimize
import shutil
from scipy.stats import norm
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, Bounds

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')

@dataclass
class DomainRange:
    names: List[str]
    low: List[float]
    high: List[float]

    @staticmethod
    def from_dict(d: Dict[str, Tuple[float, float]]) -> DomainRange:
        names = []
        low = []
        high = []

        for k, v in d.items():
            names.append(k)
            low.append(v[0])
            high.append(v[1])
        
        return DomainRange(names, low, high)
        

#######################################################
# Custom distribution for the action space

import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import io
from PIL import Image



class Distr(torch.distributions.Distribution):
    
    def denormalize_samples(self, samples):
        return samples
    
    def normalize_samples(self, samples):
        return samples
    
    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError
    
    def log_prob(self, value):
        raise NotImplementedError
        
    def entropy(self, num_samples=10000):
        # Draw samples from the distribution
        samples = self.rsample((num_samples,))
        
        # Calculate log probabilities for these samples
        log_probs = self.log_prob(samples)
        
        # Entropy is the negative expected log probability
        entropy_estimate = -log_probs.mean()
        return entropy_estimate
    

class UniformDist(Distr):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.device = "cpu"
        self.ndim = len(low)
        self.uniform = torch.distributions.Uniform(torch.tensor(low), torch.tensor(high))
    
    def volume(self):
        """
        Calculate the volume of the hyper-rectangle defined by self.low and self.high.
        """
        return torch.prod(torch.tensor(self.high) - torch.tensor(self.low))
    
    def rsample(self, sample_shape=torch.Size()):
        return self.uniform.rsample(sample_shape)
    
    def log_prob(self, value):
        return self.uniform.log_prob(value.to(self.device)).sum(-1)
    
class MultivariateNormalDist(Distr):
    def __init__(self, mean, cov, low, high):
        self.mv_mean = mean
        self.mv_cov = cov
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.ndim = self.mv_mean.shape[0]
        self.multinorm = torch.distributions.MultivariateNormal(self.mv_mean, self.mv_cov)
        
    
    def entropy(self, **kwargs):
        return self.multinorm.entropy()

    def rsample(self, sample_shape=torch.Size()):
        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        samples = self.multinorm.rsample(sample_shape)
        
        # Check which samples are out of bounds
        valid_mask = torch.all((samples >= self.low) & (samples <= self.high), dim=-1)
        invalid_mask = ~valid_mask
        
        # Generate uniform samples for the invalid ones
        uniform_samples = torch.rand(invalid_mask.sum(), self.ndim, device=samples.device)
        uniform_samples = uniform_samples * (self.high - self.low) + self.low
        
        # Replace invalid samples with uniform samples
        samples[invalid_mask] = uniform_samples
        
        return samples
    
    def log_prob(self, value):
        return self.multinorm.log_prob(value)
    

class NormFlowDist(Distr):
    def __init__(self, low, high, ndim):
        bins = 8
        self.ndim = ndim
        self.device = "cuda"
        self.low = low.to(self.device)
        self.high = high.to(self.device)
        
        self.scale = 10.0
        maf = zuko.flows.MAF(
            features=self.ndim,
            context=0,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            hidden_features=(64, 64),
            transforms=3
        )
        
        self.flow = zuko.flows.Flow(maf.transform.inv, maf.base).to(self.device)

    
    def normalize_samples(self, samples):
        """Normalize samples into standardized space"""

        if samples.ndim == 1:
            samples.reshape(1, -1)

        return ((samples - (self.low+self.high)/2.0  ) / (self.high - self.low)) * self.scale


    def denormalize_samples(self, samples):
        """Denormalize samples back in their true space"""
        samples = samples.to(self.device)
        if samples.ndim == 1:
            samples.reshape(1, -1)

        return (self.high - self.low) * samples / self.scale + (self.low+self.high)/2.0

    def log_prob(self, x):
        norm_x = self.normalize_samples(x.to(self.device))
        return self.flow().log_prob(norm_x.type(torch.FloatTensor).to(self.device))

           
    def get_params(self):
        return self.flow.parameters()
    
    def clone(self):
        # Create a new instance
        new_instance = NormFlowDist(self.low.clone(), self.high.clone(), self.ndim)
        
        # Copy the flow parameters
        new_instance.flow.load_state_dict(self.flow.state_dict())
        
        return new_instance
    
    def rsample(self, sample_shape=torch.Size()):
         
        # Normalized Truncated Normal distribution in [0, 1]
        n_valids = 0
        n_samples = sample_shape[0]
        non_valid_mask = torch.ones((n_samples)).bool()
        samples = torch.zeros((n_samples, self.ndim)).type(torch.FloatTensor).to(self.device)

        n_iters = 0
        while n_valids < n_samples:
            if(n_iters>0):
                print("regenerating iter {}".format(str(n_iters)))        
            norm_samples, _ = self.flow().rsample_and_log_prob((non_valid_mask.int().sum(),))
            norm_samples = norm_samples.type(torch.FloatTensor).to(self.device)
            samples[non_valid_mask] = self.denormalize_samples(norm_samples)
            mask_low = torch.greater_equal(samples, self.low.view(1,-1))
            mask_high = torch.less_equal(samples, self.high.view(1,-1))
            per_sample_mask_with_dim = torch.cat([mask_low, mask_high], dim=-1) # (n_samples, 2*ndim)
            per_sample_mask = torch.all(per_sample_mask_with_dim, dim=-1)
            non_valid_mask = ~per_sample_mask
            n_valids = per_sample_mask.int().sum()
            n_iters += 1
            

        if n_iters >= 10:
            print('WARNING! Sampling through the truncated normal took {n_iters} >= 10 iterations for resampling.')
        
        return samples
    
    
class BetasDist(Distr):
    def __init__(self, alphas, betas, low, high):
        super().__init__(validate_args=False)
        self.alphas = alphas
        self.betas = betas
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.dists = [torch.distributions.Beta(a, b) for a, b in zip(self.alphas, self.betas)]
        self.ndim = len(self.dists)

    def rsample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        samples = torch.stack([d.rsample(sample_shape) for d in self.dists], dim=-1).type(torch.FloatTensor)
        
        # Transform samples to the desired range
        output = self.low + (self.high - self.low) * samples
        
        # Check which samples are out of bounds
        valid_mask = torch.all((output >= self.low) & (output <= self.high), dim=-1)
        invalid_mask = ~valid_mask
        
        # Generate uniform samples for the invalid ones
        uniform_samples = torch.rand(invalid_mask.sum(), self.ndim, device=samples.device)
        uniform_samples = uniform_samples * (self.high - self.low) + self.low
        
        # Replace invalid samples with uniform samples
        output[invalid_mask] = uniform_samples
        
        return output
    
    def log_prob(self, value):
        # Transform value back to [0, 1] range
        transformed_value = (value - self.low) / (self.high - self.low)
        log_probs = torch.stack([d.log_prob(v) for d, v in zip(self.dists, transformed_value.t())])
        return log_probs.sum(dim=0)

    def to_flat(self):
        """Convert distribution parameters to a flat array."""
        return torch.cat([self.alphas, self.betas])

    @classmethod
    def from_flat(cls, flat_params, low, high):
        """Create a BetasDist instance from a flat array of parameters."""
        ndim = len(low)
        alphas = flat_params[:ndim]
        betas = flat_params[ndim:]
        return cls(alphas, betas, low, high)

    def kl_divergence(self, other):
        """
        Compute KL divergence between this BetasDist and another BetasDist or UniformDist.
        """
        kl_div = 0
        for i in range(self.ndim):
            p_dist = torch.distributions.Beta(self.alphas[i], self.betas[i])
            
            if isinstance(other, UniformDist):
                # Scale down the uniform distribution to [0, 1]
                q_dist = torch.distributions.Uniform(0, 1)
                # Compute KL divergence
                kl_div += torch.distributions.kl_divergence(p_dist, q_dist)
                # Adjust for the change in scale
                kl_div -= torch.log(self.high[i] - self.low[i])
            elif isinstance(other, BetasDist):
                q_dist = torch.distributions.Beta(other.alphas[i], other.betas[i])
                kl_div += torch.distributions.kl_divergence(p_dist, q_dist)
                # Adjust for different bounds if necessary
                if not torch.allclose(self.low[i], other.low[i]) or not torch.allclose(self.high[i], other.high[i]):
                    kl_div += torch.log((other.high[i] - other.low[i]) / (self.high[i] - self.low[i]))
            else:
                raise ValueError(f"Unsupported distribution type: {type(other)}")
        return kl_div



#######################################################
# Custom methods for the distributional reinforcement learning

class DRMethod:
    def __init__(self):
        pass
    
    def update(self, contexts, returns):
        pass
    
    def get_train_dist(self):
        raise NotImplementedError
    
    def get_test_dist(self):
        raise NotImplementedError
        
    
class FullDR(DRMethod):
    def __init__(self, domain_range: DomainRange, middle_percentage: float = 1.0, **kwargs):
        self.domain_range = domain_range
        self.middle_percentage = max(0.0, min(1.0, middle_percentage))  # Ensure it's between 0 and 1
        
    def get_train_dist(self):
        low = torch.tensor(self.domain_range.low)
        high = torch.tensor(self.domain_range.high)
        range_width = high - low
        margin = range_width * (1 - self.middle_percentage) / 2

        train_low = low + margin
        train_high = high - margin

        return UniformDist(train_low.tolist(), train_high.tolist())
    
    def get_test_dist(self):
        return UniformDist(self.domain_range.low, self.domain_range.high)
    
class NoDR(DRMethod):
    def __init__(self, domain_range: DomainRange, **kwargs):
        self.domain_range = domain_range
        self.mid_range = [(low + high) / 2 for low, high in zip(domain_range.low, domain_range.high)]

    def get_train_dist(self):
        return UniformDist(self.mid_range, self.mid_range)


    def get_test_dist(self):
        return UniformDist(self.domain_range.low, self.domain_range.high)
    
class LSDR(DRMethod):
    def __init__(self, domain_range: DomainRange, learning_rate=1e-2, num_training_iters=10, alpha=None, **kwargs):
        self.domain_range = domain_range
        self.num_training_iters = num_training_iters
        self.ndim = len(domain_range.low)
        self.alpha = alpha
        
        # Initialize mean to the center of the normalized space (0.5 for each dimension)
        self.norm_mean = nn.Parameter(torch.full((self.ndim,), 0.5, dtype=torch.float32))
        
        # Initialize covariance matrix to a diagonal matrix with small variances in normalized space
        norm_initial_cov = torch.eye(self.ndim) * 0.2
        self.norm_cov_cholesky = nn.Parameter(torch.linalg.cholesky(norm_initial_cov))
        
        self.optimizer = torch.optim.Adam([self.norm_mean, self.norm_cov_cholesky], lr=learning_rate)
        
        self.norm_target_dist = UniformDist(torch.zeros(self.ndim), torch.ones(self.ndim))

    def get_train_dist(self):
        denorm_mean = self.denormalize(self.norm_mean)
        denorm_cov = self.get_norm_covariance() * (torch.tensor(self.domain_range.high) - torch.tensor(self.domain_range.low)).unsqueeze(0) * (torch.tensor(self.domain_range.high) - torch.tensor(self.domain_range.low)).unsqueeze(1)
        return MultivariateNormalDist(denorm_mean, denorm_cov, self.domain_range.low, self.domain_range.high)

    def get_test_dist(self):
        return UniformDist(torch.tensor(self.domain_range.low), torch.tensor(self.domain_range.high))

    def get_norm_covariance(self):
        return torch.matmul(self.norm_cov_cholesky, self.norm_cov_cholesky.t())

    def normalize(self, x):
        return (x - torch.tensor(self.domain_range.low)) / (torch.tensor(self.domain_range.high) - torch.tensor(self.domain_range.low))

    def denormalize(self, x):
        return x * (torch.tensor(self.domain_range.high) - torch.tensor(self.domain_range.low)) + torch.tensor(self.domain_range.low)

    def update(self, contexts, returns):
        # Normalize contexts
        norm_contexts = self.normalize(contexts)
        
        # Normalize returns
        norm_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        print("Normalized returns:", norm_returns)
        print("Normalized contexts:", norm_contexts)
        
        for _ in range(self.num_training_iters):
            self.optimizer.zero_grad()

            norm_current_dist = self.get_train_dist()
            
            # Compute log probabilities
            log_probs = norm_current_dist.log_prob(norm_contexts)  # Normalize contexts before computing log probabilities
            # Compute loss (negative weighted log likelihood) with normalized returns
            m = norm_returns.transpose(1, 0) * log_probs
            loss = -torch.mean(m)
            
            # Add regularization to keep the distribution within the normalized range
            kl_loss = self.compute_kl_divergence(norm_current_dist, self.norm_target_dist)
            total_loss = loss + self.alpha * kl_loss  # You can adjust the coefficient

            total_loss.backward()
            self.optimizer.step()

            # Project mean back to the valid range [0, 1]
            with torch.no_grad():
                self.norm_mean.data.clamp_(0, 1)

            # Print mean and covariance in the original space
            denormalized_mean = self.denormalize(self.norm_mean.data)
            denormalized_cov = self.get_norm_covariance() * (torch.tensor(self.domain_range.high) - torch.tensor(self.domain_range.low)).unsqueeze(0) * (torch.tensor(self.domain_range.high) - torch.tensor(self.domain_range.low)).unsqueeze(1)
            print(f"LSDR Update - Mean: {denormalized_mean}, Covariance: {denormalized_cov}")
            
            # Print the loss
            print(f"LSDR Update - Loss: {loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
            
    def compute_kl_divergence(self, p, q):
        # Compute KL divergence between p (current distribution) and q (target distribution)
        # This is an approximation using Monte Carlo sampling
        samples = p.rsample((1000,))
        log_p = p.log_prob(samples)
        log_q = q.log_prob(self.normalize(samples))
        return torch.mean(log_p - log_q)

class DORAEMON(DRMethod):
    def __init__(self, domain_range: DomainRange, 
                 success_threshold: float,
                 kl_upper_bound: float = 0.1,
                 init_beta_param: float = 100.,
                 success_rate_condition: float = 0.5,
                 hard_performance_constraint: bool = True,
                 train_until_performance_lb: bool = True,
                 **kwargs):
        self.domain_range = domain_range
        self.success_threshold = success_threshold
        self.success_rate_condition = success_rate_condition
        self.kl_upper_bound = kl_upper_bound
        self.train_until_performance_lb = train_until_performance_lb
        self.hard_performance_constraint = hard_performance_constraint
        self.train_until_done = False 
        self.ndim = len(domain_range.low)
        
        self.min_bound = 0.8
        self.max_bound = init_beta_param + 10
        
        # Initialize distributions
        self.current_distr = self._create_initial_distribution(init_beta_param)
        self.target_distr = self._create_target_distribution()
        
        self.current_iter = 0
        self.distr_history = []

    def _create_initial_distribution(self, init_beta_param):
        return BetasDist(torch.ones(self.ndim) * init_beta_param, torch.ones(self.ndim) * init_beta_param, self.domain_range.low, self.domain_range.high)

    def _create_target_distribution(self):
        return UniformDist(self.domain_range.low, self.domain_range.high)

    def get_train_dist(self):
        return self.current_distr

    def get_test_dist(self):
        return self.target_distr

    def get_feasible_starting_distr(self, x0_opt, obj_fn, obj_fn_prime, kl_constraint_fn, kl_constraint_fn_prime):
        """
        Solves the inverted problem
        max J(phi_i+1) s.t. KL(phi_i+1 || phi_i) < eps
        to find an initial feasible distribution
        """
        def negative_obj_fn_with_grad(x_opt):
            try:
                obj_val = obj_fn(x_opt)
                obj_grad = obj_fn_prime(x_opt)
                
                # Check for invalid values
                if np.any(np.isnan(obj_val)) or np.any(np.isinf(obj_val)):
                    return np.inf, np.zeros_like(x_opt)
                if np.any(np.isnan(obj_grad)) or np.any(np.isinf(obj_grad)):
                    return obj_val, np.zeros_like(x_opt)
                    
                return -1 * obj_val, -1 * obj_grad
            except Exception as e:
                print(f"Warning: Error in objective function: {e}")
                return np.inf, np.zeros_like(x_opt)

        def safe_kl_constraint_fn(x_opt):
            try:
                val = kl_constraint_fn(x_opt)
                return np.clip(val, -1e10, 1e10)  # Clip to prevent extreme values
            except Exception as e:
                print(f"Warning: Error in KL constraint function: {e}")
                return np.inf

        def safe_kl_constraint_fn_prime(x_opt):
            try:
                grad = kl_constraint_fn_prime(x_opt)
                # Clip gradients to prevent numerical instability
                return np.clip(grad, -1e10, 1e10)
            except Exception as e:
                print(f"Warning: Error in KL constraint gradient: {e}")
                return np.zeros_like(x_opt)

        constraints = []
        constraints.append(
            NonlinearConstraint(
                fun=safe_kl_constraint_fn,
                lb=-np.inf,
                ub=self.kl_upper_bound-1e-5,
                jac=safe_kl_constraint_fn_prime,
                keep_feasible=True,
            )
        )

        # Add bounds to prevent extreme values
        bounds = Bounds(
            lb=-1e3 * np.ones_like(x0_opt),
            ub=1e3 * np.ones_like(x0_opt)
        )

        start = time.time()
        print("Starting optimization 2")
        
        try:
            result = minimize(
                negative_obj_fn_with_grad,
                x0_opt,
                method="trust-constr",
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={
                    "gtol": 1e-4,
                    "xtol": 1e-6,
                    "maxiter": 100,
                    "initial_tr_radius": 1.0,  # Start with a smaller trust region
                    "initial_constr_penalty": 1.0
                }
            )
        except Exception as e:
            print(f"Optimization failed with error: {e}")
            return None, None, False

        print(f"scipy inverted problem optimization time (s): {round(time.time() - start, 2)}")

        if not result.success:
            print(f"Optimization failed with message: {result.message}")
            return None, None, False
        else:
            feasible_x0_opt = result.x
            curr_step_kl = safe_kl_constraint_fn(feasible_x0_opt)
            
            # Verify the result is valid
            if np.any(np.isnan(feasible_x0_opt)) or np.any(np.isinf(feasible_x0_opt)):
                print("Warning: Optimization returned invalid values")
                return None, None, False
                
            return feasible_x0_opt, curr_step_kl, True


    def update(self, contexts, returns):
        self.current_iter += 1

        print("Updating DORAEMON")

        # Convert to numpy and ensure double precision
        contexts = torch.tensor(contexts, dtype=torch.float64)
        returns = torch.tensor(returns, dtype=torch.float64)

        print("Contexts shape:", contexts.shape)
        print("Returns shape:", returns.shape)
        print("Contexts min/max:", contexts.min().item(), contexts.max().item())
        print("Returns min/max:", returns.min().item(), returns.max().item())


        
        """
            2. Optimize KL(phi_i+1 || phi_target) s.t. J(phi_i+1) > performance_bound & KL(phi_i+1 || phi_i) < KL_bound
        """
        constraints = []


        def kl_constraint_fn(x_opt):
            """Compute KL-divergence between current and proposed distribution."""
            x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = BetasDist.from_flat(x, self.domain_range.low, self.domain_range.high)
            kl_divergence = self.current_distr.kl_divergence(proposed_distr)
            return kl_divergence.detach().numpy() 

        def kl_constraint_fn_prime(x_opt):
            """Compute the derivative for the KL-divergence (used for scipy optimizer)."""
            x_opt = torch.tensor(x_opt, requires_grad=True)
            x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = BetasDist.from_flat(x, self.domain_range.low, self.domain_range.high)
            kl_divergence = self.current_distr.kl_divergence(proposed_distr)
            grads = torch.autograd.grad(kl_divergence, x_opt)
            return np.concatenate([g.detach().numpy() for g in grads])

        constraints.append(
            NonlinearConstraint(
                fun=kl_constraint_fn,
                lb=-np.inf,
                ub=self.kl_upper_bound,
                jac=kl_constraint_fn_prime,
                keep_feasible=self.hard_performance_constraint
            )
        )

        def performance_constraint_fn(x_opt):
            """Compute the expected performance under the proposed distribution."""
            x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = BetasDist.from_flat(x, self.domain_range.low, self.domain_range.high)
            
            log_prob_proposed = proposed_distr.log_prob(contexts)
            log_prob_current = self.current_distr.log_prob(contexts)
            
            importance_sampling = torch.exp(log_prob_proposed - log_prob_current)
            
            if torch.any(torch.isnan(importance_sampling)) or torch.any(torch.isinf(importance_sampling)):
                print("Warning: NaN or Inf in importance sampling")
                print("log_prob_proposed:", log_prob_proposed)
                print("log_prob_current:", log_prob_current)
                importance_sampling = torch.nan_to_num(importance_sampling, nan=1.0, posinf=1.0, neginf=1.0)
            
            perf_values = torch.tensor(returns.detach() >= self.success_threshold, dtype=torch.float64)
            performance = torch.mean(importance_sampling * perf_values)
            
            if torch.isnan(performance) or torch.isinf(performance):
                print("Warning: NaN or Inf in performance")
                performance = torch.tensor(0.0, dtype=torch.float64)
            
            return performance.detach().numpy()

        def performance_constraint_fn_prime(x_opt):
            """Compute the derivative for the performance-constraint (used for scipy optimizer)."""
            x_opt = torch.tensor(x_opt, requires_grad=True)
            x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = BetasDist.from_flat(x, self.domain_range.low, self.domain_range.high)
            
            log_prob_proposed = proposed_distr.log_prob(contexts)
            log_prob_current = self.current_distr.log_prob(contexts)
            
            importance_sampling = torch.exp(log_prob_proposed - log_prob_current)
            
            if torch.any(torch.isnan(importance_sampling)) or torch.any(torch.isinf(importance_sampling)):
                print("Warning: NaN or Inf in importance sampling (prime)")
                importance_sampling = torch.nan_to_num(importance_sampling, nan=1.0, posinf=1.0, neginf=1.0)
            
            perf_values = torch.tensor(returns.detach() >= self.success_threshold, dtype=torch.float64)
            performance = torch.mean(importance_sampling * perf_values)
            
            if torch.isnan(performance) or torch.isinf(performance):
                print("Warning: NaN or Inf in performance (prime)")
                return np.zeros_like(x_opt.detach().numpy())
            
            grads = torch.autograd.grad(performance, x_opt)
            grad_np = np.concatenate([g.detach().numpy() for g in grads])
            
            if np.any(np.isnan(grad_np)) or np.any(np.isinf(grad_np)):
                print("Warning: NaN or Inf in gradients")
                grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            return grad_np

        constraints.append(
            NonlinearConstraint(
                fun=performance_constraint_fn,
                lb=self.success_rate_condition-1e-4,  # scipy would still complain if x0 is very close to the boundary
                ub=np.inf,
                jac=performance_constraint_fn_prime,
                keep_feasible=self.hard_performance_constraint
            )
        )



        def objective_fn(x_opt):
            """Minimize KL-divergence between the current and the target distribution,
                s.t. previously defined constraints."""
            x_opt = torch.tensor(x_opt, requires_grad=True, dtype=torch.float64)
            x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = BetasDist.from_flat(x, self.domain_range.low, self.domain_range.high)
            kl_divergence = proposed_distr.kl_divergence(self.target_distr)
            
            if not torch.isfinite(kl_divergence):
                print(f"Warning: Non-finite KL divergence detected: {kl_divergence}")
                return float('inf'), np.zeros_like(x_opt.detach().numpy())
            
            grads = torch.autograd.grad(kl_divergence, x_opt, create_graph=True)
            grad_np = np.concatenate([g.detach().numpy() for g in grads])
            
            if not np.isfinite(grad_np).all():
                print(f"Warning: Non-finite gradient detected: {grad_np}")
                return float('inf'), np.zeros_like(x_opt.detach().numpy())
            
            return kl_divergence.detach().numpy(), grad_np


        x0 = self.current_distr.to_flat()
        x0_opt = self._inv_sigmoid(x0, self.min_bound, self.max_bound)
        
        """
            Skip DORAEMON optimization at the beginning until performance_lower_bound is reached 
        """
        if self.train_until_performance_lb and not self.train_until_done:
            if performance_constraint_fn(x0_opt) < self.success_rate_condition:
                # Skip DORAEMON update
                print(f'--- DORAEMON iter {self.current_iter} skipped as performance lower bound has not been reached yet. Mean reward {performance_constraint_fn(x0_opt)} < {self.success_rate_condition}')
                return
            else:
                # Skip iterations only once, until you reach it the first time
                self.train_until_done = True
                self.n_iter_skipped = 0


        """
            Start from a feasible distribution within the trust region Kl(p||.) < eps
        """
        if performance_constraint_fn(x0_opt) < self.success_rate_condition:
            # Performance constraint not satisfied. Find a different initial distribution within the current trust region
            max_perf_x0_opt, curr_step_kl, success = self.get_feasible_starting_distr(x0_opt=x0_opt,
                                                                                        obj_fn=performance_constraint_fn,
                                                                                        obj_fn_prime=performance_constraint_fn_prime,
                                                                                        kl_constraint_fn=kl_constraint_fn,
                                                                                        kl_constraint_fn_prime=kl_constraint_fn_prime)
            if success:
                if performance_constraint_fn(max_perf_x0_opt) >= self.success_rate_condition:
                    # Feasible distribution found, Go on with this new starting distribution
                    x0_opt = max_perf_x0_opt
                    x0 = self._sigmoid(x0_opt, self.min_bound, self.max_bound)

                else:
                    # No feasible distribution within the trust region has been found
                    # Keep training with the max performance distribution within the trust region
                    new_x = self._sigmoid(max_perf_x0_opt, self.min_bound, self.max_bound)
                    self.current_distr = BetasDist.from_flat(new_x, self.domain_range.low, self.domain_range.high)
                    print(f'No distribution within the trust region satisfies the performance_constraint. ' \
                            f'Keep training with the max performant distribution in the trust region: {new_x.detach().numpy()} ' \
                            f'Est reward constraint value: {performance_constraint_fn(max_perf_x0_opt)} < {self.success_rate_condition}')
                    return

            else:
                # Inverse opt. problem had an unexpected error
                print('Warning! inverted optimization problem NOT successful.')
        
        print("Starting optimization...")
        
        try:
            result = minimize(
                objective_fn,
                x0_opt,
                method="trust-constr",
                jac=True,
                constraints=constraints,
                options={"gtol": 1e-4, "xtol": 1e-6, 'maxiter': 100},
            )

            print(f"Optimization result: {result}")
            new_x_opt = result.x
            
            # Check validity of new optimum found
            if not result.success:
                print('Warning! optimization NOT successful.')
                
                # If optimization process was not a success
                old_f = objective_fn(x0_opt)[0]
                constraints_satisfied = [const.lb <= const.fun(result.x) <= const.ub for const in constraints]

                if not (all(constraints_satisfied) and result.fun < old_f):  # keep old parameters if update was unsuccessful
                    print(f"Warning! Update effectively unsuccessful, keeping old values parameters.")
                    new_x_opt = x0_opt

            new_x = self._sigmoid(new_x_opt, self.min_bound, self.max_bound)
            self.current_distr = BetasDist.from_flat(new_x, self.domain_range.low, self.domain_range.high)
            print(f"New distribution parameters: {new_x}")
            

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            print("Keeping current distribution")
    
        

    def _sigmoid(self, x, lb=0, up=1):
        """sigmoid of x"""
        x = x if torch.is_tensor(x) else torch.tensor(x)
        sig = (up-lb)/(1+torch.exp(-x)) + lb
        return sig

    def _inv_sigmoid(self, x, lb=0, up=1):
        """return sigmoid^-1(x)"""
        x = x if torch.is_tensor(x) else torch.tensor(x)
        assert torch.all(x <= up) and torch.all(x >= lb)
        inv_sig = -torch.log((up-lb)/(x-lb) - 1)
        return inv_sig


class GOFLOW(DRMethod):
    def __init__(self, domain_range: DomainRange, num_training_iters=None, alpha=None, beta=None, max_loss=1e6, **kwargs):
        self.domain_range = domain_range
        self.alpha = alpha  # Weight for entropy maximization (KL to target)
        self.beta = beta    # Weight for similarity constraint (KL to previous)
        self.current_dist = NormFlowDist(
            torch.tensor(domain_range.low),
            torch.tensor(domain_range.high),
            ndim=len(domain_range.low)
        )
        self.dist_optimizer = torch.optim.Adam(self.current_dist.get_params(), lr=1e-3)
        self.dist_optimizer.zero_grad()
        self.num_training_iters = num_training_iters
        self.dist_history = []
        self.target_dist = UniformDist(self.domain_range.low, self.domain_range.high)
        self.max_loss = max_loss  # Add a maximum loss threshold

    def get_test_dist(self):
        return self.target_dist

    def get_train_dist(self):
        return self.current_dist

    def update(self, contexts, returns, entropy_update=True):
        print("Updating the GOFLOW distribution")
        R = torch.FloatTensor(returns).flatten().to(self.current_dist.device)
        R_ = (R - R.mean()) / (R.std() + 1e-8)

        previous_dist = self.current_dist.clone()

        for iter in range(self.num_training_iters):
            self.dist_optimizer.zero_grad()

            log_prob = self.current_dist.log_prob(contexts)
            log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)  # Clamp log probabilities

            z_target = self.target_dist.rsample([10000]).to(self.current_dist.device)
            log_p_current = self.current_dist.log_prob(z_target)
            log_p_target = self.target_dist.log_prob(z_target).to(self.current_dist.device)
            
            log_p_current = torch.clamp(log_p_current, min=-1e6, max=1e6)
            log_p_target = torch.clamp(log_p_target, min=-1e6, max=1e6)
            
            if(entropy_update):
                kl_loss_target = self.target_dist.volume()*torch.mean(torch.exp(log_p_current)*log_p_current)
            else:
                kl_loss_target = torch.mean(log_p_target - log_p_current)
            
            with torch.no_grad():
                z_previous = previous_dist.rsample([10000]).to(self.current_dist.device)
                log_p_previous = previous_dist.log_prob(z_previous)
                log_p_previous = torch.clamp(log_p_previous, min=-1e6, max=1e6)
            
            log_p_current_on_previous = self.current_dist.log_prob(z_previous)
            log_p_current_on_previous = torch.clamp(log_p_current_on_previous, min=-1e6, max=1e6)
            
            kl_loss_similarity = torch.mean(log_p_previous - log_p_current_on_previous)

            if(entropy_update):
                reward_loss = self.target_dist.volume()*((R_.detach() * log_prob * torch.exp(log_prob)).mean())
            else:
                reward_loss = -((R_.detach() * log_prob).mean())
            entropy_loss = self.alpha * kl_loss_target
            similarity_loss = self.beta * kl_loss_similarity
            total_loss = reward_loss + entropy_loss + similarity_loss

            # Check if loss is finite
            if not torch.isfinite(total_loss):
                print(f"Warning: Non-finite loss detected in iteration {iter}. Skipping update.")
                continue

            # Clip the total loss
            total_loss = torch.clamp(total_loss, max=self.max_loss)

            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.current_dist.get_params(), max_norm=1.0)
            
            self.dist_optimizer.step()

            print(f"Iteration {iter}:")
            print(f"  Reward Loss: {reward_loss.item():.4f}")
            print(f"  Entropy Loss (KL to Target): {entropy_loss.item():.4f}")
            print(f"  Similarity Loss (KL to Previous): {similarity_loss.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}")
        
        # Check if the final distribution is valid
        if not self.is_distribution_valid():
            print("Warning: Final distribution is invalid. Reverting to previous distribution.")
            self.current_dist = previous_dist

    def is_distribution_valid(self):
        # Implement checks to ensure the distribution is valid
        # For example, check if the parameters are finite and within expected ranges
        for param in self.current_dist.get_params():
            if not torch.isfinite(param).all():
                return False
        return True

        # Optionally, update the previous distribution for the next iteration
        # self.previous_dist = copy.deepcopy(self.current_dist)


class BoundarySamplingDist(Distr):
    def __init__(self, low, high, boundary_prob=0.5):
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.ndim = len(low)
        self.boundary_prob = boundary_prob
        
    def rsample(self, sample_shape=torch.Size()):
        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        samples = torch.zeros((n_samples, self.ndim))
        
        # Decide which samples will be on the boundary
        boundary_mask = torch.rand(n_samples) < self.boundary_prob
        
        # For non-boundary samples, sample uniformly
        non_boundary_samples = torch.rand((n_samples, self.ndim)) * (self.high - self.low) + self.low
        
        # For boundary samples, choose a random dimension and boundary
        boundary_dims = torch.randint(0, self.ndim, (n_samples,))
        boundary_is_high = torch.rand(n_samples) < 0.5
        
        for i in range(n_samples):
            if boundary_mask[i]:
                samples[i] = non_boundary_samples[i]
                dim = boundary_dims[i]
                samples[i, dim] = self.high[dim] if boundary_is_high[i] else self.low[dim]
            else:
                samples[i] = non_boundary_samples[i]
        
        return samples
    
    def log_prob(self, value):
        # Check if the value is within the bounds
        in_range = torch.all((value >= self.low) & (value <= self.high), dim=-1)
        
        # Calculate the volume of the sampling space
        volume = torch.prod(self.high - self.low)
        
        # Calculate the log probability for uniform sampling within the bounds
        log_prob_uniform = -torch.log(volume)
        
        # Calculate the log probability for boundary sampling
        # This is an approximation
        num_dims = len(self.low)
        log_prob_boundary = torch.log(torch.tensor(self.boundary_prob / (2 * num_dims)))
        
        # Combine probabilities
        log_prob = torch.where(
            in_range,
            torch.log(
                (1 - self.boundary_prob) * torch.exp(log_prob_uniform) +
                self.boundary_prob * torch.exp(log_prob_boundary)
            ),
            torch.tensor(float('-inf'))
        )
        
        return log_prob
    

class ADR(DRMethod):
    def __init__(self, domain_range: DomainRange, 
                 boundary_prob=0.5, 
                 success_threshold=0.5, 
                 expansion_factor=1.1, 
                 initial_dr_percentage=0.2,
                 **kwargs):
        self.domain_range = domain_range
        self.ndim = len(domain_range.low)
        self.lower_threshold = success_threshold/2.0
        self.upper_threshold = success_threshold
        self.expansion_factor = expansion_factor
        self.boundary_prob = boundary_prob
        
        mid_range = (torch.tensor(domain_range.low) + torch.tensor(domain_range.high)) / 2
        # interval = (torch.tensor(domain_range.high) - torch.tensor(domain_range.low)) * initial_dr_percentage
        self.current_low = (mid_range).tolist()
        self.current_high = (mid_range).tolist()
        
    def get_train_dist(self):
        return BoundarySamplingDist(self.current_low, self.current_high, self.boundary_prob)

    def get_test_dist(self):
        return UniformDist(self.domain_range.low, self.domain_range.high)

    def update(self, contexts, returns):
        contexts = contexts.numpy()
        returns = returns.numpy()
        
        print(contexts)
        # Randomly select a dimension to update
        dim = np.random.randint(0, self.ndim)
        
        low_boundary = self.current_low[dim]
        high_boundary = self.current_high[dim]
        
        # Identify samples near each boundary
        low_mask = np.isclose(contexts[:, dim], low_boundary, atol=1e-3)
        high_mask = np.isclose(contexts[:, dim], high_boundary, atol=1e-3)
        
        # Compute success rates for each boundary
        low_success_rate = np.mean(returns[low_mask]) if np.any(low_mask) else 0
        high_success_rate = np.mean(returns[high_mask]) if np.any(high_mask) else 0
        print("Low boundary reward: "+str(low_success_rate))
        print("High boundary reward: "+str(low_success_rate))

        # Update boundaries based on success rates
        if low_success_rate > self.upper_threshold:
            midpoint = (low_boundary + high_boundary) / 2
            new_low = max(midpoint - (midpoint - low_boundary) / self.expansion_factor, self.domain_range.low[dim])
            self.current_low[dim] = new_low
            print(f"ADR Update - Dimension {dim}, Lower boundary expanded to: {new_low:.4f}")
        elif low_success_rate < self.lower_threshold:
            midpoint = (low_boundary + high_boundary) / 2
            new_low = min(midpoint - (midpoint - low_boundary) * self.expansion_factor, (low_boundary + high_boundary) / 2)
            self.current_low[dim] = new_low
            print(f"ADR Update - Dimension {dim}, Lower boundary contracted to: {new_low:.4f}")
        
        if high_success_rate > self.upper_threshold:
            midpoint = (low_boundary + high_boundary) / 2
            new_high = min(midpoint + (high_boundary - midpoint) / self.expansion_factor, self.domain_range.high[dim])
            self.current_high[dim] = new_high
            print(f"ADR Update - Dimension {dim}, Upper boundary expanded to: {new_high:.4f}")
        elif high_success_rate < self.lower_threshold:
            midpoint = (low_boundary + high_boundary) / 2
            new_high = max(midpoint + (high_boundary - midpoint) * self.expansion_factor, (low_boundary + high_boundary) / 2)
            self.current_high[dim] = new_high
            print(f"ADR Update - Dimension {dim}, Upper boundary contracted to: {new_high:.4f}")

        print(f"Current domain: Low = {self.current_low}, High = {self.current_high}")



        
class A2CBase(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'

        # This helps in PBT when we need to restart an experiment with the exact same name, rather than
        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")

        self.config = config
        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.load_networks(params)

        self.multi_gpu = config.get('multi_gpu', False)

        
        # Remove pdfs and logs directories if they exist
        pdfs_dir = './pdfs'
        logs_dir = './logs'
        value_plots_dir = './value_plots'

        for directory in [pdfs_dir, value_plots_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Removed existing {directory} directory")

        # Create fresh directories
        os.makedirs(pdfs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(value_plots_dir, exist_ok=True)


        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        self.curr_frames = 0

        if self.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group("nccl", rank=self.global_rank, world_size=self.world_size)

            self.device_name = 'cuda:' + str(self.local_rank)
            config['device'] = self.device_name
            if self.global_rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.global_rank == 0:
            self.diagnostics = PpoDiagnostics()
        else:
            self.diagnostics = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.vec_env = None
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = config.get('vec_env', None)

        self.ppo_device = config.get('device', 'cuda:0')
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)

        if self.has_central_value:
            self.state_space = self.env_info.get('state_space', None)
            if isinstance(self.state_space,gym.spaces.Dict):
                self.state_shape = {}
                for k,v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        # TODO: do we still need it?
        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            
            if self.max_epochs == -1 and self.max_frames == -1:
                print("Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one.")
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.max_epochs

                if self.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.max_frames

                self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']

        
        
        # seq_length is used only with rnn policy and value functions
        if 'seq_len' in config:
            print('WARNING: seq_len is deprecated, use seq_length instead')

        self.seq_length = self.config.get('seq_length', 4)
        print('seq_length:', self.seq_length)
        self.bptt_len = self.config.get('bptt_length', self.seq_length) # not used right now. Didn't show that it is usefull
        self.zero_rnn_on_done = self.config.get('zero_rnn_on_done', True)

        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        print('current training device:', self.ppo_device)
        # self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_rewards = []
        self.game_contexts = []
        
        self.validating = False
        
        self.train_per_update = self.config['dr_method']['train_per_update']
        self.val_per_update = self.config['dr_method']['val_per_update']
        self.update_on_train = self.config['dr_method']['name'] == "ADR"
        self.episode_budget = self.train_per_update

        self.obs = None

        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors

        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)

        # either minibatch_size_per_env or minibatch_size should be present in a config
        # if both are present, minibatch_size is used
        # otherwise minibatch_size_per_env is used minibatch_size_per_env is used to calculate minibatch_size
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)

        assert(self.minibatch_size > 0)

        self.games_num = self.minibatch_size // self.seq_length # it is used only for current rnn implementation

        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.mini_epochs_num = self.config['mini_epochs']

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']
        print("self.summaries_dir: ", self.summaries_dir)
        if self.global_rank == 0:
            writer = SummaryWriter(self.summaries_dir, flush_secs=1)
            if self.population_based_training:
                self.writer = IntervalSummaryWriter(writer, self.config, flush_secs=1)
            else:
                self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')
        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5)
            self.advantage_mean_std = GeneralizedMovingStats((1,), momentum=momentum).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        #self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.aux_loss_dict = {}



    def plot_rewards(self, contexts, rewards, writer, frame):
        num_dims = contexts.shape[1]
        
        for dim in range(num_dims):
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(contexts[:, dim], rewards)
            plt.xlabel(f'Context Dimension {dim}')
            plt.ylabel('Rewards')
            plt.title(f'Rewards vs Context Dimension {dim}')
            
            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert the plot to a PIL Image
            image = Image.open(buf)
            
            # Convert PIL Image to a numpy array
            image_array = np.array(image)
            
            # Add the image to TensorBoard with a unique name for each dimension
            writer.add_image(f'Reward_Plot_Dim_{dim}', image_array, global_step=frame, dataformats='HWC')

            plt.close(fig)  # Close the plot to avoid showing it during training
            
    def plot_value(self, dr_method: DRMethod, initial_obs, writer, frame, res=50):
        print("Inside plot value")
        
        test_dist = dr_method.get_test_dist()
        ndims = test_dist.ndim
        obs_dim = initial_obs['obs'].shape[1]
        state_dim = initial_obs['states'].shape[1]
        env_param_dim = state_dim - obs_dim

        base_obs = initial_obs['obs'][0].unsqueeze(0).repeat(res * res, 1).cpu()
        
        os.makedirs('./value_plots', exist_ok=True)
        
        for i in range(ndims):
            for j in range(i+1, ndims):
                x = torch.linspace(test_dist.low[i], test_dist.high[i], res)
                y = torch.linspace(test_dist.low[j], test_dist.high[j], res)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                env_params = torch.zeros(res * res, env_param_dim)
                env_params[:, i] = X.flatten()
                env_params[:, j] = Y.flatten()
                
                combined_states = torch.cat([base_obs, env_params], dim=1)
                combined_states = combined_states.to(self.ppo_device)
                
                with torch.no_grad():
                    values = self.get_values({'obs': combined_states[:, :obs_dim], 'states': combined_states})
                
                values = values.cpu()
                value_grid = values.view(res, res).numpy()
                
                success_threshold = self.config['dr_method']['success_threshold']
                alpha_mask = np.where(value_grid > success_threshold, 1.0, 0.3)
                
                # Plot value function
                fig, ax = plt.subplots(figsize=(8, 8))
                pcm = ax.pcolormesh(X.numpy(), Y.numpy(), value_grid, cmap='plasma', alpha=alpha_mask, shading='auto')
                ax.set_xlabel(f'Environment Parameter {i}')
                ax.set_ylabel(f'Environment Parameter {j}')
                ax.set_title(f'Value Function (Env Params {i} vs {j})')
    
                
                pdf_filename = f'./value_plots/value_plot_params_{i}_vs_{j}_frame_{frame}.pdf'
                plt.savefig(pdf_filename, format='pdf')

                plt.colorbar(pcm, ax=ax, label='Value')
                plt.savefig(pdf_filename.replace(".pdf", "_colorbar.pdf"), format='pdf')

                # Plot mask
                fig_mask, ax_mask = plt.subplots(figsize=(8, 8))
                mask = alpha_mask < 0.5
                mask_plot = ax_mask.pcolormesh(X.numpy(), Y.numpy(), mask, cmap='binary', shading='auto')
                
                ax_mask.set_xlabel(f'Environment Parameter {i}')
                ax_mask.set_ylabel(f'Environment Parameter {j}')
                ax_mask.set_title(f'Value Mask (Env Params {i} vs {j})')
                
                mask_filename = f'./value_plots/value_mask_params_{i}_vs_{j}_frame_{frame}.pdf'
                plt.savefig(mask_filename, format='pdf')
                
                # Save the plot to a BytesIO object for TensorBoard
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                
                image = Image.open(buf)
                image_array = np.array(image)
                
                writer.add_image(f'Value_Plot_Params_{i}_vs_{j}', image_array, global_step=frame, dataformats='HWC')

                plt.close(fig)
                plt.close(fig_mask)
        
        print(f"Value function plots and masks saved as PDFs in ./value_plots folder for frame {frame}")
    
    def plot_dist(self, dr_method, writer, frame, res=50, num_samples=50):
        train_dist = dr_method.get_train_dist()
        test_dist = dr_method.get_test_dist()
        
        device = 'cpu'  # Default to CPU for plotting
        
        ndims = train_dist.ndim
        
        os.makedirs('./pdfs', exist_ok=True)
        
        # Handle single dimension case
        if ndims == 1:
            dims = [(0,0)]  # Plot dimension 0 against itself
        else:
            dims = [(i,j) for i in range(ndims) for j in range(i+1, ndims)]
            
        for i,j in dims:
            x = torch.linspace(test_dist.low[i], test_dist.high[i], res)
            if i == j:  # Single dimension case
                y = x.clone()  # Use same values for both axes
            else:
                y = torch.linspace(test_dist.low[j], test_dist.high[j], res)

            X, Y = torch.meshgrid(x, y, indexing='ij')

            points = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device)

            input_tensor = train_dist.rsample((num_samples,)).to(device)

            combined_tensor = input_tensor.unsqueeze(1).expand(-1, points.shape[0], -1).clone()

            combined_tensor[:, :, i] = points[:, 0].unsqueeze(0).expand(num_samples, -1)
            if i != j:  # Only modify j if different from i
                combined_tensor[:, :, j] = points[:, 1].unsqueeze(0).expand(num_samples, -1)

            reshaped_tensor = combined_tensor.view(-1, ndims)

            pdf_values = torch.exp(train_dist.log_prob(reshaped_tensor).detach().cpu())

            pdf_values = pdf_values.view(num_samples, res, res).mean(dim=0)

            fig, ax = plt.subplots(figsize=(8, 8))
            
            max_prob = pdf_values.max()
            alpha_mask = torch.where(pdf_values > 0.5 * max_prob, 1.0, 0.3).numpy()
            
            pcm = ax.pcolormesh(X.numpy(), Y.numpy(), pdf_values.numpy(), cmap='plasma', alpha=alpha_mask, shading='auto')
            
            if i == j:
                ax.set_xlabel(f'Dimension {i}')
                ax.set_ylabel(f'Dimension {i} (replicated)')
                ax.set_title(f'Training Distribution Marginal PDF (Dim {i})')
            else:
                ax.set_xlabel(f'Dimension {i}')
                ax.set_ylabel(f'Dimension {j}')
                ax.set_title(f'Training Distribution Marginal PDF (Dims {i} vs {j})')

            pdf_filename = f'./pdfs/distribution_plot_dims_{i}_vs_{j}_frame_{frame}.pdf'
            plt.savefig(pdf_filename, format='pdf')
            
            plt.colorbar(pcm, ax=ax, label='PDF Value')
            plt.savefig(pdf_filename.replace(".pdf", "_colorbar.pdf"), format='pdf')
            

            # Plot mask
            fig_mask, ax_mask = plt.subplots(figsize=(8, 8))
            mask = alpha_mask < 0.5
            mask_plot = ax_mask.pcolormesh(X.numpy(), Y.numpy(), mask, cmap='binary', shading='auto')
            if i == j:
                ax_mask.set_xlabel(f'Dimension {i}')
                ax_mask.set_ylabel(f'Dimension {i} (replicated)')
                ax_mask.set_title(f'Distribution Mask (Dim {i})')
            else:
                ax_mask.set_xlabel(f'Dimension {i}')
                ax_mask.set_ylabel(f'Dimension {j}')
                ax_mask.set_title(f'Distribution Mask (Dims {i} vs {j})')
            
            mask_filename = f'./pdfs/distribution_mask_dims_{i}_vs_{j}_frame_{frame}.pdf'
            plt.savefig(mask_filename, format='pdf')

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            image = Image.open(buf)
            
            image_array = np.array(image)
            
            writer.add_image(f'Distribution_Plot_Dims_{i}_vs_{j}', image_array, global_step=frame, dataformats='HWC')

            plt.close(fig)
            plt.close(fig_mask)

        print(f"Distribution plots and masks saved as PDFs in ./pdfs folder for frame {frame}")
        
    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)
        has_central_value_net = self.config.get('central_value_config') is not  None
        if has_central_value_net:
            print('Adding Central Value Network')
            if 'model' not in params['config']['central_value_config']:
                params['config']['central_value_config']['model'] = {'name': 'central_value'}
            network = builder.load(params['config']['central_value_config'])
            self.config['central_value_config']['network'] = network

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)

        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        for k,v in self.aux_loss_dict.items():
            self.writer.add_scalar('losses/' + k, torch_ext.mean_list(v).item(), frame)
        self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }
        
        with torch.no_grad():
            input_dict_copy = copy.deepcopy(input_dict)
            res_dict = self.model(input_dict_copy)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict
  
    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states
                }
                result = self.model(input_dict)
                value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def env_reset(self):
        self.init_tensors()
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        self.game_rewards = []
        self.game_contexts = []
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * masks_t
        return mb_advs

    def clear_stats(self):
        raise NotImplementedError

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['optimizer'] = self.optimizer.state_dict()

        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards
        state['last_mean_target_rewards'] = self.last_mean_target_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights, set_epoch=True):

        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        self.optimizer.load_state_dict(weights['optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)
        self.last_mean_target_rewards = weights.get('last_mean_target_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def set_central_value_function_weights(self, weights):
        self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        if self.has_central_value:
            state['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dic(weights['advantage_mean_std'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)

    def get_param(self, param_name):
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "kl_threshold",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            return getattr(self, param_name)
        elif param_name == "learning_rate":
            return self.last_lr
        else:
            raise NotImplementedError(f"Can't get param {param_name}")       

    def set_param(self, param_name, param_value):
        if param_name in [
            "grad_norm",
            "critic_coef", 
            "bounds_loss_coef",
            "entropy_coef",
            "gamma",
            "tau",
            "mini_epochs_num",
            "e_clip",
            ]:
            setattr(self, param_name, param_value)
        elif param_name == "learning_rate":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    raise NotImplementedError("Can't directly mutate LR on this schedule")
                else:
                    self.learning_rate = param_value

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
        elif param_name == "kl_threshold":
            if self.global_rank == 0:
                if self.is_adaptive_lr:
                    self.kl_threshold = param_value
                    self.scheduler.kl_threshold = param_value
                else:
                    raise NotImplementedError("Can't directly mutate kl threshold")
        else:
            raise NotImplementedError(f"No param found for {param_value}")

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def play_steps(self, validation=False):
        
    
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks) 
            else:
                res_dict = self.get_action_values(self.obs)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.perf_counter()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.perf_counter()
        
            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
            self.game_rewards += (self.current_rewards[env_done_indices]).squeeze(axis=1).tolist()
            
            # remove dimension 1 and convert to list
            
            new_contexts = infos["context"][env_done_indices, :].squeeze(axis=1).tolist()
            self.game_contexts += new_contexts
            
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict


class ContinuousA2CBase(A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)

        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        self.clip_actions = self.config.get('clip_actions', True)
        
        assert (self.config.get('dr_method', None) is not None)

        ranges = self.vec_env.env.unwrapped.get_dr_ranges()
        if(self.config['dr_method'].get('range_scale', None) is not None):
            midpoints = self.vec_env.env.unwrapped.get_midpoints()
            scale = self.config['dr_method']['range_scale']
            for k, v in ranges.items():
                ranges[k] = ((midpoints[k] - (midpoints[k] - v[0]) * scale), (midpoints[k] + (v[1] - midpoints[k]) * scale))
        
        print("Ranges")
        print(ranges)
        domain_range = DomainRange.from_dict(ranges)
            
        if(self.config['dr_method']["name"] == 'NoDR'):
            self.dr_method = NoDR(domain_range, **self.config['dr_method'])
        elif(self.config['dr_method']["name"] == 'FullDR'):
            self.dr_method = FullDR(domain_range, **self.config['dr_method'])
        elif(self.config['dr_method']["name"] == 'LSDR'):
            self.dr_method = LSDR(domain_range, **self.config['dr_method'])
        elif(self.config['dr_method']["name"] == 'GOFLOW'):
            self.dr_method = GOFLOW(domain_range, **self.config['dr_method'])
        elif(self.config['dr_method']["name"] == 'DORAEMON'):
            self.dr_method = DORAEMON(domain_range, **self.config['dr_method'])
        elif(self.config['dr_method']["name"] == 'ADR'):
            self.dr_method = ADR(domain_range, **self.config['dr_method'])

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        
        

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self, validation=False):
        
        super().train_epoch()

        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            batch_dict = self.play_steps(validation=validation)

        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()

        if not validation:  
            self.set_train()
            
            self.curr_frames = batch_dict.pop('played_frames')
            self.prepare_dataset(batch_dict)
            self.algo_observer.after_steps()
            if self.has_central_value:
                self.train_central_value()
            
            a_losses = []
            c_losses = []
            b_losses = []
            entropies = []
            kls = []
            for mini_ep in range(0, self.mini_epochs_num):
                ep_kls = []
                for i in range(len(self.dataset)):
                    a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    ep_kls.append(kl)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)

                    self.dataset.update_mu_sigma(cmu, csigma)
                    if self.schedule_type == 'legacy':
                        av_kls = kl
                        if self.multi_gpu:
                            dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                            av_kls /= self.world_size
                        self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                        self.update_lr(self.last_lr)

                av_kls = torch_ext.mean_list(ep_kls)
                if self.multi_gpu:
                    dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                    av_kls /= self.world_size
                if self.schedule_type == 'standard':
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

                kls.append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.normalize_input:
                    self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

            update_time_end = time.perf_counter()
            play_time = play_time_end - play_time_start
            update_time = update_time_end - update_time_start
            total_time = update_time_end - play_time_start

            return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul
        else:
            return None

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            if self.config.get('freeze_critic', False):
                self.value_mean_std.eval()
            else:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        self.last_mean_target_rewards = -100500

        start_time = time.perf_counter()
        total_time = 0
        rep_count = 0
        self.vec_env.env.unwrapped.set_sampling_dist(self.dr_method.get_train_dist())
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            if self.has_central_value:
                model_params.append(self.central_value_net.state_dict())
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
            if self.has_central_value:
                self.central_value_net.load_state_dict(model_params[1])

        self.writer.add_scalar('train_rewards/step', 0, 0)
        self.writer.add_scalar('train_success/step', 0, 0)

        self.writer.add_scalar('target_rewards/step', 0, 0)
        self.writer.add_scalar('target_success/step', 0, 0)

        while True:
            if(not self.validating):
                epoch_num = self.update_epoch()

            epoch_data = self.train_epoch(validation=self.validating)

            if(not self.validating):
                # cleaning memory to optimize space
                self.dataset.update_values_dict(None)
                
            should_exit = False

            if self.global_rank == 0:
            
                # print budget progress
                print(len(self.game_rewards), "/", self.episode_budget)           
                if len(self.game_rewards) >= self.episode_budget:
                
                    if(not self.validating):
                        step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = epoch_data
                        total_time += sum_time
                        frame = self.frame // self.num_agents
                    
                        self.diagnostics.epoch(self, current_epoch = epoch_num)
                        # do we need scaled_time?
                        scaled_time = self.num_agents * sum_time
                        scaled_play_time = self.num_agents * play_time
                        curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                        self.frame += curr_frames
                
                        print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                        epoch_num, self.max_epochs, frame, self.max_frames)

                        self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                        a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                        scaled_time, scaled_play_time, curr_frames)

                        if len(b_losses) > 0:
                            self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)


                        mean_rewards = np.mean(self.game_rewards)        
                        self.writer.add_scalar('train_rewards/step', mean_rewards, frame)
                        self.writer.add_scalar('train_success/step', np.mean(np.array(self.game_rewards) >= self.config['dr_method']['success_threshold']) , frame)


                        if self.has_self_play_config:
                            self.self_play_manager.update(self)

                        checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards)

                        if self.save_freq > 0:
                            if epoch_num % self.save_freq == 0:
                                self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                        if mean_rewards > self.last_mean_rewards and epoch_num >= self.save_best_after:
                            print('saving next best train rewards: ', mean_rewards)
                            self.last_mean_rewards = mean_rewards
                            self.save(os.path.join(self.nn_dir, self.config['name']+"_train"))

                                
                        print("Force resetting")
                        self.vec_env.env.unwrapped.set_sampling_dist(self.dr_method.get_test_dist())
                        self.train_game_contexts = copy.deepcopy(self.game_contexts)
                        self.train_game_rewards = copy.deepcopy(self.game_rewards)
                        self.obs = self.env_reset()
                        self.validating = True
                        self.episode_budget = self.val_per_update
                        
                    else:
                        mean_rewards = np.mean(self.game_rewards)
                        self.writer.add_scalar('target_rewards/step', mean_rewards, frame)
                        self.writer.add_scalar('target_success/step', np.mean(np.array(self.game_rewards) >= self.config['dr_method']['success_threshold']) , frame)
                        st = time.time()
                        train_dist_entropy = self.dr_method.get_train_dist().entropy()
                        print("Entropy time: "+str(time.time()-st))
                        
                        
                        self.writer.add_scalar('train_dist_entropy/step', train_dist_entropy, frame)

                        if mean_rewards > self.last_mean_target_rewards and epoch_num >= self.save_best_after:
                            print('saving next best target rewards: ', mean_rewards)
                            self.last_mean_target_rewards = mean_rewards
                            self.save(os.path.join(self.nn_dir, self.config['name']+"_target"))
                            
                        if(self.update_on_train):
                            self.dr_method.update(torch.tensor(self.train_game_contexts), torch.tensor(self.train_game_rewards))
                        else:
                            self.dr_method.update(torch.tensor(self.game_contexts), torch.tensor(self.game_rewards))
                            
                        self.plot_rewards(np.array(self.game_contexts), np.array(self.game_rewards), self.writer, self.frame)
                        self.plot_dist(dr_method=self.dr_method, writer=self.writer, frame=self.frame)
                        # self.plot_value(self.dr_method, self.obs, self.writer, self.frame)
                        
                        
                        self.vec_env.env.unwrapped.set_sampling_dist(self.dr_method.get_train_dist())
                        self.obs = self.env_reset()
                        self.validating = False
                        self.episode_budget = self.train_per_update
                

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num
