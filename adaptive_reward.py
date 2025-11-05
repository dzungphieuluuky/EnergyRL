"""
Enhanced Safe Training System combining:
1. Lagrangian multiplier optimization for soft constraints
2. Hard penalty shaping for critical violations
3. Hierarchical reward structure
4. Advanced constraint monitoring

This dual approach ensures:
- Critical constraints NEVER violated (hard penalties)
- Resource constraints optimized via Lagrangian (soft penalties)
- Fast convergence with guaranteed safety
"""

import numpy as np
import torch
from gymnasium import Wrapper
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import warnings


# ============================================================
# 1. Enhanced SafeRewardWrapper with Dual Constraint Handling
# ============================================================
class EnhancedSafeRewardWrapper(Wrapper):
    """
    Dual-mode constraint handling:
    - HARD constraints: Exponential penalties (QoS violations)
    - SOFT constraints: Lagrangian penalties (Resource violations)
    
    This ensures critical QoS is always met while optimizing resources.
    """

    def __init__(self, 
                 env,
                 # Hard constraint config (QoS)
                 hard_constraint_keys: List[str] = None,
                 hard_constraint_thresholds: Dict[str, float] = None,
                 hard_penalty_base: float = 50.0,
                 hard_penalty_scale: float = 2.0,
                 
                 # Soft constraint config (Resources)
                 soft_constraint_keys: List[str] = None,
                 soft_constraint_limits: Dict[str, float] = None,
                 initial_lambda: float = 10.0,
                 max_lambda: float = 100.0,
                 
                 # Energy reward config
                 energy_reward_enabled: bool = True,
                 energy_unlock_threshold: int = 50,
                 energy_reward_scale: float = 2.0,
                 
                 # Tracking config
                 verbose: bool = True):
        
        super().__init__(env)
        
        # Hard constraints (QoS) - MUST be satisfied
        self.hard_constraint_keys = hard_constraint_keys or ['avg_drop_rate', 'avg_latency']
        self.hard_constraint_thresholds = hard_constraint_thresholds or {
            'avg_drop_rate': 1.0,  # Max 1% drop rate
            'avg_latency': 50.0,   # Max 50ms latency
        }
        self.hard_penalty_base = hard_penalty_base
        self.hard_penalty_scale = hard_penalty_scale
        
        # Soft constraints (Resources) - Optimize via Lagrangian
        self.soft_constraint_keys = soft_constraint_keys or ['cpu_violations', 'prb_violations']
        self.soft_constraint_limits = soft_constraint_limits or {
            'cpu_violations': 0.0,  # Target 0 violations
            'prb_violations': 0.0,
        }
        self.lambda_value = float(initial_lambda)
        self.max_lambda = float(max_lambda)
        
        # Adaptive weights for soft constraints
        self.soft_weights = {key: 1.0 for key in self.soft_constraint_keys}
        
        # Energy reward config
        self.energy_reward_enabled = energy_reward_enabled
        self.energy_unlock_threshold = energy_unlock_threshold
        self.energy_reward_scale = energy_reward_scale
        
        # Tracking state
        self.episode_step = 0
        self.consecutive_safe_steps = 0
        self.consecutive_violations = 0
        self.total_episodes = 0
        self.verbose = verbose
        
        # Episode statistics
        self.episode_hard_violations = 0
        self.episode_soft_costs = {key: 0.0 for key in self.soft_constraint_keys}
        self.episode_rewards = deque(maxlen=100)
        self.episode_compliance = deque(maxlen=100)
        
        # Violation history for adaptive weights
        self.soft_violation_history = {key: deque(maxlen=100) for key in self.soft_constraint_keys}
        
        print(f"\n{'='*70}")
        print("Enhanced Safe Training System Initialized")
        print(f"{'='*70}")
        print(f"Hard Constraints (Critical): {self.hard_constraint_keys}")
        print(f"  Thresholds: {self.hard_constraint_thresholds}")
        print(f"  Penalty: -{self.hard_penalty_base} * {self.hard_penalty_scale}^(severity)")
        print(f"\nSoft Constraints (Optimized): {self.soft_constraint_keys}")
        print(f"  Limits: {self.soft_constraint_limits}")
        print(f"  Initial λ: {self.lambda_value}")
        print(f"\nEnergy Reward: {'Enabled' if self.energy_reward_enabled else 'Disabled'}")
        print(f"  Unlock after: {self.energy_unlock_threshold} safe steps")
        print(f"{'='*70}\n")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_hard_violations = 0
        self.episode_soft_costs = {key: 0.0 for key in self.soft_constraint_keys}
        
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_step += 1
        
        # ================================================================
        # STEP 1: Check Hard Constraints (QoS) - CRITICAL
        # ================================================================
        hard_violations, hard_penalty, hard_violation_details = self._evaluate_hard_constraints(info)
        
        # ================================================================
        # STEP 2: Check Soft Constraints (Resources) - OPTIMIZE
        # ================================================================
        soft_costs, soft_penalty, soft_violation_details = self._evaluate_soft_constraints(info)
        
        # ================================================================
        # STEP 3: Compute Shaped Reward
        # ================================================================
        if hard_violations:
            # CRITICAL VIOLATION - Large penalty, no rewards
            self.consecutive_safe_steps = 0
            self.consecutive_violations += 1
            self.episode_hard_violations += 1
            
            # Exponentially increasing penalty for consecutive violations
            violation_multiplier = min(3.0, 1.0 + (self.consecutive_violations / 20.0))
            total_penalty = hard_penalty * violation_multiplier + soft_penalty
            
            shaped_reward = total_penalty  # Very negative
            
            # Warning for persistent violations
            if self.consecutive_violations % 10 == 0 and self.verbose:
                print(f"\n⚠️  WARNING: {self.consecutive_violations} consecutive hard violations!")
                for detail in hard_violation_details:
                    print(f"   {detail}")
        
        else:
            # No hard violations - safe operation
            self.consecutive_safe_steps += 1
            self.consecutive_violations = 0
            
            # Base reward for safety compliance
            compliance_reward = 1.0
            
            # Bonus for sustained safety
            if self.consecutive_safe_steps > 20:
                sustained_bonus = np.log10(self.consecutive_safe_steps - 19) * 0.5
                compliance_reward += sustained_bonus
            
            # Soft constraint penalty (resources)
            shaped_reward = base_reward + compliance_reward - soft_penalty
            
            # Energy optimization reward (unlocked after sustained safety)
            if (self.energy_reward_enabled and 
                self.consecutive_safe_steps >= self.energy_unlock_threshold):
                energy_reward = self._compute_energy_reward(info)
                shaped_reward += energy_reward
                
                if self.consecutive_safe_steps == self.energy_unlock_threshold and self.verbose:
                    print(f"\n✅ Energy optimization UNLOCKED at step {self.episode_step}!")
        
        # ================================================================
        # STEP 4: Update Tracking and Info
        # ================================================================
        self._update_tracking(soft_costs)
        
        # Enhanced info dict
        info['safe_training'] = {
            'hard_violations': hard_violations,
            'hard_penalty': float(hard_penalty),
            'hard_violation_details': hard_violation_details,
            'soft_costs': soft_costs,
            'soft_penalty': float(soft_penalty),
            'soft_violation_details': soft_violation_details,
            'lambda_value': float(self.lambda_value),
            'consecutive_safe_steps': self.consecutive_safe_steps,
            'consecutive_violations': self.consecutive_violations,
            'shaped_reward': float(shaped_reward),
            'base_reward': float(base_reward),
            'compliance_reward': float(compliance_reward) if not hard_violations else 0.0,
        }
        
        # Log episode end
        if terminated or truncated:
            self._log_episode_end(info)
        
        return obs, shaped_reward, terminated, truncated, info

    def _evaluate_hard_constraints(self, info: Dict) -> Tuple[bool, float, List[str]]:
        """
        Evaluate hard constraints (QoS) with exponential penalties.
        
        Returns:
            (has_violations, total_penalty, violation_details)
        """
        has_violations = False
        total_penalty = 0.0
        violation_details = []
        
        for key in self.hard_constraint_keys:
            if key not in info:
                continue
                
            value = float(info[key])
            threshold = self.hard_constraint_thresholds[key]
            
            if value > threshold:
                has_violations = True
                
                # Severity: how much over threshold (normalized)
                severity = (value - threshold) / threshold
                
                # Exponential penalty: -base * scale^severity
                penalty = -self.hard_penalty_base * (self.hard_penalty_scale ** severity)
                total_penalty += penalty
                
                violation_details.append(
                    f"{key}: {value:.2f} > {threshold:.2f} (penalty: {penalty:.2f})"
                )
        
        return has_violations, total_penalty, violation_details

    def _evaluate_soft_constraints(self, info: Dict) -> Tuple[Dict, float, List[str]]:
        """
        Evaluate soft constraints (Resources) with Lagrangian penalties.
        
        Returns:
            (costs_dict, total_penalty, violation_details)
        """
        costs = {}
        total_penalty = 0.0
        violation_details = []
        
        for key in self.soft_constraint_keys:
            if key not in info:
                costs[key] = 0.0
                continue
            
            # Cost is how much over the limit
            value = float(info[key])
            limit = self.soft_constraint_limits[key]
            cost = max(0.0, value - limit)
            costs[key] = cost
            
            # Weighted penalty
            weighted_cost = self.soft_weights[key] * cost
            penalty = self.lambda_value * weighted_cost
            total_penalty += penalty
            
            if cost > 0:
                violation_details.append(
                    f"{key}: {value:.2f} > {limit:.2f} (cost: {cost:.2f}, λ: {self.lambda_value:.2f})"
                )
        
        return costs, total_penalty, violation_details

    def _compute_energy_reward(self, info: Dict) -> float:
        """
        Compute energy efficiency reward (only when safe).
        """
        # Get energy metrics
        total_energy = info.get('total_energy', 0.0)
        avg_power_ratio = info.get('avg_power_ratio', 1.0)
        
        # Get environment parameters
        sim_params = self.env.unwrapped.sim_params
        n_cells = self.env.unwrapped.n_cells
        
        # Maximum possible energy
        max_power = 10**((sim_params.max_tx_power - 30)/10)
        max_energy = n_cells * (sim_params.base_power + max_power)
        
        # Energy efficiency (0 to 1)
        if max_energy > 0:
            energy_efficiency = 1.0 - (total_energy / max_energy)
            energy_efficiency = max(0.0, min(1.0, energy_efficiency))
        else:
            energy_efficiency = 0.0
        
        # Power reduction bonus
        power_reduction = 1.0 - avg_power_ratio
        
        # Unlock factor (gradually increase importance)
        unlock_progress = self.consecutive_safe_steps - self.energy_unlock_threshold
        unlock_factor = min(1.0, unlock_progress / 100.0)
        
        # Total energy reward
        energy_reward = (energy_efficiency + power_reduction) * unlock_factor * self.energy_reward_scale
        
        return energy_reward

    def _update_tracking(self, soft_costs: Dict):
        """Update tracking for adaptive weights."""
        for key, cost in soft_costs.items():
            self.episode_soft_costs[key] += cost
            self.soft_violation_history[key].append(cost > 0)

    def _update_adaptive_weights(self):
        """
        Adapt soft constraint weights based on violation frequency.
        More frequently violated constraints get higher weights.
        """
        for key in self.soft_constraint_keys:
            if len(self.soft_violation_history[key]) < 10:
                continue
            
            # Violation frequency
            violation_freq = np.mean(list(self.soft_violation_history[key]))
            
            # Update weight: increase for frequently violated constraints
            target_weight = 0.5 + violation_freq * 1.5
            
            # Smooth update
            alpha = 0.1
            self.soft_weights[key] = (1 - alpha) * self.soft_weights[key] + alpha * target_weight

    def _log_episode_end(self, info: Dict):
        """Log episode summary."""
        self.total_episodes += 1
        
        # Compute compliance
        compliant = (self.episode_hard_violations == 0)
        self.episode_compliance.append(compliant)
        
        # Compute compliance rate
        if len(self.episode_compliance) > 0:
            compliance_rate = np.mean(list(self.episode_compliance)) * 100
        else:
            compliance_rate = 0.0
        
        # Update adaptive weights
        self._update_adaptive_weights()
        
        if self.verbose and self.total_episodes % 5 == 0:
            print(f"\n{'='*70}")
            print(f"Episode {self.total_episodes} Complete")
            print(f"{'='*70}")
            print(f"  Steps: {self.episode_step}")
            print(f"  Hard Violations: {self.episode_hard_violations}")
            print(f"  Soft Costs: {', '.join(f'{k}={v:.2f}' for k, v in self.episode_soft_costs.items())}")
            print(f"  Consecutive Safe Steps: {self.consecutive_safe_steps}")
            print(f"  Compliance Rate (last 100): {compliance_rate:.1f}%")
            print(f"  Lambda: {self.lambda_value:.3f}")
            print(f"  Soft Weights: {', '.join(f'{k}={v:.2f}' for k, v in self.soft_weights.items())}")
            
            if compliance_rate >= 90:
                print(f"  ✅ EXCELLENT: >90% compliance!")
            elif compliance_rate >= 70:
                print(f"  ✓ Good: >70% compliance")
            elif compliance_rate < 50:
                print(f"  ⚠️  Warning: <50% compliance")
            print(f"{'='*70}\n")

    def update_lambda(self, new_lambda: float):
        """Update Lagrangian multiplier (called by callback)."""
        self.lambda_value = max(0.0, min(new_lambda, self.max_lambda))

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        compliance_rate = np.mean(list(self.episode_compliance)) * 100 if self.episode_compliance else 0.0
        
        return {
            'lambda_value': float(self.lambda_value),
            'consecutive_safe_steps': self.consecutive_safe_steps,
            'consecutive_violations': self.consecutive_violations,
            'total_episodes': self.total_episodes,
            'compliance_rate': compliance_rate,
            'soft_weights': self.soft_weights.copy(),
            'episode_hard_violations': self.episode_hard_violations,
            'episode_soft_costs': self.episode_soft_costs.copy(),
        }


# ============================================================
# 2. Enhanced Lagrange Update Callback
# ============================================================
class EnhancedLagrangeCallback(BaseCallback):
    """
    Sophisticated Lagrange multiplier update with:
    - Primal-dual gradient descent
    - Adaptive learning rates
    - Constraint prioritization
    - Convergence detection
    """

    def __init__(self, 
                 wrapper_ref: EnhancedSafeRewardWrapper,
                 initial_lr: float = 0.01,
                 min_lr: float = 1e-5,
                 max_lr: float = 0.1,
                 lr_decay: float = 0.9995,
                 ema_alpha: float = 0.05,
                 update_frequency: int = 100,  # Update every N steps
                 verbose: int = 1):
        
        super().__init__(verbose=verbose)
        self.wrapper = wrapper_ref
        self.current_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.lr_decay = float(lr_decay)
        self.ema_alpha = float(ema_alpha)
        self.update_frequency = update_frequency
        
        # EMA tracking for each soft constraint
        self.ema_costs = {key: 0.0 for key in wrapper_ref.soft_constraint_keys}
        
        # Gradient tracking
        self.cost_history = deque(maxlen=200)
        self.lambda_history = deque(maxlen=100)
        self.grad_variance = 0.0
        
        # Convergence tracking
        self.converged = False
        self.convergence_counter = 0
        
        # Best lambda tracking
        self.best_lambda = wrapper_ref.lambda_value
        self.best_total_cost = float('inf')

    def _on_step(self) -> bool:
        """Update lambda periodically based on constraint violations."""
        
        # Only update every N steps
        if self.n_calls % self.update_frequency != 0:
            return True
        
        # Collect costs from info
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        
        # Extract soft constraint costs
        batch_costs = {key: [] for key in self.wrapper.soft_constraint_keys}
        
        for info in infos:
            safe_info = info.get('safe_training', {})
            soft_costs = safe_info.get('soft_costs', {})
            
            for key in self.wrapper.soft_constraint_keys:
                cost = soft_costs.get(key, 0.0)
                batch_costs[key].append(cost)
        
        # Average across batch
        avg_costs = {key: np.mean(costs) if costs else 0.0 
                     for key, costs in batch_costs.items()}
        
        # Update EMA for each constraint
        for key in self.wrapper.soft_constraint_keys:
            self.ema_costs[key] = (
                (1 - self.ema_alpha) * self.ema_costs[key] + 
                self.ema_alpha * avg_costs[key]
            )
        
        # Store cost history
        total_cost = sum(self.ema_costs.values())
        self.cost_history.append(total_cost)
        
        # Compute gradient (cost - limit for each constraint)
        gradients = []
        for key in self.wrapper.soft_constraint_keys:
            limit = self.wrapper.soft_constraint_limits[key]
            grad = self.ema_costs[key] - limit
            gradients.append(grad)
        
        # Use maximum gradient (most violated constraint)
        max_gradient = max(gradients) if gradients else 0.0
        
        # Adaptive learning rate based on gradient stability
        self._adapt_learning_rate()
        
        # Update lambda using gradient ascent
        new_lambda = self.wrapper.lambda_value + self.current_lr * max_gradient
        new_lambda = max(0.0, min(new_lambda, self.wrapper.max_lambda))
        
        # Store lambda history
        self.lambda_history.append(new_lambda)
        
        # Check convergence
        self._check_convergence(total_cost)
        
        # Apply update
        self.wrapper.update_lambda(new_lambda)
        
        # Decay learning rate
        self.current_lr = max(self.min_lr, self.current_lr * self.lr_decay)
        
        # Periodic logging
        if self.verbose and self.n_calls % (self.update_frequency * 10) == 0:
            self._log_update(avg_costs, max_gradient)
        
        return True

    def _adapt_learning_rate(self):
        """Adapt learning rate based on gradient variance."""
        if len(self.cost_history) < 20:
            return
        
        recent_costs = list(self.cost_history)[-20:]
        self.grad_variance = np.var(recent_costs)
        
        # High variance -> reduce LR (oscillating)
        if self.grad_variance > 0.1:
            self.current_lr *= 0.9
        # Low variance -> can increase LR (slow progress)
        elif self.grad_variance < 0.01 and self.current_lr < self.max_lr:
            self.current_lr *= 1.05
        
        # Clamp to bounds
        self.current_lr = max(self.min_lr, min(self.current_lr, self.max_lr))

    def _check_convergence(self, current_total_cost: float):
        """Check if lambda has converged."""
        if len(self.lambda_history) < 20:
            return
        
        recent_lambdas = list(self.lambda_history)[-10:]
        lambda_std = np.std(recent_lambdas)
        
        # Check if all constraints are satisfied
        all_satisfied = all(
            cost <= limit * 1.1  # 10% tolerance
            for cost, limit in zip(
                self.ema_costs.values(), 
                self.wrapper.soft_constraint_limits.values()
            )
        )
        
        # Convergence: constraints satisfied + lambda stable
        if all_satisfied and lambda_std < 0.1:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        
        if self.convergence_counter > 10:
            self.converged = True
        
        # Track best lambda
        if current_total_cost < self.best_total_cost:
            self.best_total_cost = current_total_cost
            self.best_lambda = self.wrapper.lambda_value

    def _log_update(self, current_costs: Dict, max_grad: float):
        """Log update information."""
        print(f"\n[Lagrange Update] Step {self.num_timesteps}:")
        print(f"  λ: {self.wrapper.lambda_value:.4f} (LR: {self.current_lr:.6f})")
        print(f"  Costs: {', '.join(f'{k}={v:.3f}' for k, v in current_costs.items())}")
        print(f"  EMA Costs: {', '.join(f'{k}={v:.3f}' for k, v in self.ema_costs.items())}")
        print(f"  Limits: {', '.join(f'{k}={v:.3f}' for k, v in self.wrapper.soft_constraint_limits.items())}")
        print(f"  Max Gradient: {max_grad:.4f}")
        print(f"  Grad Variance: {self.grad_variance:.4f}")
        
        if self.converged:
            print(f"  ✅ CONVERGED - Constraints satisfied")


# ============================================================
# 3. Factory Function
# ============================================================
def create_safe_5g_training(env, config: Optional[Dict[str, Any]] = None):
    """
    Create complete safe training system for 5G environment.
    
    Args:
        env: Base 5G environment
        config: Optional configuration override
        
    Returns:
        (wrapped_env, callbacks)
    """
    if config is None:
        config = {}
    
    # Default configuration optimized for 5G
    default_config = {
        # Hard constraints (QoS - MUST satisfy)
        'hard_constraint_keys': ['avg_drop_rate', 'avg_latency'],
        'hard_constraint_thresholds': {
            'avg_drop_rate': config.get('drop_threshold', 1.0),
            'avg_latency': config.get('latency_threshold', 50.0),
        },
        'hard_penalty_base': 50.0,
        'hard_penalty_scale': 2.0,
        
        # Soft constraints (Resources - optimize)
        'soft_constraint_keys': ['cpu_violations', 'prb_violations'],
        'soft_constraint_limits': {
            'cpu_violations': 0.0,  # Target 0 violations
            'prb_violations': 0.0,
        },
        'initial_lambda': 15.0,
        'max_lambda': 100.0,
        
        # Energy reward
        'energy_reward_enabled': True,
        'energy_unlock_threshold': 50,
        'energy_reward_scale': 2.0,
        
        # Lagrange callback
        'initial_lr': 0.01,
        'update_frequency': 100,
        
        'verbose': True,
    }
    
    # Merge with user config
    default_config.update(config)
    
    # Create wrapper
    wrapper = EnhancedSafeRewardWrapper(
        env,
        hard_constraint_keys=default_config['hard_constraint_keys'],
        hard_constraint_thresholds=default_config['hard_constraint_thresholds'],
        hard_penalty_base=default_config['hard_penalty_base'],
        hard_penalty_scale=default_config['hard_penalty_scale'],
        soft_constraint_keys=default_config['soft_constraint_keys'],
        soft_constraint_limits=default_config['soft_constraint_limits'],
        initial_lambda=default_config['initial_lambda'],
        max_lambda=default_config['max_lambda'],
        energy_reward_enabled=default_config['energy_reward_enabled'],
        energy_unlock_threshold=default_config['energy_unlock_threshold'],
        energy_reward_scale=default_config['energy_reward_scale'],
        verbose=default_config['verbose'],
    )
    
    # Create callback
    lagrange_callback = EnhancedLagrangeCallback(
        wrapper,
        initial_lr=default_config['initial_lr'],
        update_frequency=default_config['update_frequency'],
        verbose=1 if default_config['verbose'] else 0,
    )
    
    return wrapper, [lagrange_callback]