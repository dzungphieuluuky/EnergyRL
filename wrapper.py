import gymnasium as gym
from typing import List, Dict, Any
from fiveg_env import FiveGEnv

class LagrangianRewardWrapper(gym.Wrapper):
    """
    Implements a Lagrangian reward structure by separating the primary objective (reward)
    from the constraints (costs). The final reward is `PrimalReward - dot(Lambdas, Costs)`.
    """
    def __init__(self, env: FiveGEnv, constraint_keys: List[str]):
        super().__init__(env)
        self.env = env
        self.constraint_keys = constraint_keys
        self.lambdas = {key: 1.0 for key in self.constraint_keys}
        
    def _compute_primal_reward(self, metrics: Dict[str, Any]) -> float:
        """The agent's primary objective: maximize energy efficiency."""
        p = self.env.sim_params
        total_energy = metrics.get('total_energy', 0)
        max_power_consumption = 10**((p.maxTxPower - 30) / 10)
        max_possible_energy = self.env.n_cells * (p.idlePower + max_power_consumption)
        energy_efficiency = 1.0 - (total_energy / max(1, max_possible_energy))
        return max(0.0, energy_efficiency)

    def _compute_costs(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the non-negative cost for each constraint violation."""
        p = self.env.sim_params
        costs = {key: 0.0 for key in self.constraint_keys}
        
        # Calculate cost for each constraint as normalized severity
        if metrics.get('avg_drop_rate', 0) > p.dropCallThreshold:
            costs['drop_rate'] = (metrics['avg_drop_rate'] - p.dropCallThreshold) / p.dropCallThreshold
        if metrics.get('avg_latency', 0) > p.latencyThreshold:
            costs['latency'] = (metrics['avg_latency'] - p.latencyThreshold) / p.latencyThreshold
        if metrics.get('cpu_violations', 0) > 0:
            costs['cpu_violations'] = metrics['cpu_violations'] / self.env.n_cells
        if metrics.get('prb_violations', 0) > 0:
            costs['prb_violations'] = metrics['prb_violations'] / self.env.n_cells
        return costs

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        primal_reward = self._compute_primal_reward(info)
        costs = self._compute_costs(info)
        lagrangian_penalty = sum(self.lambdas[key] * costs[key] for key in self.constraint_keys)
        reward = primal_reward - lagrangian_penalty
        info['lagrangian_costs'] = costs # Pass costs to the callback
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def update_lambdas(self, new_lambdas: Dict[str, float]):
        """Allows the callback to update the lambda values."""
        for key, value in new_lambdas.items():
            if key in self.lambdas: self.lambdas[key] = value
class StrictConstraintWrapper(gym.Wrapper):
    """
    Zero reward if ANY constraint is violated.
    Energy efficiency reward ONLY when all constraints satisfied.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.consecutive_compliant_steps = 0
        
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        metrics = self.env.compute_metrics()
        
        # Check ALL constraints
        violations = self._check_all_constraints(metrics)
        has_violations = any(violations.values())
        
        if has_violations:
            # ZERO reward during violations
            reward = 0.0
            self.consecutive_compliant_steps = 0
        else:
            # Energy reward ONLY when compliant
            self.consecutive_compliant_steps += 1
            reward = self._compute_energy_reward(metrics)
            
            # Bonus for sustained compliance
            if self.consecutive_compliant_steps > 20:
                reward += np.log1p(self.consecutive_compliant_steps - 20) * 0.2
        
        # Enhanced logging
        info['strict_reward'] = {
            'has_violations': has_violations,
            'energy_reward': reward if not has_violations else 0.0,
            'consecutive_compliant_steps': self.consecutive_compliant_steps,
            'violations': violations
        }
        
        return obs, reward, terminated, truncated, info
    
    def _check_all_constraints(self, metrics):
        """Check all QoS and resource constraints."""
        p = self.env.sim_params
        
        violations = {
            'drop_rate': metrics['avg_drop_rate'] > p.drop_call_threshold,
            'latency': metrics['avg_latency'] > p.latency_threshold,
            'cpu': metrics['cpu_violations'] > 0,
            'prb': metrics['prb_violations'] > 0,
            'connectivity': metrics.get('connection_rate', 1.0) < 0.95
        }
        
        return violations
    
    def _compute_energy_reward(self, metrics):
        """Compute energy efficiency reward (0-1 scale)."""
        total_energy = metrics['total_energy']
        
        # Calculate maximum possible energy
        p = self.env.sim_params
        max_power = 10**((p.max_tx_power - 30)/10)
        max_energy = self.env.n_cells * (p.base_power + max_power)
        
        if max_energy > 0:
            efficiency = 1.0 - (total_energy / max_energy)
            return max(0.0, efficiency)
        return 0.0