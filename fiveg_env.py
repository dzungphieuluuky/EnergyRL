# fiveg_env.py
"""
Merged FiveG environment:
- Implements FiveGEnv class (API/variable names matching your uploaded file)
- Contains run_simulation_step(...) as the internal simulation driver (replaces sim.run_simulation_step)
- Uses Cell and UE lightweight classes for attribute access (cell.txPower, ue.rsrp, ...)
- Reuses path-loss, measurement, CPU/PRB/energy math from the translated MATLAB PDF
"""
import os
import math
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional

from fiveg_objects import Cell, UE
from numba_utils import NumbaUE, NumbaCell
from simulation_logic import *
from scenario_creator import *

neighbor_dtype = np.int32

class FiveGEnv(gym.Env):
    """A Gymnasium environment optimized for external curriculum training."""

    def __init__(self, env_config: Dict[str, Any], max_cells: int = 57) -> None:
        super().__init__()
        self.config: Dict[str, Any] = dict(env_config)
        self._set_default_config()

        self.time_step_duration: float = float(self.config['timeStep'])
        self.max_time_steps: int = int(self.config['simTime'])
        self.max_neighbors: int = 8
        self.max_cells: int = int(max_cells)
        self.state_dim_per_cell: int = 25

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_cells,), dtype=np.float32)
        state_dim: int = 17 + 14 + (self.max_cells * self.state_dim_per_cell)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.cells: List[Cell] = []
        self.ues: List[UE] = []
        self.n_cells: int = 0
        self.n_ues: int = 0
        self.current_step: int = 0
        self.total_episodes: int = 0
        self.previous_powers: np.ndarray = np.zeros(self.max_cells, dtype=np.float32)
        self.neighbor_measurements: Optional[np.ndarray] = None
        self._warm_up_numba()

    def _warm_up_numba(self):
        # This function is unchanged and correct
        print("Warming up Numba functions...")
        dummy_ue = NumbaUE()
        dummy_cell = NumbaCell()
        dummy_ues = [dummy_ue]; dummy_cells = [dummy_cell]
        numba_update_signal_measurements(dummy_ues, dummy_cells, -115.0, 0.0, 42)
        numba_update_ue_mobility(dummy_ues, 1.0, 0.0, 42, 1000.0)
        numba_update_cell_resource_usage(dummy_ues, dummy_cells)
        numba_generate_traffic(dummy_ues, 30.0, 1, 42, 0)
        print("Numba functions warmed up!")

    def _set_default_config(self) -> None:
        # This function is unchanged and correct
        defaults: Dict[str, Any] = {
            'timeStep': 1.0, 'simTime': 500, 'carrierFrequency': 3.5e9,
            'minTxPower': 30.0, 'maxTxPower': 46.0, 'basePower': 1000.0,
            'idlePower': 250.0, 'dropCallThreshold': 1.0, 'latencyThreshold': 50.0,
            'cpuThreshold': 80.0, 'prbThreshold': 80.0, 'trafficLambda': 30.0,
            'peakHourMultiplier': 1.0, 'numSites': 7, 'numUEs': 210, 'isd': 200.0,
            'deploymentScenario': 'indoor_hotspot', 'seed': 42
        }
        for key, value in defaults.items(): self.config.setdefault(key, value)

    def _setup_scenario(self):
        # This function is unchanged and correct
        sites = create_hex_layout(self.config['numSites'], self.config.get('isd', 200.0), int(self.config.get('seed', 42)))
        self.cells = configure_cells_from_sites(self.config, sites)
        ues = initialize_ues_from_config(self.config, sites, int(self.config.get('seed', 42)))
        self.ues = ues
        self.n_cells = len(self.cells); self.n_ues = len(self.ues)
        if self.n_cells > self.max_cells: raise ValueError(f"Scenario has {self.n_cells} cells, but env configured for max {self.max_cells}.")

    # ------------------------------
    # Observation builder (mirrors createRLState/mapping in your uploaded file)
    # ------------------------------
    def _get_obs(self):
        sim_features = [
            float(self.n_cells), float(self.n_ues), float(self.config['simTime']), float(self.config['timeStep']),
            float(self.current_step / max(1.0, self.config['simTime'])), float(self.config['carrierFrequency']),
            float(self.config.get('isd', 500.0)), float(self.config['minTxPower']), float(self.config['maxTxPower']),
            float(self.config['basePower']), float(self.config['idlePower']), float(self.config['dropCallThreshold']),
            float(self.config['latencyThreshold']), float(self.config['cpuThreshold']), float(self.config['prbThreshold']),
            float(self.config['trafficLambda']), float(self.config.get('peakHourMultiplier', 1.0))
        ]

        metrics = self.compute_metrics()
        network_features = [
            float(self.total_energy_kwh), float(metrics.get("activeCells", 0)), float(metrics.get("avgDropRate", 0)),
            float(metrics.get("avgLatency", 0)), float(metrics.get("totalTraffic", 0)), float(metrics.get("connectedUEs", 0)),
            float(metrics.get("connectionRate", 0)), float(metrics.get("cpuViolations", 0)), float(metrics.get("prbViolations", 0)),
            float(metrics.get("maxCpuUsage", 0)), float(metrics.get("maxPrbUsage", 0)),
            float(metrics.get("kpiViolations", 0)), float(metrics.get("totalTxPower", 0)), float(metrics.get("avgPowerRatio", 0))
        ]

        # cell features array sized to max_cells
        cell_features = np.zeros(self.max_cells * self.state_dim_per_cell, dtype=np.float32)
        for i, cell in enumerate(self.cells):
            stats = self._get_ue_stats_for_cell(cell.id)
            cell_feats_list = [
                float(cell.txPower), float(cell.energyConsumption), float(cell.cpuUsage), float(cell.prbUsage),
                float(cell.maxCapacity), float(cell.currentLoad), float(cell.currentLoad / (cell.maxCapacity or 1)),
                float(cell.ttt), float(cell.a3Offset), float(len(cell.connectedUEs)),
                float(stats['active_sessions']), float(stats['total_traffic']),
                float(stats['avg_rsrp']), float(stats['min_rsrp']), float(stats['max_rsrp']), float(stats['std_rsrp']),
                float(stats['avg_rsrq']), float(stats['min_rsrq']), float(stats['max_rsrq']), float(stats['std_rsrq']),
                float(stats['avg_sinr']), float(stats['min_sinr']), float(stats['max_sinr']), float(stats['std_sinr']),
                float(getattr(cell, 'power_ratio', 1.0))
            ]
            start_idx = i * self.state_dim_per_cell
            cell_features[start_idx: start_idx + self.state_dim_per_cell] = np.array(cell_feats_list, dtype=np.float32)

        obs = np.concatenate([np.array(sim_features, dtype=np.float32), np.array(network_features, dtype=np.float32), cell_features]).astype(np.float32)
        return obs

    def _get_ue_stats_for_cell(self, cell_id: int) -> Dict[str, float]:
        """Get UE statistics for a cell with improved numerical stability."""
        ue_metrics = [
            (ue.rsrp, ue.rsrq, ue.sinr, ue.trafficDemand, ue.sessionActive)
            for ue in self.ues if ue.servingCell == cell_id
        ]

        if not ue_metrics:
            return self._get_default_ue_stats()

        rsrps, rsrqs, sinrs, traffic, sessions = zip(*ue_metrics)
        return self._compute_safe_stats(rsrps, rsrqs, sinrs, traffic, sessions)

    def _get_default_ue_stats(self) -> Dict[str, float]:
        """Return default UE statistics for empty cells."""
        return {
            'active_sessions': 0.0, 'total_traffic': 0.0,
            'avg_rsrp': -140.0, 'min_rsrp': -140.0, 'max_rsrp': -140.0, 'std_rsrp': 0.0,
            'avg_rsrq': -20.0, 'min_rsrq': -20.0, 'max_rsrq': -20.0, 'std_rsrq': 0.0,
            'avg_sinr': -20.0, 'min_sinr': -20.0, 'max_sinr': -20.0, 'std_sinr': 0.0
        }

    def _compute_safe_stats(self, rsrps: Tuple[float, ...], rsrqs: Tuple[float, ...],
                           sinrs: Tuple[float, ...], traffic: Tuple[float, ...],
                           sessions: Tuple[bool, ...]) -> Dict[str, float]:
        """Compute statistics with NaN handling and numerical stability."""
        def safe_stats(data: Tuple[float, ...], default_val: float) -> Tuple[float, float, float, float]:
            arr: np.ndarray = np.array(data, dtype=np.float32)
            valid_data: np.ndarray = arr[np.isfinite(arr)]
            if valid_data.size == 0:
                return default_val, default_val, default_val, 0.0
            return float(np.mean(valid_data)), float(np.min(valid_data)), float(np.max(valid_data)), float(np.std(valid_data))

        avg_rsrp, min_rsrp, max_rsrp, std_rsrp = safe_stats(rsrps, -140.0)
        avg_rsrq, min_rsrq, max_rsrq, std_rsrq = safe_stats(rsrqs, -20.0)
        avg_sinr, min_sinr, max_sinr, std_sinr = safe_stats(sinrs, -20.0)

        return {
            'active_sessions': float(np.sum(sessions)),
            'total_traffic': float(np.sum(traffic)),
            'avg_rsrp': avg_rsrp, 'min_rsrp': min_rsrp, 'max_rsrp': max_rsrp, 'std_rsrp': std_rsrp,
            'avg_rsrq': avg_rsrq, 'min_rsrq': min_rsrq, 'max_rsrq': max_rsrq, 'std_rsrq': std_rsrq,
            'avg_sinr': avg_sinr, 'min_sinr': min_sinr, 'max_sinr': max_sinr, 'std_sinr': std_sinr
        }

    # ------------------------------
    # Metrics computation (merged)
    # ------------------------------
    def compute_metrics(self) -> Dict[str, Any]:
        if not self.cells:
            return {}
        total_tx_power = sum(c.txPower for c in self.cells)
        power_range = self.config['maxTxPower'] - self.config['minTxPower']
        avg_power_ratio = np.mean([(c.txPower - c.minTxPower) / power_range for c in self.cells]) if power_range > 0 else 0.0
        connected_ues = sum(1 for ue in self.ues if ue.servingCell is not None)
        drop_rates = [c.dropRate for c in self.cells if not np.isnan(c.dropRate)]
        avg_drop = float(np.mean(drop_rates)) if drop_rates else 0.0
        latencies = [c.avgLatency for c in self.cells if not np.isnan(c.avgLatency)]
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        metrics = {
            "totalEnergy": float(sum(c.energyConsumption for c in self.cells)),
            "activeCells": int(sum(1 for c in self.cells if len(c.connectedUEs) > 0)),
            "avgDropRate": avg_drop, "avgLatency": avg_latency,
            "totalTraffic": float(sum(c.currentLoad for c in self.cells)),
            "connectedUEs": int(connected_ues),
            "connectionRate": (connected_ues / max(self.n_ues, 1)), # Return as ratio, not percentage
            "cpuViolations": int(sum(1 for c in self.cells if c.cpuUsage > self.config['cpuThreshold'])),
            "prbViolations": int(sum(1 for c in self.cells if c.prbUsage > self.config['prbThreshold'])),
            "maxCpuUsage": float(max((c.cpuUsage for c in self.cells), default=0.0)),
            "maxPrbUsage": float(max((c.prbUsage for c in self.cells), default=0.0)),
            "totalTxPower": float(total_tx_power),
            "avgPowerRatio": float(avg_power_ratio)
        }
        metrics["kpiViolations"] = int(metrics["avgDropRate"] > self.config['dropCallThreshold']) + int(metrics["avgLatency"] > self.config['latencyThreshold']) + metrics["cpuViolations"] + metrics["prbViolations"]
        return metrics

    # ------------------------------
    # Reset / step methods (merged behavior)
    # ------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None: self.config['seed'] = int(seed)
        np.random.seed(int(self.config.get('seed', 42)))

        # === INITIALIZE ALL TRACKING VARIABLES HERE ===
        self.current_step = 0
        self.total_energy_kwh = 0.0  # <-- THIS WAS THE MISSING LINE CAUSING THE ERROR
        self.qos_compliant_steps = 0
        self.current_episode_reward = 0.0
        if not hasattr(self, 'total_episodes'): self.total_episodes = 0

        self._setup_scenario()
        
        active_powers = np.array([c.txPower for c in self.cells])
        self.previous_powers = np.pad(active_powers, (0, self.max_cells - len(active_powers)), 'constant')
        self.neighbor_measurements = np.full((self.n_ues, self.max_neighbors), -1, dtype=neighbor_dtype)

        self.ues, self.cells, self.neighbor_measurements = optimized_run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config, self.time_step_duration, -1, action=None
        )
        return self._get_obs(), {}

    def compute_smooth_reward(self, metrics):
        """
        Hard-constraint reward function, simplified for behavioral curriculum.
        This function now ONLY defines the final objective, not how to achieve it.
        """
        avg_drop = max(0.0, metrics.get("avgDropRate", 0.0))
        avg_latency = max(0.0, metrics.get("avgLatency", 0.0))
        connection_rate = metrics.get("connectionRate", 0.0)
        
        drop_threshold = self.config['dropCallThreshold']
        latency_threshold = self.config['latencyThreshold']
        connection_threshold = 0.95
        
        drop_violated = avg_drop > drop_threshold
        latency_violated = avg_latency > latency_threshold
        connection_violated = connection_rate < connection_threshold
        
        constraints_satisfied = not (drop_violated or latency_violated or connection_violated)
        
        if constraints_satisfied: self.qos_compliant_steps += 1
        else: self.qos_compliant_steps = 0
        
        if not constraints_satisfied:
            penalties = []
            if drop_violated: penalties.append(-5.0 * (1.0 + (avg_drop - drop_threshold) / max(drop_threshold, 1e-6)))
            if latency_violated: penalties.append(-5.0 * (1.0 + (avg_latency - latency_threshold) / max(latency_threshold, 1e-6)))
            if connection_violated: penalties.append(-10.0 * (connection_threshold - connection_rate))
            total_penalty = sum(penalties)
            
            # This return block is correct and has no curriculum logic
            return { 'reward': np.tanh(total_penalty / 10.0), 'constraints_satisfied': False, 'constraint_penalties': {'total': total_penalty}, 'metrics': {'qos_compliant_steps': self.qos_compliant_steps} }
        
        base_constraint_reward = 1.0
        drop_quality = np.clip((drop_threshold - avg_drop) / max(drop_threshold, 1e-6), 0, 1)
        latency_quality = np.clip((latency_threshold - avg_latency) / max(latency_threshold, 1e-6), 0, 1)
        connection_quality = np.clip((connection_rate - connection_threshold) / (1.0 - connection_threshold + 1e-6), 0, 1)
        qos_quality_reward = 0.5 * (drop_quality + latency_quality + connection_quality) / 3.0
        
        energy_reward = 0.0
        energy_efficiency = 0.0
        if self.n_cells > 0:
            total_current_power = sum(c.energyConsumption for c in self.cells)
            max_power_per_cell = self.config['basePower'] + 10 ** ((self.config['maxTxPower'] - 30) / 10.0)
            max_possible_power = self.n_cells * max_power_per_cell
            energy_ratio = total_current_power / max(max_possible_power, 1e-6)
            energy_efficiency = 1.0 - energy_ratio
            
            MIN_COMPLIANT_STEPS = 20
            if self.qos_compliant_steps >= MIN_COMPLIANT_STEPS:
                step_unlock = min(1.0, (self.qos_compliant_steps - MIN_COMPLIANT_STEPS) / 80.0)
                energy_reward = 2.0 * energy_efficiency * step_unlock

        # Other reward components (resource, load balance, etc.)
        resource_reward, load_balance_reward, sinr_reward, stability_reward = 0.0, 0.0, 0.0, 0.0
        
        total_reward = (base_constraint_reward + qos_quality_reward + energy_reward + resource_reward + load_balance_reward + sinr_reward + stability_reward)
        
        # Bonuses and Normalization
        streak_bonus = 0.5 * min(1.0, self.qos_compliant_steps / 100.0) if self.qos_compliant_steps >= 50 else 0.0
        total_reward += streak_bonus
        normalized_reward = np.clip(total_reward / 6.5, 0, 1)
        
        return {
            'reward': normalized_reward,
            'constraints_satisfied': True,
            'metrics': {'qos_compliant_steps': self.qos_compliant_steps},
            'components': {'energy_reward': energy_reward}
        }

    def step(self, action):
        active_action = action[:self.n_cells]
        self.ues, self.cells, self.neighbor_measurements = optimized_run_simulation_step(
            self.ues, self.cells, self.neighbor_measurements, self.config,
            self.time_step_duration, self.current_step, active_action
        )
        
        metrics = self.compute_metrics()
        reward_info = self.compute_smooth_reward(metrics)
        reward = reward_info['reward']
        
        self.current_episode_reward += reward
        self.current_step += 1
        terminated = self.current_step >= self.max_time_steps

        metrics['reward_info'] = reward_info
        metrics['is_success'] = reward_info['constraints_satisfied']
        
        if terminated:
            self.total_episodes += 1
            self._log_episode_summary()
            metrics['episode'] = { 'r': self.current_episode_reward, 'l': self.current_step }

        return self._get_obs(), float(reward), bool(terminated), False, metrics

    def _log_episode_summary(self):
        compliance_rate = (self.qos_compliant_steps / self.current_step) * 100 if self.current_step > 0 else 0
        print(f"\n{'='*60}\n[EPISODE {self.total_episodes} COMPLETE]\n  - Total Reward: {self.current_episode_reward:.2f}\n  - Final Compliance Rate: {compliance_rate:.1f}%\n{'='*60}\n")

    def analyze_constraint_feasibility(self, num_episodes=10, power_levels=[0.7, 0.8, 0.9, 1.0]):
        print("\n" + "="*70 + "\nCONSTRAINT FEASIBILITY ANALYSIS\n" + "="*70)
        for power in power_levels:
            print(f"\nTesting with constant power = {power:.1f} (action = {power})\n" + "-" * 70)
            ep_compliances = []
            for ep in range(num_episodes):
                obs, _ = self.reset()
                done = False
                ep_compliant_steps = 0
                ep_steps = 0
                while not done:
                    action = np.ones(self.action_space.shape) * power
                    obs, reward, done, _, info = self.step(action)
                    ep_steps += 1
                    if info.get('reward_info', {}).get('constraints_satisfied', False):
                        ep_compliant_steps += 1
                ep_compliances.append(100 * ep_compliant_steps / ep_steps if ep_steps > 0 else 0)
            avg_compliance = np.mean(ep_compliances)
            print(f"  Average Compliance Rate: {avg_compliance:.1f}%")
            if avg_compliance > 95: print("  ✅ Constraints are ACHIEVABLE at this power level")
            else: print("  ❌ Constraints are NOT reliably achievable")
        print("\n" + "="*70)

if __name__ == "__main__":
    cfg = {'simTime': 500, 'numSites': 3, 'numUEs': 50, 'dropCallThreshold': 1.0, 'latencyThreshold': 50.0}
    env = FiveGEnv(cfg, max_cells=12)
    print("\nRunning Constraint Feasibility Analysis...")
    env.analyze_constraint_feasibility(num_episodes=5, power_levels=[0.7, 0.8, 0.9, 1.0])
