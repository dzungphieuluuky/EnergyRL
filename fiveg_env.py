# fiveg_env.py
import os
import math
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional

from fiveg_objects import Cell, UE, SimulationParams
from simulation_logic import (
    update_ue_mobility, update_signal_measurements, check_handover_events,
    handle_disconnected_ues, update_traffic_generation,
    update_ue_drop_events, update_cell_resource_usage
)
from scenario_creator import load_scenario_config, create_sites, configure_cells, initialize_ues

def run_simulation_step(
    ues: List[UE], cells: List[Cell], sim_params: SimulationParams,
    time_step: float, current_time: float, seed: int, action: Optional[np.ndarray] = None
) -> Tuple[List[UE], List[Cell]]:
    """Runs one step of the simulation logic."""
    if action is not None:
        for i, cell in enumerate(cells):
            if i < len(action):
                power_range = cell.max_tx_power - cell.min_tx_power
                cell.tx_power = cell.min_tx_power + action[i] * power_range

    ues = update_ue_mobility(ues, time_step, current_time, seed)
    ues = update_signal_measurements(ues, cells, sim_params.rsrpMeasurementThreshold, current_time, seed)
    ues = handle_disconnected_ues(ues, cells, sim_params, time_step, current_time)
    _, ues = check_handover_events(ues, cells, current_time, sim_params, seed)
    ues, cells = update_traffic_generation(ues, cells, current_time, sim_params)
    ues, cells = update_ue_drop_events(ues, cells, current_time)
    cells = update_cell_resource_usage(cells, ues)
    return ues, cells

class FiveGEnv(gym.Env):
    """
    A Gymnasium environment for 5G network simulation.
    This base environment is responsible for the simulation physics only.
    Reward shaping is handled by wrappers.
    """
    def __init__(self, env_config: Dict[str, Any], max_cells: int = 57):
        super().__init__()
        self.sim_params = load_scenario_config(env_config.get('deploymentScenario', 'dense_urban'))
        for key, value in env_config.items():
            if hasattr(self.sim_params, key):
                setattr(self.sim_params, key, value)

        self.max_cells = int(max_cells)
        self.state_dim_per_cell = 25
        self.seed = int(env_config.get('seed', 42))

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.max_cells,), dtype=np.float32)
        state_dim = 17 + 14 + (self.max_cells * self.state_dim_per_cell)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        self.cells: List[Cell] = []
        self.ues: List[UE] = []
        self.n_cells = 0
        self.n_ues = 0
        self.current_step = 0

    def _setup_scenario(self, seed: int):
        sites = create_sites(self.sim_params, seed)
        self.cells = configure_cells(sites, self.sim_params)
        self.ues = initialize_ues(self.sim_params, sites, seed)
        self.n_cells = len(self.cells)
        self.n_ues = len(self.ues)
        if self.n_cells > self.max_cells:
            raise ValueError(f"Scenario has {self.n_cells} cells, but env configured for max {self.max_cells}.")

    def _get_obs(self) -> np.ndarray:
        p = self.sim_params
        sim_features = [
            float(self.n_cells), float(self.n_ues), float(p.simTime), float(p.timeStep),
            float(self.current_step / max(1.0, p.total_steps)), float(p.carrierFrequency),
            float(p.isd), float(p.minTxPower), float(p.maxTxPower), float(p.basePower), float(p.idlePower),
            float(p.dropCallThreshold), float(p.latencyThreshold), float(p.cpuThreshold),
            float(p.prbThreshold), float(p.trafficLambda), float(p.peakHourMultiplier)
        ]
        metrics = self.compute_metrics()
        network_features = list(metrics.values())
        
        cell_features = np.zeros(self.max_cells * self.state_dim_per_cell, dtype=np.float32)
        for i, cell in enumerate(self.cells):
            stats = self._get_ue_stats_for_cell(cell.id)
            power_range = max(1e-6, cell.max_tx_power - cell.min_tx_power)
            power_ratio = (cell.tx_power - cell.min_tx_power) / power_range
            cell_feats_list = [
                float(cell.tx_power), float(cell.energy_consumption), float(cell.cpu_usage), float(cell.prb_usage),
                float(cell.max_capacity), float(cell.current_load), float(cell.current_load / max(cell.max_capacity, 1)),
                float(cell.ttt), float(cell.a3_offset), float(len(cell.connected_ues)),
                float(stats['active_sessions']), float(stats['total_traffic']),
                *stats['rsrp_stats'], *stats['rsrq_stats'], *stats['sinr_stats'], float(power_ratio)
            ]
            start_idx = i * self.state_dim_per_cell
            cell_features[start_idx:start_idx + self.state_dim_per_cell] = cell_feats_list
            
        return np.concatenate([np.array(sim_features), np.array(network_features), cell_features]).astype(np.float32)

    def _get_ue_stats_for_cell(self, cell_id: int) -> Dict[str, Any]:
        ue_metrics = [(ue.rsrp, ue.rsrq, ue.sinr, ue.traffic_demand) for ue in self.ues if ue.serving_cell == cell_id and not ue.is_dropped]
        if not ue_metrics:
            return {'active_sessions': 0.0, 'total_traffic': 0.0, 'rsrp_stats': [-140.0]*4, 'rsrq_stats': [-20.0]*4, 'sinr_stats': [-20.0]*4}
        
        rsrps, rsrqs, sinrs, traffic = zip(*ue_metrics)
        def safe_stats(data, default):
            arr = np.array([d for d in data if np.isfinite(d)], dtype=np.float32)
            return [float(np.mean(arr)), float(np.min(arr)), float(np.max(arr)), float(np.std(arr))] if arr.size > 0 else [default]*4
        
        return {'active_sessions': float(len(ue_metrics)), 'total_traffic': float(np.sum(traffic)),
                'rsrp_stats': safe_stats(rsrps, -140.0), 'rsrq_stats': safe_stats(rsrqs, -20.0), 'sinr_stats': safe_stats(sinrs, -20.0)}

    def compute_metrics(self) -> Dict[str, float]:
        if not self.cells: return {k: 0.0 for k in ["total_energy", "active_cells", "avg_drop_rate", "avg_latency", "total_traffic", "connected_ues", "connection_rate", "cpu_violations", "prb_violations", "max_cpu_usage", "max_prb_usage", "kpi_violations", "total_tx_power", "avg_power_ratio"]}

        connected_ues = sum(len(c.connected_ues) for c in self.cells)
        power_range = self.sim_params.maxTxPower - self.sim_params.minTxPower
        
        metrics = {
            "total_energy": sum(c.energy_consumption for c in self.cells),
            "active_cells": sum(1 for c in self.cells if len(c.connected_ues) > 0),
            "avg_drop_rate": np.mean([c.drop_rate for c in self.cells]),
            "avg_latency": np.mean([c.avg_latency for c in self.cells]),
            "total_traffic": sum(c.current_load for c in self.cells),
            "connected_ues": connected_ues,
            "connection_rate": connected_ues / max(1, self.n_ues),
            "cpu_violations": sum(1 for c in self.cells if c.cpu_usage > self.sim_params.cpuThreshold),
            "prb_violations": sum(1 for c in self.cells if c.prb_usage > self.sim_params.prbThreshold),
            "max_cpu_usage": max((c.cpu_usage for c in self.cells), default=0.0),
            "max_prb_usage": max((c.prb_usage for c in self.cells), default=0.0),
            "total_tx_power": sum(c.tx_power for c in self.cells),
            "avg_power_ratio": np.mean([(c.tx_power - self.sim_params.minTxPower) / power_range for c in self.cells]) if power_range > 0 else 0.0
        }
        metrics["kpi_violations"] = float(metrics["avg_drop_rate"] > self.sim_params.dropCallThreshold) + float(metrics["avg_latency"] > self.sim_params.latencyThreshold) + metrics["cpu_violations"] + metrics["prb_violations"]
        return metrics

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.seed = seed if seed is not None else self.seed
        np.random.seed(self.seed)
        
        self.current_step = 0
        self._setup_scenario(self.seed)
        
        # Ensure all UEs start in a non-dropped state
        for ue in self.ues: ue.is_dropped = False

        self.ues, self.cells = run_simulation_step(self.ues, self.cells, self.sim_params, self.sim_params.timeStep, -1, self.seed)
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        active_action = action[:self.n_cells]
        self.ues, self.cells = run_simulation_step(self.ues, self.cells, self.sim_params, self.sim_params.timeStep, self.current_step, self.seed, active_action)
        
        metrics = self.compute_metrics()
        self.current_step += 1
        terminated = self.current_step >= self.sim_params.total_steps
        
        # Return a placeholder reward of 0.0. The wrapper will calculate the true reward.
        return self._get_obs(), 0.0, terminated, False, metrics