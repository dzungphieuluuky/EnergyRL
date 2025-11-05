from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
from collections import defaultdict
from typing import Dict, Any
import logging
import torch

class ConstraintMonitorCallback(BaseCallback):
    """Enhanced monitoring with violation tracking and adaptive penalties."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.violation_history = []
        self.compliance_history = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Check if episode just finished
        if self.locals.get('dones', [False])[0]:
            # Get final metrics from info
            info = self.locals.get('infos', [{}])[0]
            
            # Track violations
            kpi_violations = info.get('kpi_violations', 0)
            self.violation_history.append(kpi_violations)
            
            # Track compliance
            drop_ok = info.get('avg_drop_rate', 999) <= self.training_env.get_attr('sim_params')[0].dropCallThreshold
            latency_ok = info.get('avg_latency', 999) <= self.training_env.get_attr('sim_params')[0].latencyThreshold
            cpu_ok = info.get('cpu_violations', 999) == 0
            prb_ok = info.get('prb_violations', 999) == 0
            
            all_ok = drop_ok and latency_ok and cpu_ok and prb_ok
            self.compliance_history.append(all_ok)
            
            # Log every 10 episodes
            if len(self.violation_history) % 10 == 0:
                recent_compliance = np.mean(self.compliance_history[-10:]) * 100
                recent_violations = np.mean(self.violation_history[-10:])
                
                print(f"\n[Step {self.num_timesteps}] Last 10 Episodes:")
                print(f"  Compliance Rate: {recent_compliance:.1f}%")
                print(f"  Avg Violations: {recent_violations:.2f}")
                
                if recent_compliance < 50:
                    print("  ⚠️  WARNING: Low compliance rate! Agent struggling with constraints.")
                elif recent_compliance > 80:
                    print("  ✅ Good compliance! Agent learning constraint satisfaction.")
        
        return True

class AlgorithmComparisonCallback(BaseCallback):
    """Callback to track training progress and compare algorithm performance."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += sum(self.locals['rewards'])
            
        dones = self.locals.get('dones', [])
        if any(dones):
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {len(self.episode_rewards)}, Last 10 eps mean reward: {mean_reward:.3f}")
        
        return True
    
class LambdaUpdateCallback(BaseCallback):
    """Callback to update Lagrange multipliers."""
    
    def __init__(self, lambda_lr: float = 0.01, update_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr
        self.update_freq = update_freq
        self.lambda_history = defaultdict(list)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Update lambda values at the end of each rollout."""
        # Get constraint violations from all environments
        if hasattr(self.training_env, 'get_attr'):
            # VecEnv case
            try:
                # Get lambdas and violations from the first environment
                env_lambdas = self.training_env.get_attr('lambdas')[0]
                env_violations = self.training_env.get_attr('constraint_violations')[0]
                
                # Update lambdas based on violations
                for key in env_lambdas.keys():
                    violation = env_violations.get(key, 0.0)
                    old_lambda = env_lambdas[key]
                    new_lambda = max(0.0, old_lambda + self.lambda_lr * violation)
                    
                    # Set updated lambda to all environments
                    for env_idx in range(self.training_env.num_envs):
                        self.training_env.env_method('set_lambda', key, new_lambda, indices=[env_idx])
                    
                    # Store history
                    self.lambda_history[key].append(new_lambda)
                
                # Log the updates
                if self.verbose > 0 and hasattr(self, 'logger') and self.logger is not None:
                    latest_values = {f"lambda/{key}": val[-1] for key, val in self.lambda_history.items()}
                    log_string = self._format_log_string(latest_values)
                    self.logger.info(f"Lambda Update: {log_string}")
                    
            except (AttributeError, IndexError) as e:
                if self.verbose > 0:
                    print(f"Warning: Could not update lambdas: {e}")
    
    def _format_log_string(self, values: dict[str, float]) -> str:
        """Format lambda values as a readable string."""
        if not isinstance(values, dict):
            return str(values)
        
        parts = []
        for key, value in values.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

class AdamLambdaUpdateCallback(BaseCallback):
    """
    Updates Lagrange multipliers (lambdas) using the Adam optimizer.
    This provides momentum and adaptive learning rates for more stable updates.

    :param constraint_keys: A list of keys for the constraints (e.g., ['drop_rate', 'latency']).
    :param initial_lambdas: A dictionary of initial lambda values.
    :param lambda_lr: The learning rate for the Adam optimizer.
    :param update_freq: How often to update the lambdas (in steps).
    """
    def __init__(self, 
                 constraint_keys: list[str],
                 initial_lambda_value: float = 1.0,
                 lambda_lr: float = 0.01, 
                 update_freq: int = 1000, 
                 verbose: int = 1):
        super().__init__(verbose)
        self.lambda_lr = lambda_lr
        self.update_freq = update_freq
        if constraint_keys is None:
            self.constraint_keys = ['drop_rate', 'latency', 'cpu_usage', 'prb_usage']
        else:
            self.constraint_keys = constraint_keys
            
        # --- 1. Set up the Lambdas as PyTorch Parameters ---
        # We treat the lambdas as learnable parameters so we can use a PyTorch optimizer.
        # We store the log of the lambdas for numerical stability and to ensure they remain non-negative.
        initial_log_lambdas = {key: torch.tensor(np.log(initial_lambda_value), dtype=torch.float32) 
                               for key in self.constraint_keys}
        self.log_lambdas = torch.nn.ParameterDict(initial_log_lambdas)

        # --- 2. Initialize the Adam Optimizer ---
        # The optimizer will manage the updates for our log_lambdas parameters.
        self.optimizer = torch.optim.Adam(self.log_lambdas.parameters(), lr=self.lambda_lr)

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            # --- 3. Collect and Average Costs ---
            all_costs = {key: [] for key in self.constraint_keys}
            for info in self.locals.get("infos", []):
                if 'lagrangian_costs' in info:
                    for key, cost in info['lagrangian_costs'].items():
                        all_costs[key].append(cost)
            
            mean_costs = {key: np.mean(values) if values else 0.0 
                          for key, values in all_costs.items()}
            
            # --- 4. Perform the Gradient Update with Adam ---
            self.optimizer.zero_grad(set_to_none=True)
            
            # The "loss" for our dual problem is `- (lambda * cost)`.
            # To perform gradient ascent (maximize this), we minimize the negative of it.
            # This is equivalent to `d_lambda = cost`.
            loss = 0
            for key in self.constraint_keys:
                # We work with log_lambdas, so we exponentiate to get the actual lambda value.
                # This ensures lambda is always >= 0.
                lambda_val = torch.exp(self.log_lambdas[key])
                loss -= lambda_val * mean_costs[key]
            
            # Backpropagate to compute gradients (d_loss / d_lambda)
            loss.backward()
            
            # Adam takes a step to update the log_lambdas
            self.optimizer.step()
            
            # --- 5. Update Lambdas in the Environment and Log ---
            with torch.no_grad():
                current_lambdas = {key: torch.exp(self.log_lambdas[key]).item() 
                                   for key in self.constraint_keys}
            
            self.training_env.env_method('update_lambdas', current_lambdas)
            
            if self.verbose > 0 and self.n_calls % (self.update_freq * 10) == 0:
                print(f"\n[AdamLambdaUpdate] Step {self.num_timesteps}:")
                for key in self.constraint_keys:
                    print(f"  - {key:<20s} | λ: {current_lambdas[key]:<8.4f} | Cost: {mean_costs[key]:.4f}")

            # Log to TensorBoard
            for key, value in current_lambdas.items():
                self.logger.record(f'lagrangian/lambda_{key}', value)
            for key, value in mean_costs.items():
                self.logger.record(f'lagrangian/cost_{key}', value)
        return True
    
    def _format_log_string(self, values: dict[str, float]) -> str:
        """Format lambda values as a readable string."""
        if not isinstance(values, dict):
            return str(values)
        
        parts = []
        for key, value in values.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

class CurriculumLearningCallback(BaseCallback):
    """
    Callback for automatic curriculum learning - advances training stages
    based on performance metrics.
    """
    
    def __init__(self, eval_freq: int = 10000, compliance_threshold: float = 0.85, 
                 min_steps_in_stage: int = 50000, verbose: int = 1):
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.compliance_threshold = compliance_threshold
        self.min_steps_in_stage = min_steps_in_stage
        self.last_eval_step = 0
        self.current_stage = "early"
        
    def _on_step(self) -> bool:
        # Check if it's time to evaluate for stage advancement
        if self.n_calls - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.n_calls
            
            # Get compliance rate from environments
            compliance_rates = []
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'reward_computer'):
                    stats = env.env.reward_computer.get_stats()
                    compliance_rates.append(stats.get('compliance_rate', 0))
            
            if compliance_rates:
                avg_compliance = np.mean(compliance_rates) / 100.0  # Convert from percentage
                
                # Check if we should advance to next stage
                if (self.n_calls >= self.min_steps_in_stage and 
                    avg_compliance >= self.compliance_threshold):
                    
                    self._advance_training_stage()
        
        return True
    
    def _advance_training_stage(self):
        """Advance all environments to next training stage."""
        stages = ["early", "medium", "stable"]
        current_index = stages.index(self.current_stage)
        
        if current_index < len(stages) - 1:
            new_stage = stages[current_index + 1]
            self.current_stage = new_stage
            
            # Advance stage in all environments
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'reward_computer'):
                    env.env.reward_computer.advance_training_stage()
            
            if self.verbose >= 1:
                print(f"\n🎓 CURRICULUM: Advanced to {new_stage} training stage at step {self.n_calls}")

class LoggedEvalCallback(EvalCallback):
    """EvalCallback with enhanced logging for evaluation results."""
    
    def __init__(self, eval_env, log_path: str = 'eval_logs/', eval_freq: int = 10000, 
                 best_model_save_path: str = 'best_models/', verbose: int = 1):
        super(LoggedEvalCallback, self).__init__(
            eval_env=eval_env,
            log_path=log_path,
            eval_freq=eval_freq,
            best_model_save_path=best_model_save_path,
            verbose=verbose
        )
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Additional logging
        if self.n_calls % self.eval_freq == 0:
            mean_reward = np.mean(self.last_mean_reward)
            std_reward = np.std(self.last_mean_reward)
            self.logger.info(f"[Evaluation at step {self.num_timesteps}] Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return result

class FileLoggingCallback(BaseCallback):
    """
    A custom callback that intercepts Stable Baselines3's logging data
    and writes it to a structured log file.

    :param log_path: Path to the log file.
    """
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        
        # --- 1. Set up the Python logger ---
        self.file_logger = logging.getLogger(__name__)
        self.file_logger.setLevel(logging.INFO)
        
        # Prevent the logger from propagating messages to the root logger
        self.file_logger.propagate = False
        
        # Create a file handler
        file_handler = logging.FileHandler(self.log_path, mode='a') # 'w' to overwrite on each run
        file_handler.setLevel(logging.INFO)
        # Create a formatter and set it for the handler
        # We only want the raw message, no extra timestamps or levels
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        self.file_logger.addHandler(file_handler)
        
        print(f"\n[FileLoggingCallback] Logging training data to: {self.log_path}")

    def _on_step(self):
        return super()._on_step()
    
    def _on_rollout_end(self) -> bool:
        """
        This method is called at the end of each rollout.
        It's the perfect place to access the aggregated logs.
        """
        # --- 2. Get the latest logged values from the SB3 logger ---
        # This is a dictionary containing all the data like 'rollout/ep_rew_mean', etc.
        latest_values = self.locals

        if not latest_values:
            return True

        # --- 3. Format the data into a nice, readable string ---
        log_string = self._format_log_string(latest_values)
        
        # --- 4. Write the formatted string to the file ---
        self.file_logger.info(log_string)
        
        return True
    
    def _format_log_string(self, values: dict) -> str:
        """Formats the dictionary of values into a table-like string."""
        
        # Group keys by their prefix (e.g., 'lagrangian', 'rollout', 'time')
        log_groups = {}
        for key, value in values.items():
            if '/' in key:
                group, metric = key.split('/', 1)
                if group not in log_groups:
                    log_groups[group] = {}
                log_groups[group][metric] = value
        
        # Build the multi-line string
        output = f"-------------------[ Timestep {self.num_timesteps} ]-------------------\n"
        
        # Define the order of groups for consistent logging
        group_order = ['lagrangian', 'rollout', 'time', 'train']
        
        for group in group_order:
            if group in log_groups:
                output += f"| {group}/\n"
                # Sort metrics within the group for consistent order
                for metric, value in sorted(log_groups[group].items()):
                    # Format numbers to be clean and aligned
                    if isinstance(value, (float, np.floating)):
                        output += f"|    {metric:<20s} | {value:<10.3g}\n"
                    else:
                        output += f"|    {metric:<20s} | {value:<10}\n"

        output += "--------------------------------------------------------\n\n"
        return output