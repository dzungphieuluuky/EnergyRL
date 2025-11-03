# train_ppo_with_bc.py
import numpy as np
import torch
import pickle
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from fiveg_env import FiveGEnv

class BehavioralCloningLogger(BaseCallback):
    """Custom callback to log BC training progress"""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.epoch_losses = []
    
    def _on_step(self):
        return True
    
    def log_epoch(self, epoch, avg_loss, val_loss=None):
        self.epoch_losses.append(avg_loss)
        if val_loss is not None:
            print(f"BC Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            print(f"BC Epoch {epoch}: Train Loss = {avg_loss:.4f}")

class AdaptiveBehavioralCurriculum:
    """Adaptive behavioral policy that can gradually reduce power while maintaining constraints"""
    def __init__(self, action_dim, initial_range=(0.9, 1.0), final_range=(0.4, 0.7), 
                 adaptation_threshold=0.95, patience=10):
        self.action_dim = action_dim
        self.initial_low, self.initial_high = initial_range
        self.final_low, self.final_high = final_range
        self.current_low, self.current_high = initial_range
        self.adaptation_threshold = adaptation_threshold
        self.patience = patience
        self.consecutive_success = 0
        self.episode_count = 0
        
    def update_based_on_performance(self, compliance_rate):
        """Adapt power range based on constraint satisfaction"""
        self.episode_count += 1
        
        if compliance_rate >= self.adaptation_threshold:
            self.consecutive_success += 1
        else:
            self.consecutive_success = 0
            
        # Gradually reduce power range if consistently successful
        if self.consecutive_success >= self.patience and self.current_high > self.final_high:
            progress = min(1.0, self.episode_count / 200)  # Gradual over 200 episodes
            self.current_low = self.initial_low + progress * (self.final_low - self.initial_low)
            self.current_high = self.initial_high + progress * (self.final_high - self.initial_high)
            
            print(f"🎓 Curriculum adapted: [{self.current_low:.3f}, {self.current_high:.3f}]")
            self.consecutive_success = 0  # Reset after adaptation
            
    def sample_action(self, state=None):
        """Sample action from current power range"""
        return np.random.uniform(self.current_low, self.current_high, size=self.action_dim)
    
    def get_expert_action(self, state=None, deterministic=True):
        """Get expert action (can be made state-aware later)"""
        if deterministic:
            return np.full(self.action_dim, (self.current_low + self.current_high) / 2)
        return self.sample_action(state)

def create_training_env(config, n_envs=4):
    """Create vectorized training environment"""
    def make_env(rank):
        def _init():
            env = FiveGEnv(config)
            env = Monitor(env)
            return env
        return _init
    
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
    return env

def generate_expert_dataset(env, behavioral_policy, num_episodes=100, min_samples=5000):
    """
    Generate high-quality expert dataset with performance tracking
    """
    print(f"\n{'='*70}")
    print(f"GENERATING EXPERT DATASET")
    print(f"Target: {num_episodes} episodes or {min_samples} samples")
    print(f"{'='*70}")
    
    expert_observations = []
    expert_actions = []
    episode_compliance_rates = []
    
    total_samples = 0
    episode = 0
    
    while total_samples < min_samples and episode < num_episodes:
        obs = env.reset()
        done = np.array([False])
        episode_samples = 0
        episode_compliant_steps = 0
        
        while not done[0]:
            action = behavioral_policy.sample_action()
            action = np.expand_dims(action, axis=0)  # Add batch dimension for vec env
            
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            expert_observations.append(obs[0].copy())
            expert_actions.append(action[0].copy())
            
            # Track compliance
            if info[0].get('reward_info', {}).get('constraints_satisfied', False):
                episode_compliant_steps += 1
                
            obs = next_obs
            episode_samples += 1
        
        compliance_rate = episode_compliant_steps / episode_samples if episode_samples > 0 else 0
        episode_compliance_rates.append(compliance_rate)
        
        # Update curriculum based on performance
        behavioral_policy.update_based_on_performance(compliance_rate)
        
        total_samples += episode_samples
        episode += 1
        
        if episode % 10 == 0:
            current_compliance = np.mean(episode_compliance_rates[-10:]) if episode_compliance_rates else 0
            print(f"Episode {episode}: {episode_samples} samples | "
                  f"Compliance: {compliance_rate:.1%} | "
                  f"Recent Avg: {current_compliance:.1%} | "
                  f"Total: {total_samples}")

    # Convert to numpy arrays
    expert_observations = np.array(expert_observations)
    expert_actions = np.array(expert_actions)
    
    avg_compliance = np.mean(episode_compliance_rates) if episode_compliance_rates else 0
    print(f"\n✅ Dataset generation complete:")
    print(f"   Episodes: {episode}")
    print(f"   Total samples: {len(expert_actions)}")
    print(f"   Average compliance: {avg_compliance:.1%}")
    print(f"   Final power range: [{behavioral_policy.current_low:.3f}, {behavioral_policy.current_high:.3f}]")
    
    return expert_observations, expert_actions, behavioral_policy

def pretrain_with_behavioral_cloning(model, observations, actions, epochs=10, 
                                   batch_size=256, validation_split=0.2, logger=None):
    """
    Enhanced behavioral cloning with validation and early stopping
    """
    print(f"\n{'='*70}")
    print(f"BEHAVIORAL CLONING PRE-TRAINING")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Validation: {validation_split:.0%}")
    print(f"{'='*70}")
    
    # Split into training and validation
    n_val = int(len(observations) * validation_split)
    if n_val > 0:
        train_obs, val_obs = observations[:-n_val], observations[-n_val:]
        train_acts, val_acts = actions[:-n_val], actions[-n_val:]
    else:
        train_obs, val_obs = observations, observations
        train_acts, val_acts = actions, actions
    
    optimizer = model.policy.optimizer
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.policy.train()
        train_loss = 0
        n_batches = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(train_obs))
        train_obs_shuffled = train_obs[indices]
        train_acts_shuffled = train_acts[indices]
        
        for i in range(0, len(train_obs_shuffled), batch_size):
            batch_obs = train_obs_shuffled[i:i+batch_size]
            batch_acts = train_acts_shuffled[i:i+batch_size]
            
            obs_tensor = obs_as_tensor(batch_obs, model.device)
            acts_tensor = torch.tensor(batch_acts, dtype=torch.float32, device=model.device)
            
            # Get action distribution and log prob
            distribution = model.policy.get_distribution(obs_tensor)
            log_prob = distribution.log_prob(acts_tensor)
            loss = -torch.mean(log_prob)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches if n_batches > 0 else 0
        
        # Validation
        if n_val > 0:
            model.policy.eval()
            with torch.no_grad():
                val_obs_tensor = obs_as_tensor(val_obs, model.device)
                val_acts_tensor = torch.tensor(val_acts, dtype=torch.float32, device=model.device)
                
                val_distribution = model.policy.get_distribution(val_obs_tensor)
                val_log_prob = val_distribution.log_prob(val_acts_tensor)
                val_loss = -torch.mean(val_log_prob).item()
        else:
            val_loss = None
        
        # Logging
        if logger:
            logger.log_epoch(epoch + 1, avg_train_loss, val_loss)
        else:
            if val_loss is not None:
                print(f"BC Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"BC Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}")
        
        # Early stopping
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping after {epoch+1} epochs")
            break
    
    print("✅ Behavioral cloning complete!")
    return avg_train_loss, val_loss

class PowerReductionMonitor(BaseCallback):
    """Monitor power reduction and constraint satisfaction during RL training"""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_powers = []
        self.episode_compliance = []
        self.episode_rewards = []
        
    def _on_step(self):
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            
            # Extract metrics from info
            if 'reward_info' in info:
                reward_info = info['reward_info']
                self.episode_rewards.append(reward_info.get('reward', 0))
                
                # Track compliance
                constraints_satisfied = reward_info.get('constraints_satisfied', False)
                self.episode_compliance.append(1.0 if constraints_satisfied else 0.0)
            
            # Track power usage (you might need to add this to your env's info)
            if 'applied_powers' in info:
                avg_power = np.mean(info['applied_powers'])
                self.episode_powers.append(avg_power)
        
        # Log every 50 episodes
        if len(self.episode_rewards) % 50 == 0 and len(self.episode_rewards) > 0:
            n_recent = min(20, len(self.episode_rewards))
            print(f"\n📊 Training Progress (last {n_recent} episodes):")
            print(f"   Avg Reward: {np.mean(self.episode_rewards[-n_recent:]):.3f}")
            print(f"   Compliance Rate: {np.mean(self.episode_compliance[-n_recent:]):.1%}")
            if self.episode_powers:
                print(f"   Avg Power: {np.mean(self.episode_powers[-n_recent:]):.3f}")
        
        return True

def main():
    # --- Configuration ---
    TOTAL_TIMESTEPS = 1_000_000
    PRETRAIN_EPISODES = 50
    PRETRAIN_EPOCHS = 10
    N_ENVS = 10  # Parallel environments for faster training
    
    config = {
        'simTime': 500, 
        'timeStep': 1,
        'numSites': 4,
        'numUEs': 100
    }
    
    # Create directories
    os.makedirs('sb3_models', exist_ok=True)
    os.makedirs('sb3_logs', exist_ok=True)
    os.makedirs('expert_data', exist_ok=True)
    
    # --- Environment Setup ---
    print("Initializing environments...")
    env = create_training_env(config, n_envs=N_ENVS)
    eval_env = create_training_env(config, n_envs=1)
    
    # --- Behavioral Policy Setup ---
    behavioral_policy = AdaptiveBehavioralCurriculum(
        action_dim=env.action_space.shape[0],
        initial_range=(0.9, 1.0),
        final_range=(0.4, 0.7),
        adaptation_threshold=0.95,
        patience=8
    )
    
    # --- Phase 1: Generate Expert Dataset ---
    expert_obs, expert_acts, adapted_policy = generate_expert_dataset(
        env, behavioral_policy, num_episodes=PRETRAIN_EPISODES, min_samples=5000
    )
    
    # Save expert data for future use
    expert_data = {
        'observations': expert_obs,
        'actions': expert_acts,
        'policy_state': adapted_policy
    }
    with open('expert_data/behavioral_cloning_data.pkl', 'wb') as f:
        pickle.dump(expert_data, f)
    
    # --- Phase 2: Initialize PPO Model ---
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,  # Adjust for parallel envs
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Start with moderate exploration
        verbose=1,
        tensorboard_log="sb3_logs/PPO_BC/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU
        )
    )
    
    # --- Phase 3: Behavioral Cloning Pre-training ---
    bc_logger = BehavioralCloningLogger()
    pretrain_with_behavioral_cloning(
        model, expert_obs, expert_acts, 
        epochs=PRETRAIN_EPOCHS, 
        batch_size=256,
        validation_split=0.1,
        logger=bc_logger
    )
    
    # --- Phase 4: Reinforcement Learning Fine-tuning ---
    print(f"\n{'='*70}")
    print("STARTING REINFORCEMENT LEARNING FINE-TUNING")
    print(f"{'='*70}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // N_ENVS, 1),
        save_path='sb3_models/',
        name_prefix='ppo_bc'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='sb3_models/best_model/',
        log_path='sb3_logs/',
        eval_freq=max(10000 // N_ENVS, 1),
        deterministic=True,
        render=False
    )
    
    power_monitor = PowerReductionMonitor()
    
    # Fine-tune with RL
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, power_monitor],
        progress_bar=True,
        reset_num_timesteps=True
    )
    
    # --- Phase 5: Save Final Model ---
    model.save("sb3_models/ppo_bc_final")
    env.save("sb3_models/vec_normalize_final.pkl")
    
    print(f"\n{'='*70}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print("   Final model: sb3_models/ppo_bc_final.zip")
    print("   Environment stats: sb3_models/vec_normalize_final.pkl")
    print("   Expert data: expert_data/behavioral_cloning_data.pkl")
    
    # Cleanup
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()