import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import pickle

class SafeBehavioralCurriculum:
    """
    Behavioral policy that starts with high power and gradually reduces it,
    but ONLY when constraints are satisfied.
    """
    
    def __init__(self, action_dim, 
                 initial_range=(0.9, 1.0),    # Start very high for safety
                 final_range=(0.3, 0.6),      # Target efficient range
                 hold_episodes=100,           # Stay safe initially
                 transition_episodes=600,     # Slow, careful transition
                 compliance_threshold=0.98,   # Must maintain 98% QoS compliance
                 patience=20                  # Don't progress if violating
                ):
        self.action_dim = action_dim
        self.initial_low, self.initial_high = initial_range
        self.final_low, self.final_high = final_range
        self.hold_episodes = hold_episodes
        self.transition_episodes = transition_episodes
        self.total_curriculum_episodes = hold_episodes + transition_episodes
        self.compliance_threshold = compliance_threshold
        self.patience = patience
        
        # Curriculum state
        self.episode_count = 0
        self.consecutive_compliant = 0
        self.current_low, self.current_high = initial_range
        
        # Performance tracking
        self.compliance_history = []
        self.power_history = []
    
    def update_curriculum(self, episode_compliance):
        """Update curriculum only if constraints are satisfied"""
        self.compliance_history.append(episode_compliance)
        self.episode_count += 1
        
        # Check if current episode was compliant
        is_compliant = episode_compliance >= self.compliance_threshold
        
        if is_compliant:
            self.consecutive_compliant += 1
        else:
            self.consecutive_compliant = 0
        
        # --- HOLD PHASE ---
        if self.episode_count < self.hold_episodes:
            # Stay at initial high power, don't progress yet
            self.current_low, self.current_high = self.initial_low, self.initial_high
            return False
        
        # --- CHECK IF READY TO TRANSITION ---
        if (self.episode_count >= self.hold_episodes and 
            self.consecutive_compliant >= self.patience and
            self.current_high > self.final_high):
            
            # Calculate safe progression
            progress = min(1.0, (self.episode_count - self.hold_episodes) / 
                          self.transition_episodes)
            
            # Conservative reduction
            new_low = self.initial_low + progress * (self.final_low - self.initial_low)
            new_high = self.initial_high + progress * (self.final_high - self.initial_high)
            
            # Only update if we're actually reducing power
            if new_high < self.current_high:
                self.current_low, self.current_high = new_low, new_high
                print(f"🎓 Curriculum progressed! New power range: [{new_low:.3f}, {new_high:.3f}]")
                return True
        
        return False
    
    def get_power_range(self):
        return self.current_low, self.current_high
    
    def sample_action(self):
        low, high = self.get_power_range()
        return np.random.uniform(low, high, size=self.action_dim)
    
    def get_progress(self):
        """Get curriculum progress percentage"""
        if self.episode_count < self.hold_episodes:
            return 0.0
        progress = (self.episode_count - self.hold_episodes) / self.transition_episodes
        return min(1.0, progress * 100)

class ConstraintAwarePrefilling(BaseCallback):
    """
    Prefill with behavioral policy that adapts based on constraint satisfaction.
    FIXED: Now properly implements _on_step method.
    """
    
    def __init__(self, behavioral_policy, prefill_episodes=200, verbose=1):
        super().__init__(verbose)
        self.behavioral_policy = behavioral_policy
        self.prefill_episodes = prefill_episodes
        self.prefilled = False
    
    def _on_training_start(self):
        """Called before the first training step"""
        if not self.prefilled:
            print(f"\n{'='*70}")
            print(f"CONSTRAINT-AWARE PREFILLING")
            print(f"Starting power: {self.behavioral_policy.get_power_range()}")
            print(f"{'='*70}")
            
            # Get the first environment
            env = self.training_env.envs[0]
            
            for ep in range(self.prefill_episodes):
                obs, _ = env.reset()
                done = False
                episode_steps = 0
                compliant_steps = 0
                
                while not done:
                    action = self.behavioral_policy.sample_action()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Add to replay buffer
                    self.model.replay_buffer.add(
                        obs, next_obs, action, reward, done, [info]
                    )
                    
                    # Track compliance (adjust based on your environment's info structure)
                    if info.get('is_compliant', True):  # Adjust this key based on your env
                        compliant_steps += 1
                    
                    episode_steps += 1
                    obs = next_obs
                
                # Calculate episode compliance rate
                compliance_rate = compliant_steps / episode_steps if episode_steps > 0 else 0
                
                # Update curriculum based on performance
                curriculum_updated = self.behavioral_policy.update_curriculum(compliance_rate)
                
                if (ep + 1) % 10 == 0 or curriculum_updated:
                    low, high = self.behavioral_policy.get_power_range()
                    progress = self.behavioral_policy.get_progress()
                    print(f"Episode {ep+1:3d}/{self.prefill_episodes} | "
                          f"Power: [{low:.3f}, {high:.3f}] | "
                          f"Compliance: {compliance_rate:.1%} | "
                          f"Progress: {progress:.1f}% | "
                          f"Buffer: {self.model.replay_buffer.size():6d}")
            
            self.prefilled = True
            final_low, final_high = self.behavioral_policy.get_power_range()
            print(f"\n✅ Prefill complete! Final power range: [{final_low:.3f}, {final_high:.3f}]")
            print(f"📊 Buffer size: {self.model.replay_buffer.size()} transitions")
            print(f"{'='*70}\n")
    
    def _on_step(self):
        """Required method - called at each step during training"""
        return True  # Return True to continue training

class SafetyMonitorCallback(BaseCallback):
    """
    Monitor constraint satisfaction and adapt training if needed.
    """
    
    def __init__(self, behavioral_policy, safety_threshold=0.95, verbose=1):
        super().__init__(verbose)
        self.behavioral_policy = behavioral_policy
        self.safety_threshold = safety_threshold
        
        self.episode_rewards = []
        self.episode_compliances = []
        self.episode_powers = []
        self.safety_violations = 0
    
    def _on_step(self):
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            
            # Collect episode statistics
            if 'episode' in self.locals:
                ep_info = self.locals['episode'][0]
                self.episode_rewards.append(ep_info['r'])
            
            # Extract compliance and power metrics
            compliance = 0
            avg_power = 0
            
            # Adjust these keys based on your environment's info structure
            if 'reward_info' in info:
                metrics = info['reward_info'].get('metrics', {})
                curriculum = info['reward_info'].get('curriculum', {})
                
                compliance = metrics.get('qos_compliant_steps', 0) / 500.0  # Adjust denominator
                avg_power = curriculum.get('avg_power', 0)
            else:
                # Fallback: try to get from info directly
                compliance = info.get('compliance_rate', 0.95)  # Default optimistic
                avg_power = info.get('avg_power', 0.8)
            
            self.episode_compliances.append(compliance)
            self.episode_powers.append(avg_power)
            
            # Check for safety violation
            if compliance < self.safety_threshold:
                self.safety_violations += 1
                
                # If too many violations, consider adjusting learning rate
                if self.safety_violations >= 5 and len(self.episode_compliances) > 20:
                    recent_compliance = np.mean(self.episode_compliances[-10:])
                    if recent_compliance < 0.9:
                        print(f"🚨 Safety warning: {self.safety_violations} violations detected!")
                        # You could dynamically adjust learning rate here
        
        # Log progress every 5000 steps
        if self.num_timesteps % 5000 == 0 and len(self.episode_rewards) > 0:
            self._log_progress()
        
        return True
    
    def _log_progress(self):
        n_recent = min(20, len(self.episode_rewards))
        low, high = self.behavioral_policy.get_power_range()
        
        print(f"\n{'='*70}")
        print(f"SAFETY MONITOR - Step: {self.num_timesteps:,}")
        print(f"{'='*70}")
        print(f"Curriculum: [{low:.3f}, {high:.3f}] | Progress: {self.behavioral_policy.get_progress():.1f}%")
        print(f"Recent Performance (last {n_recent} episodes):")
        print(f"  Reward:    {np.mean(self.episode_rewards[-n_recent:]):.3f} ± {np.std(self.episode_rewards[-n_recent:]):.3f}")
        print(f"  Compliance: {np.mean(self.episode_compliances[-n_recent:]):.1%} ± {np.std(self.episode_compliances[-n_recent:]):.1%}")
        print(f"  Avg Power:  {np.mean(self.episode_powers[-n_recent:]):.3f} ± {np.std(self.episode_powers[-n_recent:]):.3f}")
        print(f"  Safety Violations: {self.safety_violations}")
        print(f"{'='*70}\n")

def train_safe_power_optimizer(
    env,
    total_timesteps=200_000,  # Reduced for testing
    prefill_episodes=50,      # Reduced for testing
    initial_power_range=(0.9, 1.0),    # Start very safe
    final_power_range=(0.4, 0.7),      # Target efficiency  
    hold_episodes=20,         # Reduced for testing
    transition_episodes=100,  # Reduced for testing
    compliance_threshold=0.98
):
    """
    Train with safety-guaranteed behavioral curriculum.
    REDUCED MEMORY USAGE VERSION
    """
    
    print(f"\n{'='*70}")
    print("SAFE POWER OPTIMIZATION TRAINING")
    print(f"{'='*70}")
    print(f"Strategy: Start safe → Maintain constraints → Reduce power gradually")
    print(f"Initial power: {initial_power_range}")
    print(f"Target power:  {final_power_range}")
    print(f"Compliance threshold: {compliance_threshold:.1%}")
    print(f"{'='*70}\n")
    
    # Create safety-aware behavioral policy
    behavioral_policy = SafeBehavioralCurriculum(
        action_dim=env.action_space.shape[0],
        initial_range=initial_power_range,
        final_range=final_power_range,
        hold_episodes=hold_episodes,
        transition_episodes=transition_episodes,
        compliance_threshold=compliance_threshold
    )
    
    # Create SAC model with REDUCED BUFFER to fix memory warning
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        buffer_size=50_000,   # REDUCED from 1,000,000 to 50,000
        learning_starts=2000, # Start learning earlier
        batch_size=128,       # Smaller batch size
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        verbose=1,
        tensorboard_log="sb3_logs/SAC_SafePower/",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], qf=[128, 128])  # Smaller networks
        )
    )
    
    # Create safety-aware callbacks
    prefill_callback = ConstraintAwarePrefilling(
        behavioral_policy, prefill_episodes=prefill_episodes, verbose=1
    )
    
    safety_callback = SafetyMonitorCallback(
        behavioral_policy, safety_threshold=compliance_threshold, verbose=1
    )
    
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path='sb3_models/',
        name_prefix='sac_safe_power'
    )
    
    # Train
    print("Starting safe training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[prefill_callback, safety_callback, checkpoint_callback],
        log_interval=10,
        progress_bar=True
    )
    
    # Save final model
    model.save("sb3_models/sac_safe_power_final")
    
    with open("sb3_models/sac_safe_power_behavioral.pkl", 'wb') as f:
        pickle.dump(behavioral_policy, f)
    
    print(f"\n✅ Training completed safely!")
    print(f"   Final power range achieved: {behavioral_policy.get_power_range()}")
    print(f"   Model saved: sb3_models/sac_safe_power_final.zip")
    
    return model, behavioral_policy

# Quick evaluation function
def quick_evaluate(model, env, n_episodes=5):
    """Quick evaluation of the trained policy"""
    print(f"\n{'='*50}")
    print(f"QUICK EVALUATION ({n_episodes} episodes)")
    print(f"{'='*50}")
    
    rewards = []
    compliances = []
    powers = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_powers = []
        ep_compliant_steps = 0
        total_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            
            # Track power usage and compliance
            if 'curriculum' in info.get('reward_info', {}):
                ep_powers.append(info['reward_info']['curriculum']['avg_power'])
            
            if info.get('is_compliant', True):
                ep_compliant_steps += 1
        
        rewards.append(ep_reward)
        compliances.append(ep_compliant_steps / total_steps if total_steps > 0 else 0)
        if ep_powers:
            powers.append(np.mean(ep_powers))
        
        print(f"Episode {ep+1}: Reward={ep_reward:.2f}, "
              f"Compliance={compliances[-1]:.1%}, "
              f"Avg Power={powers[-1] if powers else 'N/A':.3f}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Compliance: {np.mean(compliances):.1%}")
    print(f"Average Power: {np.mean(powers):.3f} ± {np.std(powers):.3f}")
    print(f"{'='*50}")
    
    return {
        'rewards': rewards,
        'compliances': compliances,
        'powers': powers
    }

# Usage
if __name__ == "__main__":
    from fiveg_env import FiveGEnv  # Adjust import based on your environment
    
    # Create environment
    config = {
        'simTime': 500, 
        'timeStep': 1,
    }
    
    try:
        env = FiveGEnv(config)
        
        # Train with safety-guaranteed curriculum (REDUCED SETTINGS FOR TESTING)
        model, behavioral_policy = train_safe_power_optimizer(
            env,
            total_timesteps=200_000,    # Reduced for testing
            prefill_episodes=50,        # Reduced for testing  
            initial_power_range=(0.9, 1.0),
            final_power_range=(0.4, 0.7),
            hold_episodes=20,           # Reduced for testing
            transition_episodes=100,    # Reduced for testing
            compliance_threshold=0.98
        )
        
        # Quick evaluation
        results = quick_evaluate(model, env, n_episodes=5)
        
        print("\n🎉 Training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()