# inference.py
import numpy as np
import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Import your custom environment
from fiveg_env import FiveGEnv

def load_trained_model_and_env(model_path: str, stats_path: str, inference_config: dict):
    """
    Loads a trained PPO model and the corresponding VecNormalize statistics.

    Args:
        model_path: Path to the trained model's .zip file.
        stats_path: Path to the VecNormalize .pkl file.
        inference_config: The configuration for the FiveGEnv.

    Returns:
        A tuple of (trained_model, normalized_environment).
    """
    print("--- Loading Model and Environment for Inference ---")
    
    # 1. Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Normalization stats file not found at: {stats_path}")

    # 2. Create a base, non-normalized environment
    # We use DummyVecEnv to wrap the single environment so it's compatible with VecNormalize
    base_env = DummyVecEnv([lambda: FiveGEnv(inference_config)])
    
    # 3. Load the VecNormalize statistics and wrap the environment
    # This is the most critical step. It loads the observation mean and std.
    # CRITICAL: `training=False` freezes the stats, `norm_reward=False` is for inference.
    env = VecNormalize.load(stats_path, base_env)
    env.training = False
    env.norm_reward = False
    
    print(f"Successfully loaded observation normalization stats from {stats_path}")
    
    # 4. Load the PPO model
    # SB3 will automatically reconstruct the policy network architecture from the saved file.
    # No need to specify policy_kwargs if it was a standard "MlpPolicy".
    model = PPO.load(model_path, env=env)
    
    print(f"Successfully loaded PPO model from {model_path}")
    print("--------------------------------------------------\n")
    
    return model, env

def run_inference_episode(model, env, episode_num: int):
    """
    Runs a single inference episode and logs the results.
    """
    print(f"--- Running Inference Episode #{episode_num} ---")
    
    # Reset the environment. The observation is automatically normalized by the VecNormalize wrapper.
    obs = env.reset()

    unnormalized_obs = env.get_original_obs()
    print("--- PYTHON UN-NORMALIZED OBS (STEP 1) ---")
    print(unnormalized_obs)
    print("------------------------------------------")


    done = False
    
    total_reward = 0
    step_count = 0
    compliant_steps = 0
    
    while not done:
        # Get the agent's action
        # CRITICAL: `deterministic=True` makes the agent choose the best action, not explore.
        action, _states = model.predict(obs, deterministic=True)
        
        # Perform the action in the environment
        obs, reward, done, info = env.step(action)
        
        # Log step information
        step_count += 1
        total_reward += reward[0] # reward is a numpy array in a VecEnv
        
        # Safely extract compliance info from the nested dictionary
        is_compliant = info[0].get('reward_info', {}).get('constraints_satisfied', False)
        if is_compliant:
            compliant_steps += 1
        
        status_icon = "✅" if is_compliant else "❌"
        
        print(f"  Step {step_count:3d}: Action Taken (Power Ratio)={action[0][0]:.3f}, "
              f"Step Reward={reward[0]:.3f}, Compliant={status_icon}")

    compliance_rate = (compliant_steps / step_count) * 100 if step_count > 0 else 0
    
    print("\n--- Episode Summary ---")
    print(f"  Total Steps: {step_count}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Compliance Rate: {compliance_rate:.1f}% ({compliant_steps}/{step_count} steps)")
    print("-----------------------\n")


# =================================================================
# MAIN INFERENCE EXECUTION
# =================================================================

if __name__ == "__main__":
    
    # --- 1. Configuration ---
    # Define paths to your trained model and normalization stats
    MODEL_PATH = "sb3_models/ppo_bc_final.zip"
    STATS_PATH = "sb3_models/vec_normalize_final.pkl"
    
    # This configuration should match the one used during training for consistent results
    INFERENCE_CONFIG = {
        'simTime': 500, 
        'timeStep': 1,
        'numSites': 4,
        'numUEs': 100
    }
    
    NUM_EPISODES_TO_RUN = 3
    
    try:
        # --- 2. Load the Model and Environment ---
        ppo_model, normalized_env = load_trained_model_and_env(
            model_path=MODEL_PATH,
            stats_path=STATS_PATH,
            inference_config=INFERENCE_CONFIG
        )
        
        # --- 3. Run Inference ---
        for i in range(1, NUM_EPISODES_TO_RUN + 1):
            run_inference_episode(ppo_model, normalized_env, episode_num=i)
            
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("Please ensure your model and stats files are in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- 4. Clean up ---
        if 'normalized_env' in locals():
            normalized_env.close()
        print("Inference script finished.")