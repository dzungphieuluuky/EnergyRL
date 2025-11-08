from fiveg_env import FiveGEnv
from wrapper import GatedRewardWrapper
import numpy as np

def test_fixed_wrapper():
    """Test wrapper with fixes."""
    env_config = {'deploymentScenario': 'dense_urban', 'seed': 42}
    base_env = FiveGEnv(env_config, max_cells=57)
    wrapped_env = GatedRewardWrapper(base_env)
    
    obs, _ = wrapped_env.reset(seed=42)
    
    # Test with 50% power (should work)
    action_50 = np.full(base_env.action_space.shape, 0.5)
    
    for step in range(50):
        obs, reward, terminated, truncated, info = wrapped_env.step(action_50)
        
        print(f"Step {step+1}: reward={reward:.3f}, " +
              f"terminated={terminated}, " +
              f"compliance={info.get('is_compliant', False)}")
        
        if terminated:
            print(f"  Reason: {info.get('termination_reason', 'unknown')}")
            break
    
    print(f"\nFinal compliance rate: {info.get('true_compliance_rate', 0):.1f}%")

test_fixed_wrapper()