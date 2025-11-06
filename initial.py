from stable_baselines3.common.base_class import BaseAlgorithm
import torch
import torch.nn as nn

def bias_policy_output(model: BaseAlgorithm, bias_value: float, weight_std: float = 0.01):
    """
    Initializes the output layer of an SB3 policy to a specific bias.
    This encourages the agent to start with a known "safe" action.

    Args:
        model: The SB3 model (PPO or SAC).
        bias_value: The target logit value (pre-tanh). A value of ~0.7 targets an action of 0.8.
        weight_std: The standard deviation for the final layer's weights. Should be small.
    """
    final_layer = None
    
    # --- PPO Case ---
    # For PPO's MlpPolicy, the final layer mapping features to actions is `action_net`.
    # This is a single nn.Linear layer, NOT a Sequential module.
    if hasattr(model.policy, "action_net") and isinstance(model.policy.action_net, nn.Linear):
        print("Biasing the final layer of PPO's action_net.")
        final_layer = model.policy.action_net

    # --- SAC Case ---
    # For SAC's MlpPolicy, the actor network ends with a `mu` network,
    # which IS a Sequential module. The final layer is the last element.
    elif hasattr(model.policy, "actor") and hasattr(model.policy.actor, "mu") and isinstance(model.policy.actor.mu, nn.Sequential):
        print("Biasing the final layer of SAC's actor (mu network).")
        final_layer = model.policy.actor.mu[-1]

    if final_layer is None or not isinstance(final_layer, nn.Linear):
        print("\nWARNING: Could not find a recognizable final nn.Linear policy layer to bias.")
        print("The agent will start with random actions. This may not be an error if using a custom policy.\n")
        return

    print(f"  - Original bias shape: {final_layer.bias.shape}")
    print(f"  - Setting final layer bias to {bias_value:.4f}")
    print(f"  - Setting final layer weights with std {weight_std:.4f}")

    # Initialize weights to be small to let the bias dominate the initial output
    torch.nn.init.normal_(final_layer.weight, mean=0.0, std=weight_std)
    
    # Initialize the bias to our target logit value
    torch.nn.init.constant_(final_layer.bias, bias_value)
