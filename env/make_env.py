import gymnasium as gym
import numpy as np

def make_env(env_id='CartPole-v1', seed=None, render_mode=None):
    """
    Create and configure a Gymnasium environment.
    
    Args:
        env_id (str): Environment ID (default: 'CartPole-v1')
        seed (int): Random seed for reproducibility
        render_mode (str): Render mode ('human', 'rgb_array', None)
    
    Returns:
        env: Configured Gymnasium environment
    """
    # Create the environment
    env = gym.make(env_id, render_mode=render_mode)
    
    # Set seed for reproducibility
    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)
    
    return env

def get_env_info(env):
    """
    Get basic information about the environment.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        dict: Environment information
    """
    info = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'observation_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    }
    return info

# # Example usage for testing
# if __name__ == "__main__":
#     # Create environment
#     env = make_env('CartPole-v1', seed=42)
    
#     # Get environment info
#     env_info = get_env_info(env)
#     print("Environment Info:")
#     print(f"Observation space: {env_info['observation_space']}")
#     print(f"Action space: {env_info['action_space']}")
#     print(f"Observation dim: {env_info['observation_dim']}")
#     print(f"Action dim: {env_info['action_dim']}")
    
#     # Test basic functionality
#     obs, info = env.reset()
#     print(f"Initial observation: {obs}")
    
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"After action {action}: obs={obs}, reward={reward}, done={done}")
    
#     env.close()
