import os
import json
import time
import numpy as np
from collections import deque
from datetime import datetime

class RLLogger:
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        Initialize the RL Logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"actor_critic_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.running_rewards = deque(maxlen=100)  # For running average
        
        # Training metrics
        self.start_time = time.time()
        self.episode_count = 0
        
        print(f"Logger initialized. Logs will be saved to: {self.log_file}")
    
    def log_episode(self, episode, reward, length, actor_loss=None, critic_loss=None):
        """
        Log metrics for a completed episode.
        
        Args:
            episode (int): Episode number
            reward (float): Total episode reward
            length (int): Episode length (steps)
            actor_loss (float): Actor loss for this episode
            critic_loss (float): Critic loss for this episode
        """
        self.episode_count = episode
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.running_rewards.append(reward)
        
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        
        # Calculate running average
        running_avg = np.mean(self.running_rewards)
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            elapsed_time = time.time() - self.start_time
            print(f"Episode {episode:4d} | "
                  f"Reward: {reward:7.2f} | "
                  f"Avg(100): {running_avg:7.2f} | "
                  f"Length: {length:3d} | "
                  f"Time: {elapsed_time/60:.1f}m")
    
    def get_stats(self):
        """Get current training statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'running_avg_reward': np.mean(self.running_rewards),
            'solved': np.mean(self.running_rewards) >= 195  # CartPole threshold
        }
    
    def save_logs(self):
        """Save all logged data to JSON file."""
        log_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'episodes': self.episode_count,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'stats': self.get_stats(),
            'training_time': time.time() - self.start_time
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Logs saved to: {self.log_file}")
    
    def print_final_stats(self):
        """Print final training statistics."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Average Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Max Reward: {stats['max_reward']:.2f}")
        print(f"Running Average (last 100): {stats['running_avg_reward']:.2f}")
        print(f"Average Episode Length: {stats['avg_length']:.1f}")
        print(f"Environment Solved: {'Yes' if stats['solved'] else 'No'}")
        print(f"Training Time: {(time.time() - self.start_time)/60:.1f} minutes")
        print("="*50)
