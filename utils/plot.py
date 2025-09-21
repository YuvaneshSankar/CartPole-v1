import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import seaborn as sns

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

class RLPlotter:
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the RL Plotter.
        
        Args:
            figsize (tuple): Figure size for plots
        """
        self.figsize = figsize
        plt.ion()  # Turn on interactive mode
    
    def plot_training_progress(self, episode_rewards, episode_lengths=None, 
                             actor_losses=None, critic_losses=None, 
                             window_size=100, save_path=None):
        """
        Plot comprehensive training progress.
        
        Args:
            episode_rewards (list): List of episode rewards
            episode_lengths (list): List of episode lengths
            actor_losses (list): List of actor losses
            critic_losses (list): List of critic losses
            window_size (int): Window size for moving average
            save_path (str): Path to save the plot
        """
        # Calculate number of subplots needed
        n_plots = 1  # Always have rewards
        if episode_lengths is not None:
            n_plots += 1
        if actor_losses is not None and critic_losses is not None:
            n_plots += 1
        
        fig, axes = plt.subplots(n_plots, 1, figsize=self.figsize)
        if n_plots == 1:
            axes = [axes]
        
        # Plot rewards
        self._plot_rewards(axes[0], episode_rewards, window_size)
        
        plot_idx = 1
        
        # Plot episode lengths
        if episode_lengths is not None:
            self._plot_episode_lengths(axes[plot_idx], episode_lengths, window_size)
            plot_idx += 1
        
        # Plot losses
        if actor_losses is not None and critic_losses is not None:
            self._plot_losses(axes[plot_idx], actor_losses, critic_losses, window_size)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_rewards(self, ax, rewards, window_size):
        """Plot episode rewards with moving average."""
        episodes = np.arange(len(rewards))
        
        # Plot raw rewards (faded)
        ax.plot(episodes, rewards, alpha=0.3, color='lightblue', label='Episode Reward')
        
        # Plot moving average
        if len(rewards) >= window_size:
            moving_avg = uniform_filter1d(rewards, size=window_size, mode='nearest')
            ax.plot(episodes, moving_avg, color='darkblue', linewidth=2, 
                   label=f'Moving Average ({window_size})')
        
        # Add solved line for CartPole
        ax.axhline(y=195, color='red', linestyle='--', alpha=0.7, label='Solved Threshold')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Progress: Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_episode_lengths(self, ax, lengths, window_size):
        """Plot episode lengths with moving average."""
        episodes = np.arange(len(lengths))
        
        # Plot raw lengths (faded)
        ax.plot(episodes, lengths, alpha=0.3, color='lightgreen', label='Episode Length')
        
        # Plot moving average
        if len(lengths) >= window_size:
            moving_avg = uniform_filter1d(lengths, size=window_size, mode='nearest')
            ax.plot(episodes, moving_avg, color='darkgreen', linewidth=2,
                   label=f'Moving Average ({window_size})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Training Progress: Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_losses(self, ax, actor_losses, critic_losses, window_size):
        """Plot actor and critic losses."""
        episodes = np.arange(len(actor_losses))
        
        # Plot actor losses
        if len(actor_losses) >= window_size:
            actor_smooth = uniform_filter1d(actor_losses, size=min(window_size, 50), mode='nearest')
            ax.plot(episodes, actor_smooth, color='orange', linewidth=2, label='Actor Loss')
        
        # Plot critic losses
        if len(critic_losses) >= window_size:
            critic_smooth = uniform_filter1d(critic_losses, size=min(window_size, 50), mode='nearest')
            ax.plot(episodes, critic_smooth, color='purple', linewidth=2, label='Critic Loss')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress: Network Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for losses
    
    def plot_final_evaluation(self, eval_rewards, save_path=None):
        """Plot evaluation results after training."""
        plt.figure(figsize=(10, 6))
        
        episodes = np.arange(len(eval_rewards))
        plt.plot(episodes, eval_rewards, 'o-', color='green', linewidth=2, markersize=8)
        plt.axhline(y=np.mean(eval_rewards), color='red', linestyle='--', 
                   label=f'Average: {np.mean(eval_rewards):.1f}')
        plt.axhline(y=195, color='orange', linestyle='--', alpha=0.7, label='Solved Threshold')
        
        plt.xlabel('Evaluation Episode')
        plt.ylabel('Total Reward')
        plt.title('Final Policy Evaluation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plot saved to: {save_path}")
        
        plt.show()

# Utility function to load and plot from saved logs
def plot_from_logs(log_file):
    """Plot training results from saved log file."""
    import json
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    plotter = RLPlotter()
    plotter.plot_training_progress(
        episode_rewards=data['episode_rewards'],
        episode_lengths=data.get('episode_lengths'),
        actor_losses=data.get('actor_losses'),
        critic_losses=data.get('critic_losses'),
        save_path=log_file.replace('.json', '_plot.png')
    )
