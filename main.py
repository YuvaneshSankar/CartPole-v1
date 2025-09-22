"""
Actor-Critic Reinforcement Learning for CartPole-v1
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from collections import deque

# Import your custom modules
from agent.actor import Actor
from agent.critic import Critic  # Note: You'll need to fix the Critic class name in your critic.py
from env.make_env import make_env, get_env_info
from utils.logger import RLLogger
from utils.plot import RLPlotter

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        """
        Initialize Actor-Critic Agent
        
        Args:
            state_dim (int): State space dimension
            action_dim (int): Action space dimension  
            hidden_dim (int): Hidden layer size
            actor_lr (float): Actor learning rate
            critic_lr (float): Critic learning rate
            gamma (float): Discount factor
        """
        self.gamma = gamma
        
        # Initialize networks
        self.actor = Actor(state_dim, hidden_dim, action_dim, actor_lr)
        self.critic = Critic(state_dim, hidden_dim, 1, critic_lr)  # Output dim = 1 for value
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
    def select_action(self, state):
        """
        Select action using actor network
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities from actor
        action_probs = self.actor(state)
        
        # Sample action from distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def evaluate_state(self, state):
        """
        Evaluate state value using critic network
        
        Args:
            state: Current state
            
        Returns:
            value: Estimated state value
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        value = self.critic(state)
        return value
    
    def update(self, states, actions, rewards, next_states, dones, log_probs):
        """
        Update both actor and critic networks
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            log_probs: List of action log probabilities
        """
        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        
        # Calculate state values
        state_values = self.critic(states).squeeze()
        next_state_values = self.critic(next_states).squeeze()
        
        # Calculate TD targets
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Calculate TD errors (advantages)
        td_errors = td_targets.detach() - state_values
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(state_values, td_targets.detach())
        
        # Actor loss (Policy Gradient with advantage)
        actor_loss = -(log_probs * td_errors.detach()).mean()
        
        # Update critic
        self.critic.backward(critic_loss)
        
        # Update actor
        self.actor.backward(actor_loss)
        
        return actor_loss.item(), critic_loss.item()

def train_agent(args):
    """Main training loop"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = make_env('CartPole-v1', seed=args.seed)
    env_info = get_env_info(env)
    
    # Initialize agent
    agent = ActorCriticAgent(
        state_dim=env_info['observation_dim'],
        action_dim=env_info['action_dim'],
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma
    )
    
    # Initialize logger
    logger = RLLogger(log_dir="logs", experiment_name=f"actor_critic_{args.seed}")
    
    # Training variables
    running_reward = deque(maxlen=100)
    episode_rewards = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    
    print(f"Training Actor-Critic on {env_info['observation_space']} -> {env_info['action_space']}")
    print(f"Device: {agent.device}")
    print("-" * 50)
    
    for episode in range(args.episodes):
        # Reset episode variables
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        loss_count = 0
        
        # Episode memory
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
        
        while True:
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            log_probs.append(log_prob)
            
            # Update counters
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Update networks every few steps or at episode end
            if len(states) >= args.update_freq or done or truncated:
                if len(states) > 0:
                    actor_loss, critic_loss = agent.update(
                        states, actions, rewards, next_states, dones, log_probs
                    )
                    episode_actor_loss += actor_loss
                    episode_critic_loss += critic_loss
                    loss_count += 1
                
                # Clear episode memory
                states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
            
            if done or truncated:
                break
        
        # Average losses for the episode
        if loss_count > 0:
            episode_actor_loss /= loss_count
            episode_critic_loss /= loss_count
        
        # Update tracking variables
        running_reward.append(episode_reward)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)
        
        # Log episode
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            length=episode_length,
            actor_loss=episode_actor_loss,
            critic_loss=episode_critic_loss
        )
        
        # Check if solved (average reward >= 195 over last 100 episodes)
        if len(running_reward) >= 100 and np.mean(running_reward) >= 195:
            print(f"\nEnvironment solved in {episode + 1} episodes!")
            print(f"Average reward over last 100 episodes: {np.mean(running_reward):.2f}")
            break
        
        # Save model periodically
        if (episode + 1) % args.save_freq == 0:
            save_model(agent, args.model_dir, episode + 1)
    
    # Final logging and plotting
    logger.save_logs()
    logger.print_final_stats()
    
    # Plot training progress
    if args.plot:
        plotter = RLPlotter()
        plotter.plot_training_progress(
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            actor_losses=actor_losses,
            critic_losses=critic_losses,
            save_path=os.path.join("logs", f"training_plot_{args.seed}.png")
        )
    
    # Save final model
    save_model(agent, args.model_dir, "final")
    
    # Evaluate trained agent
    if args.evaluate:
        evaluate_agent(agent, env, args.eval_episodes)
    
    env.close()

def save_model(agent, model_dir, episode):
    """Save agent models"""
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'episode': episode
    }, os.path.join(model_dir, f"agent_episode_{episode}.pth"))

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate trained agent"""
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    eval_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action, _ = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        eval_rewards.append(episode_reward)
        print(f"Eval Episode {episode + 1}: Reward = {episode_reward}")
    
    avg_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/Max Reward: {min(eval_rewards):.2f}/{max(eval_rewards):.2f}")
    
    return eval_rewards

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Actor-Critic for CartPole-v1')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--actor_lr', type=float, default=0.001, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--update_freq', type=int, default=5, help='Update frequency (steps)')
    
    # Logging and saving
    parser.add_argument('--save_freq', type=int, default=500, help='Model save frequency')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='Model save directory')
    parser.add_argument('--plot', action='store_true', help='Plot training progress')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained agent')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    train_agent(args)
