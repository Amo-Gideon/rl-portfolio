import numpy as np
import matplotlib.pyplot as plt
import torch
from network import DuelingDQN, StandardDQN
from agent import DQNAgent
from utils import make_env


def train(network_type="dueling", num_episodes=500, checkpoint_path=None):
    env = make_env("ALE/Pong-v5")
    n_actions = env.action_space.n
    input_shape = (4, 84, 84)
    
    # Choose network
    if network_type == "dueling":
        net = DuelingDQN(input_shape, n_actions)
    else:
        net = StandardDQN(input_shape, n_actions)
    
    agent = DQNAgent(
        network=net,
        n_actions=n_actions,
        lr=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=1e-5,
        batch_size=32,
        target_replace=1000,
        use_double=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    scores = []
    epsilons = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store and learn every step
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            
            score += reward
            state = next_state
        
        scores.append(score)
        epsilons.append(agent.epsilon)
        
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else score
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"[{network_type:8s}] Episode {episode:4d} | "
                  f"Avg Score: {avg_score:7.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
    
    env.close()
    
    # Save checkpoint
    if checkpoint_path:
        agent.save(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    
    return scores, epsilons, agent


def plot_results(duel_scores, std_scores=None, save_path="results/plot.png"):
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Scores
    plt.subplot(1, 2, 1)
    window = 50
    duel_smooth = np.convolve(duel_scores, np.ones(window)/window, mode='valid')
    plt.plot(duel_scores, alpha=0.2, color='red')
    plt.plot(range(window-1, len(duel_scores)), duel_smooth, 
             color='red', linewidth=2, label='Dueling DDQN')
    
    if std_scores is not None:
        std_smooth = np.convolve(std_scores, np.ones(window)/window, mode='valid')
        plt.plot(std_scores, alpha=0.2, color='blue')
        plt.plot(range(window-1, len(std_scores)), std_smooth,
                 color='blue', linewidth=2, label='Standard DDQN')
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Moving average comparison
    plt.subplot(1, 2, 2)
    plt.plot(duel_smooth, color='red', linewidth=2, label='Dueling DDQN')
    if std_scores is not None:
        plt.plot(std_smooth, color='blue', linewidth=2, label='Standard DDQN')
    plt.xlabel('Episode')
    plt.ylabel(f'Moving Avg Score (window={window})')
    plt.title('Smoothed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train Dueling
    print("=" * 60)
    print("Training Dueling DDQN...")
    print("=" * 60)
    duel_scores, duel_eps, duel_agent = train(
        network_type="dueling",
        num_episodes=500,
        checkpoint_path="checkpoints/dueling_pong.pt"
    )
    
    # Train Standard (optional — comment out if you just want one run)
    print("\n" + "=" * 60)
    print("Training Standard DDQN...")
    print("=" * 60)
    std_scores, std_eps, std_agent = train(
        network_type="standard",
        num_episodes=500,
        checkpoint_path="checkpoints/standard_pong.pt"
    )
    
    # Plot
    plot_results(duel_scores, std_scores, save_path="results/comparison.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Dueling DDQN last-50 mean: {np.mean(duel_scores[-50:]):.2f}")
    print(f"Standard DDQN last-50 mean: {np.mean(std_scores[-50:]):.2f}")