import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, state, action, reward, next_state, done):
        # Must store as a tuple, not separate args
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, network, n_actions, lr=0.0001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=1e-5,
                 batch_size=32, target_replace=1000, use_double=True,
                 device='cpu'):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_replace = target_replace
        self.use_double = use_double
        self.learn_step_counter = 0
        
        # Networks
        self.network = network.to(self.device)
        self.target_network = copy.deepcopy(network).to(self.device)
        self.target_network.eval()
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
    
    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        state: numpy array of shape (4, 84, 84)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)
    
    def learn(self):
        # Skip if not enough samples
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q(s,a): gather the Q-value for the taken action
        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target y
        with torch.no_grad():
            if self.use_double:
                # Double DQN: use online network to select action, target to evaluate
                next_actions = self.network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: max over target network
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss: Huber (smooth L1) — more stable than MSE for RL
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Backprop (correct order!)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']