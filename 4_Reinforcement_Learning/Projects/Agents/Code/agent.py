import numpy as np
import random
from typing import Tuple, List
import time

class GridWorld:
    def __init__(self, size: int = 5):
        self.size = size
        self.reset()
        
        # Place obstacles
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        # Actions: up, right, down, left
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
    def reset(self) -> int:
        """Reset environment and return initial state"""
        self.agent_pos = (0, 0)
        self.goal = (self.size-1, self.size-1)
        return self._get_state()
        
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take action and return (next_state, reward, done, info)"""
        # Get movement direction
        move = self.actions[action]
        
        # Calculate new position
        new_pos = (
            self.agent_pos[0] + move[0],
            self.agent_pos[1] + move[1]
        )
        
        # Check if valid move
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and
            new_pos not in self.obstacles):
            self.agent_pos = new_pos
            
        # Calculate reward
        if self.agent_pos == self.goal:
            reward = 100
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -50
            done = True
        else:
            reward = -1  # Small penalty for each move
            done = False
            
        return self._get_state(), reward, done, {}
    
    def _get_state(self) -> int:
        """Convert position to state number"""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def render(self):
        """Display the grid"""
        for y in range(self.size):
            for x in range(self.size):
                pos = (x, y)
                if pos == self.agent_pos:
                    print('A', end=' ')
                elif pos == self.goal:
                    print('G', end=' ')
                elif pos in self.obstacles:
                    print('O', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
    
    def learn(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using Q-learning update rule"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state, action] = new_value

def train_agent(env: GridWorld, agent: QLearningAgent, episodes: int = 1000) -> List[float]:
    """Train the agent and return rewards per episode"""
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Learn from the action
            agent.learn(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
    return rewards_history

def demonstrate_agent(env: GridWorld, agent: QLearningAgent):
    """Show a single episode of the trained agent"""
    state = env.reset()
    done = False
    env.render()
    time.sleep(1)
    
    while not done:
        action = agent.choose_action(state)
        state, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.5)

# Run the example
if __name__ == "__main__":
    # Create environment and agent
    env = GridWorld(size=5)
    agent = QLearningAgent(state_size=25, action_size=4)
    
    # Train the agent
    print("Training agent...")
    rewards = train_agent(env, agent, episodes=1000)
    
    # Show final performance
    print("\nFinal trained agent performance:")
    demonstrate_agent(env, agent)
    
    # Print average reward for last 100 episodes
    print(f"Average reward over last 100 episodes: {np.mean(rewards[-100:]):.2f}")