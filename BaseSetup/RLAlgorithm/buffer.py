import numpy as np

class ReplayBuffer():
    """
    A simple experience replay buffer for storing and sampling transitions,
    used in reinforcement learning to stabilize training.

    Attributes:
        mem_size (int): Maximum number of transitions to store.
        mem_cntr (int): Current number of transitions stored (increases until mem_size).
        state_memory (np.ndarray): Stores states.
        new_state_memory (np.ndarray): Stores next states.
        action_memory (np.ndarray): Stores actions.
        reward_memory (np.ndarray): Stores rewards.
        terminal_memory (np.ndarray): Stores done flags (True if episode ended).

    Methods:
        store_transition(state, action, reward, state_, done):
            Stores a single transition into the buffer.

        sample_buffer(batch_size):
            Samples a random mini-batch of transitions for training.
    """
    def __init__(self, max_size, input_shape, n_actions):
        """
        Initializes the ReplayBuffer.

        Parameters:
            max_size (int): Maximum number of transitions to store.
            input_shape (tuple): Shape of state observations.
            n_actions (int): Number of actions.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        """
        Stores a transition in the replay buffer.

        Parameters:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            reward (float): The reward received.
            state_ (np.ndarray): The next state.
            done (bool): True if the episode ended after this step.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Samples a batch of transitions from the buffer.

        Parameters:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batches of states, actions, rewards, next states, and dones.
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones