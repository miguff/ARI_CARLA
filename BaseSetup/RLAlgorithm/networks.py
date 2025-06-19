import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    """
    Neural network model for the Critic in Deep Deterministic Policy Gradient (DDPG).

    The Critic estimates the Q-value of a given state-action pair (Q(s, a)).

    Attributes:
        beat: Learning rate
        input_dims (tuple): Dimensions of the state input.
        fc1_dims (int): Number of units in the first fully connected layer.
        fc2_dims (int): Number of units in the second fully connected layer.
        n_actions (int): Number of action dimensions.
        name (str): Name of the network (used for saving/loading checkpoints).
        checkpoint_dir (str): Directory to store checkpoints.
        checkpoint_file (str): Full path to the model checkpoint file.
        optimizer (torch.optim.Optimizer): Optimizer for training the network.
        device (torch.device): Device on which the model is running (CPU/GPU).

    Methods:
        forward(state, action): Forward pass through the network.
        save_checkpoint(): Saves the current network weights.
        load_checkpoint(): Loads saved weights into the network.
        save_best(): Saves a special "best" checkpoint.
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')



        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        
        self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state, action):
        """
        Performs a forward pass through the Critic network.

        Parameters:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The action tensor.

        Returns:
            torch.Tensor: The predicted Q-value for the state-action pair.
        """
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        #This creates a joint embedding that combines what the state looks like and what action you're evaluating in that state.
        state_action_value = self.q(state_action_value)
        #the predicted Q-value, i.e., how good the action is in that state.

        return state_action_value

    def save_checkpoint(self):
        """
        Saves the current model weights to disk.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads model weights from disk.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        """
        Saves the current model as the best version.
        """
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)


class ActorNetwork(nn.Module):
    """
    Actor Network for Deep Deterministic Policy Gradient (DDPG) Agent.
    
    This neural network models a deterministic policy that maps states to 
    continuous actions. It uses fully connected layers with Layer Normalization 
    and ReLU activations, followed by a tanh activation to ensure output actions 
    are bounded within [-1, 1].

    Parameters:
    -----------
    alpha : float
        Learning rate for the Adam optimizer.
    input_dims : tuple
        Dimensions of the input state space.
    fc1_dims : int
        Number of units in the first fully connected layer.
    fc2_dims : int
        Number of units in the second fully connected layer.
    n_actions : int
        Dimension of the action space (number of action outputs).
    name : str
        Name identifier for saving/loading model checkpoints.
    chkpt_dir : str, optional (default='tmp/ddpg')
        Directory path for saving model checkpoints.

    Attributes:
    -----------
    fc1 : nn.Linear
        First fully connected layer.
    fc2 : nn.Linear
        Second fully connected layer.
    bn1 : nn.LayerNorm
        Layer normalization applied after fc1.
    bn2 : nn.LayerNorm
        Layer normalization applied after fc2.
    mu : nn.Linear
        Final linear layer producing raw action outputs.
    optimizer : torch.optim.Adam
        Adam optimizer for training this network.
    device : torch.device
        Device (CPU/GPU) on which model and tensors are allocated.
    checkpoint_file : str
        Full path to the checkpoint file for this model.

    Methods:
    --------
    forward(state: torch.Tensor) -> torch.Tensor:
        Performs the forward pass through the network.
        Input:
            state : Tensor of shape (batch_size, input_dims)
        Output:
            Tensor of shape (batch_size, n_actions) with actions bounded in [-1, 1].

    save_checkpoint() -> None:
        Saves the current model parameters to the checkpoint file.

    load_checkpoint() -> None:
        Loads model parameters from the checkpoint file.

    save_best() -> None:
        Saves the current model parameters as the 'best' checkpoint.
    """
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')


        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)


        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')

        self.to(self.device)


    def forward(self, state):
        """
        Forward pass of the Actor Network.

        Parameters:
        -----------
        state : torch.Tensor
            Input state tensor of shape (batch_size, input_dims).

        Returns:
        --------
        torch.Tensor
            Action tensor of shape (batch_size, n_actions), bounded by [-1, 1]
            via the tanh activation function.
        """
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        #x = T.tanh(self.mu(x))
        x = T.sigmoid(self.mu(x))

        return x

    def save_checkpoint(self):
        """
        Saves the model parameters to the checkpoint file.

        Prints a message indicating the checkpoint is saved.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Loads the model parameters from the checkpoint file.

        Prints a message indicating the checkpoint is loaded.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        """
        Saves the model parameters as the best checkpoint.

        This is useful for keeping track of the best performing model during training.
        """
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)