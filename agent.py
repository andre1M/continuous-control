from model import Actor, Critic
from utilities import ReplayBuffer

from torch import nn
from torch.optim import Adam
import torch

import random


# # # CONSTANT VALUES # # #
ACTOR_LR        = 1e-4          # Actor learn rate
CRITIC_LR       = 1e-3          # Critic learn rate
CRITIC_WD       = 1e-2          # Critic weight decay
DISCOUNT_FACTOR = 0.99          # Discounting factor for rewards
SOFT_UPDATE     = 1e-3          # Soft update ratio for target network
UPDATE_EVERY    = 1             # Soft update frequency
W_INIT_LIMIT    = 3e-3          # Network weights and biases initialization range (-val, val)
MINIBATCH_SIZE  = 64            # Number of experience tuples to be sampled for learning
BUFFER_SIZE     = int(1e6)      # Replay memory buffer size

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # # CONSTANT VALUES # # #


class DeepDeterministicPolicyGradient:
    """
    Interacts with and learns from the environment.
    Deep Deterministic Policy Gradient Agent.
    """

    def __init__(self, observation_space_size: int, action_space_size: int, seed: int):
        """
        Initialize an Agent object.

        :param observation_space: dimension of each state;
        :param action_space: dimension of each action;
        :param seed: random seed.
        """

        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.action_low = -1
        self.action_high = 1
        random.seed(seed)

        # Initialize networks and optimizers
        self.actor = Actor(self.observation_space_size, self.action_space_size, 128, 128, 128, W_INIT_LIMIT)
        self.actor_target = Actor(self.observation_space_size, self.action_space_size, 128, 128, 128, W_INIT_LIMIT)
        self.hard_update(self.actor, self.actor_target)
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(self.observation_space_size, self.action_space_size, 128, 128, 128, W_INIT_LIMIT)
        self.critic_target = Critic(self.observation_space_size, self.action_space_size, 128, 128, 128, W_INIT_LIMIT)
        self.hard_update(self.critic, self.critic_target)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR, weight_decay=CRITIC_WD)

        self.loss = nn.MSELoss()

        # Initialize replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, MINIBATCH_SIZE, seed)

        # Initialize time step (for target values soft update every $UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action: int, reward: float, next_state, done):
        """
        Save experiences in the replay memory and check if it's time to learn.

        :param state: (array_like) current state;
        :param action: action taken;
        :param reward: reward received;
        :param next_state: (array_like) next state;
        :param done: terminal state indicator; int or bool.
        """

        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        # Increment time step and compare it to the network update frequency
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Check if there is enough samples in the memory to learn
            if len(self.memory) > MINIBATCH_SIZE:
                # sample experiences from memory
                experiences = self.memory.sample()
                # learn from sampled experiences
                self.learn(experiences, DISCOUNT_FACTOR)

    def act(self, state, random_proc=None):
        """
        Returns actions for given state as per current policy.

        :param state: (array_like) current state;
        :param random_proc: (callable) random process for exploration;
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state)
        self.actor.train()

        # Add noise for exploration
        if random_proc is not None:
            action_values += torch.Tensor(random_proc()).to(DEVICE)

        # Clip action values according to environment limits
        action_values.data = action_values.data.clamp(self.action_low, self.action_high)

        return action_values

    def learn(self, experiences, gamma: float):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples;
        :param gamma: discount factor.
        """

        states, actions, rewards, next_states, dones = experiences

        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions.detach())
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Update critic
        Q_expected = self.critic(states, actions)
        loss = self.loss(Q_expected, Q_targets.detach())
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        # Update Actor
        loss = (-self.critic(states, self.actor(states))).mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        # Target network soft update
        self.soft_update(self.critic, self.critic_target, SOFT_UPDATE)
        self.soft_update(self.actor, self.actor_target, SOFT_UPDATE)

    @staticmethod
    def soft_update(local_model, target_model, tau: float):
        """
        Soft update model parameters:
        θ_target = τ * θ_local + (1 - τ) * θ_target.

        :param local_model: (PyTorch model) weights will be copied from;
        :param target_model: (PyTorch model) weights will be copied to;
        :param tau: interpolation parameter.
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(local_model, target_model):
        """
        Hard update model parameters.

        :param local_model: (PyTorch model) weights will be copied from;
        :param target_model: (PyTorch model) weights will be copied to;
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
