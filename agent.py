from model import Actor, Critic
from utilities import ReplayBuffer, OrnsteinUhlenbeckActionNoise

from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
import torch

import random


# # # CONSTANT VALUES # # #
ACTOR_LR        = 1e-4          # Actor learn rate
CRITIC_LR       = 2e-3          # Critic learn rate
CRITIC_WD       = 1e-4          # Critic weight decay
DISCOUNT_FACTOR = 0.99          # Discounting factor for rewards
SOFT_UPDATE     = 1e-3          # Soft update ratio for target network
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

    def __init__(self, observation_size: int, action_size: int, seed: int):
        """
        Initialize an Agent object.

        :param observation_size: dimension of each state;
        :param action_size: dimension of each action;
        :param seed: random seed.
        """

        self.observation_size = observation_size
        self.action_size = action_size
        self.action_low = -1
        self.action_high = 1
        random.seed(seed)

        # Initialize networks and optimizers
        self.actor_local = Actor(self.observation_size, self.action_size, 256, 512, 256, W_INIT_LIMIT).to(DEVICE)
        self.actor_target = Actor(self.observation_size, self.action_size, 256, 512, 256, W_INIT_LIMIT).to(DEVICE)
        self.hard_update(self.actor_local, self.actor_target)
        self.actor_optim = Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        self.critic_local = Critic(self.observation_size, self.action_size, 256, 512, 256, W_INIT_LIMIT)
        self.critic_target = Critic(self.observation_size, self.action_size, 256, 512, 256, W_INIT_LIMIT)
        self.hard_update(self.critic_local, self.critic_target)
        self.critic_optim = Adam(self.critic_local.parameters(), lr=CRITIC_LR, weight_decay=CRITIC_WD)

        self.noise = OrnsteinUhlenbeckActionNoise(action_size, seed)

        # Initialize replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, MINIBATCH_SIZE, seed)

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

        # Learn, if there is enough samples in memory
        if len(self.memory) > MINIBATCH_SIZE:
            # sample experiences from memory
            experiences = self.memory.sample()
            # learn from sampled experiences
            self.learn(experiences, DISCOUNT_FACTOR)

    def act(self, state, explore=True):
        """
        Returns actions for given state as per current policy.

        :param state: (array_like) current state;
        :param explore: (Bool) explore or exploit.
        """

        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # Add noise for exploration
        if explore:
            action += self.noise()

        return np.clip(action, self.action_low, self.action_high)

    def learn(self, experiences, gamma: float):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples;
        :param gamma: discount factor.
        """

        states, actions, rewards, next_states, dones = experiences

        # Update Critic
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (DISCOUNT_FACTOR * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update Actor
        action_predictions = self.actor_local(states)
        actor_loss = -self.critic_local(states, action_predictions).mean()
        actor_loss = (-self.critic_local(states, self.actor_local(states))).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Target network soft update
        self.soft_update(self.critic_local, self.critic_target, SOFT_UPDATE)
        self.soft_update(self.actor_local, self.actor_target, SOFT_UPDATE)

    def reset(self):
        self.noise.reset()

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
