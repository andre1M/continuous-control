from model import Actor, Critic
from utilities import ReplayBuffer, OrnsteinUhlenbeckActionNoise

from torch.nn import functional as F
from torch.optim import Adam
import numpy as np
import torch

import random


# # # CONSTANT VALUES # # #
BUFFER_SIZE     = int(1e6)      # Replay memory buffer size
MINIBATCH_SIZE  = 128           # Number of experience tuples to be sampled for learning
DISCOUNT_FACTOR = 0.99          # Discounting factor for rewards
SOFT_UPDATE     = 1e-3          # Soft update ratio for target network
UPDATE_EVERY    = 2             # Parameters update frequency

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
        self.actor_local = Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        self.actor_target = Actor(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        self.hard_update(self.actor_local, self.actor_target)
        self.actor_optim = Adam(self.actor_local.parameters())

        self.critic_local = Critic(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        self.critic_target = Critic(self.observation_size, self.action_size, seed=seed).to(DEVICE)
        self.hard_update(self.critic_local, self.critic_target)
        self.critic_optim = Adam(self.critic_local.parameters())

        self.noise = OrnsteinUhlenbeckActionNoise(action_size, seed, theta=0.15, sigma=0.2)

        # Initialize replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, MINIBATCH_SIZE, seed)

        self.t_step = 0
        self.eps_t = 0
        self.no_decay_steps = 0
        self.eps = 1
        self.eps_decay = 0.99995
        self.eps_min = 0.01

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

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # Learn, if there is enough samples in memory
            if len(self.memory) > MINIBATCH_SIZE:
                # sample experiences from memory
                experiences = self.memory.sample()
                # learn from sampled experiences
                self.learn(experiences)

    def act(self, state, explore=True):
        """
        Returns actions for given state as per current policy.

        :param state: (array_like) current state;
        :param explore: (Bool) explore or exploit.
        """
        self.eps_t += 1

        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state.unsqueeze(0)).cpu().data.numpy()
        self.actor_local.train()

        # Add noise for exploration
        if explore:
            if self.eps_t > self.no_decay_steps:
                action += self.eps * self.noise()
                self.eps = max(self.eps_min, self.eps * self.eps_decay)
            else:
                action += self.noise()

        return np.clip(action, self.action_low, self.action_high)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples;
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
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Target network soft update
        self.soft_update(self.critic_local, self.critic_target, SOFT_UPDATE)
        self.soft_update(self.actor_local, self.actor_target, SOFT_UPDATE)

    def reset(self):
        self.noise.reset()

    def make_checkpoint(self):
        torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')

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
