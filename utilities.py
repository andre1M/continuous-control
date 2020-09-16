from matplotlib import pyplot as plt
import numpy as np
import torch

from collections import namedtuple, deque
import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""

        random.seed(seed)
        self.mu = mu * np.ones(size)
        self.state = self.mu.copy()
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()

    def __call__(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer:
    """
    Fixed-size memory buffer to store experience tuples.
    """

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """
        Initialize a ReplayBuffer object.

        :param buffer_size: maximum size of buffer;
        :param batch_size: size of each training batch;
        :param seed: random seed.
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.experience = namedtuple(
            'Experience',
            field_names=('state', 'action', 'reward', 'next_state', 'done')
        )

        # initialize random number generator state
        random.seed(seed)

    def __len__(self):
        """
        Return the current size of internal memory.
        """

        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.

        :param state: state description;
        :param action: action taken in state;
        :param reward: reward received;
        :param next_state: next state;
        :param done: terminal state indicator.
        """

        self.memory.append(
            self.experience(state, action, reward, next_state, done)
        )

    # noinspection PyUnresolvedReferences
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.

        :return: torch tensors of states, action, rewards, next states and terminal state flags.
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(DEVICE)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(DEVICE)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(DEVICE)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(DEVICE)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(DEVICE)

        return states, actions, rewards, next_states, dones


def train(agent, env, n_episodes=2000, max_iter=350):
    """
    Train a Reinforcement Learning agent.

    :param agent: agent object to be trained;
    :param env: environment callable;
    :param n_episodes: maximum number of training episodes;
    :param max_iter: maximum number of time steps per episode;
    :return: scores per episode.
    """

    brain_name = env.brain_names[0]

    scores = []                         # list containing scores from each episode
    scores_window = deque(maxlen=100)   # last 100 scores

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        agent.reset()
        score = 0                                           # reset score for new episode

        i = 0
        while i < max_iter:
            action = agent.act(state)                               # select action
            env_info = env.step(action)[brain_name]                 # get environment response to the action
            next_state = env_info.vector_observations[0]            # get the next state
            reward = env_info.rewards[0]                            # get the reward
            done = env_info.local_done[0]                           # terminal state flag
            agent.step(state, action, reward, next_state, done)     # process experience
            state = next_state
            score += reward
            i += 1
            if done:
                break

        # save recent scores
        scores_window.append(score)
        scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            torch.save(agent.actor_local.state_dict(), 'final_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'final_checkpoint_critic.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            break
    return scores


def plot_scores(scores, filename):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(np.arange(len(scores)), scores)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Episode #', fontweight='bold')
    ax.set_title('Score evolution over training', fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
