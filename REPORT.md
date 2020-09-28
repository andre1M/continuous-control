[//]: # (Image References)

[image1]: figures/graph.png
[image2]: figures/algorithm.png
[image3]: figures/score_ddpg.png

# Continuous control with Deep Deterministic Policy Gradient (DDPG)

In this project I solved a control task in continuous action space. The complexity of the problem restricts the direct utilization of Deep Q-Network to learn optimal policy because the number of all possible actions (action space) is not a finite number. In order to predict the best action in any state in this environment there are roughly two options: discretize action space or use policy-based reinforcement learning algorithms. Discretization may work good enough for some tasks, however this approach becomes computationally expensive at exponential scale with as the number of grid blocks increases. Policy-based algorithms solve many problems in  RL: capable of learning true stochastic policy, can handle similar states pretty well, can be applied in continuous tasks. In this project I implemented Deep Deterministic Policy Gradient algorithm presented in [this work](https://arxiv.org/abs/1509.02971) with several modifications to speed up learning for this particular environment.

## Algorithm

DDPG algorithm from [original work](https://arxiv.org/abs/1509.02971):

![DDPG algorithm][image2]

In short, the idea of this algorithm is to choose an action based on Actor network (Policy-based) and the evaluate the value of this action in a particular state using Critic (Value-based) to train Actor. Critic is trained based on one-step bootstrapping (TD difference) according to target Actor and Critic. Therefore, this approach handles continuous action space by using Actor to choose an action and Critic to evaluate this action. Target networks are updated every learning step using soft update. 

## Implementation

And here is the computational graph of my particular implementation of this algorithm.

![DDPG computational graph][image1]

### Hyperparameters

A big part of this project was tuning the hyperparameters. Values were chosen according to [original work](https://arxiv.org/abs/1509.02971) with further adjustments based on observed performance. Used values are given in the table below:

| Hyperparameter            	| Value   	|
|---------------------------	|---------	|
| Replay buffer size        	| 1e6     	|
| Minibatch size            	| 128     	|
| Discount factor           	| 0.99    	|
| Actor Learning Rate       	| 1e-3    	|
| Critic Learning Rate      	| 1e-3    	|
| Epsilon initial value     	| 1       	|
| Epsilon minimal value     	| 0.01    	|
| Epsilon exponential decay 	| 0.99995 	|
| Soft update frequency     	| 2       	|
| Soft update coefficient   	| 1e-3    	|

Ornstein - Uhlenbeck Random Process for exploration is initiated with &theta; = 0.15 amd &sigma; = 0.2. 

## Results

Agent score evolution during training is given in the figure below. See [REEDME](README.md) for trained agent visualization.

![Score][image3]

## Ideas for future work

There are several improvements that can be done to this project:
- running several agents in parallel to collect samples should speedup learning;
- reward normalization can improve stability of learning process due to reduced noise in the rewards;
- apparently, DDPG is not the best algorithm for this task; more suitable approaches may work better, such as D4PG, TNPO or TRPO;
- use implemented algorithm for different environment, such as [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) to analyse stability and sample efficiency of the solution, as well as generalization capabilities.