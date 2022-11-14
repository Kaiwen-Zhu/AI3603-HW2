# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=30000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=0.8,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.3,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """comments: 
    This is the definition of the Q-Network. The architecture is as follows:
    the input is an array of shape (8,); the first hidden layer is fully connected 
    with input of size 8 and output of size 120; the second layer is a ReLU layer;
    the third layer is fully connected with input of size 120 and output of size 84;
    the fourth layer is a ReLU layer; the final output layer is fully connected with
    input of size 84 and output of size 4.
    The network is used to compute the Q-value of all actions in the given state.
    That is, if a state `s` (an array of size 8) is fed, then the network returns
    Q(s,a) for each action `a`."""
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """comments: 
    This function computes the epsilon value of current time `t` 
    according to the epsilon decay schema. The value decreases linearly
    from `start_e` to `end_e` when `t` grows from 0 to `duration`.
    When `t` is greater than `duration`, `end_e` is returned."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    """we utilize tensorboard to log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """comments: 
    Set random seed to make the result reproducible.
    Specify the device on which computation is done."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """comments: 
    Initialize the environment and set its random seed."""
    envs = make_env(args.env_id, args.seed)

    """comments: 
    Build two Q-Networks. One network is for obtaining new experiences (`q_network`),
    and the other is for evaluating `q_network`, i.e., serves as the target (`target_network`).
    `target_network` is updated as `q_network` periodically. This solves the problem
    that policy may oscillate drastically because the target is fixed in a period."""
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """comments: 
    Initialize the replay buffer, which is used to store experiences.
    When training, samples are randomly picked from replay buffer.
    This solves the problem that successive samples are correlated.
    Also, since abundant previous experiences can be utilized,
    the training will be faster."""
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """comments:
    Reset the environment, get the this initial state and begin training
    for `args.total_timesteps` steps."""
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        
        """comments: 
        Compute the epsilon value of current step `global_step`, cf. the function
        `linear_schedule`. In this function, the parameters `start_e` and `end_e` are
        given hyper-parameters; the parameter `duration` is set as a fraction of
        total steps, that is, epsilon reaches its destination when a certain fraction
        of total steps elapsed."""
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """comments: 
        The agent chooses action, exploring with probability of epsilon or
        exploiting otherwise. 'Exploring' means taking actions randomly to
        obtain more info of the environment, while 'exploiting' means utilizing
        the already learned Q-values for convergence."""
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """comments: 
        The agent takes the chosen action. Then the environment returns
        the new state (`next_obs`), the immediate rewards of this step (`rewards`),
        whether this episode terminates (`dones`) and some other information (`infos`,
        here `infos` is non-empty only if this episode terminates and the major information
        is the reward and #steps of this episode)."""
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training
        
        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        """comments: 
        Add this new transition to the replay buffer, containing the previous state,
        the action taken, the immediate reward and the new state transitioned to."""
        rb.add(obs, next_obs, actions, rewards, dones, infos)
        
        """comments: 
        If this episode terminates, then reset the environment and get the initial state,
        otherwise update the state as the new state transitioned to."""
        obs = next_obs if not dones else envs.reset()
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """comments: 
            Randomly sample a batch of size `batch_size` from the replay buffer for training."""
            data = rb.sample(args.batch_size)
            
            """comments: 
            Calculate the loss on this batch for training. The loss is set as the mean square error
            of the reward calculated by `q_network` w.r.t. `target_network`. Hence our goal is
            to minimize the gap between `q_network` and `target_network`.
            `with torch.no_grad()` is used because we only want the value and
            do not want to update the parameters of `target_network`."""
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """comments: 
            In certain steps, log the loss and Q-values using tensorboard."""
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """comments: 
            Update the parameters of `q_network` by gradient descent.
            `optimizer.zero_grad()` makes the gradient w.r.t parameters
            accumulated on the last batch return to zero.
            `loss.backward()` calculates the gradient on this batch. 
            `optimizer.step()` updates parameters using the calculated gradient."""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """comments: 
            In certain steps, copy the parameters of `q_network` to
            those of `target_network`."""
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())


    # Record the landing process after training
    envs = gym.wrappers.RecordVideo(envs, './video')
    obs = envs.reset()
    while 1:
        global_step += 1
        q_values = q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=0).cpu().numpy()
        next_obs, rewards, dones, infos = envs.step(actions)
        envs.render() # close render during training
        
        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            break

        rb.add(obs, next_obs, actions, rewards, dones, infos)
        obs = next_obs
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    
    """close the env and tensorboard logger"""
    envs.close()
    writer.close()