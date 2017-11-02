#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""

import sys
sys.path.insert(0, "/home/mep/Repos/opensim")
print(sys.path)
import opensim as osim
import numpy as np

from osim.env import *
from osim.http.client import Client
#from keras.optimizers import RMSprop
import math

from policy import Policy
from value_function import NNValueFunction
from checkpoint import Checkpoint
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import itertools

import mpi_util
import pickle

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_osim(animate=False):
    env = RunEnv(animate)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim

STATE_PELVIS_X = 1
STATE_PELVIS_Y = 2
MUSCLES_PSOAS_R = 3
MUSCLES_PSOAS_L = 11

STATE_HEAD_X = 22
STATE_HEAD_Y = 23
STATE_PELVIS_ALT_X = 24
STATE_PELVIS_ALT_Y = 25
STATE_TORSO_X = 26
STATE_TORSO_Y = 27
STATE_TOES_L_X = 28
STATE_TOES_L_Y = 29
STATE_TOES_R_X = 30
STATE_TOES_R_Y = 31
STATE_TALUS_L_X = 32
STATE_TALUS_L_Y = 33
STATE_TALUS_R_X = 34
STATE_TALUS_R_Y = 35
BLANK = -10.0

trace = {'head': [],
              'head_targ': [],
              'talus_l': [],
              'talus_l_targ': [],
              'talus_r': [],
              'talus_r_targ': [],
              'pelvis': [] }

def targs(x):
    x0=0.0
    # how quickly to accelerate stepping
    k2=2.0
    x2=x * (2/(1.0+np.exp(-k2*(x-x0)))-1.0)

    # how quickly to bend head
    k3=5.0
    x3=2/(1.0+np.exp(-k3*(x-x0)))-1.0

    head_targ = np.array([0.5*x3, BLANK])
    talus_l_targ = np.array([0.4*np.sin(np.pi*x2+np.pi),       BLANK])
    talus_r_targ = np.array([0.4*np.sin(np.pi*x2), BLANK])

    return (head_targ, talus_l_targ, talus_r_targ)

def err(x):
    return np.sqrt((x**2).sum())

def replace_none(targ, act):
    for i in range(2):
        if targ[i] == BLANK:
            targ[i] = act[i]
    
def special_reward(obs, reward, step, animate):
    error = 0.0
    
    head_abs = np.array([obs[STATE_HEAD_X], obs[STATE_HEAD_Y]])
    talus_l_abs = np.array([obs[STATE_TALUS_L_X], obs[STATE_TALUS_L_Y]])
    talus_r_abs = np.array([obs[STATE_TALUS_R_X], obs[STATE_TALUS_R_Y]])
    pelvis = np.array([obs[STATE_PELVIS_X], obs[STATE_PELVIS_Y]])

    head = head_abs - pelvis
    talus_l = talus_l_abs - pelvis
    talus_r = talus_r_abs - pelvis

    #cycle = 0.075
    cycle = 0.05
    (head_targ, talus_l_targ, talus_r_targ) = targs(step/cycle)
    replace_none(head_targ, head)
    replace_none(talus_l_targ, talus_l)
    replace_none(talus_r_targ, talus_r)

    head_diff = head_targ - head
    talus_l_diff = talus_l_targ - talus_l
    talus_r_diff = talus_r_targ - talus_r
    
    k2=0.005 # error term relative magnitude compared to reward
    k3=0.5 # rate of damping function

    error = k2 * ( err(head_diff) + err(talus_l_diff) + err(talus_r_diff))
    #error = k2*math.sqrt(l_diff*l_diff + r_diff*r_diff)*math.exp(-k3*x)
    #print("l_targ:", l_targ, "l_act:", l_act, "r_targ:", r_targ, "r_act:", r_act)
    #print("x:", x, "l_diff:", l_diff, "r_diff:", r_diff)

    if animate:
        print("step:", step, "reward:", reward, "error:", error)
        print("  PELVIS:", pelvis)
        print("  HEAD:", head, "targ:", head_targ, "diff:", head_diff, "err:", err(head_diff))
        print("  TALUS_L:", talus_l, "targ:", talus_l_targ, "diff:", talus_l_diff, "err:", err(talus_l_diff))
        print("  TALUS_R:", talus_r, "targ:", talus_r_targ, "diff:", talus_r_diff, "err:", err(talus_r_diff))

    if animate:
        # HACK
        trace["head"].append(head)
        trace["head_targ"].append(head_targ)
        trace["talus_l"].append(talus_l)
        trace["talus_l_targ"].append(talus_l_targ)
        trace["talus_r"].append(talus_r)
        trace["talus_r_targ"].append(talus_r_targ)

    return reward - error

def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = np.array(env.reset(difficulty=0))
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)

        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)

        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        #print(obs)
        reward = special_reward(obs, reward, step, animate)
        #print("reward:", reward)
        if done:
            # HACK
            if animate:
                with open("trace", 'wb') as f:
                    pickle.dump(trace, f)
                    print("wrote trace.")
        
        obs = np.array(obs)
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        # print "episode:", e, " of:", episodes, " done." #, rewards
        print(".", end="")
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })

def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, nprocs,
         policy_hid_list, valfunc_hid_list, gpu_pct, restore_path, animate):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """
    # killer = GracefulKiller()

    env, obs_dim, act_dim = init_osim(animate)
    env.seed(111 + mpi_util.rank)
    mpi_util.set_global_seeds(111 + mpi_util.rank)

    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    if mpi_util.rank == 0:
        #aigym_path = os.path.join('/tmp', env_name, now)
        #env = wrappers.Monitor(env, aigym_path, force=True)
        logger = Logger(logname=env_name, now=now)

    episode = 0

    checkpoint = Checkpoint("saves", now)
    # restore from checkpoint?
    if restore_path:
        (policy, val_func, scaler, episode, obs_dim, act_dim, kl_targ) = checkpoint.restore(restore_path)
    else:
        policy = Policy(obs_dim, act_dim, kl_targ)
        val_func = NNValueFunction(obs_dim)
        scaler = Scaler(obs_dim)

        if mpi_util.rank == 0:
            # run a few episodes (on node 0) of untrained policy to initialize scaler:
            trajectories = run_policy(env, policy, scaler, episodes=5)

            unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
            scaler.update(unscaled)  # update running statistics for scaling observations

        # broadcast policy weights, scaler, val_func
        (policy, scaler, val_func) = mpi_util.broadcast_policy_scaler_val(policy, scaler, val_func)

        if mpi_util.rank == 0: checkpoint.save(policy, val_func, scaler, episode)

    if animate:
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler, animate=animate)
        exit(0)
    
    worker_batch_size = int(batch_size / mpi_util.nworkers) # HACK
    if (worker_batch_size*mpi_util.nworkers != batch_size):
        print("batch_size:", batch_size, " is not divisible by nworkers:", mpi_util.nworkers)
        exit(1)

    batch = 0
    while episode < num_episodes:
        if mpi_util.rank == 0 and batch>0 and batch%10 == 0: checkpoint.save(policy, val_func, scaler, episode)
        batch = batch+1

        trajectories = run_policy(env, policy, scaler, episodes=worker_batch_size)
        trajectories = mpi_util.gather_trajectories(trajectories)

        if mpi_util.rank == 0:
            # concatentate trajectories into one list
            trajectories = list(itertools.chain.from_iterable(trajectories))
            print("did a batch of ",len(trajectories), " trajectories")
            print([t['rewards'].sum() for t in trajectories])

            episode += len(trajectories)
            add_value(trajectories, val_func)  # add estimated values to episodes
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            add_gae(trajectories, gamma, lam)  # calculate advantage

            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)

            # add various stats to training log:
            logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                        'Steps': np.sum([t['observes'].shape[0] for t in trajectories])})
            log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)

            policy.update(observes, actions, advantages, logger)  # update policy
            val_func.fit(observes, disc_sum_rew, logger)  # update value function

            unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
            scaler.update(unscaled)  # update running statistics for scaling observations

            logger.write(display=True)  # write logger results to file and stdout

        # if mpi_util.rank == 0 and killer.kill_now:
        #     if input('Terminate training (y/[n])? ') == 'y':
        #         break
        #     killer.kill_now = False

        # broadcast policy weights, scaler, val_func
        (policy, scaler, val_func) = mpi_util.broadcast_policy_scaler_val(policy, scaler, val_func)

    if mpi_util.rank == 0: logger.close()
    policy.close_sess()
    if mpi_util.rank == 0: val_func.close_sess()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000000000000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-r', '--restore_path', type=str,
                        help='Restore path',
                        default=None)

    parser.add_argument('-a', '--animate', action='store_true')
    parser.add_argument('--nprocs', type=int, default=1)
    parser.add_argument('--gpu_pct', type=float, default=0.0, help ='tensorflow  per_process_gpu_memory_fraction  option. .08 may work for 10 processes')
    parser.add_argument('--policy_hid_list', type=str, help='comma separated 3 layer list.  [30,40,25]', default='[]')
    parser.add_argument('--valfunc_hid_list', type=str, help='comma separated 3 layer list.  [30,40,25]', default='[]')

    args = parser.parse_args()
    args.policy_hid_list = eval(args.policy_hid_list)
    args.valfunc_hid_list = eval(args.valfunc_hid_list)
    if "parent" == mpi_util.mpi_fork(args.nprocs, gpu_pct=args.gpu_pct): sys.exit()
    main(**vars(args))
