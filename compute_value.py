

from collections import namedtuple
import argparse
import os
import glob
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import tensorflow as tf

from baselines.gail import run_mujoco
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from tqdm import tqdm
import os.path as osp
import datetime


plt.style.use('ggplot')
CONFIG = {
    'traj_limitation': [4, 11, 18, 25],
}
# gamma_list = [0.9, 0.99, 0.999]
# seed_list = [0, 1, 2, 3, 4]


def argsparser():
    parser = argparse.ArgumentParser('Do evaluation')
    parser.add_argument('--path', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    parser.add_argument('--env', type=str, default='Hopper', choices=['Hopper', 'Walker2d', 'HalfCheetah',
                                                    'Humanoid', 'HumanoidStandup', 'Ant'])
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    return parser.parse_args()


def load_dataset(expert_path):
    dataset = Mujoco_Dset(expert_path=expert_path)
    return dataset


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs, stochastic_policy,
           gamma, prefix,  save=False, reuse=False):
    # 用来评估给定策略, 在环境中采轨迹
    # Setup network
    # input: policy_func:策略网络 load_model_path: 训练策略的模型 timesteps_per_batch 每一条轨迹的最大步长 number_trajs 轨迹个数
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    #
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
            U.initialize()
            # Prepare for rollouts
            # ----------------------------------------
            if prefix == 'DAgger':
                var_list = pi.get_trainable_variables()
                U.load_state(load_model_path, var_list=var_list)
            else:
                U.load_state(load_model_path)

            obs_list = []
            acs_list = []
            len_list = []
            ret_list = []
            value_list = []
            for _ in tqdm(range(number_trajs)):
                # 采一条轨迹
                traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy, gamma=gamma)
                # ep_ret: return np.array
                obs, acs, ep_len, ep_ret, ep_value = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret'], traj['value']
                obs_list.append(obs)
                acs_list.append(acs)
                len_list.append(ep_len)
                ret_list.append(ep_ret)
                value_list.append(ep_value)
    # 存储轨迹数据
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    # 许多条轨迹的平均长度和平均reward
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    avg_value = sum(value_list) / len(value_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret, avg_value


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic, gamma=np.array([0.9, 0.99, 0.999], dtype=np.float32)):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    value = 0

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        value += np.power(gamma, t) * rew
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs, np.float32)
    rews = np.array(rews, np.float32)
    news = np.array(news)
    values = np.zeros(shape=(rews.shape[0], 3), dtype=np.float32)
    # values = np.zeros_like(rews, dtype=np.float32)
    values[-1] = rews[-1]
    i = rews.shape[0]-1
    while True:
        values[i-1] = rews[i-1] + gamma * values[i]
        if i <= 1:
            break
        i = i-1
    mean_value = np.mean(values, axis=0)
    # rew:reward ep_ret: return
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, 'value': mean_value}
    return traj


def evaluate(*, env_name, alg, traj_limit, checkpoint_dir, gamma, seed, policy_hidden_size, stochastic_policy):

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=policy_hidden_size, num_hid_layers=2)

    data_path = 'baselines/gail/dataset/deterministic.trpo.{}.0.00.npz'.format(env_name.split('-')[0])

    # data_path = os.path.join('data', 'deterministic.ppo.' + env_name + '.0.00.npz')
    dataset = load_dataset(data_path)

    # Do one evaluation
    upper_bound = sum(dataset.rets[:traj_limit])/traj_limit
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    # print(checkpoint_path)
    # assert 0
    env = gym.make(env_name + '-v2')
    env.seed(seed)
    print('Trajectory limitation: {}, Load checkpoint: {}, '.format(traj_limit, checkpoint_path))
    avg_len, avg_ret, avg_value = runner(env,
                                         policy_fn,
                                         checkpoint_path,
                                         timesteps_per_batch=1024,
                                         number_trajs=20,
                                         stochastic_policy=stochastic_policy,
                                         prefix=alg,
                                         gamma=gamma,
                                         reuse=False)
    normalized_ret = avg_ret/upper_bound
    print('Upper bound: {}, evaluation returns: {}, normalized scores: {}'.format(
        upper_bound, avg_ret, normalized_ret))
    env.close()
    result = {
        'upper_bound': upper_bound,
        'avg_ret': avg_ret,
        'avg_len': avg_len,
        'avg_value': avg_value,
        'normalized_ret': normalized_ret
    }
    return result

Result = namedtuple('Result', 'alg traj seed value upper_bound avg_ret')

def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    basedir = args.path

    for path in sorted(os.listdir(basedir)):
        if not path.startswith('gail'):
            continue
        env_name, alg, traj_limit = path.split('-')[2], path.split('-')[1], int(path.split('-')[4])
        print('process :{}'.format(osp.join(basedir, path)))
        subdirs = os.listdir(osp.join(basedir, path))
        subdirs.sort(key=lambda x: len(x))
        checkpoint_dir = subdirs[-1]
        exp_seed = int(checkpoint_dir.split('seed_')[-1])
        checkpoint_dir = osp.join(basedir, path, checkpoint_dir)

        try:
            dict_res = evaluate(
                env_name=env_name,
                alg=alg,
                policy_hidden_size=args.policy_hidden_size,
                seed=exp_seed,
                stochastic_policy=args.stochastic_policy,
                traj_limit=traj_limit,
                checkpoint_dir=checkpoint_dir,
                gamma=np.array([0.9, 0.99, 0.999], dtype=np.float32)
            )

            savedir = osp.join('results')
            os.makedirs(savedir, exist_ok=True)
            filename = osp.join(savedir, '{}-{}-{}-{}.npz'.format(alg, env_name, traj_limit, exp_seed))
            np.savez(filename, **dict_res)
            print('save into :{}'.format(filename))
        except Exception as e:
            print(e)
            print('{} fails'.format(path))

        # results['alg'].append(alg)
        # results['traj'].append(traj_limit)
        # results['seed'].append(exp_seed)
        # for key, value in dict_res.items():
        #     if key not in results:
        #         results[key] = []
        #     results[key].append(value)



if __name__ == '__main__':
    args = argsparser()
    main(args)
