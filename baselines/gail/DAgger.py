'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''
import re
import numpy as np
import datetime
from collections import defaultdict
import multiprocessing
import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm
import sys
import tensorflow as tf
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import common_arg_parser
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.gail import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail.run_mujoco import runner
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.ppo2.ppo2 import learn
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from importlib import import_module

import matplotlib.pyplot as plt
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def get_task_name(args):
    task_name = 'DAgger'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    if args.stochastic_policy:
        task_name += '.stochastic'
    else:
        task_name += '.deterministic'
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, ob=False, ret=False, use_tf=True)

    return env


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of DAgger")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='./checkpoint/DAgger')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--dagger_iters', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)

    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco',
                        default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--task_name', default='train', type=str)
    parser.add_argument('--play', default=False, action='store_true')
    return parser.parse_args()


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def plot(x, y):
    plt.scatter(x, y)
    file_name = './dagger4.png'
    plt.savefig(file_name)


def test_expert(expert_model, env):
    ob = env.reset()
    dones = np.zeros((1,))
    state = expert_model.initial_state if hasattr(expert_model, 'initial_state') else None
    ret = 0
    position_xs = []
    position_ys = []
    while True:
        # if state is not None:
        #     actions, _, state, _ = expert_model.step(ob, S=state, M=dones)
        # else:
        #     actions, _, _, _ = expert_model.step(ob)
        actions, _, state, _ = expert_model.step(ob, S=state, M=dones)
        ob, rew, new, info_ = env.step(actions)
        position_xs.append(info_['postion_x'])
        position_ys.append(info_['postion_y'])
        ret += rew
        if new:
            return position_xs, position_ys


def DAgger_traj_1_generator(expert_model, pi, env, horizon, stochastic):

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
    dones = np.zeros((1,))
    value = 0
    state = expert_model.initial_state if hasattr(expert_model, 'initial_state') else None
    while True:
        ac, vpred = pi.act(stochastic, ob)
        if state is not None:
            actions, _, state, _ = expert_model.step(ob, S=state, M=dones)
        else:
            actions, _, _, _ = expert_model.step(ob)
        obs.append(ob)
        news.append(new)
        acs.append(actions)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ob = env.reset()
        if t == horizon-1:
            break
        t += 1

    obs = np.array(obs)
    obs = np.squeeze(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    acs = np.squeeze(acs)
    # rew:reward ep_ret: return
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


def DAgger_test_traj_1_generator(expert_model, pi, env, horizon, stochastic):

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
    dones = np.zeros((1,))
    value = 0
    ret = 0
    state = expert_model.initial_state if hasattr(expert_model, 'initial_state') else None
    while True:
        ac, vpred = pi.act(stochastic, ob)
        if state is not None:
            actions, _, state, _ = expert_model.step(ob, S=state, M=dones)
        else:
            actions, _, _, _ = expert_model.step(ob)
        obs.append(ob)
        news.append(new)
        acs.append(actions)
        if t >= 20:
            ob, rew, new, _ = env.step(actions)
        else:
            ob, rew, new, _ = env.step(ac)
        rews.append(rew)
        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    obs = np.squeeze(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    acs = np.squeeze(acs)
    # rew:reward ep_ret: return
    traj = {"ret": np.sum(rews), "length": rews.shape[0]}
    return traj






def DAgger_test(expert_model, env, policy_func, dataset, optim_batch_size=128, dagger_iters=1, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()

    logger.log("Pretraining with Behavior Cloning...")
    for dagger_iter in range(dagger_iters):
        for iter_so_far in tqdm(range(int(max_iters))):
            ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
            train_loss, g = lossandgrad(ob_expert, ac_expert, True)
            adam.update(g, optim_stepsize)
            if verbose and iter_so_far % val_per_iter == 0:
                ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
                val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
                logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
        # sample data
        traj = DAgger_traj_1_generator(expert_model, pi, env, horizon=50, stochastic=False)
        obs = traj['ob']
        acs = traj['ac']
        dataset.update(obs, acs)
    print(dataset.train_acs.shape)
    print(dataset.train_obs.shape)
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    U.save_state(savedir_fname, var_list=var_list)
    return savedir_fname


def DAgger_learn(expert_model, env, policy_func, dataset, optim_batch_size=128, dagger_iters=1, each_iters=1e3,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    max_iters = int(dataset.acs.shape[0] / 50) * each_iters

    logger.log("Pretraining with Behavior Cloning...")
    for dagger_iter in range(dagger_iters):
        val_per_iter = int(max_iters / 10)
        for iter_so_far in tqdm(range(int(max_iters))):
            ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
            train_loss, g = lossandgrad(ob_expert, ac_expert, True)
            adam.update(g, optim_stepsize)
            if verbose and iter_so_far % val_per_iter == 0:
                ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
                val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
                logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
        # sample data
        traj = DAgger_traj_1_generator(expert_model, pi, env, horizon=50, stochastic=False)
        obs = traj['ob']
        acs = traj['ac']
        dataset.update(obs, acs)
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    U.save_state(savedir_fname, var_list=var_list)
    return savedir_fname


def main(args):
    # load expert model
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    import os
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        path = os.path.join("logs", "{}-{}-{}".format(
            args.alg, args.env, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")))
        configure_logger(path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    expert_model, env = train(args, extra_args)
    # train DAgger
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    logger.configure(dir=args.log_dir)
    dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    dagger_iters = args.traj_limitation
    savedir_fname = DAgger_learn(expert_model, env,
                                 policy_fn,
                                 dataset,
                                 dagger_iters=dagger_iters,
                                 each_iters=args.each_iter,
                                 ckpt_dir=args.checkpoint_dir,
                                 log_dir=args.log_dir,
                                 task_name=task_name,
                                 verbose=True)

    # test expert
    # test_trajs = 10
    # position_xs = []
    # position_ys = []
    # for i in tqdm(range(test_trajs)):
    #     tmp_position_x, tmp_position_y = test_expert(expert_model, env)
    #     tmp_position_x = np.array(tmp_position_x)
    #     tmp_position_y = np.array(tmp_position_y)
    #     position_xs.append(tmp_position_x)
    #     position_ys.append(tmp_position_y)
    # position_xs = np.concatenate(position_xs, axis=0)
    # position_ys = np.concatenate(position_ys, axis=0)
    # position = np.vstack((position_xs, position_ys))
    # np.save('./expert', position)
    # print(position.shape)



if __name__ == '__main__':

    # args = argsparser()
    main(sys.argv)
    # postion = np.load('./dagger4.npy')
    # plot(postion[0], postion[1])
