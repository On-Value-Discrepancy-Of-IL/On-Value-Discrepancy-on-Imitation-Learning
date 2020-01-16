from baselines.common.cmd_util import logger
from baselines.common import set_global_seeds, tf_util as U
from tqdm import tqdm
import tensorflow as tf
from baselines.common.mpi_adam import MpiAdam
import os.path as osp
import os
import gym
import tempfile
from baselines.gail import mlp_policy
import argparse
from baselines.common.misc_util import boolean_flag
from opl.common.dataset import MuJoCoDataset
import datetime
import numpy as np
from baselines.gail.run_mujoco import runner
traj_limitation = [5, 10, 15, 20, 25]

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=25)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--task', type=str, default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
    return parser.parse_args()


def policy_fn(name, ob_space, ac_space, reuse=False):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                reuse=reuse, hid_size=100, num_hid_layers=2)


def get_task_name(args, max_traj=None):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    if max_traj != None:
        task_name += '.traj_limitation_{}'.format(max_traj)
    else:
        task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
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
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert, rs, dones, mus, states, states2 = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert, rs, dones, mus, states, states2 = dataset.get_next_batch(optim_batch_size, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, '{}.model'.format(max_iters))
    # U.save_state(savedir_fname, var_list=pi.get_variables())
    U.save_variables(savedir_fname, variables=pi.get_variables())
    return savedir_fname


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)
    os.makedirs("logs", exist_ok=True)
    path = osp.join("logs", "{}-{}-{}-{}".format('bc', args.env_id, args.traj_limitation, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")))
    logger.configure(dir=path)
    env.seed(args.seed)
    args.log_dir = logger.get_dir()

    # if args.expert_path is None:
    #     args.expert_path = 'baselines/gail/expert/deterministic.ppo.{}-v2.0.00.npz'.format(args.env_id.split('-')[0])
    # assert osp.exists(args.expert_path),  '%s does not exist' % args.expert_path
    data_path = osp.join('dataset', args.env_id, 'expert')
    dataset = MuJoCoDataset(data_path=data_path, env_id=args.env_id, traj_limit=args.traj_limitation)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    if args.task == 'train':
        savedir_fname = learn(env,
                              policy_fn,
                              dataset,
                              max_iters=args.BC_max_iter,
                              ckpt_dir=args.checkpoint_dir,
                              log_dir=args.log_dir,
                              task_name=task_name,
                              verbose=True)
        avg_len, avg_ret = runner(env,
                                  policy_fn,
                                  savedir_fname,
                                  timesteps_per_batch=1024,
                                  number_trajs=10,
                                  stochastic_policy=args.stochastic_policy,
                                  save=args.save_sample,
                                  reuse=True)
    else:
        U.make_session(num_cpu=1).__enter__()
        rets = np.zeros(shape=len(traj_limitation))
        for i in range(len(traj_limitation)):
            max_traj = traj_limitation[i]
            args.traj_limitation = max_traj
            task_name = get_task_name(args)
            savedir_fname = osp.join('checkpoint', task_name, '{}.model'.format(args.BC_max_iter))
            if i == 0:
                reuse = False
            else:
                reuse = True
            avg_len, avg_ret = runner(env,
                                      policy_fn,
                                      savedir_fname,
                                      timesteps_per_batch=1024,
                                      number_trajs=20,
                                      stochastic_policy=args.stochastic_policy,
                                      save=args.save_sample,
                                      reuse=reuse)
            rets[i] = avg_ret
            os.makedirs("results", exist_ok=True)
            file = osp.join('results', args.env_id)
            np.save(file=file, arr=rets)


if __name__ == '__main__':
    args = argsparser()
    main(args)