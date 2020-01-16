import os
import os.path as osp
import functools

import gym
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.bench import Monitor
from baselines import logger

from env.irl_wrapper import IRLWrapper
from env.gridworld_wrapper import make_gridworld
from env.dummy_vec_env import DummyVecEnv
from env.subproc_vec_env import SubprocVecEnv


def make_env(env_id, logger_dir, rank=0, irl_wrapper=False, seed=None):
    if 'MiniGrid' in env_id:
        env = make_gridworld(env_id)
    else:
        env = gym.make(env_id)
    env_path = osp.join(logger_dir, 'monitor_{}'.format(rank))
    env = Monitor(env, env_path, allow_early_resets=True)
    if irl_wrapper:
        env = IRLWrapper(env)
    env.seed(seed)
    return env

def build_env(env_id, num_env=1, irl_wrapper=False, seed=None):
    seed = seed or 2019
    env_fns = []

    logger_dir = logger.get_dir()

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            rank=rank,
            irl_wrapper=irl_wrapper,
            seed=seed+rank,
            logger_dir=logger_dir,
        )

    for i in range(num_env):
        env_fns.append(make_thunk(rank=i))
    if num_env == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    return env
