import importlib
import shutil
import os
import os.path as osp
import argparse
import datetime

from baselines import logger

from env.env_util import build_env
from rl.agent import RLAgent
from rl.utils import get_default_network, get_env_type, set_global_seeds

def get_irl_agent(irl_alg):
    agent = importlib.import_module('irl.{}_irl'.format(
        irl_alg,
    ))
    return getattr(agent, '{}IRLAgent'.format(irl_alg.strip('_').upper()))

def get_rl_learn_fn(alg):
    alg = importlib.import_module('rl.{}'.format(alg))
    return alg.learn

def get_rl_model_fn(alg):
    alg = importlib.import_module('rl.{}'.format(alg))
    return alg.build_model

def get_rl_default_params(alg, env_type):
    alg = importlib.import_module('rl.{}'.format(alg))
    return alg.get_default_params(env_type)

def configure_log(alg, env):
    os.makedirs("logs", exist_ok=True)
    path = osp.join("logs", "{}-{}-{}".format(alg, env, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")))
    logger.configure(dir=path)
    for subdir in ['env', 'rl', 'irl']:
        shutil.copytree(os.path.join(os.getcwd(), subdir), os.path.join(path, "code", subdir))
    for subfile in ['run.py']:
        shutil.copy(subfile, os.path.join(path, "code"))

def get_expert_policy(args):
    configure_log(args.rl_alg, args.env_id)
    set_global_seeds(args.seed)

    env = build_env(args.env_id, num_env=args.num_env, seed=args.seed)
    default_params = get_rl_default_params(args.rl_alg, get_env_type(env))
    default_params['learn_params']['log_interval'] = 10
    model_fn = get_rl_model_fn(args.rl_alg)
    model = model_fn(
        env=env,
        network=get_default_network(env),
        scope='expert',
        seed=args.seed,
        **default_params['model_params']
    )
    learn_fn = get_rl_learn_fn(args.rl_alg)
    model = learn_fn(
        env=env,
        model=model,
        total_timesteps=args.rl_max_steps,
        gamma=args.gamma,
        **default_params['learn_params'],
    )
    model_path = osp.join(logger.get_dir(), 'expert.model')
    model.save(model_path)
    logger.info('saving expert model into:{}'.format(model_path))
    return model

def main(args):
    assert args.task == 'irl'
    configure_log(args.irl_alg, args.env_id)
    set_global_seeds(args.seed)

    env = build_env(args.env_id, irl_wrapper=True)
    rl_default_params = get_rl_default_params(args.rl_alg, get_env_type(env))
    rl_agent = RLAgent(
        env=env,
        algorithm=args.rl_alg,
        total_timesteps=args.rl_max_steps,
        gamma=args.gamma,
        num_eval=args.rl_num_eval,
        **rl_default_params
    )
    irl_agent_class = get_irl_agent(args.irl_alg)

    irl_agent = irl_agent_class(
        env=env,
        rl_agent=rl_agent,
        expert_path=args.irl_expert_model,
        max_iter=args.irl_max_iter,
        num_eval=args.irl_num_eval,
    )

    irl_agent.learn()

    irl_agent.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Common params
    parser.add_argument('--env_id', type=str, default='MountainCarContinuous-v0') # Continuous
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--num_env', type=int, default=2)
    parser.add_argument('--task', type=str, choices=['rl', 'irl'], default='rl')
    # RL params
    parser.add_argument('--rl_alg', type=str, default='ppo')
    parser.add_argument('--rl_max_steps', type=int, default=5e4)
    parser.add_argument('--rl_num_eval', type=int, default=4)
    # IRL params
    parser.add_argument('--irl_alg', type=str, default='qp')
    parser.add_argument('--irl_max_iter', type=int, default=20)
    parser.add_argument('--irl_num_eval', type=int, default=4)
    parser.add_argument('--irl_expert_model', type=str)

    args = parser.parse_args()
    if args.task == 'rl':
        get_expert_policy(args)
    else:
        main(args)
