import gym
import argparse
import datetime
import shutil
import os
import os.path as osp
import importlib

from baselines import logger
from baselines.run import parse_unknown_args
from baselines.common.cmd_util import set_global_seeds
from opl.common.dataset import MuJoCoDataset
import inspect


def configure_log(args):
    path = os.path.join("logs", "{}-{}-{}".format(
        args.task, args.env, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")))
    logger.configure(dir=path)

    codepath = osp.join(logger.get_dir(), "code")
    os.makedirs(codepath, exist_ok=True)
    for subdir in ['baselines', 'gym', 'irl', 'opl']:
        shutil.copytree(subdir, osp.join(codepath, subdir))

def main(args, extra_args):
    assert 'opl' not in os.getcwd(), 'Please run the program in the parent dir'
    configure_log(args)
    set_global_seeds(args.seed)

    env = gym.make(args.env)
    dataset = MuJoCoDataset(data_path=osp.join('dataset', args.env), env_id=args.env)

    if args.task == 'transition':
        learn = importlib.import_module('opl.transition_learning.{}'.format(args.transition_model)).learn
        learn_argspec = inspect.getfullargspec(learn).args
        kwargs = {}
        for key, value in extra_args:
            if key in learn_argspec:
                kwargs[key] = value
        model = learn(
            env=env,
            dataset=dataset,
            **kwargs
        )
    else:
        raise NotImplementedError




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--task', type=str, default='transition')

    parser.add_argument('--transition_model', type=str, default='behavioral_cloning')

    args, unknown_args = parser.parse_known_args()
    extra_args = parse_unknown_args(unknown_args)

    main(args, extra_args)
