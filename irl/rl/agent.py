import importlib
import os
import os.path as osp

from baselines import logger
from rl.utils import get_default_network

import numpy as np


class RLAgent:
    def __init__(
            self,
            algorithm,
            env,
            total_timesteps,
            gamma,
            num_eval,
            **alg_kwargs
    ):
        self.algorithm = algorithm
        self.env = env
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.num_eval = num_eval
        self.alg_kwargs = alg_kwargs

        self.model_params = self.alg_kwargs['model_params']
        self.learn_params = self.alg_kwargs['learn_params']

        self.model = self.load_model(env, scope='rl', load_path=None)
        self.learn_fn = self._import_learn_fn()
        self.count = 0

    def learn(self, env):
        self.model.initialize()
        model = self.learn_fn(
            env=env,
            model=self.model,
            total_timesteps=self.total_timesteps,
            **self.learn_params
        )
        self.count += 1

        model_path = osp.join(logger.get_dir(), 'models')
        os.makedirs(model_path, exist_ok=True)
        save_path = osp.join(model_path, '%s.model' % self.count)
        model.save(save_path)
        logger.info('rl agent saving model weight into :{}'.format(save_path))

        fes, rewards = [], []
        for _ in range(self.num_eval):
            result = self.play(model, env)
            fes.append(result['fe'])
            rewards.append(result['reward'])
        result = {
            'fe': np.asarray(fes).mean(axis=0),
            'reward': np.mean(rewards)
        }

        self.model = model
        return result

    def get_policy_feature_expectation(self, model, env):
        return self.play(model, env)['fe']

    def load_model(self, env, scope, load_path=None):
        build_model_fn = self._import_model_fn()
        model = build_model_fn(
            env=env,
            scope=scope,
            network=get_default_network(env),
            load_path=load_path,
            **self.model_params
        )
        return model

    def play(self, model, env):
        list_fe = []
        states = env.reset()
        gamma = self.gamma
        reward = 0
        while True:
            actions, *_ = model.step(states)
            states, rewards, dones, infos = env.step(actions)
            list_fe.append(infos[0]['feature'])
            reward += infos[0]['true_reward']
            if dones[0]:
                break
        fe_res = np.zeros(env.get_reward_weight().shape)
        for i, fe in enumerate(list_fe[::-1]):
            fe_res += (gamma ** i) * fe
        result = {
            'fe': fe_res,
            'reward': reward
        }
        return result

    def _import_learn_fn(self):
        alg = importlib.import_module('rl.{}'.format(self.algorithm))
        return alg.learn

    def _import_model_fn(self):
        alg = importlib.import_module('rl.{}'.format(self.algorithm))
        return alg.build_model
