import numpy as np
import os.path as osp

from baselines import logger
from irl.utils import qp_irl


class QPIRLAgent:
    def __init__(
            self,
            env,
            rl_agent,
            expert_path,
            num_eval,
            max_iter
    ):
        self.env = env
        self.rl_agent = rl_agent
        self.expert_model = self._load_expert_model(load_path=expert_path)
        self.num_eval = num_eval
        self.max_iter = max_iter

        self.expert_fe = self._get_expert_fe()
        logger.info('\n--------expert fe------------')
        logger.info(np.round(self.expert_fe, 3))
        # logger.info(np.reshape(np.round(self.expert_fe, 3), (6, 6)))
        logger.info('--------expert fe------------\n')
        self.list_policy_fe = []
        self.list_policy_reward = []

        self.weight = self.env.get_reward_weight()

    def learn(self):
        for iter_id in range(1, self.max_iter+1):
            fe, reward = self._optimize_policy()
            logger.info('\n------------fe--------------')
            logger.info(np.round(fe, 3))
            # logger.info(np.reshape(np.round(fe, 3), (6, 6)))
            logger.info('------------fe--------------\n')
            self.list_policy_fe.append(fe)
            self.list_policy_reward.append(reward)

            weight = self._get_optimal_reward_weight()
            logger.info('\n------------weight--------------')
            logger.info(np.round(weight, 3))
            # logger.info(np.reshape(np.round(weight, 3), (6, 6)))
            logger.info('------------weight--------------\n')
            self.weight = weight
            self._set_env_reward_weight(self.weight)

            max_value = np.dot(self.weight, fe)
            for fe in self.list_policy_fe:
                value = np.dot(self.weight, fe)
                if value > max_value: max_value = value
            expert_value = np.dot(self.weight, self.expert_fe)
            gap = expert_value - max_value
            logger.info('irl iter:{}|value gap:{:.4f}|value expert:{:.4f}|policy reward:{:.4f}'.format(iter_id, gap, expert_value, reward))

    def _optimize_policy(self):
        """Get feature expectation of optimal policy under current weight"""
        result = self.rl_agent.learn(self.env)
        fe, reward = result['fe'], result['reward']
        assert fe.shape == self.weight.shape, 'fe:{} expected to be: {}'.format(fe.shape, self.weight.shape)
        return fe, reward

    def _set_env_reward_weight(self, weight):
        self.env.set_reward_weight(weight)

    def _get_optimal_reward_weight(self):
        expert_fe, policy_fes = self.expert_fe, self.list_policy_fe
        weight = qp_irl(expert_fe, policy_fes)
        return weight

    def _load_expert_model(self, load_path):
        expert_model = self.rl_agent.load_model(env=self.env, scope='expert', load_path=load_path)
        return expert_model

    def _get_expert_fe(self):
        """Get Expert Feature Expectation"""
        expert_fes = []
        for _ in range(self.num_eval):
            expert_fes.append(self.rl_agent.get_policy_feature_expectation(model=self.expert_model, env=self.env))
        return np.array(expert_fes).mean(axis=0)

    def save(self):
        with open(osp.join(logger.get_dir(), 'weight.npy'), 'wb') as f:
            np.save(f, self.weight)
