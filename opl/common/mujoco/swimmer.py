import gym
import numpy as np

from opl.common.mujoco.mujoco_env import MuJoCoWrapper


class SwimmerEnv(MuJoCoWrapper):
    def __init__(self, transition_model):
        env = gym.make('Swimmer-v2')
        super().__init__(env, transition_model)

        self.state = None

    def reset(self, **kwargs):
        self.env.reset()
        self.state = self.env.unwrapped.get_save_variables()
        return self._get_obs()

    def step(self, action):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.state[0].copy()     # self.sim.data.qpos[0]
        self.state = self.transition_model.step(self.state, action)

        xposafter = self.state[0].copy()    # self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(action).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # return np.concatenate([qpos.flat[2:], qvel.flat])
        qpos = self.state[:self.dim_qpos]
        qvel = self.state[self.dim_qpos:]
        ob = np.concatenate([qpos.flat[2:], qvel.flat])
        assert ob.shape == self.ob_shape, 'ref:{}, real:{}'.format(self.ob_shape, ob.shape)
        return ob

    def render(self, mode='human', **kwargs):
        raise NotImplementedError
