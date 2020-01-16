import numpy as np
import gym
from opl.common.mujoco.mujoco_env import MuJoCoWrapper


class HalfCheetahEnv(MuJoCoWrapper):
    def __init__(self, transition_model):
        env = gym.make('HalfCheetah-v2')
        super().__init__(env, transition_model)

        self.state = None

    def reset(self, **kwargs):
        self.env.reset()
        self.state = self.env.unwrapped.get_save_variables()
        return self._get_obs()

    def step(self, action):
        xposbefore = self.state[0].copy()   # self.sim.data.qpos[0]
        self.state = self.transition_model.step(self.state, action)

        xposafter = self.state[0].copy()    # self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])
        qpos = self.state[:self.dim_qpos]
        qvel = self.state[self.dim_qpos:]
        ob = np.concatenate([qpos.flat[1:], qvel.flat])
        assert ob.shape == self.ob_shape, 'ref:{}, real:{}'.format(self.ob_shape, ob.shape)
        return ob

    def render(self, mode='human', **kwargs):
        raise NotImplementedError
