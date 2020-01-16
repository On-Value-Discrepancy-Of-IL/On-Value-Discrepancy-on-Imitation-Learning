import gym
import numpy as np
from opl.common.mujoco.mujoco_env import MuJoCoWrapper


class Walker2dEnv(MuJoCoWrapper):
    def __init__(self, transition_model):
        env = gym.make('Walker2d-v2')
        super().__init__(env, transition_model)

        self.state = None

    def reset(self, **kwargs):
        self.env.reset()
        self.state = self.env.unwrapped.get_save_variables()
        return self._get_obs()

    def step(self, action):
        posbefore = self.state[0].copy()    # self.sim.data.qpos[0]
        self.state = self.transition_model.step(self.state, action)

        posafter, height, ang = self.state[0:3].copy()  # self.sim.data.qpos[0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        qpos = self.state[:self.dim_qpos]
        qvel = self.state[self.dim_qpos:]
        ob = np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        assert ob.shape == self.ob_shape, 'ref:{}, real:{}'.format(self.ob_shape, ob.shape)
        return ob

    def render(self, mode='human', **kwargs):
        raise NotImplementedError

