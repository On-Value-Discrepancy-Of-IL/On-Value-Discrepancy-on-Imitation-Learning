import gym
import numpy as np

from opl.common.mujoco.mujoco_env import MuJoCoWrapper


class HopperEnv(MuJoCoWrapper):
    def __init__(self, transition_model):
        env = gym.make('Hopper-v2')
        super().__init__(env, transition_model)

        self.state = None

    def reset(self, **kwargs):
        self.env.reset()
        self.state = self.env.unwrapped.get_save_variables()
        return self._get_obs()

    def step(self, action):
        posbefore = self.state[0].copy()    # self.sim.data.qpos[0]
        self.state = self.transition_model.step(self.state, action)

        posafter, height, ang = self.state[0:3].copy()     # self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        done = not (np.isfinite(self.state).all() and (np.abs(self.state[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     np.clip(self.sim.data.qvel.flat, -10, 10)
        # ])
        qpos = self.state[:self.dim_qpos]
        qvel = self.state[self.dim_qpos:]
        ob = np.concatenate([qpos.flat[1:], np.clip(qvel.flat, -10, 10)])
        assert ob.shape == self.ob_shape, 'ref:{}, real:{}'.format(self.ob_shape, ob.shape)
        return ob

    def render(self, mode='human', **kwargs):
        raise NotImplementedError
