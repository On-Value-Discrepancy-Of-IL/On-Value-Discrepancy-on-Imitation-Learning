import numpy as np
import gym
from opl.common.mujoco.mujoco_env import MuJoCoWrapper


class AntEnv(MuJoCoWrapper):
    def __init__(self, transition_model):
        env = gym.make('Ant-v2')
        super().__init__(env, transition_model)

        self.state = None

    def reset(self, **kwargs):
        self.env.reset()
        self.state = self.env.unwrapped.get_save_variables()
        return self._get_obs()

    def step(self, action):
        xposbefore = self.state[-3].copy()   # self.get_body_com("torso")[0]
        self.state = self.transition_model.step(self.state, action)

        xposafter = self.state[-3].copy()   # self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(action).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        cfrc_ext = self.state[15+14:-3]
        assert len(cfrc_ext) == 84
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(cfrc_ext, -1, 1)))

        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        notdone = np.isfinite(self.state[:-3]).all() and self.state[2] >= 0.2 and self.state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        # qpos: (15,),  vel: (14, ), cfrc_ext: (14, 6), torso: (3,)
        # return np.concatenate([
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat,
        #     np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        # ])
        qpos = self.state[:15]
        qvel = self.state[15: 15+14]
        cfrc_ext = self.state[15+14: -3]
        ob = np.concatenate([qpos[2:].flat, qvel.flat, np.clip(cfrc_ext, -1., 1.).flat])
        assert ob.shape == self.ob_shape, 'ref:{}, real:{}'.format(self.ob_shape, ob.shape)
        return ob

    def render(self, mode='human', **kwargs):
        raise NotImplementedError
