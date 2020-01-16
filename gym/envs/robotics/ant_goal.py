import numpy as np
import gym
from gym.envs.mujoco.ant import AntEnv


def goal_distance(ag, g, info):
    assert ag.shape == g.shape
    return np.linalg.norm(ag - g, axis=-1)


class AntGoalEnv(gym.core.Wrapper):
    def __init__(self, target_range=0.4, threashold=0.05):
        env = AntEnv(frame_skip=2)
        gym.core.Wrapper.__init__(self, env)

        obs_space = self.observation_space
        ag_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        g_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            'observation': obs_space,
            'achieved_goal': ag_space,
            'desired_goal': g_space
        })

        self.target_range = target_range
        self.threashold = threashold

    def _get_obs(self):
        return np.concatenate([
            self.env.sim.data.qpos.flat[2:],
            self.env.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.env.get_body_com("torso"),
        ])

    def reset(self):
        self.env.reset()
        obs = self._get_obs()
        self.goal = self.sample_goal()
        ag = self.get_achieved_goal()
        dict_obs = {
            'observation': obs,
            'achieved_goal': ag,
            'desired_goal': self.goal.copy()
        }
        return dict_obs

    def step(self, action):
        _, inner_reward, done, info = self.env.step(action)
        obs = self._get_obs()
        ag = self.get_achieved_goal()
        g = self.goal.copy()

        dict_obs = {'observation': obs, 'achieved_goal': ag, 'desired_goal': g}
        reward = self.compute_reward(ag[None, :], g[None, :], info)[0]
        info['is_success'] = reward == 0.0
        info['is_dying'] = done

        return dict_obs, reward, False, info

    def sample_goal(self):
        init_ag = self.get_achieved_goal()
        offset = np.random.uniform(-self.target_range, self.target_range, size=2)
        while False:
            if np.linalg.norm(offset) > 0.05:
                break
            offset = np.random.uniform(-self.target_range, self.target_range, size=2)
        return init_ag + offset

    def compute_reward(self, ag, g, info):
        distance = goal_distance(ag, g, info)
        reward = np.zeros(len(ag), dtype=np.float32)
        reward.fill(-1.)
        reward[distance < self.threashold] = 0.
        return reward

    def get_achieved_goal(self):
        position = self.env.get_body_com("torso")
        return position[:2]

if __name__ == "__main__":
    env = AntGoalEnv()
    ac_space = env.action_space
    ob_space = env.observation_space
    list_ag_diff = []
    for k in range(10):
        obs = env.reset()
        ag = obs['achieved_goal'].copy()
        for t in range(50):
            act = ac_space.sample()
            obs, reward, done, info = env.step(act)
            ag_new = obs['achieved_goal'].copy()
            # print(ag_new, ag)
            ag_diff = np.linalg.norm(ag_new - ag)
            if t > 0:
                list_ag_diff.append(ag_diff)
            # print(ag_diff)
            print( reward, done, info)
            ag = ag_new
    print(np.mean(list_ag_diff))



