from gym.envs.robotics import FetchReachEnv
import numpy as np
import gym


class SequentialFetchReach(gym.Env):
    x_range = (1.0, 1.5)
    y_range = (0.45, 1.05)
    z_range = (0.45, 0.65)
    # x_range = (1.2, 1.4)
    # y_range = (0.6, 0.8)
    # z_range = (0.5, 0.6)

    def __init__(self, num_objects, T, fixed=True, include_goal_info=True, reward_type='sparse', obj_range=0.15, target_range=0.15):
        self.num_objects = num_objects
        self.T = T
        self.env = FetchReachEnv(reward_type, obj_range, target_range)
        self.action_space = self.env.action_space
        self.include_goal_info = include_goal_info
        if include_goal_info:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = self.env.observation_space.spaces['observation']

        self.metadata = self.env.metadata
        self.reward_range = self.env.reward_range
        self.spec = self.env.spec
        self.np_random = self.env.np_random
        self.goals = []
        if fixed:
            try:
                f = open('env/fetch_goal.npy', 'rb')
            except:
                f = open('fetch_goal.npy', 'rb')
            goals = np.load(f)
            self.goals = goals[:num_objects]
        else:
            xs = self.np_random.uniform(self.x_range[0], self.x_range[1], size=self.num_objects)
            ys = self.np_random.uniform(self.y_range[0], self.y_range[1], size=self.num_objects)
            zs = self.np_random.uniform(self.z_range[0], self.z_range[1], size=self.num_objects)
            for i in range(self.num_objects):
                self.goals.append([xs[i], ys[i], zs[i]])
            self.goals = np.array(self.goals)
        self.pointer = 0
        self.compute_reward = self.env.compute_reward

    def _sample_goal(self):
        assert self.pointer == 0
        return self.goals[self.pointer]

    def compute_goal_reward(self, achieved_goal, goal, info):
        return self.env.compute_reward(achieved_goal, goal, info)

    def step(self, action):
        current_desired_goal = self.goals[self.pointer]
        obs, _reward, done, info = self.env.step(action)
        achieved_goal = obs['achieved_goal']
        reward = self.compute_goal_reward(achieved_goal, current_desired_goal, {})
        reward = {-1.0: -1/self.T, 0.0: 1.0}[reward]
        if reward == 1.0:
            # if self.pointer == self.num_objects - 1 and reward == 1.0:
            #     done = True
            self.pointer = min(self.num_objects-1, self.pointer + 1)
            obs['desired_goal'] = self.goals[self.pointer]
            reward /= self.num_objects
        self.env.goal = self.goals[self.pointer]
        if self.include_goal_info:
            obs = obs
        else:
            obs = obs['observation']
        return obs, reward, done, info

    def reset(self):
        self.pointer = 0
        current_goal = self.goals[self.pointer]
        obs = self.env.reset()
        obs['desired_goal'] = current_goal
        self.env.goal = current_goal
        if self.include_goal_info:
            return obs
        else:
            return obs['observation']

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, mode='human'):
        return self.env.render(mode)

# gym.register(
#     'SequentialFetchReach-v1',
#     entry_point='env.fetch_advanced:SequentialFetchReach',
#     kwargs={'num_objects': 2, 'include_goal_info': True},
#     max_episode_steps=50,
# )

gym.register(
    'SequentialFetchReach-v2',
    entry_point='env.fetch_advanced:SequentialFetchReach',
    kwargs={'num_objects': 1, 'include_goal_info': False, 'T': 50},
    max_episode_steps=50,
)


