import gym_minigrid
from gym_minigrid.wrappers import SimpleActionWrapper, ActionWrappers
import gym
from gym.wrappers.time_limit import TimeLimit

import numpy as np


class TimePenaltyWrapper(gym.core.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

        if hasattr(self.env, 'max_episode_steps'):
            self.max_episode_steps = self.env.max_episode_steps
        elif hasattr(self.env, '_max_episode_steps'):
            self.max_episode_steps = self.env._max_episode_steps
        else:
            self.max_episode_steps = 1000
        self.penalty = -1. / self.max_episode_steps

    def reward(self, reward):
        return reward + self.penalty


class InfiniteWorld(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.height = self.unwrapped.height - 2
        self.width = self.unwrapped.width - 2
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.height * self.width,), dtype=np.float32)

        self.steps = 0
        self.max_episode_steps = self.height*self.width
        self.penalty = -1. / self.max_episode_steps

    def observation(self, observation):
        pos = self.unwrapped.agent_pos
        index = (pos[0] - 1) * self.width + (pos[1] - 1)
        assert 0 <= index <= self.height * self.width - 1, 'agent pos:{}, index:{}'.format(pos, index)
        obs = np.zeros(self.height * self.width)
        obs[index] = 1
        return obs

    def reset(self, **kwargs):
        self.steps = 0
        return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        self.steps += 1
        if self.steps > self.max_episode_steps:
            raise ValueError('{} should be less {}'.format(self.steps, self.max_episode_steps))
        state, *_, info = self.env.step(action)
        info['step'] = self.steps
        agent_pos = self.unwrapped.agent_pos
        if agent_pos[0] == self.width and agent_pos[1] == self.height:
            reward = 1.
        else:
            reward = self.penalty
        if self.steps == self.max_episode_steps:
            done = True
        else:
            done = False
        return self.observation(state), reward, done, info


class CoordinateStateWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2, ), dtype=np.float32)
        self.scale = np.sqrt(self.env.height * self.env.width)

    def observation(self, observation):
        pos = np.array(self.env.agent_pos, dtype=np.float32)
        pos /= self.scale
        return pos


class OnehotStateWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.height = self.env.height - 2
        self.width = self.env.width - 2
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.height*self.width, ), dtype=np.float32)

    def observation(self, observation):
        pos = self.env.agent_pos
        index = (pos[0] - 1) * self.width + (pos[1] - 1)
        assert 0 <= index <= self.height * self.width - 1, 'agent pos:{}, index:{}'.format(pos, index)
        obs = np.zeros(self.height*self.width)
        obs[index] = 1
        return obs

def make_gridworld(env_id, infinite_horizon=False, time_penalty=True, state_wrapper=None):
    env = gym.make(env_id)
    env = OnehotStateWrapper(env)
    env = ActionWrappers(env, action_list=['left', 'right', 'forward'])
    if infinite_horizon:
        env = SimpleActionWrapper(env, no_op=True)
    else:
        env = SimpleActionWrapper(env, no_op=False)

    if infinite_horizon:
        env = InfiniteWorld(env)
    else:
        max_episode_steps = env.unwrapped.height * env.unwrapped.width
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    if time_penalty:
        env = TimePenaltyWrapper(env)
    if state_wrapper is not None:
        assert 0
        if state_wrapper == 'coordinate':
            env = CoordinateStateWrapper(env)
        elif state_wrapper == 'onehot':
            env = OnehotStateWrapper(env)
        else:
            raise NotImplementedError
    return env


def test():
    env = make_gridworld('MiniGrid-Empty-16x16-v0')
    state = env.reset()
    print(state)
    for _ in range(10):
        act = env.action_space.sample()
        state, reward, done, info = env.step(act)
        print(state, act)


if __name__ == '__main__':
    test()
