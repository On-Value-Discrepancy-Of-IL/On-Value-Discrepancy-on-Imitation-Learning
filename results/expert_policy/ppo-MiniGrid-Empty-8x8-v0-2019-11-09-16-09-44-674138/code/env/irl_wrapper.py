from gym.core import Wrapper
import numpy as np
import gym


class IRLWrapper(Wrapper):
    def __init__(self, env, w=None, feature_fn='linear', action_feature=False):
        super().__init__(env)
        self._w = w
        if isinstance(feature_fn, str):
            self.feature_fn = {
                'linear': lambda x: x,
                'quadratic': lambda x: x**2,
            }[feature_fn]
        else:
            self.feature_fn = feature_fn
            assert callable(self.feature_fn)
        self.action_feature = action_feature
        feature_dim = self._get_space_dim(env.observation_space)
        if action_feature:
            feature_dim += self._get_space_dim(env.action_space)
        if self._w is None:
            self._w = np.random.uniform(-1, 1, feature_dim)
            self._w /= np.linalg.norm(self._w)
        assert self._w.shape == (feature_dim, )

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return state

    def step(self, action):
        state, true_reward, done, info = self.env.step(action)

        info['true_reward'] = true_reward
        info['feature'] = self._feature(state, action)
        reward = self.reward(state, action)

        return state, reward, done, info

    def reward(self, state, action):
        feature = self._feature(state, action)
        reward = np.dot(feature, self._w)
        return reward

    def state_preprocess(self, state, scale=1.):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            _state = np.zeros(self.env.action_space.n)
            _state[state] = 1
        else:
            _state = state
        _state /= scale
        _state = self.feature_fn(_state)
        return _state

    def action_preprocess(self, action, scale=1.):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            _action = np.zeros(self.env.action_space.n)
            _action[action] = 1
        else:
            _action = action
        _action /= scale
        _action = self.feature_fn(_action)
        return _action
    
    def set_reward_weight(self, w):
        self._w = w

    def get_reward_weight(self):
        return self._w.copy()

    def _feature(self, state, action):
        feature = [self.state_preprocess(state)]
        if self.action_feature:
            feature.append(self.action_preprocess(action))
        return np.array(feature).flatten()
    
    @staticmethod
    def _get_space_dim(space):
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return space.shape[0]
        else:
            raise NotImplementedError

