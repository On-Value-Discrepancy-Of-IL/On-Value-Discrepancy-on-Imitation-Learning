import gym


class MuJoCoWrapper(gym.core.Wrapper):
    def __init__(self, env, transition_model):
        super().__init__(env)
        self.transition_model = transition_model

        self.dt = env.unwrapped.dt
        self.frame_skip = env.unwrapped.frame_skip
        self.dim_qpos = env.unwrapped.sim.data.qpos.shape[0]
        self.ob_shape = self.observation_space.shape

        self.state = None

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError

    def render(self, mode='human', **kwargs):
        raise NotImplementedError

    def reset_state(self, state):
        self.state = state
