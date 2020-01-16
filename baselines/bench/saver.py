import gym
import pickle


class MujocoSaver(gym.core.Wrapper):
    def __init__(self, env, savepath):
        self.savepath = savepath
        super().__init__(env)

        self.list_variables = []
        self.nb_traj = 0

    def _save_variables(self):
        variables = self.unwrapped.get_save_variables()
        self.list_variables.append(variables)

    def reset(self, **kwargs):
        self.list_variables = []
        obs = self.env.reset(**kwargs)
        self._save_variables()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._save_variables()
        if done:
            self._dump()
        return obs, reward, done, info

    def _dump(self):
        self.nb_traj += 1
        with open(self.savepath, "ab+") as f:
            pickle.dump(self.list_variables, f)
            if self.nb_traj % 100 == 0 or self.nb_traj == 1:
                print('MuJoCoSaver save {} trajectory into :{}'.format(self.nb_traj, self.savepath))
