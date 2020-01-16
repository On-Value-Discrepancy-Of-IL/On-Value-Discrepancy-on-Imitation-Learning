import numpy as np
import pickle
import os
import os.path as osp

from baselines.common.runners import AbstractEnvRunner
from baselines import logger

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, save_samples=False):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

        self.save_samples = save_samples
        self.savedir = osp.join(logger.get_dir(), "runner")
        os.makedirs(self.savedir, exist_ok=True)
        self.ac_space = env.action_space
        self.ob_space = env.observation_space

        create_buffer = lambda: [[] for _ in range(self.nenv)]
        self.obs_save, self.acs_save, self.rs_save, self.dones_save, self.neglogpacs_save = \
            create_buffer(), create_buffer(), create_buffer(), create_buffer(), create_buffer()
        self.nb_traj = 0


    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            if self.save_samples:
                obs = self.obs.copy()
                for e in range(self.nenv):
                    self.obs_save[e].append(obs[e])
                    self.acs_save[e].append(actions[e])
                    self.neglogpacs_save[e].append(-neglogpacs[e])
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for e, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            if self.save_samples:
                dones = self.dones.copy()
                for e in range(self.nenv):
                    self.rs_save[e].append(rewards[e])
                    self.dones_save[e].append(dones[e])
                    if self.dones[e]:
                        self._dump(e)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

    def _dump(self, e):
        assert self.save_samples
        self.nb_traj += 1
        savepath = osp.join(self.savedir, "samples_{}.pkl".format(e))
        with open(savepath, "ab+") as f:
            data = dict(obs=np.asarray(self.obs_save[e], dtype=self.ob_space.dtype),
                        acs=np.asarray(self.acs_save[e], dtype=self.ac_space.dtype),
                        rs=np.asarray(self.rs_save[e], dtype=np.float32),
                        dones=np.asarray(self.dones_save[e], dtype=np.bool),
                        mus=np.exp(-np.asarray(self.neglogpacs_save[e], dtype=np.float32))),
            pickle.dump(data, f)
            if self.nb_traj % 100 == 0 or self.nb_traj == 1:
                logger.info("Runner save {} trajectory samples into:{}".format(self.nb_traj, savepath))
            self.obs_save[e].clear()
            self.acs_save[e].clear()
            self.rs_save[e].clear()
            self.dones_save[e].clear()
            self.neglogpacs_save[e].clear()

# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


