import numpy as np
import pickle
import os
import os.path as osp

from baselines import logger
from opl.common.utils import timeit


class Dset(object):
    def __init__(self, *, obs, acs, rs, dones, mus, states, states2, randomize):
        self.obs = obs
        self.acs = acs
        self.rs = rs
        self.dones = dones
        self.mus = mus
        self.states = states
        self.states2 = states2

        self.randomize = randomize
        self.num_pairs = len(obs)
        self.pointer = 0
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.obs = self.obs[idx, :]
            self.acs = self.acs[idx, :]
            self.rs = self.rs[idx]
            self.dones = self.dones[idx]
            self.mus = self.mus[idx]
            if self.states is not None:
                self.states = self.states[idx, :]
                self.states2 = self.states2[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            res = self.obs, self.acs, self.rs, self.dones, self.mus
            return res
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        obs = self.obs[self.pointer:end, :]
        acs = self.acs[self.pointer:end, :]
        rs = self.rs[self.pointer:end]
        dones = self.dones[self.pointer:end]
        mus = self.mus[self.pointer:end]
        states = self.states[self.pointer:end, :] if self.states is not None else None
        states2 = self.states2[self.pointer:end, :] if self.states2 is not None else None
        
        self.pointer = end

        return obs, acs, rs, dones, mus, states, states2


class MuJoCoDataset(object):
    def __init__(self, data_path, env_id=None, load_states=True, traj_limit=None, train_fraction=0.7, randomize=True, seed=2019):
        self.env_id = env_id or data_path.split('/')[-1].split('-')[1]
        self.saver_path = osp.join(data_path, "mujoco_saver")
        self.runner_path = osp.join(data_path, "runner")
        logger.info('Load {} from: {}'.format(env_id, data_path))
        assert osp.exists(self.saver_path), '{} does not exist'.format(self.saver_path)
        assert osp.exists(self.runner_path), '{} does not exist'.format(self.runner_path)
        np.random.seed(seed)

        self.randomize = randomize
        self.load_states = load_states
        self.traj_limit = traj_limit or np.inf
        # DataFormat: pkl. Each item is a dict
        # Each dict: obs: (T+1, n_ob); Acs: (T, n_ac); R:(T,); Done:(T, ); Mu:(T, )

        self.ret = []
        self.obs, self.acs, self.rs, self.dones, self.mus = self._load_runner()

        obs = np.concatenate(self.obs, axis=0)
        acs = np.concatenate(self.acs, axis=0)
        rs = np.concatenate(self.rs, axis=0)
        dones = np.concatenate(self.dones, axis=0)
        mus = np.concatenate(self.mus, axis=0)
        if load_states:
            self.states, self.states2 = self._load_saver()
            self._check_data(self.states, self.obs)
            states = np.concatenate(self.states, axis=0)
            states2 = np.concatenate(self.states2, axis=0)
        else:
            self.states, self.states2 = None, None
            states, states2 = None, None
        # self.obs2 = self._get_obs2_from_states2(self.states2)

        assert len(obs) == len(acs) == len(rs) == len(dones) == len(mus) == len(states) == len(states2)
        self.num_transition = len(obs)

        self.dset = Dset(obs=obs, acs=acs, rs=rs, dones=dones, mus=mus, randomize=randomize,
                         states=states if load_states else None,
                         states2=states2 if load_states else None)
        # for behavior cloning
        self.train_set = Dset(obs=obs[:int(self.num_transition*train_fraction), :],
                              acs=acs[:int(self.num_transition*train_fraction), :],
                              rs=rs[:int(self.num_transition*train_fraction), ],
                              dones=dones[:int(self.num_transition*train_fraction), ],
                              mus=mus[:int(self.num_transition*train_fraction), ],
                              states=states[:int(self.num_transition*train_fraction), :] if load_states else None,
                              states2=states2[:int(self.num_transition*train_fraction), :] if load_states else None,
                              randomize=randomize)
        self.val_set = Dset(obs=obs[int(self.num_transition*train_fraction):, :],
                            acs=acs[int(self.num_transition*train_fraction):, :],
                            rs=rs[int(self.num_transition*train_fraction):, ],
                            dones=dones[int(self.num_transition*train_fraction):, ],
                            mus=dones[int(self.num_transition*train_fraction):, ],
                            states=states[int(self.num_transition*train_fraction):, :] if load_states else None,
                            states2=states2[int(self.num_transition*train_fraction):, :] if load_states else None,
                            randomize=randomize)

        logger.log("Total transitions: %d" % len(obs))
        logger.log("Train transitions: %d" % self.train_set.num_pairs)
        logger.log("Valid transitions: %d" % self.val_set.num_pairs)
        logger.log("Average returns: %f" % np.mean(self.ret))
        logger.log("Std for returns: %f" % np.std(self.ret))
        logger.log("Observation shape: {}".format(obs.shape))
        logger.log("Action shape: {}".format(acs.shape))
        if load_states:
            logger.log("State shape: {}".format(states.shape))

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    @timeit
    def _load_saver(self, verbose=True):
        states, states2 = [], []
        for file in sorted(os.listdir(self.saver_path)):
            if file.endswith('.pkl'):
                with open(osp.join(self.saver_path, file), 'rb') as f:
                    nb = 0
                    while True:
                        try:
                            item = pickle.load(f)
                            states.append(item[:-1])
                            states2.append(item[1:])
                            nb += 1
                        except EOFError:
                            break
                        if nb >= self.traj_limit:
                            break
                    if verbose:
                        print('load {} trajectories from {}'.format(nb, osp.join(self.saver_path, file)))
        return states, states2

    @timeit
    def _load_runner(self, verbose=True):
        obs, acs, rs, dones, mus = [], [], [], [], []
        assert os.path.exists(self.runner_path), '{} does not exist'.format(self.runner_path)
        for file in sorted(os.listdir(self.runner_path)):
            if file.endswith('.pkl'):
                with open(osp.join(self.runner_path, file), 'rb') as f:
                    nb = 0
                    while True:
                        try:
                            item = pickle.load(f)[0]
                            obs.append(item['obs'])
                            acs.append(item['acs'])
                            rs.append(item['rs'])
                            self.ret.append(np.sum(item['rs']))
                            dones.append(item['dones'])
                            mus.append(item['mus'])
                            nb += 1
                        except EOFError:
                            break
                        if nb >= self.traj_limit:
                            break
                    if verbose:
                        print('load {} trajectories from {}'.format(nb, osp.join(self.runner_path, file)))
        return obs, acs, rs, dones, mus

    def _check_data(self, states, obs):
        assert isinstance(states, list) and isinstance(obs, list)
        _obs_real = np.concatenate(obs, axis=0).astype(np.float32)
        _states = np.concatenate(states, axis=0).astype(np.float32)
        _obs_ref = self._get_obs_from_states(_states)
        assert _obs_ref.shape == _obs_real.shape, '{}, obs_ref shape:{}, obs_real shape:{}'.format(self.env_id, _obs_ref.shape, _obs_real.shape)
        assert np.array_equal(_obs_ref, _obs_real), '{} obs is not coincident with states'.format(self.env_id)

    def _get_obs_from_states(self, states):
        assert isinstance(states, np.ndarray)
        if self.env_id == 'Ant-v2':
            obs = states[:, 2:-3]
        elif self.env_id == 'HalfCheetah-v2':
            obs = states[:, 1:]
        elif self.env_id == 'Hopper-v2':
            obs = states[:, 1:]
        elif self.env_id == 'Swimmer-v2':
            obs = states[:, 2:]
        elif self.env_id == 'Walker2d-v2':
            obs = states[:, 1:]
        else:
            raise NotImplementedError('{} is not implemented'.format(self.env_id))
        return obs


if __name__ == '__main__':
    for subdir in os.listdir('logs'):
        path = osp.join('logs', subdir)
        if path.endswith('/'):
            path = path[:-1]
        dataset = MuJoCoDataset(path)
        dataset.get_next_batch(64, split='train')

