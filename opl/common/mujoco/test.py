import os
import os.path as osp
import numpy as np
import argparse

from opl.common.mujoco import build_env
from opl.common.dataset import MuJoCoDataset
from opl.common.utils import timeit


class Transition(object):
    def __init__(self, dataset):
        self.dataset = dataset

        self.index = 0

        self.step_iter = 0

    def step(self, state, action):
        state2 = self.dataset.states2[self.index][self.step_iter]
        self.step_iter += 1
        return state2

    def reset(self, index):
        self.index = index
        self.step_iter = 0
        initial_state = self.dataset.states[self.index][0]
        return initial_state


def test_atom(ref, real, test_type, traj_id, step_iter):
    try:
        rtol = 1e-3  # reward from origin env is computed based on float64, but here we recompute based on float32
        np.testing.assert_allclose(actual=real, desired=ref, rtol=rtol,)
    except AssertionError:
        print('{}, traj:{}, iter:{}'.format(test_type, traj_id, step_iter))
        print('ref:{}, real:{}'.format(ref, real))
        assert 0


@timeit
def test(env_id):
    datapath = osp.join('dataset', env_id)
    dataset = MuJoCoDataset(data_path=datapath, env_id=env_id)
    transition = Transition(dataset)

    env = build_env(env_id, transition_model=transition)

    nb_traj = len(dataset.states)
    for traj_id in range(nb_traj):
        initial_state = transition.reset(traj_id)
        env.reset_state(initial_state)
        obs = env._get_obs()
        assert np.array_equal(obs, dataset.obs[traj_id][0])
        traj_length = len(dataset.states[traj_id])
        print('*****Test {}/{}, Length:{}********'.format(traj_id, nb_traj, traj_length))
        for step_iter in range(traj_length-1):
            action = dataset.acs[traj_id][step_iter]
            obs, reward, done, info = env.step(action)
            test_atom(ref=dataset.obs[traj_id][step_iter+1], real=obs, test_type='obs', traj_id=traj_id, step_iter=step_iter)
            test_atom(ref=dataset.rs[traj_id][step_iter], real=reward, test_type='reward', traj_id=traj_id, step_iter=step_iter)
            test_atom(ref=dataset.dones[traj_id][step_iter], real=done, test_type='done', traj_id=traj_id, step_iter=step_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    args = parser.parse_args()
    test(env_id=args.env)
