from opl.common.mujoco.ant import AntEnv
from opl.common.mujoco.half_cheetah import HalfCheetahEnv
from opl.common.mujoco.hopper import HopperEnv
from opl.common.mujoco.swimmer import SwimmerEnv
from opl.common.mujoco.walker2d import Walker2dEnv


def build_env(env_id, transition_model):
    if env_id == 'Ant-v2':
        return AntEnv(transition_model)
    elif env_id == 'HalfCheetah-v2':
        return HalfCheetahEnv(transition_model)
    elif env_id == 'Hopper-v2':
        return HopperEnv(transition_model)
    elif env_id == 'Swimmer-v2':
        return SwimmerEnv(transition_model)
    elif env_id == 'Walker2d-v2':
        return Walker2dEnv(transition_model)
    else:
        raise NotImplementedError('{} is not supported now!'.format(env_id))
