from gym.envs.registration import register as gym_register

env_list = []

def register(
    id,
    entry_point,
    reward_threshold=0.95,
):
    assert id.startswith("MiniGrid-")
    assert id not in env_list

    # MiniGrid-Empty-5x5-v0
    # height = int(id.split('-')[-2].split('x')[0])
    # width = int(id.split('-')[-2].split('x')[1])
    # Register the environment with OpenAI gym
    gym_register(
        id=id,
        entry_point=entry_point,
        reward_threshold=reward_threshold,
    )

    # Add the environment to the set
    env_list.append(id)
