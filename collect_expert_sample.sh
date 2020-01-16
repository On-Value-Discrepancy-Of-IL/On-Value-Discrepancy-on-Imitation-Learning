#!/usr/bin/env bash

env="Swimmer-v2"
python -m baselines.run \
    --alg ppo2 \
    --env ${env} \
    --num_timesteps 0 \
    --load_path dataset/Swimmer-v2/checkpoints/10000000.model \
    --max_traj 100 \
    --save_interval 100 \
    --save_samples 1 \
    --seed 2019 \
    --collect_sample 1 \
