#!/usr/bin/env bash

env="Ant-v2"
python -m baselines.run \
    --alg ppo2 \
    --env ${env} \
    --num_timesteps 1e5 \
    --num_env 2 \
    --ent_coef 0.01 \
    --save_interval 100 \
    --save_samples 1 \
    --seed 2019 \
    --collect_sample 1
