python -m baselines.run \
	--alg=her \
    --env=FetchReach-v4 \
	--num_timesteps=5e4 \
	--num_env 2 \
	--seed 2019 \
	--num_exp 1	\
    --replay_k 0.0