#!/usr/bin/env bash

env="Ant-v2" 
task="eval"
	python -m opl.imitation_learning.BehavioralClone \
        --env_id ${env} \
        --BC_max_iter 100000 \
	--task ${task}

