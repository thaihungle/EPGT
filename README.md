# EPGT

source code for Episodic Policy Gradient Training   
arXiv version: https://arxiv.org/abs/2112.01853 
code reference https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

# Setup  
```
pip install -r requirements.txt
mkdir logs
mkdir saved_models
```

# Mujoco tasks
run command examples for HalfCheetah
``` 
Baseline PPO: python exp_run_mjc.py --use_mem 0 --env_name HalfCheetah-v2 
EPGT + PPO: python exp_run_mjc.py --use_mem 1 --env_name HalfCheetah-v2 
```

# Atari tasks
run command examples for SpaceInvaders
``` 
Baseline PPO: python exp_run_mjc.py --use_mem 0 --env_name SpaceInvadersNoFrameskip-v4 
EPGT + PPO: python exp_run_mjc.py --use_mem 1 --env_name SpaceInvadersNoFrameskip-v4 
```