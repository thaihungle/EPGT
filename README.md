# EPGT

source code for Episodic Policy Gradient Training        
arXiv version: https://arxiv.org/abs/2112.01853    
AAAI version: https://aaai-2022.virtualchair.net/poster_aaai3607     
code reference for PG algoritms https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail   

# Setup  
```
pip install -r requirements.txt
mkdir logs
mkdir saved_models
```

# Hypaparameter optimization guide
The type of optimized hyperparameter is specificed by short strings (e.g "lr"). Check [here](/a2c_ppo_acktr/epiopt.py#L153) for the name of all supported hyperparameter types, used by different policy gradient algorithms (PPO, A2C, ACKTR). Check each algorithm file [ppo](/a2c_ppo_acktr/algo/ppo.py) or [a2c, acktr](/a2c_ppo_acktr/algo/ppo.py) for the usage of corresponding hyperparameters.

Check [here](/a2c_ppo_acktr/arguments.py) for full list of arguments. Need to set --adaptive-opt=1 and --opt-type="hyperparameter type strings" to enable EPGT. Multi-hyperparameter opt-type is supported (e.g. "lr-clip").  

# Classical tasks
run command examples for BipedalWalker
``` 
Baseline A2C: python exp_run_mjc.py --use_mem 0 --env_name BipedalWalker-v3 
EPGT + A2C: python exp_run_mjc.py --use_mem 1 --env_name BipedalWalker-v3 
```

# Mujoco tasks
run command examples for HalfCheetah
``` 
Baseline PPO: python exp_run_mjc.py --use_mem 0 --env_name HalfCheetah-v2 
EPGT + PPO: python exp_run_mjc.py --use_mem 1 --env_name HalfCheetah-v2 
```

# Atari tasks
run command examples for Qbert
``` 
Baseline ACKTR: python exp_run_mjc.py --use_mem 0 --env_name QbertNoFrameskip-v4 
EPGT + ACKTR: python exp_run_mjc.py --use_mem 1 --env_name QbertNoFrameskip-v4 
```