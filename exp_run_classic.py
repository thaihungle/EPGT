import os
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="testing atari")
    parser.add_argument("--use_mem", default=1, type=int,
                        help="mem script or not")
    parser.add_argument("--opt-type", default="lr", type=str,
                        help="mem script or not")
    parser.add_argument("--env-name", default="BipedalWalker-v3", type=str,
                        help="mem script or not")
    parser.add_argument("--run_id", default="", type=str,
                        help="mem script or not")
    parser.add_argument("--num_run", default=5, type=int,
                        help="no of eval")
    parser.add_argument("--num-env-steps", default=4000000, type=int,
                        help="no of eval")
    parser.add_argument("--num-processes", default=128, type=int,
                        help="no of eval")
    parser.add_argument("--context_train", default=1, type=int,
                        help="no of eval")
    parser.add_argument("--k", type=int, default=5,
                        help="num neighbor")
    parser.add_argument("--k_write", type=int, default=3,
                        help="num neighbor")
    parser.add_argument("--write_interval", type=int, default=10,
                        help="interval for memory writing")
    parser.add_argument("--read_interval", type=int, default=1,
                        help="interval for memory writing")
    parser.add_argument("--quan_level", type=int, default=4,
                        help="number of quantizations for discrete actions")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="number of steps")
    parser.add_argument("--episode_size", type=int, default=1,
                        help="size of a learning episode")
    parser.add_argument("--memory_size", type=int, default=5000,
                    help="size of memory")
    parser.add_argument("--context_order", type=int, default=2,
                    help="order")
    parser.add_argument("--opt-scale", type=float, default=2.0,
                        help="order")
    parser.add_argument("--plot", type=str, default="",
                        help="order")
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')


    args  = parser.parse_args()
   
    if args.use_mem==1:
        print("EPSOPT AGENT")

        for i in range(args.num_run):
            os.system(f"python main.py --env-name {args.env_name} --model-name cc --algo {args.algo} --lr {args.lr}  "
                      f"--use-gae --log-interval 1 --num-steps {args.num_steps} --num-avg-reward 100 --opt-scale {args.opt_scale}  "
                      f"--num-processes {args.num_processes} --num-env-steps {args.num_env_steps} "
                      f"--episode_size {args.episode_size} --context_order {args.context_order} "
                      f"--use_mem 1 --adaptive-opt 1 --memory_size {args.memory_size} "
                      f"--k {args.k} --k_write {args.k_write} --read_interval {args.read_interval} "
                      f"--write_interval {args.write_interval} --seed {i*10} --quan_level {args.quan_level} "
                      f"--opt-type {args.opt_type}  --context_train {args.context_train}  --run-id r{args.run_id}{i}")
    
    else:
        print("NORM AGENT")

        for i in range(args.num_run):
            os.system(f"python main.py --env-name {args.env_name} --model-name cc --algo {args.algo} --lr {args.lr} "
                      f"--use-gae --log-interval 1 --num-steps {args.num_steps} --num-avg-reward 100 --opt-scale {args.opt_scale}  "
                      f"--num-processes {args.num_processes} --num-env-steps {args.num_env_steps} "
                      f"--use_mem 0 --adaptive-opt 0 --context_train {args.context_train} "
                      f"--opt-type {args.opt_type} --run-id r{args.run_id}{i} --seed {i*10} ")