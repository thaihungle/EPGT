import os
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="testing atari")
    parser.add_argument("--use_mem", default=1, type=int,
                        help="mem script or not")
    parser.add_argument("--opt-type", default="clip-vlc", type=str,
                        help="mem script or not")
    parser.add_argument("--env-name", default="AlienNoFrameskip-v4", type=str,
                        help="mem script or not")
    parser.add_argument("--run_id", default="", type=str,
                        help="mem script or not")
    parser.add_argument("--num_run", default=3, type=int,
                        help="no of eval")
    parser.add_argument("--num-env-steps", default=10000000, type=int,
                        help="no of eval")
    parser.add_argument("--num-processes", default=32, type=int,
                        help="no of eval")
    parser.add_argument("--context_train", default=1, type=int,
                        help="no of eval")
    parser.add_argument("--k", type=int, default=5,
                        help="num neighbor")
    parser.add_argument("--k_write", type=int, default=3,
                        help="num neighbor")
    parser.add_argument("--write_interval", type=int, default=30,
                        help="interval for memory writing")
    parser.add_argument("--read_interval", type=int, default=1,
                        help="interval for memory writing")
    parser.add_argument("--quan_level", type=int, default=2,
                        help="number of quantizations for discrete actions")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="T")


    args  = parser.parse_args()
    if  args.use_mem==1:
        print("EPSOPT AGENT")

        for i in range(args.num_run):
            os.system(f"python main.py --env-name {args.env_name} --model-name atari --algo ppo "
                      f"--use-gae --log-interval 1 --num-steps {args.num_steps} "
                      f"--num-processes {args.num_processes}  --lr 2.5e-4 --entropy-coef 0.01 "
                      f"--value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 "
                      f"--gamma 0.99 --gae-lambda 0.95 --num-env-steps {args.num_env_steps} "
                      f"--use-linear-lr-decay --use-proper-time-limits --episode_size 3 "
                      f"--use_mem 1 --adaptive-opt 1 --memory_size 5000 "
                      f"--k {args.k} --k_write {args.k_write} --read_interval {args.read_interval} "
                      f"--write_interval {args.write_interval} --quan_level {args.quan_level} "
                      f"--opt-type {args.opt_type} --run-id r{args.run_id}{i}")
    else:
        print("NORM AGENT")

        for i in range(args.num_run):
            os.system(f"python main.py --env-name {args.env_name} --model-name atari --algo ppo "
                      f"--use-gae --log-interval 1 --num-steps {args.num_steps} "
                      f"--num-processes {args.num_processes}  --lr 2.5e-4 --entropy-coef 0.01 "
                      f"--value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 "
                      f"--gamma 0.99 --gae-lambda 0.95 --num-env-steps {args.num_env_steps} "
                      f"--use-linear-lr-decay --use-proper-time-limits "
                      f"--use_mem 0 --adaptive-opt 0 --context_train {args.context_train} "
                      f"--opt-type {args.opt_type} --run-id r{args.run_id}{i}")