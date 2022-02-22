import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--model-name', default='', help='model name')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=-1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--num-avg-reward',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='MountainCar-v0',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--run-id',
        default='0',
        help='run id')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=True,
        help='use a linear schedule on the learning rate')

    # episodic part
    parser.add_argument("--use_mem", type=int, default=1,
                        help="use episodic optimizer")
    parser.add_argument('--adaptive-opt',type=float, default=0,
                        help='enable adaptive optimization')
    parser.add_argument('--opt-type', type=str, default='klc-mnu',
                        help='enable adaptive optimization')
    parser.add_argument('--opt-scale', type=float, default=1.0,
                        help='enable adaptive optimization')
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="type of optimizer")
    parser.add_argument("--episode_size", type=int, default=1,
                        help="number of updates per episode")
    parser.add_argument("--context_size", type=int, default=4,
                        help="train context?")
    parser.add_argument("--context_order", type=int, default=2,
                        help="use what context")
    parser.add_argument("--context_train", type=int, default=1,
                        help="train context?")
    parser.add_argument("--context_train_limit", type=int, default=1000000,
                        help="train context?")
    parser.add_argument("--c_bs", type=int, default=8,
                        help="context bacth size")
    parser.add_argument("--ct_rate", type=float, default=0.1,
                        help="context training rate")
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="RNN hidden")
    parser.add_argument("--mem_dim", type=int, default=5,
                        help="memory dimesntion")
    parser.add_argument("--memory_size", type=int, default=1000,
                        help="memory size")
    parser.add_argument("--memory_start", type=int, default=0,
                        help="when start reading memory")
    parser.add_argument("--k", type=int, default=5,
                        help="num neighbor")
    parser.add_argument("--k_write", type=int, default=3,
                        help="num neighbor")
    parser.add_argument("--max_mode", type=int, default=0,
                        help="writing max mode")
    parser.add_argument("--write_interval", type=int, default=10,
                        help="interval for memory writing")
    parser.add_argument("--read_interval", type=int, default=1,
                        help="interval for memory writing")
    parser.add_argument("--write_lr", type=float, default=.5,
                        help="learning rate of writing")
    parser.add_argument("--num_ecw", type=int, default=10,
                        help="number of weights for context")
    parser.add_argument("--quan_level", type=int, default=5,
                        help="number of quantizations for discrete actions")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="exploration start")
    parser.add_argument("--epsilon_final", type=float, default=0.01,
                        help="exploration final")
    parser.add_argument("--epsilon_decay", type=float, default=1000.0,
                        help="exploration decay")
    parser.add_argument("--plot", type=str, default="",
                        help="plot type")
    parser.add_argument("--plot_interval", type=int, default=1000,
                        help="plot type")





    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    if args.use_mem:
        args.adaptive_opt = 1
        if args.use_mem==3:
            args.write_interval = 1

    return args
