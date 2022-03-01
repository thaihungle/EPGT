import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from tensorboard_logger import configure, log_value
from datetime import datetime
import time
from tqdm import tqdm



def main():
    args = get_args()

    if args.seed<0:
        args.seed = int(time.time())%10000

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    if not args.run_id:
        # datetime object containing current date and time
        now = datetime.now()

        print("now =", now)

        # dd/mm/YY H:M:S
        args.run_id = now.strftime("%d-%m-%Y-%H-%M-%S")


    log_dir = f"{log_dir}{args.env_name}_{args.algo}X/{args.env_name[:4]}{args.model_name}_ct{args.adaptive_opt}{args.use_mem}ot{args.opt_type}cl{args.clip_param}" \
        f"bz{args.num_mini_batch}st{args.num_steps}ps{args.num_processes}" \
        f"pe{args.ppo_epoch}lr{args.lr}-{args.run_id}"
    if args.use_mem:
        log_dir = f"{log_dir}kr{args.k}kw{args.k_write}ql{args.quan_level}os{args.opt_scale}" \
            f"co{args.context_order}es{args.episode_size}cz{args.context_size}ct{args.context_train}" \
            f"ne{args.num_ecw}ms{args.memory_size}" \
            f"ri{args.read_interval}wi{args.write_interval}hz{args.hidden_size}mz{args.mem_dim}"
    log_dir = f"{log_dir}s{args.seed}"

    plotdir = f"./plot/{args.env_name}_{args.algo}X/{args.env_name[:4]}{args.model_name}_ct{args.adaptive_opt}{args.use_mem}ot{args.opt_type}cl{args.clip_param}" \
        f"bz{args.num_mini_batch}st{args.num_steps}ps{args.num_processes}" \
        f"pe{args.ppo_epoch}lr{args.lr}-{args.run_id}"

    if args.use_mem and args.plot:
        plotdir = f"{plotdir}kr{args.k}kw{args.k_write}ql{args.quan_level}os{args.opt_scale}" \
            f"co{args.context_order}es{args.episode_size}cz{args.context_size}ct{args.context_train}" \
            f"ne{args.num_ecw}ms{args.memory_size}" \
            f"ri{args.read_interval}wi{args.write_interval}hz{args.hidden_size}mz{args.mem_dim}"
        plotdir = f"{plotdir}s{args.seed}"


        os.makedirs(plotdir, exist_ok=True)
        print(plotdir)

        args.plotdir = plotdir


    print(log_dir)
    configure(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")


    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm, args=args)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm, args=args)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True, args=args)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    agent.to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.num_avg_reward)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    pbar = tqdm(total=args.num_env_steps)

    j = total_num_steps=0
    while total_num_steps<args.num_env_steps:
    # for j in tqdm(range(num_updates)):


        num_steps = args.num_steps
        if agent.args.adaptive_opt>0 and 'nstep' in agent.epi_opt.opt_values:
            num_steps = max(agent.epi_opt.opt_values['nstep']*args.num_steps,1)
            num_steps =int(num_steps)
        
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action.cpu())

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        gamma = args.gamma

        if agent.args.adaptive_opt>0 and 'gamma' in agent.epi_opt.opt_values:
            gamma = agent.epi_opt.opt_values['gamma']

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], gamma,
                    rollouts.masks[step])

        gae_lambda = args.gae_lambda
        if agent.args.adaptive_opt>0 and 'gae' in agent.epi_opt.opt_values:
            gae_lambda = agent.epi_opt.opt_values['gae']

        rollouts.compute_returns(next_value, args.use_gae, gamma,
                                 gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, update_step=j, env_step = total_num_steps)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))
        # total_num_steps = (j + 1) * args.num_processes * args.num_steps

        num_processes = args.num_processes
        if agent.args.adaptive_opt>0 and 'np' in agent.epi_opt.opt_values:
            num_processes = args.num_processes * agent.epi_opt.opt_values['np']
            num_processes = max(1, int(num_processes))
        total_num_steps  += num_processes * num_steps
        pbar.update(num_processes * num_steps)

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer.optim if args.algo == "acktr" else agent.optimizer, total_num_steps, args.num_env_steps,
                agent.optimizer.lr if args.algo == "acktr" else args.lr, num_processes * num_steps)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:

            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            log_value('Eps. Reward/train', np.mean(np.mean(episode_rewards)), total_num_steps)
            log_value('Return/train', torch.mean(rollouts.returns), total_num_steps)

            log_value('Loss/Action loss', np.mean(list(agent.last_action_losses)[-50:]), total_num_steps)
            log_value('Loss/Value loss', np.mean(list(agent.last_value_losses)[-50:]), total_num_steps)
            if agent.args.adaptive_opt>0:
                for key in agent.epi_opt.opt_values.keys():
                    log_value(f'Mem/{key}', np.mean(list(agent.epi_opt.last_opt_values[key])[-3:]), total_num_steps)

            if agent.args.use_mem:
                log_value('Mem/size', agent.epi_opt.dnd.get_mem_size(), total_num_steps)

                if agent.args.context_train==1:
                    log_value('Mem/vael', np.mean(list(agent.epi_opt.vae_loss)[-50:]), total_num_steps)
            else:
                if args.algo!='acktr':
                    log_value('Train/lr', agent.optimizer.param_groups[0]['lr'], total_num_steps)
            # print(agent.optimizer.optim.param_groups[0]['lr'])

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            try:
                obs_rms = utils.get_vec_normalize(envs).obs_rms
            except:
                obs_rms = None
            evaluate(total_num_steps, actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        j+=1
    pbar.close()


if __name__ == "__main__":
    main()
