import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .. import epiopt

class PPO(nn.Module):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True, args=None):
        super(PPO, self).__init__()

        self.args=args
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # if args.optimizer == "adam":
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        if args.optimizer == "sgd":
            self.optimizer = optim.SGD(actor_critic.parameters(), lr=lr)

        self.last_returns = deque(maxlen=5)
        self.last_action_losses = deque(maxlen=200)
        self.last_value_losses = deque(maxlen=200)

        self.args = args
        if args.use_mem:
            self.epi_opt = epiopt.EPIOPT(actor_critic, args)


    def update(self, rollouts, update_step=0, env_step=0):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        prev_loss = 0
        gstep = 0
        R = 0
        # print(len(self.traj_buffer), self.dnd.get_mem_size())
        # print(self.traj_buffer)

       
        if self.args:
            self.epi_opt.insert2mem(torch.mean(rollouts.returns), update_step=update_step, gstep=-1)


        self.last_returns.append(torch.mean(rollouts.returns))



        for e in range(self.ppo_epoch):


            num_mini_batch = self.num_mini_batch

            if self.args.adaptive_opt>0 and 'mb' in self.epi_opt.opt_values:
                num_mini_batch = max(int(self.epi_opt.opt_values['mb']*self.num_mini_batch),1)


            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, num_mini_batch)


            for sample in data_generator:


                if self.args:
                    self.epi_opt.take_action(update_step, env_step, gstep)

                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                nps = self.args.num_processes*self.args.num_steps//self.num_mini_batch

                if self.args and self.args.adaptive_opt > 0 and 'np' in self.epi_opt.opt_values:
                    nps = nps * self.epi_opt.opt_values['np']
                    nps = max(1, int(nps))

                    action_log_probs = action_log_probs[:nps]
                    old_action_log_probs_batch = old_action_log_probs_batch[:nps]
                    values = values[:nps]
                    advantages = advantages[:nps]
                    adv_targ = adv_targ[:nps]
                    value_preds_batch = value_preds_batch[:nps]
                    return_batch = return_batch[:nps]


                approx_kl = 0.5 * torch.mean((old_action_log_probs_batch - action_log_probs) ** 2)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                if self.args.adaptive_opt > 0 and 'clip' in self.epi_opt.opt_values:
                    surr2 = torch.clamp(ratio, 1.0 -  self.epi_opt.opt_values['clip'],
                                        1.0 +  self.epi_opt.opt_values['clip']) * adv_targ


               
                action_loss = -torch.min(surr1, surr2).mean()
                  
                if self.args and gstep > 0:
                    if action_loss < prev_loss:
                        epo_reward = 0
                    else:
                        epo_reward = 0

                    self.epi_opt.add2buffer(epo_reward, gstep)

                prev_loss = action_loss
                self.last_action_losses.append(action_loss.item())

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.last_value_losses.append(value_loss.item())


                if self.args and self.args.use_mem:
                    if gstep>0:
                        self.epi_opt.compute_last_grad()

                lrs = []
                if self.args and self.args.adaptive_opt > 0 and 'lr' in self.epi_opt.opt_values:
                    for param_group in self.optimizer.param_groups:
                        lrs.append(param_group['lr'])
                        param_group['lr'] *= self.epi_opt.opt_values['lr']

                self.optimizer.zero_grad()

                value_loss_coef = self.value_loss_coef
                if self.args.adaptive_opt > 0 and 'vlc' in self.epi_opt.opt_values:
                    value_loss_coef = self.value_loss_coef*self.epi_opt.opt_values['vlc']

                entropy_coef = self.entropy_coef
                if self.args.adaptive_opt > 0 and 'enc' in self.epi_opt.opt_values:
                    entropy_coef = self.entropy_coef*self.epi_opt.opt_values['enc']



                (value_loss * value_loss_coef + action_loss -
                 dist_entropy * entropy_coef).backward()

                max_grad_norm = self.max_grad_norm
                if self.args.adaptive_opt > 0 and 'mgn' in self.epi_opt.opt_values:
                    max_grad_norm = self.max_grad_norm * self.epi_opt.opt_values['mgn']

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         max_grad_norm)
                self.optimizer.step()
                gstep += 1
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                if self.args and self.args.adaptive_opt > 0 and 'lr' in self.epi_opt.opt_values:
                    pi = 0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lrs[pi]
                        pi+=1


        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
