import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from a2c_ppo_acktr.algo.kfac import KFACOptimizer
from .. import epiopt

class A2C_ACKTR(nn.Module):
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 args = None):

        super(A2C_ACKTR, self).__init__()


        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.last_returns = deque(maxlen=5)
        self.last_action_losses = deque(maxlen=200)
        self.last_value_losses = deque(maxlen=200)
        self.last_adv = deque(maxlen=200)
        self.last_kls = deque(maxlen=200)


        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            if args.optimizer == "sgd":
                 self.optimizer = optim.SGD(actor_critic.parameters(), lr=lr)
            else:
                self.optimizer = optim.RMSprop(
                    actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.args = args
        if args.use_mem:
            self.epi_opt = epiopt.EPIOPT(actor_critic, args)

    def update(self, rollouts, update_step=0, env_step=0):
        if self.args.use_mem:
            self.epi_opt.take_action(update_step, env_step)
            self.epi_opt.add2buffer(0, update_step)


        if self.args.use_mem:
            self.epi_opt.insert2mem(torch.mean(rollouts.returns), update_step=update_step)

        self.last_returns.append(torch.mean(rollouts.returns))


        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()


        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))



        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        nps = num_processes
        ro = rollouts.returns[:-1]

        if self.args.use_mem and self.args.adaptive_opt > 0 and 'np' in self.epi_opt.opt_values:
            nps = nps * self.epi_opt.opt_values['np']
            nps = max(1,int(nps))
            action_log_probs = action_log_probs[:, :nps, :]
            values = values[:,:nps,:]
            ro = rollouts.returns[:-1][:,:nps,:]

        advantages = ro - values


        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.last_value_losses.append(value_loss.item())
        self.last_action_losses.append(action_loss.item())




        if self.acktr:
            if self.args.adaptive_opt > 0 and 'Ts' in self.epi_opt.opt_values:
                self.optimizer.Ts = self.epi_opt.opt_values['Ts']
            if self.args.adaptive_opt > 0 and 'klc' in self.epi_opt.opt_values:
                self.optimizer.kl_clip = self.epi_opt.opt_values['klc']
            if self.args.adaptive_opt > 0 and 'mnu' in self.epi_opt.opt_values:
                self.optimizer.max_nu = self.epi_opt.opt_values['mnu']
            if self.args.adaptive_opt > 0 and 'Tf' in self.epi_opt.opt_values:
                self.optimizer.Tf = self.epi_opt.opt_values['Tf']
            if self.args.adaptive_opt > 0 and 'damp' in self.epi_opt.opt_values:
                self.optimizer.kl_clip = 1e-2*self.epi_opt.opt_values['damp']
            if self.args.adaptive_opt > 0 and 'kwdc' in self.epi_opt.opt_values:
                self.optimizer.weight_decay = self.epi_opt.opt_values['kwdc']



        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss

            fic = 1
            if self.args and self.args.adaptive_opt > 0 and 'fic' in self.epi_opt.opt_values:
                fic = fic * self.epi_opt.opt_values['fic']
            fisher_loss = fisher_loss*fic
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        if self.args and self.args.use_mem:


            if update_step > 0:
                self.epi_opt.compute_last_grad()



        lrs = []
        if self.args and self.args.adaptive_opt > 0 and \
                ('lr' in self.epi_opt.opt_values or 'lrs' in self.epi_opt.opt_values or 'kmom' in self.epi_opt.opt_values):
            pg = self.optimizer.param_groups
            if self.acktr:
                old_lr = self.optimizer.lr
                pg = self.optimizer.optim.param_groups
                if 'lrs' in self.epi_opt.opt_values:
                    self.optimizer.lr *= self.epi_opt.opt_values['lrs']
                elif 'lr' in self.epi_opt.opt_values:
                    self.optimizer.lr *= self.epi_opt.opt_values['lr']

            for param_group in pg:
                lrs.append(param_group['lr'])
                if 'lrs' in self.epi_opt.opt_values:
                    param_group['lr'] *= self.epi_opt.opt_values['lrs']
                elif 'lr' in self.epi_opt.opt_values:
                    param_group['lr'] *= self.epi_opt.opt_values['lr']

                if self.args.adaptive_opt > 0 and 'kmom' in self.epi_opt.opt_values:
                    param_group['momentum'] = self.epi_opt.opt_values['kmom']




        self.optimizer.zero_grad()

        value_loss_coef = self.value_loss_coef
        if self.args and self.args.adaptive_opt > 0 and 'vlc' in self.epi_opt.opt_values:
            value_loss_coef = self.value_loss_coef * self.epi_opt.opt_values['vlc']

        entropy_coef = self.entropy_coef
        if self.args and self.args.adaptive_opt > 0 and 'enc' in self.epi_opt.opt_values:
            entropy_coef = self.entropy_coef * self.epi_opt.opt_values['enc']



        (value_loss * value_loss_coef + action_loss -
         dist_entropy * entropy_coef).backward()

        max_grad_norm = self.max_grad_norm
        if self.args and self.args.adaptive_opt > 0 and 'mgn' in self.epi_opt.opt_values:
            max_grad_norm = self.max_grad_norm * self.epi_opt.opt_values['mgn']

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     max_grad_norm)

        self.optimizer.step()

        if self.args and self.args.adaptive_opt > 0 and\
                ('lr' in self.epi_opt.opt_values or 'lrs' in self.epi_opt.opt_values):
            pi = 0
            pg = self.optimizer.param_groups
            if self.acktr:
                pg = self.optimizer.optim.param_groups
                self.optimizer.lr = old_lr

            for param_group in pg:
                param_group['lr'] = lrs[pi]
                pi += 1

        return value_loss.item(), action_loss.item(), dist_entropy.item()
