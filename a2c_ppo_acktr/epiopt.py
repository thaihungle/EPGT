import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import math
from collections import deque
import copy
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F

from . import dnd
from . import controller



USE_CUDA = torch.cuda.is_available()

def plot_keys_values(keys, values, step, plotdir):

    X_embedded = TSNE(n_components=2).fit_transform(keys)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=np.mean(values, axis=-1))
    plt.colorbar()
    plt.savefig(f'{plotdir}/kv_{step}.pdf')
    print(f'save plot to {plotdir}/kv_{step}.png')
    plt.close()


class FFWVAE(nn.Module):
    def __init__(self, isize, hsize):
        super(FFWVAE, self).__init__()

        # encoder
        self.enc1 = nn.Linear(
            isize, isize//4
        )
        self.enc2 = nn.Linear(
            isize//4, hsize*2
        )
        # self.enc3 = nn.Linear(
        #     isize//16, isize//32
        # )
        # self.enc4 = nn.Linear(
        #     isize//32, hsize
        # )

        # decoder
        self.dec1 = nn.Linear(
            hsize, isize//4
        )
        self.dec2 = nn.Linear(
            isize//4, isize
        )
        # self.dec3 = nn.Linear(
        #     isize//16, isize//4
        # )
        # self.dec4 = nn.Linear(
        #     isize//4, isize
        # )
        # self.criterion = torch.nn.BCELoss(reduction='sum')
        self.criterion = torch.nn.MSELoss()

        self.hsize = hsize

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        #print(x)
        x = F.tanh(self.enc1(x))
        x = F.tanh(self.enc2(x))
        # get `mu` and `log_var`
        mu, log_var = torch.split(x,self.hsize, dim=-1)


        return mu

    def forward(self, x):
        # encoding
        x = F.tanh(self.enc1(x))
        x = F.tanh(self.enc2(x))
        # get `mu` and `log_var`
        mu, log_var = torch.split(x,self.hsize, dim=-1)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction,z, mu, log_var

    def final_loss(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

def inverse_distance(h, h_i, epsilon=1e-3):
    return 1 / (torch.dist(h, h_i) + epsilon)

def linearly_decaying_epsilon(decay_period, step, warmup_steps=10, final_epsilon=0.01):
  """Returns the current epsilon for the agent's epsilon-greedy policy.
  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    final_epsilon: float, the final value to which to decay the epsilon parameter.
  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - final_epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - final_epsilon)
  return final_epsilon + bonus

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class EPIOPT(nn.Module):
    def __init__(self, actor_critic,
                  args=None):
        super(EPIOPT, self).__init__()

        self.args=args
        self.actor_critic = actor_critic

        self.opt_values = {}

        if 'clip' in args.opt_type:
            self.opt_values['clip'] = 1
        if 'lrs' in args.opt_type:
            self.opt_values['lrs'] = 0
        elif 'lr' in args.opt_type:
            self.opt_values['lr'] = 1
        if 'mb' in args.opt_type:
            self.opt_values['mb'] = 1
        if 'vlc' in args.opt_type:
            self.opt_values['vlc'] = 1
        if 'fic' in args.opt_type:
            self.opt_values['fic'] = 1
        if 'enc' in args.opt_type:
            self.opt_values['enc'] = 1
        if 'mgn' in args.opt_type:
            self.opt_values['mgn'] = 1
        if 'alpha' in args.opt_type:
            self.opt_values['alpha'] = 0.5
        if 'nstep' in args.opt_type:
            self.opt_values['nstep'] = 1
        if 'gae' in args.opt_type:
            self.opt_values['gae'] = 0.95
        if 'gamma' in args.opt_type:
            self.opt_values['gamma'] = 0.99
        if 'np' in args.opt_type:
            self.opt_values['np'] = 1
        if 'Ts' in args.opt_type:
            self.opt_values['Ts'] = 1
        if 'Tf' in args.opt_type:
            self.opt_values['Tf'] = 10
        if 'klc' in args.opt_type:
            self.opt_values['klc'] = 0.001
        if 'mnu' in args.opt_type:
            self.opt_values['mnu'] = 1
        if 'damp' in args.opt_type:
            self.opt_values['damp'] = 1
        if 'kwdc' in args.opt_type:
            self.opt_values['kwdc'] = 0
        if 'kmom' in args.opt_type:
            self.opt_values['kmom'] = 0.9

        print(self.opt_values)


        self.last_opt_values = {}

        for key in self.opt_values.keys():
            self.last_opt_values[key] = deque(maxlen=200)


        if  self.args.adaptive_opt>0:
            self.quan_level = args.quan_level
            self.opt_bins = {}
            self.num_bin = []
            self.num_action = 1

            for key in sorted(self.opt_values.keys()):
                self.opt_bins[key] = []
                if key == 'lrs':
                    self.opt_bins[key] = [1/100, 1/10, 1, 10, 100 ]
                elif key == 'gae':
                    self.opt_bins[key] = [0.9, 0.925,0.95,0.975, 0.99]
                elif key == 'gamma':
                    self.opt_bins[key] =  [0.95, 0.975, 0.99]
                elif key == 'nstep':
                    self.opt_bins[key] =  [1.0, 0.5, 0.25]
                elif key == 'np':
                    self.opt_bins[key] = [1.0, 0.5, 0.25]
                elif key == 'kwdc':
                    self.opt_bins[key] = [0, 0.01, 0.0001]
                elif key == 'Ts':
                    self.opt_bins[key] = [1, 2, 4]
                elif key == 'Tf':
                    self.opt_bins[key] = [10, 5, 20]
                elif key == 'kmom':
                    self.opt_bins[key] = [0.95, 0.9, 0.8]
                elif key == 'mnu':
                    self.opt_bins[key] = [1.0, 1.5, 0.5]
                elif key == 'klc':
                    self.opt_bins[key] = [0.001, 0.002, 0.003]
                elif key == 'clip':
                    self.opt_bins[key] = [0.1, 0.2, 0.3, 0.4, 0.5]
                else:
                    for i in range(self.quan_level):
                        if i==0:
                            self.opt_bins[key].append(1.0)
                        else:
                            if args.opt_scale>0:
                                self.opt_bins[key].append(i/args.opt_scale + 1.0)

                            self.opt_bins[key].append(1.0/(i/abs(args.opt_scale)+1))


                self.num_bin.append(len(self.opt_bins[key]))
                self.num_action*=len(self.opt_bins[key])

            print(self.opt_bins)
            print(self.num_bin)
            print("ACTION SPACE: ", self.num_action)
            self.read_interval = args.read_interval

        if args and args.use_mem:

            self.dnd = dnd.DND(inverse_distance, num_neighbors=args.k, max_memory=args.memory_size, lr=args.write_lr)
            epsilon_start = args.epsilon_start
            epsilon_final = args.epsilon_final
            epsilon_decay = args.epsilon_decay

            self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
                -1. * frame_idx / epsilon_decay)
            self.traj_buffer = []
            self.last_context = None
            self.last_action = None
            self.last_ecw_grads = deque(maxlen=self.args.context_order+1)
            self.last_ecb_grads = deque(maxlen=self.args.context_order+1)


            self.contex_size = args.context_size
            
            self.write_interval = args.write_interval
            self.num_ecw = args.num_ecw
            self.ecw_names = []
            self.ecb_names = []
            self.ec_feature_size = 0
            ecws = []
            ecws2 = []

            for name, param in sorted(self.actor_critic.named_parameters()):
                # print(name, param.shape,param.grad)
                if 'weight' in name:
                    ecws.append([param.shape[0], np.prod(param.shape[1:]),self.contex_size])
                    self.ecw_names.append([name,param])
                if 'bias' in name:
                    ecws2.append([param.shape[0],self.contex_size])
                    self.ecb_names.append([name,param])

            ecws = ecws[-self.num_ecw:]
            ecws2 = ecws2[-self.num_ecw:]

            self.ecw_names = self.ecw_names[-self.num_ecw:]
            self.ecb_names = self.ecb_names[-self.num_ecw:]

            for s in ecws:
                self.ec_feature_size += self.contex_size*s[0]
            for _ in self.ecb_names:
                self.ec_feature_size += self.contex_size

            self.ecw_weights = []
            self.ecb_weights = []


            for ii in range (args.context_order+1):
                for s in ecws:
                    self.ecw_weights.append(nn.Linear(s[1], s[2]))
                for s in ecws2:
                    self.ecb_weights.append(nn.Linear(s[0], s[1]))
            self.ec_feature_size*=(args.context_order+1)

            self.ecw_weights = nn.ModuleList(self.ecw_weights)
            self.ecb_weights = nn.ModuleList(self.ecb_weights)

            self.key_net = nn.Sequential(
                nn.Linear(self.ec_feature_size+1, args.mem_dim),
                nn.Tanh()
            )

            if args.context_train==0:
                for param in self.ecb_weights.parameters():
                    param.requires_grad = False
                for param in self.ecw_weights.parameters():
                    param.requires_grad = False

            elif args.context_train==1:
                self.key_net = nn.Sequential(
                    nn.Linear(args.hidden_size, args.mem_dim),
                    nn.Tanh()
                )
                self.vae_loss = deque(maxlen=200)
                self.ffwvae = FFWVAE(self.ec_feature_size+1, args.hidden_size)
                self.vae_optimizer = torch.optim.Adam(list(self.ffwvae.parameters())
                                                      + list(self.ecw_weights.parameters())
                                                      + list(self.ecb_weights.parameters()))
            for param in self.key_net.parameters():
                param.requires_grad = False
            # raise False
            print('fs: ', self.ec_feature_size)

            self.ecw_weights.apply(init_weights)
            self.ecb_weights.apply(init_weights)


        print (f"epo num p: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")


    def compute_last_grad(self):
        gradw = []
        for name, p in self.ecw_names:
            inp = p.grad.detach().clone()
            gradw.append(inp)
        self.last_ecw_grads.append(gradw)

        gradb = []

        for name, p in self.ecb_names:
            inp = p.grad.detach().clone()
            gradb.append(inp)
        self.last_ecb_grads.append(gradb)

    def decode_action(self, aid):
        num =  len(self.opt_values)
        id = []
        count = 0

        while count<num:
            # c = self.num_bin[count]
            e = num-count-1
            p = 1
            if e >0:
                p=np.prod(self.num_bin[-e:])
            x = aid//p
            id.append(x)
            aid = aid - x*p
            count+=1
        for i, k in enumerate(sorted(self.opt_values.keys())):
            self.opt_values[k]=self.opt_bins[k][id[i]]

    def addEM(self, hkey, action, R):
        # print(hkey)
        if self.args.context_train == 1:
            hkey = self.ffwvae.encode(hkey)
        hkey = self.key_net(torch.stack([hkey])).detach()
        rvec = torch.zeros(1, self.num_action)
        rvec[0, action] = R
        if USE_CUDA:
            rvec = rvec.cuda()
        # print('k',hkey)
        embedding_index = self.dnd.get_index(hkey)
        if embedding_index is None:
            self.dnd.insert(hkey, rvec.detach())

        if self.dnd.keys is not None:
            # try:
            self.dnd.lookup2write(hkey, rvec.detach(), K=self.args.k_write, max_mode=self.args.max_mode)

    def readEM(self, hkey, step=0):
        q_estimates, indexes = self.dnd.lookup(hkey, K=self.args.k)
        return q_estimates

    def choose_opt(self, context, epsilon=0, step=0):
        if self.args.use_mem<=1:
            if self.args.plot=="key-value":
                # print(self.args.plot)
                # print(step)

                if step > 0 and step % self.args.plot_interval == 0:
                    keys = self.dnd.keys.data.cpu().numpy()
                    values = self.dnd.values.data.cpu().numpy()

                    plot_keys_values(keys, values, step, self.args.plotdir)

            if random.random() > epsilon and self.args.use_mem\
                    and self.dnd.get_mem_size()>1 and step>self.args.memory_start:
                # print(epsilon)
                if self.args.context_train == 1:
                    context = self.ffwvae.encode(context)
                key = self.key_net(torch.stack([context]))
                # print(key)
                q_value = self.readEM(key.detach(), step)
                action = q_value.max(1)[1].data[0]
                # if random.random() > 0.9995:
                #     print("q lookup", q_value)
                #     print(action)
            else:
                action = random.randrange(self.num_action)
        
        return action

    def feature_extract(self, context, gstep=0):
        # print(self.actor_critic.dist.linear.weight.grad)
        # print(self.actor_critic.dist.linear.bias.grad)

        vecs = []
        i = 0
        if context is None:
            fv = torch.zeros(self.ec_feature_size)
            if USE_CUDA:
                fv = fv.cuda()
            vecs.append(fv)
            pe = torch.tensor([1]) * gstep
            if USE_CUDA:
                pe = pe.cuda()
            vecs.append(pe)
            fv = torch.cat(vecs, dim=-1)
            # print(fv.shape, self.ec_feature_size)
            return fv

        for p in context[0]:
            o = self.ecw_weights[i](p)
            vecs.append(o.view(-1))
            # print(name, o.shape)
            i+=1

        i = 0
        for p in context[1]:
            o = self.ecb_weights[i](p)
            vecs.append(o.view(-1))
            # print(name, o.shape)
            i+=1
        pe = torch.tensor([1])*gstep
        if USE_CUDA:
            pe = pe.cuda()
        vecs.append(pe)
        fv = torch.cat(vecs, dim=-1)
        # print(fv.shape, self.ec_feature_size)
        return fv

    def context_extract_raw(self):
        # print(self.actor_critic.dist.linear.weight.grad)
        # print(self.actor_critic.dist.linear.bias.grad)

        vecs = []
        if self.ecw_names[0][1].grad is None:
            return None


        i = 0

        for jj in range (self.args.context_order+1):
            ii = 0

            for name, p in self.ecw_names:
                if jj==0:
                    inp = p.detach()
                elif jj>len(self.last_ecw_grads):
                    inp = torch.zeros(p.grad.size())
                    if USE_CUDA:
                        inp = inp.cuda()
                else:
                    inp = self.last_ecw_grads[-jj][ii]
                ii+=1

                inp = inp.view(inp.shape[0],-1)
                vecs.append(inp)



        vecsb = []

        i = 0
        ii = 0

        for jj in range (self.args.context_order+1):
            ii = 0

            for name, p in self.ecb_names:
                if jj==0:
                    inp = p.detach()
                elif jj>len(self.last_ecw_grads):
                    inp = torch.zeros(p.grad.size())
                    if USE_CUDA:
                        inp = inp.cuda()
                else:
                    inp = self.last_ecb_grads[-jj][ii]
                ii+=1

                vecsb.append(inp.view(-1))


        return [vecs, vecsb]


    def take_action(self, update_step, env_step, gstep=0):
        if self.args:
            for key in self.opt_values.keys():
                self.last_opt_values[key].append(self.opt_values[key])
            cstep = gstep
            if gstep == 0:
                cstep = update_step
            if  self.args.adaptive_opt>0 and cstep % self.read_interval == 0:
                if self.args.use_mem:


                    epsilon = linearly_decaying_epsilon(self.args.num_env_steps//2 - self.args.memory_start,
                                                        env_step, self.args.memory_start,
                                                        self.args.epsilon_final)
                    # if gstep==0:
                    #     print(epsilon)
                    last_context = self.context_extract_raw()
                    last_feature = self.feature_extract(last_context, gstep)
                    self.last_context = last_context

                    self.last_action = self.choose_opt(last_feature,
                                                       epsilon=epsilon,
                                                       step=update_step)
                    self.decode_action(self.last_action)
                    # print('clip', self.clip)
                else:
                    self.last_action = self.choose_opt(context=None, epsilon=1.0)
                    self.decode_action(self.last_action)

                    # print(clip)

    def add2buffer(self, epo_reward, update_step, gstep=0):
        cstep = gstep
        if gstep == 0:
            cstep = update_step
        if self.args and self.args.use_mem and cstep % self.write_interval == 0:
          
           
            xxx = (copy.deepcopy(self.last_context),
                   self.last_action,
                   epo_reward, None, gstep)
            self.traj_buffer.append(xxx)
        

    def insert2mem(self, R, update_step, gstep=0):
        if self.args.use_mem and self.last_context is not None:
            if len(self.traj_buffer) > 1:
                self.traj_buffer.pop()
            
            xxx = (copy.deepcopy(self.last_context),
                   self.last_action,
                   R
                   , None, gstep)
            self.traj_buffer.append(xxx)


            if self.args.use_mem and update_step % self.args.episode_size == 0:
                cc = 0
                RR = 0
                states = []
                for last_context, action, reward, h, gstep in self.traj_buffer[::-1]:
                    RR += reward
                    state = self.feature_extract(last_context,gstep)
                    states.append(state)
                    if self.args.use_mem==1:
                        self.addEM(state, action, torch.as_tensor(RR))
                    cc += 1
                if self.args.use_mem == 1:
                    self.dnd.commit_insert()


                if random.random()<self.args.ct_rate and self.args.context_train==1 and update_step<self.args.context_train_limit:
                    sar = random.choices(states, k=self.args.c_bs)
                    X = []

                    for s in sar:
                        X.append(s)
                    X = torch.sigmoid(torch.stack(X, dim=0))
                    # print(X.shape)
                    reconstruction, z, mu, logvar = self.ffwvae(F.dropout(X, p=0.5))
                    self.vae_optimizer.zero_grad()
                    # print(reconstruction.shape)
                    # print(obs_processed.shape)
                    # print(z.shape)
                    bce_loss = self.ffwvae.criterion(reconstruction, X.detach())
                    loss = self.ffwvae.final_loss(bce_loss, mu, logvar)
                    loss.backward()
                    self.vae_loss.append(loss.item())
                    # print(loss)
                    self.vae_optimizer.step()

               
                del self.traj_buffer
                self.traj_buffer = []
                


