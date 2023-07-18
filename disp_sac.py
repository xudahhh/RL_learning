import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import time
from datetime import datetime
import os
from copy import deepcopy
from torch.distributions import Categorical

class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscreteActor, self).__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.action_layer = nn.Linear(64, action_dim)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = F.softmax(self.action_layer(x), dim=-1)
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) # 81
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        qs = logits.gather(-1, action) # 根据索引（action）,选择元素
        return qs
    def value(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

class SAC():
    """ train soft actor-critic """
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        actor_lr=3e-4, 
        critic_lr=3e-4, 
        alpha_lr=3e-4, 
        gamma=0.99, 
        alpha=0.2, 
        tau=0.005,
        auto_alpha = True,
        target_entropy = None,
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.replay_buffer = deque(maxlen=1000000)
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.actor = DiscreteActor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        # target critic
        self.critic1_trgt = deepcopy(self.critic1)
        self.critic2_trgt = deepcopy(self.critic2)
        self.critic1_trgt.eval()
        self.critic2_trgt.eval()

        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # alpha: weight of entropy
        self._auto_alpha = auto_alpha
        if self._auto_alpha:
            if not target_entropy:
                target_entropy = 0.98 * np.log(np.prod(self.action_dim))
            self._target_entropy = target_entropy
            self._log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self._alpha = self._log_alpha.detach().exp()
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
        else:
            self._alpha = alpha
        # other parameters
        self._tau = tau
        self._gamma = gamma
        self.device = device
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    def _sync_weight(self):
        """ synchronize weight """
        for trgt, src in zip(self.critic1_trgt.parameters(), self.critic1.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)
        for trgt, src in zip(self.critic2_trgt.parameters(), self.critic2.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)
   
    def actorward(self, obs, deterministic=False):
        """ forward propagation of actor (input stacked obs) """
        probs = self.actor(obs)
        dist = Categorical(probs)
        if deterministic:
            action = probs.argmax(axis=-1)
        else:
            action = dist.sample()

        return action, dist
    def select_action(self, state,deterministic=False):
        action,_ = self.actorward(state,deterministic)
        return action.cpu().detach().numpy()

    def update(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(np.array(action_batch)).view(batch_size, -1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).view(batch_size, -1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.LongTensor(np.array(done_batch)).view(batch_size, -1).to(self.device)

        # 更新 critic
        q1, q2 = self.critic1(state_batch, action_batch).flatten(), self.critic2(state_batch, action_batch).flatten()
        with torch.no_grad():
            _,dist_ = self.actorward(next_state_batch)
            q_ = dist_.probs * torch.min(self.critic1_trgt.value(next_state_batch), self.critic2_trgt.value(next_state_batch)) # 对比两个列表返回，最小的值
            q_ = q_.sum(dim=-1) + self._alpha * dist_.entropy() # dist_.entropy() 计算熵
            q_trgt = reward_batch.flatten() + self._gamma*(1 - done_batch.flatten())*q_.flatten()
        
        critic1_loss = F.mse_loss(q1, q_trgt)
        critic2_loss = F.mse_loss(q2, q_trgt)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # 更新 actor
        _, dist = self.actorward(state_batch)
        entropy = dist.entropy()
        q = torch.min(self.critic1.value(state_batch), self.critic2.value(state_batch))
        actor_loss = -(self._alpha*entropy + (dist.probs*q).sum(dim=-1)).flatten()

        actor_loss = actor_loss.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 更新 alpha
        if self._auto_alpha:
            log_prob = -entropy.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha*log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        # synchronize weight
        self._sync_weight()

        info = {
            "loss": {
                "actor": actor_loss.item(),
                "critic1": critic1_loss.item(),
                "critic2": critic2_loss.item()
            }
        }

        if self._auto_alpha:
            info["loss"]["alpha"] = alpha_loss.item()
            info["alpha"] = self._alpha.item()
        else:
            info["loss"]["alpha"] = 0
            info["alpha"] = self._alpha

        return info
    def save_model(self, filepath):
        """ save model """
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "alpha": self._alpha
        }
        torch.save(state_dict, filepath)

    def load_model(self, filepath):
        """ load model """
        state_dict = torch.load(filepath, map_location=torch.device("cpu"))
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self._alpha = state_dict["alpha"]
    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))