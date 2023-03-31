# Import modules
import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.distributions.categorical import Categorical

# Define dual critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.l4 = nn.Linear(state_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(state))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return torch.squeeze(q1, dim=-1), torch.squeeze(q2, dim=-1)

# Define actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.action = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.softmax(self.action(a), dim=1)

        return a


class Agent():
    def __init__(self, state_dim, action_dim, batch_size=256, lr=3e-4, gamma=0.99, tau=0.005, device="cpu"):

        # Initialisation
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Record losses
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.alpha_loss_history = []

        # Set remaining parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.min_ent = -action_dim
        self.eps = 1e-9                 # To prevent log of zero

    def choose_action(self, state, sample=False):
        with torch.no_grad():
            state = torch.Tensor([state]).to(self.device)
            if sample:
                probs = self.actor(state)
                cat_dis = Categorical(probs)
                action = cat_dis.sample()
            else:
                action = torch.argmax(self.actor(state))

        return action.cpu().numpy().flatten()

    def train(self, replay_buffer):

        # Sample batch from replay buffer
        minibatch = random.sample(replay_buffer, self.batch_size)
        state = torch.Tensor(tuple(d[0] for d in minibatch)).to(self.device)
        action = torch.Tensor(tuple(d[1] for d in minibatch)).reshape(-1, 1).long().to(self.device)
        reward = torch.Tensor(tuple(d[2] for d in minibatch)).to(self.device)
        next_state = torch.Tensor(tuple(d[3] for d in minibatch)).to(self.device)
        done = torch.Tensor(tuple(d[4] for d in minibatch)).to(self.device)

        alpha = self.log_alpha.exp().detach()

        # Critic loss #
        with torch.no_grad():
            next_actions = self.actor(next_state)
            q1_target, q2_target = self.critic_target(next_state)
            q_target = torch.min(q1_target, q2_target) - alpha * torch.log(next_actions + self.eps)
            q_target = (next_actions * q_target).sum(1)
            q_hat = reward + self.gamma * (1 - done) * q_target
        q1, q2 = self.critic(state)
        q1 = q1.gather(1, action).reshape(-1)
        q2 = q2.gather(1, action).reshape(-1)
        critic_loss = F.mse_loss(q1, q_hat) + F.mse_loss(q2, q_hat)

        self.critic_loss_history.append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor and alpha loss #
        actions = self.actor(state)
        log_actions = torch.log(actions + self.eps)
        critic_value1, critic_value2 = self.critic(state)
        critic_value = torch.min(critic_value1, critic_value2) - alpha * log_actions
        actor_loss = -((actions * critic_value).sum(1)).mean()
        alpha_loss = -(self.log_alpha * (log_actions.detach() + self.min_ent)).mean()

        self.actor_loss_history.append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_loss_history.append(alpha_loss.item())
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        ### Polyak target network updates ###
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
