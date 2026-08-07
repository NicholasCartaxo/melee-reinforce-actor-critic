import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta

# Hiperparâmetros de Treinamento
GAMMA = 0.99
H_SCALE = 0.01          # Escala da perda por entropia
MAX_GRAD_NORM = 0.5      # Limite de corte dos gradientes

class Actor(nn.Module):
    def __init__(self, input_dim=720, num_actions=10):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01)
        )
        
        # Cabeças de decisão
        self.actor_discrete = nn.Linear(512, num_actions)
        self.actor_continuous = nn.Linear(512, 4) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.net(x)
        probs = F.softmax(self.actor_discrete(x), dim=-1)
        
        ab = F.softplus(self.actor_continuous(x)) + 1.0
        ab = torch.clamp(ab, min=1.01, max=50.0)
        alpha, beta = ab.chunk(2, dim=-1)

        return probs, alpha, beta
    
    @torch.no_grad()
    def select_action(self, state):
        probs, alpha, beta = self(state)

        dist_discrete = Categorical(probs)
        action_discrete = dist_discrete.sample()
        
        dist_continuous = Beta(alpha, beta)
        action_continuous = dist_continuous.sample()

        return action_discrete, action_continuous


class Critic(nn.Module):
    def __init__(self, input_dim=720):
        super().__init__()
        
        # Rede isolada apenas para o Critic
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 1) # Saída direta do valor
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0) # Gain 1.0 is often better for value outputs
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


def train_step(actor, critic, actor_optimizer, critic_optimizer, experiences):
    """
    Agora recebe duas redes (actor, critic) e dois otimizadores correspondentes.
    """
    # 1. Extração
    states = torch.stack([e[0] for e in experiences])
    device = states.device
    actions_disc = torch.tensor([e[1] for e in experiences], dtype=torch.long, device=device)
    actions_cont = torch.stack([e[2] for e in experiences])
    rewards = torch.tensor([e[3] for e in experiences], dtype=torch.float32, device=device)
    dones = torch.tensor([e[4] for e in experiences], dtype=torch.float32, device=device)
    next_states = torch.stack([e[5] for e in experiences])

    # 2. Avaliação do Critic (Isolado)
    values = critic(states).squeeze(-1)
    with torch.no_grad():
        next_values = critic(next_states).squeeze(-1)

    # 3. Cálculo dos Retornos
    returns = torch.zeros_like(rewards)
    R = next_values[-1] * (1.0 - dones[-1])
    for t in reversed(range(len(rewards))):
        if dones[t]:
            R = 0.0
        R = rewards[t] + GAMMA * R
        returns[t] = R

    # 4. Cálculo da Vantagem
    advantages = returns - values.detach()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ==========================================
    # ATUALIZAÇÃO DO CRITIC
    # ==========================================
    critic_loss = F.mse_loss(values, returns)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
    critic_optimizer.step()

    # ==========================================
    # ATUALIZAÇÃO DO ACTOR
    # ==========================================
    probs, alpha, beta = actor(states)
    
    dist_disc = Categorical(probs)
    dist_cont = Beta(alpha, beta)

    actions_cont_clamped = torch.clamp(actions_cont, 1e-6, 1.0 - 1e-6)
    
    log_prob_disc = dist_disc.log_prob(actions_disc)
    log_prob_cont = dist_cont.log_prob(actions_cont_clamped).sum(dim=-1)
    total_log_probs = log_prob_disc + log_prob_cont

    entropy = dist_disc.entropy() + dist_cont.entropy().sum(dim=-1)

    # Perda do Actor
    actor_loss = -(total_log_probs * advantages.detach()).mean()
    entropy_loss = -entropy.mean()
    
    total_actor_loss = actor_loss + (H_SCALE * entropy_loss)

    actor_optimizer.zero_grad()
    total_actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
    actor_optimizer.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": entropy.mean().item()
    }
