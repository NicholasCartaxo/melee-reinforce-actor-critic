import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta

# Hiperparâmetros de Treinamento
GAMMA = 0.99
H_SCALE = 0.01          # Escala da perda por entropia
CRITIC_COEFF = 0.5       # Peso do erro do Critic na perda total
MAX_GRAD_NORM = 0.5      # Limite de corte dos gradientes

class ActorCriticMelee(nn.Module):
    def __init__(self, input_dim=3600, num_actions=10):
        super().__init__()
        
        # Redimensionamento para 512 neurônios com LayerNorm
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01)
        )
        
        # Cabeças de decisão específicas
        self.actor_discrete = nn.Linear(512, num_actions)
        self.actor_continuous = nn.Linear(512, 4)  # 4 parâmetros: [alpha_x, beta_x, alpha_y, beta_y]
        self.critic = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        features = self.shared(x)
        probs = F.softmax(self.actor_discrete(features), dim=-1)
        
        # Parametrização segura da distribuição Beta (alpha, beta >= 1.0)
        ab = F.softplus(self.actor_continuous(features)) + 1.0
        ab = torch.clamp(ab, min=1.01, max=50.0)
        alpha, beta = ab.chunk(2, dim=-1)

        value = self.critic(features)
        return probs, alpha, beta, value
    
    @torch.no_grad()
    def select_action(self, state):
        """Amostragem rápida sem retenção de gradientes (Usada no loop de coleta)"""
        # Se o estado veio como (3600,), adiciona dimensão de batch -> (1, 3600)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        probs, alpha, beta, _ = self(state)

        dist_discrete = Categorical(probs)
        action_discrete = dist_discrete.sample()
        
        dist_continuous = Beta(alpha, beta)
        action_continuous = dist_continuous.sample()

        return action_discrete, action_continuous


def train_step(model, optimizer, experiences):
    """
    Treina o Actor-Critic usando as experiências do loop.py.
    Estrutura da tupla esperada (6 elementos):
    (state, action_disc, action_cont, reward, done, next_state)
    """
    # 1. Extração e conversão dos dados do buffer
    states = torch.stack([e[0] for e in experiences])
    actions_disc = torch.tensor([e[1] for e in experiences], dtype=torch.long)
    actions_cont = torch.stack([e[2] for e in experiences])
    rewards = torch.tensor([e[3] for e in experiences], dtype=torch.float32)
    dones = torch.tensor([e[4] for e in experiences], dtype=torch.float32)
    next_states = torch.stack([e[5] for e in experiences])

    # 2. Reavaliação dos estados para calcular gradientes ativos
    probs, alpha, beta, values = model(states)
    values = values.squeeze(-1)

    # Reavaliação do último próximo estado para o Bootstrap
    with torch.no_grad():
        _, _, _, next_values = model(next_states)
        next_values = next_values.squeeze(-1)

    # 3. Cálculo dos Retornos N-Step (Bootstrap)
    returns = torch.zeros_like(rewards)
    R = next_values[-1] * (1.0 - dones[-1])

    for t in reversed(range(len(rewards))):
        if dones[t]:
            R = 0.0
        R = rewards[t] + GAMMA * R
        returns[t] = R

    # 4. Cálculo da Vantagem (Advantage)
    advantages = returns - values.detach()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 5. Reconstrução das distribuições e cálculo das Log-Probabilidades
    dist_disc = Categorical(probs)
    dist_cont = Beta(alpha, beta)

    # Prevenção de instabilidade nos limites [0, 1] da Beta
    actions_cont_clamped = torch.clamp(actions_cont, 1e-6, 1.0 - 1e-6)

    log_prob_disc = dist_disc.log_prob(actions_disc)
    log_prob_cont = dist_cont.log_prob(actions_cont_clamped).sum(dim=-1)
    total_log_probs = log_prob_disc + log_prob_cont

    # Entropia total (Incentivo à exploração)
    entropy = dist_disc.entropy() + dist_cont.entropy().sum(dim=-1)

    # 6. Cálculo das Perdas (Losses)
    actor_loss = -(total_log_probs * advantages).mean()
    critic_loss = F.mse_loss(values, returns)
    entropy_loss = -entropy.mean()

    total_loss = actor_loss + (CRITIC_COEFF * critic_loss) + (H_SCALE * entropy_loss)

    # 7. Otimização dos Pesos
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": entropy.mean().item()
    }