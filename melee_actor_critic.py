import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta

GAMMA = 0.99     # Desconto 
H_SCALE = 0.01    # Escala de entropia para evitar determinismo 


class LeakySoftplus(nn.Module):
  ALPHA = 0.01
  def forward(self, x):
    return torch.logsumexp(
      torch.stack([self.ALPHA * x, x], dim=0),
      dim=0
    )

class ActorCriticMelee(nn.Module):
  def __init__(self, input_dim, num_actions):
    super().__init__()
    # Arquitetura: 2 camadas ocultas de 128 neurônios
    self.shared = nn.Sequential(
      nn.Linear(input_dim, 128),
      LeakySoftplus(),
      nn.Linear(128, 128),
      LeakySoftplus()
    )
    self.actor_discrete = nn.Linear(128, num_actions) # Saída de 10 ações
    
    #Valores do analógico esquerdo
    self.actor_continuous = nn.Sequential(
      nn.Linear(128, 4),
      nn.Sigmoid() #libmelee exige valores entre 0 e 1 para analógicos
    )
    self.critic = nn.Linear(128, 1)          # Saída do valor de estado V(s)

  def forward(self, x):
    x = self.shared(x)
    
    probs = F.softmax(self.actor_discrete(x), dim=-1)
    
    ab = F.softplus(self.actor_continuous(x)) 
    ab = torch.clamp(ab, min=1e-3, max=50.0)
    alpha, beta = ab.chunk(2, dim=-1)

    value = self.critic(x)
    return probs, alpha, beta, value
    
  def select_action(self, state):
    probs, alpha, beta, value = self(state)

    dist_discrete = Categorical(probs)
    action_discrete = dist_discrete.sample()
    
    dist_continuous = Beta(alpha, beta)
    action_continuous = dist_continuous.sample()
    
    log_prob_discrete = dist_discrete.log_prob(action_discrete)
    log_prob_continuous = dist_continuous.log_prob(action_continuous).sum(dim=-1)
    
    total_log_prob = log_prob_discrete + log_prob_continuous
    total_entropy = dist_discrete.entropy() + dist_continuous.entropy().sum(dim=-1)

    return action_discrete, action_continuous, total_log_prob, total_entropy, value

def train_step(model, optimizer, experiences):
  """
  Treina o Actor-Critic usando as experiências coletadas.

  Cada experiência:
  (
    state,
    action,
    log_prob,
    reward,
    value,
    entropy,
    next_state,
    done
  )
  """

  log_probs = torch.stack([e[2] for e in experiences])
  rewards = [e[3] for e in experiences]
  values = torch.cat([e[4] for e in experiences]).squeeze()
  entropies = torch.stack([e[5] for e in experiences])
  next_states = torch.stack([e[6] for e in experiences])
  dones = [e[7] for e in experiences]

  # 1. Bootstrap do último estado
  with torch.no_grad():
    _, _, _, next_value = model(next_states[-1])
    R = next_value.squeeze()

    if dones[-1]:
      R = torch.tensor(0.0)


  # 2. Retorno n-step
  returns = []

  for i in reversed(range(len(rewards))):
    if dones[i]:
      R = torch.tensor(0.0)
    R = rewards[i] + GAMMA * R

    returns.insert(0,R)


  returns = torch.stack(returns)

  advantage = (returns - values.detach())
  if len(advantage) > 1:
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

  actor_loss = -(log_probs * advantage).mean()

  critic_loss = F.mse_loss(values,returns)

  entropy_loss = -(H_SCALE * entropies.mean())

  loss = (actor_loss + 0.5 * critic_loss + entropy_loss)

  # 8. Atualização
  optimizer.zero_grad()

  loss.backward()

  # Gradient clipping usado no paper
  torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    40
  )

  optimizer.step()

  return {
    "loss": loss.item(),
    "actor_loss": actor_loss.item(),
    "critic_loss": critic_loss.item(),
    "entropy": entropies.mean().item()
  }