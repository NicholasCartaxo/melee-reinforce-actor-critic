import torch
import torch.nn as nn
import torch.nn.functional as F


GAMMA = 0.99     # Desconto 
H_SCALE = 0.01    # Escala de entropia para evitar determinismo 
ALPHA = 0.01

class LeakySoftplus(nn.Module):

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
        self.actor = nn.Linear(128, num_actions) # Saída de 10 ações
        self.critic = nn.Linear(128, 1)          # Saída do valor de estado V(s)

    def forward(self, x):
        x = self.shared(x)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)
    
    def select_action(self, state):
        probs, value = self(state)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

def calculate_reward(prev_gs, curr_gs, port, enemy_port):
    if prev_gs is None:
        return 0.0

    reward = 0.0

    prev_player = prev_gs.players[port]
    prev_enemy = prev_gs.players[enemy_port]

    player = curr_gs.players[port]
    enemy = curr_gs.players[enemy_port]

    player_lost_stock = player.stock < prev_player.stock
    enemy_lost_stock = enemy.stock < prev_enemy.stock

    if enemy_lost_stock:
        reward += 1.0

    if player_lost_stock:
        reward -= 1.0

    damage_dealt = 0.0 if enemy_lost_stock else max(0.0, enemy.percent - prev_enemy.percent)
    damage_taken = 0.0 if player_lost_stock else max(0.0, player.percent - prev_player.percent)

    reward += 0.01 * (damage_dealt - damage_taken)

    return reward

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

    log_probs = torch.stack(
        [e[2] for e in experiences]
    )

    rewards = [
        e[3] for e in experiences
    ]

    values = torch.cat(
        [e[4] for e in experiences]
    ).squeeze()

    entropies = torch.stack(
        [e[5] for e in experiences]
    )

    next_states = torch.stack(
        [e[6] for e in experiences]
    )

    dones = [
        e[7] for e in experiences
    ]


    ####################################
    # 1. Bootstrap do último estado
    ####################################

    with torch.no_grad():

        _, next_value = model(
            next_states[-1]
        )

        R = next_value.squeeze()


        if dones[-1]:
            R = torch.tensor(
                0.0
            )


    ####################################
    # 2. Retorno n-step
    ####################################

    returns = []

    for i in reversed(range(len(rewards))):

        if dones[i]:
            R = torch.tensor(0.0)

        R = rewards[i] + GAMMA * R

        returns.insert(
            0,
            R
        )


    returns = torch.stack(
        returns
    )


    ####################################
    # 3. Advantage
    ####################################

    advantage = (
        returns -
        values.detach()
    )


    ####################################
    # 4. Actor Loss
    ####################################

    actor_loss = -(
        log_probs *
        advantage
    ).mean()


    ####################################
    # 5. Critic Loss
    ####################################

    critic_loss = F.mse_loss(
        values,
        returns
    )


    ####################################
    # 6. Entropy Bonus
    ####################################

    entropy_loss = -(
        H_SCALE *
        entropies.mean()
    )


    ####################################
    # 7. Loss total
    ####################################

    loss = (
        actor_loss
        +
        0.5 * critic_loss
        +
        entropy_loss
    )


    ####################################
    # 8. Atualização
    ####################################

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