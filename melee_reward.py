import melee

def calculate_reward(prev_gs: melee.GameState, curr_gs: melee.GameState, port: int, enemy_port: int) -> float:
    if prev_gs is None or curr_gs is None:
        return 0.0

    reward = 0.0

    prev_player = prev_gs.players[port]
    prev_enemy = prev_gs.players[enemy_port]

    player = curr_gs.players[port]
    enemy = curr_gs.players[enemy_port]

    # 1. Checagem de Perda/Ganho de Vidas (Stocks)
    player_lost_stock = player.stock < prev_player.stock
    enemy_lost_stock = enemy.stock < prev_enemy.stock

    if enemy_lost_stock:
        reward += 10.0  # Grande recompensa por nocautear o oponente

    if player_lost_stock:
        reward -= 10.0  # Grande penalidade por ser nocauteado

    # 2. Cálculo de Dano Causado / Sofrido
    # Ignora discrepâncias de porcentagem quando ocorre troca de vida
    damage_dealt = 0.0 if enemy_lost_stock else max(0.0, enemy.percent - prev_enemy.percent)
    damage_taken = 0.0 if player_lost_stock else max(0.0, player.percent - prev_player.percent)

    # 1% de dano provocado = +0.01 / 1% de dano sofrido = -0.01
    reward += 0.01 * (damage_dealt - damage_taken)

    return reward