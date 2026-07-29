import melee

def calculate_reward(prev_gs: melee.GameState, curr_gs: melee.GameState, port: int, enemy_port: int) -> float:
    if prev_gs is None or curr_gs is None:
        return 0.0

    reward = 0.0

    prev_player = prev_gs.players[port]
    prev_enemy = prev_gs.players[enemy_port]

    player = curr_gs.players[port]
    enemy = curr_gs.players[enemy_port]

    # Checagem de Perda/Ganho de Vidas (Stocks)
    player_lost_stock = player.stock < prev_player.stock
    enemy_lost_stock = enemy.stock < prev_enemy.stock

    if enemy_lost_stock:
        reward += 10.0  # Grande recompensa por nocautear o oponente

    if player_lost_stock:
        reward -= 10.0  # Grande penalidade por ser nocauteado

    # Cálculo de Dano Causado / Sofrido
    # Ignora discrepâncias de porcentagem quando ocorre troca de vida
    damage_dealt = 0.0 if enemy_lost_stock else max(0.0, enemy.percent - prev_enemy.percent)
    damage_taken = 0.0 if player_lost_stock else max(0.0, player.percent - prev_player.percent)

    # 1% de dano provocado = +0.01 / 1% de dano sofrido = -0.01
    reward += 0.01 * (damage_dealt - damage_taken)

    # Punições por Posicionamento (Shaping Rewards)
    
    # Longe do centro do palco (x = 0)
    # Penalidade gradativa quanto mais longe do eixo central
    dist_from_center = abs(player.x)
    reward -= 0.00002 * dist_from_center
    
    # Longe do oponente (distância Euclidiana)
    # Incentiva o bot a se aproximar e engajar no combate
    reward -= 0.00002 * curr_gs.distance
    
    # Offstage (Fora da plataforma principal)
    if player.off_stage:
        reward -= 0.002  # Punição contínua por frame/macro-step que ficar fora do palco
    if enemy.off_stage:
        reward += 0.002

    return reward