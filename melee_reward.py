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
        reward += 5  # Grande recompensa por nocautear o oponente

    if player_lost_stock:
        if prev_player.percent < 40.0:
            reward -= 10.0  # Penalidade maior se morreu com porcentagem baixa 
        else:
            reward -= 5.0 # Grande penalidade por ser nocauteado

    # Cálculo de Dano Causado / Sofrido
    # Ignora discrepâncias de porcentagem quando ocorre troca de vida
    damage_dealt = 0.0 if enemy_lost_stock else max(0.0, enemy.percent - prev_enemy.percent)
    damage_taken = 0.0 if player_lost_stock else max(0.0, player.percent - prev_player.percent)

    # 1% de dano provocado = +0.02 / 1% de dano sofrido = -0.02
    reward += 0.02 * (damage_dealt - damage_taken)

    # Punições por Posicionamento (Shaping Rewards)
    stage_edge_x = melee.EDGE_GROUND_POSITION[curr_gs.stage]
    
    safe_zone_limit = stage_edge_x * 0.75
    
    # Penaliza levemente o bot se ele estiver na beirada
    dist_from_center = abs(player.position.x)
    if dist_from_center > safe_zone_limit:
        reward -= 0.0001 * (dist_from_center - safe_zone_limit)
    
    # Penaliza o "camping" extremo (fugir demais) baseando-se no tamanho do mapa
    # Se a distância entre os jogadores for maior que o tamanho de meio palco
    if curr_gs.distance > stage_edge_x:
        reward -= 0.00005 * (curr_gs.distance - stage_edge_x)
    
    # Offstage (Fora da plataforma principal)
    if player.off_stage:
        reward -= 0.001  # Incentiva recuperações rápidas
    if enemy.off_stage and not player.off_stage:
        reward += 0.001  # Recompensa por manter o oponente fora do palco

    return reward
