import melee

def calculate_reward(prev_gs: melee.GameState, curr_gs: melee.GameState, port: int, enemy_port: int):
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

  l_bound = melee.BLASTZONES[curr_gs.stage][0]
  
  player_dead = player.action.value <= 0x0d
  enemy_dead = enemy.action.value <= 0x0d

  if not player_dead:
    player_dist = abs(player.position.x/l_bound)
    reward -= 0.0005 * player_dist

    if player.off_stage:
      reward -= 0.0005
  
  if not enemy_dead:
    enemy_dist = abs(enemy.position.x/l_bound)
    reward += 0.0005 * enemy_dist

    if enemy.off_stage:
      reward += 0.0005
  
  damage_dealt = 0.0 if enemy_lost_stock else max(0.0, enemy.percent - prev_enemy.percent)
  damage_taken = 0.0 if player_lost_stock else max(0.0, player.percent - prev_player.percent)

  reward += 0.01 * (damage_dealt - damage_taken)

  return reward