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

  player_went_offstage = player.off_stage and not prev_player.off_stage
  enemy_went_offstage = enemy.off_stage and not prev_enemy.off_stage
  
  if enemy_went_offstage:
    reward += 0.25

  if player_went_offstage:
    reward -= 0.25
  
  damage_dealt = 0.0 if enemy_lost_stock else max(0.0, enemy.percent - prev_enemy.percent)
  damage_taken = 0.0 if player_lost_stock else max(0.0, player.percent - prev_player.percent)

  reward += 0.005 * (damage_dealt - damage_taken)

  return reward