import melee
import numpy as np

framedata = melee.FrameData()

NUM_CHARACTERS = 26 
NUM_ATTACK_STATES = 4
NUM_STAGES = 6

def char_index(char: melee.Character) -> int:
  # Mario (0x00) through Popo (0x0a)
  if 0 <= char.value <= 10:
    return char.value
  # Pikachu (0x0c) through Roy (0x1a), shift down by 1 to fill Nana's gap
  elif 12 <= char.value <= 26:
    return char.value - 1
  # Nana, Wireframes, Bosses, or Unknown return -1 (all zeros in one-hot)
  else:
    return -1

NON_COMPETITIVE_STATES = {
  # Heavy Item Carry/Throw
  melee.Action.ITEM_PICKUP_HEAVY, melee.Action.LIFT_WAIT, melee.Action.LIFT_WALK_1, melee.Action.LIFT_WALK_2, melee.Action.LIFT_TURN,
  melee.Action.ITEM_THROW_HEAVY_FORWARD, melee.Action.ITEM_THROW_HEAVY_BACK, melee.Action.ITEM_THROW_HEAVY_HIGH, melee.Action.ITEM_THROW_HEAVY_LOW,
  melee.Action.ITEM_THROW_HEAVY_AIR_SMASH_FORWARD, melee.Action.ITEM_THROW_HEAVY_AIR_SMASH_BACK, 
  melee.Action.ITEM_THROW_HEAVY_AIR_SMASH_HIGH, melee.Action.ITEM_THROW_HEAVY_AIR_SMASH_LOW,
  
  # Non-Peach Melee Weapons
  melee.Action.BAT_SWING_1, melee.Action.BAT_SWING_2, melee.Action.BAT_SWING_3, melee.Action.BAT_SWING_4,
  melee.Action.PARASOL_SWING_1, melee.Action.PARASOL_SWING_2, melee.Action.PARASOL_SWING_3, melee.Action.PARASOL_SWING_4,
  melee.Action.FAN_SWING_1, melee.Action.FAN_SWING_2, melee.Action.FAN_SWING_3, melee.Action.FAN_SWING_4,
  melee.Action.STAR_ROD_SWING_1, melee.Action.STAR_ROD_SWING_2, melee.Action.STAR_ROD_SWING_3, melee.Action.STAR_ROD_SWING_4,
  melee.Action.LIP_STICK_SWING_1, melee.Action.LIP_STICK_SWING_2, melee.Action.LIP_STICK_SWING_3, melee.Action.LIP_STICK_SWING_4,
  
  # Ranged Items
  melee.Action.GUN_SHOOT, melee.Action.GUN_SHOOT_AIR, melee.Action.GUN_SHOOT_EMPTY, melee.Action.GUN_SHOOT_AIR_EMPTY,
  melee.Action.FIRE_FLOWER_SHOOT, melee.Action.FIRE_FLOWER_SHOOT_AIR,
  melee.Action.ITEM_SCOPE_START, melee.Action.ITEM_SCOPE_RAPID, melee.Action.ITEM_SCOPE_FIRE, melee.Action.ITEM_SCOPE_END,
  melee.Action.ITEM_SCOPE_AIR_START, melee.Action.ITEM_SCOPE_AIR_RAPID, melee.Action.ITEM_SCOPE_AIR_FIRE, melee.Action.ITEM_SCOPE_AIR_END,
  melee.Action.ITEM_SCOPE_START_EMPTY, melee.Action.ITEM_SCOPE_RAPID_EMPTY, melee.Action.ITEM_SCOPE_FIRE_EMPTY, melee.Action.ITEM_SCOPE_END_EMPTY,
  melee.Action.ITEM_SCOPE_AIR_START_EMPTY, melee.Action.ITEM_SCOPE_AIR_RAPID_EMPTY, melee.Action.ITEM_SCOPE_AIR_FIRE_EMPTY, melee.Action.ITEM_SCOPE_AIR_END_EMPTY,
  
  # Transformation / Special Items
  melee.Action.ITEM_SCREW, melee.Action.ITEM_SCREW_AIR, melee.Action.DAMAGE_SCREW, melee.Action.DAMAGE_SCREW_AIR,
  melee.Action.WARP_STAR_JUMP, melee.Action.WARP_STAP_FALL,
  melee.Action.HAMMER_WAIT, melee.Action.HAMMER_WALK, melee.Action.HAMMER_TURN, melee.Action.HAMMER_KNEE_BEND, 
  melee.Action.HAMMER_FALL, melee.Action.HAMMER_JUMP, melee.Action.HAMMER_LANDING,
  melee.Action.KINOKO_GIANT_START, melee.Action.KINOKO_GIANT_START_AIR, melee.Action.KINOKO_GIANT_END, melee.Action.KINOKO_GIANT_END_AIR,
  melee.Action.KINOKO_SMALL_START, melee.Action.KINOKO_SMALL_START_AIR, melee.Action.KINOKO_SMALL_END, melee.Action.KINOKO_SMALL_END_AIR,
  
  # PvE / Boss Elements
  melee.Action.CAPTURE_MASTERHAND, melee.Action.CAPTURE_DAMAGE_MASTERHAND, melee.Action.CAPTURE_WAIT_MASTERHAND, melee.Action.THROWN_MASTERHAND,
  melee.Action.CAPTURE_CRAZYHAND, melee.Action.CAPTURE_DAMAGE_CRAZYHAND, melee.Action.CAPTURE_WAIT_CRAZYHAND, melee.Action.THROWN_CRAZY_HAND,
  melee.Action.CAPTURE_LEA_DEAD, melee.Action.CAPTURE_LIKE_LIKE, 
  melee.Action.BARREL_WAIT, melee.Action.BARREL_CANNON_WAIT,
  
  #Nothing state
  melee.Action.NOTHING_STATE
}

COMPETITIVE_ACTIONS = [a for a in melee.Action if a not in NON_COMPETITIVE_STATES]
ACTION_TO_INDEX_MAP = {action: idx for idx, action in enumerate(COMPETITIVE_ACTIONS)}
NUM_ACTIONS = len(COMPETITIVE_ACTIONS)

left_x, right_x, upper_y, lower_y = (0,0,0,0)


framedata = melee.FrameData()

def norm_speed(speed):
  return max(-1.0, min(1.0, float(speed) / 5))

def norm_frames(frame):
  return max(-1.0, min(1.0, float(frame) / 60))

def action_index(action: melee.Action) -> int:
  return ACTION_TO_INDEX_MAP.get(action,-1)


def stage_index(stage: melee.Stage) -> int:
  stage_map = {
    melee.enums.Stage.FINAL_DESTINATION: 0,
    melee.enums.Stage.BATTLEFIELD: 1,
    melee.enums.Stage.POKEMON_STADIUM: 2,
    melee.enums.Stage.DREAMLAND: 3,
    melee.enums.Stage.FOUNTAIN_OF_DREAMS: 4,
    melee.enums.Stage.YOSHIS_STORY: 5,
  }
  
  return stage_map.get(stage, -1)


def one_hot(val: int, num_classes: int) -> list:
  vec = [0.0] * num_classes
  if 0 <= val < num_classes:
    vec[val] = 1.0
  return vec

def player_features(player: melee.PlayerState) -> list:
  
  char_list = one_hot(char_index(player.character), NUM_CHARACTERS)
  action_list = one_hot(action_index(player.action), NUM_ACTIONS)
  attack_state_list = one_hot(
    framedata.attack_state(
      player.character,
      player.action,
      player.action_frame
    ).value,
    NUM_ATTACK_STATES
  )
  
  max_frames = framedata.frame_count(player.character, player.action)
  action_progress = float(player.action_frame) / max_frames if max_frames > 0 else 0.0
  can_iasa = 1.0 if (player.iasa > 0 and player.action_frame >= player.iasa) else 0.0

  continuous_features = [
    float(player.position.x)/abs(left_x),
    float(player.position.y)/abs(upper_y),
    min(1.0,float(player.percent)/200),
    float(player.shield_strength)/60,
    float(player.is_powershield),
    float(player.facing),
    action_progress,
    float(player.invulnerable),
    norm_frames(float(player.invulnerability_left)),
    norm_frames(float(player.hitlag_left)),
    norm_frames(float(player.hitstun_frames_left)),
    float(player.jumps_left)/framedata.max_jumps(player.character),
    float(player.on_ground),
    norm_speed(float(player.speed_air_x_self)),
    norm_speed(float(player.speed_y_self)),
    norm_speed(float(player.speed_x_attack)),
    norm_speed(float(player.speed_y_attack)),
    norm_speed(float(player.speed_ground_x_self)),
    float(player.off_stage),
    can_iasa,
  ]
  
  return char_list + action_list + attack_state_list + continuous_features


def get_state(gamestate: melee.GameState, botPort: int, enemyPort: int) -> np.ndarray:
  global left_x, right_x, upper_y, lower_y
  if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
    raise RuntimeError("Current state not in game")

  if botPort not in gamestate.players.keys():
    raise RuntimeError("Bot player not connected")
  
  if enemyPort not in gamestate.players.keys():
    raise RuntimeError("Enemy player not connected")
  
  left_x, right_x, upper_y, lower_y = melee.BLASTZONES[gamestate.stage]

  bot = gamestate.players[botPort]
  enemy = gamestate.players[enemyPort]

  bot_features = player_features(bot)
  enemy_features = player_features(enemy)

  dx = (enemy.position.x - bot.position.x) / (abs(left_x) * 2)
  dy = (enemy.position.y - bot.position.y) / (abs(upper_y) * 2)

  stage_list = one_hot(stage_index(gamestate.stage), NUM_STAGES)
  general_features = stage_list + [dx, dy]

  mlp_input = np.array(bot_features + enemy_features + general_features, dtype=np.float32)
  
  return mlp_input