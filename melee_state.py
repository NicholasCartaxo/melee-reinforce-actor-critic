import melee
import numpy as np

framedata = melee.FrameData()

NUM_CHARACTERS = 26 
NUM_ACTIONS = 397
NUM_ATTACK_STATES = 4
NUM_STAGES = 6

def char_index(char: int) -> int:
  # Mario (0x00) through Popo (0x0a)
  if 0 <= char <= 10:
    return char
  # Pikachu (0x0c) through Roy (0x1a), shift down by 1 to fill Nana's gap
  elif 12 <= char <= 26:
    return char - 1
  # Nana, Wireframes, Bosses, or Unknown return -1 (all zeros in one-hot)
  else:
    return -1

def action_index(action: int) -> int:
  #TODO
  return action

def stage_index(stage: int) -> int:
  stage_map = {
    melee.enums.Stage.FINAL_DESTINATION.value: 0,
    melee.enums.Stage.BATTLEFIELD.value: 1,
    melee.enums.Stage.POKEMON_STADIUM.value: 2,
    melee.enums.Stage.DREAMLAND.value: 3,
    melee.enums.Stage.FOUNTAIN_OF_DREAMS.value: 4,
    melee.enums.Stage.YOSHIS_STORY.value: 5,
  }
  
  return stage_map.get(stage, -1)


def one_hot(val: int, num_classes: int) -> list:
  vec = [0.0] * num_classes
  if 0 <= val < num_classes:
    vec[val] = 1.0
  return vec

def player_features(player: melee.PlayerState) -> list:
  
  char_list = one_hot(char_index(player.character.value), NUM_CHARACTERS)
  action_list = one_hot(player.action.value, NUM_ACTIONS)
  attack_state_list = one_hot(
    framedata.attack_state(
      player.character,
      player.action,
      player.action_frame
    ).value,
    NUM_ATTACK_STATES
  )
  
  continuous_features = [
    float(player.position.x),
    float(player.position.y),
    float(player.percent),
    float(player.shield_strength),
    float(player.is_powershield),
    float(player.facing),
    float(player.action_frame),
    float(player.invulnerable),
    float(player.invulnerability_left),
    float(player.hitlag_left),
    float(player.hitstun_frames_left),
    float(player.jumps_left),
    float(player.on_ground),
    float(player.speed_air_x_self),
    float(player.speed_y_self),
    float(player.speed_x_attack),
    float(player.speed_y_attack),
    float(player.speed_ground_x_self),
    float(player.off_stage),
    float(player.iasa),
  ]
  
  return char_list + action_list + attack_state_list + continuous_features


def get_state(gamestate: melee.GameState, botPort: int, enemyPort: int) -> np.ndarray:
  if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
    raise RuntimeError("Current state not in game")
  
  if botPort not in gamestate.players.keys():
    raise RuntimeError("Bot player not connected")
  
  if enemyPort not in gamestate.players.keys():
    raise RuntimeError("Enemy player not connected")
  
  bot_features = player_features(gamestate.players[botPort])
  enemy_features = player_features(gamestate.players[enemyPort])

  stage_list = one_hot(stage_index(gamestate.stage.value), NUM_STAGES)
  general_features = stage_list + [gamestate.distance]

  mlp_input = np.array(bot_features + enemy_features + general_features, dtype=np.float32)
  
  return mlp_input