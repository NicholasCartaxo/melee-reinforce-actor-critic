#!/usr/bin/python3
import signal
import sys
import melee
from melee_input import get_state
from melee_output import tensor_to_controller
from melee_reward import calculate_reward
from melee_actor_critic import ActorCriticMelee
from melee_actor_critic import train_step
import torch
from dotenv import load_dotenv
import os
import time

def main():

  load_dotenv()
  
  # Create our Console object.
  #   This will be one of the primary objects that we will interface with.
  #   The Console represents the virtual or hardware system Melee is playing on.
  #   Through this object, we can get "GameState" objects per-frame so that your
  #     bot can actually "see" what's happening in the game
  console = melee.Console(
    path=os.getenv("MAINLINE_PATH"),
    fullscreen=False,
    save_replays=False,
    disable_audio=True,
    emulation_speed=0,
    gfx_backend="Null",
  )

  # Create our Controller object
  #   The controller is the second primary object your bot will interact with
  #   Your controller is your way of sending button presses to the game, whether
  #   virtual or physical.

  agent_port = 1
  enemy_port = 4

  controller = melee.Controller(
    console=console,
    port=agent_port,
    type=melee.ControllerType.STANDARD)
  
  cpuController = melee.Controller(
    console=console,
    port=enemy_port,
    type=melee.ControllerType.STANDARD
  )

  controllers = [controller,cpuController]

  # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
  def signal_handler(sig, frame):
    controller.disconnect()
    cpuController.disconnect()
    console.stop()
    sys.exit(0)

  signal.signal(signal.SIGINT, signal_handler)

  # Run the console
  console.run(iso_path=os.getenv("ISO_PATH"))

  # Connect to the console
  print("Connecting to console...")
  if not console.connect():
    print("ERROR: Failed to connect to the console.")
    sys.exit(-1)
  print("Console connected")

  # Plug our controller in
  #   Due to how named pipes work, this has to come AFTER running dolphin
  #   NOTE: If you're loading a movie file, don't connect the controller,
  #   dolphin will hang waiting for input and never receive it
  print("Connecting controller to console...")
  for c in controllers:
    if not c.connect():
      print("ERROR: Failed to connect the controller.")
      sys.exit(-1)
    print("Controller connected")

  menu_helper = melee.MenuHelper()

  ALPHA = 1e-4      # Learning rate
  N_STEPS = 20      # steps

  #Model configuration
  episode_reward = 0
  model = ActorCriticMelee(input_dim=719, num_actions=10)
  optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)
  experiences = []
  ep = 0
  in_game_flag = True
  
  prev_gamestate = None
  prev_state_tensor = None
  prev_action_idx = None
  prev_log_prob = None
  prev_value = None
  prev_entropy = None
  
  t = time.time()
  
  # Main loop
  while True:

    gamestate = console.step()

    if gamestate is None:
      continue

    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
      
      if agent_port not in gamestate.players or enemy_port not in gamestate.players:
        continue
      
      in_game_flag = True

      state_vec = get_state(gamestate,agent_port,enemy_port)

      state_tensor = torch.FloatTensor(state_vec)

      if prev_gamestate is not None:
        reward = calculate_reward(prev_gamestate, gamestate, agent_port, enemy_port)
        episode_reward += reward

        experiences.append((
          prev_state_tensor,
          prev_action_idx,
          prev_log_prob,
          reward,
          prev_value,
          prev_entropy,
          state_tensor,
          False # done = False
        ))
      
        if len(experiences) >= N_STEPS:
          metrics = train_step(model,optimizer,experiences)
          print(metrics)
          experiences = []
      
      
      action_discrete, action_continuous, log_prob, entropy, value = model.select_action(state_tensor)

      action_idx = action_discrete.item()
      stick_x = action_continuous[0].item()
      stick_y = action_continuous[1].item()

      tensor_to_controller(controller,stick_x,stick_y,action_idx)
      
      prev_gamestate = gamestate
      prev_state_tensor = state_tensor
      prev_action_idx = action_idx
      prev_log_prob = log_prob
      prev_value = value
      prev_entropy = entropy

      if gamestate.frame % 120 == 0:
        d = time.time() - t
        print("fps:",120/d)
        t = time.time()

    else:
      
      if prev_gamestate is not None:
        
        reward = calculate_reward(prev_gamestate, gamestate, agent_port, enemy_port)
        episode_reward += reward
        
        experiences.append((
          prev_state_tensor,
          prev_action_idx,
          prev_log_prob,
          reward,
          prev_value,
          prev_entropy,
          prev_state_tensor,
          True # done = True
        ))
        
        if len(experiences) > 0:
          metrics = train_step(model, optimizer, experiences)
          print(metrics)
          experiences = []
          
        prev_gamestate = None
        prev_state_tensor = None
      

      menu_helper.menu_helper_simple(
        gamestate=gamestate,
        controller=controller,
        character_selected=melee.Character.LUIGI,
        stage_selected=melee.Stage.BATTLEFIELD,
        swag=False,
        autostart=False)

      menu_helper.menu_helper_simple(
        gamestate=gamestate,
        controller=cpuController,
        character_selected=melee.Character.LUIGI,
        stage_selected=melee.Stage.BATTLEFIELD,
        cpu_level=9,
        swag=False,
        autostart=True)
      
      controller.flush()
      cpuController.flush()
        
      if in_game_flag:
        print(f'Episode {ep} reward: {episode_reward}')
        ep += 1
        in_game_flag = False
        episode_reward = 0

      
if __name__ == "__main__":
  main()