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


def main():

  load_dotenv()
  
  exi = os.getenv("EXI_PATH")
  mainline = os.getenv("MAINLINE_PATH")
  # Create our Console object.
  #   This will be one of the primary objects that we will interface with.
  #   The Console represents the virtual or hardware system Melee is playing on.
  #   Through this object, we can get "GameState" objects per-frame so that your
  #     bot can actually "see" what's happening in the game
  console = melee.Console(
    path=mainline,
    fullscreen=False,
    save_replays=False,
    enable_ffw=False,
    use_exi_inputs=False
  )

  # Create our Controller object
  #   The controller is the second primary object your bot will interact with
  #   Your controller is your way of sending button presses to the game, whether
  #   virtual or physical.

  controller = melee.Controller(
    console=console,
    port=1,
    type=melee.ControllerType.STANDARD)
  
  cpuController = melee.Controller(
    console=console,
    port=2,
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
  c = 0
  in_game_flag = True
  # Main loop
  while True:

    gamestate = console.step()

    if gamestate is None:
      continue


    if gamestate.menu_state == melee.Menu.IN_GAME:
      in_game_flag = True

      state_vec = get_state(gamestate,1,2)

      state_tensor = torch.FloatTensor(state_vec)

      action_idx, log_prob, entropy, value = model.select_action(state_tensor)

      tensor_to_controller(controller,action_idx.item())

        
      next_gamestate = console.step()
      done = False
      if next_gamestate is None:
        continue

      if next_gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        done = True
        next_state_tensor = state_tensor
      else:
        next_state_vec = get_state(next_gamestate,1,2)

        next_state_tensor = torch.FloatTensor(next_state_vec)

      reward = calculate_reward(gamestate,next_gamestate,1,2)
      episode_reward += reward

      experiences.append((
        state_tensor,
        action_idx,
        log_prob,
        reward,
        value,
        entropy,
        next_state_tensor,
        done
      ))


      if len(experiences) >= N_STEPS or done:

        metrics = train_step(model,optimizer,experiences)
        print(metrics)
        experiences = []


    else:
      menu_helper.menu_helper_simple(
        gamestate=gamestate,
        controller=controller,
        character_selected=melee.Character.LUIGI,
        stage_selected=melee.Stage.BATTLEFIELD)

      menu_helper.menu_helper_simple(
        gamestate=gamestate,
        controller=cpuController,
        character_selected=melee.Character.FOX,
        stage_selected=melee.Stage.BATTLEFIELD,
        cpu_level=9,
        autostart=True)
        
      print(f'Episode {c} reward: {episode_reward}')
      if in_game_flag:
        c += 1
        in_game_flag = False
        episode_reward = 0

      
if __name__ == "__main__":
  main()