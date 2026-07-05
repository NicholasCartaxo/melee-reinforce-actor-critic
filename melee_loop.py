#!/usr/bin/python3
import signal
import sys
import melee
import melee_input
import melee_output
import random
import numpy as np

def main():

  exi = "/home/nicholascartaxo/Slippi/exi-ai/Binaries/dolphin-emu"
  mainline = "/home/nicholascartaxo/Slippi/mainline/Binaries/dolphin-emu-nogui"
  # Create our Console object.
  #   This will be one of the primary objects that we will interface with.
  #   The Console represents the virtual or hardware system Melee is playing on.
  #   Through this object, we can get "GameState" objects per-frame so that your
  #     bot can actually "see" what's happening in the game
  console = melee.Console(
    path=mainline,
    fullscreen=True,
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
  console.run(iso_path="/home/nicholascartaxo/melee.iso")

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

  a = True

  # Main loop
  while True:
    # "step" to the next frame
    gamestate = console.step()
    if gamestate is None:
      continue

    # What menu are we in?
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
      #print(melee_input.get_state(gamestate, 1, 2))
      melee_output.tensor_to_controller(controller,
                                        random.uniform(-1,1),
                                        random.uniform(-1,1),
                                        [random.random() for _ in range(10)])
      

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
        stage_selected=melee.Stage.POKEMON_STADIUM,
        cpu_level=9,
        autostart=True)
      
if __name__ == "__main__":
  main()