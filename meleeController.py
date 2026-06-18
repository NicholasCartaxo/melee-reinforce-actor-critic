#!/usr/bin/python3
import signal
import sys
import melee
import time

def main():
  # Create our Console object.
  #   This will be one of the primary objects that we will interface with.
  #   The Console represents the virtual or hardware system Melee is playing on.
  #   Through this object, we can get "GameState" objects per-frame so that your
  #     bot can actually "see" what's happening in the game
  console = melee.Console(
    path="C:/Users/USER/AppData/Roaming/Slippi Launcher/netplay-beta",
    fullscreen=False,
    gfx_backend="Null",
    disable_audio=True,
    use_exi_inputs=True,
    enable_ffw=True,
  )

  # Create our Controller object
  #   The controller is the second primary object your bot will interact with
  #   Your controller is your way of sending button presses to the game, whether
  #   virtual or physical.

  ports = [1, 2]

  controllers = {
    port: melee.Controller(
      console=console,
      port=port,
      type=melee.ControllerType.STANDARD)
    for port in ports
  }

  # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
  def signal_handler(sig, frame):
    for controller in controllers.values():
      controller.disconnect()
    console.stop()
    sys.exit(0)

  signal.signal(signal.SIGINT, signal_handler)

  # Run the console
  console.run(iso_path="D:/melee/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso")

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
  for controller in controllers.values():
    if not controller.connect():
      print("ERROR: Failed to connect the controller.")
      sys.exit(-1)
  print("Controller connected")

  costume = 0
  framedata = melee.framedata.FrameData()

  menu_helper = melee.MenuHelper()

  d = time.time()

  # Main loop
  while True:  # Run for 10 seconds
    # "step" to the next frame
    gamestate = console.step()
    if gamestate is None:
      continue

    print(time.time()-d)
    d = time.time()

    # What menu are we in?
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:

      for port, controller in controllers.items():
        # NOTE: This is where your AI does all of its stuff!
        # This line will get hit once per frame, so here is where you read
        #   in the gamestate and decide what buttons to push on the controller
        melee.techskill.multishine(ai_state=gamestate.players[port], controller=controller)

    else:
      for port, controller in controllers.items():
        menu_helper.menu_helper_simple(
          gamestate,
          controller,
          melee.Character.FOX,
          melee.Stage.POKEMON_STADIUM,
          costume=port,
          autostart=port == 1,
          swag=False)

if __name__ == "__main__":
  main()