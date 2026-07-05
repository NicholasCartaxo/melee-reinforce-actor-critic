import melee

ACTIONS = [
  melee.Button.BUTTON_A, 
  melee.Button.BUTTON_B, 
  melee.Button.BUTTON_X, 
  melee.Button.BUTTON_Z,
  melee.Button.BUTTON_L,
  (0.5, 1.0),  # C-Up
  (0.5, 0.0),  # C-Down
  (0.0, 0.5),  # C-Left
  (1.0, 0.5),  # C-Right
  0            # No action 
]

def tensor_to_controller(controller: melee.Controller, 
             stick_x: float, 
             stick_y: float, 
             action_probs: list[float]):
  
  controller.release_all()
  
  mapped_x = (stick_x + 1.0) / 2.0
  mapped_y = (stick_y + 1.0) / 2.0
  controller.tilt_analog(melee.Button.BUTTON_MAIN, mapped_x, mapped_y)
  
  action = ACTIONS[action_probs.index(max(action_probs))]
  if isinstance(action, melee.Button):
    controller.press_button(action)
  elif isinstance(action, tuple):
    c_x, c_y = action
    controller.tilt_analog(melee.Button.BUTTON_C, c_x, c_y)