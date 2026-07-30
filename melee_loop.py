import signal
import sys
import os
import torch
import melee
import csv
import glob
import re
from dotenv import load_dotenv

from melee_input import get_state
from melee_output import tensor_to_controller
from melee_reward import calculate_reward
from melee_actor_critic import ActorCriticMelee, train_step

# Configurações do Frame Skip e RL
FRAME_SKIP = 2    # Repete a mesma ação por 2 frames (30 tomadas de decisão/s)
N_STEPS = 256
ALPHA = 1e-4
CPU_LEVEL = 3

def main():
    load_dotenv()
    
    console = melee.Console(
        path=os.getenv("MAINLINE_PATH"),
        fullscreen=False,
        save_replays=False,
        disable_audio=True,
        emulation_speed=0,
        gfx_backend="Null",
    )

    agent_port = 1
    enemy_port = 4

    controller = melee.Controller(console=console, port=agent_port, type=melee.ControllerType.STANDARD)
    cpuController = melee.Controller(console=console, port=enemy_port, type=melee.ControllerType.STANDARD)
    controllers = [controller, cpuController]

    def signal_handler(sig, frame):
        controller.disconnect()
        cpuController.disconnect()
        console.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    console.run(iso_path=os.getenv("ISO_PATH"))
    print("Connecting to console...")
    if not console.connect():
        sys.exit(-1)

    for c in controllers:
        if not c.connect():
            sys.exit(-1)

    menu_helper = melee.MenuHelper()
    os.makedirs("saved_models", exist_ok=True)

    model = ActorCriticMelee(input_dim=720, num_actions=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)

    ep = 1
    best_reward = float('-inf')

    # Carregar best_reward a partir do model_best.pt
    best_model_path = "saved_models/model_best.pt"
    if os.path.exists(best_model_path):
        print(f"Loading best reward from {best_model_path}...")
        best_checkpoint = torch.load(best_model_path, weights_only=False)
        best_reward = best_checkpoint.get('best_reward', float('-inf'))
        print(f"Current best average reward known: {best_reward:.2f}")

    # Carregar o último modelo periódico para continuar o treinamento
    model_files = glob.glob("saved_models/model_ep_*.pt")
    latest_model_path = None
    max_ep = 0
    
    for f in model_files:
        match = re.search(r'model_ep_(\d+)\.pt', f)
        if match:
            ep_num = int(match.group(1))
            if ep_num > max_ep:
                max_ep = ep_num
                latest_model_path = f

    if latest_model_path and os.path.exists(latest_model_path):
        print(f"Loading latest checkpoint to resume from {latest_model_path}...")
        checkpoint = torch.load(latest_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ep = checkpoint.get('episode', 0) + 1
        print(f"Resuming from episode {ep}")

    # Variáveis de Controle do Loop
    experiences = []
    episode_reward = 0
    episode_metrics_list = []
    in_game_flag = False
    
    # Controle de Frame Skip
    frame_skip_counter = 0
    accumulated_skip_reward = 0.0
    
    # Registros do estado inicial do Macro-Step e do Estado Anterior
    macro_start_state = None
    macro_action_idx = None
    macro_action_cont = None
    prev_state_tensor = None
    prev_gamestate = None

    # Inicialização do CSV geral
    csv_filename = "training_logs.csv"
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "CPU_Level", "Agent_Stock", "CPU_Stock"])
            
    # Inicialização do CSV exclusivo para o melhor modelo
    best_csv_filename = "best_models_log.csv"
    if not os.path.exists(best_csv_filename):
        with open(best_csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "CPU_Level", "Agent_Stock", "CPU_Stock"])

    logs_buffer = []

    while True:
        gamestate = console.step()
        if gamestate is None:
            continue

        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            if agent_port not in gamestate.players or enemy_port not in gamestate.players:
                continue

            in_game_flag = True
            state_vec = get_state(gamestate, agent_port, enemy_port)
            state_tensor = torch.FloatTensor(state_vec)

            # INÍCIO DO MACRO-STEP
            if frame_skip_counter == 0:
                macro_start_state = state_tensor

                with torch.no_grad():
                    action_discrete, action_continuous = model.select_action(state_tensor)
                
                macro_action_idx = action_discrete.item()
                macro_action_cont = action_continuous

            # EXECUÇÃO DA AÇÃO
            stick_x = macro_action_cont[0].item()
            stick_y = macro_action_cont[1].item()
            tensor_to_controller(controller, stick_x, stick_y, macro_action_idx)

            # RECOMPENSA E ACÚMULO
            if prev_gamestate is not None:
                reward = calculate_reward(prev_gamestate, gamestate, agent_port, enemy_port)
                accumulated_skip_reward += reward
                episode_reward += reward

            frame_skip_counter += 1
            prev_gamestate = gamestate
            prev_state_tensor = state_tensor

            # FIM DO MACRO-STEP: Armazena a transição de 2 frames
            if frame_skip_counter >= FRAME_SKIP:
                experiences.append((
                    macro_start_state,        # Estado no início dos 2 frames
                    macro_action_idx,         # Ação discreta executada
                    macro_action_cont,        # Ação contínua executada
                    accumulated_skip_reward,  # Recompensa TOTAL acumulada
                    False,                    # done = False
                    state_tensor              # Estado final após os 2 frames
                ))

                frame_skip_counter = 0
                accumulated_skip_reward = 0.0

                if len(experiences) >= N_STEPS:
                    metrics = train_step(model, optimizer, experiences)
                    episode_metrics_list.append(metrics)
                    experiences = []

        else:
            # TRATAMENTO DE FIM / COMEÇO DE PARTIDA
            agent_stock = 0
            cpu_stock = 0

            if prev_gamestate is not None:
                if agent_port in prev_gamestate.players:
                    agent_stock = prev_gamestate.players[agent_port].stock
                if enemy_port in prev_gamestate.players:
                    cpu_stock = prev_gamestate.players[enemy_port].stock

                reward = calculate_reward(prev_gamestate, gamestate, agent_port, enemy_port)
                accumulated_skip_reward += reward
                episode_reward += reward

                if macro_start_state is not None and prev_state_tensor is not None:
                    experiences.append((
                        macro_start_state,
                        macro_action_idx,
                        macro_action_cont,
                        accumulated_skip_reward,
                        True,               # done = True
                        prev_state_tensor
                    ))

                if len(experiences) > 0:
                    metrics = train_step(model, optimizer, experiences)
                    episode_metrics_list.append(metrics)
                    experiences = []

                prev_gamestate = None
                prev_state_tensor = None
                macro_start_state = None
                frame_skip_counter = 0
                accumulated_skip_reward = 0.0

            # Gerenciamento dos menus
            menu_helper.menu_helper_simple(
                gamestate=gamestate, controller=controller,
                character_selected=melee.Character.LUIGI,
                stage_selected=melee.Stage.BATTLEFIELD,
                swag=False, autostart=False
            )
            menu_helper.menu_helper_simple(
                gamestate=gamestate, controller=cpuController,
                character_selected=melee.Character.LUIGI,
                stage_selected=melee.Stage.BATTLEFIELD,
                cpu_level=CPU_LEVEL, swag=False, autostart=True
            )
            controller.flush()
            cpuController.flush()

            if in_game_flag:
                print(f"--- Fim do Episódio {ep} ---")
                print(f"Recompensa do Episódio: {episode_reward:.2f}")
                print(f"Estoque Final - Agent: {agent_stock} | CPU: {cpu_stock}")
                
                avg_metrics = {}
                if episode_metrics_list:
                    for k in episode_metrics_list[0].keys():
                        avg_metrics[k] = sum(m[k] for m in episode_metrics_list) / len(episode_metrics_list)
                    print(f"Métricas Médias do Episódio: {avg_metrics}")

                logs_buffer.append([ep, episode_reward, CPU_LEVEL, agent_stock, cpu_stock])

                checkpoint = {
                    'episode': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_reward': episode_reward,
                    'avg_metrics': avg_metrics,
                    'best_reward': best_reward
                }

                # Salva melhor modelo
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    checkpoint['best_reward'] = best_reward
                    torch.save(checkpoint, "saved_models/model_best.pt")
                    print(f"*** Novo melhor modelo salvo! Recompensa: {best_reward:.2f} ***")
                    
                    # Escreve log exclusivo para o modelo de melhor performance
                    with open(best_csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([ep, episode_reward, best_reward, CPU_LEVEL, agent_stock, cpu_stock])

                # Salva a cada 100 modelos
                if ep % 100 == 0:
                    torch.save(checkpoint, f"saved_models/model_ep_{ep}.pt")
                    print(f"*** Checkpoint periódico salvo: model_ep_{ep}.pt ***")
                    
                    # Atualiza CSV geral
                    with open(csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(logs_buffer)
                    logs_buffer = []

                ep += 1
                in_game_flag = False
                episode_reward = 0
                episode_metrics_list = []

if __name__ == "__main__":
    main()
