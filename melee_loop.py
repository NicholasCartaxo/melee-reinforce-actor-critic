import signal
import sys
import os
import torch
import melee
import csv
import glob
import re
import argparse
from dotenv import load_dotenv

from melee_input import get_state
from melee_output import tensor_to_controller
from melee_reward import calculate_reward
from melee_actor_critic import ActorCriticMelee, train_step

# Configurações do Frame Skip e RL
FRAME_SKIP = 2    # Repete a mesma ação por 2 frames (30 tomadas de decisão/s)
N_STEPS = 256
ALPHA = 1e-4
CPU_LEVEL = 9

def main():
    parser = argparse.ArgumentParser(description="Train Melee RL Agent")
    parser.add_argument('--self-play', action='store_true', help="Treinar o agente contra si mesmo usando o modelo salvo mais recente")
    args = parser.parse_args()

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
    enemy_controller = melee.Controller(console=console, port=enemy_port, type=melee.ControllerType.STANDARD)
    controllers = [controller, enemy_controller]

    def signal_handler(sig, frame):
        controller.disconnect()
        enemy_controller.disconnect()
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

    # Configuração do Oponente (Self-Play)
    opponent_model = None
    if args.self_play:
        print("Modo Self-Play Ativado!")
        opponent_model = ActorCriticMelee(input_dim=720, num_actions=10)
        opponent_model.eval() # Modo de avaliação para não treinar este modelo independentemente

    ep = 1

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

        if args.self_play:
            opponent_model.load_state_dict(checkpoint['model_state_dict'])
            print("Modelo mais recente carregado para o oponente.")

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
    
    # Registros para o Oponente (Self-play)
    opp_macro_action_idx = None
    opp_macro_action_cont = None

    # Inicialização do CSV geral
    csv_filename = "training_logs.csv"
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
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

                # Ação do Agente Principal
                with torch.no_grad():
                    action_discrete, action_continuous = model.select_action(state_tensor)
                macro_action_idx = action_discrete.item()
                macro_action_cont = action_continuous

                # Ação do Oponente (Self-Play)
                if args.self_play and opponent_model is not None:
                    opp_state_vec = get_state(gamestate, enemy_port, agent_port) # Inverte as portas
                    opp_state_tensor = torch.FloatTensor(opp_state_vec)
                    with torch.no_grad():
                        opp_act_d, opp_act_c = opponent_model.select_action(opp_state_tensor)
                    opp_macro_action_idx = opp_act_d.item()
                    opp_macro_action_cont = opp_act_c

            # EXECUÇÃO DA AÇÃO - Agente Principal
            stick_x = macro_action_cont[0].item()
            stick_y = macro_action_cont[1].item()
            tensor_to_controller(controller, stick_x, stick_y, macro_action_idx)

            # EXECUÇÃO DA AÇÃO - Oponente
            if args.self_play and opp_macro_action_cont is not None:
                opp_stick_x = opp_macro_action_cont[0].item()
                opp_stick_y = opp_macro_action_cont[1].item()
                tensor_to_controller(enemy_controller, opp_stick_x, opp_stick_y, opp_macro_action_idx)

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
            # Se for self-play, nível da CPU cai para 0 (Humano/Standby) para o script controlar o P4
            opp_level = 0 if args.self_play else CPU_LEVEL

            menu_helper.menu_helper_simple(
                gamestate=gamestate, controller=controller,
                character_selected=melee.Character.LUIGI,
                stage_selected=melee.Stage.BATTLEFIELD,
                swag=False, autostart=False
            )
            menu_helper.menu_helper_simple(
                gamestate=gamestate, controller=enemy_controller,
                character_selected=melee.Character.LUIGI,
                stage_selected=melee.Stage.BATTLEFIELD,
                cpu_level=opp_level, swag=False, autostart=True
            )
            
            controller.flush()
            enemy_controller.flush()

            if in_game_flag:
                print(f"--- Fim do Episódio {ep} ---")
                print(f"Recompensa do Episódio: {episode_reward:.2f}")
                print(f"Estoque Final - Agent: {agent_stock} | CPU: {cpu_stock}")
                
                avg_metrics = {}
                if episode_metrics_list:
                    for k in episode_metrics_list[0].keys():
                        avg_metrics[k] = sum(m[k] for m in episode_metrics_list) / len(episode_metrics_list)
                    print(f"Métricas Médias do Episódio: {avg_metrics}")

                logs_buffer.append([ep, episode_reward, opp_level, agent_stock, cpu_stock])

                checkpoint = {
                    'episode': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_reward': episode_reward,
                    'avg_metrics': avg_metrics,
                }

                # Salva a cada 100 modelos
                if ep % 100 == 0:
                    torch.save(checkpoint, f"saved_models/model_ep_{ep}.pt")
                    print(f"*** Checkpoint periódico salvo: model_ep_{ep}.pt ***")

                    # Atualiza o modelo do oponente de tempos em tempos com o último checkpoint do agente
                    if args.self_play and opponent_model is not None:
                        opponent_model.load_state_dict(model.state_dict())
                        print("*** Modelo do Oponente (Self-Play) atualizado com os pesos mais recentes ***")

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
