import signal
import sys
import os
import time
import torch
import melee
from dotenv import load_dotenv

from collections import deque
from melee_input import StateBuffer, get_state
from melee_output import tensor_to_controller
from melee_reward import calculate_reward
from melee_actor_critic import ActorCriticMelee, train_step

# Configurações do Frame Skip e RL
FRAME_SKIP = 4    # Repete a mesma ação por 4 frames (15 tomadas de decisão/s)
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

    model = ActorCriticMelee(input_dim=3600, num_actions=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)

    best_model_path = "saved_models/model_ep_100.pt"
    ep = 1
    best_reward = float('-inf')

    if os.path.exists(best_model_path):
        print(f"Loading checkpoint from {best_model_path}...")
        checkpoint = torch.load(best_model_path, weights_only=False)
        
        try:
            # Tenta carregar o modelo de forma estrita
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ep = checkpoint.get('episode', 0) + 1
            print(f"Checkpoint carregado com sucesso! Continuaremos a partir do Episódio {ep}.")
        except RuntimeError as e:
            # Se as dimensões da rede mudaram (ex: migração de 720D para 3600D), ignora o modelo antigo!
            print("\n" + "="*80)
            print("⚠️ AVISO DE INCOMPATIBILIDADE DE ARQUITETURA DETECTADO!")
            print("O checkpoint existente foi criado com um modelo de tamanho diferente (ex: 720D vs 3600D).")
            print("Iniciando um NOVO treinamento do zero com a nova arquitetura de 5 frames (3600D)...")
            print("="*80 + "\n")
            
            # Reseta o número de episódios e ignora os pesos velhos
            ep = 1
            best_reward = float('-inf')

    # Variáveis de Controle do Loop
    experiences = []
    episode_reward = 0
    rewards_history = []
    episode_metrics_list = []
    in_game_flag = False
    
    # Controle de Frame Skip
    frame_skip_counter = 0
    accumulated_skip_reward = 0.0
    
    # Registros do estado inicial do Macro-Step e do Estado Anterior
    macro_start_state = None
    macro_action_idx = None
    macro_action_cont = None
    prev_state_tensor = None  # 👈 1. INICIALIZADO AQUI

    prev_gamestate = None
    
    # Instancia o buffer de 5 estados
    state_buffer = StateBuffer(k=5, state_dim=720)

    # --- INÍCIO DA PARTIDA ---
    gamestate = console.step()

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

            # SE É O PRIMEIRO FRAME DA PARTIDA: Usa o seu reset() para preencher os 5 slots com o primeiro estado
            if not in_game_flag:
                in_game_flag = True
                stacked_state = state_buffer.reset(state_tensor)  # 👈 Retorna o vetor já preenchido com 3600D!
            else:
                # Nos frames seguintes, adiciona normalmente
                stacked_state = state_buffer.append(state_tensor)

            # -------------------------------------------------------------
            # 1. INÍCIO DO MACRO-STEP: Amostra uma nova ação sem gradiente
            # -------------------------------------------------------------
            if frame_skip_counter == 0:
                macro_start_state = stacked_state

                with torch.no_grad():
                    action_discrete, action_continuous = model.select_action(stacked_state)
                
                macro_action_idx = action_discrete.item()
                macro_action_cont = action_continuous

            # -------------------------------------------------------------
            # 2. EXECUÇÃO DA AÇÃO
            # -------------------------------------------------------------
            # Se macro_action_cont tem o formato (1, 2):
            if macro_action_cont.dim() > 1:
                macro_action_cont = macro_action_cont.squeeze(0) # Transforma (1, 2) em (2,)
            stick_x = macro_action_cont[0].item()
            stick_y = macro_action_cont[1].item()
            tensor_to_controller(controller, stick_x, stick_y, macro_action_idx)

            # -------------------------------------------------------------
            # 3. RECOMPENSA E ACÚMULO
            # -------------------------------------------------------------
            if prev_gamestate is not None:
                reward = calculate_reward(prev_gamestate, gamestate, agent_port, enemy_port)
                accumulated_skip_reward += reward
                episode_reward += reward

            frame_skip_counter += 1
            prev_gamestate = gamestate
            prev_state_tensor = stacked_state  # 👈 2. ATUALIZADO A CADA FRAME ATIVO

            # -------------------------------------------------------------
            # 4. FIM DO MACRO-STEP: Armazena a transição de 4 frames
            # -------------------------------------------------------------
            if frame_skip_counter >= FRAME_SKIP:
                state_to_save = stacked_state.squeeze(0) if stacked_state.dim() > 1 else stacked_state
                experiences.append((
                    macro_start_state,        # Estado no início dos 4 frames
                    macro_action_idx,         # Ação discreta executada
                    macro_action_cont,        # Ação contínua executada
                    accumulated_skip_reward,  # Recompensa TOTAL acumulada
                    False,                    # done = False
                    state_to_save              # Estado final após os 4 frames
                ))

                frame_skip_counter = 0
                accumulated_skip_reward = 0.0

                if len(experiences) >= N_STEPS:
                    metrics = train_step(model, optimizer, experiences)
                    episode_metrics_list.append(metrics)
                    experiences = []

        else:
            # -------------------------------------------------------------
            # TRATAMENTO DE FIM DE PARTIDA / MENUS
            # -------------------------------------------------------------
            if prev_gamestate is not None:
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
                        prev_state_tensor   # 👈 3. USADO AQUI COM SEGURANÇA
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
                rewards_history.append(episode_reward)
                print("Average Reward: ", sum(rewards_history)/len(rewards_history))
                print("Average Reward (last 10): ", sum(rewards_history[-10:])/min(len(rewards_history), 10))
                averege_reward_last_ten = sum(rewards_history[-10:])/min(len(rewards_history), 10)
                avg_metrics = {}
                if episode_metrics_list:
                    for k in episode_metrics_list[0].keys():
                        avg_metrics[k] = sum(m[k] for m in episode_metrics_list) / len(episode_metrics_list)
                    print(f"Métricas Médias do Episódio: {avg_metrics}")

                checkpoint = {
                    'episode': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_reward': episode_reward,
                    'avg_metrics': avg_metrics
                }

                if averege_reward_last_ten > best_reward and len(rewards_history) > 9:
                    best_reward = averege_reward_last_ten
                    torch.save(checkpoint, "saved_models/model_best.pt")
                    print(f"*** Novo modelo salvo! Recompensa: {best_reward:.2f} ***")

                ep += 1
                in_game_flag = False
                episode_reward = 0
                episode_metrics_list = []

                state_buffer.buffer.clear()

if __name__ == "__main__":
    main()
