import signal
import sys
import os
import time
import torch
import melee
import glob
import re
import csv
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
CPU_LEVEL = 5

def restart_console(old_console, old_controllers, iso_path, agent_port=1, enemy_port=4, max_retries=3):
    print("🔄 Reiniciando o console Dolphin para prevenção de deadlocks...")
    
    # 1. Desconecta controles e para o console antigo com segurança
    for c in old_controllers:
        try:
            c.disconnect()
        except Exception:
            pass
            
    try:
        old_console.stop()
    except Exception:
        pass
    
    # Pausa para o sistema operacional liberar as portas e arquivos
    time.sleep(3)
    
    # 2. Loop de tentativas instanciando um NOVO objeto Console
    for attempt in range(1, max_retries + 1):
        try:
            new_console = melee.Console(
                path=os.getenv("MAINLINE_PATH"),
                fullscreen=False,
                save_replays=False,
                disable_audio=True,
                emulation_speed=0,
                gfx_backend="Null",
            )
            
            new_controller = melee.Controller(console=new_console, port=agent_port, type=melee.ControllerType.STANDARD)
            new_cpu_controller = melee.Controller(console=new_console, port=enemy_port, type=melee.ControllerType.STANDARD)
            new_controllers = [new_controller, new_cpu_controller]
            
            new_console.run(iso_path=iso_path)
            
            if new_console.connect():
                all_connected = True
                for c in new_controllers:
                    if not c.connect():
                        all_connected = False
                        break
                
                if all_connected:
                    print(f"✅ Console reiniciado e reconectado com sucesso na tentativa {attempt}!")
                    return new_console, new_controllers, new_controller, new_cpu_controller
                    
        except Exception as e:
            print(f"⚠️ Erro ao tentar reiniciar (Tentativa {attempt}/{max_retries}): {e}")
            
        time.sleep(2)
        
    print("❌ Erro crítico ao reconectar o Dolphin. Encerrando para preservação dos logs.")
    sys.exit(-1)

def main():
    # Limita threads para evitar gargalo de processamento no WSL/CPU
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

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
        try:
            controller.disconnect()
            cpuController.disconnect()
            console.stop()
        except Exception:
            pass
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
    os.makedirs("logs", exist_ok=True)

    # Definição dos nomes e arquivos CSV de log
    csv_filename = "logs/training_log.csv"
    best_csv_filename = "logs/best_training_log.csv"
    logs_buffer = []

    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "episode_reward", "best_reward", "cpu_level", "agent_stock", "cpu_stock"])

    if not os.path.exists(best_csv_filename):
        with open(best_csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "episode_reward", "best_reward", "cpu_level", "agent_stock", "cpu_stock"])

    # Instancia o modelo atualizado com 3600D (5 frames x 720)
    model = ActorCriticMelee(input_dim=3600, num_actions=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)

    ep = 1
    best_reward = float('-inf')

    # 1. Carregar best_reward a partir do model_best.pt
    best_model_path = "saved_models/model_best.pt"
    if os.path.exists(best_model_path):
        print(f"Loading best reward from {best_model_path}...")
        try:
            best_checkpoint = torch.load(best_model_path, weights_only=False)
            best_reward = best_checkpoint.get('best_reward', float('-inf'))
            print(f"Current best average reward known: {best_reward:.2f}")
        except Exception as e:
            print(f"Não foi possível ler o best_reward do {best_model_path}: {e}")

    # 2. Carregar o último modelo periódico para continuar o treinamento
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

    # 3. Restaura o checkpoint com tratamento para mudança de arquitetura (720D vs 3600D)
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"Loading latest checkpoint to resume from {latest_model_path}...")
        checkpoint = torch.load(latest_model_path, weights_only=False)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ep = checkpoint.get('episode', 0) + 1
            print(f"Checkpoint carregado com sucesso! Continuaremos a partir do Episódio {ep}.")
            
        except RuntimeError as e:
            print("\n" + "="*80)
            print("⚠️ AVISO DE INCOMPATIBILIDADE DE ARQUITETURA DETECTADO!")
            print("O checkpoint existente foi criado com um modelo de tamanho diferente (ex: 720D vs 3600D).")
            print("Iniciando um NOVO treinamento do zero com a nova arquitetura de 5 frames (3600D)...")
            print("="*80 + "\n")
            
            ep = 1
            best_reward = float('-inf')

    # Variáveis de Controle do Loop
    experiences = []
    episode_reward = 0
    rewards_history = []
    episode_metrics_list = []
    
    # Flags de estado do jogo
    in_game_flag = False
    first_frame = True  # Controls resetting the StateBuffer for the first frame
    
    # Controle de Frame Skip
    frame_skip_counter = 0
    accumulated_skip_reward = 0.0
    
    # Registros do estado inicial do Macro-Step e do Estado Anterior
    macro_start_state = None
    macro_action_idx = None
    macro_action_cont = None
    prev_state_tensor = None

    prev_gamestate = None
    
    # Instancia o buffer de 5 estados
    state_buffer = StateBuffer(k=5, state_dim=720)

    # Monitoramento de Deadlock
    last_step_time = time.time()
    TIMEOUT_SECONDS = 60.0  # Se o Dolphin ficar 10s sem responder, reseta o console!

    while True:
        # VERIFICAÇÃO ANTI-DEADLOCK: Se o Dolphin travou sem responder
        if time.time() - last_step_time > TIMEOUT_SECONDS:
            print(f"\n⚠️ DEADLOCK DETECTADO: Dolphin não responde há {TIMEOUT_SECONDS}s. Forçando reinício...")
            console, controllers, controller, cpuController = restart_console(
                old_console=console, 
                old_controllers=controllers, 
                iso_path=os.getenv("ISO_PATH"),
                agent_port=agent_port,
                enemy_port=enemy_port
            )
            first_frame = True  # Ativa para chamar o state_buffer.reset() no 1º frame pós-reinício
            last_step_time = time.time()
            continue

        gamestate = console.step()
        if gamestate is None:
            continue

        # Reseta o temporizador de inatividade a cada frame processado
        last_step_time = time.time()

        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            if agent_port not in gamestate.players or enemy_port not in gamestate.players:
                continue

            in_game_flag = True
            state_vec = get_state(gamestate, agent_port, enemy_port)
            state_tensor = torch.FloatTensor(state_vec)

            # --- USO DA SUA FUNÇÃO DE RESET NATIVA ---
            if first_frame:
                stacked_state = state_buffer.reset(state_tensor)  # Popula os 5 slots usando sua função
                first_frame = False
            else:
                stacked_state = state_buffer.append(state_tensor) # Adiciona o frame atual e desloca

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
            if macro_action_cont.dim() > 1:
                macro_action_cont = macro_action_cont.squeeze(0)
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
            prev_state_tensor = stacked_state

            # -------------------------------------------------------------
            # 4. FIM DO MACRO-STEP: Armazena a transição de 4 frames
            # -------------------------------------------------------------
            if frame_skip_counter >= FRAME_SKIP:
                state_to_save = stacked_state.squeeze(0) if stacked_state.dim() > 1 else stacked_state
                experiences.append((
                    macro_start_state,
                    macro_action_idx,
                    macro_action_cont,
                    accumulated_skip_reward,
                    False,
                    state_to_save
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
                        True,
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

            # Gerenciamento dos menus do Melee
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
                rewards_history.append(episode_reward)
                print("Average Reward: ", sum(rewards_history)/len(rewards_history))
                print("Average Reward (last 10): ", sum(rewards_history[-10:])/min(len(rewards_history), 10))

                avg_metrics = {}
                if episode_metrics_list:
                    for k in episode_metrics_list[0].keys():
                        avg_metrics[k] = sum(m[k] for m in episode_metrics_list) / len(episode_metrics_list)
                    print(f"Métricas Médias do Episódio: {avg_metrics}")

                logs_buffer.append([ep, episode_reward, best_reward, CPU_LEVEL, agent_stock, cpu_stock])

                checkpoint = {
                    'episode': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_reward': episode_reward,
                    'avg_metrics': avg_metrics
                }

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    checkpoint['best_reward'] = best_reward
                    torch.save(checkpoint, "saved_models/model_best.pt")
                    print(f"*** Novo melhor modelo salvo! Recompensa: {best_reward:.2f} ***")
                    
                    with open(best_csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([ep, episode_reward, best_reward, CPU_LEVEL, agent_stock, cpu_stock])

                if ep % 100 == 0:
                    torch.save(checkpoint, f"saved_models/model_ep_{ep}.pt")
                    print(f"*** Checkpoint periódico salvo: model_ep_{ep}.pt ***")
                    
                    with open(csv_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(logs_buffer)
                    logs_buffer = []

                # Prepara variáveis para a próxima partida
                ep += 1
                in_game_flag = False
                first_frame = True  # Garante que o state_buffer.reset() rode no 1º frame do próximo episódio
                episode_reward = 0
                episode_metrics_list = []

                # Reinício Preventivo a cada 25 episódios
                if ep % 25 == 0:
                    torch.save(checkpoint, f"saved_models/model_ep_{ep}.pt")
                    
                    console, controllers, controller, cpuController = restart_console(
                        old_console=console, 
                        old_controllers=controllers, 
                        iso_path=os.getenv("ISO_PATH"),
                        agent_port=agent_port,
                        enemy_port=enemy_port
                    )
                    first_frame = True

if __name__ == "__main__":
    main()