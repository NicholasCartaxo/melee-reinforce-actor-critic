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
from melee_actor_critic import Actor, Critic, train_step

# Configurações do Frame Skip e RL
FRAME_SKIP = 2    # Repete a mesma ação por 2 frames (30 tomadas de decisão/s)
N_STEPS = 256
ALPHA = 1e-4
BETA = 3e-4
CPU_LEVEL = 9

def main():
    parser = argparse.ArgumentParser(description="Train Melee RL Agent")
    parser.add_argument('--self-play', action='store_true', help="Treinar o agente contra si mesmo usando o modelo salvo mais recente")
    parser.add_argument('--play-human', action='store_true', help="Jogue contra o agente. Desativa o treinamento e permite controle manual do P4.")
    args = parser.parse_args()

    if args.play_human and args.self_play:
        print("Erro: --play-human e --self-play não podem ser usados juntos.")
        sys.exit(-1)

    load_dotenv()
    
    console = melee.Console(
        path=os.getenv("MAINLINE_PATH") if not args.play_human else os.getenv("MAINLINE_PLAYER_PATH"),
        fullscreen=False,
        save_replays=False,
        disable_audio= not args.play_human,
        emulation_speed= 0 if not args.play_human else 1,
        gfx_backend="Null" if not args.play_human else "",
        copy_home_directory=True
    )

    agent_port = 1
    enemy_port = 4 if not args.play_human else 2 # inimigo humano configurado na porta 2

    controller = melee.Controller(console=console, port=agent_port, type=melee.ControllerType.STANDARD)
    
    enemy_controller = None
    controllers = [controller]
    
    if not args.play_human:
        enemy_controller = melee.Controller(console=console, port=enemy_port, type=melee.ControllerType.STANDARD)
        controllers.append(enemy_controller)

    def signal_handler(sig, frame):
        for c in controllers:
            c.disconnect()
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

    # ==========================================
    # 2. INICIALIZAÇÃO DOS MODELOS E OTIMIZADORES
    # ==========================================
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*** PyTorch using device: {device} ***")
    
    INPUT_DIM = 720
    
    actor = Actor(input_dim=INPUT_DIM, num_actions=10).to(device)
    critic = Critic(input_dim=INPUT_DIM).to(device)
    
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ALPHA)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=BETA)

    if args.play_human:
        print("Modo Humano Ativado! O agente está apenas em modo de inferência (Treinamento desativado).")
        print(f"Certifique-se de configurar seu controle no Dolphin para a Porta {enemy_port}.")
        actor.eval()
        critic.eval()

    # Configuração do Oponente (Self-Play)
    if args.self_play:
        print("Modo Self-Play Ativado!")
        opponent_actor = Actor(input_dim=720, num_actions=10).to(device)
        opponent_actor.eval()

    ep = 1

    # Carregar o último modelo periódico para continuar o treinamento ou avaliar
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

    # ==========================================
    # 3. CARREGAMENTO DOS CHECKPOINTS
    # ==========================================
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"Loading latest checkpoint from {latest_model_path}...")
        checkpoint = torch.load(latest_model_path, weights_only=False)
        
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        
        if not args.play_human:
            actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            ep = checkpoint.get('episode', 0) + 1
            print(f"Resuming training from episode {ep}")
        else:
            print("Pesos carregados para avaliação contra jogador humano.")

        if args.self_play:
            opponent_actor.load_state_dict(checkpoint['actor_state_dict'])
            print("Modelo mais recente carregado para o oponente.")


    if int(torch.__version__.split('.')[0]) >= 2:
        print("Compiling models with torch.compile()...")
        actor = torch.compile(actor)
        critic = torch.compile(critic)
        
        print("Pre-warming the PyTorch compiler (this will take 30-60 seconds)...")
        dummy_state = torch.zeros(INPUT_DIM, dtype=torch.float32, device=device)
        with torch.no_grad():
            actor(dummy_state)
            critic(dummy_state)
        print("Warmup complete! Now connecting to Dolphin...")

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

    # Inicialização do CSV geral (apenas se treinando)
    csv_filename = "training_logs.csv"
    if not args.play_human and not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Atualizado para incluir as novas métricas de log
            writer.writerow(["Episode", "Reward", "CPU_Level", "Agent_Stock", "CPU_Stock", "Actor_Loss", "Critic_Loss", "Entropy"])

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
            state_tensor = torch.FloatTensor(state_vec).to(device)

            # INÍCIO DO MACRO-STEP
            if frame_skip_counter == 0:
                macro_start_state = state_tensor

                # Ação do Agente Principal
                with torch.no_grad():
                    action_discrete, action_continuous = actor.select_action(state_tensor)
                macro_action_idx = action_discrete.item()
                macro_action_cont = action_continuous

                # Ação do Oponente (Self-Play)
                if args.self_play and opponent_actor is not None:
                    opp_state_vec = get_state(gamestate, enemy_port, agent_port)
                    opp_state_tensor = torch.FloatTensor(opp_state_vec)
                    with torch.no_grad():
                        opp_act_d, opp_act_c = opponent_actor.select_action(opp_state_tensor)
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

            # FIM DO MACRO-STEP: Armazena a transição e Treina
            if frame_skip_counter >= FRAME_SKIP:
                if not args.play_human:
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

                if not args.play_human and len(experiences) >= N_STEPS:
                    # ==========================================
                    # 4. CHAMADA DE TREINAMENTO ATUALIZADA
                    # ==========================================
                    metrics = train_step(actor, critic, actor_optimizer, critic_optimizer, experiences)
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

                if not args.play_human and macro_start_state is not None and prev_state_tensor is not None:
                    experiences.append((
                        macro_start_state,
                        macro_action_idx,
                        macro_action_cont,
                        accumulated_skip_reward,
                        True,               # done = True
                        prev_state_tensor
                    ))

                if not args.play_human and len(experiences) > 0:
                    metrics = train_step(actor, critic, actor_optimizer, critic_optimizer, experiences)
                    episode_metrics_list.append(metrics)
                    experiences = []

                prev_gamestate = None
                prev_state_tensor = None
                macro_start_state = None
                frame_skip_counter = 0
                accumulated_skip_reward = 0.0

            # Gerenciamento dos menus
            # Se humano joga, deixamos o autostart desligado para o jogador poder configurar e dar o start manual
            opp_level = 0 if (args.self_play or args.play_human) else CPU_LEVEL

            menu_helper.menu_helper_simple(
                gamestate=gamestate, controller=controller,
                character_selected=melee.Character.LUIGI,
                stage_selected=melee.Stage.BATTLEFIELD,
                swag=False, autostart=False
            )
            
            if not args.play_human:
                menu_helper.menu_helper_simple(
                    gamestate=gamestate, controller=enemy_controller,
                    character_selected=melee.Character.LUIGI,
                    stage_selected=melee.Stage.BATTLEFIELD,
                    cpu_level=opp_level, swag=False, autostart=True
                )
                enemy_controller.flush()
            
            controller.flush()

            if in_game_flag:
                print(f"--- Fim do Episódio {ep} ---")
                print(f"Recompensa do Episódio: {episode_reward:.2f}")
                print(f"Estoque Final - Agent: {agent_stock} | Enemy: {cpu_stock}")
                
                if not args.play_human:
                    avg_metrics = {}
                    if episode_metrics_list:
                        for k in episode_metrics_list[0].keys():
                            avg_metrics[k] = sum(m[k] for m in episode_metrics_list) / len(episode_metrics_list)
                        print(f"Métricas Médias do Episódio: {avg_metrics}")

                    # Extrai os valores usando o .get para evitar key errors caso as chaves não existam ou estejam vazias
                    a_loss = avg_metrics.get('actor_loss', 0.0)
                    c_loss = avg_metrics.get('critic_loss', 0.0)
                    ent = avg_metrics.get('entropy', 0.0)

                    # Adiciona as novas métricas ao buffer
                    logs_buffer.append([ep, episode_reward, opp_level, agent_stock, cpu_stock, a_loss, c_loss, ent])

                    # ==========================================
                    # 5. SALVAMENTO DE CHECKPOINT ATUALIZADO
                    # ==========================================
                    checkpoint = {
                        'episode': ep,
                        'actor_state_dict': actor.state_dict(),
                        'critic_state_dict': critic.state_dict(),
                        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                        'episode_reward': episode_reward,
                        'avg_metrics': avg_metrics,
                    }

                    # Salva a cada 100 modelos
                    if ep % 100 == 0:
                        torch.save(checkpoint, f"saved_models/model_ep_{ep}.pt")
                        print(f"*** Checkpoint periódico salvo: model_ep_{ep}.pt ***")

                        # Atualiza o modelo do oponente de tempos em tempos com o último checkpoint do agente
                        if args.self_play and opponent_actor is not None:
                            opponent_actor.load_state_dict(actor.state_dict())
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
