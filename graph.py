import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
nome_do_arquivo = 'training_logs.csv'
df = pd.read_csv(nome_do_arquivo)

# 2. Definir a lógica de Vitória/Derrota
def determinar_resultado(row):
    if row['Agent_Stock'] > row['CPU_Stock']:
        return 'Vitória'
    elif row['Agent_Stock'] < row['CPU_Stock']:
        return 'Derrota'
    else:
        return 'Empate'

# Cria uma nova coluna no DataFrame com o resultado
df['Resultado'] = df.apply(determinar_resultado, axis=1)

# 3. Configurar o estilo e criar uma grade de gráficos (2 linhas, 2 colunas)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

# Definir as cores para o gráfico de recompensa
cores = {
    'Vitória': '#2ca02c', # Verde
    'Derrota': '#d62728', # Vermelho
    'Empate': '#7f7f7f'   # Cinza
}

# --- GRÁFICO 1: Reward (Recompensa) ---
sns.scatterplot(
    data=df, x='Episode', y='Reward', hue='Resultado',
    palette=cores, alpha=0.7, edgecolor=None, s=30, ax=axes[0, 0]
)
axes[0, 0].set_title('Recompensa (Reward) por Episódio', fontsize=14)
axes[0, 0].set_xlabel('Episódio')
axes[0, 0].set_ylabel('Recompensa Total')
axes[0, 0].legend(title='Resultado', loc='best')

# --- GRÁFICO 2: Actor Loss ---
# Usamos lineplot ou scatterplot. Como Loss de RL flutua muito, um gráfico de linha com certa transparência ajuda.
sns.lineplot(data=df, x='Episode', y='Actor_Loss', color='blue', alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_title('Actor Loss ao longo do Treinamento', fontsize=14)
axes[0, 1].set_xlabel('Episódio')
axes[0, 1].set_ylabel('Actor Loss')

# --- GRÁFICO 3: Critic Loss ---
sns.lineplot(data=df, x='Episode', y='Critic_Loss', color='orange', alpha=0.7, ax=axes[1, 0])
axes[1, 0].set_title('Critic Loss ao longo do Treinamento', fontsize=14)
axes[1, 0].set_xlabel('Episódio')
axes[1, 0].set_ylabel('Critic Loss')

# --- GRÁFICO 4: Entropy ---
sns.lineplot(data=df, x='Episode', y='Entropy', color='purple', alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title('Entropy (Exploração) ao longo do Treinamento', fontsize=14)
axes[1, 1].set_xlabel('Episódio')
axes[1, 1].set_ylabel('Entropy')

# Ajustar o layout para que os títulos não se sobreponham
plt.tight_layout()

# 4. Exibir e/ou salvar o gráfico final
plt.savefig('grafico_treinamento_completo.png', dpi=300)
print("Gráfico 'grafico_treinamento_completo.png' salvo com sucesso!")