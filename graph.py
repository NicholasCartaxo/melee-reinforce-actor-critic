import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados (substitua 'dados.csv' pelo nome do seu arquivo)
nome_do_arquivo = 'training_logs.csv'
df = pd.read_csv(nome_do_arquivo)

# 2. Definir a lógica de Vitória/Derrota
# O agente vence se tiver mais "Stock" (vidas) que a CPU no fim do episódio
def determinar_resultado(row):
    if row['Agent_Stock'] > row['CPU_Stock']:
        return 'Vitória'
    elif row['Agent_Stock'] < row['CPU_Stock']:
        return 'Derrota'
    else:
        return 'Empate'

# Cria uma nova coluna no DataFrame com o resultado
df['Resultado'] = df.apply(determinar_resultado, axis=1)

# 3. Configurar o estilo e tamanho do gráfico
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Definir as cores para cada possível resultado
cores = {
    'Vitória': '#2ca02c', # Verde
    'Derrota': '#d62728', # Vermelho
    'Empate': '#7f7f7f'   # Cinza
}

# 4. Criar o gráfico de dispersão (scatter plot)
grafico = sns.scatterplot(
    data=df,
    x='Episode',
    y='Reward',
    hue='Resultado',
    palette=cores,
    alpha=0.7,       # Leve transparência para pontos sobrepostos
    edgecolor=None,
    s=30             # Tamanho dos pontos
)

# 5. Títulos e rótulos
plt.title('Recompensa (Reward) por Episódio ao longo do Treinamento', fontsize=14, pad=15)
plt.xlabel('Episódio', fontsize=12)
plt.ylabel('Recompensa Total', fontsize=12)

# Ajustar a legenda
plt.legend(title='Resultado', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout() # Evita que a legenda fique cortada

# 6. Exibir e/ou salvar o gráfico
plt.savefig('grafico_recompensas.png', dpi=300) # Remova o '#' para salvar a imagem
