"""Classe responsável pelas visualizações e registros dos resultados"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AnalyticUtils:

    @staticmethod
    def save_rewards_file(total_rewards: list):
        """
        Cria um arquivo .txt com o histórico de todas as recompensas totais de cada época na execução
        """
        with open('rewards.txt', 'w') as rewards_file:
            for epoch_index, epoch_reward in enumerate(total_rewards):
                rewards_file.write(f"| Epoch {epoch_index + 1} | Recompensa: {epoch_reward} |\n")

    
    @staticmethod
    def plot_rewards_graph(total_rewards: list):
        """
        Plota e salva um gráfico da evolução das recompensas totais por época
        """

        # Usando Dataframes para facilitar no armazenamento e visualização
        df = pd.DataFrame({
            'Época' : range(len(total_rewards)),
            'Recompensas totais' : total_rewards
        })

        # Define um tema para o gráfico
        sns.set_theme(style='darkgrid')

        # Cria o gráfico
        plt.figure(figsize=(10,6)) # Tamanho padrão, pode ser mudado se necessário
        sns.lineplot(x='Época', y='Recompensas totais', data=df)

        # Nomeando os eixos e dando título ao gráfico
        plt.xlabel('Época de Treinamento')
        plt.ylabel('Recompensa Total Acumulada')
        plt.ylim(bottom=0)
        plt.title('Evolução da Recompensa Total ao longo das épocas')

        # Salva a figura em um arquivo de imagem
        plt.savefig('rewards_evolution.png')

        # Mostra o gráfico na interface de comando
        plt.show()


    @staticmethod
    def plot_optimal_policy(optimal_policy : dict):
        """
        Prepara um Dataframe explicitando a política ótima para cada estado, e plota um gráfico de heatmap
        """
        
        policy_df = pd.DataFrame(optimal_policy).T

        # Criamos um novo dataframe para as probabilidades (inicialmente todas zero)
        prob_df = pd.DataFrame(0, index=policy_df.index, columns=policy_df.columns)

        # Para cada estado(linha), encontramos a ação de maior valor e marcamos sua probabilidade como 1
        for state in prob_df.index:
            best_action = policy_df.loc[state].idxmax()
            prob_df.loc[state, best_action] = 1
        # Ou seja, temos um dataframe cujas melhores ações possuem probabilidade 1 e as demais zero

        # Plotando a figura
        sns.set_theme(style='darkgrid')
        plt.figure(figsize=(10,6))
        sns.heatmap(data=prob_df, annot=True, cmap='viridis')

        # Legendas
        plt.title('Probabilidade do agente realizar cada ação em cada estado numa política ótima')
        plt.xlabel('Ação')
        plt.ylabel('Estado')

        # Salvando a figura e exibindo no terminal
        plt.savefig('optimal_policy.png')

        plt.show()



