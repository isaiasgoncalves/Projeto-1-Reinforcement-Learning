"""Arquivo principal de treinamento do nosso agente"""
from src.agent import Agent
from src.analytic_utils import AnalyticUtils
import config as cf
import numpy as np

def main():

    agent = Agent()
    total_rewards = []
    average_rewards = []

    # Lista para armazenar as recompensas para tirar a média
    tmp_rewards = []

    for _ in range(cf.EPOCHS):
        # Rodamos uma época com nosso agente e salvamos a recompensa total
        print(f"Executando a Epoch {_ + 1}")
        total_rewards.append(agent.learn())

        if _ % 30 == 1:
            # A cada dez épocas, armazena a média de recompensas
            average_rewards.append(np.mean(np.array(tmp_rewards)))
            tmp_rewards = [agent.learn()]

        else:
            # Coleta a recompensa para calcular a média
            tmp_rewards.append(agent.learn())
            

    # Escreve e salva o arquivo com os registros de 
    AnalyticUtils.save_rewards_file(total_rewards)

    # Plota um gráfico da evolução das recompensas totais
    AnalyticUtils.plot_rewards_graph(total_rewards)

    # Plota um heatmap com a política ótima
    AnalyticUtils.plot_optimal_policy(agent.q_table)

    # Salva um gráfico de evolução das médias das recompensas
    AnalyticUtils.plot_average_rewards(average_rewards)

if __name__ == "__main__":
    main()


