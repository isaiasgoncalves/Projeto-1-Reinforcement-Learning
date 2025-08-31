"""Arquivo principal de treinamento do nosso agente"""
import numpy as np

from src.agent import Agent
from src.analytic_utils import AnalyticUtils
import config as cf

def main():

    agent = Agent()
    total_rewards = []

    for _ in range(cf.EPOCHS):
        # Rodamos uma época com nosso agente e salvamos a recompensa total
        print(f"Executando a Epoch {_ + 1}")
        total_rewards.append(agent.learn())

    # Escreve e salva o arquivo com os registros de 
    AnalyticUtils.save_rewards_file(total_rewards)

    # Plota um gráfico da evolução das recompensas totais
    AnalyticUtils.plot_rewards_graph(total_rewards)

    # Plota um heatmap com a política ótima
    AnalyticUtils.plot_optimal_policy(agent.q_table)
    

if __name__ == "__main__":
    main()


