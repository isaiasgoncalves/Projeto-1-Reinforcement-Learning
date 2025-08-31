"""Classe que gerencia um agente a ser treinado"""
import numpy as np

import config as cf
from .robot import Robot


class Agent:
    def __init__(self, appr = cf.APPR, gamma = cf.GAMMA, epsilon = cf.EPSILON):
        self.appr = appr   # Taxa de aprendizado
        self.gamma = gamma # Fator de desconto
        self.epsilon = epsilon # Probabilidade de exploração

        self.q_table = {
            "high" : {
                        "search": 0.0,
                        "wait": 0.0
                      },
            "low" : {
                        "search": 0.0,
                        "wait": 0.0,
                        "recharge" : 0.0
                      }

        }

    def _choose_action(self, current_battery):
        """
        Escolhe qual será a próxima ação do agente
        Com probabilidade epsilon, ele irá escolher a melhor ação dentre as conhecidas, 
        e com probabilidade 1-epsilon ele irá explorar uma nova ação aleatória
        """

        # Pega o dicionário com as ações possíveis
        possible_actions = self.q_table[current_battery]

        if np.random.random() > self.epsilon:

            # Escolhe a melhor ação dentre as possíveis    
            return max(possible_actions.items(), key=lambda x: x[1])[0]

        else:
            # Escolhe uma ação aleatória
            return np.random.choice(list(possible_actions.keys()))
        

    def _update(self, current_battery_state, action, reward, next_battery_state):
        """
        Atualiza o valor de Q para o par (state, action) usando a fórmula do Q-Learning
        """

        # Valor do Q antigo
        old_value = self.q_table[current_battery_state][action]

        # Melhor valor de Q para o próximo estado
        next_max = max(self.q_table[next_battery_state].values())

        # Nosso novo valor estimado ótimo para esse estado
        new_estimate = reward + self.gamma * next_max
        
        # Atualiza o valor antigo na direção do novo
        new_value = old_value + self.appr * (new_estimate - old_value)
        self.q_table[current_battery_state][action] = new_value


    def learn(self):
        """
        Função responsável por gerenciar uma época de treinamento com vários passos.
        Recebe um agente e cria um Ambiente (robô) próprio, e realiza a sucessão de passos do treinamento.
        Retorna o somatório das recompensas obtidas
        """

        # Preparamos o ambiente para nossa época
        robot = Robot()

        for _ in range(cf.STEPS):
            # Iteramos em cada passo

            current_state = robot.battery
            # Escolhemos nossa próxima ação baseado na matriz de valores do Agente
            next_action = self._choose_action(robot.battery)

            # Executa a próxima ação e obtém a recompensa imediata
            if next_action == "search":
                action_reward = robot.search()
    
            elif next_action == "wait":
                action_reward = robot.wait()

            else:
                action_reward = robot.recharge()

            # Obtém o próxim estado
            next_state = robot.battery

            # O agente atualiza os valores do estado
            self._update(current_state, next_action, action_reward, next_state)

        # Ao final, obtemos nossa recompensa final
        return robot.total_reward




        

