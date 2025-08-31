"""Classe que representa um robô com suas propriedades dentro de uma simulação"""
import config as cf
import numpy as np


class Robot:
    def __init__(self):
        self.battery = "high"
        self.last_action = None
        self.total_reward = 0


    def search(self):
        if self.battery == "high":
            if np.random.random() > cf.ALPHA:
                # Mantém bateria alta
                self.total_reward += cf.RSearch
                self.last_action = "search"

                return cf.RSearch

            else:
                # Bateria cai para low
                self.total_reward += cf.RSearch
                self.last_action = "search"
                self.battery = "low"
            
                return cf.RSearch

        else:
            if np.random.random() > cf.BETA:
                # Mantém bateria no low
                self.total_reward += cf.RSearch
                self.last_action = "search"
                
                return cf.RSearch

            else:
                # Descarregou
                self.total_reward += -3
                self.last_action = "search"
                self.battery = "high" # Robô compulsoriamente recarregado
                
                return -3

    def wait(self):
        # Mantém o mesmo estado de bateria, e adiciona a recompensa R_wait

        self.total_reward += cf.RWait
        self.last_action = "wait"
        
        return cf.RWait
        
        
    def recharge(self):
        if self.battery == "high":
            # Não é possível recarregar com bateria alta
            raise Exception("Por alguma razão o robô tentou recarregar com bateria alta")
        
        self.last_action = "recharge"
        self.last_reward = 0
        self.battery = "high"
        
        return 0
    







