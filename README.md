# Projeto 1 - Aprendizado por Reforço
Por:
* Isaías Gouvêa Gonçalves
* Joyce Figueiró Braga

### Proposta do Projeto: Aprendizado por Reforço para o Robô de Reciclagem

O projeto tem como objetivo implementar um algoritmo de *Aprendizado por Reforço (Reinforcement Learning - RL)* para o problema clássico do *Robô de Reciclagem*, descrito no Exemplo 3.3 do livro-texto.

O robô pode estar em dois estados de energia: *alta (high)* ou *baixa (low). Em cada estado, ele pode escolher entre diferentes ações: **procurar (search), **esperar (wait)* ou *recarregar (recharge)*. Cada ação tem uma probabilidade de sucesso e um valor de recompensa associado. Por exemplo:
- Procurar traz recompensas, mas pode reduzir o nível de energia do robô.
- Esperar é mais seguro, mas gera menor recompensa.
- Recarregar só pode ser feito no estado de energia baixa, restaurando o estado de energia para alto.

O aprendizado será realizado utilizando o algoritmo de *Temporal Difference (TD)*, que atualiza os valores de cada estado a partir das experiências acumuladas pelo robô.  
O objetivo é que o robô aprenda uma *política ótima, isto é, a melhor estratégia de ações em cada estado para **maximizar a soma total de recompensas* ao longo do tempo.

Ao final do projeto:
- Será registrado o total de recompensas acumuladas ao longo de várias épocas de treinamento.
- Serão gerados gráficos mostrando a evolução da recompensa acumulada.
- A política ótima aprendida será representada em um *heatmap*, mostrando as ações preferidas do robô em cada estado.

Este projeto une teoria e prática de RL, fornecendo uma aplicação didática do algoritmo TD em um ambiente de decisão estocástico simples, mas ilustrativo.

### Algoritmo Usado

O algoritmo que utilizamos para o aprendizado do robô de reciclagem é o Temporal Difference (TD). Ele é um dos métodos mais importantes do Aprendizado por Reforço, pois permite que o agente aprenda diretamente com a experiência, sem precisar conhecer de antemão todas as probabilidades do ambiente.

No algoritmo utilizado, a taxa de aprendizado (learning rate, α) controla o quanto o agente ajusta suas estimativas de valores $(Q(s,a) ou V(s))$ depois de observar uma nova experiência.

Em outras palavras, ela define o peso que a nova informação tem sobre a estimativa antiga.

A fórmula típica de atualização é assim:

Novo valor ← Antigo valor + $\alpha$[Recompensa observada + $\gamma$ Estimativa futura − Antigo valor]


Se $\alpha = 0$, o agente não aprende nada (sempre mantém o valor antigo).

Se $\alpha = 1$, o agente ignora o que sabia antes e confia apenas na experiência mais recente.

Se $0 < \alpha < 1$, há um equilíbrio entre histórico e nova experiência.

Além disso, existe o fator de desconto $\gamma$, que controla o peso das recompensas futuras:

Se $\gamma = 0$, o agente se preocupa apenas com a recompensa imediata.

Se $\gamma$ estiver próximo de 1, ele valoriza mais as recompensas que podem vir no futuro.

No caso desse exemplo de busca do robô, escolhemos  o $\gamma = 0.9$ (um valor alto) porque:
O robô não está resolvendo um problema de "curto prazo" apenas. Ele precisa considerar que, mesmo que andar agora não dê uma recompensa imediata, pode ser essencial para chegar ao objetivo (a busca).


Com $\gamma$ alto (próximo de 1) → o agente valoriza recompensas futuras quase tanto quanto as imediatas. Isso faz sentido para tarefas de navegação/exploração, onde o objetivo pode estar a vários passos de distância.


Se tivéssemos escolhido, por exemplo, $\gamma$ baixo ($0.3$ ou $0.5$) → o robô tenderia a se preocupar apenas com as recompensas imediatas (ex.: esperar dá logo +1, então ele poderia preferir sempre esperar, sem nunca buscar a recompensa maior que está mais distante).


Então, o $\gamma = 0.9$ foi escolhido para modelar que o robô tem “paciência” e está disposto a agir agora em troca de uma recompensa maior no futuro.
Em outras palavras:
$\gamma$ alto → robô “visionário” (valoriza o futuro).

$\gamma$ baixo → robô “impaciente” (valoriza só o agora).

O processo de aprendizado acontece repetidamente, ao longo de várias épocas, em que o agente age, recebe recompensas e ajusta suas estimativas. Com o tempo, ele aprende uma política ótima, ou seja, a melhor estratégia de ações em cada estado para maximizar a soma das recompensas no longo prazo.


### Como executar

Antes de tudo, garanta ter instalado em seu ambiente as bibliotecas necessárias (numpy, matplotlib e seaborn). Isso pode ser feito facilmente utilizando o comando abaixo no seu terminal:

`pip install -r requirements.txt`

No arquivo `config.py` você poderá editar os hiperparâmetros do projeto, veja abaixo:

```python
/config.py

########################################## Execução ###########################################

# Número de ciclos por época
STEPS = 1000
# Quantidade de épocas
EPOCHS = 1000



########################################## Ambiente ###########################################
ALPHA = .8
BETA = .8

# Recompensas
RSearch = 2
RWait = 0.5


############################################ Agente ###########################################

# Taxas de aprendizado 
APPR = .8

# Fator de desconto
GAMMA = .5

# Probabilidade de exploração
EPSILON = .3

```

Tendo os parâmetros conforme desejado, execute o treinamento do agente com o comando `python3 main.py` ou `python main.py` no seu console no diretório do projeto.

Como retorno o código indicará a execução das épocas, e ao final será gerado um arquivo `requirements.txt` com o histórico das recompensas máximas em cada época.
Serão gerados também dois gráfico em `optimal_policy.png` e `rewards_evolution.py` que apresentam a política ótima e a evolução da recompensa total respectivamente.


### Explicação sobre o código

O projeto está organizado da seguinte forma:

```
│ 
│=====[Código Fonte]=====
│ 
├── src/
│   ├── agent.py
│   ├── analytic_utils.py
│   └── robot.py
├── .gitignore
├── config.py
├── main.py
│ 
│====[Demais arquivos]====
│ 
├── optimal_policy.png
├── rewards_evolution.png
├── rewards.txt
├── requeriments.txt
└── README.md

```

O arquivo `config.py` como já mencionado contém as configurações gerais dos hiperparâmetros do modelo, e o `main.py` é o arquivo principal da execução do código, consolidando os métodos centrais.

A estrutura das funções do código foi feita utilizando o paratigma de Orientação a Objetos básico

O arquivo `robot.py` conta com a classe `Robot` que representa o ambiente onde o Agente irá aprender. Ali estão determinadas as regras de negócio especificadas anteriormente; todas as ações de um robô em cada estado; suas consequências e sistema de recompensas. Todos esses recursos foram implementados diretamente como características e métodos da classe, sinta-se à vontade para consultar o código comentado.

De forma semelhante, o arquivo `agent.py` contém a classe `Agent`, referente ao próprio agente que será treinado. Como um de seus atributos temos a matriz $Q(s, a)$ que contém o valor de cada uma das ações diante de cada um dos estados. 

Os três métodos da classe são cruciais para o aprendizado. O primeiro, `Agent._choose_action(...)` é responsável por determinar qual será a próxima ação tomada pelo agente. A partir do estado de um robô, ele irá tomar a decisão da melhor ação considerando uma probabilidade de explorar uma ação aleatória (essa probabilidade é escolhida em `config.py`).

O método `Agent._update(...)` é responsável por atualizar a matriz $Q(s,a)$ do nosso agente utilizando os dados mais recentes do robô no ambiente na equação de Bellman:

```python
# Nosso novo valor estimado ótimo para esse estado
new_estimate = reward + self.gamma * next_max

# Atualiza o valor antigo na direção do novo
new_value = old_value + self.appr * (new_estimate - old_value)

self.q_table[current_battery_state][action] = new_value
```

Por fim, o método `Agent.learn()`, representa uma época de aprendizado do agente. Ele começa um novo robô "do zero" e a partir daí começa a tomar decisões baseado na sua matriz Q-learning previamente existente. A ideia é em cada iteração essa matriz ser atualizada visando obter a máxima recompensa no final. Ao terminar uma época, a função retorna a recompensa total obtida pelo robô.

Já o arquivo `analytic_utils.py` contém métodos responsáveis por gerar os arquivos de resultado, o `rewards.txt` e os dois gráficos salvos como imagem no diretório.


### Resultados obtidos

