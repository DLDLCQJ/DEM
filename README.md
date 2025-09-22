# DEM (Directed Evolution Model) : 
"<u>Selection specifies the candidates for the mutation, while mutation improves the odds of producing stronger offsprings</u>"

Inspired by chain-of-thought derived from directed evolution, we propose the Directed Evolution Model (DEM), a hybrid approach extending evolutionary computational strategies for directed evolution within a continual reinforcement learning framework, to address both domain shift and label scarcity issues. DEM can provide a unique solution for uncertainty exploration through mimicking the trial-and-error process of directed evolution. 



## Model Description

A directed evolution algorithm implementing an intelligent model capable of rapidly adapting to a new, unseen environment, the method comprising:
-extending evolutionary computational strategies for directed evolution within a reinforcement learning (RL) framework to address both domain shift and label scarcity issues;
-leveraging selection, mutation, and confidence calibration strategies to create a dynamic agent-environment interaction. We develop a novel directed evolution model (DEM) that is RL from the supervised feedback mechanism. This ensures DEM can be applied to precision-critical scenarios such as neural prediction, biological drug development, and autonomous systems. Moreover, the screening-to-evolving learning facilitates model application to large-scared data and reduces computing cost;
-implementing uncertainty exploration following an evolutionary chain of thought, selecting the candidates with desired properties while evolution improves the odds of producing desired offsprings. This evolution computation strategies guide uncertainty exploration within RL, effectively implementing the processes of screening and evolving learning. Moreover, we adapt beam search strategy to store and prioritize old states based on their importance(rewards). These computational techniques derived from directed evolution ensure the uncertain exploration process in a more targeted manner and enhance the efficiency and stationarity of uncertain exploration when uncertainties change;
-incorporating pseudo-labeling strategies to address label scarcity, allowing the model to work with unlabeled target data in a supervised manner;
-integrating replay buffer with continual backpropagate to achieve a better trade-off between exploitation and exploration during uncertain exploration.

## 
