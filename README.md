# DEM: Directed evolution algorithm drives neural prediction

<u aling="center"> 
***Selection specifies the candidates for the mutation, while mutation improves the odds of producing stronger offsprings*** 
</u>
<p align="center">
<img width="736" height="294" alt="Screenshot 2025-09-22 at 4 13 33 PM" src="https://github.com/user-attachments/assets/167bdd7a-a674-4ba1-90d2-7f7d435a5704" />
</p>



Inspired by chain-of-thought derived from directed evolution, we propose the Directed Evolution Model (DEM), a hybrid approach extending evolutionary computational strategies for directed evolution within a continual reinforcement learning framework, to address both domain shift and label scarcity issues. DEM can provide a unique solution for uncertainty exploration through mimicking the trial-and-error process of directed evolution. Importantly, our algorithm has the potential to be transformative not only for medical domain but also for other domains, especially new environments where labeled data is expensive and sparse.


## Introduction

A directed evolution algorithm implementing an intelligent model capable of rapidly adapting to a new, unseen environment, the method comprising:

-extending evolutionary computational strategies for directed evolution within a reinforcement learning (RL) framework to address both domain shift and label scarcity issues; \
-leveraging selection, mutation, and confidence calibration strategies to create a dynamic agent-environment interaction. We develop a novel directed evolution model (DEM) that is RL from the supervised feedback mechanism. This ensures DEM can be applied to precision-critical scenarios such as neural prediction, biological drug development, and autonomous systems. Moreover, the screening-to-evolving learning facilitates model application to large-scared data and reduces computing cost;\
-implementing uncertainty exploration following an evolutionary chain-of-thought, selecting the candidates with desired properties while evolution improves the odds of producing desired offsprings. This evolution computation strategies guide uncertainty exploration within RL, effectively implementing the processes of screening and evolving learning. Moreover, we adapt beam search strategy to store and prioritize old states based on their importance(rewards). These computational techniques derived from directed evolution ensure the uncertain exploration process in a more targeted manner and enhance the efficiency and stationarity of uncertain exploration when uncertainties change;\
-incorporating pseudo-labeling strategies to address label scarcity, allowing the model to work with unlabeled target data in a supervised manner;\
-integrating replay buffer with continual backpropagate to achieve a better trade-off between exploitation and exploration during uncertain exploration.


## Configuration
#### Step1: Pretraining
We first pretrained a source-led model, where labelled source training data and unlabelled target data as inputs, and then performance is measured on the source testing dataset. 

#### Step2: Instruction Tuning
Next, we utilized pretrained model to acquire target predictions as the initial pseudo-labeled target samples. Finally, high-confidence samples and pseudo-label variants will be iteratively produced from screening and evolving processes separately. Therefore, the continual reinforcement neural networks were iteratively trained on selective samples (screening phase) and refined pseudo-labels (evolving phase), and then their performances were measured on the target testing dataset.

## Citation
If you use this code or use our model for your research, please cite our paper:
```bibtex
@article{DEM-V0,
  title = {Directed evolution algorithm drives neural prediction},
  author = {Yanlin Wang, Nancy Young, Patrick Wong}
  journal = {},
  volume = {},
  pages ={},
  year = {2025},
  doi ={}
}
```
## Acknowledgement

We appreciate open source projects including: [["loss-of-plasticity" | https://github.com/shibhansh/loss-of-plasticity]]
