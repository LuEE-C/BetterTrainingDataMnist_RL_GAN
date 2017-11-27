# BetterTrainingDataMnist_RL_GAN


Using a reinforcement learning model to generate "better" training samples for mnist. By “better", I mean training samples that, when used to train a model, give consistently better f1_scores on mnist then the original sample. In order to achieve this result, I will use as a reward the difference between the original sample accuracy on the mnist training set and the modified sample accuracy when training a model from those samples. I train a model with the samples on every batch and, taking advantage of the fact that the reward does not have to be differentiable, unlike GAN discriminators, I use a lightgbm model that can be trained to a decent accuracy much faster than a neural network to generate my reward signal.

It uses Proximal Policy Optimisation https://arxiv.org/abs/1707.06347 as well as noisy layers for exploration https://arxiv.org/abs/1706.10295 .

![Alt text](/relative/CodeOutline.jpg?raw=true "Optional Title")

  
  
  
