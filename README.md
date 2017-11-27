# BetterTrainingDataMnist_RL_GAN


Using a reinforcement learning model to generate "better" training samples for mnist. By “better", I mean training samples that, when used to train a model, give consistently better f1_scores on mnist then the original sample. In order to achieve this result, I will use as a reward the difference between the original sample accuracy on the mnist training set and the modified sample accuracy when training a model from those samples. I train a model with the samples on every batch and, taking advantage of the fact that the reward does not have to be differentiable, unlike GAN discriminators, I use a lightgbm model that can be trained to a decent accuracy much faster than a neural network to generate my reward signal.


General flow


Split mnist data in 3 groups, 50000 training data, 5000 training target, 5000 validation target
normalize pixel values from -1 to 1


While(!convergence)
  
  
  sample 10 different digit from the training data
  make a lightgbm model trained on only those 10 digits
  predict the class of the training target
  get the by class f1_score for the training target
  
  
  feed the sampled digits to the actor model to produce additive modifications
  add 0 centered noise to the predictions, scaled with how far we are in the training
  add the changes to the original digits to make the transformed set of digits
  make a lightgbm model trained on only the transformed digits
  predict the class of the training target and validation target
  get the by class f1_score for the training target and the validation target
  
  
  calculate the rewards : the new training target f1_score minus the original f1_score
  calculate the validation : the new validation target f1_score minus the orinal f1_score
  
  train the model with Proximal Policy Optimisation https://arxiv.org/abs/1707.06347
  
  
  
  
