# Applications of the Generalized Stiefel Manifold in Neural Networks

This project investigates the use of the Stiefel manifold in neural networks, where weight matrices are constrained to lie on matrix manifolds. It also explores different optimization methods for training neural networks under these geometric constraints.

## CIFAR-10
CNN with 4 convolutional layers. Each projects with that architecture has two versions: fixed learning rate and learning rate with scheduler in the main and lr-scheduler branches respectively. 
**1. cifar_10_cnn_adam:** Used ADAM and SGD optimiziation methods.
**2. CIFAR_10+cnn+stiefel:** The weight matrices are constrained to lie on the Stiefel manifold. The optimization methods are Cayley SGD and Cayley ADAM. For the running project we need these extra files: gutils_modify, utils_modify, stiefel_optimizer_modify. 




