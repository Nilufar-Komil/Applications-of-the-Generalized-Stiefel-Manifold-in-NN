# Applications of the Generalized Stiefel Manifold in Neural Networks

This project investigates the use of the Stiefel manifold in neural networks, where weight matrices are constrained to lie on matrix manifolds. It also explores different optimization methods for training neural networks under these geometric constraints.

## CIFAR-10
CNN with 4 convolutional layers. Each projects with that architecture has two versions: fixed learning rate and learning rate with scheduler in the main and lr-scheduler branches respectively. 

**1. cifar_10_cnn_adam:** Used *ADAM* and *SGD* optimiziation methods.

**2. CIFAR_10+cnn+stiefel:** The *weight matrices are* constrained to lie on the *Stiefel manifold*. The optimization methods are *Cayley SGD* and *Cayley ADAM*. For the running project we need these extra files: gutils_modify, utils_modify, stiefel_optimizer_modify. 

**3. CIFAR_10+cnn+Gstiefel+cayley (1):** The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *fixed learning metric* for each layer. The optimization methods are *Cayley SGD* and *Cayley ADAM*. For the running project we need these extra files: gutils_modify, utils_modify, stiefel_optimizer_modify. 

**4. CIFAR_10+cnn+Gstiefel+cayley+trivial (3):** The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *fixed learning metric* for each layer. The optimization methods are *Cayley SGD* and *Cayley ADAM*. In the project we *use idea trivializetion*, split manifold to product of two manifolds. For the running project we need these extra files: gutils_modify, utils_modify, stiefel_optimizer_modify.

**5. trivial_+_train_Q+L:** The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *training $B$s* for each layer. The optimization methods are *Cayley SGD* and *Cayley ADAM*. In the project we *use idea trivializetion*, split manifold to product of two manifolds. We use _spectral decompostion_ of $B$ and _train its eigenvalue and eigenvector matrices_ in per epoch. For the running project we need these extra files: gutils_modify, utils_modify, stiefel_optimizer_modify.

**6. CL+COL+GSt+GSt:** The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *training $B$s* for each layer. The optimization methods are *Cayley SGD* and *Cayley ADAM*. In the project we *use idea trivializetion*, split manifold to product of two manifolds. We use _spectral decompostion_ of $B$ and _train its eigenvalue and eigenvector matrices_ in per epoch. We also extract _CF and COL topological layers ideas_ from the paper "Topological Convolutional Neural Networks". For the running project we need these extra files: gutils_modify, utils_modify, stiefel_optimizer_modify, klein. 

# Applications of the Generalized Stiefel Manifold in Neural Networks

This project explores applications of the Stiefel and Generalized Stiefel manifolds in neural networks, where weight matrices are constrained to lie on matrix manifolds. It also compares several optimization methods for training neural networks under these geometric constraints.

## CIFAR-10

The experiments are based on a CNN with 4 convolutional layers. Each architecture is provided in two versions: one with a fixed learning rate and one with a learning-rate scheduler, available in the `main` and `lr-scheduler` branches, respectively.

**1. `cifar_10_cnn_adam`**  
Uses standard *Adam* and *SGD* optimization methods.

**2. `CIFAR_10+cnn+stiefel`**  
The weight matrices are constrained to lie on the *Stiefel manifold*. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following extra files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**3. `CIFAR_10+cnn+Gstiefel+cayley (1)`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with a *fixed learning metric* for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following extra files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**4. `CIFAR_10+cnn+Gstiefel+cayley+trivial (3)`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with a *fixed learning metric* for each layer. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following extra files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**5. `trivial_+_train_Q+L`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *trainable* $B$ matrices for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. It also uses the *spectral decomposition* of $B$ and trains its eigenvalue and eigenvector matrices at each epoch. To run this project, the following extra files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**6. `CL+COL+GSt+GSt`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *trainable* $B$ matrices for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. It also uses the *spectral decomposition* of $B$ and trains its eigenvalue and eigenvector matrices at each epoch. In addition, it incorporates *CF* and *COL* topological layer ideas inspired by the paper *Topological Convolutional Neural Networks*. To run this project, the following extra files are required: `gutils_modify`, `utils_modify`, `stiefel_optimizer_modify`, and `klein`.



