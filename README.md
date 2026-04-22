# Applications of the Generalized Stiefel Manifold in Neural Networks

This project explores applications of the Stiefel and Generalized Stiefel manifolds in neural networks, where weight matrices are constrained to lie on matrix manifolds. It also compares several optimization methods for training neural networks under these geometric constraints.

## WRN-28-10

WRN-28-10 is a Wide Residual Network with 28 layers and a widening factor of 10. For the CIFAR-10 dataset, it begins with a 3×3 convolution, followed by 3 groups of residual blocks with channel sizes 160, 320, and 640. Each group contains 4 residual blocks, and each block consists of 2 convolutional layers. The network ends with global average pooling and a fully connected classification layer.

Each architecture is provided in two versions: one with a fixed learning rate and one with a learning-rate scheduler, available in the `main` and `lr-scheduler` branches, respectively.

**1. `WRN_28_10_adam`**  
Uses standard *Adam* and *SGD* optimization methods.

**2. `WRN_28_10+cnn+stiefel`**  
The weight matrices are constrained to lie on the *Stiefel manifold*. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**3. `WRN_28_10_gs_all_with_B_fixed`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with a *fixed learning metric* for each layer. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**4. `WRN_28_10_gs_all_with_B_train`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *trainable* $B$ matrices for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. It also uses the *spectral decomposition* of $B$ and trains its eigenvalue and eigenvector matrices at each epoch. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**5. `WRN_28_10_cf+col+_B_train`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *trainable* $B$ matrices for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. It also uses the *spectral decomposition* of $B$ and trains its eigenvalue and eigenvector matrices at each epoch. In addition, it incorporates *CF* and *COL* topological layer ideas inspired by the paper *Topological Convolutional Neural Networks*. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, `stiefel_optimizer_modify`, and `klein`.

## CIFAR-10 CNN

These experiments are based on a CNN with 4 convolutional layers. Each architecture is provided in two versions: one with a fixed learning rate and one with a learning-rate scheduler, available in the `main` and `lr-scheduler` branches, respectively.

**1. `cifar_10_cnn_adam`**  
Uses standard *Adam* and *SGD* optimization methods.

**2. `CIFAR_10+cnn+stiefel`**  
The weight matrices are constrained to lie on the *Stiefel manifold*. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**3. `CIFAR_10+cnn+Gstiefel+cayley (1)`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with a *fixed learning metric* for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**4. `CIFAR_10+cnn+Gstiefel+cayley+trivial (3)`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with a *fixed learning metric* for each layer. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. The optimization methods are *Cayley SGD* and *Cayley Adam*. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**5. `trivial_+_train_Q+L`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *trainable* $B$ matrices for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. It also uses the *spectral decomposition* of $B$ and trains its eigenvalue and eigenvector matrices at each epoch. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, and `stiefel_optimizer_modify`.

**6. `CL+COL+GSt+GSt`**  
The weight matrices are constrained to lie on the *Generalized Stiefel manifold* with *trainable* $B$ matrices for each layer. The optimization methods are *Cayley SGD* and *Cayley Adam*. This project uses the idea of *trivialization*, splitting the manifold into a product of two manifolds. It also uses the *spectral decomposition* of $B$ and trains its eigenvalue and eigenvector matrices at each epoch. In addition, it incorporates *CF* and *COL* topological layer ideas inspired by the paper *Topological Convolutional Neural Networks*. To run this project, the following additional files are required: `gutils_modify`, `utils_modify`, `stiefel_optimizer_modify`, and `klein`.



