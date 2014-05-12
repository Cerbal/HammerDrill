HammerDrill
===========
HammerDrill is a Java framework that allows to train deep Neural Networks. Its efficiency comes from:
- the segmentation of the dataset in several chunks treated in parallel by different cores or (ultimately) machines. This segmentation is very similar to the project SandBlaster from Google.
- the computation of matrix multiplication in native code (OpenBlas or MKL) via the Java Framework MTJ.

The project is in its very early step. So far no release is available.

Example
=========
Here is an example of how to use HammerDrill. The following code train a denoising autoencoder on the MNIST dataset.

![Alt text](/pictures/code.png)

which produces a network able to reconstruct a noisy input:

![Alt text](/pictures/reconstruction.png)
