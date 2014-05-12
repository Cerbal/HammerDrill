HammerDrill
===========
HammerDrill is a Java framework that allows to train deep Neural Network. Its efficiency comes from:
- the segmentation of the dataset in several chunks treated in parallel by different cores or (ultimately) machines. This segmentation is very similar to the project SandBlaster from Google.
- the computation of matrix multiplication in native code (OpenBlas or MKL) via the Java Framework MTJ.

The project is in its very early step. So far no release is available.

Example
=========
Here is an example of how to use HammerDrill. The following code train a denoising autoencoder on the MNIST dataset.

<code>
/** load the data from MNIST */
DenseMatrix data=Data.loadFromMatlabFormat("MNIST/MNIST_train.mat", "data");
/** corrupt 70% of the data */
DenseMatrix corrupted_data=Data.corruptData(data, 0.7);
/** define a new MLP, 784-144-784 with ReLU activation */
NeuralNetworks nn=new NeuralNetworks(new int[]{784,144,784}, Activation.ReLU);
nn.initialize();
/** Train the network in parallel*/
nn=Trainer.train(nn, corrupted_data, data);
/** Load the test data of the MNIST, and feed the network with it */
DenseMatrix test_data=Data.loadFromMatlabFormat("MNIST/MNIST_test.mat", "data");
DenseMatrix output_test=nn.feedforward(test_data);
/** Save the output and the neural network in matlab format for further exploitation */
Data.saveOnMatlabFormat(output_test, "output_test.mat");
NeuralNetworks.saveOnMatlabFormat(nn, "neural_network.mat");
</code>
